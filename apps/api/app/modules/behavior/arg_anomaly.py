"""Record-only argument anomaly check — per-key baselines for tool-call args.

Static allow/deny rules catch shape, not intent: a tool call can be
schema-valid and still be far outside what that agent, in that workspace,
normally does with that tool and argument path. This module learns per-key
baselines keyed on (workspace_id, agent_key, tool_name, arg_path) and emits
advisory findings when a value lands far outside its own history.

This lives under modules/behavior rather than app/guard on purpose: Guard
decides what may happen before a call executes, this only describes what
already happened, after the fact, through the audit trail.

Posture, deliberately narrow for v1:
- Observes filesystem-write tools only (via tool_groups.normalize_tool).
- Record-only: findings surface as "audited" advisory events in the audit
  trail. Nothing in this module can BLOCK or APPROVAL-gate a call.
- Fail-open by design: an advisory check must never break a call. The caller
  (guard_check_impl) wraps observe() and rolls back on any error, so a
  failure degrades to a skipped observation.
- Raw string argument values are never persisted — baselines store
  workspace-salted sha256 prefixes and numeric features only.
- Gated by guard_config.arg_anomaly_enabled (default off). Enforcement is a
  follow-up once the false-positive rate on real traffic is known.

Known v1 behavior, accepted for the record-only phase:
- A flagged outlier is still folded into the baseline after scoring, so
  sustained drift flags its first occurrences and then normalizes.
- An arg with just under MAX_DISTINCT_VALUES legitimate values never disarms
  and will flag rare-but-legitimate values.
- An arg that has been constant for MIN_SAMPLES calls flags on any change,
  without reference to ZSCORE_THRESHOLD, because a constant has no variance
  to score against. Expect this to be the highest-volume numeric rule.
All three are exactly what the false-positive measurement is for.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING, Any

import structlog

from app.modules.guard.tool_groups import normalize_tool

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

log = structlog.get_logger(__name__)


# ─── Tuning constants (starting values — see the proposal issue) ──────────────

OBSERVED_TOOL_GROUPS = {"filesystem-write"}

# Below this many recorded calls for a key the check never fires (cold start
# stays silent). Starting value; the false-positive rate on real traffic
# decides where it settles.
MIN_SAMPLES = 200

# Numeric args with real variance: flag when |x - mean| / stddev meets this.
# Conservative on purpose, but note it does not govern the constant-baseline
# case below, which is the higher-volume rule in practice.
ZSCORE_THRESHOLD = 6.0

# Categorical args: once a key has seen this many distinct values, it is not
# categorical (free text, unique paths) — novelty detection disarms itself.
MAX_DISTINCT_VALUES = 200

# Cap observations per call so a pathological tool_input cannot fan out into
# unbounded work per request.
MAX_ARG_PATHS_PER_CALL = 8

# Cap distinct baseline rows per (workspace, agent_key, tool_name) so callers
# minting ever-new nested keys cannot grow the table without bound. Past the
# cap, unseen arg_paths are simply not tracked.
MAX_TRACKED_ARG_PATHS = 64

# Tenant-wide ceiling across every agent/tool/arg_path in a workspace. On
# insert past the cap, the least-recently-updated rows are evicted to make
# room, so a workspace that churns keys rotates rather than grows. The
# per-tool cap above bounds fan-out within one tool; this bounds the tenant.
MAX_ROWS_PER_WORKSPACE = 5000

# Below this, treat the baseline as constant — guards both true zero variance
# and float-noise m2 that would otherwise produce astronomical z-scores.
_STDDEV_FLOOR = 1e-9

_FLATTEN_MAX_DEPTH = 3

# Strings longer than this are content, not categories — observe length only.
_MAX_CATEGORICAL_LEN = 128

# Arg names whose values are filesystem paths: reduced to a low-cardinality
# feature (leading directories + extension) instead of the raw path.
_PATH_ARG_KEYS = {"file_path", "path"}

# Arg names carrying file content or patch text. These are near-unique per
# call at any length, so they are observed by length only, never as
# categories. Without this, every short write mints a novel value and the
# novelty rule dominates the findings it is supposed to measure.
_FREE_TEXT_ARG_KEYS = {"content", "old_string", "new_string", "text", "body", "patch"}

RULE_NUMERIC = "behavior.arg_anomaly.numeric"
RULE_NOVEL_VALUE = "behavior.arg_anomaly.novel_value"


# ─── Tunable thresholds ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class Thresholds:
    """The two knobs the false-positive checkpoint will actually move.

    Workspaces override them through guard_config so tuning is a config PATCH
    rather than a deploy. NULL in the database means "use the module default",
    which keeps the default itself in exactly one place (the constants above).
    """

    zscore: float = ZSCORE_THRESHOLD
    min_samples: int = MIN_SAMPLES

    @classmethod
    def from_config(cls, cfg: Any) -> Thresholds:
        """Read overrides off a GuardConfig row. Anything missing, unset or
        nonsensical falls back to the default rather than raising: this is an
        advisory path and a bad config value must not cost a finding."""
        return cls(
            zscore=_positive(
                getattr(cfg, "arg_anomaly_zscore_threshold", None), ZSCORE_THRESHOLD
            ),
            min_samples=int(
                _positive(getattr(cfg, "arg_anomaly_min_samples", None), MIN_SAMPLES)
            ),
        )


def _positive(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


# ─── Flattening — tool_input dict to (arg_path, kind, value) ─────────────────


def _path_feature(raw: str) -> str:
    """Reduce a filesystem path to '<first two dir segments>:<extension>'.

    'apps/api/app/guard/policy.py' -> 'apps/api:.py'. Keeps the baseline
    categorical (bounded cardinality) where raw paths are near-unique.
    '..' segments survive (traversal is signal) and a dotfile leaf is its
    own extension ('.env' stays distinct from an extensionless 'README').
    """
    norm = raw.replace("\\", "/").strip().lower()
    while norm.startswith("./"):
        norm = norm[2:]
    segments = [s for s in norm.lstrip("/").split("/") if s]
    prefix = "/".join(segments[:-1][:2])
    leaf = segments[-1] if segments else ""
    if leaf.startswith(".") and "." not in leaf[1:]:
        ext = leaf
    elif "." in leaf[1:]:
        ext = "." + leaf.rsplit(".", 1)[-1]
    else:
        ext = ""
    return f"{prefix}:{ext}"


def _fingerprint(raw: str, salt: str) -> str:
    """Workspace-salted fingerprint of a normalized string value. The raw
    value is never stored; salting keeps the 16-hex prefix from being a
    cross-workspace dictionary-attack target for low-entropy values."""
    normalized = raw.strip().lower()[:_MAX_CATEGORICAL_LEN]
    return hashlib.sha256(f"{salt}|{normalized}".encode()).hexdigest()[:16]


def _flatten_args(
    tool_input: dict, _prefix: str = "", _depth: int = 0
) -> Iterator[tuple[str, str, float | str]]:
    """Yield (arg_path, kind, value) observations from a tool_input dict.

    kind is 'numeric' (value: float) or 'category' (value: str, fingerprinted
    before storage). Strings at path-like keys become a low-cardinality path
    feature; long strings and lists are observed by length only. Keys are
    walked in sorted order so the per-call cap (applied by the caller) is
    deterministic.
    """
    if _depth >= _FLATTEN_MAX_DEPTH:
        return
    for key in sorted(tool_input):
        value = tool_input[key]
        arg_path = f"{_prefix}{key}"
        if isinstance(value, bool):
            continue  # true/false carries no distribution worth modeling
        if isinstance(value, (int, float)):
            yield arg_path, "numeric", float(value)
        elif isinstance(value, str):
            if key in _PATH_ARG_KEYS:
                yield arg_path, "category", _path_feature(value)
            elif key in _FREE_TEXT_ARG_KEYS or len(value) > _MAX_CATEGORICAL_LEN:
                yield f"{arg_path}#len", "numeric", float(len(value))
            else:
                yield arg_path, "category", value
        elif isinstance(value, list):
            yield f"{arg_path}#len", "numeric", float(len(value))
        elif isinstance(value, dict):
            yield from _flatten_args(value, _prefix=f"{arg_path}.", _depth=_depth + 1)


# ─── Scoring — pure, DB-free (baseline is any object with the stat attrs) ────


def _score_and_update(
    baseline: Any,
    kind: str,
    value: float | str,
    thresholds: Thresholds = Thresholds(),
) -> dict | None:
    """Score one observation against its baseline, then fold it in.

    For kind 'category', value is the already-computed fingerprint, never the
    raw string. Scoring happens before the update so the observation cannot
    vouch for itself. Returns a finding dict or None. Mutates baseline.count /
    mean / m2 / values / overflow_count in place; the caller owns persistence.
    """
    if baseline.kind != kind:
        return None  # arg changed type across calls — skip rather than mix stats

    finding: dict | None = None

    if kind == "numeric":
        assert isinstance(value, float)
        if baseline.count >= thresholds.min_samples:
            stddev = (
                math.sqrt(baseline.m2 / (baseline.count - 1))
                if baseline.count > 1
                else 0.0
            )
            if stddev < _STDDEV_FLOOR:
                # Constant baseline: no variance to score against, so any
                # change flags and ZSCORE_THRESHOLD does not apply. Expect
                # this to be the highest-volume numeric rule, and expect the
                # findings to look like ordinary traffic to a reviewer. The
                # message says so explicitly, because "arg changed for the
                # first time in 200 calls" is the whole signal here and is
                # not, on its own, evidence of anything wrong.
                if value != baseline.mean:
                    finding = {
                        "rule_id": RULE_NUMERIC,
                        "arg_path": baseline.arg_path,
                        "message": (
                            f"arg '{baseline.arg_path}' value {value:.4g} is the first "
                            f"change from a constant baseline of {baseline.mean:.4g} in "
                            f"{baseline.count} calls (no variance to score against, so "
                            f"the z-score threshold does not apply)"
                        ),
                    }
            else:
                zscore = abs(value - baseline.mean) / stddev
                if zscore >= thresholds.zscore:
                    finding = {
                        "rule_id": RULE_NUMERIC,
                        "arg_path": baseline.arg_path,
                        "message": (
                            f"arg '{baseline.arg_path}' value {value:.4g} is {zscore:.1f} "
                            f"standard deviations from its baseline (n={baseline.count})"
                        ),
                    }
        # Welford — numerically stable rolling mean/variance.
        baseline.count += 1
        delta = value - baseline.mean
        baseline.mean += delta / baseline.count
        baseline.m2 += delta * (value - baseline.mean)
        return finding

    # kind == "category" — value is the fingerprint
    assert isinstance(value, str)
    values = dict(baseline.values or {})  # reassigned below — JSONB in-place
    #                                       mutation is invisible to the ORM
    if value in values:
        values[value] += 1
    elif len(values) >= MAX_DISTINCT_VALUES:
        # High-cardinality arg — not categorical. Disarm novelty permanently
        # for this key; overflow_count > 0 is the disarmed marker.
        baseline.overflow_count += 1
    else:
        if baseline.count >= thresholds.min_samples and baseline.overflow_count == 0:
            finding = {
                "rule_id": RULE_NOVEL_VALUE,
                "arg_path": baseline.arg_path,
                "message": (
                    f"arg '{baseline.arg_path}' value unseen in {baseline.count} prior "
                    f"calls for this key (value stored as hash only)"
                ),
            }
        values[value] = 1
    baseline.count += 1
    baseline.values = values
    return finding


# ─── Observe (public API) ────────────────────────────────────────────────────


def _new_baseline(
    workspace_id: uuid.UUID, agent_key: str, tool_name: str, arg_path: str, kind: str
) -> Any:
    """Fully-initialized ArgBaseline. Column defaults apply at flush, not
    construction — a freshly built row must be scorable immediately, so
    every stat attr is set explicitly here."""
    from app.modules.behavior.models import ArgBaseline

    return ArgBaseline(
        workspace_id=workspace_id,
        agent_key=agent_key,
        tool_name=tool_name,
        arg_path=arg_path,
        kind=kind,
        count=0,
        mean=0.0,
        m2=0.0,
        values={} if kind == "category" else None,
        overflow_count=0,
    )


def _apply_observations(
    rows: dict[str, Any],
    observations: list[tuple[str, str, float | str]],
    salt: str,
    new_baseline: Callable[[str, str], Any | None],
    thresholds: Thresholds = Thresholds(),
) -> list[dict]:
    """Score observations against their rows, creating rows via new_baseline
    (which returns None once the tracking cap is hit). Category values are
    fingerprinted here, before scoring — raw strings never reach the
    baselines. Pure aside from mutating rows; unit-testable without a DB.
    """
    findings: list[dict] = []
    for arg_path, kind, value in observations:
        baseline = rows.get(arg_path)
        if baseline is None:
            baseline = new_baseline(arg_path, kind)
            if baseline is None:
                continue  # row-growth cap — unseen arg_paths stop being tracked
            rows[arg_path] = baseline
        if kind == "category":
            value = _fingerprint(value, salt)
        finding = _score_and_update(baseline, kind, value, thresholds)
        if finding:
            findings.append(finding)
    return findings


def observe(
    db: Session,
    *,
    workspace_id: uuid.UUID,
    agent_key: str,
    tool_name: str,
    tool_input: dict,
    thresholds: Thresholds | None = None,
) -> list[dict]:
    """Score one tool call against its per-key baselines and fold it in.

    Returns advisory findings (possibly empty). Never raises by contract of
    its caller — guard_check_impl wraps this in a fail-open handler and rolls
    the session back on error, so a concurrent-insert IntegrityError on the
    unique key degrades to a skipped observation, not a failed call.

    The caller is responsible for the guard_config.arg_anomaly_enabled gate
    and for resolving a real agent_key; this function assumes both.
    """
    if not isinstance(tool_input, dict):
        return []  # some frameworks send strings/arrays — nothing to model
    if normalize_tool(tool_name) not in OBSERVED_TOOL_GROUPS:
        return []
    observations = list(islice(_flatten_args(tool_input), MAX_ARG_PATHS_PER_CALL))
    if not observations:
        return []

    from sqlalchemy import func

    from app.modules.behavior.models import ArgBaseline

    tool = tool_name.lower()
    paths = [path for path, _, _ in observations]
    rows = {
        row.arg_path: row
        for row in (
            db.query(ArgBaseline)
            .filter(
                ArgBaseline.workspace_id == workspace_id,
                ArgBaseline.agent_key == agent_key,
                ArgBaseline.tool_name == tool,
                ArgBaseline.arg_path.in_(paths),
            )
            .with_for_update()
            .all()
        )
    }

    def _count_tool_rows() -> int:
        return (
            db.query(func.count(ArgBaseline.id))
            .filter(
                ArgBaseline.workspace_id == workspace_id,
                ArgBaseline.agent_key == agent_key,
                ArgBaseline.tool_name == tool,
            )
            .scalar()
            or 0
        )

    def _count_workspace_rows() -> int:
        return (
            db.query(func.count(ArgBaseline.id))
            .filter(ArgBaseline.workspace_id == workspace_id)
            .scalar()
            or 0
        )

    def _evict(needed: int) -> int:
        # Flush first so rows created earlier in this same call have ids and
        # can be protected; without it they are in the table but unprotected.
        db.flush()
        return _evict_lru(
            db,
            ArgBaseline,
            workspace_id=workspace_id,
            needed=needed,
            protected_ids=[r.id for r in rows.values()],
        )

    budget = _RowBudget(_count_tool_rows, _count_workspace_rows, _evict)

    def _budgeted_baseline(arg_path: str, kind: str) -> Any | None:
        if not budget.allows_new_row():
            return None
        baseline = _new_baseline(workspace_id, agent_key, tool, arg_path, kind)
        db.add(baseline)
        return baseline

    findings = _apply_observations(
        rows,
        observations,
        str(workspace_id),
        _budgeted_baseline,
        thresholds or Thresholds(),
    )
    db.commit()
    return findings


class _RowBudget:
    """Decides whether one more baseline row may be created.

    Enforces the per-(workspace, agent_key, tool) fan-out cap and the
    tenant-wide ceiling, evicting least-recently-used rows for the latter.
    Both counts are lazy: they cost a query only when a call actually creates
    a key, which is rare once a baseline is established. State lives for one
    observe() call. Counters and the evictor are injected so this is testable
    without a database.
    """

    def __init__(
        self,
        count_tool_rows: Callable[[], int],
        count_workspace_rows: Callable[[], int],
        evict: Callable[[int], int],
    ):
        self._count_tool_rows = count_tool_rows
        self._count_workspace_rows = count_workspace_rows
        self._evict = evict
        self.tool_rows: int | None = None
        self.ws_rows: int | None = None
        self.evict_exhausted = False

    def allows_new_row(self) -> bool:
        if self.tool_rows is None:
            self.tool_rows = self._count_tool_rows()
        if self.tool_rows >= MAX_TRACKED_ARG_PATHS:
            return False

        if self.ws_rows is None:
            self.ws_rows = self._count_workspace_rows()
        if self.ws_rows >= MAX_ROWS_PER_WORKSPACE:
            if self.evict_exhausted:
                return False  # already tried this call, do not re-scan per arg
            # Evict enough to leave room for exactly one more row.
            evicted = self._evict(self.ws_rows - MAX_ROWS_PER_WORKSPACE + 1)
            self.ws_rows -= evicted
            if self.ws_rows >= MAX_ROWS_PER_WORKSPACE:
                self.evict_exhausted = True
                return False  # nothing evictable, skip rather than exceed the cap

        # tool_rows is not decremented when eviction removes rows belonging to
        # this same tool, so the per-tool cap can read high after an eviction.
        # That errs toward tracking fewer keys, which is the safe direction.
        self.tool_rows += 1
        self.ws_rows += 1
        return True


def _evict_lru(
    db: Session,
    model: type[Any],
    *,
    workspace_id: uuid.UUID,
    needed: int,
    protected_ids: list[uuid.UUID],
) -> int:
    """Delete the `needed` least-recently-updated rows for a workspace.

    updated_at is the LRU key. Two things are never evicted: rows this
    transaction holds locked (passed in as protected_ids, since evicting them
    would lose the update we are about to write), and rows another transaction
    holds locked, via SKIP LOCKED. The latter matters because a stale row is
    both the best eviction candidate and, if an agent has just come back after
    a long gap, the row that agent is writing right now. Waiting on that lock
    would put a record-only path in the way of a live call. Returns the number
    actually deleted, which can be short of `needed`.
    """
    victims = db.query(model.id).filter(model.workspace_id == workspace_id)
    if protected_ids:
        victims = victims.filter(model.id.notin_(protected_ids))
    victim_ids = [
        row.id
        for row in victims.order_by(model.updated_at.asc())
        .limit(needed)
        .with_for_update(skip_locked=True)
        .all()
    ]
    if not victim_ids:
        return 0
    deleted = (
        db.query(model)
        .filter(model.id.in_(victim_ids))
        .delete(synchronize_session=False)
    )
    log.info(
        "behavior.arg_anomaly.baselines_evicted",
        workspace_id=str(workspace_id),
        evicted=deleted,
    )
    return deleted
