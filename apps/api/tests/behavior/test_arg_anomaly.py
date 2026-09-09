"""Unit tests for app.modules.behavior.arg_anomaly — flattening, scoring, row init.

Scoring is DB-free by design: it operates on any object with the stat attrs
(SimpleNamespace here). observe()'s ORM row construction is covered via
_new_baseline, which exists precisely because Column defaults apply at flush,
not construction; observe() end-to-end is exercised by the docker smoke test
with a seeded baseline.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

from app.modules.behavior.arg_anomaly import (
    MAX_ARG_PATHS_PER_CALL,
    MAX_DISTINCT_VALUES,
    MAX_ROWS_PER_WORKSPACE,
    MAX_TRACKED_ARG_PATHS,
    MIN_SAMPLES,
    RULE_NOVEL_VALUE,
    RULE_NUMERIC,
    _apply_observations,
    _evict_lru,
    _fingerprint,
    _flatten_args,
    _new_baseline,
    _path_feature,
    _RowBudget,
    _score_and_update,
    Thresholds,
    ZSCORE_THRESHOLD,
)


def _baseline(**over):
    base = {
        "arg_path": "file_path",
        "kind": "numeric",
        "count": 0,
        "mean": 0.0,
        "m2": 0.0,
        "values": None,
        "overflow_count": 0,
    }
    base.update(over)
    return SimpleNamespace(**base)


# ─── _flatten_args ───────────────────────────────────────────────────────────


def test_flatten_numeric_and_path_and_nested():
    obs = list(
        _flatten_args(
            {
                "file_path": "apps/api/app/guard/policy.py",
                "mode": "append",
                "retries": 3,
                "opts": {"atomic": True, "chunk_size": 512},
            }
        )
    )
    as_dict = {path: (kind, value) for path, kind, value in obs}
    assert as_dict["file_path"] == ("category", "apps/api:.py")
    assert as_dict["mode"] == ("category", "append")
    assert as_dict["retries"] == ("numeric", 3.0)
    assert as_dict["opts.chunk_size"] == ("numeric", 512.0)
    assert "opts.atomic" not in as_dict  # bools carry no distribution


def test_flatten_long_string_becomes_length():
    obs = list(_flatten_args({"summary": "x" * 5000}))
    assert obs == [("summary#len", "numeric", 5000.0)]


def test_flatten_free_text_args_are_length_only_at_any_size():
    # content/old_string/new_string are near-unique per call. Treating a short
    # one as a category would mint a novel value on almost every write and
    # swamp the false-positive measurement this check exists to produce.
    obs = list(_flatten_args({"content": "ok", "old_string": "a", "new_string": "b"}))
    assert obs == [
        ("content#len", "numeric", 2.0),
        ("new_string#len", "numeric", 1.0),
        ("old_string#len", "numeric", 1.0),
    ]


def test_flatten_list_becomes_length():
    obs = list(_flatten_args({"edits": [1, 2, 3]}))
    assert obs == [("edits#len", "numeric", 3.0)]


def test_flatten_is_deterministic_regardless_of_insertion_order():
    tool_input = {f"k{i:02d}": i for i in range(20)}
    reversed_input = dict(reversed(list(tool_input.items())))
    first = list(_flatten_args(tool_input))[:MAX_ARG_PATHS_PER_CALL]
    second = list(_flatten_args(reversed_input))[:MAX_ARG_PATHS_PER_CALL]
    assert first == second  # sorted() walk keeps the cap deterministic
    assert len(first) == MAX_ARG_PATHS_PER_CALL


def test_path_feature_bounds_cardinality():
    assert _path_feature("apps/api/app/guard/policy.py") == "apps/api:.py"
    assert _path_feature("apps/api/app/guard/audit.py") == "apps/api:.py"
    assert _path_feature("C:\\Users\\dev\\notes.txt") == "c:/users:.txt"
    assert _path_feature("README") == ":"


def test_path_feature_keeps_dotfiles_and_traversal_signal():
    assert _path_feature(".env") == ":.env"  # dotfile != extensionless README
    assert _path_feature("./.env") == ":.env"
    assert _path_feature("..\\..\\key.pem") == "../..:.pem"  # traversal survives
    assert _path_feature(".ssh/config") == ".ssh:"


def test_fingerprint_normalizes_salts_and_never_returns_raw():
    assert _fingerprint("  Secret-Value  ", "ws-1") == _fingerprint(
        "secret-value", "ws-1"
    )
    assert _fingerprint("secret-value", "ws-1") != _fingerprint("secret-value", "ws-2")
    assert "secret" not in _fingerprint("secret-value", "ws-1")


# ─── row initialization ──────────────────────────────────────────────────────


def test_new_baseline_is_scorable_before_flush():
    # Column(default=...) applies at flush — a freshly constructed row must
    # already carry real zeros or the very first observation of every key
    # dies on None arithmetic (and fail-open hides the corpse).
    ws = uuid.uuid4()
    numeric = _new_baseline(ws, "user-1", "write", "retries", "numeric")
    assert (numeric.count, numeric.mean, numeric.m2) == (0, 0.0, 0.0)
    assert numeric.overflow_count == 0
    assert _score_and_update(numeric, "numeric", 1.0) is None
    assert numeric.count == 1

    category = _new_baseline(ws, "user-1", "write", "mode", "category")
    assert category.values == {}
    assert _score_and_update(category, "category", "fp-abc") is None
    assert category.values == {"fp-abc": 1}


# ─── numeric scoring ─────────────────────────────────────────────────────────


def _feed(baseline, values):
    for v in values:
        _score_and_update(baseline, "numeric", float(v))


def test_numeric_silent_below_min_samples():
    b = _baseline()
    _feed(b, [10.0] * (MIN_SAMPLES - 1))
    assert _score_and_update(b, "numeric", 1e9) is None  # cold start never flags


def test_numeric_outlier_flags_and_near_mean_does_not():
    b = _baseline()
    # Alternating values -> mean 15, stddev ~5, n = 300.
    _feed(b, [10.0, 20.0] * 150)
    assert _score_and_update(b, "numeric", 16.0) is None
    finding = _score_and_update(b, "numeric", 500.0)
    assert finding is not None
    assert finding["rule_id"] == RULE_NUMERIC
    assert "standard deviations" in finding["message"]


def test_numeric_zero_variance_flags_any_change():
    b = _baseline()
    _feed(b, [42.0] * (MIN_SAMPLES + 1))
    assert _score_and_update(b, "numeric", 42.0) is None
    finding = _score_and_update(b, "numeric", 43.0)
    assert finding is not None
    assert "first change from a constant baseline" in finding["message"]


def test_numeric_scores_before_updating():
    b = _baseline()
    _feed(b, [10.0, 20.0] * 150)
    mean_before = b.mean
    _score_and_update(b, "numeric", 500.0)
    assert b.mean > mean_before  # outlier folded in after scoring, not before


def test_kind_mismatch_is_skipped():
    b = _baseline(kind="numeric", count=MIN_SAMPLES + 50)
    assert _score_and_update(b, "category", "fp-surprise") is None
    assert b.count == MIN_SAMPLES + 50  # untouched


# ─── Thresholds (tunable via guard_config, NULL = module default) ────────────


def test_thresholds_default_to_module_constants():
    t = Thresholds()
    assert (t.zscore, t.min_samples) == (ZSCORE_THRESHOLD, MIN_SAMPLES)


def test_thresholds_from_config_reads_overrides():
    cfg = SimpleNamespace(arg_anomaly_zscore_threshold=3.5, arg_anomaly_min_samples=50)
    t = Thresholds.from_config(cfg)
    assert (t.zscore, t.min_samples) == (3.5, 50)
    assert isinstance(t.min_samples, int)


def test_thresholds_from_config_falls_back_on_null_or_junk():
    # NULL means "use the default". A nonsensical value must not cost a
    # finding on an advisory path, so it falls back rather than raising.
    for cfg in (
        SimpleNamespace(
            arg_anomaly_zscore_threshold=None, arg_anomaly_min_samples=None
        ),
        SimpleNamespace(arg_anomaly_zscore_threshold=0, arg_anomaly_min_samples=-5),
        SimpleNamespace(
            arg_anomaly_zscore_threshold="junk", arg_anomaly_min_samples="x"
        ),
        SimpleNamespace(),  # pre-migration row with neither column
    ):
        t = Thresholds.from_config(cfg)
        assert (t.zscore, t.min_samples) == (ZSCORE_THRESHOLD, MIN_SAMPLES)


def test_lowered_threshold_flags_what_the_default_would_not():
    # The point of the knob: the same observation, two verdicts.
    tuned = Thresholds(zscore=2.0, min_samples=10)
    strict = _baseline()
    loose = _baseline()
    _feed(strict, [10.0, 20.0] * 150)
    _feed(loose, [10.0, 20.0] * 150)
    assert _score_and_update(strict, "numeric", 33.0) is None
    assert _score_and_update(loose, "numeric", 33.0, tuned) is not None


def test_lowered_min_samples_ends_the_cold_start_early():
    b = _baseline()
    _feed(b, [5.0, 15.0] * 10)  # n = 20, far below the default 200
    assert _score_and_update(b, "numeric", 900.0) is None
    b2 = _baseline()
    _feed(b2, [5.0, 15.0] * 10)
    finding = _score_and_update(b2, "numeric", 900.0, Thresholds(min_samples=10))
    assert finding is not None


# ─── categorical scoring (values are fingerprints — observe() hashes) ────────


def test_category_novel_value_flags_after_min_samples():
    b = _baseline(kind="category", count=MIN_SAMPLES, values={"fp-seen": MIN_SAMPLES})
    assert _score_and_update(b, "category", "fp-seen") is None
    finding = _score_and_update(b, "category", "fp-never-before")
    assert finding is not None
    assert finding["rule_id"] == RULE_NOVEL_VALUE
    assert "fp-never-before" not in finding["message"]  # value never surfaced


def test_category_silent_below_min_samples():
    b = _baseline(kind="category", count=5, values={"fp-a": 5})
    assert _score_and_update(b, "category", "fp-brand-new") is None


def test_category_overflow_disarms_novelty():
    values = {f"fp-{i}": 1 for i in range(MAX_DISTINCT_VALUES)}
    b = _baseline(kind="category", count=MIN_SAMPLES + 500, values=values)
    assert _score_and_update(b, "category", "fp-one-too-many") is None
    assert b.overflow_count == 1
    # Disarmed for good, even for values that would fit again.
    assert _score_and_update(b, "category", "fp-another") is None


def test_category_values_reassigned_not_mutated():
    original = {"fp-a": 1}
    b = _baseline(kind="category", count=1, values=original)
    _score_and_update(b, "category", "fp-b")
    assert b.values is not original  # ORM change tracking needs reassignment


# ─── _apply_observations (observe()'s core, DB-free) ─────────────────────────


def test_apply_observations_never_stores_raw_category_values():
    rows = {}
    raw = "rm -rf /tmp/target"
    _apply_observations(
        rows,
        [("command", "category", raw)],
        salt="ws-1",
        new_baseline=lambda path, kind: _baseline(arg_path=path, kind=kind, values={}),
    )
    stored_keys = list(rows["command"].values.keys())
    assert stored_keys == [_fingerprint(raw, "ws-1")]
    assert raw not in stored_keys


def test_apply_observations_respects_tracking_cap():
    rows = {}
    created = []

    def capped(path, kind):
        if len(created) >= 2:
            return None  # cap reached — same contract as observe()'s closure
        b = _baseline(arg_path=path, kind=kind)
        created.append(path)
        return b

    findings = _apply_observations(
        rows,
        [("a", "numeric", 1.0), ("b", "numeric", 2.0), ("c", "numeric", 3.0)],
        salt="ws-1",
        new_baseline=capped,
    )
    assert created == ["a", "b"]
    assert "c" not in rows  # untracked past the cap, silently
    assert findings == []


def test_caps_are_ordered_sensibly():
    # Per-tool fan-out cap must be well under the tenant ceiling, or the
    # tenant cap could never be reached by a single tool and the LRU path
    # would be dead code.
    assert MAX_TRACKED_ARG_PATHS < MAX_ROWS_PER_WORKSPACE
    assert MAX_ROWS_PER_WORKSPACE == 5000


# ─── _evict_lru (LRU eviction, fake query chain) ─────────────────────────────


class _Clause:
    """A recorded filter predicate the fake query can actually apply."""

    def __init__(self, op, column, values=None):
        self.op, self.column, self.values = op, column, values

    def keeps(self, row):
        if self.op == "in":
            return getattr(row, self.column) in self.values
        if self.op == "notin":
            return getattr(row, self.column) not in self.values
        if self.op == "eq":
            return getattr(row, self.column) == self.values
        raise AssertionError(f"unhandled predicate: {self.op}")


class _FakeColumn:
    """Column stand-in supporting the operators _evict_lru calls."""

    def __init__(self, name):
        self.name = name

    def asc(self):
        return _Clause("asc", self.name)

    def in_(self, values):
        return _Clause("in", self.name, list(values))

    def notin_(self, values):
        return _Clause("notin", self.name, list(values))

    def __eq__(self, other):
        return _Clause("eq", self.name, other)


class _FakeQuery:
    """Stand-in for the SQLAlchemy query chain _evict_lru uses. Filters are
    really applied, so a missing predicate in the production code shows up as
    a failing assertion rather than passing silently."""

    def __init__(self, rows, recorder):
        self._rows = rows
        self._rec = recorder

    def filter(self, *clauses):
        self._rec["filters"] = self._rec.get("filters", 0) + len(clauses)
        for clause in clauses:
            self._rows = [r for r in self._rows if clause.keeps(r)]
        return self

    def order_by(self, clause):
        self._rec["order_by"] = (clause.column, clause.op)
        self._rows = sorted(self._rows, key=lambda r: r.updated_at)
        return self

    def limit(self, n):
        self._rec["limit"] = n
        self._rows = self._rows[:n]
        return self

    def with_for_update(self, skip_locked=False):
        self._rec["skip_locked"] = skip_locked
        self._rows = [r for r in self._rows if not getattr(r, "locked", False)]
        return self

    def all(self):
        return self._rows

    def delete(self, synchronize_session=False):
        self._rec["deleted"] = [r.id for r in self._rows]
        return len(self._rows)


class _FakeDb:
    def __init__(self, rows):
        self._rows = rows
        self.rec = {}

    def query(self, *args):
        return _FakeQuery(list(self._rows), self.rec)


class _Model:
    id = _FakeColumn("id")
    workspace_id = _FakeColumn("workspace_id")
    updated_at = _FakeColumn("updated_at")


_WS = uuid.uuid4()
_OTHER_WS = uuid.uuid4()


def _row(rid, age, workspace_id=_WS, locked=False):
    """age is an ordering key: lower = less recently updated."""
    return SimpleNamespace(
        id=rid, updated_at=age, workspace_id=workspace_id, locked=locked
    )


def test_evict_lru_deletes_the_least_recently_updated_first():
    # Insertion order is deliberately not LRU order, so a production change to
    # a different sort column (or none) fails this test.
    rows = [_row("newest", 50), _row("oldest", 1), _row("middle", 10)]
    db = _FakeDb(rows)
    deleted = _evict_lru(db, _Model, workspace_id=_WS, needed=2, protected_ids=[])
    assert deleted == 2
    assert db.rec["order_by"] == ("updated_at", "asc")
    assert db.rec["limit"] == 2
    assert db.rec["deleted"] == ["oldest", "middle"]


def test_evict_lru_is_scoped_to_one_workspace():
    # The predicate whose loss would be catastrophic in a multi-tenant table.
    rows = [_row("theirs", 1, workspace_id=_OTHER_WS), _row("ours", 5)]
    db = _FakeDb(rows)
    deleted = _evict_lru(db, _Model, workspace_id=_WS, needed=2, protected_ids=[])
    assert deleted == 1
    assert db.rec["deleted"] == ["ours"]  # another tenant's older row survives


def test_evict_lru_skips_rows_locked_by_other_transactions():
    # SKIP LOCKED: a stale row is both the best victim and, if an agent just
    # came back, the row being written. Never wait on it from a record-only path.
    rows = [_row("locked_elsewhere", 1, locked=True), _row("free", 9)]
    db = _FakeDb(rows)
    deleted = _evict_lru(db, _Model, workspace_id=_WS, needed=2, protected_ids=[])
    assert db.rec["skip_locked"] is True
    assert deleted == 1
    assert db.rec["deleted"] == ["free"]


def test_evict_lru_returns_zero_when_every_candidate_is_protected():
    # The table is full, but everything left belongs to this transaction.
    db = _FakeDb([_row("held", 1)])
    deleted = _evict_lru(db, _Model, workspace_id=_WS, needed=3, protected_ids=["held"])
    assert deleted == 0
    assert "deleted" not in db.rec  # no DELETE issued on an empty victim set


def test_evict_lru_never_evicts_a_protected_row():
    # A row this transaction holds locked must survive even when it is the
    # least-recently-updated candidate.
    db = _FakeDb([_row("held", 1), _row("r1", 2)])
    deleted = _evict_lru(db, _Model, workspace_id=_WS, needed=2, protected_ids=["held"])
    assert deleted == 1
    assert db.rec["deleted"] == ["r1"]


# ─── _RowBudget (the real cap/evict decision, DB-free) ───────────────────────


def _budget(tool_rows=0, ws_rows=0, evict=None):
    calls = {"evictions": []}

    def _evict(needed):
        calls["evictions"].append(needed)
        return evict(needed) if evict else 0

    b = _RowBudget(lambda: tool_rows, lambda: ws_rows, _evict)
    b.calls = calls
    return b


def test_budget_allows_rows_with_room_to_spare():
    b = _budget(tool_rows=0, ws_rows=0)
    assert b.allows_new_row() is True
    assert (b.tool_rows, b.ws_rows) == (1, 1)
    assert b.calls["evictions"] == []  # no eviction scan when under the cap


def test_budget_refuses_past_the_per_tool_cap_without_counting_workspace():
    b = _budget(tool_rows=MAX_TRACKED_ARG_PATHS, ws_rows=0)
    assert b.allows_new_row() is False
    assert b.ws_rows is None  # cheaper check short-circuits the wider one


def test_budget_evicts_exactly_enough_to_fit_one_more():
    # At the cap, one row must go so the new one lands exactly on the cap.
    b = _budget(ws_rows=MAX_ROWS_PER_WORKSPACE, evict=lambda n: n)
    assert b.allows_new_row() is True
    assert b.calls["evictions"] == [1]
    assert b.ws_rows == MAX_ROWS_PER_WORKSPACE


def test_budget_evicts_the_overshoot_plus_one_when_over_cap():
    b = _budget(ws_rows=MAX_ROWS_PER_WORKSPACE + 3, evict=lambda n: n)
    assert b.allows_new_row() is True
    assert b.calls["evictions"] == [4]  # 3 over, plus room for 1
    assert b.ws_rows == MAX_ROWS_PER_WORKSPACE


def test_budget_refuses_when_nothing_could_be_evicted():
    b = _budget(ws_rows=MAX_ROWS_PER_WORKSPACE, evict=lambda n: 0)
    assert b.allows_new_row() is False
    assert b.ws_rows == MAX_ROWS_PER_WORKSPACE  # never exceeds the cap


def test_budget_does_not_rescan_after_a_failed_eviction():
    # Eight args in one call must not mean eight full evict scans.
    b = _budget(ws_rows=MAX_ROWS_PER_WORKSPACE, evict=lambda n: 0)
    for _ in range(5):
        assert b.allows_new_row() is False
    assert b.calls["evictions"] == [1]  # tried once, then short-circuited


def test_budget_counts_lazily():
    counted = {"tool": 0, "ws": 0}

    def count_tool():
        counted["tool"] += 1
        return 0

    def count_ws():
        counted["ws"] += 1
        return 0

    b = _RowBudget(count_tool, count_ws, lambda n: 0)
    for _ in range(4):
        b.allows_new_row()
    assert counted == {"tool": 1, "ws": 1}  # one query each, not one per arg


def test_apply_observations_reuses_existing_rows():
    b = _baseline(arg_path="retries", kind="numeric", count=10, mean=5.0)
    rows = {"retries": b}
    _apply_observations(
        rows,
        [("retries", "numeric", 5.0)],
        salt="ws-1",
        new_baseline=lambda path, kind: (_ for _ in ()).throw(
            AssertionError("no new row expected")
        ),
    )
    assert b.count == 11
