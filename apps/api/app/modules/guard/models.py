import hashlib
import uuid
from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, LargeBinary, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import ARRAY, JSON, JSONB, UUID
from pgvector.sqlalchemy import Vector

from app.core.database import Base


def get_policy_hash(db, ws_uuid, persona: str = "agent") -> str | None:
    """Snapshot the active policy version_hash for a workspace at decision time. Returns None if no cache yet."""
    row = db.get(GuardPolicyCache, (ws_uuid, persona))
    return row.version_hash if row else None


def chain_hash_for_insert(db, ws_uuid, ts: datetime, tool_call, decision: str):
    """Returns (previous_hash, entry_hash) for a new GuardAuditEvent row.
    Acquires a per-workspace row lock to serialise concurrent inserts."""
    last = (
        db.query(GuardAuditEvent.entry_hash)
        .filter(GuardAuditEvent.workspace_id == ws_uuid,
                GuardAuditEvent.entry_hash.isnot(None))
        .order_by(GuardAuditEvent.ts.desc())
        .with_for_update(skip_locked=False)
        .first()
    )
    prev = last.entry_hash if last else ""
    _tool = tool_call or ""
    entry = hashlib.sha256(f"{ts.isoformat()}|{_tool}|{decision}|{prev}".encode()).hexdigest()
    return prev, entry


class GuardConfig(Base):
    """One Guard config per workspace — workspace IS the Guard team."""

    __tablename__ = "guard_config"

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True)
    invite_code = Column(Text, nullable=False)
    slug = Column(Text, nullable=True)
    alert_channel = Column(Text, nullable=True)
    enforcement_mode = Column(String(20), nullable=False, default="warn")
    fail_mode = Column(String(20), nullable=False, default="fail_closed")  # fail_open | fail_closed (CLI behavior on outage)
    notify_on_block = Column(Boolean, nullable=False, default=True)
    notify_on_budget = Column(Boolean, nullable=False, default=True)
    resync_requested_at = Column(DateTime(timezone=True), nullable=True)
    token_guardrails = Column(JSONB, nullable=True)
    guardrail_snapshot = Column(JSONB, nullable=True)
    alert_slack_integration_id = Column(UUID(as_uuid=True), nullable=True)
    automation_security_scan = Column(Boolean, nullable=False, default=False)
    automation_workflow_trigger = Column(Boolean, nullable=False, default=False)
    # Dev-time persona: applies to MCP hook + daemon sync (Claude Code, Cursor, etc.).
    # 'conservative' | 'standard' | 'developer' — admin-managed; member can override.
    persona = Column(String(20), nullable=False, default="agent")
    # Runtime persona: applies to workflow execution (guard_block.py). Defaults to
    # 'conservative' so production runs always enforce the strictest rule set
    # regardless of what dev persona the workspace runs locally. Admin-only edit.
    runtime_persona = Column(String(20), nullable=False, default="agent")
    deny_on_error = Column(Boolean, nullable=False, default=True)  # fail-closed on policy eval error
    notify_on_fail_open = Column(Boolean, nullable=False, default=True)  # customer-facing WARNING when Guard engine falls open (#1520)
    advisory_mode = Column(Boolean, nullable=False, default=False)  # log all, block nothing
    arg_anomaly_enabled = Column(Boolean, nullable=False, default=False)  # record-only arg-drift observation, advisory audit events only
    # Tunables for the record-only checkpoint, so thresholds move by config
    # PATCH instead of a deploy. NULL = use the module default in
    # app.modules.behavior.arg_anomaly, keeping the default in one place.
    arg_anomaly_zscore_threshold = Column(Float, nullable=True)
    arg_anomaly_min_samples = Column(Integer, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(DateTime(timezone=True), nullable=True)


class GuardMemberConfig(Base):
    """Per-workspace CLI token for a Clerk user. Role is always read from workspace_users."""

    __tablename__ = "guard_member_config"

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True)
    clerk_user_id = Column(Text, nullable=False, primary_key=True)
    member_token = Column(Text, nullable=False)
    active = Column(Boolean, nullable=False, default=True)
    # NULL = inherit workspace default persona from guard_config.persona
    persona = Column(String(20), nullable=True)
    # 'user' = self-selected via conduct init; 'admin' = locked by admin
    assigned_by = Column(String(10), nullable=False, default="user")
    # Version string of workspace_instructions last synced by this member's CLI.
    # Null = never synced. Written by conduct guard sync.
    instructions_version = Column(String(64), nullable=True)
    joined_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    # Added by revision 0062 — model was missing this FK (#1284).
    agent_identity_id = Column(
        String(36),
        ForeignKey("agent_identities.id", ondelete="SET NULL", name="fk_guard_member_config_agent_identity"),
        nullable=True,
        index=True,
    )


class GuardSession(Base):
    __tablename__ = "guard_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    clerk_user_id = Column(Text, nullable=True)
    user_email = Column(String(255), nullable=True)
    ai_tool = Column(String(50), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    total_tokens_before = Column(Integer, nullable=False, default=0)
    total_tokens_after = Column(Integer, nullable=False, default=0)
    total_cost_usd = Column(Float, nullable=False, default=0.0)
    total_saved_usd = Column(Float, nullable=False, default=0.0)
    event_count = Column(Integer, nullable=False, default=0)
    violations_count = Column(Integer, nullable=False, default=0)
    client_ip = Column(String(64), nullable=True)
    os_info = Column(String(128), nullable=True)
    hostname = Column(String(255), nullable=True)
    intent = Column(Text, nullable=True)
    tool_sequence = Column(JSONB, nullable=True)
    session_parse_status = Column(String(20), nullable=True)  # ok|partial|failed|unsupported
    session_parser = Column(String(30), nullable=True)        # claude_code_v1|codex_v1


class GuardNotificationChannel(Base):
    """Per-action notification routing (#1142 Phase 1).

    One row per (workspace, action, channel_type, channel_ref). Phase 1 supports
    channel_type='slack'; Phase 2 adds email/pagerduty/webhook.

    Legacy guard_config.alert_channel + notify_on_block/notify_on_budget stay in
    place; the router auto-seeds this table from them on first read.
    """

    __tablename__ = "guard_notification_channels"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
    )
    action = Column(String(20), nullable=False)  # block | warn | audit | approval
    channel_type = Column(String(20), nullable=False, default="slack")
    integration_id = Column(UUID(as_uuid=True), nullable=True)
    channel_ref = Column(String(200), nullable=False)
    enabled = Column(Boolean, nullable=False, default=True)
    dedupe_window_sec = Column(Integer, nullable=False, default=300)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_guard_notif_workspace_action", "workspace_id", "action"),
        Index("idx_guard_notif_workspace_enabled", "workspace_id", "enabled"),
    )


class GuardAuditEvent(Base):
    __tablename__ = "guard_audit_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    clerk_user_id = Column(Text, nullable=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("guard_sessions.id"), nullable=True)
    user_email = Column(String(255), nullable=True)
    ai_tool = Column(String(50), nullable=False)
    tool_call = Column(String(255), nullable=True)  # nullable: proxy rows have no tool name
    source = Column(String(20), nullable=False, default="hook")   # 'hook' | 'proxy' | 'mcp'
    provider = Column(String(30), nullable=True)    # 'anthropic' | 'openai' | 'perplexity' (proxy only)
    model = Column(String(100), nullable=True)      # vendor model id (proxy only)
    input_summary = Column(Text, nullable=True)
    decision = Column(String(20), nullable=False)
    rule_id = Column(String(100), nullable=True)
    rule_message = Column(Text, nullable=True)
    tokens_before = Column(Integer, nullable=True)
    tokens_after = Column(Integer, nullable=True)
    tokens_saved = Column(Integer, nullable=True)
    cost_usd_before = Column(Float, nullable=True)
    cost_usd_after = Column(Float, nullable=True)
    tool_use_id = Column(Text, nullable=True, index=True)
    hook_session_id = Column(Text, nullable=True, index=True)  # raw string session_id from hook stdin
    conductai_run_id = Column(String(255), nullable=True)
    conductai_workflow = Column(String(255), nullable=True)
    conductai_workflow_id = Column(String(255), nullable=True)
    blast_radius = Column(JSONB, nullable=True)
    # Added by revision 0056 — model was missing these columns (#1284).
    os_info = Column(String(128), nullable=True)
    hostname = Column(String(255), nullable=True)
    ts = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    duration_ms = Column(Integer, nullable=True)
    execution_status = Column(String(20), nullable=True)   # success | error | timeout
    result_summary = Column(Text, nullable=True)
    # hash-chain integrity — do not UPDATE or DELETE rows, chain breaks
    previous_hash = Column(Text, nullable=True)
    entry_hash = Column(Text, nullable=True)
    policy_hash = Column(Text, nullable=True)  # version_hash from GuardPolicyCache at decision time
    # session goal set by `conduct session start`
    goal_id   = Column(String(255), nullable=True)
    goal_name = Column(String(255), nullable=True)
    # layered verdict envelope (#1150 phase 1) — nullable so pre-migration and
    # non-eval audit paths (auth, approval, guard block) stay as-is
    evaluated_rules = Column(JSONB, nullable=True)   # list of {rule_id, severity, action, message}
    defense_score   = Column(Integer, nullable=True)  # weighted aggregate across matched rules
    # PR B.6 (#1347) — persisted routing decision for LLM proxy calls.
    # Only populated when the caller sent a tier form ("balanced" etc.);
    # NULL when a concrete model ID was forwarded straight through.
    routing_meta    = Column(JSONB, nullable=True)
    # Added by revision 0102 (#1340) — was already read by _event_to_dict via
    # getattr hotfix (#1338). Declaring on ORM closes schema drift.
    # ondelete=SET NULL: audit history must survive identity deletion.
    agent_identity_id = Column(
        String(36),
        ForeignKey("agent_identities.id", ondelete="SET NULL"),
        nullable=True,
    )
    # Added by revision 0116 — sha256 of a short-lived share token, present
    # only for trial-workspace blocks. Enables anonymous receipt lookup for
    # signup users who don't yet have a login; workspace rows leave it NULL.
    share_token_hash = Column(Text, nullable=True)

    __table_args__ = (
        Index("ix_guard_audit_events_source", "workspace_id", "source", "ts"),
        Index(
            "ix_guard_audit_events_provider",
            "workspace_id", "provider", "ts",
            postgresql_where=sa.text("provider IS NOT NULL"),
        ),
        Index("ix_guard_audit_events_entry_hash", "entry_hash"),
        Index(
            "ix_guard_audit_events_evaluated_rules_gin",
            "evaluated_rules",
            postgresql_using="gin",
        ),
        Index("ix_guard_audit_events_ws_ts", "workspace_id", sa.text("ts DESC")),
        Index("ix_guard_audit_events_agent_identity_id", "agent_identity_id"),
    )


class GuardSavings(Base):
    """Per-developer RTK + Agent Booster token savings snapshot, pushed by `conduct guard sync`."""

    __tablename__ = "guard_savings"

    id = Column(Integer, primary_key=True)
    workspace_id = Column(Text, nullable=False, index=True)
    member_email = Column(Text, nullable=False)
    rtk_saved_tokens = Column(BigInteger, nullable=False, default=0)
    rtk_savings_pct = Column(Float, nullable=False, default=0.0)
    rtk_total_commands = Column(Integer, nullable=False, default=0)
    booster_saved_tokens = Column(BigInteger, nullable=False, default=0)
    booster_savings_pct = Column(Float, nullable=False, default=0.0)
    booster_total_reads = Column(Integer, nullable=False, default=0)
    period_start = Column(DateTime(timezone=True), nullable=True)
    period_end = Column(DateTime(timezone=True), nullable=False)
    recorded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class GuardDeveloperTools(Base):
    """Per-developer AI tool coverage snapshot, pushed by conduct login / guard sync."""
    __tablename__ = "guard_developer_tools"

    id = Column(Integer, primary_key=True)
    workspace_id = Column(Text, nullable=False, index=True)
    user_email = Column(Text, nullable=False)
    detected_tools = Column(JSONB, nullable=False, default=list)   # ["claude-code", "vscode", ...]
    mcp_registered = Column(JSONB, nullable=False, default=list)   # tools where conduct-mcp is wired
    hook_registered = Column(JSONB, nullable=False, default=list)  # tools where Guard hook is wired
    reported_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("workspace_id", "user_email", name="uq_guard_dev_tools"),
    )


class GuardSpendBudget(Base):
    __tablename__ = "guard_spend_budgets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    clerk_user_id = Column(Text, nullable=True)
    monthly_limit_usd = Column(Float, nullable=False)
    alert_threshold_pct = Column(Integer, nullable=False, default=80)
    hard_limit_usd = Column(Float, nullable=True)
    default_per_developer_usd = Column(Float, nullable=True)
    last_alert_pct_bucket = Column(Integer, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index(
            "uq_guard_spend_workspace_default", "workspace_id",
            unique=True,
            postgresql_where=sa.text("clerk_user_id IS NULL"),
        ),
        Index(
            "uq_guard_spend_workspace_member", "workspace_id", "clerk_user_id",
            unique=True,
            postgresql_where=sa.text("clerk_user_id IS NOT NULL"),
        ),
    )


class GuardRateLimit(Base):
    """Per-workspace / per-agent-identity RPM+TPM caps (#980).

    workspace_id + agent_identity_id=NULL row = workspace default.
    workspace_id + agent_identity_id=X row  = override for that agent identity.
    Enforced by app.modules.guard.rate_limit.check_rate_limit at proxy step 4d.2.
    """
    __tablename__ = "guard_rate_limits"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    agent_identity_id = Column(String(36), ForeignKey("agent_identities.id", ondelete="CASCADE"), nullable=True)
    rpm = Column(Integer, nullable=True)
    tpm = Column(Integer, nullable=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        UniqueConstraint("workspace_id", "agent_identity_id", name="uq_guard_rate_limits_scope"),
        Index("idx_guard_rate_limits_ws", "workspace_id"),
    )


class SessionReport(Base):
    __tablename__ = "session_reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
    )
    clerk_user_id = Column(Text, nullable=True)
    developer_email = Column(String(255), nullable=False)
    archetype = Column(String(100), nullable=True)
    autonomy_score = Column(Float, nullable=True)
    planning_ratio = Column(Float, nullable=True)
    sessions = Column(Integer, nullable=False, default=0)
    prompts = Column(Integer, nullable=False, default=0)
    commits = Column(Integer, nullable=False, default=0)
    lines_per_hour = Column(Float, nullable=True)
    active_days = Column(Integer, nullable=True)
    tools_json = Column(JSONB, nullable=True)
    report_md = Column(Text, nullable=True)
    embedding = Column(Vector(1536), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("ix_session_reports_workspace", "workspace_id"),
        Index("ix_session_reports_email", "developer_email"),
    )


# ── Skill Pack Model ───────────────────────────────────────────────────────────

class SkillPack(Base):
    """Catalog of available skill packs. Rules live here, not per-workspace."""

    __tablename__ = "skill_packs"

    slug        = Column(Text, primary_key=True)            # "conduct-base", "conduct-soc2"
    version     = Column(Text, primary_key=True)            # "1.0.0"
    name        = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    tier        = Column(Text, nullable=False, default="free")  # free / paid / enterprise
    rules       = Column(JSONB, nullable=False, default=list)   # [{id, match_tool, match_pattern, ...}]
    published_at = Column(DateTime(timezone=True), nullable=True)
    created_at  = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class WorkspaceSkillPack(Base):
    """Which skill packs each workspace has installed."""

    __tablename__ = "workspace_skill_packs"

    workspace_id    = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True)
    pack_slug       = Column(Text, primary_key=True)
    pinned_version  = Column(Text, nullable=True)   # null = always latest
    installed_by    = Column(Text, nullable=True)   # clerk_user_id
    installed_at    = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_workspace_skill_packs_workspace", "workspace_id"),
    )


class WorkspaceCustomRule(Base):
    """Per-workspace custom guard rules. Replaces the legacy guard_policies
    table for non-pack rules. Pack rules live in skill_packs.rules JSONB."""

    __tablename__ = "workspace_custom_rules"

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True)
    rule_id      = Column(Text, primary_key=True)
    persona      = Column(Text, nullable=False, default="agent")  # "agent" or "proxy"
    body         = Column(JSONB, nullable=False)            # full rule shape (id, match_*, action, message, severity, ...)
    enabled      = Column(Boolean, nullable=False, default=True)
    created_by   = Column(Text, nullable=True)              # clerk_user_id
    created_at   = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at   = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc),
                          onupdate=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_workspace_custom_rules_workspace", "workspace_id"),
    )


class GuardRuleOverride(Base):
    """Per-workspace overrides on top of skill pack defaults."""

    __tablename__ = "guard_rule_overrides"

    workspace_id    = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True)
    rule_id         = Column(Text, primary_key=True)
    action          = Column(Text, nullable=True)           # null = use pack default
    disabled        = Column(Boolean, nullable=False, default=False)
    custom_message  = Column(Text, nullable=True)
    match_pattern   = Column(Text, nullable=True)           # null = use pack default
    overridden_by   = Column(Text, nullable=True)           # clerk_user_id
    overridden_at   = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    reason           = Column(Text, nullable=True)           # required for security-relaxing exceptions
    expires_at       = Column(DateTime(timezone=True), nullable=True)
    use_audited_at    = Column(DateTime(timezone=True), nullable=True)
    expiry_audited_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_guard_rule_overrides_workspace", "workspace_id"),
    )


class GuardPolicyCache(Base):
    """Pre-computed flattened policy per workspace+persona. Invalidated on pack/override change."""

    __tablename__ = "guard_policy_cache"

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True)
    persona      = Column(Text, primary_key=True)
    payload      = Column(JSONB, nullable=False, default=list)
    version_hash = Column(Text, nullable=False)
    computed_at  = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class WorkspaceSigningKey(Base):
    """One HMAC-SHA256 signing key per workspace, used to sign GET /guard/policies/sync responses.

    The raw key_bytes are returned only at POST (generate/rotate) time. All subsequent
    reads return the fingerprint only. The CLI writes the key to ~/.conductguard/signing.key
    and verifies each fetched policy before caching it to disk.
    """

    __tablename__ = "workspace_signing_keys"

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True)
    key_bytes    = Column(LargeBinary(32), nullable=False)
    fingerprint  = Column(Text, nullable=False)
    created_at   = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    rotated_at   = Column(DateTime(timezone=True), nullable=True)


class DiscoveryScan(Base):
    __tablename__ = "discovery_scans"

    id             = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id   = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    triggered_by   = Column(String(20), nullable=False)    # cli | schedule
    status         = Column(String(20), nullable=False)    # running | complete | failed
    agents_found   = Column(Integer, nullable=True)
    guard_coverage = Column(Integer, nullable=True)        # count under Guard
    scan_config    = Column(JSONB, nullable=True)
    started_at     = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    completed_at   = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_discovery_scans_workspace", "workspace_id"),
    )


class PolicyCertification(Base):
    __tablename__ = "policy_certifications"

    id             = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id   = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    pack_slug      = Column(Text, nullable=False)
    certified_by   = Column(Text, nullable=False)   # clerk_user_id
    policy_version = Column(Text, nullable=True)    # version_hash snapshot
    certified_at   = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_policy_cert_ws_pack_ts", "workspace_id", "pack_slug", sa.text("certified_at DESC")),
    )


class DiscoveredAgent(Base):
    __tablename__ = "discovered_agents"

    id            = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id  = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    scan_id       = Column(UUID(as_uuid=True), ForeignKey("discovery_scans.id", ondelete="CASCADE"), nullable=True)
    name          = Column(Text, nullable=True)
    framework     = Column(String(50), nullable=True)   # langchain|crewai|autogen|claude-code|copilot|cursor|codex|windsurf
    source        = Column(String(20), nullable=True)   # config | process
    location      = Column(Text, nullable=True)         # config path | process cmd
    evidence      = Column(JSONB, nullable=True)
    risk_score    = Column(Integer, nullable=True)      # 0-100
    under_guard   = Column(Boolean, nullable=False, default=False)
    proxy_routed  = Column(Boolean, nullable=False, default=False)
    first_seen_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    last_seen_at  = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint(
            "workspace_id", "framework", "source",
            name="uq_discovered_agents_workspace_framework_source",
        ),
        Index("ix_discovered_agents_workspace", "workspace_id"),
        Index("ix_discovered_agents_scan", "scan_id"),
    )


class GuardVerifyRun(Base):
    """Persisted result of a Guard Verify adversarial test battery execution."""

    __tablename__ = "guard_verify_runs"

    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    score        = Column(Integer, nullable=False)
    grade        = Column(String(2), nullable=False)
    results      = Column(JSONB, nullable=False)   # list of test result dicts
    total_tests  = Column(Integer, nullable=False)
    passed_tests = Column(Integer, nullable=False)
    created_at   = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class GuardKnowledgeIndex(Base):
    """Unified semantic index across Guard sources — audit events, rules, discovered agents.
    Used by GLens for intent-based search ('blocks related to secrets', etc.)."""

    __tablename__ = "guard_knowledge_index"

    id            = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id  = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    source_kind   = Column(Text, nullable=False)   # audit_event | rule | discovered_agent
    source_id     = Column(Text, nullable=False)
    canonical_text = Column(Text, nullable=False)
    meta          = Column("metadata", JSONB, nullable=False, default=dict)
    content_hash  = Column(Text, nullable=False)
    embedding     = Column(Vector(1536), nullable=True)
    updated_at    = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        UniqueConstraint("workspace_id", "source_kind", "source_id", name="guard_knowledge_index_workspace_id_source_kind_source_id_key"),
        Index("guard_knowledge_index_workspace_id_source_kind_idx", "workspace_id", "source_kind"),
        Index(
            "guard_knowledge_index_embedding_idx", "embedding",
            postgresql_using="ivfflat",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"lists": "100"},
        ),
    )


class GuardApprovalRequest(Base):
    """HITL approval request created by a rule with action=approval (#1140).

    Surface-agnostic: created from the CLI/MCP hook, LLM proxy, or workflow
    runtime. When triggered inside a workflow, source_run_id/source_block_id
    link back so the decide endpoint can resume the paused run using the same
    approval_received run_event that DSL approve blocks already emit.
    """

    __tablename__ = "guard_approval_requests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)

    rule_id      = Column(String(200), nullable=False)
    rule_pack    = Column(String(100), nullable=True)
    rule_message = Column(Text, nullable=True)

    tool_name  = Column(String(100), nullable=True)
    tool_input = Column(JSONB, nullable=False, default=dict)

    requester_email       = Column(String(255), nullable=True)
    requester_user_id     = Column(String(255), nullable=True)
    requester_agent_ident = Column(String(255), nullable=True)

    surface    = Column(String(50), nullable=False, default="unknown")
    session_id = Column(String(255), nullable=True)

    __table_args__ = (
        Index("idx_guard_approvals_ws_status", "workspace_id", "status", "created_at"),
        Index("idx_guard_approvals_ws_requester", "workspace_id", "requester_email"),
        Index(
            "idx_guard_approvals_source_run", "source_run_id",
            postgresql_where=sa.text("source_run_id IS NOT NULL"),
        ),
        Index(
            "idx_guard_approvals_pending_timeout", "timeout_at",
            postgresql_where=sa.text("status = 'pending'"),
        ),
    )

    source_run_id   = Column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="SET NULL"), nullable=True)
    source_block_id = Column(String(255), nullable=True)

    approval_group = Column(String(100), nullable=True)
    approval_type  = Column(String(20), nullable=False, default="any_authorized")

    status             = Column(String(20), nullable=False, default="pending")
    decided_by_email   = Column(String(255), nullable=True)
    decided_by_user_id = Column(String(255), nullable=True)
    decided_reason     = Column(Text, nullable=True)
    decided_at         = Column(DateTime(timezone=True), nullable=True)
    latency_ms         = Column(BigInteger, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    timeout_at = Column(DateTime(timezone=True), nullable=False)
