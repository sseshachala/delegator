"""
Guard workspace config endpoints.

GET   /guard/config?workspace_id          — get config (creates if not exists)
PATCH /guard/config?workspace_id          — update alert_channel, notify_on_block, notify_on_budget
GET   /guard/config/installed?workspace_id — returns {installed, workspace_id, invite_code}
"""
import secrets
import uuid
from datetime import datetime, timezone

import structlog

log = structlog.get_logger(__name__)

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.auth import get_workspace_id, get_user_id, require_workspace_role, require_permission
from app.core.database import get_db
from app.models.workspace import Workspace
from app.modules.guard.models import GuardConfig, GuardMemberConfig, GuardSpendBudget, WorkspaceSkillPack

router = APIRouter(prefix="/guard/config", tags=["guard-config"])


# ── Pydantic models ────────────────────────────────────────────────────────────

_VALID_ENFORCEMENT_MODES = {"block", "warn", "audit"}


class ConfigOut(BaseModel):
    workspace_id: str
    invite_code: str
    slug: str | None
    alert_channel: str | None
    alert_slack_integration_id: str | None
    enforcement_mode: str
    fail_mode: str
    notify_on_block: bool
    notify_on_budget: bool
    automation_security_scan: bool = False
    automation_workflow_trigger: bool = False
    created_at: datetime
    updated_at: datetime | None
    automation_warnings: list[str] = []
    deny_on_error: bool = True
    advisory_mode: bool = False
    notify_on_fail_open: bool = True
    arg_anomaly_enabled: bool = False
    arg_anomaly_zscore_threshold: float | None = None
    arg_anomaly_min_samples: int | None = None
    spend_limit_usd: float | None = None

    class Config:
        from_attributes = True


class ConfigPatch(BaseModel):
    alert_channel: str | None = None
    alert_slack_integration_id: str | None = None
    enforcement_mode: str | None = None
    fail_mode: str | None = None
    notify_on_block: bool | None = None
    notify_on_budget: bool | None = None
    automation_security_scan: bool | None = None
    automation_workflow_trigger: bool | None = None
    deny_on_error: bool | None = None
    advisory_mode: bool | None = None
    notify_on_fail_open: bool | None = None
    arg_anomaly_enabled: bool | None = None
    # NULL leaves the override untouched; the module default applies whenever
    # the column is NULL. See app.modules.behavior.arg_anomaly.Thresholds.
    arg_anomaly_zscore_threshold: float | None = None
    arg_anomaly_min_samples: int | None = None


class InstallStatusOut(BaseModel):
    installed: bool
    workspace_id: str | None = None
    invite_code: str | None = None
    member_token: str | None = None
    agent_token: str | None = None
    user_email: str | None = None
    clerk_user_id: str | None = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_or_create_config(db: Session, workspace_id: str) -> GuardConfig:
    """Return existing GuardConfig or create one (with seeded policies)."""
    ws_uuid = uuid.UUID(workspace_id)
    config = db.query(GuardConfig).filter(GuardConfig.workspace_id == ws_uuid).first()
    if config:
        return config

    config = GuardConfig(
        workspace_id=ws_uuid,
        invite_code=secrets.token_hex(16),
    )
    db.add(config)
    db.flush()
    # Auto-install conduct-base for new workspaces
    existing = db.get(WorkspaceSkillPack, (ws_uuid, "conduct-base"))
    if not existing:
        db.add(WorkspaceSkillPack(workspace_id=ws_uuid, pack_slug="conduct-base", installed_by="system:onboard"))
    db.commit()
    db.refresh(config)
    log.info("guard.config_created", workspace_id=workspace_id)
    return config


def _config_to_out(cfg: GuardConfig) -> ConfigOut:
    return ConfigOut(
        workspace_id=str(cfg.workspace_id),
        invite_code=cfg.invite_code,
        slug=cfg.slug,
        alert_channel=cfg.alert_channel,
        alert_slack_integration_id=str(cfg.alert_slack_integration_id) if cfg.alert_slack_integration_id else None,
        enforcement_mode=cfg.enforcement_mode,
        fail_mode=getattr(cfg, "fail_mode", "fail_open"),
        notify_on_block=cfg.notify_on_block,
        notify_on_budget=cfg.notify_on_budget,
        automation_security_scan=bool(cfg.automation_security_scan),
        automation_workflow_trigger=bool(cfg.automation_workflow_trigger),
        deny_on_error=getattr(cfg, "deny_on_error", True),
        advisory_mode=bool(getattr(cfg, "advisory_mode", False)),
        notify_on_fail_open=bool(getattr(cfg, "notify_on_fail_open", True)),
        arg_anomaly_enabled=bool(getattr(cfg, "arg_anomaly_enabled", False)),
        arg_anomaly_zscore_threshold=getattr(cfg, "arg_anomaly_zscore_threshold", None),
        arg_anomaly_min_samples=getattr(cfg, "arg_anomaly_min_samples", None),
        created_at=cfg.created_at,
        updated_at=cfg.updated_at,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/installed", response_model=InstallStatusOut)
def get_install_status(
    db: Session = Depends(get_db),
    workspace_id: str = Depends(get_workspace_id),
    user_id: str = Depends(get_user_id),
):
    """Return whether Guard is installed for the workspace.
    Guard is considered installed if ANY workspace in the org has a GuardConfig row.
    Auto-provisions a guard_member_config entry for the calling user if they are
    a workspace member (idempotent).
    """
    try:
        ws_uuid = uuid.UUID(workspace_id)
    except ValueError:
        return InstallStatusOut(installed=False)

    # Check org-level: org_id → owner_id (same person's workspaces) → single workspace
    ws = db.query(Workspace).filter(Workspace.id == ws_uuid).first()
    if ws and ws.org_id:
        org_ws_subq = db.query(Workspace.id).filter(Workspace.org_id == ws.org_id)
    elif ws and ws.owner_id:
        org_ws_subq = db.query(Workspace.id).filter(Workspace.owner_id == ws.owner_id)
    else:
        org_ws_subq = db.query(Workspace.id).filter(Workspace.id == ws_uuid)
    config = db.query(GuardConfig).filter(GuardConfig.workspace_id.in_(org_ws_subq)).first()
    if not config:
        return InstallStatusOut(installed=False)

    # Idempotent member provisioning — only for human Clerk sessions (not machine tokens)
    if not user_id:
        return InstallStatusOut(installed=True, workspace_id=workspace_id)

    try:
        from sqlalchemy import text
        existing = db.execute(
            text("""
                SELECT 1 FROM guard_member_config
                WHERE workspace_id = :ws AND clerk_user_id = :uid
                LIMIT 1
            """),
            {"ws": workspace_id, "uid": user_id},
        ).fetchone()
        if not existing:
            db.execute(
                text("""
                    INSERT INTO guard_member_config (workspace_id, clerk_user_id, member_token, active, joined_at)
                    VALUES (:ws, :uid, :token, true, :now)
                    ON CONFLICT (workspace_id, clerk_user_id) DO NOTHING
                """),
                {
                    "ws": workspace_id,
                    "uid": user_id,
                    "token": secrets.token_hex(32),
                    "now": datetime.now(timezone.utc),
                },
            )
            db.commit()
            log.info("guard.member_provisioned", workspace_id=workspace_id, user_id=user_id)
    except Exception:
        db.rollback()  # non-fatal

    # Fetch member_token + agent_identity_id for the calling user so CLI can use it
    token_row = db.execute(
        text("SELECT member_token, agent_identity_id FROM guard_member_config WHERE workspace_id = :ws AND clerk_user_id = :uid LIMIT 1"),
        {"ws": workspace_id, "uid": user_id},
    ).fetchone()

    # Resolve agent_token — decrypt existing FK, re-mint if missing/expired/decrypt fails
    agent_token: str | None = None
    if token_row and token_row.agent_identity_id:
        from app.modules.agent_identity.models import AgentIdentity
        from app.core.crypto import decrypt as _decrypt
        from datetime import datetime, timezone as _tz
        ai_row = db.query(AgentIdentity).filter(AgentIdentity.id == token_row.agent_identity_id).first()
        if ai_row and ai_row.token_encrypted:
            expired = ai_row.expires_at and ai_row.expires_at < datetime.now(_tz.utc)
            if not expired:
                try:
                    agent_token = _decrypt(ai_row.token_encrypted).get("token")
                except Exception:
                    pass

    if not agent_token:
        from app.modules.agent_identity.router import mint_agent_identity
        try:
            identity_row, agent_token = mint_agent_identity(db, workspace_id, f"{user_id} (auto)")
            db.flush()  # persist agent_identity row before FK reference
            if token_row:
                db.execute(
                    text("UPDATE guard_member_config SET agent_identity_id = :aid WHERE workspace_id = :ws AND clerk_user_id = :uid"),
                    {"aid": identity_row.id, "ws": workspace_id, "uid": user_id},
                )
            else:
                db.execute(
                    text("INSERT INTO guard_member_config (workspace_id, clerk_user_id, agent_identity_id, active, joined_at) VALUES (:ws, :uid, :aid, true, now()) ON CONFLICT DO NOTHING"),
                    {"ws": workspace_id, "uid": user_id, "aid": identity_row.id},
                )
            db.commit()
        except Exception as _e:
            log.error("agent_identity.lazy_mint_failed", error=str(_e), workspace_id=workspace_id, user_id=user_id)
            agent_token = None

    from app.core.auth import get_clerk_user_email as _get_email
    user_email = _get_email(user_id)
    return InstallStatusOut(
        installed=True,
        workspace_id=workspace_id,
        invite_code=config.invite_code,
        member_token=token_row.member_token if token_row else None,
        agent_token=agent_token,
        user_email=user_email,
        clerk_user_id=user_id,
    )


@router.get("", response_model=ConfigOut)
def get_config(
    db: Session = Depends(get_db),
    workspace_id: str = Depends(get_workspace_id),
):
    """Return Guard config for the workspace, creating it if it does not yet exist."""
    config = _get_or_create_config(db, workspace_id)
    ws_uuid = uuid.UUID(workspace_id)
    workspace_budget = (
        db.query(GuardSpendBudget)
        .filter(GuardSpendBudget.workspace_id == ws_uuid, GuardSpendBudget.clerk_user_id.is_(None))
        .first()
    )
    out = _config_to_out(config)
    return out.model_copy(update={"spend_limit_usd": workspace_budget.monthly_limit_usd if workspace_budget else None})


@router.patch("", response_model=ConfigOut)
def patch_config(
    body: ConfigPatch,
    db: Session = Depends(get_db),
    workspace_id: str = Depends(get_workspace_id),
    _: str = Depends(require_permission("guard.settings.edit")),
):
    """Update Guard notification/channel settings for the workspace.

    Requires ``guard.settings.edit`` (admin). Prior versions of this
    endpoint had no explicit permission dependency — any authenticated
    caller in the workspace could flip fail_mode / deny_on_error /
    enforcement_mode. Sibling endpoints (persona, runtime-persona) have
    always enforced this permission; the general PATCH did not, until
    #1520 PR 2. This closes that gap for every field the endpoint
    accepts, not just notify_on_fail_open.
    """
    from fastapi import HTTPException
    config = _get_or_create_config(db, workspace_id)
    if body.alert_channel is not None:
        config.alert_channel = body.alert_channel
    if body.alert_slack_integration_id is not None:
        import uuid as _uuid
        config.alert_slack_integration_id = _uuid.UUID(body.alert_slack_integration_id) if body.alert_slack_integration_id else None
    if body.enforcement_mode is not None:
        if body.enforcement_mode not in _VALID_ENFORCEMENT_MODES:
            raise HTTPException(
                status_code=422,
                detail=f"enforcement_mode must be one of: {', '.join(sorted(_VALID_ENFORCEMENT_MODES))}",
            )
        config.enforcement_mode = body.enforcement_mode
    if body.fail_mode is not None:
        if body.fail_mode not in {"fail_open", "fail_closed"}:
            raise HTTPException(
                status_code=422,
                detail="fail_mode must be 'fail_open' or 'fail_closed'",
            )
        config.fail_mode = body.fail_mode
    if body.notify_on_block is not None:
        config.notify_on_block = body.notify_on_block
    if body.notify_on_budget is not None:
        config.notify_on_budget = body.notify_on_budget
    if body.automation_security_scan is not None:
        config.automation_security_scan = body.automation_security_scan
    if body.automation_workflow_trigger is not None:
        config.automation_workflow_trigger = body.automation_workflow_trigger
    if body.deny_on_error is not None:
        config.deny_on_error = body.deny_on_error
    if body.notify_on_fail_open is not None:
        config.notify_on_fail_open = body.notify_on_fail_open
    if body.advisory_mode is not None:
        config.advisory_mode = body.advisory_mode
    if body.arg_anomaly_enabled is not None:
        config.arg_anomaly_enabled = body.arg_anomaly_enabled
    if body.arg_anomaly_zscore_threshold is not None:
        if body.arg_anomaly_zscore_threshold <= 0:
            raise HTTPException(status_code=422, detail="arg_anomaly_zscore_threshold must be greater than 0")
        config.arg_anomaly_zscore_threshold = body.arg_anomaly_zscore_threshold
    if body.arg_anomaly_min_samples is not None:
        if body.arg_anomaly_min_samples < 1:
            raise HTTPException(status_code=422, detail="arg_anomaly_min_samples must be at least 1")
        config.arg_anomaly_min_samples = body.arg_anomaly_min_samples
    config.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(config)

    out = _config_to_out(config)
    return out


@router.delete("", status_code=204)
def delete_config(
    db: Session = Depends(get_db),
    workspace_id: str = Depends(get_workspace_id),
):
    """Uninstall Guard for the workspace — removes guard_config and all guard_member_config rows."""
    try:
        ws_uuid = uuid.UUID(workspace_id)
    except ValueError:
        return
    db.query(GuardConfig).filter(GuardConfig.workspace_id == ws_uuid).delete()
    db.query(GuardMemberConfig).filter(GuardMemberConfig.workspace_id == ws_uuid).delete()
    db.commit()
    log.info("guard.config_deleted", workspace_id=workspace_id)


class ResyncOut(BaseModel):
    ok: bool
    resync_requested_at: str


_VALID_PERSONAS = {"agent", "proxy"}


class PersonaOut(BaseModel):
    persona: str                                  # active dev persona for the calling user
    assigned_by: str                              # 'user' or 'admin'
    workspace_default: str                        # workspace dev_persona default
    workspace_runtime_persona: str = "conservative"  # admin-only, applies to workflow runtime


class PersonaPatch(BaseModel):
    persona: str
    developer_id: str | None = None   # if set, override for that member only; else workspace default


class RuntimePersonaPatch(BaseModel):
    persona: str


@router.get("/persona", response_model=PersonaOut)
def get_persona(
    workspace_id: str = Depends(get_workspace_id),
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db),
):
    """Return the active persona for the calling user.
    Resolves: member override → workspace default → 'standard'.
    """
    try:
        ws_uuid = uuid.UUID(workspace_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid workspace_id")

    cfg = db.query(GuardConfig).filter(GuardConfig.workspace_id == ws_uuid).first()
    workspace_default = (cfg.persona if cfg and cfg.persona else "agent")
    workspace_runtime = (cfg.runtime_persona if cfg and cfg.runtime_persona else "conservative")

    member = (
        db.query(GuardMemberConfig)
        .filter(
            GuardMemberConfig.workspace_id == ws_uuid,
            GuardMemberConfig.clerk_user_id == user_id,
        )
        .first()
    )

    if member and member.persona:
        return PersonaOut(
            persona=member.persona,
            assigned_by=member.assigned_by or "user",
            workspace_default=workspace_default,
            workspace_runtime_persona=workspace_runtime,
        )

    return PersonaOut(
        persona=workspace_default,
        assigned_by="user",
        workspace_default=workspace_default,
        workspace_runtime_persona=workspace_runtime,
    )


@router.patch("/persona", response_model=PersonaOut)
def set_persona(
    body: PersonaPatch,
    workspace_id: str = Depends(get_workspace_id),
    _: str = Depends(require_permission("guard.settings.edit")),
    db: Session = Depends(get_db),
):
    """Admin sets the persona for the workspace or a specific developer.

    - No developer_id → sets workspace default for all members
    - developer_id → sets per-member override (assigned_by: admin, locked)
    """
    if body.persona not in _VALID_PERSONAS:
        raise HTTPException(status_code=422, detail=f"persona must be one of {sorted(_VALID_PERSONAS)}")

    try:
        ws_uuid = uuid.UUID(workspace_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid workspace_id")

    if body.developer_id:
        # Per-member override — admin-locked
        member = (
            db.query(GuardMemberConfig)
            .filter(
                GuardMemberConfig.workspace_id == ws_uuid,
                GuardMemberConfig.clerk_user_id == body.developer_id,
            )
            .first()
        )
        if not member:
            raise HTTPException(status_code=404, detail="Developer not in this workspace")
        member.persona = body.persona
        member.assigned_by = "admin"
        db.commit()
        log.info("guard.persona_set_member", workspace_id=workspace_id, developer_id=body.developer_id, persona=body.persona)
        cfg = db.query(GuardConfig).filter(GuardConfig.workspace_id == ws_uuid).first()
        workspace_default = cfg.persona if cfg else "agent"
        workspace_runtime = cfg.runtime_persona if cfg else "agent"
        return PersonaOut(
            persona=body.persona, assigned_by="admin",
            workspace_default=workspace_default,
            workspace_runtime_persona=workspace_runtime,
        )

    # Workspace-wide default
    cfg = db.query(GuardConfig).filter(GuardConfig.workspace_id == ws_uuid).first()
    if not cfg:
        raise HTTPException(status_code=404, detail="Guard not installed for this workspace")
    cfg.persona = body.persona
    db.commit()
    log.info("guard.persona_set_workspace", workspace_id=workspace_id, persona=body.persona)
    return PersonaOut(
        persona=body.persona, assigned_by="admin",
        workspace_default=body.persona,
        workspace_runtime_persona=cfg.runtime_persona or "agent",
    )


@router.patch("/runtime-persona", response_model=PersonaOut)
def set_runtime_persona(
    body: RuntimePersonaPatch,
    workspace_id: str = Depends(get_workspace_id),
    _: str = Depends(require_permission("guard.settings.edit")),
    db: Session = Depends(get_db),
):
    """Admin sets the runtime persona for the workspace. Applies only to
    workflow execution (guard_block.py) - independent of dev/MCP persona."""
    if body.persona not in _VALID_PERSONAS:
        raise HTTPException(status_code=422, detail=f"persona must be one of {sorted(_VALID_PERSONAS)}")
    try:
        ws_uuid = uuid.UUID(workspace_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid workspace_id")

    cfg = db.query(GuardConfig).filter(GuardConfig.workspace_id == ws_uuid).first()
    if not cfg:
        raise HTTPException(status_code=404, detail="Guard not installed for this workspace")
    cfg.runtime_persona = body.persona
    db.commit()

    from app.modules.guard.policy_engine import invalidate_policy_cache
    invalidate_policy_cache(db, ws_uuid)
    db.commit()

    log.info("guard.runtime_persona_set", workspace_id=workspace_id, persona=body.persona)
    return PersonaOut(
        persona=cfg.persona or "agent",
        assigned_by="admin",
        workspace_default=cfg.persona or "agent",
        workspace_runtime_persona=body.persona,
    )


@router.post("/resync", response_model=ResyncOut, status_code=200)
def request_resync(
    workspace_id: str = Query(...),
    db: Session = Depends(get_db),
    _role: str = Depends(require_workspace_role("admin", "developer")),
):
    """Bump resync_requested_at so the CLI detects a version change within its next 60s poll."""
    try:
        ws_uuid = uuid.UUID(workspace_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid workspace_id")
    config = db.query(GuardConfig).filter(GuardConfig.workspace_id == ws_uuid).first()
    if not config:
        raise HTTPException(status_code=404, detail="Guard not installed")
    config.resync_requested_at = datetime.now(timezone.utc)
    db.commit()
    log.info("guard.resync_requested", workspace_id=workspace_id)
    return ResyncOut(ok=True, resync_requested_at=config.resync_requested_at.isoformat())


class JoinIn(BaseModel):
    invite_code: str
    email: str


class JoinOut(BaseModel):
    workspace_id: str
    member_token: str
    agent_token: str | None = None
    policy: dict


# Standalone router so this doesn't need /guard/config prefix
join_router = APIRouter(prefix="/guard", tags=["guard-config"])


@join_router.post("/join", response_model=JoinOut)
def join_guard(body: JoinIn, db: Session = Depends(get_db)):
    """Developer joins Guard via invite code. Returns workspace_id + member_token + policy."""
    import uuid
    config = db.query(GuardConfig).filter(GuardConfig.invite_code == body.invite_code).first()
    if not config:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Invalid invite code")

    workspace_id = str(config.workspace_id)

    # Upsert guard_member_config keyed on (workspace_id, email)
    from sqlalchemy import text
    existing = db.execute(
        text("""
            SELECT member_token FROM guard_member_config
            WHERE workspace_id = :ws AND clerk_user_id = :email
            LIMIT 1
        """),
        {"ws": workspace_id, "email": body.email},
    ).fetchone()

    if existing:
        member_token = existing.member_token
        agent_token: str | None = None
    else:
        from app.modules.agent_identity.router import mint_agent_identity
        member_token = secrets.token_hex(32)
        identity_row, agent_token = mint_agent_identity(
            db, workspace_id, name=f"{body.email} (auto)"
        )
        db.execute(
            text("""
                INSERT INTO guard_member_config (workspace_id, clerk_user_id, member_token, agent_identity_id, active, joined_at)
                VALUES (:ws, :email, :token, :agent_id, true, :now)
                ON CONFLICT (workspace_id, clerk_user_id) DO NOTHING
            """),
            {
                "ws": workspace_id,
                "email": body.email,
                "token": member_token,
                "agent_id": identity_row.id,
                "now": datetime.now(timezone.utc),
            },
        )
        db.commit()
        log.info("guard.developer_joined", workspace_id=workspace_id, email=body.email)

    # Active ruleset comes from skill_packs JSONB via compute_policy().
    # Persona is read from guard_config above (config.persona, defaults to 'standard').
    # compute_policy already filters to enabled packs per WorkspaceSkillPack — a rule
    # from a pack that is not installed will not appear here.
    from app.modules.guard.enforcement import is_hook_applicable_rule
    from app.modules.guard.policy_engine import compute_policy
    persona = (config.persona or "agent")
    computed = compute_policy(db, config.workspace_id, persona)
    rules = [
        {
            "rule_id":           r.get("id") or r.get("rule_id"),
            "match_tool":        r.get("match_tool") or "*",
            "match_ai_tool":     r.get("match_ai_tool"),
            "match_pattern":     r.get("match_pattern"),
            "match_path_pattern": r.get("match_path_pattern"),
            "action":            r.get("action"),
            "message":           r.get("message"),
            # #1048: propagate except_paths so pack authors can declaratively
            # exclude paths without editing hook code.
            "except_paths":      r.get("except_paths"),
            # #1048: source_pack lets operators trace 'which pack put this rule
            # in my local cache' when diagnosing stale-cache issues.
            "source_pack":       r.get("source_pack") or r.get("_pack_slug"),
        }
        for r in computed
        if is_hook_applicable_rule(r)
    ]
    policy = {"workspace_id": workspace_id, "version": "1", "rules": rules}

    return JoinOut(workspace_id=workspace_id, member_token=member_token, agent_token=agent_token, policy=policy)


class InviteRegenOut(BaseModel):
    invite_code: str


@router.post("/invite/regenerate", response_model=InviteRegenOut)
def regenerate_invite(
    db: Session = Depends(get_db),
    workspace_id: str = Depends(get_workspace_id),
):
    """Generate a new invite code for the workspace Guard config."""
    config = _get_or_create_config(db, workspace_id)
    config.invite_code = secrets.token_hex(16)
    config.updated_at = datetime.now(timezone.utc)
    db.commit()
    log.info("guard.invite_regenerated", workspace_id=workspace_id)
    return InviteRegenOut(invite_code=config.invite_code)
