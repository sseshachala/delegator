"""
ConductGuard — Remote MCP server over HTTP.

POST /guard/mcp   — stateless JSON-RPC 2.0 endpoint (MCP HTTP transport)

Claude.ai / Claude for Work users add this URL in their MCP settings:
  https://api.conductai.ai/guard/mcp?workspace_id=<uuid>
  Authorization: Bearer <member_token>

Claude Desktop users can also point here instead of running a local process.
Auth: workspace_id in query, member_token in `Authorization: Bearer` header.
Legacy `?token=` query param accepted for Claude.ai web compat — emits a
deprecation warning (issue #800). Slated for removal in issue #810.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlalchemy.orm import Session

from sqlalchemy import text as _sql

from app.core.database import SessionLocal
from app.core.auth import get_clerk_user_email, resolve_agent_token
from app.core.pii import redact_secrets
from app.models.workspace import Workspace
from app.modules.guard.models import DiscoveredAgent, GuardAuditEvent, GuardConfig, GuardMemberConfig, chain_hash_for_insert, get_policy_hash
from app.modules.guard.models import GuardApprovalRequest
from app.modules.guard import approval as _approval
from app.modules.guard.policy_engine import compute_policy
from app.modules.guard.tool_groups import expand_match_tool

router = APIRouter(prefix="/guard/mcp", tags=["guard-mcp"])

PROTOCOL_VERSION = "2024-11-05"

_TOOLS = [
    {
        "name": "guard_status",
        "description": (
            "Returns current ConductGuard policy status: workspace_id, "
            "workspace_name, your email, number of active rules, and the "
            "policy version timestamp."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "conduct_current_workspace",
        "description": (
            "Return the active workspace this MCP session is scoped to: "
            "workspace_id, workspace_name, and the caller's role in that workspace. "
            "Use when the user asks 'which workspace am I in' or when you (the model) "
            "need to confirm the workspace context before running side-effectful tools. "
            "Read-only; no side effects."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "guard_check",
        "description": (
            "Check the intent about to be executed against team policy. "
            # #997 UX: call once per intent, not per action. Reduces transcript noise.
            "Call ONCE at the start of a task or when scope changes (reads → writes, local → network, "
            "new destination or command family). Do NOT call before every read/write in a batch. "
            "Response: 'ok' or empty means proceed silently — do NOT narrate it. "
            "'BLOCKED — <reason>' means stop and tell the user the rule. "
            "'WARNING — <reason>' means proceed but surface the warning inline. "
            "Pass tool_name as the ACTION FAMILY (e.g. 'bash', 'write_file', 'curl', 'git') "
            "and tool_input as the specific parameters."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool_name":  {"type": "string", "description": "The action you are about to take (e.g. bash, read_file, write_file, curl, git)"},
                "tool_input": {"type": "object", "description": "Relevant parameters — e.g. {\"command\": \"rm -rf /\"} or {\"file_path\": \"/etc/passwd\"}"},
                "conduct_run_id":   {"type": "string", "description": "Conduct run ID if called from within a workflow run — pass the value from your run context."},
                "conduct_workflow": {"type": "string", "description": "Conduct workflow slug if called from within a workflow run."},
                "pack": {"type": "string", "description": "Optional. Scope this check to a specific compliance pack (e.g. 'conduct-eu-ai-act', 'conduct-hipaa'). conduct-base is always enforced on top. Returns ERROR if the pack is not installed for this workspace."},
                "prompt": {"type": "string", "description": "Optional. The prompt or action description being checked. Stored in the audit trail for traceability — useful for agentic apps passing context about why an action is being taken."},
            },
            "required": ["tool_name"],
        },
    },
    {
        "name": "guard_sync",
        "description": "Returns current active ruleset (no-op for remote MCP — policy is always live).",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "guard_enable",
        "description": (
            "Call this when the user asks to 'enable conductguard', 'load mcp', 'activate guard', "
            "or any similar onboarding request. Confirms ConductGuard is connected, returns the "
            "number of active policy rules, and provides the Project Instruction snippet the user "
            "should paste into their Claude.ai Project settings to make guard_check fire automatically."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "guard_spend",
        "description": (
            "Returns LLM spend through the Conduct Guard Proxy, grouped by provider and model. "
            "Use when the user asks 'how much did I spend on LLMs today?' or 'what's our team's "
            "Claude bill this week?'. Optional 'days' argument (default 1, max 30) widens the "
            "window. Only proxy-routed calls are counted."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Lookback window in days (default 1, max 30)"},
            },
            "required": [],
        },
    },
    {
        "name": "guard_local_risks",
        "description": (
            "Returns open local key risk findings — pre-existing real provider API keys "
            "(sk-ant-, sk-, pplx-) detected on developers' machines during conduct guard sync. "
            "Use when the user asks 'do we still have raw API keys on dev laptops?' or for "
            "CISO audit prep. Each finding includes provider, file path, masked fragment, "
            "and which developer it was on."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "guard_activity",
        "description": (
            "ALWAYS call this at the start of every conversation, immediately after the user sends "
            "their first message. Pass a one-line summary of what the user is asking you to do. "
            "This logs session intent to the team's ConductGuard audit trail so admins can see "
            "what work is being done across the team's AI usage."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "One-line summary of what the user is asking you to do in this conversation.",
                },
                "category": {
                    "type": "string",
                    "description": "Optional category: coding, debugging, review, research, writing, devops, security, other",
                },
                "conduct_run_id":   {"type": "string", "description": "Conduct run ID if called from within a workflow run."},
                "conduct_workflow": {"type": "string", "description": "Conduct workflow slug if called from within a workflow run."},
            },
            "required": ["summary"],
        },
    },
    {
        "name": "guard_recent_activity",
        "description": (
            "Read-only: show recent Guard audit events for the caller in this workspace. "
            "Complements guard_activity (which is write-only). Returns a compact list of "
            "'time  decision  rule_id  tool_call' rows so agents can see what they have done recently."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "days":     {"type": "integer", "description": "Window in days (1-30). Default 1.", "default": 1},
                "limit":    {"type": "integer", "description": "Max events to return (1-100). Default 20.", "default": 20},
                "decision": {"type": "string", "description": "Optional filter: allowed / blocked / warned / audited (alias: ok → allowed)"},
                "rule_id":  {"type": "string", "description": "Optional filter to a specific rule_id"},
            },
            "required": [],
        },
    },
    {
        "name": "guard_discover",
        "description": "Show all AI agents discovered in this org and Guard coverage. Returns total, coverage %, and the full agent inventory — each entry has a `governed: true|false` flag so callers can filter to shadow (ungoverned) or governed as needed.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "guard_discover_register",
        "description": "Bring a discovered shadow agent under Guard governance by agent ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent ID from guard_discover results"},
            },
            "required": ["agent_id"],
        },
    },
    {
        "name": "post_finding",
        "description": (
            "Report a security vulnerability or finding directly to Conduct's Security Loop. "
            "Use this when you detect a secret leak, injection risk, path traversal, auth bypass, "
            "or any other security issue in the code you're reviewing. "
            "Conduct will auto-triage it and can trigger an automated fix via the security-autopilot-fix playbook."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool":         {"type": "string", "description": "Tool or scanner that found this (e.g. 'claude_code', 'semgrep', 'bughunter')"},
                "severity":     {"type": "string", "description": "critical | high | medium | low | info"},
                "type":         {"type": "string", "description": "injection | path-traversal | secret-leak | auth-bypass | crypto | guard_violation | other"},
                "description":  {"type": "string", "description": "Clear description of the vulnerability"},
                "file":         {"type": "string", "description": "File path where the issue was found"},
                "line":         {"type": "integer", "description": "Line number"},
                "repo_full_name": {"type": "string", "description": "GitHub repo (e.g. 'org/repo') — required for trigger_fix to open a PR"},
                "suggested_fix": {"type": "string", "description": "Optional suggested remediation"},
            },
            "required": ["tool", "severity", "type", "description"],
        },
    },
    {
        "name": "trigger_fix",
        "description": (
            "Trigger the security-autopilot-fix playbook for a finding that was previously reported via post_finding. "
            "Conduct will open a PR with an automated fix. The finding must have a repo_full_name set."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "finding_id": {"type": "string", "description": "Finding UUID returned by post_finding"},
            },
            "required": ["finding_id"],
        },
    },
    {
        "name": "conduct_list_agents",
        "description": "List all installed agents in your Conduct workspace.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "conduct_list_projects",
        "description": "List all projects in your Conduct workspace.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "conduct_list_playbooks",
        "description": "List available Conduct playbooks (workflow templates).",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "conduct_run_workflow",
        "description": "Trigger a workflow run in Conduct. Returns the run ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string", "description": "The workflow UUID to run"},
                "payload":     {"type": "object", "description": "Optional trigger payload (key-value pairs)"},
            },
            "required": ["workflow_id"],
        },
    },
    {
        "name": "conduct_get_run",
        "description": "Get the status and result of a workflow run.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string"},
                "run_id":      {"type": "string"},
            },
            "required": ["workflow_id", "run_id"],
        },
    },
]



# ── Conduct platform helpers ──────────────────────────────────────────────────

def _list_agents(db, ws_uuid: uuid.UUID) -> list[dict]:
    rows = db.execute(
        _sql("SELECT id, name, status FROM agents WHERE workspace_id = :w ORDER BY name LIMIT 100"),
        {"w": str(ws_uuid)},
    ).fetchall()
    return [{"id": str(r.id), "name": r.name, "status": r.status} for r in rows]


def _list_projects(db, ws_uuid: uuid.UUID) -> list[dict]:
    rows = db.execute(
        _sql("SELECT id, name, description FROM projects WHERE workspace_id = :w ORDER BY name LIMIT 100"),
        {"w": str(ws_uuid)},
    ).fetchall()
    return [{"id": str(r.id), "name": r.name, "description": r.description} for r in rows]


def _list_playbooks(db, ws_uuid: uuid.UUID) -> list[dict]:
    rows = db.execute(
        _sql("SELECT id, name, description FROM playbooks WHERE workspace_id = :w ORDER BY name LIMIT 100"),
        {"w": str(ws_uuid)},
    ).fetchall()
    return [{"id": str(r.id), "name": r.name, "description": r.description} for r in rows]


def _run_workflow(db, ws_uuid: uuid.UUID, workflow_id: str, payload: dict, user_email: str) -> dict:
    row = db.execute(
        _sql("SELECT id FROM workflows WHERE id = :wf AND workspace_id = :ws LIMIT 1"),
        {"wf": workflow_id, "ws": str(ws_uuid)},
    ).fetchone()
    if not row:
        raise ValueError(f"workflow {workflow_id} not found")
    run_id = str(uuid.uuid4())
    db.execute(
        _sql("""
            INSERT INTO runs (id, workflow_id, workspace_id, status, triggered_by, payload, created_at)
            VALUES (:id, :wf, :ws, 'pending', :by, :pl, :ts)
        """),
        {"id": run_id, "wf": workflow_id, "ws": str(ws_uuid),
         "by": user_email, "pl": json.dumps(payload), "ts": datetime.now(timezone.utc)},
    )
    db.commit()
    return {"run_id": run_id, "status": "pending"}


def _get_run_status(db, ws_uuid: uuid.UUID, workflow_id: str, run_id: str) -> dict:
    row = db.execute(
        _sql("SELECT status, outcome, started_at, completed_at FROM runs WHERE id = :r AND workflow_id = :wf AND workspace_id = :ws LIMIT 1"),
        {"r": run_id, "wf": workflow_id, "ws": str(ws_uuid)},
    ).fetchone()
    if not row:
        raise ValueError(f"run {run_id} not found")
    return {
        "status":       row.status,
        "outcome":      row.outcome,
        "started_at":   row.started_at.isoformat() if row.started_at else None,
        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
    }


# ── Surface detection ─────────────────────────────────────────────────────────

def _detect_surface(client_info: dict) -> str:
    name = (client_info.get("name") or "").lower()
    if "desktop" in name:
        return "claude_desktop"
    if "work" in name or "teams" in name or "enterprise" in name:
        return "claude_work"
    if "claude" in name:
        return "claude_chat"
    if "codex" in name:
        return "codex"
    if "cursor" in name:
        return "cursor"
    if "windsurf" in name:
        return "windsurf"
    if "copilot" in name or "github" in name:
        return "copilot"
    return "unknown"


# ── Policy matching ───────────────────────────────────────────────────────────

_ACTION_PRIORITY = {"block": 0, "approval": 1, "warn": 2, "audit": 3}


def _match_policy(tool_name: str, tool_input: dict, rules: list) -> dict | None:
    """Return the most restrictive matching rule (block > approval > warn > audit)."""
    inp_text  = json.dumps(tool_input)
    path_keys = ["file_path", "path", "command"]
    path_text = " ".join(str(tool_input.get(k, "")) for k in path_keys)

    best: dict | None = None
    best_priority = 999

    for rule in rules:
        match_tool = (rule.get("match_tool") or "*").lower()
        if match_tool != "*":
            allowed = expand_match_tool(match_tool)
            if tool_name.lower() not in allowed:
                continue

        pattern = rule.get("match_pattern")
        if pattern:
            try:
                if not re.search(pattern, inp_text, re.IGNORECASE):
                    continue
            except re.error:
                continue

        path_pattern = rule.get("match_path_pattern")
        if path_pattern:
            try:
                if not re.search(path_pattern, path_text, re.IGNORECASE):
                    continue
            except re.error:
                continue

        priority = _ACTION_PRIORITY.get(rule.get("action", "audit"), 3)
        if priority < best_priority:
            best_priority = priority
            best = rule

    return best


# ── DB helpers ────────────────────────────────────────────────────────────────

_APPROVAL_FIELDS = ("approval_group", "approval_type", "approval_timeout_sec", "approval_notification", "guidance", "inject_guidance")


def _project_rule(r: dict) -> dict:
    """Trim a raw rule to the fields the matcher + downstream handlers use.
    Preserves approval_* fields so action=approval can honour rule spec.
    Preserves enforcement contract so surface-scoped matchers can honour
    proxy/hook/mcp/runtime = not_supported / conditional / hard."""
    out = {
        "rule_id":           r.get("id") or r.get("rule_id"),
        "match_tool":        r.get("match_tool"),
        "match_pattern":     r.get("match_pattern"),
        "match_path_pattern": r.get("match_path_pattern"),
        "action":            r.get("action"),
        "message":           r.get("message"),
        "pack":              r.get("pack") or r.get("pack_slug"),
        "enforcement":       r.get("enforcement") or {},
    }
    for k in _APPROVAL_FIELDS:
        if k in r and r[k] is not None:
            out[k] = r[k]
    return out


def _get_rules(db: Session, ws_uuid: uuid.UUID) -> list[dict]:
    """Active ruleset for the agent persona — governs what AI does on the machine."""
    rules = compute_policy(db, ws_uuid, "agent")
    return [_project_rule(r) for r in rules]


_PERSONAS = ["agent", "proxy"]

def _get_rules_for_pack(db: Session, ws_uuid: uuid.UUID, pack_slug: str) -> list[dict] | str:
    """Rules from conduct-base + pack_slug only. Returns error string if pack not installed."""
    if pack_slug == "conduct-base":
        return 'ERROR — "conduct-base" is always enforced and cannot be selected directly. Use a compliance pack (e.g. "conduct-eu-ai-act").'
    row = db.execute(_sql("SELECT 1 FROM workspace_skill_packs WHERE workspace_id=:ws AND pack_slug=:p"),
                     {"ws": ws_uuid, "p": pack_slug}).fetchone()
    if not row:
        installed = [r[0] for r in db.execute(_sql(
            "SELECT pack_slug FROM workspace_skill_packs WHERE workspace_id=:ws AND pack_slug != 'conduct-base'"
        ), {"ws": ws_uuid}).fetchall()]
        available = ", ".join(sorted(installed)) or "none"
        return f'ERROR — pack "{pack_slug}" is not installed for this workspace. Installed packs: {available}.'
    rules: dict[str, dict] = {}
    for slug in ("conduct-base", pack_slug):
        sp = db.execute(_sql(
            "SELECT rules FROM skill_packs WHERE slug=:slug ORDER BY published_at DESC LIMIT 1"
        ), {"slug": slug}).fetchone()
        if not sp:
            continue
        for rule in (sp[0] or []):
            persona = rule.get("persona")
            if persona and persona != "agent":
                continue
            if not persona and "agent" not in rule.get("persona_affinity", _PERSONAS):
                continue
            projected = _project_rule(rule)
            projected["pack"] = projected.get("pack") or slug
            rules[rule["id"]] = projected
    return list(rules.values())


def _record_event(
    db: Session,
    ws_uuid: uuid.UUID,
    tool_name: str,
    tool_input: dict,
    decision: str,
    rule_id: str | None,
    ai_tool: str,
    user_email: str,
    session_id: str,
    *,
    conductai_run_id: str | None = None,
    conductai_workflow: str | None = None,
    prompt: str | None = None,
    source: str = "mcp",
    rule_message: str | None = None,
) -> None:
    ts = datetime.now(timezone.utc)
    prev_hash, entry_hash = chain_hash_for_insert(db, ws_uuid, ts, tool_name, decision)
    policy_hash = get_policy_hash(db, ws_uuid)

    raw_summary = redact_secrets(json.dumps(tool_input)[:500])[0][:200]
    if prompt:
        raw_summary = f"[prompt: {prompt[:100]}] {raw_summary}"

    event = GuardAuditEvent(
        workspace_id=ws_uuid,
        clerk_user_id=user_email,
        user_email=user_email,
        ai_tool=ai_tool,
        tool_call=tool_name,
        source=source,
        input_summary=raw_summary,
        decision=decision,
        rule_id=rule_id,
        rule_message=rule_message,
        hook_session_id=session_id,
        ts=ts,
        conductai_run_id=conductai_run_id,
        conductai_workflow=conductai_workflow,
        previous_hash=prev_hash,
        entry_hash=entry_hash,
        policy_hash=policy_hash,
    )
    db.add(event)
    db.commit()

    # Slack + webhook + PagerDuty + email fan-out for blocks/warns.
    # Uses caller-provided `source` so Guard Activity attribution is precise
    # (mcp vs hook vs runtime vs proxy).
    if decision in ("blocked", "warned"):
        try:
            from app.modules.guard.routers.events import notify_guard_block
            notify_guard_block(db, ws_uuid, decision=decision, rule_id=rule_id,
                               user_email=user_email, tool=tool_name, source=source)
        except Exception:
            pass


# ── JSON-RPC response helpers ─────────────────────────────────────────────────

def _ok(msg_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _err(msg_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


# Governance: when a tool_call handler runs, we stamp the response text with
# [ws:xxxxxxxx] so the model can spot silent workspace drift (dashboard switch,
# token rotation, etc) without having to poll guard_status every turn.
# Init / OAuth responses leave this unset, so their text stays unchanged.
_tool_ws_ctx: ContextVar[uuid.UUID | None] = ContextVar("guard_mcp_tool_ws", default=None)


def _text(msg_id, text: str) -> dict:
    _ws = _tool_ws_ctx.get()
    if _ws is not None:
        text = f"[ws:{str(_ws)[:8]}] {text}"
    return _ok(msg_id, {"content": [{"type": "text", "text": text}]})


# ── Main endpoint ─────────────────────────────────────────────────────────────

def _extract_token(request: Request) -> str | None:
    """Authorization header only. Header may be `Bearer <token>` or a bare token
    (Smithery-style). ?token= query param was retired with issue #800."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip() or None
    if auth:
        return auth.strip() or None
    return None


@router.get("")
async def mcp_sse(
    request: Request,
    workspace_id: str | None = Query(None),
):
    """SSE endpoint required by MCP Streamable HTTP transport (GET establishes the stream).
    For stateless policy checks we don't push server-initiated messages, so this just
    holds the connection open with keepalive pings until the client disconnects.

    Auth: Authorization: Bearer <token> header required.
    workspace_id is optional — when omitted, the Bearer token identifies the workspace.
    """
    import asyncio

    if not _extract_token(request):
        ua = request.headers.get("User-Agent", "")
        # Only send OAuth discovery header to clients that support it.
        # Smithery and similar tools fall back to API key auth — sending WWW-Authenticate
        # breaks them. Claude.ai sends "claude-mcp"; VS Code Copilot sends "github-copilot".
        _ua = ua.lower()
        is_claude = "claude" in _ua or "github-copilot" in _ua or "vscode" in _ua or "cursor" in _ua
        resp_headers = {}
        if is_claude:
            ws_param = f"?workspace_id={workspace_id}" if workspace_id else ""
            resp_headers["WWW-Authenticate"] = (
                'Bearer realm="https://api.conductai.ai/guard/mcp",'
                f' resource_metadata="https://api.conductai.ai/.well-known/oauth-protected-resource/guard/mcp{ws_param}"'
            )
        return JSONResponse(
            status_code=401,
            content={"error": "missing or invalid token"},
            headers=resp_headers,
        )

    # Build the POST endpoint URL from the request so it works across envs.
    # Behind Render's TLS proxy request.base_url is http:// unless uvicorn is
    # started with --proxy-headers; trust X-Forwarded-Proto as fallback so the
    # SSE endpoint origin matches the connection origin (MCP requires match).
    fwd_proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
    base_url = str(request.base_url).rstrip("/")
    if fwd_proto in ("http", "https") and "://" in base_url:
        base_url = f"{fwd_proto}://{base_url.split('://', 1)[1]}"
    post_url = f"{base_url}/guard/mcp"

    async def event_stream():
        # MCP SSE transport: client waits for this before sending initialize
        yield f"event: endpoint\ndata: {post_url}\n\n"
        yield ": keepalive\n\n"
        while True:
            await asyncio.sleep(15)
            yield ": keepalive\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("")
async def mcp_terminate():
    """Streamable HTTP session termination — stateless server, just ack."""
    return JSONResponse(status_code=204, content=None)


@router.post("")
async def mcp_endpoint(
    request: Request,
    workspace_id: str | None = Query(None, description="Guard workspace UUID — optional when using OAuth Bearer token"),
):
    """Stateless MCP JSON-RPC endpoint for Claude.ai / Claude Desktop / Claude for Work."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid JSON"})

    msg_id = body.get("id")
    method = body.get("method", "")
    params = body.get("params") or {}

    resolved_token = _extract_token(request)
    if not resolved_token:
        return JSONResponse(status_code=401, content=_err(msg_id, -32600, "missing token (use Authorization: Bearer)"))

    db = SessionLocal()
    try:
        ident = resolve_agent_token(resolved_token, db)
        if not ident:
            return JSONResponse(status_code=401, content=_err(msg_id, -32600, "invalid token"))
        _ws_id_str, clerk_user_id = ident

        # Validate workspace matches URL param if provided
        if workspace_id:
            try:
                ws_uuid = uuid.UUID(workspace_id)
            except ValueError:
                return JSONResponse(status_code=422, content=_err(msg_id, -32600, "invalid workspace_id"))
            if str(ws_uuid) != _ws_id_str:
                return JSONResponse(status_code=401, content=_err(msg_id, -32600, "token does not belong to this workspace"))
        else:
            ws_uuid = uuid.UUID(_ws_id_str)

        user_email = get_clerk_user_email(clerk_user_id) or clerk_user_id

        # ponytail: auto-provision GuardConfig on first MCP call — token already
        # authenticated, workspace exists, missing config was a 401 loop for
        # Claude.ai OAuth (89cc839 re-OAuth trick never converges).
        from app.modules.guard.routers.config import _get_or_create_config
        config = _get_or_create_config(db, str(ws_uuid))

        if method == "initialize":
            client_info = params.get("clientInfo") or {}
            surface = _detect_surface(client_info)
            # Echo the client's requested protocolVersion so strict clients (Copilot
            # rmcp) don't treat the connection as a version mismatch and skip
            # tools/list. Fall back to our default when the client omits it.
            negotiated_version = params.get("protocolVersion") or PROTOCOL_VERSION
            # Streamable HTTP: clients (Copilot rmcp, etc) expect Mcp-Session-Id.
            # We're stateless, so any stable-per-response uuid satisfies the contract.
            session_hdr = str(uuid.uuid4())
            return JSONResponse(
                _ok(msg_id, {
                    "protocolVersion": negotiated_version,
                    "capabilities":    {"tools": {}},
                    "serverInfo":      {"name": "conductguard", "version": "1.0.0"},
                    "_surface":        surface,
                    "instructions": (
                        "ConductGuard is active and enforcing your team's security policy. "
                        # #997: once per intent, not per action — protects the transcript.
                        "Call guard_activity ONCE at the start of a user request with a one-line summary. "
                        "Call guard_check ONCE per intent (not per file/command). Re-check only when scope changes "
                        "— e.g. moving from reads to writes, from local files to network, or entering a new task. "
                        "If the response is BLOCKED: stop and explain the policy rule to the user. "
                        "If WARNING: proceed but surface the warning inline. "
                        "If the response is empty or 'ok': proceed silently — do not narrate it in the chat. "
                        "Do not repeat guard_check for the same intent within a single response."
                    ),
                }),
                headers={"Mcp-Session-Id": session_hdr},
            )

        elif method == "notifications/initialized":
            # Response() not JSONResponse(content=None): the latter serializes
            # `null` (4 bytes) with Content-Length: 0, uvicorn raises
            # RuntimeError("Response content longer than Content-Length"),
            # Claude.ai's toolbox proxy sees a broken response and returns 502.
            return Response(status_code=204)

        elif method == "tools/list":
            return JSONResponse(_ok(msg_id, {"tools": _TOOLS}))

        elif method == "tools/call":
            # ponytail: stamps every tool response with [ws:xxxxxxxx] so the
            # model can detect silent workspace changes without polling.
            _tool_ws_ctx.set(ws_uuid)
            tool_name = params.get("name", "")
            arguments = params.get("arguments") or {}
            # Try explicit surface header, then clientInfo (rarely on tools/call),
            # then User-Agent (Copilot rmcp reveals itself here). Default to
            # 'unknown' — misattributing to claude_chat hides Copilot traffic.
            ai_tool = (
                request.headers.get("x-claude-surface")
                or _detect_surface(params.get("clientInfo") or {})
                or "unknown"
            )
            if ai_tool == "unknown":
                ua_surface = _detect_surface({"name": request.headers.get("User-Agent", "")})
                if ua_surface != "unknown":
                    ai_tool = ua_surface
            session_id = request.headers.get("x-session-id", str(uuid.uuid4()))

            # Self-register: every tool call proves this agent is under Guard.
            try:
                _now = datetime.now(timezone.utc)
                db.execute(_sql("""
                    INSERT INTO discovered_agents
                        (id, workspace_id, name, framework, source, location, under_guard, first_seen_at, last_seen_at)
                    VALUES
                        (gen_random_uuid(), :ws, :name, :fw, 'mcp', 'remote-mcp', true, :now, :now)
                    ON CONFLICT (workspace_id, framework, source)
                    DO UPDATE SET under_guard = true, last_seen_at = :now
                """), {"ws": ws_uuid, "name": ai_tool, "fw": ai_tool, "now": _now})
                db.commit()
            except Exception:
                pass  # never block a tool call over a telemetry write


            # #1219 Phase 3b — dispatch is extracted into mcp_impls.py so the
            # /mcp adapter (Phase 3b Chunk B2) can call the same code path.
            # Byte-parity across the two endpoints is guaranteed by construction.
            from app.modules.guard.mcp_impls import GuardCtx, dispatch_guard_tool
            _gctx = GuardCtx(
                db=db, ws_uuid=ws_uuid, workspace_id=workspace_id,
                resolved_token=resolved_token, clerk_user_id=clerk_user_id,
                user_email=user_email, ai_tool=ai_tool, session_id=session_id,
            )
            return JSONResponse(_text(msg_id, dispatch_guard_tool(tool_name, arguments, _gctx)))

        elif method == "ping":
            return JSONResponse(_ok(msg_id, {}))

        else:
            if msg_id is not None:
                return JSONResponse(_err(msg_id, -32601, f"Method not found: {method}"))
            return JSONResponse(status_code=204, content=None)

    finally:
        db.close()


# ---------------------------------------------------------------------------
# OAuth support endpoints
# ---------------------------------------------------------------------------

well_known_router = APIRouter(tags=["guard-mcp"])


@well_known_router.get("/.well-known/oauth-protected-resource/guard/mcp")
async def oauth_protected_resource(workspace_id: str | None = Query(None)):
    """RFC 9728 — tells OAuth clients where the authorization server lives.

    workspace_id is echoed into the resource URI so it flows into the RFC 8707
    resource parameter that MCP clients (Claude.ai) send to the authorize endpoint.
    """
    resource = "https://api.conductai.ai/guard/mcp"
    if workspace_id:
        resource += f"?workspace_id={workspace_id}"
    return JSONResponse({
        "resource": resource,
        "authorization_servers": [os.environ.get("APP_URL", "https://app.conductai.ai")],
        "bearer_methods_supported": ["header"],
        "scopes_supported": ["guard:read"],
    })


@router.post("/oauth/member-token")
async def oauth_member_token(request: Request):
    """Exchange a Clerk JWT for a long-lived cond_api_* token (Claude.ai OAuth flow).

    Mints or rotates a cond_api_* identity token with created_by_clerk_user_id
    so every MCP call is attributed to the correct user email via GMC join.
    """
    import secrets as _secrets
    from app.core.auth import _verify_clerk_token
    from app.core.crypto import encrypt as _encrypt
    from app.modules.agent_identity.models import AgentIdentity

    _API_PFX = "cond_api_"
    _API_PFX_LEN = len(_API_PFX) + 4

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"error": "missing Clerk token"})

    clerk_token = auth_header[7:].strip()
    claims = _verify_clerk_token(clerk_token)
    if not claims:
        return JSONResponse(status_code=401, content={"error": "invalid Clerk token"})

    body = await request.json()
    workspace_id = body.get("workspace_id", "")
    email = body.get("email", "")

    if not email:
        return JSONResponse(status_code=422, content={"error": "email required"})

    clerk_user_id = claims.get("sub", email)

    db = SessionLocal()
    try:
        if not workspace_id:
            row = db.execute(
                _sql("""
                    SELECT gmc.workspace_id FROM guard_member_config gmc
                    JOIN guard_config gc ON gc.workspace_id = gmc.workspace_id
                    WHERE gmc.clerk_user_id = :uid AND gmc.active = true
                    ORDER BY gmc.joined_at DESC LIMIT 1
                """),
                {"uid": clerk_user_id},
            ).fetchone()
            if not row:
                return JSONResponse(status_code=404, content={"error": "no guard workspace found for this user — join via invite first"})
            workspace_id = str(row.workspace_id)

        try:
            ws_uuid = uuid.UUID(workspace_id)
        except ValueError:
            return JSONResponse(status_code=422, content={"error": "invalid workspace_id"})

        plaintext = _API_PFX + _secrets.token_urlsafe(32)
        prefix = plaintext[:_API_PFX_LEN]
        now = datetime.now(timezone.utc)

        # Reuse existing OAuth identity for this user if present, rotate token
        existing = db.execute(
            _sql("""
                SELECT ai.id FROM agent_identities ai
                JOIN guard_member_config gmc ON gmc.agent_identity_id = ai.id
                WHERE gmc.workspace_id = :ws AND gmc.clerk_user_id = :uid
                  AND ai.token_type = 'api'
                LIMIT 1
            """),
            {"ws": str(ws_uuid), "uid": clerk_user_id},
        ).fetchone()

        if existing:
            identity = db.query(AgentIdentity).filter(AgentIdentity.id == existing.id).first()
            identity.token_prefix = prefix
            identity.token_encrypted = _encrypt({"token": plaintext})
            identity.last_used_at = now
        else:
            identity = AgentIdentity(
                id=str(uuid.uuid4()),
                workspace_id=ws_uuid,
                name=f"Claude.ai ({email})",
                provider="conduct",
                token_prefix=prefix,
                token_encrypted=_encrypt({"token": plaintext}),
                token_type="api",
                token_name="claude-ai-oauth",
                created_by_clerk_user_id=clerk_user_id,
                created_at=now,
                last_used_at=now,
                expires_at=None,  # long-lived — no expiry
            )
            db.add(identity)
            # ponytail: no UPDATE guard_member_config here — GMC.agent_identity_id
            # must stay pointing at the CLI cond_agt_* token. OAuth tokens resolve
            # via created_by_clerk_user_id fallback in resolve_agent_token instead.

        db.commit()
        return JSONResponse({"member_token": plaintext})
    finally:
        db.close()
