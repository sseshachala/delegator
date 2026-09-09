from app.models.organization import Organization
from app.models.workspace import Workspace
from app.models.workspace_llm_primitives import WorkspaceLLMPrimitives
from app.models.rbac import Role, Permission  # noqa
from app.models.user import User
from app.models.workspace_user import WorkspaceUser
from app.models.project import Project
from app.models.environment import Environment  # noqa — must be before Workflow to satisfy Workspace.environments relationship
from app.models.workflow import Workflow, WorkflowVersion
from app.models.integration import Integration
from app.models.run import Run, RunEvent
from app.models.run_analytics_event import RunAnalyticsEvent
from app.models.run_trace import RunTrace
from app.models.audit_log import AuditLog
from app.models.email_template import EmailTemplate
from app.models.playbook_submission import PlaybookSubmission

# Models registered here so their tables appear in Base.metadata for
# alembic check. Without these imports, alembic sees the tables as
# "orphan" and proposes to drop them — see #1284 for the audit.
from app.models.agent_memory import AgentMemory  # noqa
from app.models.mcp_server import McpServer  # noqa
from app.models.run_block_state import RunBlockState  # noqa
from app.models.run_online_score import RunOnlineScore  # noqa
from app.models.security_finding import SecurityFinding  # noqa
from app.models.team_session_memory import TeamSessionMemory  # noqa
from app.models.watchdog_event import WatchdogEvent  # noqa
from app.models.workspace_config import WorkspaceConfig  # noqa
from app.models.workspace_instructions import WorkspaceInstructions  # noqa
from app.models.project_template import ProjectTemplate  # noqa
from app.models.workspace_invite import WorkspaceInvite  # noqa
from app.models.workspace_report_layout import WorkspaceReportLayout  # noqa
from app.models.security_config import SecurityConfig  # noqa
from app.models.security_policy import SecurityPolicy  # noqa
from app.models.model_routing_policy import ModelRoutingPolicy  # noqa
from app.models.cred_retrieval_token import CredRetrievalToken  # noqa
from app.models.oauth import OauthClient, OauthAuthCode  # noqa — OAuth 2.1 tables (migration 0118)

from app.modules.guard.models import (
    GuardConfig,
    GuardMemberConfig,
    GuardSession,
    GuardAuditEvent,
    GuardSpendBudget,
)
from app.modules.behavior.models import ArgBaseline  # noqa
from app.modules.glens.models import GlensChatSession  # noqa
from app.modules.telemetry.models import TelemetryEvent  # noqa
from app.modules.agent_identity.models import AgentIdentity
from app.modules.agent_identity.run_token_model import AgentRunToken  # noqa

__all__ = [
    "Organization", "Workspace", "User", "WorkspaceUser", "Project",
    "Environment",
    "Workflow", "WorkflowVersion", "Integration", "Run", "RunEvent",
    "RunAnalyticsEvent", "RunTrace", "AuditLog", "EmailTemplate",
    "PlaybookSubmission",
    "AgentMemory", "McpServer", "RunBlockState", "RunOnlineScore",
    "SecurityFinding", "TeamSessionMemory", "WatchdogEvent",
    "WorkspaceConfig", "WorkspaceInstructions",
    "ProjectTemplate", "WorkspaceInvite", "SecurityConfig",
    "SecurityPolicy", "ModelRoutingPolicy", "CredRetrievalToken",
    "GuardConfig", "GuardMemberConfig", "GuardSession",
    "GuardAuditEvent", "GuardSpendBudget",
    "GlensChatSession", "TelemetryEvent",
    "Role", "Permission",
    "AgentIdentity",
    "AgentRunToken",
    "WorkspaceLLMPrimitives",
]
