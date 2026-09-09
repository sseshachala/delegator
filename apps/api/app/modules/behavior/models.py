"""SQLAlchemy model for behavior_arg_baselines (#1594).

Behavior signals observe what already happened; Guard decides what may happen.
This table belongs to the former story, so it lives here rather than under
modules/guard.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

from app.core.database import Base


class ArgBaseline(Base):
    """Per-key rolling baseline for tool-call argument values.

    One row per (workspace, agent_key, tool_name, arg_path). agent_key is the
    calling user id on the MCP guard_check path (the only writer today; a
    future surface may pass an agent_identity_id). Numeric args keep Welford
    aggregates (count/mean/m2); categorical args keep hashed value counts,
    raw values are never stored. Read and written only by
    app.modules.behavior.arg_anomaly.observe, behind
    guard_config.arg_anomaly_enabled.

    Row growth is bounded twice: MAX_TRACKED_ARG_PATHS per
    (workspace, agent_key, tool_name), and MAX_ROWS_PER_WORKSPACE overall with
    least-recently-updated eviction on insert.
    """

    __tablename__ = "behavior_arg_baselines"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
    )
    agent_key = Column(Text, nullable=False)
    tool_name = Column(Text, nullable=False)
    arg_path = Column(Text, nullable=False)
    kind = Column(String(10), nullable=False)  # 'numeric' | 'category'
    count = Column(BigInteger, nullable=False, default=0)
    mean = Column(Float, nullable=False, default=0.0)
    m2 = Column(Float, nullable=False, default=0.0)
    # {salted sha256[:16] of normalized value: occurrences} — capped at
    # MAX_DISTINCT_VALUES; overflow_count > 0 marks the key as
    # non-categorical (novelty detection disarmed).
    values = Column(JSONB, nullable=True)
    overflow_count = Column(BigInteger, nullable=False, default=0)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    # Doubles as the LRU key for eviction.
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            "agent_key",
            "tool_name",
            "arg_path",
            name="uq_behavior_arg_baselines_key",
        ),
        Index("idx_behavior_arg_baselines_ws", "workspace_id"),
        # Supports the LRU eviction scan.
        Index("idx_behavior_arg_baselines_ws_updated", "workspace_id", "updated_at"),
    )
