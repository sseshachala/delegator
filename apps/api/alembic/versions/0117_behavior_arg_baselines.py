"""Argument anomaly baselines + opt-in flag (record-only v1, #1594).

Two changes, one story — the record-only argument anomaly check:

1. New ``behavior_arg_baselines``: per-(workspace, agent_key, tool_name,
   arg_path) rolling aggregates. Welford count/mean/m2 for numeric args,
   salted hashed value counts for categorical args (raw values never
   stored). Read and written only by
   ``app.modules.behavior.arg_anomaly.observe``. Row growth is bounded per
   workspace with least-recently-updated eviction, hence the
   (workspace_id, updated_at) index.

2. ``guard_config.arg_anomaly_enabled`` (default FALSE) — the opt-in gate.
   The flag lives on guard_config because it gates behavior on the
   guard_check surface; the observations themselves are a behavior-module
   concern.

Merging changes nothing for existing workspaces until someone flips the flag.

Revision ID: 0117
Revises: 0116
Create Date: 2026-09-08
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0117"
down_revision = "0116"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "behavior_arg_baselines",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "workspace_id",
            UUID(as_uuid=True),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("agent_key", sa.Text, nullable=False),
        sa.Column("tool_name", sa.Text, nullable=False),
        sa.Column("arg_path", sa.Text, nullable=False),
        sa.Column("kind", sa.String(10), nullable=False),
        sa.Column("count", sa.BigInteger, nullable=False, server_default=sa.text("0")),
        sa.Column("mean", sa.Float, nullable=False, server_default=sa.text("0")),
        sa.Column("m2", sa.Float, nullable=False, server_default=sa.text("0")),
        sa.Column("values", JSONB, nullable=True),
        sa.Column(
            "overflow_count", sa.BigInteger, nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "workspace_id",
            "agent_key",
            "tool_name",
            "arg_path",
            name="uq_behavior_arg_baselines_key",
        ),
    )
    op.create_index(
        "idx_behavior_arg_baselines_ws",
        "behavior_arg_baselines",
        ["workspace_id"],
    )
    op.create_index(
        "idx_behavior_arg_baselines_ws_updated",
        "behavior_arg_baselines",
        ["workspace_id", "updated_at"],
    )
    op.add_column(
        "guard_config",
        sa.Column(
            "arg_anomaly_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    # Nullable on purpose: NULL means "use the module default", so the default
    # value itself lives only in the code and cannot drift from the schema.
    op.add_column(
        "guard_config",
        sa.Column("arg_anomaly_zscore_threshold", sa.Float(), nullable=True),
    )
    op.add_column(
        "guard_config",
        sa.Column("arg_anomaly_min_samples", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("guard_config", "arg_anomaly_min_samples")
    op.drop_column("guard_config", "arg_anomaly_zscore_threshold")
    op.drop_column("guard_config", "arg_anomaly_enabled")
    op.drop_index(
        "idx_behavior_arg_baselines_ws_updated", table_name="behavior_arg_baselines"
    )
    op.drop_index("idx_behavior_arg_baselines_ws", table_name="behavior_arg_baselines")
    op.drop_table("behavior_arg_baselines")
