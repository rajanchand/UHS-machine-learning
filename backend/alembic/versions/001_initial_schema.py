"""Initial schema — flows, predictions, alerts, ml_models with TimescaleDB hypertables.

Revision ID: 001_initial_schema
Revises: None
Create Date: 2024-01-01 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Enable TimescaleDB extension
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")

    # Create enum types
    op.execute("CREATE TYPE alertstatus AS ENUM ('open', 'acknowledged', 'resolved')")
    op.execute("CREATE TYPE alertseverity AS ENUM ('low', 'medium', 'high', 'critical')")

    # --- flows table ---
    op.create_table(
        "flows",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        # Network identifiers
        sa.Column("src_ip", sa.String(45), nullable=False),
        sa.Column("src_port", sa.Integer, nullable=False),
        sa.Column("dst_ip", sa.String(45), nullable=False),
        sa.Column("dst_port", sa.Integer, nullable=False),
        sa.Column("protocol", sa.Integer, nullable=False),
        # Flow features - NSL-KDD 41 standard + 7 derived features
        sa.Column("duration", sa.Float, nullable=False, server_default="0"),
        sa.Column("protocol_type", sa.Float, nullable=False, server_default="0"),
        sa.Column("service", sa.Float, nullable=False, server_default="0"),
        sa.Column("flag", sa.Float, nullable=False, server_default="0"),
        sa.Column("src_bytes", sa.Float, nullable=False, server_default="0"),
        sa.Column("dst_bytes", sa.Float, nullable=False, server_default="0"),
        sa.Column("land", sa.Float, nullable=False, server_default="0"),
        sa.Column("wrong_fragment", sa.Float, nullable=False, server_default="0"),
        sa.Column("urgent", sa.Float, nullable=False, server_default="0"),
        sa.Column("hot", sa.Float, nullable=False, server_default="0"),
        sa.Column("num_failed_logins", sa.Float, nullable=False, server_default="0"),
        sa.Column("logged_in", sa.Float, nullable=False, server_default="0"),
        sa.Column("num_compromised", sa.Float, nullable=False, server_default="0"),
        sa.Column("root_shell", sa.Float, nullable=False, server_default="0"),
        sa.Column("su_attempted", sa.Float, nullable=False, server_default="0"),
        sa.Column("num_root", sa.Float, nullable=False, server_default="0"),
        sa.Column("num_file_creations", sa.Float, nullable=False, server_default="0"),
        sa.Column("num_shells", sa.Float, nullable=False, server_default="0"),
        sa.Column("num_access_files", sa.Float, nullable=False, server_default="0"),
        sa.Column("num_outbound_cmds", sa.Float, nullable=False, server_default="0"),
        sa.Column("is_host_login", sa.Float, nullable=False, server_default="0"),
        sa.Column("is_guest_login", sa.Float, nullable=False, server_default="0"),
        sa.Column("count", sa.Float, nullable=False, server_default="0"),
        sa.Column("srv_count", sa.Float, nullable=False, server_default="0"),
        sa.Column("serror_rate", sa.Float, nullable=False, server_default="0"),
        sa.Column("srv_serror_rate", sa.Float, nullable=False, server_default="0"),
        sa.Column("rerror_rate", sa.Float, nullable=False, server_default="0"),
        sa.Column("srv_rerror_rate", sa.Float, nullable=False, server_default="0"),
        sa.Column("same_srv_rate", sa.Float, nullable=False, server_default="0"),
        sa.Column("diff_srv_rate", sa.Float, nullable=False, server_default="0"),
        sa.Column("srv_diff_host_rate", sa.Float, nullable=False, server_default="0"),
        sa.Column("dst_host_count", sa.Float, nullable=False, server_default="0"),
        sa.Column("dst_host_srv_count", sa.Float, nullable=False, server_default="0"),
        sa.Column("dst_host_same_srv_rate", sa.Float, nullable=False, server_default="0"),
        sa.Column("dst_host_diff_srv_rate", sa.Float, nullable=False, server_default="0"),
        sa.Column("dst_host_same_src_port_rate", sa.Float, nullable=False, server_default="0"),
        sa.Column("dst_host_srv_diff_host_rate", sa.Float, nullable=False, server_default="0"),
        sa.Column("dst_host_serror_rate", sa.Float, nullable=False, server_default="0"),
        sa.Column("dst_host_srv_serror_rate", sa.Float, nullable=False, server_default="0"),
        sa.Column("dst_host_rerror_rate", sa.Float, nullable=False, server_default="0"),
        sa.Column("dst_host_srv_rerror_rate", sa.Float, nullable=False, server_default="0"),
        sa.Column("packet_rate", sa.Float, nullable=False, server_default="0"),
        sa.Column("byte_rate", sa.Float, nullable=False, server_default="0"),
        sa.Column("avg_packet_size", sa.Float, nullable=False, server_default="0"),
        sa.Column("flow_duration", sa.Float, nullable=False, server_default="0"),
        sa.Column("inter_arrival_time", sa.Float, nullable=False, server_default="0"),
        sa.Column("fwd_bwd_ratio", sa.Float, nullable=False, server_default="0"),
        sa.Column("port_entropy", sa.Float, nullable=False, server_default="0"),
        # Label
        sa.Column("label", sa.String(64), nullable=True),
    )
    op.create_index("ix_flows_ts", "flows", ["ts"])

    # Convert flows to TimescaleDB hypertable
    op.execute("SELECT create_hypertable('flows', 'ts', migrate_data => true)")

    # --- predictions table ---
    op.create_table(
        "predictions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "flow_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("flows.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("model_name", sa.String(128), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("is_anomaly", sa.Boolean, nullable=False),
        sa.Column("threshold", sa.Float, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_predictions_created_at", "predictions", ["created_at"])
    op.create_index("ix_predictions_flow_id", "predictions", ["flow_id"])

    # Convert predictions to TimescaleDB hypertable
    op.execute("SELECT create_hypertable('predictions', 'created_at', migrate_data => true)")

    # --- alerts table ---
    op.create_table(
        "alerts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "flow_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("flows.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "severity",
            postgresql.ENUM("low", "medium", "high", "critical", name="alertseverity", create_type=False),
            nullable=False,
            server_default="medium",
        ),
        sa.Column("suspected_attack_type", sa.String(64), nullable=True),
        sa.Column(
            "status",
            postgresql.ENUM("open", "acknowledged", "resolved", name="alertstatus", create_type=False),
            nullable=False,
            server_default="open",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_alerts_status", "alerts", ["status"])
    op.create_index("ix_alerts_created_at", "alerts", ["created_at"])

    # --- ml_models table ---
    op.create_table(
        "ml_models",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False, unique=True),
        sa.Column("version", sa.String(64), nullable=False),
        sa.Column(
            "trained_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("metrics_json", postgresql.JSONB, nullable=True),
        sa.Column("artifact_path", sa.Text, nullable=False),
        sa.Column("threshold", sa.Float, nullable=False, server_default="0.5"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("description", sa.Text, nullable=True),
    )


def downgrade() -> None:
    op.drop_table("ml_models")
    op.drop_table("alerts")
    op.drop_table("predictions")
    op.drop_table("flows")
    op.execute("DROP TYPE IF EXISTS alertstatus")
    op.execute("DROP TYPE IF EXISTS alertseverity")
    op.execute("DROP EXTENSION IF EXISTS timescaledb CASCADE")
