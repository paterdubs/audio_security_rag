"""Lược đồ khởi tạo — SYSTEM.md §4.3

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-16
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBED_DIM = 1024   # khớp models.EMBED_DIM và .env EMBED_DIM


def upgrade() -> None:
    # Phải bật extension TRƯỚC khi tạo cột Vector, nếu không CREATE TABLE lỗi
    # "type vector does not exist".
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "locations",
        sa.Column("location_id", sa.Text(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("area_type", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
    )

    op.create_table(
        "security_events",
        sa.Column("event_id", sa.Text(), primary_key=True),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("location_id", sa.Text(), sa.ForeignKey("locations.location_id"), nullable=False),
        sa.Column("caption_en", sa.Text(), nullable=False),
        sa.Column("caption_vi", sa.Text(), nullable=False),
        sa.Column("severity", sa.Text(), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("audio_path", sa.Text(), nullable=True),
        sa.Column("embedding", Vector(EMBED_DIM), nullable=True),
        sa.Column("grounding_score", sa.Float(), nullable=True),
        sa.Column("model_versions", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "event_detections",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "event_id",
            sa.Text(),
            sa.ForeignKey("security_events.event_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("class_id", sa.Text(), nullable=False),
        sa.Column("onset_sec", sa.Float(), nullable=False),
        sa.Column("offset_sec", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
    )
    op.create_index("ix_event_detections_event_id", "event_detections", ["event_id"])

    op.create_table(
        "event_feedback",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("event_id", sa.Text(), sa.ForeignKey("security_events.event_id"), nullable=False),
        sa.Column("is_true_alarm", sa.Boolean(), nullable=True),
        sa.Column("correct_classes", sa.ARRAY(sa.Text()), nullable=True),
        sa.Column("caption_ok", sa.Boolean(), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("reviewed_by", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "inference_metrics",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("ts", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("model_version", sa.String(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("stage", sa.String(), nullable=True),
        sa.Column("mean_confidence", sa.Float(), nullable=True),
        sa.Column("class_histogram", postgresql.JSONB(), nullable=True),
    )

    # HNSW + cosine: §7.2 truy xuất bằng cosine distance (`<=>`). Index phải dùng đúng
    # opclass của toán tử sẽ query, nếu không Postgres bỏ qua index và quét toàn bảng.
    op.execute(
        "CREATE INDEX ix_security_events_embedding_hnsw "
        "ON security_events USING hnsw (embedding vector_cosine_ops)"
    )
    # Ba index dưới phục vụ bộ lọc metadata của RAG (khoảng thời gian, địa điểm, mức rủi ro)
    # — chính là lợi thế "lọc + vector trong MỘT câu SQL" của pgvector (§4.2).
    op.execute("CREATE INDEX ix_security_events_window_start ON security_events (window_start DESC)")
    op.execute(
        "CREATE INDEX ix_security_events_location_window "
        "ON security_events (location_id, window_start DESC)"
    )
    op.execute(
        "CREATE INDEX ix_security_events_severity_window "
        "ON security_events (severity, window_start DESC)"
    )


def downgrade() -> None:
    op.drop_table("inference_metrics")
    op.drop_table("event_feedback")
    op.drop_index("ix_event_detections_event_id", table_name="event_detections")
    op.drop_table("event_detections")
    op.drop_table("security_events")
    op.drop_table("locations")
    # CỐ Ý không DROP EXTENSION vector: extension có thể đang được schema khác dùng.
