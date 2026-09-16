"""ORM — khớp đúng lược đồ trong SYSTEM.md §4.3.

Sửa bảng ở đây thì PHẢI sinh migration Alembic tương ứng trong cùng commit: lược đồ thật
của database là thứ Alembic nói, không phải thứ file này nói.
"""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBED_DIM = 1024   # khớp .env EMBED_DIM và migration; đổi ⇒ migration mới


class Base(DeclarativeBase):
    pass


class Location(Base):
    __tablename__ = "locations"

    location_id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    area_type: Mapped[str] = mapped_column(Text, nullable=False)   # school|parking|residential|factory
    description: Mapped[str | None] = mapped_column(Text)


class SecurityEvent(Base):
    """Đơn vị dữ liệu trung tâm của toàn hệ thống."""

    __tablename__ = "security_events"

    event_id: Mapped[str] = mapped_column(Text, primary_key=True)       # EVT_YYYYMMDD_NNNNNN
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    location_id: Mapped[str] = mapped_column(Text, ForeignKey("locations.location_id"), nullable=False)
    caption_en: Mapped[str] = mapped_column(Text, nullable=False)
    caption_vi: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(Text, nullable=False)          # LOW|MEDIUM|HIGH|CRITICAL
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    # NULL khi retention đã xoá audio, hoặc khi AUDIO_RETENTION_DAYS=0 (không lưu audio).
    audio_path: Mapped[str | None] = mapped_column(Text)
    # Embed caption_vi, KHÔNG phải caption_en — người dùng hỏi bằng tiếng Việt (§7.4).
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM))
    grounding_score: Mapped[float | None] = mapped_column(Float)
    model_versions: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    detections: Mapped[list["EventDetection"]] = relationship(
        back_populates="event", cascade="all, delete-orphan", lazy="selectin"
    )


class EventDetection(Base):
    __tablename__ = "event_detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(
        Text, ForeignKey("security_events.event_id", ondelete="CASCADE"), nullable=False
    )
    class_id: Mapped[str] = mapped_column(Text, nullable=False)     # 1 trong 16 class
    onset_sec: Mapped[float] = mapped_column(Float, nullable=False)  # tương đối so với window_start
    offset_sec: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    event: Mapped[SecurityEvent] = relationship(back_populates="detections")


class EventFeedback(Base):
    """Phản hồi người vận hành → nhiên liệu retraining (§9.4).

    `caption_ok` là trường quý nhất: nó cho nhãn hallucination TRONG THỰC ĐỊA, thứ không
    dataset công khai nào có.
    """

    __tablename__ = "event_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(Text, ForeignKey("security_events.event_id"), nullable=False)
    is_true_alarm: Mapped[bool | None] = mapped_column(Boolean)
    correct_classes: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    caption_ok: Mapped[bool | None] = mapped_column(Boolean)
    note: Mapped[str | None] = mapped_column(Text)
    reviewed_by: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class InferenceMetric(Base):
    __tablename__ = "inference_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    model_version: Mapped[str | None] = mapped_column(String)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    stage: Mapped[str | None] = mapped_column(String)   # sed|caption|aggregate|e2e
    mean_confidence: Mapped[float | None] = mapped_column(Float)
    class_histogram: Mapped[dict | None] = mapped_column(JSONB)
