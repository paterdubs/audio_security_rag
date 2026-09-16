"""Pydantic v2 — validate ở MỌI biên hệ thống (SYSTEM.md §4.2)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

SEVERITIES = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


class DetectionOut(BaseModel):
    class_id: str
    onset: float
    offset: float
    confidence: float


class LocationOut(BaseModel):
    id: str
    name: str
    area_type: str


class EventSummary(BaseModel):
    event_id: str
    window_start: datetime
    window_end: datetime
    location: LocationOut | None = None
    caption_vi: str
    caption_en: str
    severity: str
    risk_score: float
    n_detections: int


class EventDetail(EventSummary):
    detections: list[DetectionOut]
    grounding_score: float | None = None
    model_versions: dict[str, str]
    has_audio: bool


class UploadResponse(BaseModel):
    event_id: str
    severity: str
    risk_score: float
    caption_vi: str
    detections: list[DetectionOut]


class RagQuery(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    # Lọc metadata đi CÙNG truy vấn ngữ nghĩa trong một câu SQL — §4.2.
    from_: datetime | None = Field(default=None, alias="from")
    to: datetime | None = None
    location: str | None = None
    severity: str | None = None

    model_config = {"populate_by_name": True}


class Citation(BaseModel):
    """Một citation LUÔN trỏ tới event_id thật. §4.4: câu trả lời không có citation bị
    coi là lỗi hệ thống, không phải câu trả lời hợp lệ."""

    event_id: str
    window_start: datetime
    caption_vi: str
    severity: str
    similarity: float


class RagAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_events: list[str]
    provider: str


class FeedbackIn(BaseModel):
    is_true_alarm: bool | None = None
    correct_classes: list[str] | None = None
    # Trường quý nhất của cả vòng lặp MLOps: nhãn hallucination trong thực địa (§9.4).
    caption_ok: bool | None = None
    note: str | None = Field(default=None, max_length=2000)
    reviewed_by: str | None = Field(default=None, max_length=200)
