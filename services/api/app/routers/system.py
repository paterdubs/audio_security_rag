"""Feedback, trạng thái model, số liệu dashboard — SYSTEM.md §4.4, §9.3, §9.4."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.envelope import ok
from app.models import EventDetection, EventFeedback, SecurityEvent
from app.schemas import FeedbackIn

router = APIRouter(tags=["system"])


@router.post("/events/{event_id}/feedback")
async def submit_feedback(
    event_id: str, payload: FeedbackIn, session: AsyncSession = Depends(get_session)
) -> dict:
    """Ghi phản hồi người vận hành — nhiên liệu cho vòng lặp retraining (§9.4)."""
    if await session.get(SecurityEvent, event_id) is None:
        raise HTTPException(status_code=404, detail=f"Không có sự kiện {event_id}")

    feedback = EventFeedback(
        event_id=event_id,
        is_true_alarm=payload.is_true_alarm,
        correct_classes=payload.correct_classes,
        caption_ok=payload.caption_ok,
        note=payload.note,
        reviewed_by=payload.reviewed_by,
    )
    session.add(feedback)
    await session.commit()
    return ok({"event_id": event_id, "feedback_id": feedback.id})


@router.get("/models/status")
async def models_status(session: AsyncSession = Depends(get_session)) -> dict:
    """Phiên bản model đang phục vụ.

    ⚠️ Bản W1 dùng PANNs + caption template làm TẠM (đúng kế hoạch walking skeleton).
    Model đề xuất thật (BEATs + Conformer + BART, grounded decoding) là W4–W5. Trường
    `is_placeholder` nói thẳng điều đó để không ai đọc nhầm số đo của bản tạm thành số
    đo của đóng góp nghiên cứu.
    """
    latest = await session.scalar(
        select(SecurityEvent.model_versions).order_by(SecurityEvent.created_at.desc()).limit(1)
    )
    return ok(
        {
            "active": latest or {},
            "is_placeholder": True,
            "note": "PANNs CNN14 + caption template (walking skeleton W1); Grounded AAC ở W4-W5",
            "rag_answer_provider": get_settings().rag_answer_provider,
        }
    )


@router.get("/metrics/summary")
async def metrics_summary(session: AsyncSession = Depends(get_session)) -> dict:
    """Số liệu cho dashboard. Đếm thật trong DB, không cache — quy mô một node."""
    total_events = await session.scalar(select(func.count()).select_from(SecurityEvent)) or 0

    by_severity = {
        row.severity: row.n
        for row in await session.execute(
            select(SecurityEvent.severity, func.count().label("n")).group_by(SecurityEvent.severity)
        )
    }
    by_class = {
        row.class_id: row.n
        for row in await session.execute(
            select(EventDetection.class_id, func.count().label("n"))
            .group_by(EventDetection.class_id)
            .order_by(func.count().desc())
        )
    }
    feedback_total = await session.scalar(select(func.count()).select_from(EventFeedback)) or 0
    true_alarms = await session.scalar(
        select(func.count()).select_from(EventFeedback).where(EventFeedback.is_true_alarm.is_(True))
    ) or 0

    return ok(
        {
            "total_events": total_events,
            "by_severity": by_severity,
            "by_class": by_class,
            "feedback": {
                "total": feedback_total,
                "true_alarms": true_alarms,
                # FAR chỉ tính được khi đã có phản hồi; trả None thay vì 0 để không ai
                # đọc nhầm "chưa có dữ liệu" thành "tỉ lệ báo động giả bằng 0".
                "false_alarm_rate": (
                    round(1 - true_alarms / feedback_total, 4) if feedback_total else None
                ),
            },
        }
    )
