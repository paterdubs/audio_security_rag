"""CRUD sự kiện + phát lại audio — SYSTEM.md §4.4."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db import get_session
from app.envelope import ok
from app.models import EventDetection, Location, SecurityEvent
from app.schemas import DetectionOut, EventDetail, EventSummary, LocationOut

router = APIRouter(prefix="/events", tags=["events"])

MAX_PAGE_SIZE = 200


def _location_out(location: Location | None) -> LocationOut | None:
    if location is None:
        return None
    return LocationOut(id=location.location_id, name=location.name, area_type=location.area_type)


@router.get("")
async def list_events(
    session: AsyncSession = Depends(get_session),
    from_: datetime | None = Query(default=None, alias="from"),
    to: datetime | None = None,
    location: str | None = None,
    severity: str | None = None,
    class_id: str | None = Query(default=None, alias="class"),
    limit: int = Query(default=50, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """Danh sách sự kiện, mới nhất trước.

    `limit` bị chặn trần: không có trần thì một request `?limit=999999` kéo cả bảng vào
    bộ nhớ và làm chết service — dashboard chỉ hiển thị được vài chục dòng một lúc.
    """
    stmt = select(SecurityEvent).options(selectinload(SecurityEvent.detections))
    count_stmt = select(func.count()).select_from(SecurityEvent)

    filters = []
    if from_ is not None:
        filters.append(SecurityEvent.window_start >= from_)
    if to is not None:
        filters.append(SecurityEvent.window_start <= to)
    if location is not None:
        filters.append(SecurityEvent.location_id == location)
    if severity is not None:
        filters.append(SecurityEvent.severity == severity)
    if class_id is not None:
        # Lọc theo lớp phải đi qua bảng detection, không có cột nào trên security_events
        # chứa danh sách lớp.
        filters.append(
            SecurityEvent.event_id.in_(
                select(EventDetection.event_id).where(EventDetection.class_id == class_id)
            )
        )
    for condition in filters:
        stmt = stmt.where(condition)
        count_stmt = count_stmt.where(condition)

    total = await session.scalar(count_stmt) or 0
    rows = (
        await session.scalars(
            stmt.order_by(SecurityEvent.window_start.desc()).limit(limit).offset(offset)
        )
    ).all()

    locations = {
        loc.location_id: loc
        for loc in (await session.scalars(select(Location))).all()
    }
    data = [
        EventSummary(
            event_id=event.event_id,
            window_start=event.window_start,
            window_end=event.window_end,
            location=_location_out(locations.get(event.location_id)),
            caption_vi=event.caption_vi,
            caption_en=event.caption_en,
            severity=event.severity,
            risk_score=event.risk_score,
            n_detections=len(event.detections),
        )
        for event in rows
    ]
    return ok(data, meta={"total": total, "limit": limit, "offset": offset})


@router.get("/{event_id}")
async def get_event(event_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    event = await session.get(
        SecurityEvent, event_id, options=[selectinload(SecurityEvent.detections)]
    )
    if event is None:
        raise HTTPException(status_code=404, detail=f"Không có sự kiện {event_id}")

    location = await session.get(Location, event.location_id)
    detail = EventDetail(
        event_id=event.event_id,
        window_start=event.window_start,
        window_end=event.window_end,
        location=_location_out(location),
        caption_vi=event.caption_vi,
        caption_en=event.caption_en,
        severity=event.severity,
        risk_score=event.risk_score,
        n_detections=len(event.detections),
        detections=[
            DetectionOut(
                class_id=d.class_id, onset=d.onset_sec, offset=d.offset_sec, confidence=d.confidence
            )
            for d in sorted(event.detections, key=lambda d: d.onset_sec)
        ],
        grounding_score=event.grounding_score,
        model_versions=event.model_versions,
        has_audio=bool(event.audio_path),
    )
    return ok(detail)


@router.get("/{event_id}/audio")
async def get_event_audio(event_id: str, session: AsyncSession = Depends(get_session)) -> FileResponse:
    event = await session.get(SecurityEvent, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"Không có sự kiện {event_id}")
    # audio_path NULL là trạng thái HỢP LỆ: retention đã xoá, hoặc cấu hình không lưu
    # audio (AUDIO_RETENTION_DAYS=0). Phải phân biệt với "sự kiện không tồn tại".
    if not event.audio_path:
        raise HTTPException(
            status_code=410,
            detail="Audio đã hết thời hạn lưu trữ hoặc hệ thống được cấu hình không lưu audio",
        )
    path = Path(event.audio_path)
    if not path.exists():
        raise HTTPException(status_code=410, detail="File audio không còn trên đĩa")
    return FileResponse(path, media_type="audio/wav", filename=f"{event_id}.wav")
