"""Ingest audio → sự kiện an ninh — đường đi đầu-cuối của walking skeleton.

    upload → inference (PANNs + caption) → risk scoring → DB → embedding → alert

Đây là đường OFFLINE. Đường streaming thời gian thực (Redis Streams + Temporal
Aggregator) là việc của W7 — SYSTEM.md §6.1–6.2.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.envelope import ok
from app.event_ids import next_event_id
from app.inference_client import InferenceClient
from app.models import EventDetection, Location, SecurityEvent
from app.ontology import class_tiers
from app.rag.document import build_document
from app.risk import Detection, score_event
from app.routers.alerts import broadcast_event
from app.schemas import DetectionOut, UploadResponse

router = APIRouter(prefix="/audio", tags=["ingest"])

MAX_UPLOAD_BYTES = 100 * 1024 * 1024   # 100 MB — quá mức này gần như chắc chắn là nhầm file


async def _allocate_event_id(session: AsyncSession, moment: datetime) -> str:
    """Xin id kế tiếp trong ngày. Đọc id lớn nhất của ĐÚNG ngày đó rồi +1."""
    prefix = f"EVT_{moment:%Y%m%d}_"
    latest = await session.scalar(
        select(SecurityEvent.event_id)
        .where(SecurityEvent.event_id.like(f"{prefix}%"))
        .order_by(SecurityEvent.event_id.desc())
        .limit(1)
    )
    return next_event_id(moment, latest)


@router.post("/upload")
async def upload_audio(
    file: UploadFile = File(...),
    location_id: str = Form(default="HALL_03"),
    session: AsyncSession = Depends(get_session),
) -> dict:
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="File rỗng")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File vượt {MAX_UPLOAD_BYTES // 1024 // 1024} MB")

    # Location phải có thật: FK sẽ chặn, nhưng báo lỗi 400 rõ ràng tốt hơn là để lỗi
    # ràng buộc khoá ngoại nổ ra dưới tầng DB.
    location = await session.get(Location, location_id)
    if location is None:
        known = list((await session.scalars(select(Location.location_id))).all())
        raise HTTPException(status_code=400, detail=f"location_id {location_id!r} không có. Đang có: {known}")

    settings = get_settings()
    result = await InferenceClient().infer(audio_bytes, file.filename or "upload.wav")

    risk = score_event(
        [
            Detection(d["class_id"], float(d["onset"]), float(d["offset"]), float(d["confidence"]))
            for d in result.detections
        ],
        class_tiers(),
    )

    now = datetime.now(timezone.utc)
    event_id = await _allocate_event_id(session, now)

    audio_path = None
    if settings.audio_retention_days > 0:
        settings.audio_dir.mkdir(parents=True, exist_ok=True)
        destination = settings.audio_dir / f"{event_id}.wav"
        destination.write_bytes(audio_bytes)
        audio_path = str(destination)

    event = SecurityEvent(
        event_id=event_id,
        window_start=now,
        window_end=now + timedelta(seconds=result.duration),
        location_id=location_id,
        caption_en=result.caption_en,
        caption_vi=result.caption_vi,
        severity=risk.severity,
        risk_score=risk.risk_score,
        audio_path=audio_path,
        model_versions=result.model_versions,
    )
    session.add(event)
    for d in result.detections:
        session.add(
            EventDetection(
                event_id=event_id,
                class_id=d["class_id"],
                onset_sec=float(d["onset"]),
                offset_sec=float(d["offset"]),
                confidence=float(d["confidence"]),
            )
        )

    # Embedding: nhúng tiếng Việt (KHÔNG phải caption_en) vì người dùng hỏi tiếng Việt — §7.4.
    #
    # Nhúng VĂN BẢN GIÀU THÔNG TIN chứ không phải mỗi caption: caption trần là câu template
    # nên mọi sự kiện giống nhau 0.83–0.92 và RAG trả về cùng một sự kiện cho mọi câu hỏi.
    # Xem app/rag/document.py để có số đo đầy đủ.
    #
    # Lỗi embedding không được làm mất sự kiện: sự kiện vẫn phải vào DB và vẫn cảnh báo
    # được, chỉ là tạm thời chưa truy xuất được bằng RAG.
    document = build_document(
        caption_vi=result.caption_vi,
        class_ids=[d["class_id"] for d in result.detections],
        window_start=now,
        location_name=location.name,
        area_type=location.area_type,
        severity=risk.severity,
    )
    try:
        vectors = await InferenceClient().embed([document])
        event.embedding = vectors[0]
    except Exception:  # noqa: BLE001
        event.embedding = None

    await session.commit()

    payload = UploadResponse(
        event_id=event_id,
        severity=risk.severity,
        risk_score=risk.risk_score,
        caption_vi=result.caption_vi,
        detections=[
            DetectionOut(
                class_id=d["class_id"],
                onset=float(d["onset"]),
                offset=float(d["offset"]),
                confidence=float(d["confidence"]),
            )
            for d in result.detections
        ],
    )
    # Đường NÓNG: đẩy thẳng ra WebSocket, KHÔNG đi qua LLM (§4.1).
    await broadcast_event(
        {
            "event_id": event_id,
            "severity": risk.severity,
            "risk_score": risk.risk_score,
            "caption_vi": result.caption_vi,
            "location_id": location_id,
            "window_start": now.isoformat(),
        }
    )
    return ok(payload, meta={"guards_applied": risk.guards_applied, "risk_components": risk.components})
