"""Truy vấn lịch sử bằng tiếng Việt, có trích dẫn — SYSTEM.md §7.

Đường LỊCH SỬ, tách hẳn khỏi đường NÓNG (alerts). Chậm hơn nhưng phải giải thích được:
mọi câu trả lời kèm citations[] trỏ tới event_id thật.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.envelope import ok
from app.inference_client import InferenceClient
from app.rag.answer import get_provider
from app.rag.retrieval import retrieve
from app.schemas import Citation, RagAnswer, RagQuery

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query")
async def rag_query(payload: RagQuery, session: AsyncSession = Depends(get_session)) -> dict:
    settings = get_settings()

    try:
        vectors = await InferenceClient().embed([payload.question])
    except Exception as error:  # noqa: BLE001
        # Không có embedding thì KHÔNG được rơi về tìm kiếm chuỗi rồi trả lời như thường:
        # câu trả lời sẽ dựa trên bằng chứng kém hơn hẳn mà người dùng không biết.
        raise HTTPException(
            status_code=503,
            detail=f"Service inference chưa sẵn sàng để nhúng câu hỏi: {type(error).__name__}",
        ) from error

    events = await retrieve(
        session,
        vectors[0],
        top_k=settings.rag_top_k,
        from_ts=payload.from_,
        to_ts=payload.to,
        location=payload.location,
        severity=payload.severity,
        min_similarity=settings.rag_min_similarity,
    )

    provider = get_provider(settings.rag_answer_provider)
    answer = RagAnswer(
        answer=provider.answer(payload.question, events),
        citations=[
            Citation(
                event_id=e.event_id,
                window_start=e.window_start,
                caption_vi=e.caption_vi,
                severity=e.severity,
                similarity=round(e.similarity, 4),
            )
            for e in events
        ],
        retrieved_events=[e.event_id for e in events],
        provider=provider.name,
    )
    return ok(answer)
