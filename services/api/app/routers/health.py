"""Health check — dùng cho Docker healthcheck và `depends_on: service_healthy`."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.envelope import ok
from app.inference_client import InferenceClient

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(session: AsyncSession = Depends(get_session)) -> dict:
    """Readiness THẬT, không phải `return {"ok": true}`.

    Một healthcheck luôn xanh bất kể database sống hay chết còn tệ hơn không có: nó làm
    `depends_on: service_healthy` mất tác dụng và đẩy lỗi xuống tận request đầu tiên.
    """
    checks: dict[str, str] = {}

    try:
        await session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as error:  # noqa: BLE001 — health check phải nuốt lỗi để báo cáo được
        checks["database"] = f"loi: {type(error).__name__}"

    # `inference` KHÔNG phải điều kiện sống của api: api vẫn phục vụ được /events và
    # /rag/query khi inference đang nạp model. Báo trạng thái, không làm api chết theo.
    checks["inference"] = "ok" if await InferenceClient().healthy() else "chua san sang"

    healthy = checks["database"] == "ok"
    return ok({"status": "healthy" if healthy else "degraded", "checks": checks})
