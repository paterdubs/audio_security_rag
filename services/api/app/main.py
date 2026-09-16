"""FastAPI app cho service `api` — SYSTEM.md §4.2, §4.4."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import get_settings
from app.db import get_session_factory
from app.envelope import fail
from app.routers import alerts, audio, events, health, rag, system
from app.seed import seed_locations

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    # Seed idempotent, chạy mỗi lần khởi động — `docker compose up` từ máy trắng phải có
    # sẵn location để upload được ngay (§9.5), không bắt người dùng tự INSERT bằng tay.
    try:
        async with get_session_factory()() as session:
            added = await seed_locations(session)
            if added:
                logger.info("seed %d location", added)
    except Exception as error:  # noqa: BLE001
        # Seed hỏng không được chặn service khởi động: healthcheck sẽ báo database lỗi,
        # và như vậy nguyên nhân hiện ra ở đúng chỗ người ta nhìn.
        logger.warning("seed that bai: %s", error)
    yield


settings = get_settings()
app = FastAPI(
    title="Audio Security RAG API",
    version="0.1.0",
    description="Giám sát an ninh bằng âm thanh + truy xuất cảnh báo qua RAG",
    lifespan=lifespan,
)

# Dashboard chạy ở cổng khác (3000) nên trình duyệt coi là cross-origin. Ở bản một node
# này chấp nhận mở; siết lại theo domain thật khi triển khai ngoài máy cá nhân.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Lỗi cũng phải đi qua envelope: client chỉ cần MỘT chỗ xử lý lỗi (§4.4)."""
    return JSONResponse(
        status_code=exc.status_code,
        content=fail(code=f"http_{exc.status_code}", message=str(exc.detail)),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=fail(code="validation_error", message="Dữ liệu gửi lên không hợp lệ",
                     meta={"errors": exc.errors()}),
    )


for router in (health.router, events.router, audio.router, rag.router, system.router, alerts.router):
    app.include_router(router, prefix=settings.api_prefix)
