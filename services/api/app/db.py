"""Engine + session async cho SQLAlchemy 2.0."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        settings = get_settings()
        # pool_pre_ping: container db có thể khởi động lại độc lập với api; không có nó
        # thì api giữ connection đã chết và mọi request sau đó lỗi cho tới khi restart tay.
        _engine = create_async_engine(settings.database_url, pool_pre_ping=True, future=True)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency của FastAPI: một session cho mỗi request, đóng chắc chắn khi xong."""
    async with get_session_factory()() as session:
        yield session
