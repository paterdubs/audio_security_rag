"""Seed dữ liệu tối thiểu để hệ thống chạy được ngay sau `docker compose up` (§9.5).

Chỉ seed `locations`: `security_events` phải đến từ audio thật, không được tạo sẵn sự
kiện giả — dashboard hiển thị sự kiện bịa sẽ làm người xem tưởng hệ thống đang hoạt động
trong khi chưa hề xử lý byte audio nào.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Location

# Bốn khu vực triển khai của đề tài (SYSTEM.md §1.4).
DEFAULT_LOCATIONS = [
    Location(location_id="HALL_03", name="Hành lang tầng 2 nhà B", area_type="school",
             description="Khu vực lớp học, đông người giờ ra chơi"),
    Location(location_id="PARK_01", name="Nhà xe sinh viên", area_type="parking",
             description="Nhà xe có mái, vọng âm mạnh"),
    Location(location_id="RESI_01", name="Hẻm khu dân cư", area_type="residential",
             description="Khu dân cư, nhiều pháo dịp lễ Tết"),
    Location(location_id="FACT_01", name="Xưởng cơ khí", area_type="factory",
             description="Nền ồn máy móc liên tục"),
]


async def seed_locations(session: AsyncSession) -> int:
    """Thêm các location chưa có. Idempotent: chạy lại mỗi lần khởi động không sao."""
    existing = set((await session.scalars(select(Location.location_id))).all())
    added = [loc for loc in DEFAULT_LOCATIONS if loc.location_id not in existing]
    for location in added:
        session.add(location)
    if added:
        await session.commit()
    return len(added)
