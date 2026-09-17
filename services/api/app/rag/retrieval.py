"""Truy xuất lai ghép — SYSTEM.md §7.2.

Điểm cốt lõi: lọc metadata (khoảng thời gian, địa điểm, mức rủi ro) và tìm kiếm ngữ nghĩa
nằm trong **MỘT câu SQL**, không phải hai lượt gọi rồi tự merge. Đây chính là lý do chọn
pgvector thay Qdrant (§4.2) — nếu tách làm hai bước thì lợi thế kiến trúc ấy mất sạch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class RetrievedEvent:
    event_id: str
    window_start: datetime
    caption_vi: str
    severity: str
    similarity: float


# `<=>` là cosine distance của pgvector (0 = giống hệt). Index HNSW được tạo với
# vector_cosine_ops nên phải dùng đúng toán tử này, nếu không Postgres bỏ qua index.
_SQL = """
SELECT event_id, window_start, caption_vi, severity,
       1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity
FROM security_events
WHERE embedding IS NOT NULL
  AND (CAST(:from_ts AS timestamptz) IS NULL OR window_start >= CAST(:from_ts AS timestamptz))
  AND (CAST(:to_ts   AS timestamptz) IS NULL OR window_start <= CAST(:to_ts   AS timestamptz))
  AND (CAST(:location AS text) IS NULL OR location_id = CAST(:location AS text))
  AND (CAST(:severity AS text) IS NULL OR severity   = CAST(:severity AS text))
  -- Ngưỡng bằng chứng. Viết theo KHOẢNG CÁCH (<=) chứ không theo similarity (>=) để
  -- Postgres còn dùng được index HNSW; hai cách tương đương vì similarity = 1 - distance.
  -- Thiếu dòng này thì mọi câu hỏi đều có citation, kể cả câu lạc đề hoàn toàn — nhánh
  -- "không có sự kiện nào" của §4.4 trở thành code chết.
  AND (embedding <=> CAST(:query_vec AS vector)) <= :max_distance
ORDER BY embedding <=> CAST(:query_vec AS vector)
LIMIT :top_k
"""


async def retrieve(
    session: AsyncSession,
    query_vector: list[float],
    *,
    top_k: int,
    from_ts: datetime | None = None,
    to_ts: datetime | None = None,
    location: str | None = None,
    severity: str | None = None,
    min_similarity: float = 0.0,
) -> list[RetrievedEvent]:
    rows = await session.execute(
        text(_SQL),
        {
            # pgvector nhận literal dạng '[1,2,3]'; truyền list Python thẳng vào sẽ
            # thành mảng Postgres và câu lệnh lỗi kiểu.
            "query_vec": "[" + ",".join(f"{v:.6f}" for v in query_vector) + "]",
            "from_ts": from_ts,
            "to_ts": to_ts,
            "location": location,
            "severity": severity,
            "max_distance": 1.0 - min_similarity,
            "top_k": top_k,
        },
    )
    return [
        RetrievedEvent(
            event_id=row.event_id,
            window_start=row.window_start,
            caption_vi=row.caption_vi,
            severity=row.severity,
            similarity=float(row.similarity),
        )
        for row in rows
    ]
