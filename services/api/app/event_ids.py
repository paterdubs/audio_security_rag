"""Sinh `event_id` dạng `EVT_YYYYMMDD_NNNNNN` — SYSTEM.md §4.3.

Vì sao không dùng UUID: `event_id` xuất hiện trong citation của RAG, trong báo cáo, và
trong lúc người vận hành đọc to cho nhau nghe. Một id mang sẵn ngày tháng và số thứ tự
tăng dần đọc được, sắp xếp được, và nhìn là biết sự kiện xảy ra hôm nào.
"""

from __future__ import annotations

import re
from datetime import datetime

EVENT_ID_RE = re.compile(r"^EVT_(\d{8})_(\d{6})$")
SEQUENCE_WIDTH = 6


def format_event_id(moment: datetime, sequence: int) -> str:
    if sequence < 1 or sequence >= 10**SEQUENCE_WIDTH:
        raise ValueError(f"sequence ngoài khoảng 1..{10**SEQUENCE_WIDTH - 1}: {sequence}")
    return f"EVT_{moment:%Y%m%d}_{sequence:0{SEQUENCE_WIDTH}d}"


def next_event_id(moment: datetime, latest_today: str | None) -> str:
    """id kế tiếp trong NGÀY của `moment`.

    `latest_today` là event_id lớn nhất đã có của chính ngày đó (None nếu chưa có).
    Đếm lại từ 1 mỗi ngày là có chủ đích — số thứ tự trong ngày cho biết ngay "hôm nay
    đã có bao nhiêu sự kiện", thứ mà một bộ đếm chạy suốt đời không nói được.
    """
    if latest_today is None:
        return format_event_id(moment, 1)

    match = EVENT_ID_RE.match(latest_today)
    if not match:
        raise ValueError(f"event_id không đúng định dạng: {latest_today!r}")
    if match.group(1) != f"{moment:%Y%m%d}":
        # Ngày khác nghĩa là gọi sai: đếm tiếp từ id của hôm qua sẽ làm số thứ tự
        # trong ngày mới bắt đầu ở giữa chừng và không còn nghĩa gì.
        raise ValueError(f"latest_today {latest_today!r} không thuộc ngày {moment:%Y%m%d}")
    return format_event_id(moment, int(match.group(2)) + 1)
