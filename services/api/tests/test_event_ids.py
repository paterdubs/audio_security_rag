"""Test sinh event_id — SYSTEM.md §4.3.

`event_id` là thứ RAG trích dẫn và người vận hành đọc cho nhau nghe. Sinh trùng id nghĩa
là hai sự kiện khác nhau chỉ về một dòng trong DB, và citation trỏ sai sự kiện.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.event_ids import format_event_id, next_event_id  # noqa: E402

MOMENT = datetime(2026, 9, 16, 22, 31, 5)


def test_dinh_dang_dung_spec():
    assert format_event_id(MOMENT, 123) == "EVT_20260916_000123"


def test_ngay_moi_dem_lai_tu_1():
    """Đếm lại mỗi ngày là có chủ đích: số thứ tự cho biết ngay "hôm nay đã có bao nhiêu
    sự kiện", thứ mà bộ đếm chạy suốt đời không nói được."""
    assert next_event_id(MOMENT, None) == "EVT_20260916_000001"


def test_dem_tiep_trong_ngay():
    assert next_event_id(MOMENT, "EVT_20260916_000123") == "EVT_20260916_000124"


def test_qua_moc_nghin_khong_tran_so():
    assert next_event_id(MOMENT, "EVT_20260916_000999") == "EVT_20260916_001000"


def test_id_cua_ngay_khac_bi_tu_choi():
    """Đếm tiếp từ id hôm qua sẽ làm số thứ tự ngày mới bắt đầu ở giữa chừng — thà dừng
    còn hơn sinh ra một dãy id vô nghĩa mà không ai nhận ra."""
    with pytest.raises(ValueError, match="không thuộc ngày"):
        next_event_id(MOMENT, "EVT_20260915_000123")


def test_id_sai_dinh_dang_bi_tu_choi():
    with pytest.raises(ValueError, match="không đúng định dạng"):
        next_event_id(MOMENT, "khong-phai-event-id")


def test_sequence_ngoai_khoang_bi_tu_choi():
    with pytest.raises(ValueError):
        format_event_id(MOMENT, 0)
    with pytest.raises(ValueError):
        format_event_id(MOMENT, 1_000_000)


def test_id_sap_xep_duoc_theo_thu_tu_thoi_gian():
    # Zero-pad đủ rộng thì sort chuỗi = sort thời gian. Thiếu pad thì "EVT_..._10" đứng
    # trước "EVT_..._9" và mọi bảng hiển thị theo id đều sai thứ tự.
    ids = [format_event_id(MOMENT, n) for n in (9, 10, 100)]
    assert ids == sorted(ids)
