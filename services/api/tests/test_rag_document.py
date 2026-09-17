"""Test cho văn bản được lập chỉ mục — app/rag/document.py.

Bài học đứng sau file này: bản đầu nhúng mỗi `caption_vi` nên mọi sự kiện giống nhau
0.83–0.92 và RAG trả về cùng một sự kiện cho mọi câu hỏi. Các test dưới đây khoá lại
những tính chất đã ngăn được chuyện đó.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.rag.document import LOCAL_TZ, build_document, part_of_day


def test_van_ban_chua_ten_lop_tieng_viet():
    """Tên lớp phải xuất hiện tường minh — đây là từ phân biệt chính giữa các sự kiện."""
    doc = build_document(
        caption_vi="Nghe thấy tiếng súng.",
        class_ids=["gunshot"],
        window_start=datetime(2026, 9, 17, 3, 0, tzinfo=timezone.utc),
    )
    assert "tiếng súng" in doc


def _ti_le_token_chung(a: str, b: str) -> float:
    tokens_a, tokens_b = set(a.split()), set(b.split())
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def test_van_ban_phan_biet_hon_han_caption_tran():
    """Hồi quy cho lỗi đã đo thật: nhúng mỗi caption làm "tiếng súng" và "tiếng pháo"
    giống nhau 0.88, và mọi câu hỏi đổ về cùng một sự kiện.

    So sánh TƯƠNG ĐỐI với cách cũ thay vì đặt một ngưỡng tuỳ ý — điều cần khoá lại là
    "khá hơn hẳn caption trần", vì đó mới đúng là thứ đã hỏng. Tỉ lệ token chung chỉ là
    xấp xỉ cho khoảng cách nhúng; phán quyết thật nằm ở phép đo đầu-cuối với BGE-M3.
    """
    when = datetime(2026, 9, 17, 3, 0, tzinfo=timezone.utc)
    caption_sung, caption_phao = "Nghe thấy tiếng súng.", "Nghe thấy tiếng pháo."

    cach_cu = _ti_le_token_chung(caption_sung, caption_phao)

    sung = build_document(caption_vi=caption_sung, class_ids=["gunshot"],
                          window_start=when, location_name="Nhà xe sinh viên",
                          area_type="parking", severity="CRITICAL")
    phao = build_document(caption_vi=caption_phao, class_ids=["fireworks"],
                          window_start=when, location_name="Hẻm khu dân cư",
                          area_type="residential", severity="LOW")
    cach_moi = _ti_le_token_chung(sung, phao)

    assert cach_moi < cach_cu * 0.75, (
        f"caption trần dùng chung {cach_cu:.0%} token, văn bản mới {cach_moi:.0%} "
        "— chưa đủ tách biệt"
    )


def test_gio_duoc_doi_sang_gio_viet_nam():
    """DB lưu UTC, nhưng người dùng hỏi theo giờ của họ. 17:56 UTC = 00:56 hôm sau."""
    doc = build_document(
        caption_vi="Nghe thấy tiếng kính vỡ.",
        class_ids=["glass_breaking"],
        window_start=datetime(2026, 9, 16, 17, 56, tzinfo=timezone.utc),
    )
    assert "00:56 ngày 17/09/2026" in doc
    assert "ban đêm" in doc


def test_datetime_ngay_tho_duoc_coi_la_utc():
    """Đoán nhầm sang giờ địa phương sẽ lệch 7 tiếng mà không báo lỗi gì."""
    ngay_tho = datetime(2026, 9, 16, 17, 56)
    co_tz = datetime(2026, 9, 16, 17, 56, tzinfo=timezone.utc)
    assert build_document(caption_vi="x", class_ids=[], window_start=ngay_tho) == \
           build_document(caption_vi="x", class_ids=[], window_start=co_tz)


def test_su_kien_khong_co_detection_van_co_dong_mo_ta():
    """Sự kiện 'không nghe thấy gì' cũng phải truy xuất được bằng chính câu hỏi ấy."""
    doc = build_document(
        caption_vi="Không phát hiện sự kiện âm thanh đáng chú ý.",
        class_ids=[],
        window_start=datetime(2026, 9, 17, 3, 0, tzinfo=timezone.utc),
    )
    assert "không có sự kiện đáng chú ý" in doc


def test_dia_diem_kem_loai_khu_vuc_tieng_viet():
    doc = build_document(
        caption_vi="Nghe thấy tiếng súng.", class_ids=["gunshot"],
        window_start=datetime(2026, 9, 17, 3, 0, tzinfo=timezone.utc),
        location_name="Nhà xe sinh viên", area_type="parking",
    )
    assert "Nhà xe sinh viên (nhà xe)" in doc


def test_area_type_la_khong_giu_nguyen_thay_vi_bia():
    """Loại khu vực chưa dịch thì hiện nguyên bản — đừng bịa tên tiếng Việt."""
    doc = build_document(
        caption_vi="x", class_ids=[], window_start=datetime(2026, 9, 17, tzinfo=timezone.utc),
        location_name="Kho lạnh", area_type="warehouse",
    )
    assert "Kho lạnh (warehouse)" in doc


@pytest.mark.parametrize(
    ("hour", "mong_doi"),
    [(0, "ban đêm"), (4, "ban đêm"), (7, "buổi sáng"), (12, "buổi trưa"),
     (16, "buổi chiều"), (20, "buổi tối"), (23, "ban đêm")],
)
def test_buoi_trong_ngay(hour: int, mong_doi: str):
    assert part_of_day(hour) == mong_doi


def test_mui_gio_la_utc_cong_7():
    assert LOCAL_TZ.utcoffset(None) == timedelta(hours=7)
