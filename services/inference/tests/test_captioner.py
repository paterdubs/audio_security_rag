"""Test caption template — baseline B0 của SYSTEM.md §5.5.

B0 là cận trên về grounding: mọi từ đều sinh từ detection có thật nên EHR ≈ 0. Test ở đây
bảo vệ đúng tính chất ấy — nếu caption nhắc tới thứ không có trong detection thì B0 mất
vai trò đường cơ sở và mọi so sánh ở §8.3 hỏng theo.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.captioner import NO_EVENT_EN, NO_EVENT_VI, caption_en, caption_vi  # noqa: E402


def _d(class_id: str, onset: float, offset: float) -> dict:
    return {"class_id": class_id, "onset": onset, "offset": offset, "confidence": 0.9}


# ── Không có sự kiện ─────────────────────────────────────────────────────────


def test_khong_co_su_kien_thi_noi_thang():
    assert caption_vi([]) == NO_EVENT_VI
    assert caption_en([]) == NO_EVENT_EN


# ── Một sự kiện ──────────────────────────────────────────────────────────────


def test_mot_su_kien():
    assert caption_vi([_d("glass_breaking", 1.0, 2.0)]) == "Nghe thấy tiếng kính vỡ."
    assert caption_en([_d("glass_breaking", 1.0, 2.0)]) == "Glass breaking is heard."


# ── Thứ tự thời gian: điều kiện cần để đo TOA (§8.2) ────────────────────────


def test_su_kien_noi_tiep_dung_thu_tu():
    detections = [_d("scream", 3.0, 5.0), _d("glass_breaking", 1.0, 2.0)]
    assert caption_vi(detections) == "Nghe thấy tiếng kính vỡ, sau đó là tiếng hét hoảng loạn."


def test_su_kien_chong_lan_noi_cung_luc():
    """Chồng lấn và nối tiếp là hai quan hệ thời gian KHÁC NHAU. Nói "sau đó" cho hai âm
    xảy ra cùng lúc là một dạng hallucination về thứ tự, không phải lỗi từ ngữ."""
    detections = [_d("glass_breaking", 1.0, 3.0), _d("scream", 2.0, 4.0)]
    assert "cùng lúc" in caption_vi(detections)
    assert "same time" in caption_en(detections)


def test_ba_su_kien_noi_tiep():
    detections = [
        _d("glass_breaking", 1.0, 2.0),
        _d("scream", 3.0, 4.0),
        _d("running_footsteps", 5.0, 7.0),
    ]
    result = caption_vi(detections)
    assert result == "Nghe thấy tiếng kính vỡ, sau đó là tiếng hét hoảng loạn và tiếng bước chân chạy."


# ── Grounding: không nhắc thứ không có trong detection ──────────────────────


def test_caption_chi_nhac_lop_co_trong_detection():
    result = caption_vi([_d("fireworks", 0.0, 2.0)])
    assert "tiếng pháo" in result
    for khong_co in ("súng", "nổ", "hét", "kính"):
        assert khong_co not in result


def test_lop_la_giu_nguyen_id_khong_bia_ten():
    # Lớp chưa có trong glossary: hiện đúng id thay vì bịa một tên tiếng Việt.
    assert "lop_moi_chua_dich" in caption_vi([_d("lop_moi_chua_dich", 0.0, 1.0)])
