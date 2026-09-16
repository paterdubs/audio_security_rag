"""Test sinh câu trả lời RAG — ràng buộc cứng của SYSTEM.md §4.4 và §7.3.

Ràng buộc "không có bằng chứng thì không bịa" là điều kiện cần của cả đóng góp C2 (đo
hallucination). Nếu chính tầng RAG bịa thì mọi con số EHR đo ở tầng caption đều vô nghĩa.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.answer import NO_EVIDENCE_VI, TemplateProvider, get_provider  # noqa: E402
from app.rag.retrieval import RetrievedEvent  # noqa: E402


def _event(event_id: str = "EVT_20260916_000001", caption: str = "Tiếng kính vỡ, sau đó là tiếng hét.") -> RetrievedEvent:
    return RetrievedEvent(
        event_id=event_id,
        window_start=datetime(2026, 9, 16, 22, 31, 5),
        caption_vi=caption,
        severity="CRITICAL",
        similarity=0.87,
    )


def test_khong_co_bang_chung_thi_tu_choi_chu_khong_bia():
    answer = TemplateProvider().answer("tối qua có tiếng súng không?", [])
    assert answer == NO_EVIDENCE_VI
    assert "không suy đoán" in answer


def test_cau_tra_loi_chua_event_id_de_truy_nguoc():
    answer = TemplateProvider().answer("có gì bất thường không?", [_event()])
    assert "EVT_20260916_000001" in answer


def test_moi_cau_chu_deu_lay_tu_caption_da_luu():
    """Provider trích xuất: không được thêm thông tin không có trong bằng chứng."""
    caption = "Tiếng chuông báo động kéo dài."
    answer = TemplateProvider().answer("có cháy không?", [_event(caption=caption)])
    assert caption in answer
    # Câu hỏi nhắc tới "cháy" nhưng bằng chứng không nói gì về cháy — câu trả lời không
    # được tự suy ra có cháy.
    assert "cháy" not in answer.lower()


def test_nhieu_su_kien_thi_liet_ke_het_khong_chon_bua_mot_cai():
    events = [_event("EVT_20260916_000001"), _event("EVT_20260916_000002")]
    answer = TemplateProvider().answer("tối qua thế nào?", events)
    assert "EVT_20260916_000001" in answer and "EVT_20260916_000002" in answer


def test_hien_thi_thoi_diem_de_nguoi_doc_hieu_ngay():
    answer = TemplateProvider().answer("khi nào?", [_event()])
    assert "22:31 ngày 16/09/2026" in answer


def test_provider_la_thi_dung_ngay_khong_am_tham_ve_template():
    """Tưởng đang chạy Claude mà thực ra chạy template sẽ cho số liệu đánh giá sai hoàn
    toàn mà không ai biết — thà chết lúc khởi động."""
    with pytest.raises(ValueError, match="chưa được hiện thực"):
        get_provider("claude-khong-ton-tai")


def test_provider_mac_dinh_la_template():
    assert get_provider("template").name == "template"
