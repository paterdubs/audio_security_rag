"""Test cho pipeline nối B0 (`services/inference/app/captioner.py`) với bộ metric
hallucination (`ml/evaluation/hallucination.py`) — SYSTEM.md §8.2/§8.5.

Đây là phép đo TỰ NHẤT QUÁN trên `data/synthetic/dev`: caption B0 sinh thẳng từ nhãn dự
báo của chính model, đối chiếu với nhãn tham chiếu strong-label sẵn có. Nó KHÔNG cần
G2 (chưa có), vì EHR/EOR/GS/TOA/CHR chỉ cần văn bản caption + tập sự kiện tham chiếu, không
cần human reference. Đây là phép "nối ống", không phải phép đo cuối cùng cho khoá luận.

Ba chỗ nguy hiểm:

1. `frames_to_events()`/`tham_chieu()` trả khoá `event_label`; `caption_en()`/`caption_vi()`
   đòi khoá `class_id`. Lẫn khoá là lỗi câm — caption sẽ luôn rỗng mà không ném lỗi nào.
2. B0 là "cận trên grounding": EHR của nó chỉ có thể dương vì (a) lexicon không nhận ra
   cụm caption sinh ra, hoặc (b) model DỰ BÁO sai (detection không khớp tham chiếu). Test
   phải phân biệt được — lexicon sai là lỗi hạ tầng, model dự báo sai là tín hiệu thật.
3. Tổng hợp nhiều clip phải BỎ QUA `None`, không được coi `None` là 0 khi tính trung bình.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "services" / "inference"))

from ml.evaluation.caption_b0 import (  # noqa: E402
    danh_gia_mot_clip,
    doi_dinh_dang_su_kien,
    lop_that_cua_clip,
    onset_theo_lop,
    tong_hop,
)


def _sk(nhan: str, on: float, off: float) -> dict:
    return {"event_label": nhan, "onset": on, "offset": off}


# ── Đổi định dạng: event_label -> class_id ──────────────────────────────────


def test_doi_dinh_dang_su_kien_doi_dung_khoa():
    """Bẫy có thật: frames_to_events()/tham_chieu() dùng 'event_label', captioner đòi
    'class_id'. Lẫn khoá làm caption luôn rỗng mà không lỗi nào báo."""
    ra = doi_dinh_dang_su_kien([_sk("gunshot", 1.0, 2.0)])
    assert ra == [{"class_id": "gunshot", "onset": 1.0, "offset": 2.0}]


def test_doi_dinh_dang_su_kien_danh_sach_rong():
    assert doi_dinh_dang_su_kien([]) == []


# ── Tập lớp / onset tham chiếu ───────────────────────────────────────────────


def test_lop_that_cua_clip_gom_khong_trung():
    kq = lop_that_cua_clip([_sk("gunshot", 1.0, 2.0), _sk("gunshot", 5.0, 6.0),
                            _sk("siren", 3.0, 4.0)])
    assert kq == {"gunshot", "siren"}


def test_lop_that_cua_clip_rong():
    assert lop_that_cua_clip([]) == set()


def test_onset_theo_lop_lay_onset_som_nhat():
    kq = onset_theo_lop([_sk("gunshot", 5.0, 6.0), _sk("gunshot", 1.0, 2.0)])
    assert kq["gunshot"] == 1.0


# ── Đánh giá một clip ────────────────────────────────────────────────────────


def test_danh_gia_mot_clip_du_bao_dung_het_thi_ehr_bang_0():
    """B0 là cận trên grounding: nếu detection = tham chiếu hệt nhau thì EHR/EOR phải 0."""
    du_bao = [_sk("gunshot", 1.0, 2.0)]
    tham_chieu = [_sk("gunshot", 1.0, 2.0)]
    kq = danh_gia_mot_clip(du_bao, tham_chieu, ngon_ngu="en")
    assert kq["ehr"] == 0.0
    assert kq["eor"] == 0.0
    assert kq["gs"] == 1.0


def test_danh_gia_mot_clip_du_bao_them_lop_gia_thi_ehr_duong():
    """Model dự báo thừa một lớp không có thật -> EHR phải bắt được, vì đây là tín hiệu
    THẬT (model sai), không phải lỗi hạ tầng của lexicon."""
    du_bao = [_sk("gunshot", 1.0, 2.0), _sk("siren", 5.0, 6.0)]
    tham_chieu = [_sk("gunshot", 1.0, 2.0)]
    kq = danh_gia_mot_clip(du_bao, tham_chieu, ngon_ngu="en")
    assert kq["ehr"] == 0.5
    assert kq["eor"] == 0.0


def test_danh_gia_mot_clip_khong_du_bao_gi_ehr_none():
    """Caption rỗng (không detection nào) -> |P|=0 -> EHR None, không phải 0 hay 1."""
    kq = danh_gia_mot_clip([], [_sk("gunshot", 1.0, 2.0)], ngon_ngu="en")
    assert kq["ehr"] is None
    assert kq["eor"] == 1.0


def test_danh_gia_mot_clip_ghi_ca_caption_va_hai_tap():
    kq = danh_gia_mot_clip([_sk("gunshot", 1.0, 2.0)], [_sk("gunshot", 1.0, 2.0)],
                           ngon_ngu="en")
    assert kq["caption"] == "A gunshot is heard."
    assert kq["P"] == ["gunshot"]
    assert kq["G"] == ["gunshot"]


def test_danh_gia_mot_clip_ngon_ngu_vi():
    kq = danh_gia_mot_clip([_sk("gunshot", 1.0, 2.0)], [_sk("gunshot", 1.0, 2.0)],
                           ngon_ngu="vi")
    assert "tiếng súng" in kq["caption"]


# ── Tổng hợp nhiều clip ──────────────────────────────────────────────────────


def test_tong_hop_bo_qua_none_khong_tinh_la_0():
    """Đây là chỗ dễ sai nhất: nếu None bị coi là 0, một clip caption rỗng (EHR=None) sẽ
    kéo trung bình EHR xuống sai, làm model trông "ít ảo giác hơn" một cách giả tạo."""
    ket_qua = [
        {"ehr": 0.5, "eor": 0.0, "gs": 0.5, "chr": None, "toa": None},
        {"ehr": None, "eor": 1.0, "gs": 0.0, "chr": None, "toa": None},
        {"ehr": 0.3, "eor": 0.0, "gs": 0.7, "chr": None, "toa": None},
    ]
    tt = tong_hop(ket_qua)
    assert abs(tt["ehr"]["trung_binh"] - 0.4) < 1e-9   # (0.5+0.3)/2, KHÔNG /3
    assert tt["ehr"]["n_do_duoc"] == 2
    assert tt["ehr"]["n_chua_do"] == 1
    assert tt["chr"]["n_do_duoc"] == 0
    assert tt["chr"]["trung_binh"] is None


def test_tong_hop_danh_sach_rong_khong_no():
    tt = tong_hop([])
    assert tt["ehr"]["trung_binh"] is None
    assert tt["ehr"]["n_do_duoc"] == 0
