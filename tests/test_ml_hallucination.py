"""Test cho bộ metric hallucination — đóng góp C2, SYSTEM.md §8.2.

Đây là metric TỰ CÀI, không có thư viện tham chiếu để đối chứng, nên test là thứ duy
nhất đứng giữa một con số sai và cả một chương khoá luận dựng trên nó. SYSTEM.md §8.2 nói
thẳng: bộ trích phải được kiểm định thủ công trên 100 caption, **nếu không thì metric mới
không đáng tin**. Test ở đây phủ phần logic; phần kiểm định thủ công là việc riêng.

Bốn chỗ nguy hiểm nhất:

1. **Phủ định.** "no gunshots" mà tính thành "có nhắc tới gunshot" sẽ biến một caption
   đúng thành một ca ảo giác. Đây là lỗi làm hỏng metric theo hướng bi quan, và nó im lặng.
2. **Biên từ.** "snapshot" chứa chuỗi "shot" — khớp chuỗi thô sẽ đếm nhầm.
3. **Mẫu số 0.** Caption không nhắc sự kiện nào thì EHR **không xác định**, không phải 0.
   Trả 0 sẽ khiến "nói càng ít càng tốt" thành chiến lược tối ưu giả tạo — model im lặng
   sẽ có EHR hoàn hảo.
4. **TOA cần ít nhất hai sự kiện chung**, và cặp có onset bằng nhau thì thứ tự không xác
   định nên phải loại khỏi mẫu số, không được tính là sai.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.hallucination import (  # noqa: E402
    EVENT_LEXICON,
    chr_critical,
    ehr,
    eor,
    grounding_score,
    toa,
    trich_su_kien,
)

# ── Từ vựng ──────────────────────────────────────────────────────────────────


def test_lexicon_phu_du_16_lop():
    """Thiếu một lớp nghĩa là lớp đó KHÔNG BAO GIỜ bị tính là được nhắc tới — EOR của nó
    luôn bằng 1 và không ai biết vì sao."""
    assert len(EVENT_LEXICON) == 16
    for lop, cum in EVENT_LEXICON.items():
        assert cum, f"lớp {lop} không có cụm từ nào"


def test_moi_cum_phu_dinh_deu_la_chuoi_khong_phai_boolean():
    """Bẫy THẬT đã xảy ra 22/09: YAML 1.1 đọc `no` không nháy thành boolean `False`, nên
    từ phủ định phổ biến nhất tiếng Anh biến mất khỏi danh sách. Nếu code lúc đó dùng
    `p in text` thay vì regex thì nó đã hỏng IM LẶNG thay vì nổ."""
    from ml.evaluation.hallucination import CUM_PHU_DINH

    assert all(isinstance(p, str) for p in CUM_PHU_DINH), CUM_PHU_DINH
    assert "no" in CUM_PHU_DINH


def test_lexicon_nhan_ra_moi_cum_tu_cua_captioner_B0():
    """Bất biến nối hai module: từ vựng phải nhận ra CHÍNH caption mà B0 sinh ra.

    B0 (`services/inference/app/captioner.py`) là cận trên về grounding — mọi từ đều sinh
    từ detection có thật nên EHR của nó phải ≈ 0. Nếu lexicon không khớp cụm của B0 thì
    EHR của B0 sẽ dương một cách vô lý và cả thang đo mất mốc chuẩn. Ai sửa `vocab.py` mà
    quên `event_lexicon.yaml` sẽ làm hỏng điều đó IM LẶNG — test này là thứ bắt được.
    """
    sys.path.insert(0, str(REPO_ROOT / "services" / "inference"))
    from app.vocab import CLASS_EN

    thieu = [lop for lop, cum in CLASS_EN.items()
             if lop not in trich_su_kien(f"{cum} is heard.")]
    assert thieu == [], f"lexicon không nhận ra cụm của B0 cho lớp: {thieu}"


# ── Trích sự kiện từ caption ─────────────────────────────────────────────────


def test_trich_su_kien_khop_cum_tu_co_ban():
    assert trich_su_kien("A gunshot is heard.") == {"gunshot"}


def test_trich_su_kien_khop_nhieu_lop():
    kq = trich_su_kien("Glass breaking, then a person screaming.")
    assert kq == {"glass_breaking", "scream"}


def test_trich_su_kien_khong_phan_biet_hoa_thuong():
    assert trich_su_kien("GUNSHOT") == {"gunshot"}


def test_trich_su_kien_ton_trong_bien_tu():
    """'snapshot' chứa chuỗi 'shot'. Khớp chuỗi thô sẽ đếm nhầm thành gunshot."""
    assert "gunshot" not in trich_su_kien("He took a snapshot of the room.")


# ── Phủ định ─────────────────────────────────────────────────────────────────


def test_phu_dinh_khong_tinh_la_nhac_toi():
    assert trich_su_kien("There is no gunshot in this clip.") == set()


def test_phu_dinh_khong_lan_sang_lop_khac():
    """'no screaming' phủ định screaming, KHÔNG phủ định door slam đứng trước nó."""
    kq = trich_su_kien("A door slamming, but no screaming.")
    assert kq == {"door_slam"}


def test_phu_dinh_phu_ca_cum_lien_ke():
    kq = trich_su_kien("No gunshot or explosion was heard.")
    assert kq == set()


def test_phu_dinh_dang_rut_gon():
    assert trich_su_kien("There isn't a siren.") == set()


# ── EHR / EOR / GS ───────────────────────────────────────────────────────────


def test_ehr_dem_dung_phan_bia_ra():
    # nhắc 3, thật có 2 trong số đó → 1/3 bịa
    assert abs(ehr({"gunshot", "siren", "laughter"}, {"gunshot", "siren"}) - 1 / 3) < 1e-9


def test_ehr_khong_nhac_gi_tra_none_khong_phai_0():
    """Caption im lặng KHÔNG phải caption hoàn hảo. Trả 0 ở đây làm 'nói càng ít càng
    tốt' thành chiến lược thắng giả tạo."""
    assert ehr(set(), {"gunshot"}) is None


def test_eor_dem_dung_phan_bo_sot():
    assert abs(eor({"gunshot"}, {"gunshot", "siren"}) - 0.5) < 1e-9


def test_eor_khong_co_su_kien_that_tra_none():
    assert eor({"gunshot"}, set()) is None


def test_grounding_score_la_jaccard():
    assert abs(grounding_score({"a", "b"}, {"b", "c"}) - 1 / 3) < 1e-9


def test_grounding_score_hai_tap_rong_tra_none():
    assert grounding_score(set(), set()) is None


# ── CHR — chỉ tính trên lớp tier Critical ────────────────────────────────────


def test_chr_chi_tinh_lop_critical():
    """Bịa 'tiếng súng' nguy hiểm hơn bịa 'bước chân' — CHR phải bỏ qua lớp không Critical."""
    # nhắc gunshot (Critical, bịa) + laughter (không Critical, bịa) → CHR chỉ thấy gunshot
    assert chr_critical({"gunshot", "laughter"}, {"laughter"}) == 1.0


def test_chr_khong_nhac_lop_critical_nao_tra_none():
    assert chr_critical({"laughter"}, {"laughter"}) is None


# ── TOA ──────────────────────────────────────────────────────────────────────


def test_toa_thu_tu_dung_hoan_toan():
    caption = "A gunshot, then a siren."
    onset = {"gunshot": 1.0, "siren": 5.0}
    assert toa(caption, onset) == 1.0


def test_toa_thu_tu_nguoc():
    caption = "A siren, then a gunshot."
    onset = {"gunshot": 1.0, "siren": 5.0}
    assert toa(caption, onset) == 0.0


def test_toa_duoi_hai_su_kien_chung_tra_none():
    """Một sự kiện thì không có cặp nào để xét thứ tự — CHƯA ĐO, không phải 0."""
    assert toa("A gunshot.", {"gunshot": 1.0}) is None


def test_toa_cap_cung_onset_bi_loai_khoi_mau_so():
    """Hai sự kiện khởi đầu cùng lúc thì không có thứ tự đúng — tính là sai sẽ phạt oan
    caption nói 'cùng lúc'."""
    caption = "A gunshot and a siren."
    onset = {"gunshot": 2.0, "siren": 2.0}
    assert toa(caption, onset) is None
