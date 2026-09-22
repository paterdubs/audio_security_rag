"""Test cho `scripts/kiem_dinh_lexicon.py` — điều kiện cần của C2 (SYSTEM.md §8.2):
kiểm định thủ công bộ trích lexicon trên 100 caption, báo cáo độ chính xác của CHÍNH bộ
trích (không phải grounding của model).

Ba chỗ nguy hiểm:

1. Mẫu 100 caption phải RẢI ĐỀU theo độ phức tạp câu (0/1/2/≥3 sự kiện được nhắc) — B0
   dùng ba cấu trúc cú pháp khác nhau tuỳ số sự kiện (`captioner.py`: một câu đơn, "sau
   đó", "cùng lúc"). Lấy mẫu ngẫu nhiên thuần tuý có thể toàn câu đơn giản nhất, không
   kiểm được nhánh cú pháp phức tạp.
2. Cột `sai`/`thieu` người điền phải phân biệt được: `sai` = lexicon trích NHẦM (báo có
   mà caption không thật sự nói); `thieu` = lexicon TRÍCH SÓT (caption có nói mà lexicon
   không bắt được). Hai loại lỗi này cho ra Precision/Recall khác nhau của CHÍNH bộ trích,
   không phải của model.
3. Caption/hàng rỗng (không sự kiện nào được trích) không được làm mẫu số về 0 một cách
   im lặng — phải trả None đúng kỷ luật "chưa đo ≠ 0" đã dùng xuyên suốt dự án.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from kiem_dinh_lexicon import (  # noqa: E402
    chon_mau_kiem_dinh,
    doc_bang_kiem,
    tach_danh_sach,
    tinh_do_chinh_xac,
    viet_bang_kiem,
)


def _r(clip_id: str, caption: str, P: list[str]) -> dict:
    return {"clip_id": clip_id, "caption": caption, "P": P}


# ── Chọn mẫu rải theo độ phức tạp câu ────────────────────────────────────────


def test_chon_mau_rai_deu_theo_so_su_kien():
    ket_qua = (
        [_r(f"don_{i}", "x", ["a"]) for i in range(20)]
        + [_r(f"rong_{i}", "y", []) for i in range(20)]
        + [_r(f"doi_{i}", "z", ["a", "b"]) for i in range(20)]
        + [_r(f"ba_{i}", "w", ["a", "b", "c"]) for i in range(20)]
    )
    mau = chon_mau_kiem_dinh(ket_qua, n=8, seed="s")
    nhom = {min(len(r["P"]), 3) for r in mau}
    assert nhom == {0, 1, 2, 3}, f"thiếu nhóm độ phức tạp: {nhom}"


def test_chon_mau_deterministic():
    ket_qua = [_r(f"c{i}", "x", ["a"] * (i % 4)) for i in range(40)]
    a = chon_mau_kiem_dinh(ket_qua, n=10, seed="hat-giong")
    b = chon_mau_kiem_dinh(ket_qua, n=10, seed="hat-giong")
    assert [r["clip_id"] for r in a] == [r["clip_id"] for r in b]


def test_chon_mau_khong_vuot_qua_n():
    ket_qua = [_r(f"c{i}", "x", ["a"]) for i in range(200)]
    mau = chon_mau_kiem_dinh(ket_qua, n=100, seed="s")
    assert len(mau) == 100


def test_chon_mau_nhom_thieu_khong_lam_chet_vong_lap():
    """Chỉ có nhóm 0 và 1 sự kiện — round-robin phải bỏ qua nhóm 2/3 rỗng, không treo."""
    ket_qua = [_r(f"c{i}", "x", []) for i in range(5)] + [_r(f"d{i}", "y", ["a"]) for i in range(5)]
    mau = chon_mau_kiem_dinh(ket_qua, n=8, seed="s")
    assert len(mau) == 8


# ── Đọc/ghi bảng kiểm ─────────────────────────────────────────────────────────


def test_viet_va_doc_bang_kiem_roundtrip(tmp_path):
    mau = [_r("c1", "A gunshot is heard.", ["gunshot"])]
    path = tmp_path / "bang.csv"
    viet_bang_kiem(mau, path)
    rows = doc_bang_kiem(path)
    assert rows[0]["clip_id"] == "c1"
    assert rows[0]["caption"] == "A gunshot is heard."
    assert rows[0]["P_tu_dong"] == "gunshot"
    assert rows[0]["sai"] == ""
    assert rows[0]["thieu"] == ""


def test_tach_danh_sach_rong_tra_danh_sach_rong():
    assert tach_danh_sach("") == []


def test_tach_danh_sach_nhieu_phan_tu_boi_dau_phay():
    assert tach_danh_sach("gunshot, siren") == ["gunshot", "siren"]


# ── Tính độ chính xác từ bảng đã người điền ─────────────────────────────────


def test_tinh_do_chinh_xac_khong_loi_nao_thi_bang_1():
    rows = [{"P_tu_dong": "gunshot;siren", "sai": "", "thieu": ""} for _ in range(5)]
    kq = tinh_do_chinh_xac(rows)
    assert kq["do_chinh_xac"] == 1.0
    assert kq["do_bao_phu"] == 1.0
    assert kq["ti_le_hoan_toan_dung"] == 1.0


def test_tinh_do_chinh_xac_dem_dung_ca_sai_lan_thieu():
    rows = [
        {"P_tu_dong": "gunshot;siren", "sai": "siren", "thieu": ""},   # 1 sai
        {"P_tu_dong": "laughter", "sai": "", "thieu": "applause_cheering"},  # 1 thiếu
        {"P_tu_dong": "scream", "sai": "", "thieu": ""},               # sạch
    ]
    kq = tinh_do_chinh_xac(rows)
    # tổng trích = 4 (2+1+1), sai = 1 -> chính xác = 3/4
    assert abs(kq["do_chinh_xac"] - 0.75) < 1e-9
    # đúng-thật = 3 (4 trích - 1 sai), thiếu = 1 -> bao phủ = 3/(3+1) = 0.75
    assert abs(kq["do_bao_phu"] - 0.75) < 1e-9
    assert abs(kq["ti_le_hoan_toan_dung"] - 1 / 3) < 1e-9


def test_tinh_do_chinh_xac_khong_co_hang_nao_trich_duoc_gi_tra_none():
    """Mọi caption đều rỗng (P_tu_dong rỗng ở mọi hàng) -> mẫu số 0 -> None, KHÔNG phải 1
    hay 0. Trả 1 sẽ khoe 'chính xác tuyệt đối' một cách giả tạo khi thực ra chưa đo được
    gì; trả 0 thì phạt oan một lexicon đúng nhưng gặp toàn clip không sự kiện."""
    rows = [{"P_tu_dong": "", "sai": "", "thieu": ""} for _ in range(3)]
    kq = tinh_do_chinh_xac(rows)
    assert kq["do_chinh_xac"] is None


def test_tinh_do_chinh_xac_danh_sach_rong_tra_none_het():
    kq = tinh_do_chinh_xac([])
    assert kq["do_chinh_xac"] is None
    assert kq["do_bao_phu"] is None
    assert kq["ti_le_hoan_toan_dung"] is None
