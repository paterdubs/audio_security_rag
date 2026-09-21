"""Test cho hướng điều tra thứ tư của `long_event` — cắt chéo với các lát cắt khác.

Ba ứng viên trước đều bị đo bác bỏ (cửa sổ lọc hẹp, trần cửa sổ thấp, khe hở năng lượng
thật). Cả ba đều đo trên HÀNG `long_event` của bảng lát cắt, mà hàng đó lẫn lộn: 52/144
clip `long_event` của dev đồng thời là `reverb`, 45 là `low_snr`. Giả thuyết ở đây là
`long_event` xấu KHÔNG phải vì độ dài mà vì trùng lát cắt khó khác — và giả thuyết đó chỉ
kiểm được bằng bảng 2×2, nên mọi test dưới đây dựng cả hai nhánh để so, không đo một nhánh
rồi suy diễn.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.long_event_crosscut import (  # noqa: E402
    ti_le_cap_nham,
    ti_le_phan_manh,
)


def sk(lop: str, on: float, off: float) -> dict:
    return {"event_label": lop, "onset": on, "offset": off}


def test_ti_le_phan_manh_dem_dung_tu_so_va_mau_so():
    tham_chieu = {"a": [sk("siren", 0.0, 5.0)], "b": [sk("siren", 0.0, 5.0)]}
    du_bao = {"a": [sk("siren", 0.0, 2.0), sk("siren", 3.0, 5.0)],
              "b": [sk("siren", 0.1, 4.9)]}
    kq = ti_le_phan_manh(tham_chieu, du_bao, ["a", "b"])
    assert kq["n_ref"] == 2
    assert kq["n_vo"] == 1
    assert kq["ti_le"] == 0.5


def test_ti_le_phan_manh_o_rong_tra_ve_none_chu_khong_phai_0():
    """Ô 2×2 rỗng là CHƯA ĐO, không phải "tỉ lệ vỡ bằng 0" — trả 0 ở đây sẽ tạo ra đúng
    kiểu kết luận sai mà dự án đã dặn: F1=0 vì chưa đo bị đọc thành F1 bằng 0."""
    kq = ti_le_phan_manh({}, {}, [])
    assert kq["n_ref"] == 0
    assert kq["ti_le"] is None


def test_ti_le_phan_manh_bo_qua_clip_ngoai_danh_sach():
    tham_chieu = {"a": [sk("siren", 0.0, 5.0)], "b": [sk("siren", 0.0, 5.0)]}
    du_bao = {"a": [sk("siren", 0.0, 2.0), sk("siren", 3.0, 5.0)], "b": []}
    kq = ti_le_phan_manh(tham_chieu, du_bao, ["b"])
    assert kq["n_ref"] == 1
    assert kq["n_vo"] == 0


# ── Ma trận nhầm theo từng ô ─────────────────────────────────────────────────
#
# `ma_tran_nham()` nhận KetQuaPhanLoai của TOÀN tập; chưa ai gọi nó trên một tập con clip.
# Lấy nhầm clip ngoài ô vào ma trận là đúng cái bẫy mẫu số đã gặp ở `nhom_theo_lat_cat` —
# cặp nhầm của ô khác sẽ hiện lên như cặp nhầm của ô này mà không có lỗi nào bắn ra.


def test_cap_nham_theo_o_chi_dem_clip_trong_o():
    tham_chieu = {"a": [sk("siren", 0.0, 1.0)], "b": [sk("gunshot", 0.0, 1.0)]}
    du_bao = {"a": [sk("alarm_bell", 0.0, 1.0)], "b": [sk("fireworks", 0.0, 1.0)]}
    lop = ["siren", "alarm_bell", "gunshot", "fireworks"]
    cap = ti_le_cap_nham(tham_chieu, du_bao, ["a"], lop)
    assert [(c["ref_lop"], c["pred_lop"], c["n"]) for c in cap] == [("siren", "alarm_bell", 1)]


def test_ti_le_cap_nham_sap_giam_dan_theo_so_luot():
    tham_chieu = {"a": [sk("siren", 0.0, 1.0)], "b": [sk("siren", 0.0, 1.0)],
                  "c": [sk("gunshot", 0.0, 1.0)]}
    du_bao = {"a": [sk("alarm_bell", 0.0, 1.0)], "b": [sk("alarm_bell", 0.0, 1.0)],
              "c": [sk("fireworks", 0.0, 1.0)]}
    lop = ["siren", "alarm_bell", "gunshot", "fireworks"]
    cap = ti_le_cap_nham(tham_chieu, du_bao, ["a", "b", "c"], lop)
    assert [c["n"] for c in cap] == [2, 1]
    assert cap[0]["ref_lop"] == "siren"


def test_ti_le_cap_nham_bo_duong_cheo_va_o_rong():
    """Đường chéo của `ma_tran_nham` gồm CẢ `dung` LẪN `bien` — giữ nó lại sẽ làm cặp
    'nhầm' đậm nhất luôn là cặp lớp-với-chính-nó, che mất cặp nhầm thật."""
    tham_chieu = {"a": [sk("siren", 0.0, 1.0)]}
    du_bao = {"a": [sk("siren", 0.0, 1.0)]}
    assert ti_le_cap_nham(tham_chieu, du_bao, ["a"], ["siren", "alarm_bell"]) == []
    assert ti_le_cap_nham({}, {}, [], ["siren"]) == []
