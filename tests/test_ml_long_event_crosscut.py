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

from ml.evaluation.long_event_crosscut import ti_le_phan_manh  # noqa: E402


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
