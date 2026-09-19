"""Test cho hiệu chuẩn xác suất — vì sao cả ba run panns_ft_v1/v2/v3 đỉnh F1 ở θ ≈ 0,85–0,95.

Chưa ai đo hiệu chuẩn trong đề tài này, chỉ mới quan sát triệu chứng (θ* luôn cao). ECE
tự viết chứ không có thư viện tham chiếu như `sed_eval` cho F1, nên mọi phép tính đều
phải kiểm được bằng tay: xây trường hợp cân bằng hoàn hảo (ECE=0) và trường hợp quá tự
tin đã biết trước diff (ECE=diff), không dùng dữ liệu ngẫu nhiên rồi đoán khoảng.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.calibration import bang_hieu_chuan, chia_bin, do_tin_cay, ece  # noqa: E402
from ml.evaluation.predictions import DuDoan  # noqa: E402


def _du_doan_hieu_chuan_gia() -> DuDoan:
    """Một clip, một đoạn (T=1), ratio=n_frames → `khung()` lặp đúng một giá trị hằng số
    ra toàn bộ 10 khung, không đụng nhánh đệm của `khung_tu_doan`. Ba lớp, ba mức hiệu
    chuẩn biết trước: 'a' dự báo 0,6 nhưng KHÔNG có sự kiện (thật=0) → lệch 0,6; 'b' dự
    báo 0,6 và có sự kiện phủ hết clip (thật=1) → lệch 0,4; 'c' dự báo 0,5 và có sự kiện
    phủ hết clip → lệch 0,5."""
    n_frames = 10
    class_ids = ["a", "b", "c"]
    doan_prob = np.array([[[0.6, 0.6, 0.5]]], dtype=np.float16)
    return DuDoan(
        doan_prob=doan_prob, clip_prob=np.zeros((1, 3), dtype=np.float16),
        clip_ids=["x"], class_ids=class_ids,
        ref_clip=np.array([0, 0]), ref_class=np.array([1, 2]),
        ref_onset=np.array([0.0, 0.0]), ref_offset=np.array([1.0, 1.0]),
        meta={"n_frames": n_frames, "duration": 1.0, "ratio": n_frames},
    )


# ── Bin và ECE ───────────────────────────────────────────────────────────────


def test_chia_bin_co_dung_so_canh_va_hai_dau_mut():
    canh = chia_bin(10)
    assert len(canh) == 11
    assert canh[0] == 0.0
    assert canh[-1] == 1.0


def test_ece_bang_khong_khi_hieu_chuan_hoan_hao():
    """3/10 mẫu dương, dự báo hằng số 0,3 — trung bình dự báo khớp đúng tỉ lệ dương thật."""
    xac_suat = np.full(10, 0.3)
    nhan = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    assert ece(xac_suat, nhan, n_bins=1) == pytest.approx(0.0, abs=1e-9)


def test_ece_bang_dung_do_lech_khi_qua_tu_tin():
    """Dự báo hằng số 0,99 nhưng chỉ 20% mẫu thật dương — ECE phải bằng ĐÚNG |0,99-0,2|,
    không phải một số 'lớn' mơ hồ. Đây là chữ ký của mô hình quá tự tin."""
    xac_suat = np.full(10, 0.99)
    nhan = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    assert ece(xac_suat, nhan, n_bins=1) == pytest.approx(0.79, abs=1e-9)


def test_do_tin_cay_tong_so_luong_bang_tong_mau():
    """Mỗi mẫu phải rơi vào đúng một bin — tổng so_luong lệch tổng mẫu nghĩa là có mẫu bị
    đếm hai lần hoặc rơi ra ngoài do lỗi làm tròn biên bin."""
    xac_suat = np.array([0.0, 0.05, 0.15, 0.55, 0.95, 1.0])
    nhan = np.array([0, 0, 1, 0, 1, 1], dtype=np.float32)
    hang = do_tin_cay(xac_suat, nhan, n_bins=10)
    assert len(hang) == 10
    assert sum(h["so_luong"] for h in hang) == len(xac_suat)


def test_do_tin_cay_bin_rong_khong_gay_loi_chi_ra_nan():
    hang = do_tin_cay(np.array([0.05]), np.array([0.0]), n_bins=10)
    trong = [h for h in hang if h["so_luong"] == 0]
    assert len(trong) == 9
    assert all(np.isnan(h["trung_binh_du_bao"]) for h in trong)


# ── Bảng theo lớp ────────────────────────────────────────────────────────────


def test_bang_hieu_chuan_sap_xep_ece_giam_dan():
    """Lớp lệch hiệu chuẩn nhiều nhất phải đứng đầu — người đọc báo cáo lướt từ trên
    xuống, xếp sai thứ tự thì lớp đáng lo nhất bị chìm dưới bảng.

    Dung sai 2e-3: `doan_prob` là float16 (xem docstring `predictions.DuDoan`, sai số
    lượng tử ~5e-4) — so bằng tuyệt đối thì test vỡ vì lượng tử hoá, không phải vì logic
    sai."""
    du_doan = _du_doan_hieu_chuan_gia()
    bang = bang_hieu_chuan(du_doan, n_bins=1)

    thu_tu = [h["class_id"] for h in bang["theo_lop"]]
    assert thu_tu == ["a", "c", "b"]
    diem = {h["class_id"]: h["ece"] for h in bang["theo_lop"]}
    assert diem["a"] == pytest.approx(0.6, abs=2e-3)
    assert diem["c"] == pytest.approx(0.5, abs=2e-3)
    assert diem["b"] == pytest.approx(0.4, abs=2e-3)
