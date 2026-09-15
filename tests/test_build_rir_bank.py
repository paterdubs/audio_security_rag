"""Test cho scripts/build_rir_bank.py.

Lỗi trong bank RIR không làm gì hỏng lộ liễu — clip vẫn phát được, vẫn "nghe như có
hiệu ứng gì đó". Nhưng nó làm lát cắt `reverb` mô tả sai thứ thực sự đã sinh ra, và
lát cắt đó là một trong các con số báo cáo theo slice của khoá luận.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_rir_bank as brb  # noqa: E402


def _rir_gia_lap(rt60: float, sample_rate: int = 16000, pre_delay_ms: float = 0.0) -> np.ndarray:
    """Nhiễu trắng tắt dần theo hàm mũ — mô hình RIR đơn giản nhất có RT60 biết trước."""
    rng = np.random.default_rng(0)
    n = int(sample_rate * rt60 * 1.5)
    t = np.arange(n) / sample_rate
    decay = 10 ** (-3 * t / rt60)            # −60 dB đúng tại t = rt60
    rir = rng.normal(0, 1, n) * decay
    rir[0] = 1.0                             # xung trực tiếp
    if pre_delay_ms:
        rir = np.concatenate([np.zeros(int(pre_delay_ms * sample_rate / 1000)), rir])
    return rir.astype(np.float32)


# ── Đo RT60 ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("rt60", [0.3, 0.8, 2.0])
def test_do_lai_dung_rt60_da_dung_len(rt60: float):
    do_duoc = brb.rt60_from_rir(_rir_gia_lap(rt60), 16000)
    assert abs(do_duoc - rt60) / rt60 < 0.25


def test_rir_rong_khong_lam_no():
    assert brb.rt60_from_rir(np.zeros(1000), 16000) == 0.0


# ── Xếp nhóm theo RT60 đo được ───────────────────────────────────────────────


@pytest.mark.parametrize("rt60,expected", [
    (0.05, None),            # buồng tiêu âm — KHÔNG phải RIR có vang
    (0.14, None),
    (0.20, "small_room"),    # phòng học, văn phòng
    (0.49, "small_room"),
    (0.50, "medium_room"),   # hành lang, sảnh
    (1.19, "medium_room"),
    (1.20, "large_room"),    # nhà xe, xưởng
    (3.00, "large_room"),
])
def test_xep_nhom_theo_nguong(rt60: float, expected: str | None):
    assert brb.bin_by_rt60(rt60) == expected


def test_loai_phong_cam():
    """RWCP có 36 file đo trong buồng tiêu âm. Đưa vào bank reverb thì lát cắt
    "có vang" lại chứa clip không vang, và tỉ lệ 40% trong báo cáo mô tả sai."""
    assert brb.bin_by_rt60(0.0) is None


# ── Cắt RIR ──────────────────────────────────────────────────────────────────


def test_cat_bo_do_tre_lan_truyen():
    """Giữ độ trễ lại thì tích chập đẩy TOÀN BỘ sự kiện lùi đi, trong khi .jams vẫn
    ghi mốc cũ — audio và nhãn lệch nhau một cách âm thầm."""
    rir = _rir_gia_lap(0.5, pre_delay_ms=50.0)
    cat = brb.trim_rir(rir, 16000)
    # Xung trực tiếp phải nằm gần đầu, chỉ còn phần đệm PRE_DIRECT_MS.
    assert int(np.argmax(np.abs(cat))) <= int(brb.PRE_DIRECT_MS * 16 + 1)


def test_cat_duoi_qua_dai():
    rir = _rir_gia_lap(6.0)
    assert len(brb.trim_rir(rir, 16000)) <= int(brb.MAX_RIR_SEC * 16000)


def test_rir_ngan_khong_bi_cat_mat():
    rir = _rir_gia_lap(0.2)
    assert len(brb.trim_rir(rir, 16000)) > 0


# ── Chuẩn hoá ────────────────────────────────────────────────────────────────


def test_chuan_hoa_theo_nang_luong_khong_theo_dinh():
    """Chuẩn theo đỉnh thì RIR vang dài làm clip to vọt, RIR khô làm clip nhỏ đi —
    và SNR mà Scaper vừa đặt sẽ sai lệch THEO ĐỘ VANG, tức sai có hệ thống."""
    for rt60 in (0.2, 2.0):
        chuan = brb.normalize_rir(brb.trim_rir(_rir_gia_lap(rt60), 16000))
        assert np.sqrt(np.sum(chuan.astype(np.float64) ** 2)) == pytest.approx(1.0, abs=1e-5)


def test_rir_toan_khong_khong_chia_cho_0():
    assert np.array_equal(brb.normalize_rir(np.zeros(100)), np.zeros(100))
