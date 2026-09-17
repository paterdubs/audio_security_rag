"""Test cho Mixup và hậu xử lý thích ứng.

Cả hai đều là loại thay đổi KHÔNG bao giờ báo lỗi khi làm sai: mixup trộn nhãn lệch pha
với sóng âm, hay cửa sổ lọc rộng hơn chính sự kiện — đều chạy trơn tru và chỉ làm điểm
số tệ đi một cách khó truy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.sed_metrics import (  # noqa: E402
    MAX_MEDIAN_FRAMES, MIN_MEDIAN_FRAMES, adaptive_median_sizes, class_event_durations,
    frames_to_events, median_filter,
)
from ml.training.mixup import LABEL_HARD, LABEL_SOFT, mix_batch  # noqa: E402


# ── Mixup ────────────────────────────────────────────────────────────────────


def _batch(n=4, samples=100, frames=10, classes=3):
    torch.manual_seed(0)
    return (torch.randn(n, samples),
            torch.zeros(n, frames, classes),
            torch.zeros(n, classes))


def test_alpha_bang_khong_thi_khong_doi_gi():
    """Tắt mixup phải là đường đi KHÔNG chạm vào dữ liệu, để bật/tắt so được sòng phẳng."""
    waveform, frame_target, clip_target = _batch()
    a, b, c = mix_batch(waveform, frame_target, clip_target, alpha=0.0)
    assert torch.equal(a, waveform) and torch.equal(b, frame_target) and torch.equal(c, clip_target)


def test_giu_nguyen_hinh_dang():
    waveform, frame_target, clip_target = _batch()
    a, b, c = mix_batch(waveform, frame_target, clip_target, alpha=0.2)
    assert a.shape == waveform.shape
    assert b.shape == frame_target.shape
    assert c.shape == clip_target.shape


def test_nhan_hard_la_hop_cua_hai_tap_nhan():
    """Trộn hai tín hiệu thì CẢ HAI sự kiện đều nghe được, nên nhãn hợp lý là hợp của
    hai tập — và với lớp hiếm, cách này giữ nguyên cường độ tín hiệu dương."""
    waveform = torch.zeros(2, 10)
    clip_target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    frame_target = torch.zeros(2, 4, 3)
    frame_target[0, :, 0] = 1.0
    frame_target[1, :, 1] = 1.0

    _, mixed_frame, mixed_clip = mix_batch(
        waveform, frame_target, clip_target, alpha=1.0, label_mode=LABEL_HARD
    )
    # Dù hoán vị ra sao, nhãn hard chỉ nhận giá trị 0 hoặc 1 — không có giá trị trung gian.
    assert set(np.unique(mixed_clip.numpy())).issubset({0.0, 1.0})
    assert set(np.unique(mixed_frame.numpy())).issubset({0.0, 1.0})
    # Và không được làm MẤT nhãn nào đang bật.
    assert mixed_clip.sum() >= clip_target.sum()


def test_nhan_soft_nam_giua_hai_nhan_goc():
    waveform = torch.zeros(2, 10)
    clip_target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    frame_target = torch.zeros(2, 4, 2)

    _, _, mixed_clip = mix_batch(
        waveform, frame_target, clip_target, alpha=1.0, label_mode=LABEL_SOFT
    )
    assert ((mixed_clip >= 0.0) & (mixed_clip <= 1.0)).all()
    # Tổng nhãn mềm được bảo toàn: λ·1 + (1−λ)·1 = 1 cho mỗi mẫu.
    assert mixed_clip.sum(dim=1).allclose(torch.ones(2), atol=1e-5)


def test_song_am_va_nhan_dung_CUNG_mot_hoan_vi():
    """Lỗi kinh điển: xáo sóng âm bằng một hoán vị, xáo nhãn bằng hoán vị khác. Model vẫn
    huấn luyện bình thường, chỉ là học trên nhãn của clip khác."""
    waveform = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    clip_target = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    frame_target = torch.zeros(4, 2, 1)

    mixed_waveform, _, mixed_clip = mix_batch(
        waveform, frame_target, clip_target, alpha=1.0, label_mode=LABEL_SOFT
    )
    # Sóng âm và nhãn clip mang cùng giá trị nên nếu hoán vị khớp, hai kết quả trùng khít.
    assert torch.allclose(mixed_waveform, mixed_clip, atol=1e-5)


# ── Hậu xử lý thích ứng ──────────────────────────────────────────────────────


def test_cua_so_theo_do_dai_lop():
    """Tiếng súng ngắn phải được cửa sổ hẹp, tiếng còi dài được cửa sổ rộng."""
    sizes = adaptive_median_sizes([0.08, 3.0], frames_per_second=100.0)
    assert sizes[0] < sizes[1]


def test_cua_so_luon_le():
    """Cửa sổ chẵn làm kết quả lệch nửa khung về một phía, và lệch pha giữa các lớp là
    thứ không ai nhìn ra khi đọc điểm số."""
    sizes = adaptive_median_sizes([0.04, 0.10, 0.20, 1.0, 9.0], frames_per_second=100.0)
    assert all(int(s) % 2 == 1 for s in sizes)


def test_cua_so_bi_chan_tren_duoi():
    sizes = adaptive_median_sizes([0.0001, 1000.0], frames_per_second=100.0)
    assert sizes[0] >= MIN_MEDIAN_FRAMES
    assert sizes[1] <= MAX_MEDIAN_FRAMES + 1     # +1 vì bước ép lẻ


def test_lop_khong_co_du_lieu_roi_ve_mac_dinh():
    """NaN (lớp chưa từng xuất hiện) không được biến thành cửa sổ 0 hay âm."""
    sizes = adaptive_median_sizes([np.nan, 0.5], frames_per_second=100.0)
    assert sizes[0] >= MIN_MEDIAN_FRAMES


def test_phan_vi_do_dai_dung_lop():
    events_per_clip = [
        [(0, 0.0, 0.1), (1, 0.0, 2.0)],
        [(0, 0.0, 0.2), (1, 0.0, 3.0)],
    ]
    durations = class_event_durations(events_per_clip, n_classes=3, percentile=50)
    assert durations[0] == pytest.approx(0.15, abs=1e-6)
    assert durations[1] == pytest.approx(2.5, abs=1e-6)
    assert np.isnan(durations[2])            # lớp 2 không xuất hiện


def test_cua_so_rong_xoa_mat_su_kien_ngan_con_cua_so_hep_thi_khong():
    """Đây chính là lý do tồn tại của cả tính năng này: cùng một tín hiệu tiếng súng
    (8 khung ≈ 80 ms), cửa sổ chung 51 khung xoá sạch nó, cửa sổ hẹp thì giữ được."""
    probabilities = np.zeros((200, 1), dtype=np.float32)
    probabilities[100:108, 0] = 0.9

    assert len(frames_to_events(probabilities, ["gunshot"], 2.0, median_size=51)) == 0
    assert len(frames_to_events(probabilities, ["gunshot"], 2.0, median_size=3)) == 1


def test_cua_so_theo_lop_ap_dung_dung_cot():
    """Cửa sổ của lớp A không được ảnh hưởng lớp B."""
    probabilities = np.zeros((200, 2), dtype=np.float32)
    probabilities[100:108, 0] = 0.9          # sự kiện ngắn ở lớp 0
    probabilities[100:108, 1] = 0.9          # sự kiện ngắn y hệt ở lớp 1

    smoothed = median_filter(probabilities, [3, 51])
    assert smoothed[100:108, 0].max() > 0.5, "cửa sổ hẹp phải giữ được sự kiện"
    assert smoothed[100:108, 1].max() < 0.5, "cửa sổ rộng phải xoá sự kiện ngắn"


def test_loc_mot_so_va_loc_theo_mang_cho_ket_qua_giong_nhau_khi_cung_cua_so():
    """Đường đi 'một số' và đường đi 'mảng' phải tương đương — nếu lệch thì mọi so sánh
    cố định/thích ứng đều nhiễu bởi chính sự khác biệt cài đặt."""
    rng = np.random.default_rng(0)
    probabilities = rng.random((120, 4)).astype(np.float32)
    assert np.allclose(median_filter(probabilities, 7), median_filter(probabilities, [7] * 4))
