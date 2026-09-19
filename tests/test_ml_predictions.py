"""Test cho Pha 3 — lưu dự đoán thô mức đoạn.

Lỗi ở đây đều IM LẶNG: dựng lại mức khung sai một bước thì mọi onset lệch đi vài chục
mili-giây và F1 mức sự kiện tụt mà không ai biết vì sao; lưu thiếu clip rỗng thì điểm
CAO LÊN một cách sai.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.predictions import doc, khung_tu_doan, luu  # noqa: E402

N_FRAMES = 1001


# ── Dựng lại mức khung từ mức đoạn ───────────────────────────────────────────


@pytest.mark.parametrize("time_pool_blocks,n_doan", [(5, 31), (3, 125)])
def test_dung_lai_khung_tu_doan_khop_dau_ra_model(time_pool_blocks, n_doan):
    """Đây là điều khoản mà toàn bộ Pha 3 đứng lên: lưu 31 đoạn thay vì 1001 khung CHỈ
    hợp lệ khi 1001 khung dựng lại được từ 31 đoạn không sai một phần tử nào.

    Nếu `panns_inference` đổi cách nội suy (hay đổi cách đệm khung cuối), test này vỡ —
    còn không có test này thì dữ liệu đã lưu lặng lẽ trôi khỏi dữ liệu thật.
    """
    import torch

    from ml.models.panns_sed import PannsSed

    model = PannsSed(15, checkpoint_path=None, time_pool_blocks=time_pool_blocks).eval()
    torch.manual_seed(0)
    x = torch.randn(1, 320000) * 0.1
    with torch.no_grad():
        output = model(x)

    doan = torch.sigmoid(output["segment_logits"].float())[0].numpy()
    khung_that = torch.sigmoid(output["frame_logits"].float())[0].numpy()

    assert doan.shape[0] == n_doan
    dung_lai = khung_tu_doan(doan, N_FRAMES, ratio=2 ** time_pool_blocks)

    assert dung_lai.shape == khung_that.shape == (N_FRAMES, 15)
    np.testing.assert_array_equal(dung_lai, khung_that)


def test_dem_khung_cuoi_chu_khong_dem_khong():
    """31 × 32 = 992, thiếu 9 khung. `pad_framewise_output` lặp KHUNG CUỐI; đệm 0 sẽ tạo
    ra một hố im lặng 90 ms ở cuối mọi clip và cắt cụt mọi sự kiện chạm mép."""
    doan = np.arange(31, dtype=np.float32)[:, None]
    khung = khung_tu_doan(doan, N_FRAMES, ratio=32)
    assert khung.shape == (1001, 1)
    assert khung[-9:, 0].tolist() == [30.0] * 9


def test_ratio_sai_thi_bao_loi_chu_khong_cat_bot():
    """Ratio không khớp time_pool_blocks là lỗi cấu hình, không phải chuyện cắt cho vừa."""
    with pytest.raises(ValueError, match="ratio không khớp"):
        khung_tu_doan(np.zeros((125, 15), dtype=np.float32), N_FRAMES, ratio=32)


def test_float16_du_min_cho_buoc_quet_nguong():
    """Lưu float16 chỉ hợp lệ nếu sai số lượng tử nhỏ hơn hẳn bước quét ngưỡng (0,01)."""
    rng = np.random.default_rng(0)
    prob = rng.random((200, 15)).astype(np.float32)
    assert np.abs(prob.astype(np.float16).astype(np.float32) - prob).max() < 1e-3


# ── Ghi / đọc ────────────────────────────────────────────────────────────────


def _viet_mau(tmp_path: Path, su_kien) -> Path:
    doan_prob = np.full((len(su_kien), 31, 2), 0.25, dtype=np.float16)
    clip_prob = np.full((len(su_kien), 2), 0.25, dtype=np.float16)
    meta = {"ratio": 32, "n_frames": N_FRAMES, "duration": 10.0}
    return luu(tmp_path / "p.npz", doan_prob, clip_prob,
               [f"c_{i}" for i in range(len(su_kien))], ["alarm_bell", "gunshot"],
               su_kien, meta)


def test_ghi_doc_giu_nguyen_su_kien_tham_chieu(tmp_path):
    path = _viet_mau(tmp_path, [[(0, 1.0, 2.0), (1, 3.0, 3.5)], [(1, 0.0, 9.0)]])
    du_doan = doc(path)

    assert du_doan.clip_ids == ["c_0", "c_1"]
    tham_chieu = du_doan.tham_chieu()
    assert tham_chieu["c_0"] == [
        {"event_label": "alarm_bell", "onset": 1.0, "offset": 2.0},
        {"event_label": "gunshot", "onset": 3.0, "offset": 3.5},
    ]
    assert tham_chieu["c_1"] == [{"event_label": "gunshot", "onset": 0.0, "offset": 9.0}]


def test_clip_khong_su_kien_van_co_mat_o_tham_chieu(tmp_path):
    """sed_eval hiểu file thiếu là 'không đánh giá' chứ không phải 'đoán rỗng' — bỏ clip
    im lặng ra ngoài làm mọi dương tính giả trong đó biến mất khỏi mẫu số."""
    path = _viet_mau(tmp_path, [[], [(0, 1.0, 2.0)], []])
    tham_chieu = doc(path).tham_chieu()

    assert set(tham_chieu) == {"c_0", "c_1", "c_2"}
    assert tham_chieu["c_0"] == []
    assert tham_chieu["c_2"] == []


def test_clip_target_suy_dung_tu_su_kien(tmp_path):
    path = _viet_mau(tmp_path, [[(0, 1.0, 2.0), (0, 5.0, 6.0)], [], [(1, 0.0, 1.0)]])
    assert doc(path).clip_target().tolist() == [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]


def test_khung_dung_lai_dung_so_khung_sau_khi_doc(tmp_path):
    path = _viet_mau(tmp_path, [[(0, 1.0, 2.0)]])
    assert doc(path).khung(0).shape == (N_FRAMES, 2)
