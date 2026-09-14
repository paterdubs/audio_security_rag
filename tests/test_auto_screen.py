"""Test cho scripts/auto_screen.py.

Phần định tuyến và đề xuất biên tách khỏi phần chạy mô hình chính là để test được ở
đây, trên máy không cài torch. Hai loại lỗi cần chặn:
  - Sai định tuyến: clip rõ ràng sai lớp lọt vào hàng đợi (tốn giờ công), hoặc tệ hơn,
    ca khó bị tự động nhận (bank bẩn mà không ai biết).
  - Sai đề xuất biên: người duyệt bấm "nhận, giữ biên đề xuất" hàng loạt, nên biên sai
    đi thẳng vào nhãn mà không qua mắt ai.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import auto_screen as asc  # noqa: E402


# ── Định tuyến — DATA_PLAN §4.2 ──────────────────────────────────────────────


def test_chac_chan_va_vuot_lop_de_nham_thi_tu_dong_nhan():
    assert asc.route(0.80, 0.10).decision == "auto_accept"


def test_dung_nguong_0_30_van_duoc_nhan():
    assert asc.route(0.30, 0.10).decision == "auto_accept"


def test_duoi_nguong_mot_chut_thi_khong_duoc_tu_dong_nhan():
    assert asc.route(0.299, 0.10).decision == "review"


def test_gan_nhu_khong_co_lop_dich_thi_tu_dong_loai():
    assert asc.route(0.01, 0.02).decision == "auto_reject"


def test_dung_nguong_0_05_thi_chua_loai():
    assert asc.route(0.05, 0.02).decision == "review"


def test_lop_de_nham_thang_thi_uu_tien_cao():
    # Chính là những ca quyết định tỉ lệ báo động giả — phải đến tay người trước.
    routing = asc.route(0.20, 0.60)
    assert routing.decision == "review" and routing.priority == "high"


def test_khong_chac_nhung_khong_bi_lop_khac_thang_thi_uu_tien_thuong():
    routing = asc.route(0.20, 0.10)
    assert routing.decision == "review" and routing.priority == "normal"


def test_chac_chan_nhung_lop_de_nham_con_chac_hon_thi_khong_duoc_tu_dong_nhan():
    # p_target 0.55 vượt ngưỡng 0.30 nhưng thua fireworks 0.70. Nhận tự động ở đây là
    # đưa một clip pháo hoa vào bank gunshot — đúng cặp nhầm lẫn cả taxonomy dựng ra để đo.
    routing = asc.route(0.55, 0.70)
    assert routing.decision == "review" and routing.priority == "high"


def test_luat_loai_thang_luat_uu_tien_cao_khi_hai_luat_chong_nhau():
    """p_target=0.02 + p_confusable=0.50 thoả CẢ luật 2 lẫn luật 3.

    Theo đúng thứ tự bảng DATA_PLAN thì luật 2 thắng. Đảo lại sẽ dồn hàng nghìn clip
    rõ ràng sai lớp vào hàng đợi người — đúng thứ việc tự động hoá sinh ra để tránh.
    """
    assert asc.route(0.02, 0.50).decision == "auto_reject"


def test_hang_doi_xep_ca_kho_len_truoc():
    cao = asc.Routing("review", "high", "")
    thuong = asc.Routing("review", "normal", "")
    assert asc.priority_rank(cao) < asc.priority_rank(thuong)


# ── Lọc trung vị ─────────────────────────────────────────────────────────────


def test_loc_trung_vi_xoa_duoc_mot_frame_tut_le_loi():
    # Một frame 0 giữa vùng sự kiện sẽ cắt đôi "vùng liên tục dài nhất" nếu không lọc.
    values = np.array([0.9, 0.9, 0.0, 0.9, 0.9])
    assert asc.median_filter(values, 3)[2] == pytest.approx(0.9)


def test_loc_trung_vi_giu_nguyen_do_dai():
    values = np.random.rand(50)
    assert asc.median_filter(values, 5).size == 50


def test_cua_so_1_thi_khong_doi_gi():
    values = np.array([0.1, 0.9, 0.2])
    assert np.array_equal(asc.median_filter(values, 1), values)


def test_trung_vi_khong_bi_mot_frame_0_keo_ca_vung_xuong():
    # Nếu dùng trung bình thì giá trị tại đây tụt xuống 0.6 và có thể rơi dưới ngưỡng.
    values = np.array([0.9, 0.9, 0.0, 0.9, 0.9])
    assert asc.median_filter(values, 3)[2] > 0.8


# ── Vùng liên tục dài nhất ───────────────────────────────────────────────────


@pytest.mark.parametrize("mask,expected", [
    ([0, 1, 1, 1, 0], (1, 4)),
    ([1, 1, 0, 1, 1, 1], (3, 6)),          # dải sau dài hơn
    ([1, 1, 1, 0, 1], (0, 3)),             # dải trước dài hơn
    ([1, 1, 1], (0, 3)),                   # chạm hết mảng
    ([0, 0, 0], (0, 0)),                   # không có dải nào
])
def test_tim_dung_dai_dai_nhat(mask, expected):
    assert asc.longest_run(np.array(mask, dtype=bool)) == expected


# ── Đề xuất biên — DATA_PLAN §4.3 ────────────────────────────────────────────


def test_bien_bam_vao_vung_xac_suat_cao():
    frames = np.array([0.0] * 20 + [0.9] * 20 + [0.0] * 20)
    onset, offset = asc.propose_boundary(frames, duration=6.0)
    # Vùng sự kiện là giây 2–4, cộng nới ±50 ms.
    assert onset == pytest.approx(1.95, abs=0.06)
    assert offset == pytest.approx(4.05, abs=0.06)


def test_clip_may_khong_chac_van_cho_bien_dung():
    """Ngưỡng là 0.5 × ĐỈNH, không phải hằng số tuyệt đối.

    Ngưỡng tuyệt đối sẽ trả vùng rỗng cho đúng những clip cần người xem nhất — clip
    mà máy chỉ chắc 0.2 nhưng đường cong vẫn chỉ đúng chỗ sự kiện.
    """
    frames = np.array([0.01] * 20 + [0.20] * 20 + [0.01] * 20)
    onset, offset = asc.propose_boundary(frames, duration=6.0)
    assert 1.8 < onset < 2.1 and 3.9 < offset < 4.2


def test_bien_khong_bao_gio_tran_ra_ngoai_clip():
    frames = np.array([0.9] * 30)
    onset, offset = asc.propose_boundary(frames, duration=3.0)
    assert onset == 0.0 and offset == 3.0


def test_clip_im_lang_thi_tra_ve_ca_clip():
    # Không bịa ra một vùng nào — trả cả clip để người tự quyết.
    assert asc.propose_boundary(np.zeros(30), duration=3.0) == (0.0, 3.0)


def test_khong_co_frame_nao_thi_khong_no():
    assert asc.propose_boundary(np.array([]), duration=3.0) == (0.0, 3.0)


def test_bien_giu_nguyen_thu_tu_onset_truoc_offset():
    for _ in range(50):
        frames = np.random.rand(60)
        onset, offset = asc.propose_boundary(frames, duration=6.0)
        assert onset < offset


# ── Chấm điểm ────────────────────────────────────────────────────────────────


def test_lay_xac_suat_LON_NHAT_trong_cac_ma_cua_lop():
    # Một lớp của ta gộp nhiều mã AudioSet; lấy max chứ không phải trung bình, vì
    # trung bình sẽ dìm clip khớp hoàn hảo với MỘT mã trong nhóm.
    probs = np.array([0.1, 0.9, 0.3, 0.2])
    p_target, _, _ = asc.score_clip(probs, target=[0, 1], confusable={})
    assert p_target == pytest.approx(0.9)


def test_neu_ten_lop_de_nham_manh_nhat_duoc_ghi_lai():
    probs = np.array([0.2, 0.3, 0.8])
    _, p_conf, name = asc.score_clip(probs, target=[0], confusable={"fireworks": [1], "explosion": [2]})
    assert name == "explosion" and p_conf == pytest.approx(0.8)


def test_lop_khong_co_ma_nao_khop_thi_ve_0_chu_khong_no():
    p_target, p_conf, name = asc.score_clip(np.array([0.5]), target=[], confusable={})
    assert (p_target, p_conf, name) == (0.0, 0.0, "")
