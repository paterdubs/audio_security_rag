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


# ── Đệm clip ngắn ────────────────────────────────────────────────────────────
#
# 1076/5260 clip của bank dưới 1 giây, và CNN14 SẬP với chúng. Nhưng cách đệm mới là
# chỗ nguy hiểm: đệm im lặng chạy trơn tru và cho ra số, chỉ là số sai.


def test_clip_ngan_duoc_keo_dai_du_cho_mo_hinh():
    ngan = np.ones(3200)                       # 0.1 giây
    assert len(asc.pad_to_minimum(ngan)) == asc.MIN_SAMPLES


def test_keo_dai_bang_LAP_LAI_chu_khong_phai_im_lang():
    """Đo trên 40 clip ngắn thật: đệm im lặng cho p_target trung bình 0.012 và làm
    37/40 clip bị tự động loại; lặp lại cho 0.400 và chỉ loại 2.

    Tức đệm im lặng sẽ xoá ~1000 clip khỏi bank với lý do "sai lớp" trong khi chúng
    không hề sai lớp — và mất mát tập trung đúng vào các lớp xung kích.
    """
    ngan = np.array([0.5, -0.5] * 1600)        # 0.1 giây, không có mẫu 0 nào
    keo_dai = asc.pad_to_minimum(ngan)
    assert not np.any(keo_dai == 0), "có mẫu 0 nghĩa là đang đệm im lặng"
    assert np.array_equal(keo_dai[:3200], ngan)


def test_clip_du_dai_thi_giu_nguyen():
    dai = np.ones(asc.MIN_SAMPLES * 2)
    assert np.array_equal(asc.pad_to_minimum(dai), dai)


def test_clip_dung_1_giay_khong_bi_dong_vao():
    vua = np.ones(asc.MIN_SAMPLES)
    assert np.array_equal(asc.pad_to_minimum(vua), vua)


def test_clip_rong_khong_lam_no_vong_lap():
    assert len(asc.pad_to_minimum(np.array([]))) == 0


def test_chi_doc_phan_audio_that_khi_de_xuat_bien():
    """Đọc cả phần lặp sẽ cho ra biên trỏ vào bản sao thứ ba của sự kiện — một khoảng
    thời gian KHÔNG TỒN TẠI trong file gốc, mà người duyệt bấm 'nhận' là nó vào nhãn."""
    # Clip 0.25 s được lặp 4 lần thành 1 s → chỉ 1/4 số frame là thật.
    assert asc.real_frame_count(frames=100, original_samples=asc.MIN_SAMPLES // 4) == 25


def test_clip_du_dai_thi_doc_het_frame():
    assert asc.real_frame_count(frames=100, original_samples=asc.MIN_SAMPLES * 3) == 100


def test_luon_con_it_nhat_mot_frame_de_doc():
    assert asc.real_frame_count(frames=10, original_samples=1) >= 1


# ── Guard: lớp mà tagger không đủ phân giải ──────────────────────────────────
#
# Đo thật trên bank 5260 clip: shout_yell bị loại 93%, object_drop_dishes 87%. Chẩn
# đoán cho thấy PANNs nghe ra "Speech"/"Groan"/"Gasp" ở shout_yell và
# "Chink, clink"/"Coin dropping"/"Glass" ở object_drop_dishes — tức nghe ĐÚNG nội dung
# âm học, chỉ gán vào lớp lân cận. Mà các clip ấy đã qua cổng PP của FSD50K, nghĩa là
# CON NGƯỜI đã chấm nhãn là có mặt và nổi trội.


def _diem(class_id: str, decision: str, n: int) -> list[dict]:
    return [{"class_id": class_id, "decision": decision, "priority": "", "file_id": f"{class_id}_{decision}_{i}"}
            for i in range(n)]


def test_lop_bi_loai_qua_nua_thi_huy_moi_quyet_dinh_loai():
    scored = _diem("shout_yell", "auto_reject", 93) + _diem("shout_yell", "review", 7)
    cuu = asc.guard_low_resolution_classes(scored)
    assert cuu == {"shout_yell": 93}
    assert all(r["decision"] == "review" for r in scored)


def test_lop_binh_thuong_khong_bi_dong_vao():
    scored = _diem("siren", "auto_reject", 7) + _diem("siren", "auto_accept", 293)
    assert asc.guard_low_resolution_classes(scored) == {}
    assert sum(1 for r in scored if r["decision"] == "auto_reject") == 7


def test_dung_muc_mot_nua_thi_chua_kich_hoat():
    # Nửa là ranh giới tự nhiên chứ không phải số tinh chỉnh; phải VƯỢT nửa.
    scored = _diem("door_slam", "auto_reject", 50) + _diem("door_slam", "auto_accept", 50)
    assert asc.guard_low_resolution_classes(scored) == {}


def test_moi_lop_duoc_xet_rieng():
    scored = (_diem("shout_yell", "auto_reject", 93) + _diem("shout_yell", "review", 7)
              + _diem("siren", "auto_reject", 7) + _diem("siren", "auto_accept", 293))
    cuu = asc.guard_low_resolution_classes(scored)
    assert set(cuu) == {"shout_yell"}
    assert sum(1 for r in scored if r["decision"] == "auto_reject") == 7    # siren giữ nguyên


def test_clip_duoc_cuu_vao_hang_doi_uu_tien_THUONG():
    # Không phải ưu tiên cao: chúng không khó vì lẫn lớp, chúng chỉ nằm ngoài tầm
    # phân giải của tagger. Đẩy lên ưu tiên cao sẽ dìm mất những ca thật sự khó.
    bi_loai = _diem("shout_yell", "auto_reject", 93)
    scored = bi_loai + _diem("shout_yell", "review", 7)
    asc.guard_low_resolution_classes(scored)
    assert all(r["priority"] == "normal" for r in bi_loai)


def test_lop_khong_co_clip_nao_khong_lam_no():
    assert asc.guard_low_resolution_classes([]) == {}


# ── Gộp điểm mới vào screen_scores.csv, không ghi đè cả file ────────────────


def test_chay_lop_nay_khong_xoa_lop_khac():
    """Đúng lỗi thật: chạy --class-id applause_cheering sau --class-id laughter đã
    xoá sạch điểm của laughter khỏi screen_scores.csv vì write_rows ghi đè cả file."""
    existing = [{"file_id": "a", "class_id": "laughter"}, {"file_id": "b", "class_id": "siren"}]
    fresh = [{"file_id": "c", "class_id": "applause_cheering"}]
    ket_qua = asc.merge_scored(existing, fresh, rescanned_classes={"applause_cheering"})
    assert {r["file_id"] for r in ket_qua} == {"a", "b", "c"}


def test_quet_lai_cung_lop_thi_thay_diem_cu():
    existing = [{"file_id": "a", "class_id": "laughter", "p_target": 0.1}]
    fresh = [{"file_id": "a", "class_id": "laughter", "p_target": 0.9}]
    ket_qua = asc.merge_scored(existing, fresh, rescanned_classes={"laughter"})
    assert ket_qua == [{"file_id": "a", "class_id": "laughter", "p_target": 0.9}]


