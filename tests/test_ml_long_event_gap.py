"""Test cho ứng viên cuối của `long_event` — khe hở năng lượng thật bên trong sự kiện dài.

Hai giả thuyết trước đã bị đo bác bỏ (cửa sổ lọc hẹp, rồi trần cửa sổ thấp — xem
docs/measurements/long_event_postproc_20260919.md và max_median_ceiling_20260919.md).
Ứng viên cuối: Scaper có sinh khe hở năng lượng THẬT bên trong nhãn hay không. "Có khe
hở" một mình không chứng minh gì — chỉ chênh lệch giữa nhóm sự kiện BỊ vỡ và nhóm KHÔNG
vỡ mới là bằng chứng, nên mọi test ở đây đều dựng cả hai nhóm để so, không đo một nhóm rồi
suy diễn.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.long_event_gap import (  # noqa: E402
    bien_do_rms,
    danh_dau_phan_manh,
    do_khe_ho_long_event,
    khe_ho_dai_nhat_sec,
    tom_tat,
)


def sk(lop: str, on: float, off: float) -> dict:
    return {"event_label": lop, "onset": on, "offset": off}


# ── Đánh dấu phân mảnh ───────────────────────────────────────────────────────


def test_danh_dau_phan_manh_vo_thanh_hai_manh():
    ref = [sk("siren", 0.0, 5.0)]
    pred = [sk("siren", 0.0, 2.0), sk("siren", 3.0, 5.0)]
    assert danh_dau_phan_manh(ref, pred) == [True]


def test_danh_dau_phan_manh_khong_vo_khi_mot_du_bao_khop_dung():
    ref = [sk("siren", 0.0, 5.0)]
    pred = [sk("siren", 0.1, 4.9)]
    assert danh_dau_phan_manh(ref, pred) == [False]


def test_danh_dau_phan_manh_khac_lop_khong_tinh():
    """Hai dự báo chồng lấn nhưng khác lớp không phải phân mảnh — đó là Substitution/
    Insertion, một loại lỗi khác hẳn."""
    ref = [sk("siren", 0.0, 5.0)]
    pred = [sk("siren", 0.0, 2.0), sk("gunshot", 3.0, 3.3)]
    assert danh_dau_phan_manh(ref, pred) == [False]


# ── Đường bao năng lượng ─────────────────────────────────────────────────────


def test_bien_do_rms_phat_hien_doan_im_giua_tin_hieu():
    sr = 1000
    tone = np.ones(300, dtype=np.float32)
    im = np.zeros(300, dtype=np.float32)
    waveform = np.concatenate([tone, im, tone])
    bien_do = bien_do_rms(waveform, sr, onset=0.0, offset=0.9, hop_sec=0.05)
    assert bien_do.max() == pytest.approx(1.0, abs=1e-6)
    assert bien_do.min() == pytest.approx(0.0, abs=1e-6)


def test_khe_ho_dai_nhat_do_dung_do_dai_doan_im():
    hop_sec = 0.05
    bien_do = np.array([1.0] * 4 + [0.0] * 6 + [1.0] * 4)
    khe_ho = khe_ho_dai_nhat_sec(bien_do, hop_sec, nguong_ti_le=0.2)
    assert khe_ho == pytest.approx(6 * hop_sec)


def test_khe_ho_bang_khong_khi_khong_co_doan_im():
    bien_do = np.array([1.0, 0.9, 1.1, 1.0, 0.95])
    assert khe_ho_dai_nhat_sec(bien_do, 0.02, nguong_ti_le=0.2) == 0.0


def test_khe_ho_mang_rong_khong_nem_loi():
    assert khe_ho_dai_nhat_sec(np.array([]), 0.02) == 0.0


# ── Gộp thành phép đo trên một lát cắt ───────────────────────────────────────


def test_do_khe_ho_long_event_bo_qua_su_kien_ngan_hon_nguong(tmp_path):
    """Sự kiện ngắn hơn ngưỡng không thể chứa khe hở nhiều trăm ms có ý nghĩa — cực ngắn
    thì càng chắc chắn không vỡ được. Lẫn nó vào nhóm 'nguyên' sẽ kéo trung vị nhóm đó
    xuống một cách giả tạo do độ dài, không phải vì Scaper hiền lành hơn ở đó."""
    sr = 1000
    waveform = np.concatenate([
        np.ones(2000, dtype=np.float32), np.zeros(1000, dtype=np.float32),
        np.ones(2000, dtype=np.float32),
    ])
    sf.write(str(tmp_path / "a.wav"), waveform, sr)

    tham_chieu = {"a": [sk("siren", 0.0, 5.0), sk("gunshot", 4.9, 5.2)]}
    du_bao = {"a": [sk("siren", 0.0, 2.0), sk("siren", 3.0, 5.0), sk("gunshot", 4.9, 5.2)]}

    ket_qua = do_khe_ho_long_event(tham_chieu, du_bao, tmp_path, ["a"],
                                   hop_sec=0.05, nguong_ti_le=0.2, min_duration_sec=1.0)

    assert len(ket_qua) == 1
    assert ket_qua[0]["lop"] == "siren"
    assert ket_qua[0]["phan_manh"] is True
    assert ket_qua[0]["khe_ho_sec"] > 0.5


def test_do_khe_ho_long_event_clip_khong_co_su_kien_dat_nguong_thi_bo_qua_khong_doc_file(tmp_path):
    """Không có sự kiện nào đạt min_duration_sec thì KHÔNG được mở file audio — clip thiếu
    .wav (dữ liệu tạm thời đang sinh dở) không được làm chết lượt đo vì một clip không
    liên quan gì tới phép so sánh."""
    tham_chieu = {"b": [sk("gunshot", 0.0, 0.3)]}
    du_bao = {"b": [sk("gunshot", 0.0, 0.3)]}

    ket_qua = do_khe_ho_long_event(tham_chieu, du_bao, tmp_path, ["b"],
                                   hop_sec=0.05, nguong_ti_le=0.2, min_duration_sec=1.0)
    assert ket_qua == []


# ── Tóm tắt hai nhóm ─────────────────────────────────────────────────────────


def test_tom_tat_tach_dung_hai_nhom():
    ket_qua = [
        {"phan_manh": True, "khe_ho_sec": 1.0},
        {"phan_manh": True, "khe_ho_sec": 2.0},
        {"phan_manh": False, "khe_ho_sec": 0.0},
        {"phan_manh": False, "khe_ho_sec": 0.1},
    ]
    tt = tom_tat(ket_qua)
    assert tt["n_vo"] == 2
    assert tt["n_nguyen"] == 2
    assert tt["vo_trung_vi"] == pytest.approx(1.5)
    assert tt["nguyen_trung_vi"] == pytest.approx(0.05)


def test_tom_tat_nhom_rong_ra_nan_khong_nem_loi():
    tt = tom_tat([{"phan_manh": False, "khe_ho_sec": 0.2}])
    assert tt["n_vo"] == 0
    assert np.isnan(tt["vo_trung_vi"])
