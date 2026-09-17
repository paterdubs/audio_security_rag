"""Test phần suy luận KHÔNG cần torch — ánh xạ nhãn, đệm clip, đề xuất biên.

Ba thứ được test ở đây đều đã sai một lần trên dữ liệu thật ở scripts/auto_screen.py.
Tách chúng khỏi phần gọi model chính là để test được mà không phải nạp 300 MB checkpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.tagger import (  # noqa: E402
    MIN_SAMPLES,
    class_to_indices,
    detections_from_frames,
    longest_run,
    mid_to_index,
    pad_to_minimum,
    propose_boundary,
)


# ── Đệm clip ngắn ────────────────────────────────────────────────────────────


def test_clip_ngan_duoc_lap_lai_khong_dem_im_lang():
    """Đo thật trên 40 clip <0.6s: đệm im lặng → 37/40 bị loại oan; lặp lại → 2/40.
    CNN14 gộp cả trung bình lẫn cực đại theo thời gian, nên im lặng dìm xác suất xuống."""
    audio = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32)
    padded = pad_to_minimum(audio)

    assert len(padded) == MIN_SAMPLES
    assert not np.any(padded == 0.0)          # không có mẫu im lặng nào
    assert np.allclose(padded[:4], audio)     # bắt đầu vẫn là chính clip gốc


def test_clip_du_dai_khong_bi_dong_vao():
    audio = np.ones(MIN_SAMPLES + 100, dtype=np.float32)
    assert len(pad_to_minimum(audio)) == MIN_SAMPLES + 100


def test_clip_rong_khong_lam_no():
    assert len(pad_to_minimum(np.array([], dtype=np.float32))) == 0


# ── Ánh xạ nhãn theo MÃ AudioSet ─────────────────────────────────────────────


def test_ghep_theo_ma_khong_ghep_theo_ten(tmp_path):
    """Tên hiển thị có dấu phẩy, ngoặc, và trùng nhau giữa các nhánh ontology — ghép
    theo tên là chỗ dễ sai nhất."""
    ontology = tmp_path / "ontology.json"
    ontology.write_text(
        json.dumps([
            {"id": "/m/032s66", "name": "Gunshot, gunfire"},
            {"id": "/m/03qc9zr", "name": "Screaming"},
        ]),
        encoding="utf-8",
    )
    mapping = mid_to_index(["Speech", "Gunshot, gunfire", "Screaming"], ontology)
    assert mapping == {"/m/032s66": 1, "/m/03qc9zr": 2}


def test_lop_khong_co_ma_nao_khop_thi_bi_bo_qua():
    taxonomy = {"classes": {
        "gunshot": {"audioset_ids": ["/m/032s66"]},
        "vehicle_crash": {"audioset_ids": ["/m/khong_co_trong_panns"]},
    }}
    result = class_to_indices(taxonomy, {"/m/032s66": 1})
    assert "gunshot" in result and "vehicle_crash" not in result


# ── Đề xuất biên ─────────────────────────────────────────────────────────────


def test_bien_bam_theo_vung_xac_suat_cao():
    probs = np.array([0.01, 0.02, 0.9, 0.95, 0.88, 0.03, 0.01])
    onset, offset = propose_boundary(probs, duration=7.0)
    assert 1.5 < onset < 2.5
    assert 4.5 < offset < 5.5


def test_nguong_theo_dinh_chu_khong_phai_hang_so():
    """Clip máy chỉ chắc 0.2 vẫn phải có biên đúng — ngưỡng tuyệt đối sẽ trả vùng rỗng
    cho đúng những clip cần người xem nhất."""
    probs = np.array([0.01, 0.01, 0.18, 0.20, 0.17, 0.01])
    onset, offset = propose_boundary(probs, duration=6.0)
    assert offset > onset


def test_khong_co_tin_hieu_thi_tra_ca_clip():
    onset, offset = propose_boundary(np.zeros(10), duration=5.0)
    assert (onset, offset) == (0.0, 5.0)


def test_lay_doan_dai_nhat_khong_lay_moi_doan():
    # Một tiếng lách tách lẻ không được kéo biên sự kiện ra toàn clip.
    mask = np.array([True, False, False, True, True, True, False])
    assert longest_run(mask) == (3, 6)


# ── Ghép detection từ xác suất frame ────────────────────────────────────────


def _frames(n_frames: int = 10, n_classes: int = 5) -> np.ndarray:
    return np.zeros((n_frames, n_classes))


def test_duoi_nguong_thi_khong_thanh_detection():
    framewise = _frames()
    framewise[3:6, 0] = 0.05      # dưới DETECTION_THRESHOLD
    result = detections_from_frames(framewise, {"gunshot": [0]}, duration=10.0)
    assert result == []


def test_nhieu_ma_cua_cung_mot_lop_lay_max_khong_cong_don():
    """`siren` gồm 5 mã AudioSet. Nghe thấy còi cứu thương hay cứu hoả đều là siren —
    cộng dồn sẽ cho ra xác suất > 1, thứ vô nghĩa."""
    framewise = _frames()
    framewise[2:5, 0] = 0.6
    framewise[2:5, 1] = 0.7
    result = detections_from_frames(framewise, {"siren": [0, 1]}, duration=10.0)
    assert len(result) == 1
    assert result[0].confidence == 0.7


def test_nhieu_lop_duoc_sap_theo_onset():
    framewise = _frames()
    framewise[1:3, 0] = 0.9      # lớp A sớm hơn
    framewise[6:9, 1] = 0.8      # lớp B muộn hơn
    result = detections_from_frames(framewise, {"scream": [1], "glass_breaking": [0]}, duration=10.0)
    assert [d.class_id for d in result] == ["glass_breaking", "scream"]
