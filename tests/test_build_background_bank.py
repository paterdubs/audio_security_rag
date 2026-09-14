"""Test cho scripts/build_background_bank.py.

Nền bẩn là lỗi tệ nhất trong cả pipeline, và là lỗi KHÔNG có triệu chứng: một tiếng
chuông lọt vào nền sẽ được Scaper gán nhãn "không có sự kiện", và model học rằng
tiếng chuông nghĩa là bình thường. Sai theo hướng BỎ SÓT — đúng hướng nguy hiểm nhất
cho một hệ giám sát, và không phép đo nào trên tập dev synthetic phát hiện được, vì
tập dev cũng sinh ra từ chính bank bẩn đó.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_background_bank as bg  # noqa: E402


# ── Lấy lớp Nhóm A từ ontology, không liệt kê tay ────────────────────────────


def test_lay_lop_nhom_A_tu_ontology():
    """Liệt kê tay thì thêm một lớp Nhóm A mới sẽ để lọt đúng lớp vừa thêm."""
    ontology = {"classes": {
        "gunshot": {"group": "A", "audioset_ids": ["/m/1", "/m/2"]},
        "speech_normal": {"group": "B", "audioset_ids": ["/m/3"]},
    }}
    found = bg.group_a_indices(ontology, {"/m/1": 10, "/m/2": 11, "/m/3": 12})
    assert found == {"gunshot": [10, 11]}


def test_lop_nhom_B_khong_bi_coi_la_nhiem():
    # speech_normal là nền hợp lệ, không phải chất gây nhiễm.
    ontology = {"classes": {"speech_normal": {"group": "B", "audioset_ids": ["/m/3"]}}}
    assert bg.group_a_indices(ontology, {"/m/3": 12}) == {}


def test_lop_khong_co_ma_nao_khop_thi_bo_qua():
    ontology = {"classes": {"vehicle_crash": {"group": "A", "audioset_ids": ["/m/khong_co"]}}}
    assert bg.group_a_indices(ontology, {"/m/1": 0}) == {}


# ── Đo nhiễm: lấy ĐỈNH trên mọi frame ────────────────────────────────────────


def test_mot_su_kien_ngan_trong_doan_dai_van_bi_bat():
    """Tiếng chuông 0.5s trong đoạn nền 10s chỉ làm TRUNG BÌNH nhích vài phần trăm,
    nhưng vẫn đủ để Scaper gán nhãn sai cho cả clip. Nên phải lấy đỉnh."""
    frames = np.zeros((100, 5))
    frames[50:55, 2] = 0.9                      # 5/100 frame có sự kiện
    name, prob = bg.worst_contamination(frames, {"alarm_bell": [2]})
    assert name == "alarm_bell" and prob == pytest.approx(0.9)


def test_bao_lai_lop_nhiem_NANG_NHAT():
    frames = np.zeros((10, 5))
    frames[0, 1] = 0.3
    frames[0, 3] = 0.7
    name, prob = bg.worst_contamination(frames, {"scream": [1], "gunshot": [3]})
    assert name == "gunshot" and prob == pytest.approx(0.7)


def test_nen_sach_thi_khong_co_lop_nao_bi_neu():
    frames = np.zeros((10, 5))
    assert bg.worst_contamination(frames, {"gunshot": [3]}) == ("", 0.0)


def test_khong_co_frame_nao_khong_lam_no():
    assert bg.worst_contamination(np.array([]), {"gunshot": [0]}) == ("", 0.0)


# ── Ngưỡng 0.15: thà báo nhầm còn hơn lọt ────────────────────────────────────


@pytest.mark.parametrize("prob,expected", [
    (0.00, "clean"),
    (0.14, "clean"),
    (0.15, "clean"),      # đúng ngưỡng thì chưa cờ
    (0.16, "flagged"),
    (0.90, "flagged"),
])
def test_nguong_nhiem(prob: float, expected: str):
    assert bg.verdict_for(prob, threshold=0.15) == expected


def test_nguong_nen_thap_hon_han_nguong_foreground():
    """0.15 cho nền so với 0.30 cho foreground, và sự chênh lệch là CÓ CHỦ Ý.

    Hai bên chịu hai loại rủi ro khác nhau: cờ nhầm một đoạn nền chỉ tốn công nghe
    lại, còn để lọt một sự kiện vào nền thì dạy model sai vĩnh viễn.
    """
    import yaml
    config = yaml.safe_load(bg.CONFIG_PATH.read_text(encoding="utf-8"))
    import auto_screen as asc
    assert config["contamination_threshold"] < asc.ACCEPT_THRESHOLD


# ── Ánh xạ khu vực ───────────────────────────────────────────────────────────


def test_moi_khu_vuc_deu_co_it_nhat_mot_nguon():
    import yaml
    config = yaml.safe_load(bg.CONFIG_PATH.read_text(encoding="utf-8"))
    da_phu = set(config.get("tau2019_partial", {}).values()) | set(config.get("urbansound8k", {}).values())
    thieu = set(config["area_types"]) - da_phu
    assert not thieu, f"khu vực không có nguồn nào: {sorted(thieu)}"


def test_khong_anh_xa_lop_nhom_A_cua_chinh_ta_vao_nen():
    """siren và gun_shot của UrbanSound8K là lớp Nhóm A của ta. Dù salience=2, đưa
    chúng vào nền là tự tay tạo ra đúng thứ bộ sàng lọc sinh ra để chặn."""
    import yaml
    config = yaml.safe_load(bg.CONFIG_PATH.read_text(encoding="utf-8"))
    assert "siren" not in config["urbansound8k"]
    assert "gun_shot" not in config["urbansound8k"]
