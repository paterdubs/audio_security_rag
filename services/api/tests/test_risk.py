"""Test risk scoring — SYSTEM.md §6.4.

Đây là chỗ quyết định False Alarm Rate của cả hệ thống. Một lỗi ở đây không làm gì hỏng
lộ liễu: hệ thống vẫn chạy, vẫn sinh cảnh báo, chỉ là sinh SAI — và chỉ lộ ra khi người
vận hành đã mất niềm tin vì bị đánh thức lúc 2 giờ sáng vì một tràng pháo.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.risk import (  # noqa: E402
    CONFIDENCE_FLOOR,
    Detection,
    score_event,
    severity_for,
)

TIERS = {
    "gunshot": "Critical",
    "scream": "Critical",
    "shout_yell": "High",
    "siren": "Medium",
    "fireworks": "Negative",
    "applause_cheering": "Negative",
}


# ── Ánh xạ ngưỡng severity ───────────────────────────────────────────────────


def test_bien_nguong_dung_theo_spec():
    # LOW < 0.35 ≤ MEDIUM < 0.60 ≤ HIGH < 0.80 ≤ CRITICAL — kiểm đúng tại biên,
    # vì lệch một nấc ở đây đổi hẳn hành vi cảnh báo.
    assert severity_for(0.349) == "LOW"
    assert severity_for(0.35) == "MEDIUM"
    assert severity_for(0.599) == "MEDIUM"
    assert severity_for(0.60) == "HIGH"
    assert severity_for(0.799) == "HIGH"
    assert severity_for(0.80) == "CRITICAL"


# ── Công thức max(w·p) ───────────────────────────────────────────────────────


def test_lay_max_khong_lay_tong():
    """Cộng dồn nhiều detection sẽ làm một clip ồn ào nhiều sự kiện nhỏ vượt mặt một
    tiếng súng rõ ràng — spec ghi max, không phải tổng."""
    result = score_event(
        [
            Detection("siren", 0.0, 1.0, 0.9),       # 0.4 × 0.9 = 0.36
            Detection("shout_yell", 1.0, 2.0, 0.9),  # 0.7 × 0.9 = 0.63
        ],
        TIERS,
    )
    assert result.risk_score == 0.63


def test_sung_ro_rang_thanh_critical():
    result = score_event([Detection("gunshot", 1.0, 1.4, 0.93)], TIERS)
    assert result.severity == "CRITICAL"
    assert result.risk_score == 0.93


def test_diem_khong_bao_gio_vuot_1():
    result = score_event([Detection("gunshot", 0.0, 1.0, 1.0)], TIERS)
    assert result.risk_score <= 1.0


# ── Guard: chặn confidence thấp (θ_low) ─────────────────────────────────────


def test_detection_confidence_thap_khong_duoc_tinh():
    result = score_event([Detection("gunshot", 0.0, 1.0, CONFIDENCE_FLOOR - 0.01)], TIERS)
    assert result.risk_score == 0.0
    assert result.severity == "LOW"
    assert "confidence_floor" in result.guards_applied


# ── Guard: chặn Nhóm B ───────────────────────────────────────────────────────


def test_phao_hoa_khong_duoc_bao_dong(capsys):
    """Nguồn báo động giả số 1 ở bối cảnh VN (Tết, đám cưới). Không có guard này thì
    mỗi đêm giao thừa là một chuỗi cảnh báo."""
    result = score_event([Detection("fireworks", 0.0, 3.0, 0.98)], TIERS)
    assert result.severity == "LOW"
    assert "group_b_block" in result.guards_applied


def test_nhom_b_khong_che_duoc_su_kien_that():
    """Pháo hoa to hơn nhưng có tiếng súng vượt θ_high → KHÔNG được ép xuống LOW.
    Đây là nửa còn lại của guard: chặn báo động giả mà vẫn không bỏ sót thật."""
    result = score_event(
        [
            Detection("fireworks", 0.0, 3.0, 0.99),
            Detection("gunshot", 1.0, 1.4, 0.85),
        ],
        TIERS,
    )
    assert result.severity == "CRITICAL"
    assert "group_b_block" not in result.guards_applied


def test_nhom_a_yeu_khong_go_duoc_guard():
    # gunshot 0.3 < θ_high (0.60) → vẫn coi là cảnh nền có pháo, không phải sự cố.
    result = score_event(
        [
            Detection("applause_cheering", 0.0, 3.0, 0.95),
            Detection("gunshot", 1.0, 1.2, 0.30),
        ],
        TIERS,
    )
    assert result.severity == "LOW"
    assert "group_b_block" in result.guards_applied


# ── Ca biên ──────────────────────────────────────────────────────────────────


def test_khong_co_detection_nao():
    result = score_event([], TIERS)
    assert (result.risk_score, result.severity) == (0.0, "LOW")


def test_lop_la_bi_bo_qua_khong_doan_trong_so():
    """Lớp không có trong taxonomy: bỏ qua, KHÔNG gán trọng số mặc định — đoán trọng số
    cho lớp không biết là tự tạo cảnh báo từ hư không."""
    result = score_event([Detection("tieng_gi_do_la", 0.0, 1.0, 0.99)], TIERS)
    assert result.risk_score == 0.0


def test_components_noi_ro_phan_chua_lam():
    # combo/ctx = 0.0 tường minh để chỗ gọi biết đây là bản W1 chưa đủ, không phải
    # đã tính ra 0.
    result = score_event([Detection("gunshot", 0.0, 1.0, 0.9)], TIERS)
    assert result.components["combo"] == 0.0
    assert result.components["ctx"] == 0.0
    assert result.components["max_weighted"] > 0
