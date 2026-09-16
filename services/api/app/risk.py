"""Risk scoring theo luật — SYSTEM.md §6.4.

Cố ý dùng luật, không dùng model: minh bạch, giải thích được trước hội đồng, và **không
có dữ liệu nhãn rủi ro thật** để huấn luyện. Nâng cấp ML-based là hướng phát triển, sau
khi thu đủ `event_feedback`.

    R = min(1, max_i(w_ci · p_i) + β·combo + γ·ctx)

Bản này (walking skeleton W1) chỉ hiện thực **phần max(w·p) + guard**. `combo` và `ctx`
cần lịch sử sự kiện và cấu hình khu vực hạn chế — thuộc W7 cùng Temporal Aggregator.
Trả về `components` để chỗ nào gọi cũng biết phần nào đã tính, phần nào còn 0.

⚠️ Bốn quy tắc guard quan trọng hơn công thức: chúng mới là nơi thực sự kiểm soát False
Alarm Rate. Bỏ guard Nhóm B thì mỗi tràng pháo Tết là một cảnh báo CRITICAL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

# Trọng số theo tier — SYSTEM.md §6.4. Tier của từng lớp KHÔNG hardcode ở đây, đọc từ
# ml/configs/ontology_map.yaml (nguồn chân lý duy nhất cho taxonomy).
TIER_WEIGHTS = {"Critical": 1.0, "High": 0.7, "Medium": 0.4, "Negative": 0.05}

# LOW < 0.35 ≤ MEDIUM < 0.60 ≤ HIGH < 0.80 ≤ CRITICAL
SEVERITY_THRESHOLDS = ((0.80, "CRITICAL"), (0.60, "HIGH"), (0.35, "MEDIUM"))
LOWEST_SEVERITY = "LOW"

CONFIDENCE_FLOOR = 0.10   # θ_low: dưới mức này không được tham gia tính risk
GROUP_A_ALERT = 0.60      # θ_high: Nhóm A phải vượt mức này mới thắng được guard Nhóm B


@dataclass(frozen=True)
class Detection:
    class_id: str
    onset: float
    offset: float
    confidence: float


@dataclass(frozen=True)
class RiskResult:
    risk_score: float
    severity: str
    components: dict[str, float] = field(default_factory=dict)
    guards_applied: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def load_class_tiers(config_path: str) -> dict[str, str]:
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return {name: spec["tier"] for name, spec in data["classes"].items()}


def severity_for(score: float) -> str:
    for threshold, label in SEVERITY_THRESHOLDS:
        if score >= threshold:
            return label
    return LOWEST_SEVERITY


def _downgrade_to_low(severity: str) -> str:
    return LOWEST_SEVERITY if severity != LOWEST_SEVERITY else severity


def score_event(detections: list[Detection], tiers: dict[str, str]) -> RiskResult:
    """Điểm rủi ro + severity cho một sự kiện.

    Lớp lạ (không có trong taxonomy) bị bỏ qua thay vì gán trọng số mặc định: đoán một
    trọng số cho lớp không biết là tự tạo ra cảnh báo từ hư không.
    """
    usable = [d for d in detections if d.confidence >= CONFIDENCE_FLOOR and d.class_id in tiers]
    guards: list[str] = []
    if len(usable) < len(detections):
        guards.append("confidence_floor")

    if not usable:
        return RiskResult(0.0, LOWEST_SEVERITY, {"max_weighted": 0.0, "combo": 0.0, "ctx": 0.0}, guards)

    max_weighted = max(TIER_WEIGHTS[tiers[d.class_id]] * d.confidence for d in usable)
    score = min(1.0, max_weighted)
    severity = severity_for(score)

    # Guard Nhóm B: lớp trội là Negative và KHÔNG lớp Nhóm A nào vượt θ_high → ép ≤ LOW.
    #
    # "Lớp trội" ở đây là lớp model NGHE RÕ NHẤT (confidence cao nhất), KHÔNG phải lớp
    # có điểm đã nhân trọng số cao nhất. Dùng điểm đã nhân trọng số làm guard gần như
    # vô dụng: `gunshot` chỉ cần 0.05 confidence là đã vượt `fireworks` 0.95 (1.0×0.05 >
    # 0.05×0.95), nên mọi tràng pháo có lẫn chút nhiễu giống súng đều thoát guard —
    # đúng thứ guard sinh ra để chặn.
    dominant = max(usable, key=lambda d: d.confidence)
    group_a_alerting = any(
        tiers[d.class_id] != "Negative" and d.confidence >= GROUP_A_ALERT for d in usable
    )
    if tiers[dominant.class_id] == "Negative" and not group_a_alerting:
        severity = _downgrade_to_low(severity)
        guards.append("group_b_block")

    return RiskResult(
        risk_score=round(score, 4),
        severity=severity,
        # combo/ctx = 0.0 tường minh, không phải quên: xem docstring đầu file.
        components={"max_weighted": round(max_weighted, 4), "combo": 0.0, "ctx": 0.0},
        guards_applied=guards,
    )
