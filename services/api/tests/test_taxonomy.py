"""Taxonomy phải khớp giữa ontology_map.yaml và từ điển tiếng Việt.

Đây không phải test cho vui: taxonomy ĐÃ lệch thật một lần (tách `laughter_cheering`
thành `laughter` + `applause_cheering`), và chỗ lệch kiểu đó không làm gì nổ cả — nó chỉ
khiến dashboard lặng lẽ hiện `class_id` thô thay vì tên tiếng Việt. Test này là thứ duy
nhất bắt được nó trước khi người dùng thấy.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.captions import CLASS_VI

ONTOLOGY_PATH = Path(__file__).resolve().parents[3] / "ml" / "configs" / "ontology_map.yaml"


@pytest.fixture(scope="module")
def ontology_classes() -> dict:
    if not ONTOLOGY_PATH.exists():
        pytest.skip(f"Không thấy {ONTOLOGY_PATH}")
    return yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8"))["classes"]


def test_moi_lop_trong_ontology_deu_co_ten_tieng_viet(ontology_classes: dict) -> None:
    thieu = sorted(set(ontology_classes) - set(CLASS_VI))
    assert not thieu, f"Các lớp chưa có tên tiếng Việt trong captions.py: {thieu}"


def test_khong_co_ten_tieng_viet_thua(ontology_classes: dict) -> None:
    # Chiều ngược lại cũng quan trọng: một lớp đã bị xoá khỏi ontology mà tên tiếng Việt
    # còn sót lại là dấu hiệu CLASS_VI đang mô tả một taxonomy cũ.
    thua = sorted(set(CLASS_VI) - set(ontology_classes))
    assert not thua, f"captions.py còn tên của lớp không còn trong ontology: {thua}"


def test_so_lop_khop_voi_meta() -> None:
    config = yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    assert len(config["classes"]) == config["meta"]["n_classes"]


def test_moi_lop_deu_co_group_va_tier(ontology_classes: dict) -> None:
    # class_groups() và class_tiers() đọc thẳng hai khoá này; thiếu một khoá là KeyError
    # lúc chạy, và nó sẽ nổ ở endpoint /taxonomy chứ không phải lúc nạp cấu hình.
    for class_id, spec in ontology_classes.items():
        assert spec.get("group") in {"A", "B"}, f"{class_id}: group không hợp lệ"
        assert spec.get("tier"), f"{class_id}: thiếu tier"
