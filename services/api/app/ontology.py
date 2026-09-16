"""Đọc taxonomy từ ml/configs/ontology_map.yaml — nguồn chân lý duy nhất.

File này được mount read-only vào container. Service KHÔNG chép lại danh sách 16 lớp:
mọi lần taxonomy đổi (đã đổi thật một lần: tách `laughter_cheering` → `laughter` +
`applause_cheering`), bản sao thứ hai sẽ âm thầm lệch đi mà không có gì báo.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from app.config import get_settings


@lru_cache(maxsize=1)
def _load(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        raise RuntimeError(
            f"Không thấy ontology tại {path}. File này phải được mount read-only vào container "
            "(xem docker-compose.yml) — service không được đoán taxonomy."
        )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _config() -> dict:
    return _load(str(get_settings().ontology_path))


def class_tiers() -> dict[str, str]:
    """{class_id: tier} — dùng cho risk scoring (§6.4)."""
    return {name: spec["tier"] for name, spec in _config()["classes"].items()}


def class_ids() -> set[str]:
    return set(_config()["classes"])


def group_of(class_id: str) -> str | None:
    spec = _config()["classes"].get(class_id)
    return spec.get("group") if spec else None
