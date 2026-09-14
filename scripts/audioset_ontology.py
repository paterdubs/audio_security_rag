"""Xử lý quan hệ cha–con trong ontology AudioSet.

Tồn tại vì một cái bẫy đã suýt làm hỏng việc chọn clip FSD50K:

    Gunshot, gunfire  ⊂  Explosion
    Fireworks         ⊂  Explosion
    Siren             ⊂  Alarm

FSD50K gắn nhãn theo phân cấp — clip nào có `Gunshot` thì **luôn** kèm cả `Explosion`.
Nếu coi "clip có hai nhãn thuộc hai lớp của ta" là clip nhiều sự kiện thì
`gunshot`, `siren`, `fireworks` đều ra **0 clip dùng được**, và ta sẽ kết luận nhầm
rằng FSD50K không có ba lớp đó.

Cách xử lý: bỏ nhãn TỔ TIÊN khi nhãn con của nó cũng có mặt, giữ nhãn cụ thể nhất.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_parents(ontology_path: Path) -> dict[str, list[str]]:
    """{mã con: [mã cha]} suy từ trường `child_ids` của file ontology gốc."""
    nodes = json.loads(ontology_path.read_text(encoding="utf-8"))
    parents: dict[str, list[str]] = {}
    for node in nodes:
        for child in node.get("child_ids", []):
            parents.setdefault(child, []).append(node["id"])
    return parents


def ancestors_of(mid: str, parents: dict[str, list[str]]) -> set[str]:
    """Mọi tổ tiên của một mã. Duyệt theo chiều rộng vì ontology là DAG, không phải cây:
    một lớp có thể có nhiều cha, và có thể tồn tại chu trình do lỗi dữ liệu."""
    found: set[str] = set()
    queue = list(parents.get(mid, []))
    while queue:
        current = queue.pop()
        if current in found:
            continue
        found.add(current)
        queue.extend(parents.get(current, []))
    return found


def collapse_to_most_specific(mids: set[str], parents: dict[str, list[str]]) -> set[str]:
    """Bỏ mọi mã là tổ tiên của một mã khác trong cùng tập.

    {Gunshot, Explosion} → {Gunshot}      (phân cấp, một sự kiện)
    {Gunshot, Fireworks} → {Gunshot, Fireworks}  (hai anh em, đúng là hai sự kiện)
    """
    redundant = {ancestor for mid in mids for ancestor in ancestors_of(mid, parents) if ancestor in mids}
    return mids - redundant
