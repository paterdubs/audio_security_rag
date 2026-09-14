"""Kiểm tra ml/configs/ontology_map.yaml đối chiếu với các file metadata gốc.

Chạy: python scripts/verify_ontology.py
Thoát 1 nếu có ERROR. Dùng trong CI để không ai sửa ontology_map.yaml mà làm hỏng nó.

Triết lý: file YAML là nguồn chân lý cho *ý định* của ta; các file trong data/reference/
là nguồn chân lý cho *sự thật* của dataset. Script này bắt chỗ hai bên lệch nhau.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from common import REPO_ROOT, enable_utf8_output

CONFIG_PATH = REPO_ROOT / "ml" / "configs" / "ontology_map.yaml"

EXPECTED_N_CLASSES = 15
EXPECTED_GROUP_SIZES = {"A": 10, "B": 5}
REQUIRED_CLASS_KEYS = (
    "group",
    "audioset_ids",
    "audioset_names",
    "fsd50k_labels",
    "esc50_labels",
    "urbansound_labels",
    "desed_labels",
    "mivia_labels",
    "confusable_with",
    "target_foreground",
    "min_acceptable",
)

ERROR, WARN, INFO = "ERROR", "WARN", "INFO"


@dataclass(frozen=True)
class Issue:
    level: str
    code: str
    message: str


# ── Nạp dữ liệu tham chiếu ───────────────────────────────────────────────────


def load_ontology(path: Path) -> dict[str, str]:
    """Trả về {audioset_id: display_name} từ file ontology gốc."""
    nodes = json.loads(path.read_text(encoding="utf-8"))
    return {node["id"]: node["name"] for node in nodes}


def load_strong_vocab(path: Path) -> set[str]:
    """Trả về tập id CÓ strong label (bộ 456 lớp của AudioSet-strong)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return {line.split("\t")[0] for line in lines if line.strip()}


def load_esc50_categories(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["category"] for row in csv.DictReader(handle)}


# ── Các phép kiểm tra ────────────────────────────────────────────────────────


def check_structure(classes: dict) -> list[Issue]:
    issues: list[Issue] = []
    if len(classes) != EXPECTED_N_CLASSES:
        issues.append(Issue(ERROR, "C01", f"Cần {EXPECTED_N_CLASSES} lớp, đang có {len(classes)}"))

    for name, spec in classes.items():
        missing = [key for key in REQUIRED_CLASS_KEYS if key not in spec]
        if missing:
            issues.append(Issue(ERROR, "C11", f"{name}: thiếu khoá {missing}"))
        if spec.get("target_foreground", 0) < spec.get("min_acceptable", 0):
            issues.append(Issue(ERROR, "C07", f"{name}: target_foreground < min_acceptable"))

    for group, expected in EXPECTED_GROUP_SIZES.items():
        actual = sum(1 for spec in classes.values() if spec.get("group") == group)
        if actual != expected:
            issues.append(Issue(ERROR, "C01", f"Nhóm {group}: cần {expected} lớp, đang có {actual}"))
    return issues


def check_audioset_ids(classes: dict, ontology: dict[str, str]) -> list[Issue]:
    """Mọi id phải tồn tại, và tên đi kèm phải khớp tên thật — bắt lỗi chép nhầm."""
    issues: list[Issue] = []
    for name, spec in classes.items():
        ids = spec.get("audioset_ids", [])
        names = spec.get("audioset_names", [])
        if len(ids) != len(names):
            issues.append(Issue(ERROR, "C03", f"{name}: {len(ids)} id nhưng {len(names)} tên"))
            continue
        for audioset_id, declared in zip(ids, names):
            actual = ontology.get(audioset_id)
            if actual is None:
                issues.append(Issue(ERROR, "C02", f"{name}: id {audioset_id} không có trong ontology gốc"))
            elif actual != declared:
                issues.append(
                    Issue(ERROR, "C03", f"{name}: {audioset_id} tên thật là {actual!r}, file ghi {declared!r}")
                )
    return issues


def check_id_uniqueness(classes: dict) -> list[Issue]:
    """Một id thuộc hai lớp nghĩa là taxonomy nhập nhằng — phải quyết, không để lửng."""
    owners: dict[str, list[str]] = {}
    for name, spec in classes.items():
        for audioset_id in spec.get("audioset_ids", []):
            owners.setdefault(audioset_id, []).append(name)
    return [
        Issue(ERROR, "C05", f"id {audioset_id} bị dùng bởi nhiều lớp: {names}")
        for audioset_id, names in owners.items()
        if len(names) > 1
    ]


def check_strong_availability(classes: dict, strong_ids: set[str]) -> list[Issue]:
    """Id không có strong label thì phải được KHAI BÁO rõ ở audioset_weak_only."""
    issues: list[Issue] = []
    for name, spec in classes.items():
        declared_weak = set(spec.get("audioset_weak_only", []))
        actual_weak = {i for i in spec.get("audioset_ids", []) if i not in strong_ids}

        for audioset_id in sorted(actual_weak - declared_weak):
            issues.append(
                Issue(WARN, "C04", f"{name}: {audioset_id} KHÔNG có strong label — thêm vào audioset_weak_only")
            )
        for audioset_id in sorted(declared_weak - actual_weak):
            issues.append(
                Issue(ERROR, "C04", f"{name}: {audioset_id} khai là weak-only nhưng thực tế CÓ strong label")
            )
    return issues


def check_confusable(classes: dict) -> list[Issue]:
    """confusable_with điều khiển luật định tuyến sang người — sai là hỏng quy trình gán nhãn."""
    issues: list[Issue] = []
    for name, spec in classes.items():
        for other in spec.get("confusable_with", []):
            if other == name:
                issues.append(Issue(ERROR, "C06", f"{name}: tự nhầm với chính nó"))
            elif other not in classes:
                issues.append(Issue(ERROR, "C06", f"{name}: nhầm với lớp không tồn tại {other!r}"))
            elif name not in classes[other].get("confusable_with", []):
                issues.append(Issue(ERROR, "C06", f"Không đối xứng: {name}→{other} nhưng {other}↛{name}"))
    return issues


def check_esc50(classes: dict, categories: set[str]) -> list[Issue]:
    return [
        Issue(ERROR, "C08", f"{name}: ESC-50 không có lớp {label!r}")
        for name, spec in classes.items()
        for label in spec.get("esc50_labels", [])
        if label not in categories
    ]


def load_fsd50k_vocab(path: Path) -> dict[str, str]:
    """{mã AudioSet: nhãn FSD50K} — FSD50K công bố mã ngay trong vocabulary.csv."""
    with path.open(encoding="utf-8", newline="") as handle:
        return {row[2]: row[1] for row in csv.reader(handle) if len(row) >= 3}


def check_fsd50k(classes: dict, vocab: dict[str, str]) -> list[Issue]:
    """fsd50k_labels phải SUY RA ĐƯỢC từ audioset_ids, không phải gõ tay.

    Ánh xạ bằng mã AudioSet chắc hơn khớp chuỗi tên nhiều: FSD50K đặt tên lớp theo
    quy ước riêng (`Walk_and_footsteps`, `Dishes_and_pots_and_pans`) nên đoán tên là
    sai lặng lẽ — lớp vẫn tồn tại, chỉ là không khớp và ta tưởng nguồn không có.
    """
    issues: list[Issue] = []
    for name, spec in classes.items():
        expected = [vocab[i] for i in spec.get("audioset_ids", []) if i in vocab]
        declared = spec.get("fsd50k_labels", [])
        if sorted(declared) != sorted(expected):
            issues.append(Issue(ERROR, "C12", f"{name}: fsd50k_labels là {declared}, suy từ mã phải là {expected}"))
    return issues


def check_background_rejects(config: dict, ontology: dict[str, str]) -> list[Issue]:
    """Mọi id chặn nền phải tồn tại và phải thuộc một lớp ta đang theo dõi."""
    issues: list[Issue] = []
    known = {i for spec in config["classes"].values() for i in spec.get("audioset_ids", [])}
    for audioset_id in config.get("background_reject_ids", []):
        if audioset_id not in ontology:
            issues.append(Issue(ERROR, "C10", f"background_reject_ids: {audioset_id} không có trong ontology"))
        elif audioset_id not in known:
            issues.append(Issue(WARN, "C10", f"background_reject_ids: {audioset_id} không thuộc lớp nào"))
    return issues


# ── Điều phối ────────────────────────────────────────────────────────────────


def resolve(path_str: str) -> Path:
    return REPO_ROOT / path_str


def run_checks(config: dict) -> list[Issue]:
    classes = config["classes"]
    refs = config["meta"]["reference_files"]

    issues = check_structure(classes)
    issues += check_id_uniqueness(classes)
    issues += check_confusable(classes)

    ontology_path = resolve(refs["audioset_ontology"])
    if ontology_path.exists():
        ontology = load_ontology(ontology_path)
        issues += check_audioset_ids(classes, ontology)
        issues += check_background_rejects(config, ontology)
    else:
        issues.append(Issue(ERROR, "C09", f"Thiếu file ontology gốc: {ontology_path}"))

    strong_path = resolve(refs["audioset_strong_vocab"])
    if strong_path.exists():
        issues += check_strong_availability(classes, load_strong_vocab(strong_path))
    else:
        issues.append(Issue(WARN, "C09", f"Chưa có {strong_path} — bỏ qua kiểm tra strong label"))

    fsd50k_path = resolve(refs["fsd50k_vocab"]) if "fsd50k_vocab" in refs else None
    if fsd50k_path and fsd50k_path.exists():
        issues += check_fsd50k(classes, load_fsd50k_vocab(fsd50k_path))
    elif fsd50k_path:
        issues.append(Issue(WARN, "C09", f"Chưa có {fsd50k_path} — bỏ qua kiểm tra FSD50K"))

    esc50_path = resolve(refs["esc50_meta"])
    if esc50_path.exists():
        issues += check_esc50(classes, load_esc50_categories(esc50_path))
    else:
        issues.append(Issue(WARN, "C09", f"Chưa có {esc50_path} — bỏ qua kiểm tra ESC-50"))

    for key, path_str in config["meta"].get("reference_files_pending", {}).items():
        if not resolve(path_str).exists():
            issues.append(Issue(INFO, "C09", f"Chưa tải {key} → nhãn nguồn đó vẫn ở trạng thái CẦN XÁC MINH"))
    return issues


def report(issues: list[Issue]) -> int:
    for level in (ERROR, WARN, INFO):
        selected = [i for i in issues if i.level == level]
        if not selected:
            continue
        print(f"\n{level} ({len(selected)}):")
        for issue in selected:
            print(f"  [{issue.code}] {issue.message}")

    n_errors = sum(1 for i in issues if i.level == ERROR)
    n_warns = sum(1 for i in issues if i.level == WARN)
    print(f"\n{'✗ THẤT BẠI' if n_errors else '✓ ĐẠT'} — {n_errors} lỗi, {n_warns} cảnh báo")
    return 1 if n_errors else 0


def main() -> int:
    enable_utf8_output()
    if not CONFIG_PATH.exists():
        print(f"Không tìm thấy {CONFIG_PATH}", file=sys.stderr)
        return 1

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    print(f"Kiểm tra {CONFIG_PATH.relative_to(REPO_ROOT)} (v{config['meta']['version']}, "
          f"{len(config['classes'])} lớp)")
    return report(run_checks(config))


if __name__ == "__main__":
    sys.exit(main())
