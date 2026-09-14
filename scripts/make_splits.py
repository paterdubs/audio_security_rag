"""Chia dữ liệu thành foreground_bank_train / dev / gold_test — DATA_PLAN §6.

    python scripts/make_splits.py
    python scripts/make_splits.py --dry-run

Ba điều script này làm mà chia ngẫu nhiên theo file không làm được:
  1. GỘP NHÓM BẮC CẦU trước khi chia. DESED có hai id Freesound khác nhau trỏ cùng
     một audio giống từng byte; chia theo id nguyên bản là rò rỉ có thật, đã đo.
  2. RÀNG BUỘC THEO NGUỒN. Mẩu sự kiện đã cắt rời không thay thế được cảnh thật
     trong gold_test — xem ml/configs/splits.yaml.
  3. GÁN THEO BĂM, không xáo trộn danh sách. Thêm dữ liệu mới không làm xáo lại
     nhóm đã gán, nên kết quả thí nghiệm cũ vẫn so sánh được với kết quả mới.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from common import REPO_ROOT, enable_utf8_output

MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "raw_manifest.csv"
EXCLUSIONS_PATH = REPO_ROOT / "data" / "manifests" / "exclusions.csv"
DEDUP_PATH = REPO_ROOT / "data" / "manifests" / "dedup_groups.csv"
SPLITS_PATH = REPO_ROOT / "data" / "manifests" / "splits.csv"
CONFIG_PATH = REPO_ROOT / "ml" / "configs" / "splits.yaml"

SPLIT_FIELDS = ["file_id", "class_id", "source_dataset", "source_group_id", "merged_group_id", "split"]
ALL_SPLITS = ["foreground_bank_train", "dev", "gold_test"]


# ── Gộp nhóm bắc cầu ─────────────────────────────────────────────────────────


class UnionFind:
    """Gộp các nhóm nguồn bị nối với nhau qua bản sao giống hệt.

    Quan hệ "trùng byte" có tính bắc cầu: A≡B và B≡C thì A, B, C phải cùng một tập.
    Gộp từng cặp một mà không truy vết bắc cầu sẽ bỏ sót đúng những chuỗi dài nhất.
    """

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:      # nén đường đi
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            # Luôn lấy id nhỏ hơn làm gốc → merged_group_id ổn định giữa các lần chạy,
            # không phụ thuộc thứ tự đọc file.
            low, high = sorted([root_a, root_b])
            self.parent[high] = low


def merge_groups(manifest_rows: list[dict], dedup_rows: list[dict]) -> dict[str, str]:
    """{source_group_id: merged_group_id}, đã gộp mọi nhóm nối nhau qua bản trùng."""
    union = UnionFind()
    for row in manifest_rows:
        union.find(row["source_group_id"])
    for row in dedup_rows:
        members = [g for g in row["source_group_ids"].split("|") if g]
        for other in members[1:]:
            union.union(members[0], other)
    return {group: union.find(group) for group in union.parent}


# ── Gán tập ──────────────────────────────────────────────────────────────────


def eligible_splits(config: dict, merged: str, datasets: set[str]) -> list[str]:
    """Các tập mà MỌI nguồn trong nhóm đều chấp nhận.

    Nhóm gộp có thể chứa nhiều nguồn (chính là chuyện bản trùng bắc cầu). Nếu giao
    của chúng rỗng thì có một mẩu bank giống hệt một clip dùng để đánh giá — không
    cách gán nào an toàn, phải người xử lý. Dừng, không chọn bừa một bên.
    """
    rules = config["eligibility"]
    unknown = sorted(datasets - set(rules))
    if unknown:
        raise SystemExit(
            f"❌ nguồn chưa khai trong splits.yaml: {unknown}\n"
            f"   thêm vào `eligibility` — không có mặc định, vì chọn nhầm tập là rò rỉ"
        )
    allowed: set[str] = set(ALL_SPLITS)
    for dataset in datasets:
        allowed &= set(rules[dataset])
    if not allowed:
        raise SystemExit(
            f"❌ nhóm {merged} gộp các nguồn không dùng chung tập nào: {sorted(datasets)}\n"
            f"   nghĩa là một mẩu bank trùng byte với một clip đánh giá.\n"
            f"   Xử lý tay: loại một bên vào exclusions.csv rồi chạy lại."
        )
    return [s for s in ALL_SPLITS if s in allowed]


def hash_fraction(seed: str, merged: str) -> float:
    """Số thực ổn định trong [0,1) từ id nhóm. Băm chứ không xáo trộn danh sách:
    thêm nhóm mới thì nhóm cũ giữ nguyên tập, kết quả cũ vẫn so sánh được."""
    digest = hashlib.sha256(f"{seed}|{merged}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_split(config: dict, merged: str, allowed: list[str]) -> str:
    if len(allowed) == 1:
        return allowed[0]
    weights = [float(config["ratios"].get(s, 0.0)) for s in allowed]
    total = sum(weights)
    if total <= 0:
        raise SystemExit(f"❌ splits.yaml thiếu `ratios` cho {allowed}")
    point = hash_fraction(config["seed"], merged) * total
    running = 0.0
    for split, weight in zip(allowed, weights):
        running += weight
        if point < running:
            return split
    return allowed[-1]


# ── Vào/ra ───────────────────────────────────────────────────────────────────


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def excluded_file_ids() -> set[str]:
    return {row["file_id"] for row in read_csv(EXCLUSIONS_PATH)}


def build_splits(manifest_rows: list[dict], dedup_rows: list[dict], config: dict) -> list[dict]:
    group_of = merge_groups(manifest_rows, dedup_rows)

    datasets_in_group: dict[str, set[str]] = defaultdict(set)
    for row in manifest_rows:
        datasets_in_group[group_of[row["source_group_id"]]].add(row["source_dataset"])

    split_of_group = {
        merged: assign_split(config, merged, eligible_splits(config, merged, datasets))
        for merged, datasets in sorted(datasets_in_group.items())
    }

    return [
        {
            "file_id": row["file_id"],
            "class_id": row["claimed_class"],
            "source_dataset": row["source_dataset"],
            "source_group_id": row["source_group_id"],
            "merged_group_id": group_of[row["source_group_id"]],
            "split": split_of_group[group_of[row["source_group_id"]]],
        }
        for row in manifest_rows
    ]


def write_splits(rows: list[dict]) -> None:
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SPLITS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPLIT_FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["split"], r["class_id"], r["file_id"])))


def report(rows: list[dict], group_of: dict[str, str], dedup_rows: list[dict]) -> None:
    bridging = sum(1 for row in dedup_rows if len({g for g in row["source_group_ids"].split("|") if g}) > 1)
    shrunk = len(group_of) - len(set(group_of.values()))
    print(f"\n▶ {len(set(group_of.values()))} nhóm sau khi gộp "
          f"({len(group_of)} nhóm nguyên bản, {shrunk} bị gộp vào nhóm khác "
          f"vì {bridging} bản trùng bắc cầu)")

    per_split = Counter(row["split"] for row in rows)
    print(f"\n{'tập':<24}{'clip':>7}{'nhóm':>7}  lớp")
    print("─" * 78)
    for split in ALL_SPLITS:
        subset = [r for r in rows if r["split"] == split]
        groups = len({r["merged_group_id"] for r in subset})
        classes = sorted({r["class_id"] for r in subset})
        shown = ", ".join(classes[:4]) + (f" +{len(classes) - 4}" if len(classes) > 4 else "")
        print(f"{split:<24}{per_split[split]:>7}{groups:>7}  {shown or '—'}")

    if not per_split["gold_test"]:
        print("\n⚠️ gold_test RỖNG. Không nguồn audio thật nào đã tải — cần AudioSet-strong,")
        print("   MIVIA (đang chờ duyệt), hoặc thu thực địa tại IUH. Cổng D7 chưa qua được.")
    if not per_split["dev"]:
        print("⚠️ dev RỖNG — cùng lý do trên.")


def main() -> int:
    enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="chỉ báo cáo, không ghi splits.csv")
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    dropped = excluded_file_ids()
    manifest_rows = [r for r in read_csv(MANIFEST_PATH) if r["file_id"] not in dropped]
    if not manifest_rows:
        print("raw_manifest.csv rỗng hoặc mọi dòng đã bị loại. Chạy build_manifest.py trước.")
        return 1
    print(f"▶ {len(manifest_rows)} clip còn lại ({len(dropped)} đã bị loại ở exclusions.csv)")

    dedup_rows = read_csv(DEDUP_PATH)
    rows = build_splits(manifest_rows, dedup_rows, config)
    group_of = merge_groups(manifest_rows, dedup_rows)
    report(rows, group_of, dedup_rows)

    if args.dry_run:
        print("\n(--dry-run: chưa ghi splits.csv)")
    else:
        write_splits(rows)
        print(f"\n✓ đã ghi {SPLITS_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
