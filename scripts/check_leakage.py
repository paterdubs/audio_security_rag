"""Ba kiểm tra rò rỉ giữa các tập — DATA_PLAN §6. Chạy trong CI.

    python scripts/check_leakage.py

Trả mã 0 khi sạch, 1 khi có rò rỉ. Rò rỉ audio không làm gì hỏng ngay: nó chỉ làm
điểm số đẹp lên. Không có cổng tự động thì không ai phát hiện ra, và con số sai sẽ
đi thẳng vào khoá luận. Vì vậy đây là kiểm tra CHẶN, không phải cảnh báo.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

from common import REPO_ROOT, enable_utf8_output

MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "raw_manifest.csv"
DEDUP_PATH = REPO_ROOT / "data" / "manifests" / "dedup_groups.csv"
SPLITS_PATH = REPO_ROOT / "data" / "manifests" / "splits.csv"

MAX_SHOWN = 5      # in vài ví dụ là đủ để lần ra; đổ hết ra chỉ làm log không đọc được


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def show(title: str, offenders: list[str]) -> bool:
    """In kết quả một kiểm tra, trả True nếu ĐẠT."""
    if not offenders:
        print(f"  ✓ {title}")
        return True
    print(f"  ✗ {title} — {len(offenders)} trường hợp")
    for line in offenders[:MAX_SHOWN]:
        print(f"      {line}")
    if len(offenders) > MAX_SHOWN:
        print(f"      … còn {len(offenders) - MAX_SHOWN} trường hợp nữa")
    return False


# ── Kiểm tra 1 ───────────────────────────────────────────────────────────────


def check_group_not_split(split_rows: list[dict]) -> list[str]:
    """Không source_group_id nào xuất hiện ở hai tập.

    Kiểm tra trên id NGUYÊN BẢN chứ không phải id đã gộp: gộp là việc của
    make_splits.py, còn đây phải kiểm tra độc lập với nó. Kiểm tra lại chính đầu ra
    của bước gộp thì luôn đạt, kể cả khi bước gộp sai.
    """
    splits_of: dict[str, set[str]] = defaultdict(set)
    for row in split_rows:
        splits_of[row["source_group_id"]].add(row["split"])
    return [f"{group}: {sorted(splits)}" for group, splits in sorted(splits_of.items()) if len(splits) > 1]


# ── Kiểm tra 2 ───────────────────────────────────────────────────────────────


def check_no_shared_audio(split_rows: list[dict], manifest_rows: list[dict]) -> list[str]:
    """Không audio nào (theo SHA-256) nằm ở hai tập.

    Bao trùm cả trường hợp hai file khác tên, khác nguồn, khác id nhóm nhưng giống
    hệt nhau — đúng tình huống đã đo được ở DESED.
    """
    checksum_of = {row["file_id"]: row["checksum_sha256"] for row in manifest_rows}
    splits_of: dict[str, set[str]] = defaultdict(set)
    files_of: dict[str, list[str]] = defaultdict(list)
    for row in split_rows:
        checksum = checksum_of.get(row["file_id"])
        if not checksum:
            continue
        splits_of[checksum].add(row["split"])
        files_of[checksum].append(row["file_id"])
    return [
        f"{checksum[:12]}… ở {sorted(splits)}: {', '.join(sorted(files_of[checksum])[:3])}"
        for checksum, splits in sorted(splits_of.items())
        if len(splits) > 1
    ]


# ── Kiểm tra 3 ───────────────────────────────────────────────────────────────


def check_dedup_groups_intact(split_rows: list[dict], dedup_rows: list[dict]) -> list[str]:
    """Không nhóm trùng lặp nào bắc cầu giữa hai tập.

    Khác kiểm tra 2: ở đây soi theo dedup_groups.csv, nên bắt được cả nhóm mà một
    thành viên đã bị loại khỏi manifest (build_manifest gộp mất) — chỉ cần hai
    thành viên còn lại rơi vào hai tập là hỏng.
    """
    split_of_group: dict[str, str] = {}
    for row in split_rows:
        split_of_group[row["source_group_id"]] = row["split"]

    offenders = []
    for row in dedup_rows:
        members = [g for g in row["source_group_ids"].split("|") if g]
        splits = {split_of_group[g] for g in members if g in split_of_group}
        if len(splits) > 1:
            offenders.append(f"{row['checksum_sha256'][:12]}… nối {members} qua {sorted(splits)}")
    return offenders


# ── Chạy ─────────────────────────────────────────────────────────────────────


def main() -> int:
    enable_utf8_output()
    split_rows = read_csv(SPLITS_PATH)
    if not split_rows:
        print("❌ chưa có data/manifests/splits.csv — chạy scripts/make_splits.py trước")
        return 1

    manifest_rows = read_csv(MANIFEST_PATH)
    dedup_rows = read_csv(DEDUP_PATH)
    print(f"▶ kiểm tra {len(split_rows)} clip trên {len({r['split'] for r in split_rows})} tập\n")

    passed = [
        show("không nhóm nguồn nào nằm ở hai tập", check_group_not_split(split_rows)),
        show("không audio nào (SHA-256) nằm ở hai tập", check_no_shared_audio(split_rows, manifest_rows)),
        show("không nhóm trùng lặp nào bắc cầu giữa hai tập", check_dedup_groups_intact(split_rows, dedup_rows)),
    ]

    if all(passed):
        print("\n✓ không phát hiện rò rỉ")
        return 0
    print(f"\n❌ {passed.count(False)}/3 kiểm tra KHÔNG đạt — sửa trước khi train")
    return 1


if __name__ == "__main__":
    sys.exit(main())
