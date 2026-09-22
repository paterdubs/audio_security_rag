"""Năm kiểm tra rò rỉ giữa các tập — DATA_PLAN §6. Chạy trong CI.

    python scripts/check_leakage.py
    python scripts/check_leakage.py --nhanh     # bỏ kiểm tra 4–5 (không đọc JAMS)

Trả mã 0 khi sạch, 1 khi có rò rỉ. Rò rỉ audio không làm gì hỏng ngay: nó chỉ làm
điểm số đẹp lên. Không có cổng tự động thì không ai phát hiện ra, và con số sai sẽ
đi thẳng vào khoá luận. Vì vậy đây là kiểm tra CHẶN, không phải cảnh báo.

Kiểm tra 1–3 soi `splits.csv` — tức soi Ý ĐỊNH chia tập, chạy tức thì.
Kiểm tra 4 soi ĐẦU RA thật (JAMS đã sinh), vì bộ lọc đầu vào của `scaper_generate.py`
tự khai báo là đủ thì không ai kiểm lại; dự án đã bị `run_manifest.doc_hop_dong` khai
PASSED trong khi thiếu file báo cáo. Đọc ~9.360 file JAMS, khoảng 20–60 s.
Kiểm tra 5 KHÔNG phải cổng: synthetic train và dev cố ý sinh từ cùng một bank nên
trùng nguồn là theo thiết kế. Nó chỉ in số để phần Hạn chế của khoá luận nói được cụ
thể thay vì câu chung chung "cùng recipe B0–B9".
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from common import REPO_ROOT, enable_utf8_output

MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "raw_manifest.csv"
DEDUP_PATH = REPO_ROOT / "data" / "manifests" / "dedup_groups.csv"
SPLITS_PATH = REPO_ROOT / "data" / "manifests" / "splits.csv"
SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"

BANK_SPLIT = "foreground_bank_train"   # tập duy nhất được phép làm nguyên liệu synthetic
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


# ── Kiểm tra 4 ───────────────────────────────────────────────────────────────


def source_id_from_path(source_file: str) -> str:
    """Đường dẫn nguồn trong JAMS → `file_id` (chính là stem của file bank).

    JAMS ghi đường dẫn TUYỆT ĐỐI của máy sinh dữ liệu, thường là đường dẫn Windows.
    Phải đổi `\\` thành `/` trước khi cắt, nếu không thì trên POSIX cả chuỗi bị coi là
    một tên file duy nhất và mọi phép so đều trượt — tức cổng luôn báo sạch.
    """
    name = source_file.replace("\\", "/").rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0] if "." in name else name


def read_jams_sources(path: Path) -> dict[str, set[str]]:
    """{'foreground': {...}, 'background': {...}} của một clip synthetic.

    JAMS hỏng (tiến trình sinh bị kill giữa chừng) trả rỗng thay vì ném: một file hỏng
    không được làm sập cả lượt kiểm 9.360 file. Số clip đọc được có in ra để một đống
    file hỏng không đi qua thành "sạch" mà không ai thấy.
    """
    out: dict[str, set[str]] = {"foreground": set(), "background": set()}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return out
    for ann in data.get("annotations", []):
        for obs in ann.get("data", []):
            value = obs.get("value") or {}
            role, source = value.get("role"), value.get("source_file")
            if role in out and source:
                out[role].add(source_id_from_path(source))
    return out


def collect_sources(jams_dir: Path) -> dict[str, dict[str, set[str]]]:
    """{clip_id: {'foreground': {...}, 'background': {...}}} cho cả một tập."""
    return {p.stem: read_jams_sources(p) for p in sorted(jams_dir.glob("*.jams"))}


def merge_role(clip_sources: dict[str, dict[str, set[str]]], role: str) -> set[str]:
    out: set[str] = set()
    for roles in clip_sources.values():
        out |= roles[role]
    return out


def check_synthetic_sources_in_bank(fg_sources: set[str], eligible: set[str],
                                    split_name: str) -> list[str]:
    """Nguồn foreground dùng THẬT mà không thuộc `foreground_bank_train`.

    Ca thảm hoạ nó bắt: một clip `gold_test` lọt vào nguyên liệu synthetic. Kiểm tra
    1–3 không bắt được vì chúng chỉ soi bảng chia tập, không soi thứ đã thực sự dùng.
    """
    return [f"{split_name}: {sid}" for sid in sorted(fg_sources - eligible)]


# ── Kiểm tra 5 — SỐ PHẢI KHAI BÁO, không phải cổng ──────────────────────────


def shared_source_stats(train_sources: set[str], dev_sources: set[str]) -> dict:
    """Mức dùng chung nguồn. Mẫu số 0 → `ti_le_dev=None` (CHƯA ĐO), không phải 0."""
    shared = train_sources & dev_sources
    return {"n_train": len(train_sources), "n_dev": len(dev_sources),
            "n_chung": len(shared),
            "ti_le_dev": (len(shared) / len(dev_sources)) if dev_sources else None}


def clips_touching(clip_sources: dict[str, set[str]], other: set[str]) -> dict:
    """Bao nhiêu clip có ÍT NHẤT một nguồn dùng lại từ tập kia."""
    n_clip = len(clip_sources)
    n_dinh = sum(1 for sources in clip_sources.values() if sources & other)
    return {"n_clip": n_clip, "n_dinh": n_dinh,
            "ti_le": (n_dinh / n_clip) if n_clip else None}


def so(x: float | None) -> str:
    return "chưa đo" if x is None else f"{x:.4f}".replace(".", ",")


# ── Chạy ─────────────────────────────────────────────────────────────────────


def main() -> int:
    enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nhanh", action="store_true",
                        help="bỏ kiểm tra 4–5 (không đọc ~9.360 file JAMS)")
    args = parser.parse_args()

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

    if not args.nhanh:
        eligible = {r["file_id"] for r in split_rows if r["split"] == BANK_SPLIT}
        by_split = {}
        for name in ("train", "dev"):
            jams_dir = SYNTHETIC_DIR / name / "jams"
            if not jams_dir.is_dir():
                print(f"  ⚠️ chưa có {jams_dir.relative_to(REPO_ROOT)} — bỏ qua tập {name}")
                continue
            by_split[name] = collect_sources(jams_dir)
        for name, clips in by_split.items():
            fg = merge_role(clips, "foreground")
            passed.append(show(
                f"synthetic {name}: {len(clips)} clip, {len(fg)} nguồn foreground đều thuộc {BANK_SPLIT}",
                check_synthetic_sources_in_bank(fg, eligible, name)))

        if {"train", "dev"} <= set(by_split):
            print("\n▶ dùng chung nguồn synthetic train ↔ dev (KHÔNG phải cổng — theo thiết kế)")
            for role in ("foreground", "background"):
                t = shared_source_stats(merge_role(by_split["train"], role),
                                        merge_role(by_split["dev"], role))
                print(f"    {role}: train {t['n_train']} · dev {t['n_dev']} · chung {t['n_chung']}"
                      f" → {so(t['ti_le_dev'])} số nguồn của dev cũng có trong train")
            c = clips_touching({k: v["foreground"] for k, v in by_split["dev"].items()},
                               merge_role(by_split["train"], "foreground"))
            print(f"    mức clip: {c['n_dinh']}/{c['n_clip']} clip dev ({so(c['ti_le'])})"
                  " có ít nhất một nguồn foreground từng dùng trong train")

    if all(passed):
        print("\n✓ không phát hiện rò rỉ")
        return 0
    print(f"\n❌ {passed.count(False)}/{len(passed)} kiểm tra KHÔNG đạt — sửa trước khi train")
    return 1


if __name__ == "__main__":
    sys.exit(main())
