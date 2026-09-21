"""Chọn N clip pilot cho gold test set — DATA_PLAN §8.3 bước 1.

    python scripts/select_gold_pilot.py --n 30 --seed kltn-2026-gold-pilot-v1

Round-robin theo lớp, KHÔNG chọn ngẫu nhiên thuần tuý trên toàn bộ `gold_test`: lớp
nhiều file nhất (`speech_normal`, 210/367) sẽ lấn át lớp ít nhất (`door_slam`, 11/367)
nếu chọn đều — một pilot 30 clip toàn `speech_normal` không phát hiện được lỗ hổng
guideline riêng của các lớp hiếm cho tới tận lần gán đại trà, lúc sửa đã tốn 6 giờ công.

⚠️ Cột `classes_tham_khao` trong file kết quả CHỈ dùng để chọn mẫu đa dạng, không phải
để gán nhãn. Chế độ GOLD cấm máy đề xuất (`annotation_guideline.md` §7.2) — người gán
KHÔNG được nhìn cột này khi mở file audio để nghe.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import REPO_ROOT, enable_utf8_output  # noqa: E402

SPLITS_PATH = REPO_ROOT / "data" / "manifests" / "splits.csv"
SEGMENTS_PATH = REPO_ROOT / "data" / "raw" / "audioset_strong" / "segments.jsonl"
OUT_PATH = REPO_ROOT / "data" / "gold" / "pilot_v1_candidates.csv"

DEFAULT_N = 30
DEFAULT_SEED = "kltn-2026-gold-pilot-v1"


def hash_fraction(seed: str, key: str) -> float:
    """Số thực ổn định trong [0,1) từ một khoá — cùng công thức với
    `make_splits.hash_fraction`, để thứ tự "file nào trước trong một lớp" cũng tái lập
    được qua các lần chạy, không phụ thuộc thứ tự file_id tình cờ đến từ đâu."""
    digest = hashlib.sha256(f"{seed}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def chon_pilot(theo_lop: dict[str, list[str]], n: int, seed: str) -> list[str]:
    """Round-robin qua các lớp (thứ tự tên lớp, ổn định): mỗi vòng lấy MỘT file CHƯA
    chọn của mỗi lớp, cho tới khi đủ `n` hoặc hết file. Một file có thể thuộc nhiều lớp
    (một ổ 10s có nhiều sự kiện) — khi đã chọn ở lớp này thì lớp khác bỏ qua nó, không
    đếm hai lần.
    """
    con_lai = {
        lop: sorted(set(files), key=lambda f: hash_fraction(seed, f))
        for lop, files in theo_lop.items()
    }
    da_chon: set[str] = set()
    ket_qua: list[str] = []
    lop_theo_thu_tu = sorted(theo_lop)
    con_tien_trien = True
    while len(ket_qua) < n and con_tien_trien:
        con_tien_trien = False
        for lop in lop_theo_thu_tu:
            if len(ket_qua) >= n:
                break
            while con_lai[lop] and con_lai[lop][0] in da_chon:
                con_lai[lop].pop(0)
            if con_lai[lop]:
                file_id = con_lai[lop].pop(0)
                da_chon.add(file_id)
                ket_qua.append(file_id)
                con_tien_trien = True
    return ket_qua


def doc_gold_theo_lop(splits_path: Path, segments_path: Path) -> dict[str, dict[str, list[str]]]:
    """{lớp: [file_id]} CHỈ cho các nhóm nguồn đã được `make_splits.py` gán vào
    `gold_test`, cộng {file_id: path/classes} để ghi báo cáo."""
    gold_groups: set[str] = set()
    with splits_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["source_dataset"] == "audioset_strong" and row["split"] == "gold_test":
                gold_groups.add(row["source_group_id"])

    theo_lop: dict[str, list[str]] = defaultdict(list)
    thong_tin: dict[str, dict] = {}
    with segments_path.open(encoding="utf-8") as handle:
        for line in handle:
            segment = json.loads(line)
            if f"youtube_{segment['ytid']}" not in gold_groups:
                continue
            classes = sorted({e["class_id"] for e in segment["events"]})
            thong_tin[segment["file_id"]] = {"path": segment["path"], "classes": classes}
            for class_id in classes:
                theo_lop[class_id].append(segment["file_id"])
    return {"theo_lop": dict(theo_lop), "thong_tin": thong_tin}


def run(args: argparse.Namespace) -> int:
    if not args.splits.exists():
        print(f"❌ chưa có {args.splits}\n   Chạy: scripts/make_splits.py trước.")
        return 1
    if not args.segments.exists():
        print(f"❌ chưa có {args.segments}\n   Chạy: scripts/fetch_audioset_strong.py trước.")
        return 1

    du_lieu = doc_gold_theo_lop(args.splits, args.segments)
    theo_lop, thong_tin = du_lieu["theo_lop"], du_lieu["thong_tin"]
    n_file_gold = len(thong_tin)
    print(f"▶ {n_file_gold} file WAV trong gold_test, {len(theo_lop)} lớp có mặt")

    chon = chon_pilot(theo_lop, args.n, args.seed)
    print(f"  chọn {len(chon)}/{args.n} file (seed={args.seed!r})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_id", "path", "classes_tham_khao"])
        for file_id in chon:
            info = thong_tin[file_id]
            writer.writerow([file_id, info["path"], ";".join(info["classes"])])

    print(f"\n✓ {args.out.relative_to(REPO_ROOT)}")
    print("  ⚠️ Cột classes_tham_khao chỉ để tham khảo chọn mẫu — KHÔNG nhìn khi gán nhãn.")
    return 0


def main() -> int:
    enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--splits", type=Path, default=SPLITS_PATH)
    parser.add_argument("--segments", type=Path, default=SEGMENTS_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
