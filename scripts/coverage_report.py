"""Báo cáo độ phủ dữ liệu theo lớp — trả lời "còn thiếu gì, thiếu bao nhiêu".

    python scripts/coverage_report.py
    python scripts/coverage_report.py --stage raw       # trước khi chuẩn hoá
    python scripts/coverage_report.py --markdown        # dán vào docs/data_inventory.md

Đây là báo cáo dùng ở **cổng quyết định D7** (DATA_PLAN §11): lớp nào còn dưới
`min_acceptable` ở mốc đó phải chọn một trong bốn phương án và ghi thành ADR.

Số liệu lấy từ `interim/normalized/` chứ không từ manifest, vì chỉ clip đã qua cổng
chất lượng mới thực sự dùng được — báo cáo theo manifest sẽ lạc quan hơn thực tế.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import soundfile
import yaml

from common import REPO_ROOT, enable_utf8_output

ONTOLOGY_PATH = REPO_ROOT / "ml" / "configs" / "ontology_map.yaml"
MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "raw_manifest.csv"
EXCLUSIONS_PATH = REPO_ROOT / "data" / "manifests" / "exclusions.csv"
NORMALIZED_DIR = REPO_ROOT / "data" / "interim" / "normalized"


@dataclass
class ClassCoverage:
    class_id: str
    group: str
    n_clips: int
    n_source_groups: int
    minutes: float
    target: int
    minimum: int
    sources: list[str]

    @property
    def status(self) -> str:
        if self.target == 0:
            return "—"
        if self.n_clips >= self.target:
            return "✅ đủ"
        if self.n_clips >= self.minimum:
            return "🟡 trên mức tối thiểu"
        if self.n_clips > 0:
            return "🔴 thiếu"
        return "⛔ chưa có"

    @property
    def clips_per_group(self) -> float:
        """Cao thì rủi ro rò rỉ cao: nhiều clip cắt từ cùng một bản ghi gốc."""
        return self.n_clips / self.n_source_groups if self.n_source_groups else 0.0


def load_manifest_index() -> dict[str, dict]:
    if not MANIFEST_PATH.exists():
        return {}
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as handle:
        return {row["file_id"]: row for row in csv.DictReader(handle)}


def collect_normalized(class_id: str) -> list[str]:
    directory = NORMALIZED_DIR / class_id
    return [p.stem for p in directory.glob("*.wav")] if directory.exists() else []


def collect_raw(index: dict[str, dict], class_id: str) -> list[str]:
    return [fid for fid, row in index.items() if row["claimed_class"] == class_id]


def measure_minutes(class_id: str, file_ids: list[str], index: dict[str, dict], stage: str) -> float:
    if stage == "raw":
        return sum(float(index[fid]["duration"]) for fid in file_ids if fid in index) / 60
    total = sum(soundfile.info(str(NORMALIZED_DIR / class_id / f"{fid}.wav")).duration for fid in file_ids)
    return total / 60


def build_coverage(ontology: dict, index: dict[str, dict], stage: str) -> list[ClassCoverage]:
    coverage = []
    for class_id, spec in ontology["classes"].items():
        file_ids = collect_raw(index, class_id) if stage == "raw" else collect_normalized(class_id)
        rows = [index[fid] for fid in file_ids if fid in index]
        coverage.append(ClassCoverage(
            class_id=class_id,
            group=spec.get("group", "?"),
            n_clips=len(file_ids),
            n_source_groups=len({r["source_group_id"] for r in rows}),
            minutes=measure_minutes(class_id, file_ids, index, stage),
            target=spec.get("target_foreground", 0),
            minimum=spec.get("min_acceptable", 0),
            sources=sorted({r["source_dataset"] for r in rows}),
        ))
    return coverage


def count_exclusions() -> dict[str, int]:
    if not EXCLUSIONS_PATH.exists():
        return {}
    with EXCLUSIONS_PATH.open(encoding="utf-8", newline="") as handle:
        counts: dict[str, int] = {}
        for row in csv.DictReader(handle):
            counts[row["reason_code"]] = counts.get(row["reason_code"], 0) + 1
    return counts


def print_table(coverage: list[ClassCoverage], stage: str) -> None:
    print(f"\nĐộ phủ ở giai đoạn: {stage}\n")
    print(f"{'lớp':<21}{'nh':>3}{'clip':>6}{'nhóm':>6}{'clip/nhóm':>11}{'phút':>7}"
          f"{'mục tiêu':>10}  {'trạng thái':<24}nguồn")
    print("─" * 118)
    for item in sorted(coverage, key=lambda c: (c.group, c.class_id)):
        print(f"{item.class_id:<21}{item.group:>3}{item.n_clips:>6}{item.n_source_groups:>6}"
              f"{item.clips_per_group:>11.1f}{item.minutes:>7.1f}{item.target:>10}  "
              f"{item.status:<24}{', '.join(item.sources)[:34]}")
    print("─" * 118)
    total_clips = sum(c.n_clips for c in coverage)
    total_minutes = sum(c.minutes for c in coverage)
    print(f"{'TỔNG':<21}{'':>3}{total_clips:>6}{'':>6}{'':>11}{total_minutes:>7.1f}")


def print_gaps(coverage: list[ClassCoverage]) -> None:
    below = [c for c in coverage if c.target > 0 and c.n_clips < c.minimum]
    if not below:
        print("\n✅ Mọi lớp đều đạt mức tối thiểu.")
        return

    print(f"\n⚠️ {len(below)} lớp chưa đạt mức tối thiểu — đây là danh sách cho cổng D7:")
    for item in sorted(below, key=lambda c: c.n_clips - c.minimum):
        print(f"   {item.class_id:<21} {item.n_clips:>4}/{item.minimum:<4} còn thiếu {item.minimum - item.n_clips}")

    risky = [c for c in coverage if c.clips_per_group >= 2.0]
    if risky:
        print("\n⚠️ Lớp có nhiều clip cắt từ cùng một bản ghi gốc (rủi ro rò rỉ khi chia tập):")
        for item in sorted(risky, key=lambda c: -c.clips_per_group):
            print(f"   {item.class_id:<21} {item.clips_per_group:.1f} clip/nhóm"
                  f" ({item.n_clips} clip từ {item.n_source_groups} bản ghi)")


def print_markdown(coverage: list[ClassCoverage]) -> None:
    print("| Lớp | Nhóm | Clip | Nhóm nguồn | Phút | Mục tiêu | Tối thiểu | Trạng thái |")
    print("|---|---|---:|---:|---:|---:|---:|---|")
    for item in sorted(coverage, key=lambda c: (c.group, c.class_id)):
        print(f"| `{item.class_id}` | {item.group} | {item.n_clips} | {item.n_source_groups} | "
              f"{item.minutes:.1f} | {item.target} | {item.minimum} | {item.status} |")


def main() -> int:
    enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["raw", "normalized"], default="normalized",
                        help="raw = mọi file đã tải · normalized = đã qua cổng chất lượng (mặc định)")
    parser.add_argument("--markdown", action="store_true", help="in bảng Markdown để dán vào tài liệu")
    args = parser.parse_args()

    ontology = yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    coverage = build_coverage(ontology, load_manifest_index(), args.stage)

    if args.markdown:
        print_markdown(coverage)
        return 0

    print_table(coverage, args.stage)
    if counts := count_exclusions():
        total = sum(counts.values())
        breakdown = " · ".join(f"{reason} {count}" for reason, count in sorted(counts.items(), key=lambda kv: -kv[1]))
        print(f"\nĐã loại {total} clip: {breakdown}")
    print_gaps(coverage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
