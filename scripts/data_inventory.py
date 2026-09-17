"""Sinh docs/data_inventory.md — bảng tổng kết dữ liệu, đọc thẳng từ manifest thật.

    python scripts/data_inventory.py

File sinh ra là **nguồn duy nhất** cho câu "class_id → nguồn → số giờ" trong khoá
luận (SYSTEM.md §3.2) — không được cập nhật tay, chạy lại script khi dữ liệu đổi.

Khác `coverage_report.py`: script đó đo ở `interim/normalized/` (mọi clip ĐỦ ĐIỀU
KIỆN vào bank, dùng cho cổng D7 — "còn thiếu gì"), còn script này đo ở
`foreground_manifest.csv` (clip ĐÃ THỰC SỰ vào bank sau sàng lọc + duyệt người —
"đang có gì"). Hai con số khác nhau có chủ đích, không phải trùng lặp.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml
from common import REPO_ROOT, enable_utf8_output

ONTOLOGY_PATH = REPO_ROOT / "ml" / "configs" / "ontology_map.yaml"
FOREGROUND_MANIFEST = REPO_ROOT / "data" / "manifests" / "foreground_manifest.csv"
BACKGROUND_MANIFEST = REPO_ROOT / "data" / "manifests" / "background_manifest.csv"
RIR_MANIFEST = REPO_ROOT / "data" / "manifests" / "rir_manifest.csv"
SPLITS_PATH = REPO_ROOT / "data" / "manifests" / "splits.csv"
RAW_MANIFEST = REPO_ROOT / "data" / "manifests" / "raw_manifest.csv"
SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"
OUTPUT_PATH = REPO_ROOT / "docs" / "data_inventory.md"

REAL_AUDIO_SPLITS = ["dev", "gold_test"]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# ── Foreground bank ──────────────────────────────────────────────────────────


def foreground_rows_by_class(rows: list[dict]) -> dict[str, list[dict]]:
    by_class: dict[str, list[dict]] = {}
    for row in rows:
        by_class.setdefault(row["class_id"], []).append(row)
    return by_class


def foreground_table(ontology: dict, rows: list[dict]) -> list[dict]:
    by_class = foreground_rows_by_class(rows)
    table = []
    for class_id, spec in ontology["classes"].items():
        clips = by_class.get(class_id, [])
        minutes = sum(float(r["duration_bank"]) for r in clips) / 60
        target = spec.get("target_foreground", 0)
        minimum = spec.get("min_acceptable", 0)
        if target == 0:
            status = "—"
        elif len(clips) >= target:
            status = "✅ đủ"
        elif len(clips) >= minimum:
            status = "🟡 trên tối thiểu"
        elif clips:
            status = "🔴 dưới tối thiểu"
        else:
            status = "⛔ chưa có"
        table.append({
            "class_id": class_id,
            "group": spec.get("group", "?"),
            "n_clips": len(clips),
            "minutes": round(minutes, 1),
            "target": target,
            "minimum": minimum,
            "status": status,
            "sources": sorted({r["source_dataset"] for r in clips}),
        })
    return table


# ── Background bank ──────────────────────────────────────────────────────────


def background_table(rows: list[dict]) -> list[dict]:
    clean = [r for r in rows if r.get("verdict") == "clean"]
    by_area: dict[str, list[dict]] = {}
    for row in clean:
        by_area.setdefault(row["area_type"], []).append(row)
    return [
        {"area_type": area, "n_clips": len(items),
         "minutes": round(sum(float(r["duration"]) for r in items) / 60, 1)}
        for area, items in sorted(by_area.items())
    ]


# ── RIR bank ──────────────────────────────────────────────────────────────────


def rir_table(rows: list[dict]) -> list[dict]:
    by_space: dict[str, list[dict]] = {}
    for row in rows:
        by_space.setdefault(row["space_type"], []).append(row)
    return [
        {"space_type": space, "n_rir": len(items),
         "median_rt60": round(statistics.median(float(r["rt60_sec"]) for r in items), 2)}
        for space, items in sorted(by_space.items())
    ]


# ── Synthetic train/dev (Scaper) ─────────────────────────────────────────────


def synthetic_summary(split: str) -> dict | None:
    index_path = SYNTHETIC_DIR / split / "slice_index.jsonl"
    if not index_path.exists():
        return None
    with index_path.open(encoding="utf-8") as handle:
        clips = [json.loads(line) for line in handle if line.strip()]
    if not clips:
        return None

    slice_counts: dict[str, int] = {}
    for clip in clips:
        for slice_name in clip.get("slices", []):
            slice_counts[slice_name] = slice_counts.get(slice_name, 0) + 1
    n_silent = sum(1 for c in clips if c["n_events"] == 0)

    return {
        "split": split,
        "n_clips": len(clips),
        "hours": round(len(clips) * 10.0 / 3600, 1),
        "n_silent": n_silent,
        "slice_counts": slice_counts,
    }


# ── Dev/gold_test — audio thật, strong label ────────────────────────────────


def real_audio_table(splits_rows: list[dict], raw_index: dict[str, dict]) -> list[dict]:
    table = []
    for split in REAL_AUDIO_SPLITS:
        file_ids = [r["file_id"] for r in splits_rows if r["split"] == split]
        minutes = sum(float(raw_index[fid]["duration"]) for fid in file_ids if fid in raw_index) / 60
        n_classes = len({raw_index[fid]["claimed_class"] for fid in file_ids if fid in raw_index})
        table.append({"split": split, "n_clips": len(file_ids),
                      "minutes": round(minutes, 1), "n_classes": n_classes})
    return table


# ── Ghi Markdown ─────────────────────────────────────────────────────────────


def render(fg: list[dict], bg: list[dict], rir: list[dict],
           synth: list[dict], real_audio: list[dict]) -> str:
    lines = [
        "# data_inventory.md — Tổng kết dữ liệu (sinh tự động)",
        "",
        "> **KHÔNG sửa tay.** Chạy `.venv/Scripts/python.exe scripts/data_inventory.py` để sinh lại.",
        f"> Snapshot UTC: {datetime.now(UTC).isoformat(timespec='seconds')}. Xem [STATUS.md](STATUS.md).",
        "> Bảng bank đọc manifest; synthetic đọc slice_index, không chứng nhận audio/slice đạt QA.",
        "> Real dev/gold đọc splits.csv + raw_manifest.csv; AudioSet 937 WAV/segments đã tải rồi dừng 17/09 nhưng chưa nhập manifest nên không được tính.",
        "> Taxonomy 16 lớp, SED 15 đầu ra; ambient_noise không cần foreground. Shout_yell giữ 56/80 theo ADR-0005.",
        "",
        "## Foreground bank",
        "",
        "| Lớp | Nhóm | Clip | Phút | Mục tiêu | Tối thiểu | Trạng thái | Nguồn |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in sorted(fg, key=lambda r: (r["group"], r["class_id"])):
        lines.append(f"| `{row['class_id']}` | {row['group']} | {row['n_clips']} | {row['minutes']} | "
                     f"{row['target']} | {row['minimum']} | {row['status']} | {', '.join(row['sources'])} |")
    total_fg_clips = sum(r["n_clips"] for r in fg)
    total_fg_minutes = sum(r["minutes"] for r in fg)
    lines += ["", f"**Tổng: {total_fg_clips} clip / {total_fg_minutes:.1f} phút.**", ""]

    lines += ["## Background bank", "", "| Khu vực | Clip | Phút |", "|---|---:|---:|"]
    for row in bg:
        lines.append(f"| {row['area_type']} | {row['n_clips']} | {row['minutes']} |")
    lines.append("")

    lines += ["## RIR bank", "", "| Loại phòng | Số RIR | RT60 trung vị (s) |", "|---|---:|---:|"]
    for row in rir:
        lines.append(f"| {row['space_type']} | {row['n_rir']} | {row['median_rt60']} |")
    lines.append("")

    lines += ["## Train/dev tổng hợp (Scaper)", "",
              "Lát cắt dưới đây là kế hoạch, không phải số clip vượt kiểm định hợp đồng. Không có index thì chưa hiện trong bảng.", "",
              "| Tập | Clip | Giờ | Clip im lặng | Lát cắt |", "|---|---:|---:|---:|---|"]
    for row in synth:
        slice_str = ", ".join(f"{k}={v}" for k, v in sorted(row["slice_counts"].items()))
        lines.append(f"| {row['split']} | {row['n_clips']} | {row['hours']} | {row['n_silent']} | {slice_str} |")
    lines.append("")

    lines += ["## dev / gold_test — audio thật", "",
              "| Tập | Clip | Phút | Số lớp có mặt |", "|---|---:|---:|---:|"]
    for row in real_audio:
        lines.append(f"| {row['split']} | {row['n_clips']} | {row['minutes']} | {row['n_classes']} |")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    enable_utf8_output()
    argparse.ArgumentParser(description=__doc__).parse_args()

    ontology = yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    fg = foreground_table(ontology, read_csv(FOREGROUND_MANIFEST))
    bg = background_table(read_csv(BACKGROUND_MANIFEST))
    rir = rir_table(read_csv(RIR_MANIFEST))
    synth = [s for s in (synthetic_summary(split) for split in ("train", "dev")) if s]
    raw_index = {row["file_id"]: row for row in read_csv(RAW_MANIFEST)}
    real_audio = real_audio_table(read_csv(SPLITS_PATH), raw_index)

    OUTPUT_PATH.write_text(render(fg, bg, rir, synth, real_audio), encoding="utf-8")
    print(f"✓ {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
