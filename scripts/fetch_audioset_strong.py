"""Tải nhãn strong-label của AudioSet và audio tương ứng qua yt-dlp — DATA_PLAN §6.

    python scripts/fetch_audioset_strong.py --labels-only     # chỉ tải 3 file TSV, vài trăm KB
    python scripts/fetch_audioset_strong.py                   # + tải audio (cần yt-dlp, ffmpeg)
    python scripts/fetch_audioset_strong.py --limit 20         # chạy thử

Đây là nguồn DUY NHẤT cho `dev`/`gold_test` không phụ thuộc MIVIA (đang chờ duyệt) hay
buổi thu thực địa IUH (chưa diễn ra). Không có nguồn này, cổng D7 (gold_test rỗng) không
qua được và không đo được EHR/EOR/GS/TOA/CHR — tức mất luôn đóng góp M5 của khoá luận.

Ba điều khác biệt so với các nguồn khác trong sources.yaml:
  1. Nhãn là STRONG THẬT (onset/offset theo giây), nhưng đơn vị audio ta tải là một
     Ổ SỔ 10 GIÂY (segment_id = f"{ytid}_{window_start_ms}"), và MỘT ổ có thể chứa
     NHIỀU sự kiện — khác các adapter khác vốn một file = một sự kiện. Vì vậy kết quả
     ở đây không phải RawEntry trực tiếp: script ghi ra segments.jsonl, để
     build_manifest.adapt_audioset_strong sinh NHIỀU dòng manifest trỏ cùng một file .wav.
  2. Audio thuộc về chủ video YouTube, KHÔNG được phát hành lại — chỉ công bố YTID +
     mốc thời gian (đã ghi trong sources.yaml). Vì vậy segments.jsonl chỉ chứa audio
     tải cho MÁY CỦA TA dùng, không đưa vào data công bố.
  3. 15–30% video dự kiến không tải được (đã gỡ / chặn vùng / private). Đây là hụt dữ
     liệu BÌNH THƯỜNG của nguồn YouTube, không phải lỗi — nhưng PHẢI ghi vào
     exclusions.csv với mã `unavailable` và báo cáo trong khoá luận (N3).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import yaml

from common import REPO_ROOT, enable_utf8_output, write_exclusions

CONFIG_PATH = REPO_ROOT / "ml" / "configs" / "sources.yaml"
ONTOLOGY_PATH = REPO_ROOT / "ml" / "configs" / "ontology_map.yaml"
RAW_ROOT = REPO_ROOT / "data" / "raw" / "audioset_strong"
LABELS_DIR = RAW_ROOT / "labels"
AUDIO_DIR = RAW_ROOT / "audio"
SEGMENTS_PATH = RAW_ROOT / "segments.jsonl"

WINDOW_SECONDS = 10.0          # AudioSet strong nhãn theo cửa sổ 10 giây gốc của AudioSet
STAGE = "fetch_audioset_strong"


# ── Tải 3 file TSV nhãn ───────────────────────────────────────────────────────


def download_labels(urls: list[str]) -> None:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    for url in urls:
        dest = LABELS_DIR / url.rsplit("/", 1)[-1]
        if dest.exists():
            continue
        print(f"  ↓ {dest.name}")
        with urllib.request.urlopen(url, timeout=60) as response:
            dest.write_bytes(response.read())


# ── Phân tích nhãn ───────────────────────────────────────────────────────────


def load_mid_to_display_name(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) >= 2 and row[0].startswith("/"):
                mapping[row[0]] = row[1]
    return mapping


def parse_segment_id(segment_id: str) -> tuple[str, int]:
    """"<ytid>_<window_start_ms>" → (ytid, window_start_ms).

    YTID của YouTube có thể chứa dấu gạch dưới, nên phải tách từ BÊN PHẢI —
    tách từ bên trái cắt nhầm giữa ytid, làm sai luôn cả video lẫn mốc thời gian.
    """
    ytid, _, start_ms = segment_id.rpartition("_")
    return ytid, int(start_ms)


def load_strong_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def mid_to_class_map(ontology: dict, known_mids: set[str]) -> dict[str, str]:
    """{mid: class_id}, giới hạn ở các mid THỰC SỰ có trong bộ vocab strong-label.

    audioset_ids trong ontology_map.yaml gồm cả mid chỉ có ở tập weak (527 lớp) —
    đối chiếu với known_mids (456 lớp strong) trước khi dùng, kẻo lọc ra rỗng mà
    không ai biết vì sao.
    """
    mapping: dict[str, str] = {}
    for class_id, spec in ontology["classes"].items():
        for mid in spec.get("audioset_ids", []):
            if mid in known_mids:
                mapping[mid] = class_id
    return mapping


def select_segments(rows: list[dict], mid_to_class: dict[str, str]) -> dict[tuple[str, int], list[dict]]:
    """{(ytid, window_start_ms): [{class_id, onset, offset}]} — chỉ nhãn PRESENT khớp 16 lớp.

    UNCERTAIN bị loại: đây là nhãn gold dùng để TÍNH ĐIỂM model, lẫn nhãn không chắc vào
    ground truth thì mọi số EHR/EOR đo trên nó đều sai mà không lộ triệu chứng.
    """
    segments: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        if row.get("present") != "PRESENT":
            continue
        class_id = mid_to_class.get(row["label"])
        if class_id is None:
            continue
        ytid, window_start_ms = parse_segment_id(row["segment_id"])
        key = (ytid, window_start_ms)
        segments.setdefault(key, []).append({
            "class_id": class_id,
            "onset": round(float(row["start_time_seconds"]), 3),
            "offset": round(float(row["end_time_seconds"]), 3),
        })
    return segments


# ── Tải audio bằng yt-dlp ────────────────────────────────────────────────────


def download_window(ytid: str, window_start_ms: int, dest: Path) -> bool:
    """Tải đúng 10 giây [start, start+10) của video bằng yt-dlp. True nếu thành công.

    --download-sections cắt TRƯỚC KHI tải hết video — video 30 phút mà chỉ cần
    10 giây, tải cả video là phí băng thông gấp trăm lần không cần thiết.
    """
    start = window_start_ms / 1000
    end = start + WINDOW_SECONDS
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "yt-dlp", "-f", "bestaudio",
            "--download-sections", f"*{start}-{end}",
            "--extract-audio", "--audio-format", "wav",
            "-o", str(dest.with_suffix("")) + ".%(ext)s",
            "--no-playlist", "--quiet", "--no-warnings",
            f"https://www.youtube.com/watch?v={ytid}",
        ],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and dest.exists()


# ── Điều phối ────────────────────────────────────────────────────────────────


def append_segments(new_rows: list[dict]) -> None:
    """Thêm dòng mới, KHÔNG ghi đè — chạy lại script không được xoá kết quả cũ."""
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    existing_keys = set()
    if SEGMENTS_PATH.exists():
        with SEGMENTS_PATH.open(encoding="utf-8") as handle:
            existing_keys = {json.loads(line)["file_id"] for line in handle if line.strip()}
    with SEGMENTS_PATH.open("a", encoding="utf-8") as handle:
        for row in new_rows:
            if row["file_id"] not in existing_keys:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(limit: int | None, labels_only: bool) -> int:
    sources = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["sources"]["audioset_strong"]
    ontology = yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8"))

    download_labels(sources["label_urls"])
    train_path = LABELS_DIR / "audioset_train_strong.tsv"
    eval_path = LABELS_DIR / "audioset_eval_strong.tsv"
    vocab_path = LABELS_DIR / "mid_to_display_name.tsv"

    mid_to_name = load_mid_to_display_name(vocab_path)
    mid_to_class = mid_to_class_map(ontology, set(mid_to_name))
    print(f"  {len(mid_to_class)} mã AudioSet strong khớp với 16 lớp của ta")

    rows = load_strong_rows(train_path) + load_strong_rows(eval_path)
    segments = select_segments(rows, mid_to_class)
    print(f"  {len(segments)} ổ 10 giây có ít nhất một sự kiện thuộc 16 lớp")

    if labels_only:
        return 0

    keys = sorted(segments)[:limit] if limit is not None else sorted(segments)
    ok, failed = 0, {}
    new_rows = []
    for ytid, window_start_ms in keys:
        file_id = f"as_strong_{ytid}_{window_start_ms}"
        dest = AUDIO_DIR / f"{file_id}.wav"
        if not dest.exists():
            if not download_window(ytid, window_start_ms, dest):
                failed[file_id] = ("unavailable", f"yt-dlp thất bại: {ytid}")
                continue
        ok += 1
        new_rows.append({
            "file_id": file_id,
            "path": str(dest.relative_to(REPO_ROOT)).replace("\\", "/"),
            "ytid": ytid,
            "window_start_ms": window_start_ms,
            "events": segments[(ytid, window_start_ms)],
        })

    append_segments(new_rows)
    write_exclusions(STAGE, failed, {f"as_strong_{y}_{w}" for y, w in keys})
    print(f"  tải được {ok}/{len(keys)} — {len(failed)} lỗi ghi vào exclusions.csv")
    return 0


def main() -> int:
    enable_utf8_output()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--labels-only", action="store_true")
    args = parser.parse_args()
    return run(args.limit, args.labels_only)


if __name__ == "__main__":
    sys.exit(main())
