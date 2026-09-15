"""Tải bộ vehicle_crash_cc (HuggingFace, CC-BY) — thay thế MIVIA Road cho lớp `vehicle_crash`.

    python scripts/fetch_vehicle_crash_cc.py

Nguồn: https://huggingface.co/datasets/Titung/car-crash-audio-cc — 46 clip cắt từ 38 video
YouTube, license CC-BY 3.0. Không dùng `download_sources.py` vì đây không phải MỘT archive:
`metadata.csv` liệt kê nhiều file lẻ, mỗi file tải qua URL `resolve/main/<file_name>` riêng.

Ba điều dễ làm sai với nguồn này:
  1. `video_id` trong metadata.csv PHẢI dùng làm `source_group_id` — vài video góp nhiều
     clip (hậu tố `_00`, `_01`... trong `file_name`), chia theo file là rò rỉ (N2).
  2. `label` trong metadata.csv luôn là "Car Crash" (không có lớp nào khác) — không cần
     ánh xạ qua ontology_map.yaml như các nguồn nhiều lớp khác.
  3. `start_sec`/`end_sec` là mốc trong VIDEO GỐC, không phải trong file đã tải — file đã
     tải VỐN ĐÃ là đúng đoạn đó, nên trong manifest onset=0/offset=duration (như DESED
     soundbank), không phải orig_onset=start_sec.
"""

from __future__ import annotations

import csv
import io
import sys
import urllib.request
from pathlib import Path

from common import REPO_ROOT, enable_utf8_output

BASE_URL = "https://huggingface.co/datasets/Titung/car-crash-audio-cc/resolve/main/"
RAW_DIR = REPO_ROOT / "data" / "raw" / "vehicle_crash_cc"
METADATA_PATH = RAW_DIR / "metadata.csv"


def download_metadata() -> list[dict]:
    with urllib.request.urlopen(BASE_URL + "metadata.csv", timeout=60) as response:
        text = response.read().decode("utf-8")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(text, encoding="utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def download_clip(file_name: str) -> Path:
    dest = RAW_DIR / file_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with urllib.request.urlopen(BASE_URL + file_name, timeout=60) as response:
            dest.write_bytes(response.read())
    return dest


def run() -> int:
    rows = download_metadata()
    print(f"▶ {len(rows)} clip trong metadata.csv")

    ok = 0
    for row in rows:
        try:
            download_clip(row["file_name"])
            ok += 1
        except OSError as error:
            print(f"  ⚠️ {row['file_name']}: {error}")

    n_videos = len({row["video_id"] for row in rows})
    print(f"✓ tải {ok}/{len(rows)} clip từ {n_videos} video gốc → {RAW_DIR.relative_to(REPO_ROOT)}")
    return 0 if ok == len(rows) else 1


def main() -> int:
    enable_utf8_output()
    return run()


if __name__ == "__main__":
    sys.exit(main())
