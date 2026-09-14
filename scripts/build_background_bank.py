"""Dựng background bank và sàng lọc nhiễm sự kiện Nhóm A — DATA_PLAN §5.

    python scripts/build_background_bank.py --dry-run
    python scripts/build_background_bank.py
    python scripts/build_background_bank.py --report-only

Yêu cầu duy nhất nhưng nghiêm ngặt: nền KHÔNG được chứa sự kiện Nhóm A. Một tiếng
chuông lọt vào nền sẽ được Scaper gán nhãn "không có sự kiện", và model học sai một
cách hệ thống — sai theo hướng bỏ sót, tức đúng hướng nguy hiểm nhất cho giám sát.

Vì vậy ngưỡng nhiễm đặt ở 0.15, thấp hơn hẳn ngưỡng 0.30 của foreground: ở đây thà
báo nhầm nhiều còn hơn lọt một.

Đầu ra:
  data/manifests/background_manifest.csv   mọi đoạn đã xét, kèm điểm nhiễm
  data/banks/background/<area_type>/       đoạn đã sạch
  review_flags.csv (stage=background)      đoạn bị cờ, chờ người nghe
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

from common import REPO_ROOT, enable_utf8_output

CONFIG_PATH = REPO_ROOT / "ml" / "configs" / "background_map.yaml"
PREPROCESS_PATH = REPO_ROOT / "ml" / "configs" / "preprocessing.yaml"
ONTOLOGY_PATH = REPO_ROOT / "ml" / "configs" / "ontology_map.yaml"
MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "background_manifest.csv"
INTERIM_DIR = REPO_ROOT / "data" / "interim" / "background"
BANK_DIR = REPO_ROOT / "data" / "banks" / "background"

TAU_ROOT = REPO_ROOT / "data" / "raw" / "tau2019_partial" / "TAU-urban-acoustic-scenes-2019-development"
US8K_ROOT = REPO_ROOT / "data" / "raw" / "urbansound8k" / "UrbanSound8K"
SAMPLE_RATE = 32000          # PANNs — giống auto_screen.py

FIELDS = ["file_id", "area_type", "source_dataset", "source_id", "source_group_id",
          "path_raw", "duration", "worst_class", "worst_prob", "verdict"]


# ── Liệt kê ứng viên ─────────────────────────────────────────────────────────


def tau_candidates(config: dict) -> list[dict]:
    """Clip TAU2019 thuộc các cảnh đã ánh xạ. meta.csv phân tách bằng TAB, không phải dấu phẩy."""
    meta = TAU_ROOT / "meta.csv"
    if not meta.exists():
        return []
    mapping = config.get("tau2019_partial", {})
    rows = []
    with meta.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            area = mapping.get(row["scene_label"])
            if not area:
                continue
            path = TAU_ROOT / row["filename"]
            if not path.exists():          # mới tải 1/21 zip nên phần lớn chưa có
                continue
            # Tên file dạng `scene-thanhpho-<id địa điểm>-<id đoạn>-<thiết bị>.wav`.
            # Cùng một ĐỊA ĐIỂM thu ra nhiều đoạn → đó là nhóm chống rò rỉ (N2).
            parts = Path(row["filename"]).stem.split("-")
            location = "-".join(parts[:3]) if len(parts) >= 3 else parts[0]
            rows.append({
                "area_type": area,
                "source_dataset": "tau2019",
                "source_id": Path(row["filename"]).stem,
                "source_group_id": f"tau_{location}",
                "path_raw": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
            })
    return rows


def us8k_candidates(config: dict) -> list[dict]:
    """Clip UrbanSound8K có salience=2 thuộc các lớp đã ánh xạ.

    salience=2 nghĩa là lớp được gán nằm ở HẬU CẢNH — tức phần nổi trội của clip chính
    là nền. Đó đúng là thứ background bank cần, và những clip này đã bị adapter
    foreground loại ra với mã `background_salience`.
    """
    meta = US8K_ROOT / "metadata" / "UrbanSound8K.csv"
    if not meta.exists():
        return []
    mapping = config.get("urbansound8k", {})
    rows = []
    with meta.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            area = mapping.get(row["class"])
            if not area or row["salience"] != "2":
                continue
            path = US8K_ROOT / "audio" / f"fold{row['fold']}" / row["slice_file_name"]
            if not path.exists():
                continue
            rows.append({
                "area_type": area,
                "source_dataset": "urbansound8k",
                "source_id": row["slice_file_name"],
                "source_group_id": f"freesound_{row['fsID']}",
                "path_raw": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
            })
    return rows


def candidates(config: dict) -> list[dict]:
    rows = tau_candidates(config) + us8k_candidates(config)
    for index, row in enumerate(rows):
        row["file_id"] = f"bg_{row['area_type']}_{row['source_dataset']}_{index:05d}"
    return rows


# ── Sàng lọc nhiễm ───────────────────────────────────────────────────────────


def group_a_indices(ontology: dict, index_of: dict[str, int]) -> dict[str, list[int]]:
    """{lớp Nhóm A: chỉ số cột PANNs}. Nhóm A là các lớp sự kiện an ninh.

    Lấy TỪ ontology_map.yaml chứ không liệt kê tay: thêm một lớp Nhóm A mới mà quên
    cập nhật bộ sàng lọc nền sẽ để lọt đúng lớp vừa thêm.
    """
    result = {}
    for class_id, spec in ontology["classes"].items():
        if spec.get("group") != "A":
            continue
        found = [index_of[mid] for mid in spec.get("audioset_ids", []) if mid in index_of]
        if found:
            result[class_id] = found
    return result


def worst_contamination(frame_probs: np.ndarray, group_a: dict[str, list[int]]) -> tuple[str, float]:
    """(lớp Nhóm A nhiễm nặng nhất, xác suất đỉnh của nó trên MỌI frame).

    Lấy đỉnh trên toàn bộ frame chứ không lấy trung bình: một tiếng chuông 0.5 giây
    trong đoạn nền 10 giây chỉ làm trung bình nhích lên vài phần trăm, nhưng nó vẫn
    đủ để Scaper gán nhãn sai cho cả clip.
    """
    worst_name, worst = "", 0.0
    for class_id, indices in group_a.items():
        peak = float(frame_probs[:, indices].max()) if frame_probs.size else 0.0
        if peak > worst:
            worst_name, worst = class_id, peak
    return worst_name, worst


def verdict_for(worst_prob: float, threshold: float) -> str:
    return "flagged" if worst_prob > threshold else "clean"


# ── Báo cáo ──────────────────────────────────────────────────────────────────


def report(rows: list[dict], config: dict) -> bool:
    from collections import Counter

    print(f"\n{'khu vực':<14}{'đoạn':>7}{'sạch':>7}{'cờ':>6}{'phút':>8}   nguồn")
    print("─" * 74)
    ok = True
    for area in config["area_types"]:
        subset = [r for r in rows if r["area_type"] == area]
        clean = [r for r in subset if r["verdict"] == "clean"]
        minutes = sum(float(r["duration"]) for r in clean) / 60
        sources = ", ".join(sorted({r["source_dataset"] for r in subset})) or "—"
        mark = "" if len(clean) >= config["min_clips_per_area"] else "  ⚠️"
        print(f"{area:<14}{len(subset):>7}{len(clean):>7}{len(subset) - len(clean):>6}{minutes:>8.1f}   {sources}{mark}")
        if len(clean) < config["min_clips_per_area"]:
            ok = False

    flagged = [r for r in rows if r["verdict"] == "flagged"]
    if flagged:
        worst = Counter(r["worst_class"] for r in flagged)
        print(f"\n{len(flagged)} đoạn bị cờ nhiễm — lớp gây cờ nhiều nhất:")
        for name, count in worst.most_common(5):
            print(f"    {count:>4}  {name}")
        print("   Người phải nghe các đoạn này: cắt bỏ phần nhiễm, hoặc loại cả đoạn.")

    if not ok:
        print(f"\n⚠️ Có khu vực dưới {config['min_clips_per_area']} đoạn sạch.")
        print("   `factory` không có trong TAU2019 — chỉ buổi thu thực địa tại IUH mới lấp được.")
    return ok


# ── Chạy ─────────────────────────────────────────────────────────────────────


def write_manifest(rows: list[dict]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["area_type"], r["file_id"])))


def read_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true", help="chỉ liệt kê ứng viên, không chạy mô hình")
    parser.add_argument("--report-only", action="store_true", help="đọc lại manifest đã có")
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    if args.report_only:
        rows = read_manifest()
        if not rows:
            print("chưa có background_manifest.csv — chạy không kèm --report-only trước")
            return 1
        return 0 if report(rows, config) else 1

    rows = candidates(config)
    if not rows:
        print("❌ không có ứng viên nào. Đã tải tau2019_partial và urbansound8k chưa?")
        return 1
    rows = rows[: args.limit] if args.limit else rows
    print(f"▶ {len(rows)} đoạn ứng viên từ {len({r['source_dataset'] for r in rows})} nguồn")

    if args.dry_run:
        for area in config["area_types"]:
            count = sum(1 for r in rows if r["area_type"] == area)
            print(f"    {area:<14}{count:>6}")
        print("\n(--dry-run: chưa chuẩn hoá, chưa sàng lọc)")
        return 0

    # Nền và tiền cảnh phải qua ĐÚNG một quy trình chuẩn hoá — lệch thì tham số SNR
    # của Scaper không còn nghĩa gì.
    import librosa
    import soundfile

    import auto_screen as asc
    from normalize_audio import process_row

    preprocess = yaml.safe_load(PREPROCESS_PATH.read_text(encoding="utf-8"))
    ontology = yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    _, detector, _ = asc.load_tagger()
    from panns_inference.config import labels
    index_of = asc.mid_to_index(list(labels), REPO_ROOT / "data" / "reference" / "audioset_ontology.json")
    group_a = group_a_indices(ontology, index_of)
    print(f"  sàng lọc {len(group_a)} lớp Nhóm A, ngưỡng nhiễm {config['contamination_threshold']}")

    threshold = float(config["contamination_threshold"])
    kept, dropped = [], 0
    for number, row in enumerate(rows, 1):
        if number % 200 == 0:
            print(f"  {number}/{len(rows)}…", flush=True)
        rejection, _ = process_row({**row, "claimed_class": row["area_type"]},
                                   preprocess, dry_run=False,
                                   out_dir=INTERIM_DIR, subdir_key="area_type")
        if rejection:
            dropped += 1
            continue

        normalized = INTERIM_DIR / row["area_type"] / f"{row['file_id']}.wav"
        audio, _ = librosa.load(str(normalized), sr=SAMPLE_RATE, mono=True)
        frame_probs = detector.inference(asc.pad_to_minimum(audio)[None, :])[0]
        worst_class, worst_prob = worst_contamination(frame_probs, group_a)

        row["duration"] = round(len(audio) / SAMPLE_RATE, 3)
        row["worst_class"] = worst_class
        row["worst_prob"] = round(worst_prob, 4)
        row["verdict"] = verdict_for(worst_prob, threshold)
        if row["verdict"] == "clean":
            destination = BANK_DIR / row["area_type"] / f"{row['file_id']}.wav"
            destination.parent.mkdir(parents=True, exist_ok=True)
            soundfile.write(destination, *soundfile.read(normalized))
        kept.append(row)

    if dropped:
        print(f"  {dropped} đoạn không qua cổng chất lượng (cùng cổng với foreground)")
    write_manifest(kept)
    ok = report(kept, config)
    print(f"\n✓ {MANIFEST_PATH.relative_to(REPO_ROOT)} · {BANK_DIR.relative_to(REPO_ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
