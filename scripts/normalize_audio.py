"""Giai đoạn 1 — chuẩn hoá kỹ thuật (DATA_PLAN §4.1). Hoàn toàn tự động.

    python scripts/normalize_audio.py --source esc50
    python scripts/normalize_audio.py --source esc50 --limit 20 --dry-run

data/raw/  →  [cổng chất lượng]  →  16 kHz mono, EBU R128  →  data/interim/normalized/
                     ↓ trượt
              data/manifests/exclusions.csv   (N3: mọi file bị loại đều có lý do)

Hai điều dễ làm sai:

  · **Chuẩn hoá độ to, không phải chuẩn hoá đỉnh.** Peak normalize chia cả clip cho
    đúng một mẫu to nhất, nên một tiếng lách tách cũng dìm cả clip xuống. Với dữ liệu
    an ninh thì đó là thảm hoạ: tiếng súng và tiếng kính vỡ vốn có đỉnh nhọn.
  · **Máy chỉ được loại theo tiêu chí ĐO ĐƯỢC** (im lặng, méo, quá ngắn, nâng mẫu).
    Những thứ cần tai người — sai lớp, nhiều sự kiện, lẫn nhạc — không thuộc bước này.
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm
import soundfile
import yaml

from common import REPO_ROOT, enable_utf8_output, write_exclusions

MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "raw_manifest.csv"
REVIEW_PATH = REPO_ROOT / "data" / "manifests" / "review_flags.csv"
NORMALIZED_DIR = REPO_ROOT / "data" / "interim" / "normalized"
CONFIG_PATH = REPO_ROOT / "ml" / "configs" / "preprocessing.yaml"

STAGE = "normalize"
REVIEW_FIELDS = ["file_id", "stage", "flag", "value", "detail"]

# pyloudnorm cảnh báo "possible clipped samples" NGAY SAU khi chỉnh gain, tức là
# trước khi ta hạ trần ở apply_peak_ceiling(). Ta đã xử lý việc đó nên tắt cảnh báo,
# nếu không 300 dòng cảnh báo sẽ che mất báo cáo thật.
warnings.filterwarnings("ignore", message="Possible clipped samples in output")


@dataclass(frozen=True)
class Rejection:
    reason_code: str
    detail: str


@dataclass(frozen=True)
class ReviewFlag:
    """Không đủ tệ để máy tự loại, nhưng người phải nghe trước khi cho vào bank."""

    flag: str
    value: float
    detail: str


# ── Phát hiện méo ────────────────────────────────────────────────────────────


def flat_top_ratio(audio: np.ndarray, min_run: int, ceiling: float) -> float:
    """Tỉ lệ mẫu nằm trong các đoạn đỉnh bị cắt phẳng — dấu hiệu méo thật sự.

    Phải gọi trên TÍN HIỆU GỐC, trước khi resample. Đo bằng cách đếm mẫu chạm trần
    sẽ thổi phồng (resampler tự vượt đỉnh), còn đo đoạn phẳng sau khi resample lại
    bỏ sót (phép nội suy làm mượt đỉnh phẳng thành cong). Số liệu ở preprocessing.yaml.
    """
    at_ceiling = np.abs(audio) >= ceiling
    if not at_ceiling.any():
        return 0.0

    padded = np.concatenate(([0], at_ceiling.view(np.int8), [0]))
    edges = np.diff(padded)
    run_lengths = np.where(edges == -1)[0] - np.where(edges == 1)[0]
    return float(run_lengths[run_lengths >= min_run].sum()) / len(audio)


# ── Cổng chất lượng ──────────────────────────────────────────────────────────


def check_quality(
    audio: np.ndarray, sample_rate: int, original_rate: int, clipped_ratio: float, gates: dict
) -> tuple[Rejection | None, ReviewFlag | None]:
    """(lý do loại, cờ cần người xem). Máy chỉ dùng tiêu chí đo được."""
    duration = len(audio) / sample_rate

    if duration < gates["min_duration_sec"]:
        return Rejection("too_short", f"{duration:.2f}s < {gates['min_duration_sec']}s"), None
    if duration > gates["max_duration_sec"]:
        return Rejection("too_long", f"{duration:.2f}s > {gates['max_duration_sec']}s"), None
    if original_rate < gates["min_original_sample_rate"]:
        return Rejection("upsampled", f"sample rate gốc {original_rate} Hz"), None

    rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
    if rms < gates["silence_rms_threshold"]:
        return Rejection("silent", f"RMS {rms:.6f}"), None

    if clipped_ratio > gates["max_clipped_ratio"]:
        return Rejection("clipped", f"{clipped_ratio:.2%} đoạn cắt phẳng"), None
    if clipped_ratio > gates["clipping_review_ratio"]:
        return None, ReviewFlag("clipping", clipped_ratio, f"{clipped_ratio:.3%} đoạn cắt phẳng — nghe kiểm tra")

    return None, None


# ── Biến đổi ─────────────────────────────────────────────────────────────────


def load_mono(path: Path, target_rate: int, resample_type: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Đọc file → (tín hiệu đã resample, tín hiệu GỐC mono, sample rate GỐC).

    Trả về cả bản gốc vì hai phép kiểm tra chỉ đúng khi đo trước resample:
    sample rate gốc (phát hiện nâng mẫu) và đoạn cắt phẳng (phát hiện méo).
    """
    original, original_rate = soundfile.read(str(path), always_2d=True)
    original_mono = original.mean(axis=1)
    audio, _ = librosa.load(str(path), sr=target_rate, mono=True, res_type=resample_type)
    return audio, original_mono, original_rate


def normalize_loudness(audio: np.ndarray, sample_rate: int, settings: dict) -> tuple[np.ndarray, float | None]:
    """Chuẩn hoá về target LUFS. Trả về (tín hiệu, LUFS đo được trước khi chỉnh)."""
    duration = len(audio) / sample_rate
    if duration < settings["min_duration_for_lufs"]:
        return audio, None

    meter = pyloudnorm.Meter(sample_rate)
    measured = meter.integrated_loudness(audio)
    if not np.isfinite(measured):
        return audio, None

    adjusted = pyloudnorm.normalize.loudness(audio, measured, settings["target_lufs"])
    return apply_peak_ceiling(adjusted, settings["peak_ceiling_dbfs"]), float(measured)


def apply_peak_ceiling(audio: np.ndarray, ceiling_dbfs: float) -> np.ndarray:
    """Hạ gain nếu chỉnh độ to làm tín hiệu vượt trần. Không bao giờ nâng gain."""
    ceiling = 10 ** (ceiling_dbfs / 20)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return audio * (ceiling / peak) if peak > ceiling else audio


# ── Xử lý một file ───────────────────────────────────────────────────────────


def process_row(row: dict, config: dict, dry_run: bool,
                out_dir: Path | None = None, subdir_key: str = "claimed_class",
                ) -> tuple[Rejection | None, ReviewFlag | None]:
    """Chuẩn hoá một clip. `out_dir`/`subdir_key` để background bank dùng lại được
    cùng bộ cổng chất lượng và cùng phép chuẩn hoá độ to — nền và tiền cảnh phải qua
    ĐÚNG một quy trình, nếu lệch thì Scaper trộn hai thứ khác mức độ to với nhau và
    tham số SNR trong config không còn nghĩa gì."""
    source_path = REPO_ROOT / row["path_raw"]
    if not source_path.exists():
        return Rejection("missing_file", row["path_raw"]), None

    audio_settings = config["audio"]
    gates = config["quality_gates"]
    audio, original_mono, original_rate = load_mono(
        source_path, audio_settings["target_sample_rate"], audio_settings["resample_type"]
    )

    # Đo méo trên tín hiệu GỐC — sau resample thì dấu vết đã bị làm mượt.
    clipped_ratio = flat_top_ratio(
        original_mono, gates["clipping_min_run_samples"], gates["clipping_ceiling"]
    )
    rejection, flag = check_quality(
        audio, audio_settings["target_sample_rate"], original_rate, clipped_ratio, gates
    )
    if rejection:
        return rejection, None

    audio, _ = normalize_loudness(audio, audio_settings["target_sample_rate"], config["loudness"])
    if dry_run:
        return None, flag

    destination = (out_dir or NORMALIZED_DIR) / row[subdir_key] / f"{row['file_id']}.wav"
    destination.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(destination, audio, audio_settings["target_sample_rate"], subtype="PCM_16")
    return None, flag


# ── Ghi loại bỏ ──────────────────────────────────────────────────────────────


def write_review_flags(flags: dict[str, ReviewFlag], processed: set[str], source_prefix: str = "") -> None:
    """Hàng đợi cho người nghe ở giai đoạn 3 — không phải danh sách loại bỏ.

    `file_id in processed` không bắt được dòng CŨ mang class_id đã mất (ví dụ sau khi
    tách một lớp): file_id đổi theo class_id, nên nó không còn khớp bất cứ gì trong
    lần quét mới và nằm lại vĩnh viễn. Với `source_prefix`, bất kỳ dòng cùng nguồn mà
    không nằm trong `processed` — tức đã bị đổi tên/xoá — cũng bị dọn theo.
    """
    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    kept: list[dict] = []
    if REVIEW_PATH.exists():
        with REVIEW_PATH.open(encoding="utf-8", newline="") as handle:
            kept = [r for r in csv.DictReader(handle)
                    if not (r["stage"] == STAGE and (r["file_id"] in processed
                            or (source_prefix and r["file_id"].startswith(source_prefix))))]
    with REVIEW_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(kept)
        for file_id, flag in sorted(flags.items(), key=lambda kv: -kv[1].value):
            writer.writerow({
                "file_id": file_id, "stage": STAGE,
                "flag": flag.flag, "value": f"{flag.value:.6f}", "detail": flag.detail,
            })


def report(total: int, rejections: dict[str, Rejection], flags: dict[str, ReviewFlag], dry_run: bool) -> None:
    kept = total - len(rejections)
    print(f"\n{'(chạy thử) ' if dry_run else ''}giữ {kept}/{total} clip ({kept / total:.1%})"
          f" · {len(flags)} clip cần người nghe kiểm tra")

    if not rejections:
        print("Không clip nào bị loại ở cổng kỹ thuật.")
        return

    counts: dict[str, int] = {}
    for rejection in rejections.values():
        counts[rejection.reason_code] = counts.get(rejection.reason_code, 0) + 1

    print(f"\n{'lý do loại':<20}{'số clip':>9}")
    print("─" * 30)
    for reason, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"{reason:<20}{count:>9}")

    examples = list(rejections.items())[:3]
    print("\nví dụ bị loại: " + " · ".join(f"{fid} ({r.reason_code}: {r.detail})" for fid, r in examples))
    if flags:
        top = sorted(flags.items(), key=lambda kv: -kv[1].value)[:3]
        print("nặng nhất trong hàng đợi người nghe: "
              + " · ".join(f"{fid} ({f.detail})" for fid, f in top))


def main() -> int:
    enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", help="chỉ xử lý nguồn này (mặc định: tất cả)")
    parser.add_argument("--limit", type=int, help="chỉ xử lý N file đầu")
    parser.add_argument("--dry-run", action="store_true", help="chỉ báo cáo, không ghi file")
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if not args.source or r["source_dataset"] == args.source]
    rows = rows[: args.limit]

    if not rows:
        print("Manifest không có dòng nào khớp. Chạy build_manifest.py trước.")
        return 1

    print(f"▶ chuẩn hoá {len(rows)} clip → {config['audio']['target_sample_rate']} Hz mono, "
          f"{config['loudness']['target_lufs']} LUFS")

    rejections: dict[str, Rejection] = {}
    flags: dict[str, ReviewFlag] = {}
    for index, row in enumerate(rows, 1):
        if index % 100 == 0:
            print(f"  {index}/{len(rows)}…", flush=True)
        rejection, flag = process_row(row, config, args.dry_run)
        if rejection:
            rejections[row["file_id"]] = rejection
        elif flag:
            flags[row["file_id"]] = flag

    if not args.dry_run:
        processed = {r["file_id"] for r in rows}
        # Chỉ dọn dòng mồ côi (class đã đổi tên) khi quét TOÀN BỘ nguồn, không --limit —
        # xem drop_stale_rows trong build_manifest.py cho lý do đầy đủ.
        source_prefix = f"{args.source}_" if args.source and args.limit is None else ""
        write_exclusions(STAGE, {fid: (r.reason_code, r.detail) for fid, r in rejections.items()}, processed)
        write_review_flags(flags, processed, source_prefix)
    report(len(rows), rejections, flags, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
