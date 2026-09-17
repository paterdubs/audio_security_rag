"""Nạp trước sóng âm vào một memmap duy nhất — bước `Precompute` của PLAN.md W2.

    .venv/Scripts/python.exe -m ml.features.precompute_waveform --split train

Vì sao cần: một epoch đọc 6500 file .wav rời rạc rồi resample 16k→32k tốn CPU đúng bằng
nhau ở MỌI epoch, mà CPU lúc này còn đang chia cho tiến trình sinh Scaper. Trả trước một
lần rồi đọc memmap khiến epoch sau gần như chỉ còn chi phí GPU.

Lưu int16 chứ không phải float32: nguồn vốn là PCM_16 nên không mất mát gì, mà dung lượng
giảm một nửa (6500 clip × 10 s × 32 kHz: 8,3 GB → 4,2 GB). Ổ D chỉ còn ~24 GB.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Console Windows mặc định cp1252 và sẽ CHẾT ở ký tự "▶" hay tên lớp tiếng Việt — lỗi
# nổ ra ở dòng print đầu tiên, sau khi đã làm hết phần việc tốn thời gian.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from common import enable_utf8_output  # noqa: E402

enable_utf8_output()

from ml.datasets.synthetic_sed import event_class_ids, load_split  # noqa: E402

SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"
FEATURES_DIR = REPO_ROOT / "data" / "features"
ONTOLOGY_PATH = REPO_ROOT / "ml" / "configs" / "ontology_map.yaml"

INT16_SCALE = 32767.0


def precompute(split: str, sample_rate: int, duration: float, limit: int | None) -> int:
    class_ids = event_class_ids(ONTOLOGY_PATH)
    clips = load_split(SYNTHETIC_DIR / split, class_ids, limit=limit)
    if not clips:
        print(f"❌ không có clip nào trong {SYNTHETIC_DIR / split}")
        return 1

    n_samples = int(round(sample_rate * duration))
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    wave_path = FEATURES_DIR / f"{split}_wave{sample_rate // 1000}k.npy"
    meta_path = FEATURES_DIR / f"{split}_meta.json"

    print(f"▶ {split}: {len(clips)} clip × {duration}s @ {sample_rate} Hz "
          f"→ {len(clips) * n_samples * 2 / 1e9:.2f} GB")

    waves = np.lib.format.open_memmap(
        wave_path, mode="w+", dtype=np.int16, shape=(len(clips), n_samples)
    )
    for i, clip in enumerate(clips):
        audio, _ = librosa.load(str(clip.audio_path), sr=sample_rate, mono=True)
        # Clip có thể lệch vài mẫu so với danh nghĩa (Scaper làm tròn) — cắt/đệm về
        # đúng độ dài thay vì để lỗi shape nổ ra lúc ghép batch.
        if len(audio) < n_samples:
            audio = np.pad(audio, (0, n_samples - len(audio)))
        waves[i] = np.clip(audio[:n_samples] * INT16_SCALE, -INT16_SCALE, INT16_SCALE).astype(np.int16)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(clips)}…", flush=True)
    waves.flush()

    meta_path.write_text(
        json.dumps(
            {
                "split": split,
                "sample_rate": sample_rate,
                "duration": duration,
                "class_ids": class_ids,
                "clips": [
                    {"clip_id": c.clip_id, "events": [list(e) for e in c.events]} for c in clips
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"✓ {wave_path.relative_to(REPO_ROOT)} + {meta_path.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--sample-rate", type=int, default=32000,
                        help="32000 cho PANNs (đã pretrain ở tần số này), 16000 cho BEATs")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    return precompute(args.split, args.sample_rate, args.duration, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
