"""Bọc PANNs CNN14 — phần thực sự gọi torch.

Tách khỏi `tagger.py` có chủ đích: mọi logic (ánh xạ nhãn, đệm clip, đề xuất biên) nằm ở
`tagger.py` và test được mà không cần nạp 300 MB checkpoint. File này chỉ lo nạp model và
đẩy sóng âm qua nó.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import numpy as np

from app import tagger

logger = logging.getLogger(__name__)

ONTOLOGY_JSON = Path(os.environ.get("AUDIOSET_ONTOLOGY", "/config/audioset_ontology.json"))
TAXONOMY_YAML = Path(os.environ.get("ONTOLOGY_PATH", "/config/ontology_map.yaml"))

_sed = None
_class_indices: dict[str, list[int]] = {}


def load_model() -> None:
    global _sed, _class_indices
    if _sed is not None:
        return

    import yaml

    from app.bootstrap import ensure_panns_data

    # PHẢI gọi trước khi import panns_inference: module đó đọc file nhãn ngay lúc import
    # và sẽ chết nếu thiếu (xem app/bootstrap.py).
    checkpoint = ensure_panns_data()

    from panns_inference import SoundEventDetection
    from panns_inference.config import labels

    logger.info("nap PANNs CNN14 (CPU)...")
    _sed = SoundEventDetection(checkpoint_path=str(checkpoint), device="cpu")

    index_of = tagger.mid_to_index(list(labels), ONTOLOGY_JSON)
    taxonomy = yaml.safe_load(TAXONOMY_YAML.read_text(encoding="utf-8"))
    _class_indices = tagger.class_to_indices(taxonomy, index_of)
    logger.info(
        "ghep %d/%d nhan PANNs; %d/%d lop cua ta co ma khop",
        len(index_of), len(labels), len(_class_indices), len(taxonomy["classes"]),
    )


def is_ready() -> bool:
    return _sed is not None


def read_wav(audio_bytes: bytes) -> tuple[np.ndarray, float]:
    """Đọc wav → mono float32 @32 kHz + thời lượng THẬT (giây).

    Thời lượng tính TRƯỚC khi resample/đệm: nhãn onset/offset phải tính theo audio gốc
    người dùng gửi lên, không phải theo tensor đã bị kéo dài để vừa model.
    """
    import librosa

    audio, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=tagger.SAMPLE_RATE, mono=True)
    duration = float(len(audio)) / tagger.SAMPLE_RATE
    return audio.astype(np.float32), duration


def detect(audio_bytes: bytes) -> tuple[list[tagger.Detection], float]:
    if _sed is None:
        raise RuntimeError("PANNs chưa nạp xong")

    audio, duration = read_wav(audio_bytes)
    if duration <= 0:
        return [], 0.0

    model_input = tagger.pad_to_minimum(audio)
    framewise = _sed.inference(model_input[None, :])[0]   # (n_frames, 527)

    # Chỉ đọc phần khung ứng với audio THẬT: phần còn lại là bản lặp do đệm, dùng nó sẽ
    # cho ra biên sự kiện nằm ngoài độ dài clip gốc.
    real_frames = max(1, int(round(framewise.shape[0] * min(1.0, duration * tagger.SAMPLE_RATE / len(model_input)))))
    framewise = framewise[:real_frames]

    detections = tagger.detections_from_frames(framewise, _class_indices, duration)
    return detections, duration
