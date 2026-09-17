"""Dataset đọc memmap đã precompute — không đụng tới file .wav trong lúc huấn luyện."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.datasets.synthetic_sed import clip_targets, frame_targets
from ml.features.precompute_waveform import INT16_SCALE


class PrecomputedSedDataset(Dataset):
    """(sóng âm float32, nhãn khung, nhãn clip) từ memmap + meta.json.

    `n_frames` phải khớp độ dài đầu ra của model. Đặt ở đây thay vì hardcode: CNN14 ở
    10 s/32 kHz/hop 320 cho 1001 khung, còn BEATs sẽ cho con số khác hẳn.
    """

    def __init__(self, features_dir: Path, split: str, n_frames: int, sample_rate: int = 32000,
                 indices: np.ndarray | None = None):
        meta = json.loads((features_dir / f"{split}_meta.json").read_text(encoding="utf-8"))
        self.class_ids: list[str] = meta["class_ids"]
        self.duration: float = float(meta["duration"])
        self.n_frames = n_frames
        self.clips = meta["clips"]

        # mmap_mode="r": không nạp 4 GB vào RAM, để hệ điều hành lo phần cache trang.
        self.waves = np.load(features_dir / f"{split}_wave{sample_rate // 1000}k.npy", mmap_mode="r")
        if len(self.waves) != len(self.clips):
            raise ValueError(
                f"memmap có {len(self.waves)} dòng nhưng meta có {len(self.clips)} clip — "
                "hai file lệch nhau, phải chạy lại precompute_waveform."
            )
        self.indices = np.arange(len(self.clips)) if indices is None else indices

    def __len__(self) -> int:
        return len(self.indices)

    def events_of(self, position: int) -> list[tuple[int, float, float]]:
        row = self.clips[self.indices[position]]
        return [(int(c), float(on), float(off)) for c, on, off in row["events"]]

    def clip_id(self, position: int) -> str:
        return self.clips[self.indices[position]]["clip_id"]

    def __getitem__(self, position: int) -> dict:
        index = self.indices[position]
        waveform = np.asarray(self.waves[index], dtype=np.float32) / INT16_SCALE
        events = self.events_of(position)
        n_classes = len(self.class_ids)
        return {
            "waveform": torch.from_numpy(waveform),
            "frame_target": torch.from_numpy(
                frame_targets(events, self.n_frames, n_classes, self.duration)
            ),
            "clip_target": torch.from_numpy(clip_targets(events, n_classes)),
            "index": position,
        }


def split_indices(n: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Tách validation TẠM THỜI từ chính tập train synthetic.

    ⚠️ Đây KHÔNG phải tập dev thật của đề tài. `splits.yaml` định nghĩa dev là audio THẬT
    (audioset_strong / desed_real / mivia); tập tách ở đây dùng chung bank nguồn với train
    nên một clip ESC-50 có thể xuất hiện ở cả hai bên — điểm sẽ CAO HƠN thực tế. Nó chỉ
    dùng để dừng sớm và chọn siêu tham số khi bộ synthetic dev chưa sinh xong, và con số
    trên nó KHÔNG được đưa vào khoá luận như kết quả đánh giá.
    """
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    n_val = max(1, int(round(n * val_ratio)))
    return permutation[n_val:], permutation[:n_val]
