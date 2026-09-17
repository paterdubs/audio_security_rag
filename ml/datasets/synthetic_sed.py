"""Đọc bộ synthetic do Scaper sinh thành dữ liệu huấn luyện SED.

Nhãn lấy từ file .jams — KHÔNG lấy từ .tsv: Scaper ở cấu hình hiện tại không ghi tsv
(thư mục `tsv/` rỗng, 6500 jams / 0 tsv), và .jams mới là thứ chứa đủ onset, offset,
nhãn, SNR lẫn file nguồn. Dựa vào tsv sẽ ra một dataset RỖNG mà không báo lỗi gì.

Hai quyết định về nhãn, ghi ở đây vì chúng thay đổi ý nghĩa của mọi con số đo được:

1. `ambient_noise` KHÔNG nằm trong head. Trong clip Scaper, nền luôn có mặt suốt 10 giây
   nên nhãn này sẽ bật 100% thời gian ở 100% clip — model học được nó trong một bước và
   AP của nó luôn ≈ 1.0, làm mAP tổng bị thổi phồng mà không phản ánh năng lực nào. Nó là
   trạng thái "không có sự kiện", không phải một sự kiện. Bank foreground cũng chỉ có
   đúng 15 thư mục, không có ambient_noise — hai chỗ khớp nhau.

2. Chỉ lấy observation có `role == "foreground"`. Observation nền mang nhãn KHU VỰC
   (`parking`, `school`…) chứ không phải nhãn lớp âm thanh; đưa vào sẽ tạo ra 4 lớp ma.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

# Lớp nền, xem lý do ở docstring. Để tên ở hằng số thay vì rải chuỗi trong code.
BACKGROUND_CLASS = "ambient_noise"


def event_class_ids(ontology_path: Path) -> list[str]:
    """15 lớp sự kiện, thứ tự cố định (sorted) để chỉ số lớp không đổi giữa các lần chạy.

    Thứ tự phải ổn định: chỉ số lớp được nướng vào checkpoint. Dùng thứ tự của dict YAML
    thì thêm một lớp vào giữa file sẽ âm thầm đổi ý nghĩa mọi cột trong head đã huấn luyện.
    """
    config = yaml.safe_load(ontology_path.read_text(encoding="utf-8"))
    return sorted(name for name in config["classes"] if name != BACKGROUND_CLASS)


@dataclass(frozen=True)
class Clip:
    """Một clip cùng danh sách sự kiện đã quy về chỉ số lớp."""

    clip_id: str
    audio_path: Path
    events: tuple[tuple[int, float, float], ...]   # (class_idx, onset_sec, offset_sec)


def parse_jams(path: Path, class_to_idx: dict[str, int]) -> tuple[tuple[int, float, float], ...]:
    """Bóc sự kiện foreground từ một file .jams của Scaper."""
    data = json.loads(path.read_text(encoding="utf-8"))
    annotation = data["annotations"][0]
    events: list[tuple[int, float, float]] = []
    for observation in annotation["data"]:
        value = observation["value"]
        if not isinstance(value, dict) or value.get("role") != "foreground":
            continue
        label = value["label"]
        if label not in class_to_idx:
            # Lớp lạ (taxonomy đã đổi sau khi sinh dữ liệu) — bỏ qua NHƯNG phải thấy được.
            # Nuốt lặng sẽ làm dataset thiếu nhãn mà mọi thứ vẫn chạy bình thường.
            raise KeyError(
                f"{path.name}: nhãn {label!r} không có trong taxonomy. "
                "Dữ liệu synthetic sinh bằng một taxonomy khác với ontology_map.yaml hiện tại."
            )
        onset = float(observation["time"])
        offset = onset + float(observation["duration"])
        events.append((class_to_idx[label], onset, offset))
    return tuple(events)


def load_split(split_dir: Path, class_ids: list[str], limit: int | None = None) -> list[Clip]:
    """Ghép audio/*.wav với jams/*.jams theo tên, bỏ qua clip thiếu một trong hai.

    Bỏ qua clip lệch đôi là CỐ Ý: bộ synthetic có thể đang được sinh dở ở tiến trình khác,
    và một clip có .wav nhưng chưa có .jams (hoặc ngược lại) là trạng thái tạm thời bình
    thường, không phải hỏng dữ liệu.
    """
    class_to_idx = {name: i for i, name in enumerate(class_ids)}
    audio_dir, jams_dir = split_dir / "audio", split_dir / "jams"
    clips: list[Clip] = []
    for jams_path in sorted(jams_dir.glob("*.jams")):
        audio_path = audio_dir / f"{jams_path.stem}.wav"
        if not audio_path.exists():
            continue
        clips.append(Clip(jams_path.stem, audio_path, parse_jams(jams_path, class_to_idx)))
        if limit is not None and len(clips) >= limit:
            break
    return clips


def frame_targets(events, n_frames: int, n_classes: int, duration: float) -> np.ndarray:
    """Ma trận nhãn mức khung (n_frames, n_classes), 1 = lớp đang kêu tại khung đó.

    Làm tròn onset XUỐNG và offset LÊN: một sự kiện ngắn hơn một khung (tiếng súng ~80 ms,
    khung ~10 ms nên không gặp, nhưng va chạm chén đĩa thì có) vẫn phải chiếm ít nhất một
    khung. Làm tròn cả hai đầu theo cùng một chiều sẽ làm sự kiện ngắn biến mất hoàn toàn
    khỏi nhãn, và model học được rằng chúng không tồn tại.
    """
    target = np.zeros((n_frames, n_classes), dtype=np.float32)
    for class_idx, onset, offset in events:
        start = int(np.floor(onset / duration * n_frames))
        end = int(np.ceil(offset / duration * n_frames))
        start = max(0, min(start, n_frames - 1))
        end = max(start + 1, min(end, n_frames))
        target[start:end, class_idx] = 1.0
    return target


def clip_targets(events, n_classes: int) -> np.ndarray:
    """Nhãn mức clip (n_classes,) — dùng cho mAP và cho nhánh tagging."""
    target = np.zeros(n_classes, dtype=np.float32)
    for class_idx, _, _ in events:
        target[class_idx] = 1.0
    return target
