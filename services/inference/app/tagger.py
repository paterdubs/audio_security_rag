"""PANNs CNN14 → detection theo 16 lớp của đề tài.

⚠️ Đây là model TẠM của walking skeleton W1. Model đề xuất thật (BEATs đóng băng +
Conformer + BART, grounded decoding) là W4–W5 — SYSTEM.md §5. Đừng đọc số của bản tạm
này như số của đóng góp nghiên cứu.

Ba chi tiết dưới đây KHÔNG phải tuỳ chọn, đều đã trả giá bằng dữ liệu thật ở
`scripts/auto_screen.py`:

  1. Ghép nhãn theo **mã AudioSet** (`/m/xxxxx`), không theo tên hiển thị. Tên có dấu
     phẩy, có ngoặc, và trùng nhau giữa các nhánh ontology.
  2. Resample lên **32 kHz** trước khi suy luận. PANNs huấn luyện ở 32 kHz; đưa 16 kHz
     vào thì model vẫn chạy, vẫn ra số — nhưng là số của một tín hiệu bị dịch một quãng tám.
  3. Clip < 1 giây phải kéo dài bằng cách **lặp lại**, không đệm im lặng. Đo trên 40 clip
     thật: đệm im lặng → p trung bình 0.012 (37/40 bị loại oan); lặp lại → 0.400 (2/40).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 32000          # PANNs huấn luyện ở 32 kHz — không đổi
MIN_SAMPLES = SAMPLE_RATE    # CNN14 sập với clip ngắn hơn 1 giây
DETECTION_THRESHOLD = 0.20   # dưới mức này coi như không nghe thấy gì
BOUNDARY_FRACTION = 0.5      # ngưỡng biên = 0.5 × đỉnh, không phải hằng số tuyệt đối
BOUNDARY_PAD_SEC = 0.05


@dataclass(frozen=True)
class Detection:
    class_id: str
    onset: float
    offset: float
    confidence: float


def pad_to_minimum(audio: np.ndarray) -> np.ndarray:
    """Kéo dài clip ngắn bằng cách LẶP LẠI chính nó — xem ghi chú đầu file."""
    if len(audio) >= MIN_SAMPLES or len(audio) == 0:
        return audio
    repeats = int(np.ceil(MIN_SAMPLES / len(audio)))
    return np.tile(audio, repeats)[:MIN_SAMPLES]


def mid_to_index(labels: list[str], ontology_json: Path) -> dict[str, int]:
    """{mã AudioSet: chỉ số cột đầu ra của PANNs}. Ghép theo MÃ, không theo tên."""
    ontology = json.loads(ontology_json.read_text(encoding="utf-8"))
    by_name = {entry["name"]: entry["id"] for entry in ontology}
    mapping: dict[str, int] = {}
    for index, name in enumerate(labels):
        mid = by_name.get(name)
        if mid:
            mapping[mid] = index
    return mapping


def class_to_indices(taxonomy: dict, index_of: dict[str, int]) -> dict[str, list[int]]:
    """{class_id: [chỉ số PANNs]} cho cả 16 lớp."""
    result: dict[str, list[int]] = {}
    for class_id, spec in taxonomy["classes"].items():
        indices = [index_of[mid] for mid in spec.get("audioset_ids", []) if mid in index_of]
        if indices:
            result[class_id] = indices
    return result


def longest_run(mask: np.ndarray) -> tuple[int, int]:
    """Đoạn True dài nhất — [start, end). Lấy đoạn dài nhất thay vì mọi đoạn vượt ngưỡng
    để một tiếng lách tách lẻ không kéo biên sự kiện ra toàn clip."""
    best_start = best_end = start = 0
    running = False
    for i, value in enumerate(mask):
        if value and not running:
            start, running = i, True
        elif not value and running:
            running = False
            if i - start > best_end - best_start:
                best_start, best_end = start, i
    if running and len(mask) - start > best_end - best_start:
        best_start, best_end = start, len(mask)
    return best_start, best_end


def propose_boundary(frame_probs: np.ndarray, duration: float) -> tuple[float, float]:
    """[onset, offset] từ xác suất mức frame.

    Ngưỡng 0.5 × ĐỈNH chứ không phải hằng số tuyệt đối: một clip máy chỉ chắc 0.2 vẫn có
    biên đúng, chỉ là cả đường cong thấp. Ngưỡng tuyệt đối sẽ trả vùng rỗng cho đúng
    những clip cần người xem nhất.
    """
    if frame_probs.size == 0 or duration <= 0:
        return 0.0, max(duration, 0.0)
    peak = float(frame_probs.max())
    if peak <= 0:
        return 0.0, duration

    start, end = longest_run(frame_probs >= peak * BOUNDARY_FRACTION)
    if end <= start:
        return 0.0, duration

    seconds_per_frame = duration / frame_probs.size
    onset = max(0.0, start * seconds_per_frame - BOUNDARY_PAD_SEC)
    offset = min(duration, end * seconds_per_frame + BOUNDARY_PAD_SEC)
    return round(onset, 3), round(offset, 3)


def detections_from_frames(
    framewise: np.ndarray,
    class_indices: dict[str, list[int]],
    duration: float,
    threshold: float = DETECTION_THRESHOLD,
) -> list[Detection]:
    """Xác suất mức frame (n_frames × 527) → danh sách detection.

    Một lớp có nhiều mã AudioSet (ví dụ `siren` gồm 5 mã) → lấy MAX qua các mã: nghe
    thấy còi cứu thương hay còi cứu hoả đều là `siren`, không cộng dồn thành >1.
    """
    detections: list[Detection] = []
    for class_id, indices in sorted(class_indices.items()):
        class_frames = framewise[:, indices].max(axis=1)
        confidence = float(class_frames.max())
        if confidence < threshold:
            continue
        onset, offset = propose_boundary(class_frames, duration)
        detections.append(Detection(class_id, onset, offset, round(confidence, 4)))
    return sorted(detections, key=lambda d: d.onset)
