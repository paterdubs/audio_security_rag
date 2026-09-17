"""Đánh giá một checkpoint SED, so hậu xử lý CỐ ĐỊNH với hậu xử lý THÍCH ỨNG.

    .venv/Scripts/python.exe -m ml.evaluation.eval_sed --run panns_ft_v1

Hậu xử lý chỉ chạy lúc suy luận, không đụng tới trọng số — nên đây là một phép so sánh
A/B sạch trên CÙNG một model: mọi khác biệt về điểm đều do bộ lọc, không do may rủi của
lần huấn luyện. Sinh lại kết quả này rẻ, không cần train lại.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from common import enable_utf8_output  # noqa: E402

enable_utf8_output()

from ml.evaluation.sed_metrics import (  # noqa: E402
    DEFAULT_MEDIAN_FILTER_FRAMES, adaptive_median_sizes, class_event_durations,
    clip_level_map, event_and_segment_f1, frames_to_events,
)
from ml.models.panns_sed import DEFAULT_TIME_POOL_BLOCKS, PannsSed  # noqa: E402
from ml.training.sed_data import PrecomputedSedDataset, split_indices  # noqa: E402

FEATURES_DIR = REPO_ROOT / "data" / "features"
RUNS_DIR = REPO_ROOT / "ml" / "runs"


@torch.no_grad()
def collect_predictions(model, loader, dataset, device):
    """Chạy model MỘT lần, giữ lại xác suất thô.

    Cố ý tách khỏi bước tính điểm: quét nhiều cấu hình hậu xử lý mà chạy lại model mỗi
    lần thì tốn GPU vô ích, và tệ hơn là dễ vô tình so hai lần chạy khác nhau.
    """
    model.eval()
    clip_scores, clip_true, frame_probabilities, clip_ids = [], [], [], []
    for batch in loader:
        waveform = batch["waveform"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(waveform)
        clip_scores.append(torch.sigmoid(output["clip_logits"].float()).cpu().numpy())
        clip_true.append(batch["clip_target"].numpy())
        frame_probabilities.append(torch.sigmoid(output["frame_logits"].float()).cpu().numpy())
        clip_ids.extend(dataset.clip_id(p) for p in batch["index"].tolist())
    return (np.concatenate(clip_scores), np.concatenate(clip_true),
            np.concatenate(frame_probabilities), clip_ids)


def score_with(frame_probabilities, clip_ids, reference, dataset, threshold, median_size):
    predicted = {
        clip_id: frames_to_events(frame_probabilities[row], dataset.class_ids,
                                  dataset.duration, threshold, median_size)
        for row, clip_id in enumerate(clip_ids)
    }
    return event_and_segment_f1(reference, predicted, dataset.class_ids, dataset.duration)


def doc_time_pool_blocks(run_dir: Path, checkpoint: dict) -> int:
    """`time_pool_blocks` mà run này ĐÃ train, đọc theo thứ tự tin cậy giảm dần.

    1. checkpoint — nguồn chắc chắn nhất; các run từ 17/09 trở đi đều ghi
    2. history.json → args — ghi lúc train, luôn khớp; dùng cho run cũ
    3. mặc định, kèm CẢNH BÁO — đoán sai ở đây cho ra một bộ số vô nghĩa mà
       chương trình vẫn chạy trót lọt, nên im lặng là không chấp nhận được
    """
    if "time_pool_blocks" in checkpoint:
        return int(checkpoint["time_pool_blocks"])

    history_path = run_dir / "history.json"
    if history_path.exists():
        args_da_train = json.loads(history_path.read_text(encoding="utf-8")).get("args", {})
        if "time_pool_blocks" in args_da_train:
            return int(args_da_train["time_pool_blocks"])

    print(f"⚠️ {run_dir.name} không ghi time_pool_blocks ở đâu cả — dùng mặc định "
          f"{DEFAULT_TIME_POOL_BLOCKS}. Nếu run này train ở giá trị khác thì mọi số "
          f"dưới đây đều SAI mà không có triệu chứng.")
    return DEFAULT_TIME_POOL_BLOCKS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="tên thư mục trong ml/runs/")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--sample-rate", type=int, default=32000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(RUNS_DIR / args.run / "best.pt", map_location="cpu", weights_only=False)
    class_ids = checkpoint["class_ids"]
    print(f"checkpoint: epoch {checkpoint['epoch']} · mAP lúc lưu {checkpoint['clip_map']:.4f}")

    # Độ phân giải thời gian phải khớp ĐÚNG lúc train. `panns_ft_v2` train ở
    # time_pool_blocks=3 (80 ms/frame) còn mặc định là 5 (323 ms/frame); dựng sai thì
    # hoặc load_state_dict nổ vì lệch kích thước, hoặc — tệ hơn — chạy trót lọt và cho
    # ra một bộ số vô nghĩa. Đọc từ chính run thay vì thêm một cờ CLI: cờ thì đặt sai
    # được, đọc từ run thì không.
    time_pool_blocks = doc_time_pool_blocks(RUNS_DIR / args.run, checkpoint)
    print(f"time_pool_blocks = {time_pool_blocks} "
          f"({1000 * 1024 * 2 ** time_pool_blocks / args.sample_rate:.0f} ms mỗi frame)")

    model = PannsSed(len(class_ids), checkpoint_path=None,
                     time_pool_blocks=time_pool_blocks).to(device)
    model.load_state_dict(checkpoint["model"])

    n_frames = 1001
    full = PrecomputedSedDataset(FEATURES_DIR, args.split, n_frames, args.sample_rate)
    train_idx, val_idx = split_indices(len(full), args.val_ratio, args.seed)
    train_set = PrecomputedSedDataset(FEATURES_DIR, args.split, n_frames, args.sample_rate, train_idx)
    val_set = PrecomputedSedDataset(FEATURES_DIR, args.split, n_frames, args.sample_rate, val_idx)

    loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    scores, targets, frame_probabilities, clip_ids = collect_predictions(model, loader, val_set, device)

    clip_map, per_class_ap = clip_level_map(scores, targets, class_ids)
    print(f"\nmAP mức clip: {clip_map:.4f}  (không phụ thuộc hậu xử lý)\n")

    reference = {
        val_set.clip_id(p): [{"event_label": class_ids[c], "onset": on, "offset": off}
                             for c, on, off in val_set.events_of(p)]
        for p in range(len(val_set))
    }

    # Cửa sổ thích ứng suy từ tập TRAIN, không phải tập đang chấm điểm.
    durations = class_event_durations(
        (train_set.events_of(i) for i in range(len(train_set))), len(class_ids)
    )
    adaptive = adaptive_median_sizes(durations, frames_per_second=n_frames / val_set.duration)

    print(f"{'lớp':<22}{'phân vị 25 (s)':>16}{'cửa sổ':>9}")
    print("-" * 47)
    for class_id, duration_sec, window in zip(class_ids, durations, adaptive):
        shown = f"{duration_sec:.3f}" if np.isfinite(duration_sec) else "—"
        print(f"{class_id:<22}{shown:>16}{int(window):>9}")

    results = {}
    for label, median_size in (("cố định (7)", DEFAULT_MEDIAN_FILTER_FRAMES),
                               ("thích ứng", adaptive)):
        segment_f1, event_f1, per_class = score_with(
            frame_probabilities, clip_ids, reference, val_set, args.threshold, median_size
        )
        results[label] = {"segment_f1": segment_f1, "event_f1": event_f1,
                          "event_f1_per_class": per_class}
        print(f"\n[{label}]  F1(đoạn 1s)={segment_f1:.4f}  F1(sự kiện)={event_f1:.4f}")

    co_dinh, thich_ung = results["cố định (7)"], results["thích ứng"]
    delta_event = thich_ung["event_f1"] - co_dinh["event_f1"]
    delta_segment = thich_ung["segment_f1"] - co_dinh["segment_f1"]
    print(f"\nCHÊNH LỆCH  F1(sự kiện) {delta_event:+.4f}   F1(đoạn) {delta_segment:+.4f}")

    print(f"\n{'lớp':<22}{'cố định':>10}{'thích ứng':>12}{'chênh':>9}")
    print("-" * 53)
    for class_id in class_ids:
        a = co_dinh["event_f1_per_class"].get(class_id, 0.0)
        b = thich_ung["event_f1_per_class"].get(class_id, 0.0)
        print(f"{class_id:<22}{a:>10.4f}{b:>12.4f}{b - a:>+9.4f}")

    out = RUNS_DIR / args.run / "postproc_ab.json"
    out.write_text(json.dumps(
        {"clip_map": clip_map, "clip_ap_per_class": per_class_ap,
         "median_sizes": {c: int(w) for c, w in zip(class_ids, adaptive)},
         "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
