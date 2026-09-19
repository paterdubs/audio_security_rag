"""Lưu dự đoán thô mức ĐOẠN — Pha 3 của docs/TRAINING_OPS_PLAN.md.

    .venv/Scripts/python.exe -m ml.evaluation.predictions --run panns_ft_v3 --split dev

Vì sao phải có: ngày 17/09 cả ba run v1/v2/v3 đều xong, v3 có event-F1 0,1077 còn v2 có
0,1897 — và **không ai trả lời được vì sao**, vì dự đoán không được lưu. Muốn biết chênh
lệch đó là do dữ liệu khó hơn hay do ngưỡng 0,5 không còn đúng thì phải inference lại
toàn bộ checkpoint. Đó là chi phí thật của việc thiếu file này, không phải rủi ro lý thuyết.

**Lưu mức đoạn, không lưu mức khung.** `frame_logits` là bản lặp nguyên xi của
`segment_logits`: interpolator lặp mỗi đoạn đúng `2**time_pool_blocks` lần rồi
`pad_framewise_output` đệm khung cuối. Ở cấu hình v1 (31 đoạn → 1001 khung) lưu mức khung
tốn gấp 32 lần mà không thêm một bit nào. `khung_tu_doan()` dựng lại mức khung và
`tests/test_ml_predictions.py` khẳng định hai đường trùng khít từng phần tử — nếu thư viện
đổi cách nội suy, test đó vỡ chứ dữ liệu không trôi âm thầm.

float16 là đủ: đây là xác suất sau sigmoid trong [0,1], sai số lượng tử ~5e-4, nhỏ hơn
hai bậc so với bước quét ngưỡng thô nhất mà ta dùng (0,05).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from common import enable_utf8_output  # noqa: E402

enable_utf8_output()

from ml.evaluation.eval_sed import doc_time_pool_blocks  # noqa: E402
from ml.models.panns_sed import PannsSed  # noqa: E402
from ml.tracking.fingerprint import van_tay_chia_tap, van_tay_du_lieu  # noqa: E402
from ml.training.sed_data import PrecomputedSedDataset, split_indices  # noqa: E402

FEATURES_DIR = REPO_ROOT / "data" / "features"
RUNS_DIR = REPO_ROOT / "ml" / "runs"

# Số khung mel của CNN14 ở 10 s / 32 kHz / hop 320. Không phụ thuộc time_pool_blocks:
# interpolator + pad_framewise_output luôn đưa đầu ra về đúng số khung mel.
N_FRAMES = 1001


def khung_tu_doan(doan: np.ndarray, n_frames: int, ratio: int) -> np.ndarray:
    """(T, C) mức đoạn → (n_frames, C) mức khung — nghịch đảo của interpolator PANNs.

    Hai bước, đúng thứ tự thư viện gốc làm:
      1. `Interpolator`: lặp mỗi đoạn `ratio` lần → T × ratio khung
      2. `pad_framewise_output`: đệm cho đủ `n_frames` bằng cách LẶP KHUNG CUỐI
         (31 × 32 = 992, thiếu 9 khung; 125 × 8 = 1000, thiếu 1 khung)
    """
    lap = np.repeat(doan, ratio, axis=0)
    if len(lap) > n_frames:
        raise ValueError(
            f"{len(doan)} đoạn × ratio {ratio} = {len(lap)} khung, vượt {n_frames} khung mel. "
            "ratio không khớp time_pool_blocks của checkpoint."
        )
    if len(lap) < n_frames:
        lap = np.concatenate([lap, np.repeat(lap[-1:], n_frames - len(lap), axis=0)])
    return lap


@dataclass(frozen=True)
class DuDoan:
    """Nội dung một file predictions — đủ để chấm lại trên CPU, không cần GPU hay checkpoint."""

    doan_prob: np.ndarray          # (N, T, C) float16
    clip_prob: np.ndarray          # (N, C) float16
    clip_ids: list[str]
    class_ids: list[str]
    ref_clip: np.ndarray           # (M,) chỉ số clip của từng sự kiện tham chiếu
    ref_class: np.ndarray          # (M,)
    ref_onset: np.ndarray          # (M,) giây
    ref_offset: np.ndarray         # (M,)
    meta: dict

    @property
    def ratio(self) -> int:
        return int(self.meta["ratio"])

    @property
    def n_frames(self) -> int:
        return int(self.meta["n_frames"])

    @property
    def duration(self) -> float:
        return float(self.meta["duration"])

    def khung(self, row: int) -> np.ndarray:
        """Xác suất mức khung của clip thứ `row`, dựng lại từ mức đoạn (float32)."""
        return khung_tu_doan(self.doan_prob[row].astype(np.float32), self.n_frames, self.ratio)

    def clip_target(self) -> np.ndarray:
        """(N, C) nhãn mức clip, suy từ danh sách sự kiện tham chiếu."""
        target = np.zeros((len(self.clip_ids), len(self.class_ids)), dtype=np.float32)
        target[self.ref_clip, self.ref_class] = 1.0
        return target

    def tham_chieu(self) -> dict[str, list[dict]]:
        """{clip_id: [{event_label, onset, offset}]} — định dạng sed_eval cần.

        MỌI clip đều có mặt, kể cả clip không sự kiện nào: thiếu clip ở một phía làm
        sed_eval hiểu là "không đánh giá" thay vì "đoán rỗng", và điểm cao lên một cách sai.
        """
        ket_qua: dict[str, list[dict]] = {clip_id: [] for clip_id in self.clip_ids}
        for row, class_idx, onset, offset in zip(self.ref_clip, self.ref_class,
                                                 self.ref_onset, self.ref_offset, strict=False):
            ket_qua[self.clip_ids[int(row)]].append({
                "event_label": self.class_ids[int(class_idx)],
                "onset": float(onset), "offset": float(offset),
            })
        return ket_qua


def doc(path: Path) -> DuDoan:
    with np.load(path, allow_pickle=False) as data:
        return DuDoan(
            doan_prob=data["doan_prob"], clip_prob=data["clip_prob"],
            clip_ids=[str(x) for x in data["clip_ids"]],
            class_ids=[str(x) for x in data["class_ids"]],
            ref_clip=data["ref_clip"], ref_class=data["ref_class"],
            ref_onset=data["ref_onset"], ref_offset=data["ref_offset"],
            meta=json.loads(str(data["meta"])),
        )


@torch.no_grad()
def thu_thap(model, loader: DataLoader, dataset: PrecomputedSedDataset,
             device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str], list[list]]:
    """Chạy model một lượt, giữ xác suất mức đoạn và mức clip."""
    model.eval()
    doan, clip, clip_ids, su_kien = [], [], [], []
    for batch in loader:
        waveform = batch["waveform"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(waveform)
        doan.append(torch.sigmoid(output["segment_logits"].float()).cpu().numpy().astype(np.float16))
        clip.append(torch.sigmoid(output["clip_logits"].float()).cpu().numpy().astype(np.float16))
        for position in batch["index"].tolist():
            clip_ids.append(dataset.clip_id(position))
            su_kien.append(dataset.events_of(position))
    return np.concatenate(doan), np.concatenate(clip), clip_ids, su_kien


def luu(path: Path, doan_prob, clip_prob, clip_ids, class_ids, su_kien, meta: dict) -> Path:
    """Ghi .npz nén — sự kiện tham chiếu để phẳng vì mỗi clip có số sự kiện khác nhau."""
    ref_clip, ref_class, ref_onset, ref_offset = [], [], [], []
    for row, events in enumerate(su_kien):
        for class_idx, onset, offset in events:
            ref_clip.append(row)
            ref_class.append(int(class_idx))
            ref_onset.append(float(onset))
            ref_offset.append(float(offset))

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        doan_prob=doan_prob, clip_prob=clip_prob,
        clip_ids=np.array(clip_ids), class_ids=np.array(class_ids),
        ref_clip=np.array(ref_clip, dtype=np.int32),
        ref_class=np.array(ref_class, dtype=np.int16),
        ref_onset=np.array(ref_onset, dtype=np.float32),
        ref_offset=np.array(ref_offset, dtype=np.float32),
        meta=json.dumps(meta, ensure_ascii=False),
    )
    return path


def run(args: argparse.Namespace) -> int:
    run_dir = RUNS_DIR / args.run
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        print(f"❌ không thấy {ckpt_path}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    class_ids = checkpoint["class_ids"]
    time_pool_blocks = doc_time_pool_blocks(run_dir, checkpoint)
    ratio = 2 ** time_pool_blocks
    print(f"checkpoint: epoch {checkpoint['epoch']} · mAP lúc lưu {checkpoint['clip_map']:.4f} "
          f"· time_pool_blocks={time_pool_blocks} (ratio {ratio})")

    model = PannsSed(len(class_ids), checkpoint_path=None,
                     time_pool_blocks=time_pool_blocks).to(device)
    model.load_state_dict(checkpoint["model"])

    full = PrecomputedSedDataset(FEATURES_DIR, args.split, N_FRAMES, args.sample_rate)
    if args.subset == "all":
        dataset, chia_tap = full, None
        # v1/v2 train trên lô legacy đã bị xoá; `--subset val` trên cache hiện tại KHÔNG
        # dựng lại được tập val của chúng. Đánh giá cả split (dev) là giao thức chung duy
        # nhất mà cả ba run đều chưa từng huấn luyện trên đó.
    else:
        train_idx, val_idx = split_indices(len(full), args.val_ratio, args.seed)
        indices = val_idx if args.subset == "val" else train_idx
        dataset = PrecomputedSedDataset(FEATURES_DIR, args.split, N_FRAMES,
                                        args.sample_rate, indices)
        chia_tap = van_tay_chia_tap(train_idx, val_idx)
        print("⚠️ subset=val tái dựng phép chia từ CACHE HIỆN TẠI. Nếu cache đã sinh lại "
              "kể từ lần train thì đây không phải tập val của run đó.")

    print(f"▶ {args.run} · split {args.split} · subset {args.subset} · {len(dataset)} clip · {device}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)

    bat_dau = time.time()
    doan_prob, clip_prob, clip_ids, su_kien = thu_thap(model, loader, dataset, device)
    giay = time.time() - bat_dau

    meta = {
        "run": args.run, "split": args.split, "subset": args.subset,
        "n_clips": len(clip_ids), "n_doan": int(doan_prob.shape[1]),
        "n_frames": N_FRAMES, "ratio": ratio, "time_pool_blocks": time_pool_blocks,
        "duration": float(full.duration), "sample_rate": args.sample_rate,
        "checkpoint": {"epoch": int(checkpoint["epoch"]),
                       "clip_map_luc_luu": float(checkpoint["clip_map"])},
        "chia_tap": chia_tap,
        "du_lieu": van_tay_du_lieu(FEATURES_DIR, args.split, args.sample_rate,
                                   bam_waveform=not args.khong_bam_waveform),
        "torch": torch.__version__, "device": device.type,
        "giay_inference": round(giay, 1),
        # Ngưỡng và cửa sổ lọc KHÔNG được nướng vào file này — đó chính là hai thứ ta
        # muốn quét lại. Ghi giá trị mặc định lúc train chỉ để đối chiếu.
        "threshold_luc_train": 0.5,
    }
    path = luu(run_dir / "predictions" / f"{args.split}_{args.subset}.npz",
               doan_prob, clip_prob, clip_ids, class_ids, su_kien, meta)
    print(f"✓ {path.relative_to(REPO_ROOT)} · {path.stat().st_size / 1e6:.1f} MB "
          f"· {giay:.0f}s inference")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="tên thư mục trong ml/runs/")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--subset", choices=["all", "val", "train"], default="all")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--khong-bam-waveform", action="store_true",
                        help="bỏ băm waveform (nhanh hơn, nhưng vân tay mất khả năng "
                             "phát hiện cache đã sinh lại)")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
