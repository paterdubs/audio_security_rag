"""Huấn luyện SED trên bộ synthetic — baseline PANNs CNN14.

    .venv/Scripts/python.exe -m ml.training.train_sed --epochs 20
    .venv/Scripts/python.exe -m ml.training.train_sed --freeze-backbone --epochs 15
    .venv/Scripts/python.exe -m ml.training.train_sed --zero-shot        # không huấn luyện

`--zero-shot` đo model KHI CHƯA học gì (head khởi tạo ngẫu nhiên): nó tồn tại để có một
mốc sàn tuyệt đối. Một baseline "học được" mà không hơn nổi mốc này thì nghĩa là vòng
huấn luyện đang hỏng ở đâu đó, chứ không phải bài toán khó.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from common import enable_utf8_output  # noqa: E402

enable_utf8_output()

from ml.evaluation.sed_metrics import (  # noqa: E402
    DEFAULT_MEDIAN_FILTER_FRAMES,
    SedScores,
    adaptive_median_sizes,
    class_event_durations,
    clip_level_map,
    event_and_segment_f1,
    frames_to_events,
)
from ml.models.panns_sed import (  # noqa: E402
    DEFAULT_TIME_POOL_BLOCKS,
    PannsSed,
    trainable_parameters,
)
from ml.tracking.fingerprint import van_tay_chia_tap, van_tay_du_lieu  # noqa: E402
from ml.tracking.run_manifest import ghi as ghi_manifest  # noqa: E402
from ml.tracking.run_manifest import ghi_epoch, tao_manifest  # noqa: E402
from ml.training.mixup import LABEL_HARD, LABEL_SOFT, mix_batch  # noqa: E402
from ml.training.sed_data import PrecomputedSedDataset, split_indices  # noqa: E402

FEATURES_DIR = REPO_ROOT / "data" / "features"
RUNS_DIR = REPO_ROOT / "ml" / "runs"
CHECKPOINT_PATH = REPO_ROOT / "data" / "reference" / "Cnn14_DecisionLevelMax.pth"

# Trần cho pos_weight. Lớp hiếm nhất (vehicle_crash, 33 clip bank) có tỉ lệ khung dương
# cực thấp; để pos_weight tự do sẽ ra hệ số hàng nghìn, biến loss thành gần như chỉ còn
# một lớp đó và mọi lớp khác bị bỏ rơi.
MAX_POS_WEIGHT = 30.0


def gieo_mam(seed: int) -> dict:
    """Gieo MỌI nguồn ngẫu nhiên, không chỉ phép chia train/val.

    Trước 19/09/2026 `--seed` chỉ đi vào `split_indices`; thứ tự xáo của DataLoader, hệ
    số trộn của mixup và SpecAugment đều chạy trên RNG toàn cục chưa gieo. Hệ quả: v1/v2/v3
    KHÔNG lặp lại được, và chênh lệch giữa hai run không tách được phần do thay đổi thật
    với phần do may rủi. Đây là yêu cầu R1 của TRAINING_OPS_PLAN.

    `PYTHONHASHSEED` chỉ có tác dụng với tiến trình con (DataLoader worker); tiến trình
    hiện tại đã khởi động xong nên đặt ở đây là để worker kế thừa, không phải cho bản thân nó.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return {"split": seed, "python": seed, "numpy": seed, "torch": seed, "cuda": seed}


def kiem_cong_hop_dong(ket_qua: str, bo_qua: bool) -> str | None:
    """`None` = được train tiếp; chuỗi = thông điệp chặn, in ra rồi dừng.

    Trước 19/09 cổng này chỉ IN CẢNH BÁO khi `ket_qua != "PASSED"` — một lô `FAILED` vẫn
    train được, và v1/v2 đã train trên lô legacy không đạt hợp đồng mà không có gì chặn
    lại. `KHONG_RO` ("chưa ai kiểm") bị chặn GIỐNG HỆT `FAILED` ("đã kiểm, hỏng") — "chưa
    ai kiểm" không phải lý do để train, nó là lý do để KHÔNG train cho tới khi kiểm.
    """
    if ket_qua == "PASSED" or bo_qua:
        return None
    return (f"hợp đồng dữ liệu: {ket_qua} — chặn train. Dùng --force-du-lieu-chua-dat để "
            "train dù vậy (cờ này được ghi vào manifest.json).")


def luu_checkpoint_resume(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                          scheduler, scaler: torch.amp.GradScaler, epoch: int,
                          best_map: float) -> None:
    """Trạng thái ĐẦY ĐỦ để resume — ghi đè MỖI epoch, khác `best.pt` (chỉ lưu model ở
    epoch mAP tốt nhất, dùng cho inference). Không gộp hai file: đổi schema của `best.pt`
    sẽ vỡ `predictions.py`/`eval_sed.py`, cả hai đọc đúng 5 khoá của nó.

    RNG lưu cả bốn nguồn `gieo_mam()` đã gieo — thiếu một nguồn thì lượt resume và lượt
    chạy liền mạch rẽ nhánh ngay từ mẫu tiếp theo mà không có triệu chứng nào.
    """
    torch.save({
        "epoch": epoch, "best_map": best_map,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
        "rng": {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }, path)


def nap_checkpoint_resume(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                          scheduler, scaler: torch.amp.GradScaler) -> tuple[int, float]:
    """Nạp lại đầy đủ → (epoch đã xong, best_map). Vòng lặp train tiếp tục từ epoch+1.

    Khôi phục RNG SAU KHI đã dựng model/optimizer/scheduler mới (chúng có thể tiêu thụ vài
    số ngẫu nhiên lúc khởi tạo) — nếu khôi phục trước, bước dựng lại sẽ lệch RNG đi so với
    lúc lưu.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    random.setstate(checkpoint["rng"]["python"])
    np.random.set_state(checkpoint["rng"]["numpy"])
    torch.set_rng_state(checkpoint["rng"]["torch"])
    if checkpoint["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
    return int(checkpoint["epoch"]), float(checkpoint["best_map"])


def detect_frame_count(model: nn.Module, device: torch.device, n_samples: int) -> int:
    """Hỏi thẳng model xem nó trả về bao nhiêu khung, thay vì tính tay từ hop size.

    Tính tay là chỗ rất dễ lệch một khung (STFT center padding, interpolate, pad cuối), và
    lệch một khung thì nhãn bị trượt pha so với dự đoán suốt cả quá trình huấn luyện.
    """
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, n_samples, device=device))
    return int(output["frame_logits"].shape[1])


def positive_weights(dataset: PrecomputedSedDataset, n_classes: int) -> torch.Tensor:
    """pos_weight theo lớp = (số khung âm)/(số khung dương), chặn trần."""
    positive = np.zeros(n_classes, dtype=np.float64)
    total = 0.0
    for position in range(len(dataset)):
        events = dataset.events_of(position)
        total += dataset.duration
        for class_idx, onset, offset in events:
            positive[class_idx] += max(0.0, offset - onset)
    negative = np.maximum(total - positive, 1e-6)
    weights = np.where(positive > 0, negative / np.maximum(positive, 1e-6), 1.0)
    return torch.tensor(np.minimum(weights, MAX_POS_WEIGHT), dtype=torch.float32)


def median_sizes_for(dataset: PrecomputedSedDataset, n_classes: int, n_frames: int,
                     adaptive: bool):
    """Cửa sổ lọc trung vị: một số chung, hoặc một cửa sổ cho mỗi lớp.

    Thống kê độ dài lấy từ tập ĐANG huấn luyện chứ không phải tập đánh giá — dùng nhãn
    của tập đánh giá để chọn siêu tham số hậu xử lý là rò rỉ, dù chỉ là một con số.
    """
    if not adaptive:
        return DEFAULT_MEDIAN_FILTER_FRAMES
    durations = class_event_durations(
        (dataset.events_of(i) for i in range(len(dataset))), n_classes
    )
    return adaptive_median_sizes(durations, frames_per_second=n_frames / dataset.duration)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, dataset: PrecomputedSedDataset,
             device: torch.device, threshold: float, full: bool,
             median_size=DEFAULT_MEDIAN_FILTER_FRAMES) -> SedScores:
    model.eval()
    clip_scores, clip_true = [], []
    reference: dict[str, list[dict]] = {}
    predicted: dict[str, list[dict]] = {}

    for batch in loader:
        waveform = batch["waveform"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(waveform)
        clip_scores.append(torch.sigmoid(output["clip_logits"].float()).cpu().numpy())
        clip_true.append(batch["clip_target"].numpy())

        if not full:
            continue
        frame_probability = torch.sigmoid(output["frame_logits"].float()).cpu().numpy()
        for row, position in enumerate(batch["index"].tolist()):
            clip_id = dataset.clip_id(position)
            predicted[clip_id] = frames_to_events(
                frame_probability[row], dataset.class_ids, dataset.duration, threshold,
                median_size,
            )
            reference[clip_id] = [
                {"event_label": dataset.class_ids[c], "onset": on, "offset": off}
                for c, on, off in dataset.events_of(position)
            ]

    scores = np.concatenate(clip_scores)
    targets = np.concatenate(clip_true)
    clip_map, per_class_ap = clip_level_map(scores, targets, dataset.class_ids)

    segment_f1 = event_f1 = 0.0
    per_class_event: dict[str, float] = {}
    if full:
        segment_f1, event_f1, per_class_event = event_and_segment_f1(
            reference, predicted, dataset.class_ids, dataset.duration
        )
    return SedScores(clip_map, per_class_ap, segment_f1, event_f1, per_class_event)


def run(args: argparse.Namespace) -> int:
    seeds = gieo_mam(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = CHECKPOINT_PATH if CHECKPOINT_PATH.exists() else None
    if checkpoint is None:
        print(f"⚠️ không thấy checkpoint PANNs tại {CHECKPOINT_PATH} — backbone khởi tạo NGẪU NHIÊN.")
        print("   Kết quả sẽ vô nghĩa. Xem scripts/fetch_panns_checkpoint.py")

    meta_path = FEATURES_DIR / f"{args.split}_meta.json"
    if not meta_path.exists():
        print(f"❌ chưa precompute: thiếu {meta_path}")
        print("   Chạy: .venv/Scripts/python.exe -m ml.features.precompute_waveform --split train")
        return 1
    n_classes = len(json.loads(meta_path.read_text(encoding="utf-8"))["class_ids"])

    model = PannsSed(n_classes, checkpoint, freeze_backbone=args.freeze_backbone,
                     time_pool_blocks=args.time_pool_blocks).to(device)
    n_samples = int(round(args.sample_rate * args.duration))
    n_frames = detect_frame_count(model, device, n_samples)
    print(f"model: {trainable_parameters(model):,} tham số học được · {n_frames} khung/clip · {device}")

    full = PrecomputedSedDataset(FEATURES_DIR, args.split, n_frames, args.sample_rate)
    train_idx, val_idx = split_indices(len(full), args.val_ratio, args.seed)
    train_set = PrecomputedSedDataset(FEATURES_DIR, args.split, n_frames, args.sample_rate, train_idx)
    val_set = PrecomputedSedDataset(FEATURES_DIR, args.split, n_frames, args.sample_rate, val_idx)
    print(f"dữ liệu: {len(train_set)} train / {len(val_set)} val (tách tạm từ synthetic train)")

    common = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": True}
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS_DIR / (args.name or time.strftime("sed_%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    # Manifest ghi TRƯỚC epoch đầu tiên: một run bị ngắt giữa chừng vẫn phải để lại đủ
    # lineage để biết checkpoint dở dang đó thuộc về dữ liệu nào, mã nào, cấu hình nào.
    manifest = tao_manifest(
        run_dir.name, vars(args), split=args.split,
        van_tay=van_tay_du_lieu(FEATURES_DIR, args.split, args.sample_rate,
                                bam_waveform=not args.khong_bam_waveform),
        chia_tap=van_tay_chia_tap(train_idx, val_idx), seeds=seeds,
    )
    manifest["timing"]["bat_dau"] = datetime.now(UTC).isoformat(timespec="seconds")
    manifest["model"] = {"n_frames": n_frames, "n_classes": n_classes,
                         "tham_so_hoc_duoc": trainable_parameters(model)}
    manifest["data"]["hop_dong"]["cong_chan_bo_qua"] = bool(args.force_du_lieu_chua_dat)
    ghi_manifest(run_dir, manifest)
    hop_dong = manifest["data"]["hop_dong"]["ket_qua"]
    thong_diep_chan = kiem_cong_hop_dong(hop_dong, args.force_du_lieu_chua_dat)
    if thong_diep_chan is not None:
        print(f"❌ {thong_diep_chan}")
        return 1
    if hop_dong != "PASSED":
        print(f"⚠️ hợp đồng dữ liệu: {hop_dong} nhưng --force-du-lieu-chua-dat đã bật — "
              "tiếp tục train, cờ đã ghi vào manifest.json.")

    median_size = median_sizes_for(train_set, n_classes, n_frames, args.adaptive_postproc)
    if args.adaptive_postproc:
        print("cửa sổ lọc theo lớp: " + ", ".join(
            f"{c}={int(w)}" for c, w in zip(full.class_ids, median_size)))

    if args.zero_shot:
        scores = evaluate(model, val_loader, val_set, device, args.threshold, full=True,
                          median_size=median_size)
        print(f"\n[zero-shot, head chưa học] {scores.summary()}")
        (run_dir / "zero_shot.json").write_text(
            json.dumps(scores.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    pos_weight = positive_weights(train_set, n_classes).to(device)
    frame_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    clip_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Learning rate PHÂN TẦNG: backbone đã pretrain trên AudioSet 2M clip, head thì ngẫu
    # nhiên hoàn toàn. Dùng chung một lr đủ lớn cho head sẽ xoá sạch đặc trưng đã học
    # trong vài chục bước đầu — và triệu chứng của nó là "train được nhưng thua cả
    # zero-shot", rất dễ bị đọc nhầm thành "dữ liệu synthetic vô dụng".
    head_parameters = list(model.head.parameters())
    head_ids = {id(p) for p in head_parameters}
    backbone_parameters = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": head_parameters, "lr": args.lr}]
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": args.lr * args.backbone_lr_scale})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g["lr"] for g in groups],
        epochs=args.epochs, steps_per_epoch=len(train_loader)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history, best_map, epoch_bat_dau = [], -1.0, 1
    resume_path = run_dir / "checkpoint_resume.pt"
    if args.resume:
        if resume_path.exists():
            da_xong, best_map = nap_checkpoint_resume(resume_path, model, optimizer,
                                                       scheduler, scaler)
            epoch_bat_dau = da_xong + 1
            history_path = run_dir / "history.json"
            if history_path.exists():
                history = json.loads(history_path.read_text(encoding="utf-8"))["history"]
            print(f"↻ resume từ epoch {da_xong} đã xong · best_map={best_map:.4f} · "
                  f"tiếp tục từ epoch {epoch_bat_dau}/{args.epochs}", flush=True)
        else:
            print(f"⚠️ --resume bật nhưng không thấy {resume_path} — bắt đầu từ epoch 1.")
    if epoch_bat_dau > args.epochs:
        print(f"epoch {epoch_bat_dau - 1} đã là epoch cuối ({args.epochs}) — không còn gì "
              "để resume.")
        return 0

    for epoch in range(epoch_bat_dau, args.epochs + 1):
        model.train()
        started, total_loss = time.time(), 0.0
        for batch in train_loader:
            waveform = batch["waveform"].to(device, non_blocking=True)
            frame_target = batch["frame_target"].to(device, non_blocking=True)
            clip_target = batch["clip_target"].to(device, non_blocking=True)

            # Mixup CHỈ ở nhánh huấn luyện, và chỉ ở một phần batch (mixup-prob): trộn
            # mọi batch thì model không bao giờ thấy một clip sạch nào.
            if args.mixup_alpha > 0 and np.random.rand() < args.mixup_prob:
                waveform, frame_target, clip_target = mix_batch(
                    waveform, frame_target, clip_target, args.mixup_alpha, args.mixup_label
                )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(waveform)
                loss = (frame_loss_fn(output["frame_logits"], frame_target)
                        + args.clip_loss_weight * clip_loss_fn(output["clip_logits"], clip_target))
            scaler.scale(loss).backward()
            # Cắt gradient sau khi unscale: pos_weight tới 30 làm loss của lớp hiếm dựng
            # đứng ở vài batch đầu, và một bước nhảy đủ lớn là đủ đẩy trọng số ra NaN.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        # F1 mức sự kiện đắt (sed_eval chạy trên CPU) — chỉ tính ở epoch cuối và mỗi
        # `--full-eval-every` epoch, còn lại theo dõi bằng mAP cho nhanh.
        full_eval = epoch % args.full_eval_every == 0 or epoch == args.epochs
        scores = evaluate(model, val_loader, val_set, device, args.threshold, full=full_eval,
                          median_size=median_size)
        elapsed = time.time() - started
        print(f"epoch {epoch:2d}/{args.epochs}  loss={total_loss / len(train_loader):.4f}  "
              f"{scores.summary()}  ({elapsed:.0f}s)", flush=True)
        history.append({"epoch": epoch, "loss": total_loss / len(train_loader),
                        "clip_map": scores.clip_map, "segment_f1": scores.segment_f1,
                        "event_f1": scores.event_f1, "seconds": elapsed})
        ghi_epoch(run_dir, manifest, {
            "epoch": epoch, "seconds": round(elapsed, 1), "full_eval": full_eval,
            "clip_map": scores.clip_map,
            "ket_thuc": datetime.now(UTC).isoformat(timespec="seconds"),
        })

        if scores.clip_map > best_map:
            best_map = scores.clip_map
            # `time_pool_blocks` phải đi CÙNG trọng số. Thiếu nó thì khâu đánh giá
            # phải đoán, và đoán sai (mặc định 5 trong khi v2 train ở 3) cho ra một
            # bộ số vô nghĩa mà chương trình vẫn chạy trót lọt — xem
            # `ml/evaluation/eval_sed.doc_time_pool_blocks`.
            torch.save({"model": model.state_dict(), "class_ids": full.class_ids,
                        "epoch": epoch, "clip_map": best_map,
                        "time_pool_blocks": args.time_pool_blocks,
                        "sample_rate": args.sample_rate, "duration": args.duration},
                       run_dir / "best.pt")

        # Ghi đè MỖI epoch, không chỉ epoch tốt nhất — resume phải tiếp tục từ epoch vừa
        # xong, không phải nhảy lùi về epoch tốt nhất trước đó.
        luu_checkpoint_resume(resume_path, model, optimizer, scheduler, scaler,
                              epoch, best_map)

    (run_dir / "history.json").write_text(
        json.dumps({"args": vars(args), "history": history,
                    "final": scores.__dict__}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    manifest["ket_qua_cuoi"] = scores.__dict__
    manifest["best_clip_map"] = best_map
    ghi_manifest(run_dir, manifest)
    print(f"\n✓ {run_dir.relative_to(REPO_ROOT)} · mAP tốt nhất {best_map:.4f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3, help="lr của head")
    parser.add_argument("--backbone-lr-scale", type=float, default=0.05,
                        help="lr backbone = lr × hệ số này")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--clip-loss-weight", type=float, default=0.5)
    parser.add_argument("--full-eval-every", type=int, default=5)
    parser.add_argument("--mixup-alpha", type=float, default=0.0,
                        help="0 = tắt. DCASE thường dùng 0.2")
    parser.add_argument("--mixup-prob", type=float, default=0.5,
                        help="tỉ lệ batch được trộn; 1.0 thì model không thấy clip sạch nào")
    parser.add_argument("--mixup-label", choices=[LABEL_SOFT, LABEL_HARD], default=LABEL_SOFT)
    parser.add_argument("--adaptive-postproc", action="store_true",
                        help="cửa sổ lọc trung vị riêng cho từng lớp, suy từ độ dài sự kiện thật")
    parser.add_argument("--time-pool-blocks", type=int, default=DEFAULT_TIME_POOL_BLOCKS,
                        help="5=323ms (CNN14 gốc) · 4=161ms · 3=80ms · 2=40ms. "
                             "Collar của DCASE là 200ms nên 5 là KHÔNG ĐỦ")
    parser.add_argument("--khong-bam-waveform", action="store_true",
                        help="bo bam 5 GB waveform khi lap manifest (chay thu nhanh); "
                             "van tay se KHONG phat hien duoc cache da sinh lai")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--zero-shot", action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--force-du-lieu-chua-dat", action="store_true",
                        help="cho phép train dù hợp đồng dữ liệu FAILED/KHONG_RO; cờ này "
                             "được ghi vào manifest.json để nhận diện run nào đã bỏ qua cổng")
    parser.add_argument("--resume", action="store_true",
                        help="tiếp tục từ ml/runs/<name>/checkpoint_resume.pt nếu có "
                             "(optimizer/scheduler/RNG đầy đủ), không thì bắt đầu epoch 1")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
