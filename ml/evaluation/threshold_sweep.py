"""Quét ngưỡng trên dự đoán đã lưu — không cần GPU, không train lại.

    .venv/Scripts/python.exe -m ml.evaluation.threshold_sweep --run panns_ft_v3 --split dev

Đây là thứ mở khoá một kết luận đang bị chặn: bảng v1/v2/v3 trong STATUS §3 đều chấm ở
ngưỡng **cố định 0,5**, và TRAINING_OPS_PLAN đã ghi thẳng rằng không ai biết chênh lệch
F1 giữa ba run là do dữ liệu khó hơn hay do 0,5 không còn đúng. Quét ngưỡng trả lời đúng
vế thứ hai.

⚠️ **Ngưỡng chọn ở đây là ngưỡng chọn trên DEV.** Theo thoả thuận ở CLAUDE.md §5, ngưỡng
chọn trên dev rồi KHOÁ LẠI; test chạy một lần. Thêm một giới hạn phải nói rõ: dev tổng
hợp dùng CHUNG foreground bank với train, nên nó không phải tập độc lập — nó chỉ đủ tư
cách để chọn siêu tham số, không đủ tư cách làm kết quả báo cáo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from common import enable_utf8_output  # noqa: E402

enable_utf8_output()

from ml.evaluation.predictions import doc  # noqa: E402
from ml.evaluation.sed_metrics import (  # noqa: E402
    DEFAULT_MEDIAN_FILTER_FRAMES,
    adaptive_median_sizes,
    class_event_durations,
    clip_level_map,
    event_and_segment_f1,
    frames_to_events,
)

RUNS_DIR = REPO_ROOT / "ml" / "runs"


def luoi_nguong(spec: str) -> list[float]:
    """'0.05:0.95:0.05' → [0.05, 0.10, …, 0.95]."""
    dau, cuoi, buoc = (float(x) for x in spec.split(":"))
    n = int(round((cuoi - dau) / buoc)) + 1
    return [round(dau + i * buoc, 4) for i in range(n)]


def cua_so_loc(du_doan, thich_ung: bool):
    """Cửa sổ lọc trung vị: một số chung, hoặc một cửa sổ cho mỗi lớp.

    ⚠️ Cửa sổ thích ứng ở đây suy từ độ dài sự kiện của CHÍNH tập đang chấm. Đó là rò rỉ
    nhẹ (dùng nhãn của tập đánh giá để chọn một siêu tham số hậu xử lý) và chỉ chấp nhận
    được vì đây là dev, không phải test. Lúc train, `median_sizes_for` lấy thống kê từ
    tập train chứ không từ tập val — hai chỗ khác nhau có chủ đích.
    """
    if not thich_ung:
        return DEFAULT_MEDIAN_FILTER_FRAMES
    su_kien = [[] for _ in du_doan.clip_ids]
    for row, c, on, off in zip(du_doan.ref_clip, du_doan.ref_class,
                               du_doan.ref_onset, du_doan.ref_offset, strict=True):
        su_kien[int(row)].append((int(c), float(on), float(off)))
    do_dai = class_event_durations(su_kien, len(du_doan.class_ids))
    return adaptive_median_sizes(do_dai, frames_per_second=du_doan.n_frames / du_doan.duration)


def quet(du_doan, nguongs: list[float], median_size) -> list[dict]:
    """Chấm điểm ở từng ngưỡng. Dựng khung MỘT lần, chỉ đổi ngưỡng."""
    tham_chieu = du_doan.tham_chieu()
    khung = [du_doan.khung(i) for i in range(len(du_doan.clip_ids))]

    ket_qua = []
    for nguong in nguongs:
        bat_dau = time.time()
        du_bao = {
            clip_id: frames_to_events(khung[i], du_doan.class_ids, du_doan.duration,
                                      nguong, median_size)
            for i, clip_id in enumerate(du_doan.clip_ids)
        }
        n_su_kien = sum(len(v) for v in du_bao.values())
        segment_f1, event_f1, theo_lop = event_and_segment_f1(
            tham_chieu, du_bao, du_doan.class_ids, du_doan.duration
        )
        ket_qua.append({"threshold": nguong, "segment_f1": segment_f1, "event_f1": event_f1,
                        "n_su_kien_du_bao": n_su_kien, "event_f1_theo_lop": theo_lop,
                        "giay": round(time.time() - bat_dau, 1)})
        print(f"  θ={nguong:.2f}  F1(đoạn)={segment_f1:.4f}  F1(sự kiện)={event_f1:.4f}  "
              f"({n_su_kien:>6} sự kiện dự báo, {ket_qua[-1]['giay']:.0f}s)", flush=True)
    return ket_qua


def run(args: argparse.Namespace) -> int:
    path = RUNS_DIR / args.run / "predictions" / f"{args.split}_{args.subset}.npz"
    if not path.exists():
        print(f"❌ chưa có {path}\n   Chạy: -m ml.evaluation.predictions --run {args.run} "
              f"--split {args.split}")
        return 1

    du_doan = doc(path)
    nguongs = luoi_nguong(args.thresholds)
    median_size = cua_so_loc(du_doan, args.adaptive_postproc)

    print(f"▶ {args.run} · {args.split}/{args.subset} · {len(du_doan.clip_ids)} clip · "
          f"{du_doan.doan_prob.shape[1]} đoạn · lọc "
          f"{'thích ứng' if args.adaptive_postproc else DEFAULT_MEDIAN_FILTER_FRAMES}")

    # mAP mức clip KHÔNG phụ thuộc ngưỡng — in một lần làm mốc đối chiếu với lúc train.
    clip_map, ap_theo_lop = clip_level_map(
        du_doan.clip_prob.astype(np.float32), du_doan.clip_target(), du_doan.class_ids
    )
    print(f"  mAP(clip) = {clip_map:.4f} (không phụ thuộc ngưỡng)\n")

    ket_qua = quet(du_doan, nguongs, median_size)
    tot_event = max(ket_qua, key=lambda r: r["event_f1"])
    tot_segment = max(ket_qua, key=lambda r: r["segment_f1"])
    tai_05 = min(ket_qua, key=lambda r: abs(r["threshold"] - 0.5))

    print(f"\n{'chỉ số':<18}{'θ=0,50':>10}{'θ tốt nhất':>13}{'θ':>7}{'chênh':>10}")
    print("-" * 58)
    print(f"{'F1(sự kiện)':<18}{tai_05['event_f1']:>10.4f}{tot_event['event_f1']:>13.4f}"
          f"{tot_event['threshold']:>7.2f}{tot_event['event_f1'] - tai_05['event_f1']:>+10.4f}")
    print(f"{'F1(đoạn 1s)':<18}{tai_05['segment_f1']:>10.4f}{tot_segment['segment_f1']:>13.4f}"
          f"{tot_segment['threshold']:>7.2f}{tot_segment['segment_f1'] - tai_05['segment_f1']:>+10.4f}")

    print(f"\n{'lớp':<22}{'AP(clip)':>10}{'F1(sk) θ=0,50':>15}{'F1(sk) θ tốt':>14}")
    print("-" * 61)
    for class_id in du_doan.class_ids:
        print(f"{class_id:<22}{ap_theo_lop.get(class_id, float('nan')):>10.4f}"
              f"{tai_05['event_f1_theo_lop'].get(class_id, 0.0):>15.4f}"
              f"{tot_event['event_f1_theo_lop'].get(class_id, 0.0):>14.4f}")

    out = RUNS_DIR / args.run / f"threshold_sweep_{args.split}_{args.subset}.json"
    out.write_text(json.dumps({
        "run": args.run, "split": args.split, "subset": args.subset,
        "n_clips": len(du_doan.clip_ids), "meta_du_doan": du_doan.meta,
        "median_size": (int(median_size) if np.isscalar(median_size)
                        else {c: int(w) for c, w in zip(du_doan.class_ids, median_size, strict=True)}),
        "clip_map": clip_map, "clip_ap_theo_lop": ap_theo_lop,
        "quet": ket_qua,
        "tot_nhat": {"event_f1": tot_event["threshold"], "segment_f1": tot_segment["threshold"]},
        "ghi_chu": "Ngưỡng chọn trên DEV. Dev tổng hợp dùng chung foreground bank với "
                   "train nên không phải tập độc lập; chỉ dùng để chọn siêu tham số.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ {out.relative_to(REPO_ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--subset", default="all")
    parser.add_argument("--thresholds", default="0.05:0.95:0.05",
                        help="dau:cuoi:buoc")
    parser.add_argument("--adaptive-postproc", action="store_true",
                        help="cửa sổ lọc riêng từng lớp; xem cảnh báo rò rỉ ở cua_so_loc()")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
