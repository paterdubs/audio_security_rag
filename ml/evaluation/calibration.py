"""Hiệu chuẩn xác suất — vì sao cả ba run panns_ft_v1/v2/v3 đỉnh F1(sự kiện) ở
θ ≈ 0,85–0,95 thay vì gần 0,5.

    .venv/Scripts/python.exe -m ml.evaluation.calibration --run panns_ft_v3 --split dev

Không train lại, không cần GPU: đọc `predictions/{split}_{subset}.npz` như
`error_analysis.py`. Đây mới là triệu chứng đang được đo, chưa ai kết luận nguyên nhân —
reliability diagram + ECE chỉ trả lời "lệch bao nhiêu, lệch theo hướng nào", không trả
lời "vì sao lệch". `pos_weight` là ứng viên, nhưng ablation nó cần train lại, khác hẳn
chi phí của trang này.

⚠️ ECE của một lớp có ít khung dương (sự kiện hiếm trong dev) không đáng tin: khi một bin
gần như trống, `trung_binh_du_bao`/`ti_le_duong` là NaN và bị loại khỏi trung bình — số
ECE khi đó dựa trên rất ít bin.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from common import enable_utf8_output  # noqa: E402

enable_utf8_output()

from ml.datasets.synthetic_sed import frame_targets  # noqa: E402
from ml.evaluation.error_analysis import so, viet_dan  # noqa: E402
from ml.evaluation.predictions import doc  # noqa: E402

RUNS_DIR = REPO_ROOT / "ml" / "runs"
DEFAULT_N_BINS = 10


def chia_bin(n_bins: int = DEFAULT_N_BINS) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1)


def do_tin_cay(xac_suat: np.ndarray, nhan: np.ndarray,
               n_bins: int = DEFAULT_N_BINS) -> list[dict]:
    """Reliability diagram: mỗi bin so trung bình xác suất dự báo với tỉ lệ dương thật.

    Hiệu chuẩn tốt thì hai cột đó bằng nhau ở mọi bin — model 'nói' 0,7 thì đúng 70% số
    lần. Bin trống (không mẫu nào rơi vào) trả NaN thay vì 0, để không lẫn với "hiệu
    chuẩn hoàn hảo tình cờ tại một bin không ai dùng tới".
    """
    canh = chia_bin(n_bins)
    chi_so = np.clip(np.digitize(xac_suat, canh[1:-1], right=True), 0, n_bins - 1)
    hang = []
    for b in range(n_bins):
        mat_na = chi_so == b
        so_luong = int(mat_na.sum())
        hang.append({
            "bin_duoi": float(canh[b]), "bin_tren": float(canh[b + 1]),
            "so_luong": so_luong,
            "trung_binh_du_bao": float(xac_suat[mat_na].mean()) if so_luong else float("nan"),
            "ti_le_duong": float(nhan[mat_na].mean()) if so_luong else float("nan"),
        })
    return hang


def ece(xac_suat: np.ndarray, nhan: np.ndarray, n_bins: int = DEFAULT_N_BINS) -> float:
    """Expected Calibration Error: trung bình |dự báo − thật| theo bin, cân theo số mẫu.

    Bin trống bị loại khỏi trung bình (không góp NaN vào tổng) — không phải "coi lệch
    bằng 0", mà là "bin đó không có gì để đo".
    """
    hang = do_tin_cay(xac_suat, nhan, n_bins)
    tong = sum(h["so_luong"] for h in hang)
    if tong == 0:
        return float("nan")
    return sum(h["so_luong"] * abs(h["trung_binh_du_bao"] - h["ti_le_duong"])
              for h in hang if h["so_luong"] > 0) / tong


def thu_thap(du_doan) -> tuple[np.ndarray, np.ndarray]:
    """(N·T, C) xác suất và nhãn mức khung, gộp mọi clip.

    Nhãn dựng bằng đúng `frame_targets` mà lúc train dùng để tạo nhãn khung
    (`sed_data.py::PrecomputedSedDataset.__getitem__`) — không tự viết lại công thức làm
    tròn onset/offset một lần nữa ở đây.
    """
    n_classes = len(du_doan.class_ids)
    su_kien: list[list[tuple[int, float, float]]] = [[] for _ in du_doan.clip_ids]
    for row, c, on, off in zip(du_doan.ref_clip, du_doan.ref_class,
                               du_doan.ref_onset, du_doan.ref_offset, strict=True):
        su_kien[int(row)].append((int(c), float(on), float(off)))
    xac_suat = np.concatenate([du_doan.khung(i) for i in range(len(du_doan.clip_ids))], axis=0)
    nhan = np.concatenate([
        frame_targets(su_kien[i], du_doan.n_frames, n_classes, du_doan.duration)
        for i in range(len(du_doan.clip_ids))
    ], axis=0)
    return xac_suat, nhan


def bang_hieu_chuan(du_doan, n_bins: int = DEFAULT_N_BINS) -> dict:
    """ECE + reliability diagram từng lớp, sắp xếp lớp lệch nhiều nhất lên đầu."""
    xac_suat, nhan = thu_thap(du_doan)
    theo_lop = []
    for idx, class_id in enumerate(du_doan.class_ids):
        theo_lop.append({
            "class_id": class_id,
            "n_khung_duong": int(nhan[:, idx].sum()),
            "ece": ece(xac_suat[:, idx], nhan[:, idx], n_bins),
            "reliability": do_tin_cay(xac_suat[:, idx], nhan[:, idx], n_bins),
        })
    theo_lop.sort(key=lambda h: h["ece"], reverse=True)
    return {
        "n_bins": n_bins,
        "ece_tong": ece(xac_suat.ravel(), nhan.ravel(), n_bins),
        "theo_lop": theo_lop,
    }


# ── Báo cáo ──────────────────────────────────────────────────────────────────


def _bang(tieu_de: list[str], hang: list[list[str]]) -> Iterator[str]:
    yield "| " + " | ".join(tieu_de) + " |"
    yield "|" + "|".join("---" for _ in tieu_de) + "|"
    for h in hang:
        yield "| " + " | ".join(h) + " |"


def dong_bao_cao(payload: dict) -> Iterator[str]:
    yield f"# Hiệu chuẩn xác suất — {payload['run']} ({payload['split']}/{payload['subset']})"
    yield ""
    yield (f"{payload['n_clip']} clip · {payload['n_bins']} bin · "
          f"ECE tổng (gộp mọi lớp) = **{so(payload['ece_tong'], 4)}**.")
    yield ""
    yield ("⚠️ ECE của lớp có ít khung dương không đáng tin — xem cột 'khung dương'. Trang "
          "này đo TRIỆU CHỨNG (lệch bao nhiêu), không kết luận nguyên nhân.")
    yield ""
    yield "## Theo lớp, lệch nhiều nhất trước"
    yield ""
    yield from _bang(
        ["lớp", "khung dương", "ECE"],
        [[h["class_id"], str(h["n_khung_duong"]), so(h["ece"], 4)] for h in payload["theo_lop"]],
    )
    yield ""
    yield "## Reliability diagram — 3 lớp lệch nhiều nhất"
    for h in payload["theo_lop"][:3]:
        yield ""
        yield f"### {h['class_id']} (ECE = {so(h['ece'], 4)})"
        yield ""
        yield from _bang(
            ["bin", "dự báo trung bình", "tỉ lệ dương thật", "số khung"],
            [[f"{so(r['bin_duoi'])}–{so(r['bin_tren'])}",
              so(r["trung_binh_du_bao"], 4) if r["so_luong"] else "—",
              so(r["ti_le_duong"], 4) if r["so_luong"] else "—",
              str(r["so_luong"])]
             for r in h["reliability"]],
        )


# ── Chạy ─────────────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> int:
    nguon = RUNS_DIR / args.run / "predictions" / f"{args.split}_{args.subset}.npz"
    if not nguon.exists():
        print(f"❌ chưa có {nguon}\n   Chạy: -m ml.evaluation.predictions --run {args.run} "
              f"--split {args.split}")
        return 1

    bat_dau = time.time()
    du_doan = doc(nguon)
    bang = bang_hieu_chuan(du_doan, args.bins)
    payload = {
        "run": args.run, "split": args.split, "subset": args.subset,
        "n_clip": len(du_doan.clip_ids),
        **bang,
        "giay": round(time.time() - bat_dau, 1),
        "ghi_chu": "ECE của lớp có ít khung dương không đáng tin; trang này đo triệu "
                   "chứng, không kết luận nguyên nhân.",
    }
    print(f"  ECE tổng = {payload['ece_tong']:.4f} ({payload['n_clip']} clip, "
          f"{args.bins} bin)", flush=True)

    out_json = RUNS_DIR / args.run / "calibration.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md = viet_dan(RUNS_DIR / args.run / "calibration.md", dong_bao_cao(payload))
    print(f"\n✓ {out_json.relative_to(REPO_ROOT)}\n✓ {out_md.relative_to(REPO_ROOT)} "
          f"· {payload['giay']:.0f}s", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="tên thư mục trong ml/runs/")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--subset", default="all")
    parser.add_argument("--bins", type=int, default=DEFAULT_N_BINS)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
