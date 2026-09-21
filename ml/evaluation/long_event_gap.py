"""Ứng viên cuối cùng của `long_event` — khe hở năng lượng THẬT bên trong sự kiện dài.

    .venv/Scripts/python.exe -m ml.evaluation.long_event_gap --run panns_ft_v3 --split dev

Hai giả thuyết trước đã bị đo bác bỏ: cửa sổ lọc trung vị 7 khung quá hẹp
(docs/measurements/long_event_postproc_20260919.md), rồi trần `MAX_MEDIAN_FRAMES=51` quá
thấp (docs/measurements/max_median_ceiling_20260919.md). Ứng viên còn lại: Scaper sinh
soundscape bằng cách chồng foreground lên nền, và một sự kiện `long_event` 4+ giây (tiếng
còi, báo động) có thể có khoảng lặng thật ở giữa mà nhãn không phản ánh — nhãn chỉ có một
onset/offset bao trọn.

Không đủ để chỉ đo "có khe hở hay không" — mọi âm thanh thực đều có lúc trầm lúc bổng.
Phải so **nhóm sự kiện bị phân mảnh** với **nhóm không bị phân mảnh**, trong CÙNG lát cắt
`long_event`: nếu khe hở của hai nhóm giống nhau thì giả thuyết bị bác bỏ, không phải vì
không có khe hở, mà vì khe hở không giải thích được vì sao một sự kiện vỡ còn sự kiện kia
thì không.

Không train lại, không GPU: đọc dự đoán đã lưu (Pha 3) + audio gốc trong
`data/synthetic/{split}/audio/`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from common import enable_utf8_output  # noqa: E402

enable_utf8_output()

from ml.evaluation.error_analysis import (  # noqa: E402
    doc_lat_cat,
    giai_ma,
    nhom_theo_lat_cat,
    so,
    viet_dan,
)
from ml.evaluation.error_taxonomy import chong_lan  # noqa: E402
from ml.evaluation.predictions import doc  # noqa: E402
from ml.evaluation.sed_metrics import DEFAULT_MEDIAN_FILTER_FRAMES  # noqa: E402

RUNS_DIR = REPO_ROOT / "ml" / "runs"
DATA_DIR = REPO_ROOT / "data" / "synthetic"

HOP_SEC_MAC_DINH = 0.02          # ~20 ms — đủ mịn để bắt khe hở cỡ trăm ms
NGUONG_TI_LE_MAC_DINH = 0.2      # RMS dưới 20% trung vị của chính sự kiện đó = "im"
MIN_DURATION_SEC_MAC_DINH = 1.0  # sự kiện ngắn hơn không thể chứa khe hở có ý nghĩa


def danh_dau_phan_manh(ref: list[dict], pred: list[dict]) -> list[bool]:
    """Cờ song song với `ref` — True nếu sự kiện đó bị ≥2 dự báo CÙNG LỚP, có chồng lấn.

    Cùng định nghĩa phân mảnh với `error_taxonomy._dem_cau_truc`, khác ở chỗ hàm đó chỉ
    đếm số lượng còn hàm này phơi ra ĐÚNG sự kiện nào bị vỡ — cần biết cụ thể sự kiện nào
    để tra khe hở năng lượng của riêng nó, không phải một con số tổng.
    """
    return [
        sum(1 for p in pred if p["event_label"] == r["event_label"] and chong_lan(r, p) > 0) >= 2
        for r in ref
    ]


def bien_do_rms(waveform: np.ndarray, sr: int, onset: float, offset: float,
                hop_sec: float = HOP_SEC_MAC_DINH) -> np.ndarray:
    """RMS từng cửa sổ `hop_sec` trong [onset, offset] của audio GỐC — không qua model,
    không median-filter. Đo trực tiếp xem Scaper có sinh khoảng lặng thật hay không."""
    lo = max(0, int(round(onset * sr)))
    hi = min(len(waveform), int(round(offset * sr)))
    doan = waveform[lo:hi]
    hop = max(1, int(round(hop_sec * sr)))
    n_khung = len(doan) // hop
    if n_khung == 0:
        return np.array([]) if len(doan) == 0 else np.array(
            [np.sqrt(np.mean(doan.astype(np.float64) ** 2))]
        )
    return np.array([
        np.sqrt(np.mean(doan[i * hop:(i + 1) * hop].astype(np.float64) ** 2))
        for i in range(n_khung)
    ])


def khe_ho_dai_nhat_sec(bien_do: np.ndarray, hop_sec: float,
                        nguong_ti_le: float = NGUONG_TI_LE_MAC_DINH) -> float:
    """Chuỗi liên tục dài nhất mà RMS < `nguong_ti_le` × trung vị RMS của CHÍNH sự kiện đó.

    So với trung vị của chính nó, không phải một ngưỡng tuyệt đối chung: hai sự kiện to
    nhỏ khác hẳn nhau (còi hú vs báo động xa) không so được bằng cùng một mức decibel.
    """
    if len(bien_do) == 0:
        return 0.0
    nguong = nguong_ti_le * float(np.median(bien_do))
    dai_nhat = dai_hien_tai = 0
    for duoi_nguong in bien_do < nguong:
        dai_hien_tai = dai_hien_tai + 1 if duoi_nguong else 0
        dai_nhat = max(dai_nhat, dai_hien_tai)
    return dai_nhat * hop_sec


def do_khe_ho_long_event(tham_chieu: dict[str, list[dict]], du_bao: dict[str, list[dict]],
                         audio_dir: Path, clip_ids: list[str], *,
                         hop_sec: float = HOP_SEC_MAC_DINH,
                         nguong_ti_le: float = NGUONG_TI_LE_MAC_DINH,
                         min_duration_sec: float = MIN_DURATION_SEC_MAC_DINH) -> list[dict]:
    """Với mỗi clip trong `clip_ids`: ghép phân mảnh, đo khe hở cho sự kiện đủ dài.

    Chỉ mở file audio khi có ít nhất một sự kiện đạt `min_duration_sec` — clip không liên
    quan (mọi sự kiện đều ngắn) không được phép làm lượt đo chết vì thiếu file audio.
    """
    ket_qua: list[dict] = []
    for clip_id in clip_ids:
        ref = tham_chieu.get(clip_id, [])
        pred = du_bao.get(clip_id, [])
        co_manh = danh_dau_phan_manh(ref, pred)
        can_do = [(r, manh) for r, manh in zip(ref, co_manh, strict=True)
                 if r["offset"] - r["onset"] >= min_duration_sec]
        if not can_do:
            continue
        waveform, sr = sf.read(str(audio_dir / f"{clip_id}.wav"), dtype="float32",
                               always_2d=False)
        for r, manh in can_do:
            bien_do = bien_do_rms(waveform, sr, r["onset"], r["offset"], hop_sec)
            ket_qua.append({
                "clip_id": clip_id, "lop": r["event_label"], "onset": r["onset"],
                "do_dai": round(r["offset"] - r["onset"], 3), "phan_manh": manh,
                "khe_ho_sec": khe_ho_dai_nhat_sec(bien_do, hop_sec, nguong_ti_le),
            })
    return ket_qua


def tom_tat(ket_qua: list[dict]) -> dict:
    """So khe hở giữa nhóm vỡ và nhóm nguyên — trung vị + p75, không kiểm định hình thức
    (mẫu nhỏ). NaN khi một nhóm rỗng, không phải 0 — 0 sẽ đọc nhầm thành 'khe hở bằng 0'."""
    vo = [r["khe_ho_sec"] for r in ket_qua if r["phan_manh"]]
    nguyen = [r["khe_ho_sec"] for r in ket_qua if not r["phan_manh"]]

    def _pct(xs: list[float], q: float) -> float:
        return float(np.percentile(xs, q)) if xs else float("nan")

    return {
        "n_vo": len(vo), "n_nguyen": len(nguyen),
        "vo_trung_vi": _pct(vo, 50), "vo_p75": _pct(vo, 75),
        "nguyen_trung_vi": _pct(nguyen, 50), "nguyen_p75": _pct(nguyen, 75),
    }


# ── Báo cáo ──────────────────────────────────────────────────────────────────


def dong_bao_cao(p: dict) -> Iterator[str]:
    tt = p["tom_tat"]
    yield f"# Khe hở năng lượng bên trong `long_event` — {p['run']} ({p['split']}/{p['subset']})"
    yield ""
    yield (f"Sinh bởi `ml/evaluation/long_event_gap.py`. {p['n_clip_long_event']} clip "
          f"lát cắt `long_event`, {p['n_su_kien_xet']} sự kiện đạt ngưỡng độ dài "
          f"≥{so(p['min_duration_sec'])}s (đã bỏ sự kiện ngắn hơn — không thể chứa khe hở "
          f"có ý nghĩa). Ngưỡng \"im\": RMS < {so(p['nguong_ti_le'] * 100, 0)}% trung vị "
          f"RMS của chính sự kiện đó.")
    yield ""
    yield ("> ⚠️ `data/synthetic/dev` sinh cùng recipe B0–B9 với train của v3 → thiên vị "
          "v3 theo thiết kế. θ\\* dùng ở đây lấy từ `analysis.json` của chính run — vẫn "
          "chọn trên tập đang chấm.")
    yield ""
    yield "## So khe hở: nhóm bị phân mảnh vs nhóm không, trong CÙNG lát cắt"
    yield ""
    yield "| nhóm | n sự kiện | khe hở trung vị | khe hở p75 |"
    yield "|---|---|---|---|"
    yield (f"| bị phân mảnh | {tt['n_vo']} | {so(tt['vo_trung_vi'], 3)}s | "
          f"{so(tt['vo_p75'], 3)}s |")
    yield (f"| không phân mảnh | {tt['n_nguyen']} | {so(tt['nguyen_trung_vi'], 3)}s | "
          f"{so(tt['nguyen_p75'], 3)}s |")
    yield ""
    if tt["n_vo"] == 0 or tt["n_nguyen"] == 0:
        yield ("**Một trong hai nhóm rỗng — không so được.** Nới `--min-duration-sec` "
              "hoặc kiểm tra lại lát cắt `long_event` của run này.")
        return
    chenh = tt["vo_trung_vi"] - tt["nguyen_trung_vi"]
    yield f"Chênh lệch trung vị (vỡ − nguyên) = **{so(chenh, 3)}s**."
    yield ""
    if chenh > 0.1:
        yield ("Nhóm bị phân mảnh có khe hở dài hơn rõ rệt — **ủng hộ** giả thuyết khe hở "
              "năng lượng thật. Cần nghe trực tiếp vài clip trong nhóm vỡ để xác nhận "
              "bằng tai, con số một mình không đủ.")
    else:
        yield ("Hai nhóm gần như không khác nhau — giả thuyết khe hở năng lượng thật bị "
              "**bác bỏ**. `long_event` đã loại BA ứng viên liên tiếp (cửa sổ lọc hẹp, "
              "trần cửa sổ thấp, khe hở năng lượng); nguyên nhân riêng của nó **vẫn chưa "
              "xác định**.")


# ── Chạy ─────────────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> int:
    nguon = RUNS_DIR / args.run / "predictions" / f"{args.split}_{args.subset}.npz"
    if not nguon.exists():
        print(f"❌ chưa có {nguon}\n   Chạy: -m ml.evaluation.predictions --run {args.run} "
              f"--split {args.split}")
        return 1
    slice_path = DATA_DIR / args.split / "slice_index.jsonl"
    if not slice_path.exists():
        print(f"❌ chưa có {slice_path} — không dựng được lát cắt long_event")
        return 1
    analysis_path = RUNS_DIR / args.run / "analysis.json"
    if not analysis_path.exists():
        print(f"❌ chưa có {analysis_path}\n   Chạy: -m ml.evaluation.error_analysis "
              f"--run {args.run} --split {args.split}  (cần θ* của run này)")
        return 1

    bat_dau = time.time()
    theta_sao = json.loads(analysis_path.read_text(encoding="utf-8"))["theta_sao"]
    du_doan = doc(nguon)
    khung = [du_doan.khung(i) for i in range(len(du_doan.clip_ids))]
    du_bao_sao = giai_ma(du_doan, khung, theta_sao, DEFAULT_MEDIAN_FILTER_FRAMES)
    tham_chieu = du_doan.tham_chieu()

    nhom = nhom_theo_lat_cat(doc_lat_cat(slice_path), du_doan.clip_ids)
    clip_ids_long_event = nhom.get("long_event", [])
    print(f"▶ {args.run} · {args.split}/{args.subset} · θ*={so(theta_sao)} · "
         f"{len(clip_ids_long_event)} clip long_event", flush=True)

    audio_dir = DATA_DIR / args.split / "audio"
    ket_qua = do_khe_ho_long_event(
        tham_chieu, du_bao_sao, audio_dir, clip_ids_long_event,
        hop_sec=args.hop_ms / 1000, nguong_ti_le=args.nguong_ti_le,
        min_duration_sec=args.min_duration_sec,
    )
    tt = tom_tat(ket_qua)
    print(f"  vỡ: n={tt['n_vo']} trung_vi={tt['vo_trung_vi']:.3f}s · "
         f"nguyên: n={tt['n_nguyen']} trung_vi={tt['nguyen_trung_vi']:.3f}s", flush=True)

    payload = {
        "run": args.run, "split": args.split, "subset": args.subset,
        "theta_sao": theta_sao, "hop_sec": args.hop_ms / 1000,
        "nguong_ti_le": args.nguong_ti_le, "min_duration_sec": args.min_duration_sec,
        "n_clip_long_event": len(clip_ids_long_event), "n_su_kien_xet": len(ket_qua),
        "su_kien": ket_qua, "tom_tat": tt,
        "giay": round(time.time() - bat_dau, 1),
        "ghi_chu": "dev tổng hợp thiên vị v3; θ* chọn trên chính tập đang chấm.",
    }
    out_json = RUNS_DIR / args.run / "long_event_gap.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md = viet_dan(RUNS_DIR / args.run / "long_event_gap.md", dong_bao_cao(payload))
    print(f"\n✓ {out_json.relative_to(REPO_ROOT)}\n✓ {out_md.relative_to(REPO_ROOT)} "
         f"· {payload['giay']:.0f}s", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--subset", default="all")
    parser.add_argument("--nguong-ti-le", type=float, default=NGUONG_TI_LE_MAC_DINH)
    parser.add_argument("--hop-ms", type=float, default=HOP_SEC_MAC_DINH * 1000)
    parser.add_argument("--min-duration-sec", type=float, default=MIN_DURATION_SEC_MAC_DINH)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
