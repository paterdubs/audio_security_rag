"""Nối B0 (`services/inference/app/captioner.py`) với bộ metric hallucination — W4,
SYSTEM.md §8.2/§8.5, tiếp theo `ml/evaluation/hallucination.py`.

    .venv/Scripts/python.exe -m ml.evaluation.caption_b0 --run panns_ft_pw1 --split dev

Đây là phép đo **tự nhất quán**, không phải phép đo cuối cùng cho khoá luận: B0 sinh
caption thẳng từ TIMELINE DỰ BÁO của chính model (không phải từ nhãn tham chiếu), rồi
EHR/EOR/GS/TOA/CHR so caption đó với NHÃN THAM CHIẾU strong-label sẵn có trên
`data/synthetic/dev`. Không cần G2 (chưa có) vì các metric này không cần human reference,
chỉ cần văn bản caption + tập sự kiện thật.

**Đọc kết quả cho đúng:** B0 là cận trên về grounding theo định nghĩa (mọi từ trong caption
đều sinh từ một detection có thật). Nếu EHR ở đây > 0 thì lý do CHỈ có thể là detection của
model không khớp tham chiếu (model dự báo sai) — đó là tín hiệu thật về model. Nếu lexicon
không nhận ra cụm mà chính B0 sinh ra thì đó là lỗi hạ tầng (xem
`tests/test_ml_hallucination.py::test_lexicon_nhan_ra_moi_cum_tu_cua_captioner_B0`), không
phải hallucination — hai nguyên nhân này KHÔNG được trộn khi đọc số ra từ đây.

Dùng chung θ\* và cửa sổ lọc thích ứng-từ-train đã ghi sẵn trong `analysis_adaptive_train.json`
của run — đúng chuẩn báo cáo mới (docs/measurements/chuan_bao_cao_20260922.md), không quét
ngưỡng lại.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "services" / "inference"))

from common import enable_utf8_output  # noqa: E402

enable_utf8_output()

from app.captioner import caption_en, caption_vi  # noqa: E402

from ml.evaluation.error_analysis import cua_so_loc_tu_train, giai_ma  # noqa: E402
from ml.evaluation.hallucination import (  # noqa: E402
    chr_critical,
    ehr,
    eor,
    grounding_score,
    toa,
    trich_su_kien,
)
from ml.evaluation.predictions import doc  # noqa: E402

RUNS_DIR = REPO_ROOT / "ml" / "runs"

CAPTIONER = {"en": caption_en, "vi": caption_vi}
METRICS = ("ehr", "eor", "gs", "chr", "toa")


# ── Đổi định dạng ─────────────────────────────────────────────────────────────


def doi_dinh_dang_su_kien(events: list[dict]) -> list[dict]:
    """`{event_label, onset, offset}` (sed_eval/dự báo) → `{class_id, onset, offset}`
    (captioner đòi khoá này). Lẫn khoá là lỗi câm: caption sẽ luôn rỗng, không lỗi nào báo."""
    return [{"class_id": e["event_label"], "onset": e["onset"], "offset": e["offset"]}
            for e in events]


def lop_that_cua_clip(events: list[dict]) -> set[str]:
    return {e["event_label"] for e in events}


def onset_theo_lop(events: list[dict]) -> dict[str, float]:
    """Onset SỚM NHẤT của mỗi lớp trong tham chiếu — dùng làm `onset_that` cho `toa()`."""
    ra: dict[str, float] = {}
    for e in events:
        lop, on = e["event_label"], e["onset"]
        if lop not in ra or on < ra[lop]:
            ra[lop] = on
    return ra


# ── Đánh giá một clip ────────────────────────────────────────────────────────


def danh_gia_mot_clip(du_bao_events: list[dict], tham_chieu_events: list[dict],
                      ngon_ngu: str = "en") -> dict:
    """`trich_su_kien()` dùng CHUNG một lexicon gộp EN+VI (xem `event_lexicon.yaml`),
    nên hoạt động như nhau bất kể `ngon_ngu` — không cần nhánh riêng cho tiếng Việt."""
    caption = CAPTIONER[ngon_ngu](doi_dinh_dang_su_kien(du_bao_events))
    P = trich_su_kien(caption)
    G = lop_that_cua_clip(tham_chieu_events)
    onset_that = onset_theo_lop(tham_chieu_events)
    return {
        "caption": caption, "P": sorted(P), "G": sorted(G),
        "ehr": ehr(P, G), "eor": eor(P, G), "gs": grounding_score(P, G),
        "chr": chr_critical(P, G), "toa": toa(caption, onset_that),
    }


# ── Tổng hợp ─────────────────────────────────────────────────────────────────


def tong_hop(ket_qua: list[dict]) -> dict[str, dict]:
    """Trung bình MỖI metric, bỏ qua `None` — không được coi `None` là 0."""
    ra: dict[str, dict] = {}
    for m in METRICS:
        gia_tri = [k[m] for k in ket_qua if k[m] is not None]
        ra[m] = {
            "trung_binh": (sum(gia_tri) / len(gia_tri)) if gia_tri else None,
            "n_do_duoc": len(gia_tri),
            "n_chua_do": len(ket_qua) - len(gia_tri),
        }
    return ra


# ── Báo cáo ──────────────────────────────────────────────────────────────────


def dong_bao_cao(run: str, tong: dict[str, dict], n_clip: int) -> Iterator[str]:
    yield f"# B0 × hallucination — {run} trên dev (tự nhất quán, không phải G2) — 2026-09-22"
    yield ""
    yield ("Caption B0 sinh từ timeline DỰ BÁO của chính model, so với nhãn tham chiếu "
           "strong-label. EHR > 0 ở đây nghĩa là model dự báo sai, KHÔNG phải lỗi lexicon "
           "(lexicon đã kiểm bất biến với mọi cụm B0 sinh ra — xem test_ml_hallucination.py).")
    yield ""
    yield f"`n_clip` = {n_clip}."
    yield ""
    yield "| metric | trung bình | đo được | chưa đo |"
    yield "|---|---:|---:|---:|"
    ten = {"ehr": "EHR↓", "eor": "EOR↓", "gs": "GS↑", "chr": "CHR↓", "toa": "TOA↑"}
    for m in METRICS:
        t = tong[m]
        tb = "—" if t["trung_binh"] is None else f"{t['trung_binh']:.4f}".replace(".", ",")
        yield f"| {ten[m]} | {tb} | {t['n_do_duoc']} | {t['n_chua_do']} |"
    yield ""


# ── Chạy ─────────────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> int:
    run_dir = RUNS_DIR / args.run
    nguon = run_dir / "predictions" / f"{args.split}_{args.subset}.npz"
    ana_path = run_dir / "analysis_adaptive_train.json"
    if not nguon.exists():
        print(f"❌ chưa có {nguon}")
        return 1
    if not ana_path.exists():
        print(f"❌ chưa có {ana_path}\n   Chạy: -m ml.evaluation.error_analysis --run "
              f"{args.run} --split {args.split} --adaptive-postproc --adaptive-source train")
        return 1

    bat_dau = time.time()
    du_doan = doc(nguon)
    theta_sao = json.loads(ana_path.read_text(encoding="utf-8"))["theta_sao"]
    median_size = cua_so_loc_tu_train(du_doan.class_ids, du_doan.n_frames / du_doan.duration,
                                      max_frames=args.max_median_frames)
    khung = [du_doan.khung(i) for i in range(len(du_doan.clip_ids))]
    du_bao = giai_ma(du_doan, khung, theta_sao, median_size)
    tham_chieu = du_doan.tham_chieu()

    ket_qua = [danh_gia_mot_clip(du_bao[c], tham_chieu[c], args.ngon_ngu)
              for c in du_doan.clip_ids]
    tong = tong_hop(ket_qua)

    duong_dan = run_dir / f"caption_b0_{args.split}.json"
    duong_dan.write_text(json.dumps({
        "run": args.run, "split": args.split, "ngon_ngu": args.ngon_ngu,
        "theta_sao": theta_sao, "n_clip": len(du_doan.clip_ids),
        "tong_hop": tong,
        "vi_du": ket_qua[:5],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = run_dir / f"caption_b0_{args.split}.md"
    with md_path.open("w", encoding="utf-8") as f:
        for dong in dong_bao_cao(args.run, tong, len(du_doan.clip_ids)):
            f.write(dong + "\n")

    print(f"✓ {duong_dan.relative_to(REPO_ROOT)} · {time.time() - bat_dau:.0f}s", flush=True)
    for m in METRICS:
        t = tong[m]
        tb = "chưa đo" if t["trung_binh"] is None else f"{t['trung_binh']:.4f}"
        print(f"  {m}: {tb} (đo được {t['n_do_duoc']}/{len(ket_qua)})", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--subset", default="all")
    parser.add_argument("--ngon-ngu", choices=("en", "vi"), default="en", dest="ngon_ngu")
    parser.add_argument("--max-median-frames", type=int, default=51)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
