"""Kiểm định thủ công bộ trích lexicon trên 100 caption — điều kiện CẦN của đóng góp C2
(SYSTEM.md §8.2). Hai bước, hai lệnh:

    # 1. Sinh bảng 100 caption cần người đọc và điền tay
    .venv/Scripts/python.exe scripts/kiem_dinh_lexicon.py sinh --run panns_ft_pw1

    # (mở data/gold/kiem_dinh_lexicon_100.csv, điền cột "sai" và "thieu" — xem §2 dưới)

    # 2. Sau khi điền xong: tính độ chính xác của bộ trích
    .venv/Scripts/python.exe scripts/kiem_dinh_lexicon.py cham

⚠️ Đây là kiểm định LEXICON (bước trích P từ văn bản caption), KHÔNG PHẢI kiểm định
grounding của model. Hai thứ khác nhau: `ml/evaluation/caption_b0.py` đo model có nói
đúng sự thật không (EHR/EOR so P với G thật); script này đo lexicon có ĐỌC ĐÚNG caption
không (câu nói X, lexicon có bắt được X không) — độc lập với việc X có đúng sự thật hay
không. SYSTEM.md §8.2 đòi phải đo riêng cái thứ hai, và báo cáo độ chính xác của CHÍNH
bộ trích, nếu không thì EHR/EOR không đáng tin dù model có tốt cỡ nào.

Cách điền hai cột trong CSV sinh ra:

    sai   = các lớp trong cột P_tu_dong mà caption KHÔNG THỰC SỰ nói tới (lexicon trích
            nhầm) — ví dụ P_tu_dong có "gunshot" nhưng câu chỉ nói "he took a snapshot".
    thieu = các lớp caption CÓ nói tới mà P_tu_dong bỏ sót (lexicon trích sót) — ví dụ
            câu nói "a person screaming" nhưng P_tu_dong không có "scream".

Cả hai cột đều để TRỐNG nếu không có lỗi. Nhiều lớp thì cách nhau bằng dấu phẩy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from common import enable_utf8_output  # noqa: E402

enable_utf8_output()

from ml.evaluation.caption_b0 import danh_gia_mot_clip  # noqa: E402
from ml.evaluation.error_analysis import cua_so_loc_tu_train, giai_ma  # noqa: E402
from ml.evaluation.predictions import doc  # noqa: E402

RUNS_DIR = REPO_ROOT / "ml" / "runs"
BANG_PATH = REPO_ROOT / "data" / "gold" / "kiem_dinh_lexicon_100.csv"
BAO_CAO_PATH = REPO_ROOT / "docs" / "measurements" / "kiem_dinh_lexicon_20260922.md"

NHOM_TOI_DA = 3   # bucket "≥3 sự kiện" gộp chung — câu 3 hay 5 sự kiện cùng một cấu trúc


# ── Chọn mẫu rải theo độ phức tạp câu ────────────────────────────────────────


def _hash_fraction(seed: str, key: str) -> float:
    """Cùng công thức với `select_gold_pilot.hash_fraction` — số thực ổn định [0,1)."""
    digest = hashlib.sha256(f"{seed}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def chon_mau_kiem_dinh(ket_qua: list[dict], n: int, seed: str) -> list[dict]:
    """Round-robin theo SỐ SỰ KIỆN caption nhắc tới (0/1/2/≥3) — cùng thuật toán với
    `select_gold_pilot.chon_pilot`, chỉ đổi trục nhóm từ "lớp" sang "độ phức tạp câu".

    Lý do phải rải đều: B0 dùng BA cấu trúc cú pháp khác nhau tuỳ số sự kiện (câu đơn,
    "X sau đó Y", "X và Y cùng lúc" — xem `captioner.py`). Mẫu ngẫu nhiên thuần tuý có
    thể ra toàn câu đơn giản nhất, không kiểm được lexicon trên câu phức.
    """
    nhom: dict[int, list[dict]] = defaultdict(list)
    for r in ket_qua:
        nhom[min(len(r["P"]), NHOM_TOI_DA)].append(r)
    for so in nhom:
        nhom[so] = sorted(nhom[so], key=lambda r: _hash_fraction(seed, r["clip_id"]))

    da_chon: list[dict] = []
    thu_tu_nhom = sorted(nhom)
    con_tien_trien = True
    while len(da_chon) < n and con_tien_trien:
        con_tien_trien = False
        for so in thu_tu_nhom:
            if len(da_chon) >= n:
                break
            if nhom[so]:
                da_chon.append(nhom[so].pop(0))
                con_tien_trien = True
    return da_chon


# ── Đọc / ghi bảng kiểm ───────────────────────────────────────────────────────


def viet_bang_kiem(mau: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_id", "caption", "P_tu_dong", "sai", "thieu", "ghi_chu"])
        for r in mau:
            writer.writerow([r["clip_id"], r["caption"], ";".join(r["P"]), "", "", ""])


def doc_bang_kiem(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def tach_danh_sach(o: str) -> list[str]:
    """'gunshot, siren' -> ['gunshot', 'siren']; '' -> []. Chấp nhận cả phẩy lẫn chấm phẩy
    vì người điền tay dễ lẫn hai thứ này."""
    o = (o or "").strip()
    if not o:
        return []
    return [x.strip() for x in o.replace(";", ",").split(",") if x.strip()]


# ── Tính độ chính xác ─────────────────────────────────────────────────────────


def tinh_do_chinh_xac(rows: list[dict]) -> dict:
    """Precision/Recall của CHÍNH bộ trích lexicon, từ bảng đã người điền `sai`/`thieu`.

    `do_chinh_xac` (precision): trong số lớp lexicon đã trích, bao nhiêu phần trăm là
    trích đúng. `do_bao_phu` (recall): trong số lớp lexicon LẼ RA phải trích được (trích
    đúng + trích sót), bao nhiêu phần trăm nó bắt được. Không hàng nào trích được gì
    (mẫu số 0) -> None, không phải 1 hay 0 — xem test cho lý do cụ thể.
    """
    tong_trich = tong_sai = tong_thieu = 0
    n_hoan_toan_dung = 0
    for r in rows:
        P = tach_danh_sach(r["P_tu_dong"])
        sai = tach_danh_sach(r["sai"])
        thieu = tach_danh_sach(r["thieu"])
        tong_trich += len(P)
        tong_sai += len(sai)
        tong_thieu += len(thieu)
        if not sai and not thieu:
            n_hoan_toan_dung += 1

    dung_that = tong_trich - tong_sai
    return {
        "n_caption": len(rows),
        "do_chinh_xac": (dung_that / tong_trich) if tong_trich else None,
        "do_bao_phu": (dung_that / (dung_that + tong_thieu)) if (dung_that + tong_thieu) else None,
        "ti_le_hoan_toan_dung": (n_hoan_toan_dung / len(rows)) if rows else None,
        "tong_trich": tong_trich, "tong_sai": tong_sai, "tong_thieu": tong_thieu,
    }


# ── Lệnh "sinh" ────────────────────────────────────────────────────────────────


def gom_ket_qua(run: str, split: str = "dev", subset: str = "all") -> list[dict]:
    """Chấm caption B0 cho TOÀN BỘ clip của một run — dùng chung θ*/cửa sổ thích ứng-từ-
    train đã có trong `analysis_adaptive_train.json`, không quét ngưỡng lại."""
    import json

    run_dir = RUNS_DIR / run
    du_doan = doc(run_dir / "predictions" / f"{split}_{subset}.npz")
    theta_sao = json.loads((run_dir / "analysis_adaptive_train.json")
                          .read_text(encoding="utf-8"))["theta_sao"]
    median_size = cua_so_loc_tu_train(du_doan.class_ids, du_doan.n_frames / du_doan.duration)
    khung = [du_doan.khung(i) for i in range(len(du_doan.clip_ids))]
    du_bao = giai_ma(du_doan, khung, theta_sao, median_size)
    tham_chieu = du_doan.tham_chieu()
    return [
        {"clip_id": c, **danh_gia_mot_clip(du_bao[c], tham_chieu[c])}
        for c in du_doan.clip_ids
    ]


def lenh_sinh(args: argparse.Namespace) -> int:
    ket_qua = gom_ket_qua(args.run, args.split)
    mau = chon_mau_kiem_dinh(ket_qua, args.n, args.seed)
    viet_bang_kiem(mau, args.out)
    print(f"✓ {args.out.relative_to(REPO_ROOT)} · {len(mau)} caption cần kiểm định thủ công")
    print("  Điền cột 'sai' (lexicon trích NHẦM) và 'thieu' (lexicon trích SÓT), để trống")
    print("  nếu không có lỗi. Xong thì chạy: kiem_dinh_lexicon.py cham")
    return 0


# ── Lệnh "chấm" ─────────────────────────────────────────────────────────────────


def lenh_cham(args: argparse.Namespace) -> int:
    if not args.bang.exists():
        print(f"❌ chưa có {args.bang}\n   Chạy: kiem_dinh_lexicon.py sinh trước.")
        return 1
    rows = doc_bang_kiem(args.bang)
    kq = tinh_do_chinh_xac(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        f.write("# Kiểm định thủ công bộ trích lexicon — 100 caption (SYSTEM.md §8.2)\n\n")
        f.write(f"`n_caption` = {kq['n_caption']}. Tổng cụm trích = {kq['tong_trich']}, "
               f"sai = {kq['tong_sai']}, thiếu = {kq['tong_thieu']}.\n\n")
        for ten, gia_tri in (("Độ chính xác (precision)", kq["do_chinh_xac"]),
                             ("Độ bao phủ (recall)", kq["do_bao_phu"]),
                             ("Tỉ lệ caption hoàn toàn đúng", kq["ti_le_hoan_toan_dung"])):
            f.write(f"- **{ten}**: " + ("chưa đo" if gia_tri is None
                                        else f"{gia_tri:.4f}".replace(".", ",")) + "\n")

    print(f"✓ {args.out.relative_to(REPO_ROOT)}")
    for ten, gia_tri in (("độ chính xác", kq["do_chinh_xac"]), ("độ bao phủ", kq["do_bao_phu"]),
                        ("tỉ lệ hoàn toàn đúng", kq["ti_le_hoan_toan_dung"])):
        print(f"  {ten}: " + ("chưa đo" if gia_tri is None else f"{gia_tri:.4f}"))
    print("\n  ⚠️ Nếu số liệu chấp nhận được: mở ml/configs/event_lexicon.yaml, đổi "
          "meta.da_kiem_dinh_thu_cong -> true, ghi số đo vào đó.")
    return 0


# ── Chạy ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="lenh", required=True)

    p_sinh = sub.add_parser("sinh")
    p_sinh.add_argument("--run", required=True)
    p_sinh.add_argument("--split", default="dev")
    p_sinh.add_argument("--n", type=int, default=100)
    p_sinh.add_argument("--seed", default="kltn-2026-kiem-dinh-lexicon")
    p_sinh.add_argument("--out", type=Path, default=BANG_PATH)

    p_cham = sub.add_parser("cham")
    p_cham.add_argument("--bang", type=Path, default=BANG_PATH)
    p_cham.add_argument("--out", type=Path, default=BAO_CAO_PATH)

    args = parser.parse_args()
    return lenh_sinh(args) if args.lenh == "sinh" else lenh_cham(args)


if __name__ == "__main__":
    raise SystemExit(main())
