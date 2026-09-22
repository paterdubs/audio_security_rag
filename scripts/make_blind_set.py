"""Dựng bộ gán lại mù cho pilot lần 2 — DATA_PLAN §8.3 bước 2 / §8.4.

    python scripts/make_blind_set.py --n-moi 30 --seed kltn-2026-gold-blind-v1-lan2

§8.4 đòi 4 điều kiện mù. Script này lo điều kiện 2 (băm tên + xáo thứ tự) và điều kiện
4 (trộn clip mồi để không lộ khối 30 cũ). Điều kiện 1 (dự án Label Studio mới, không
import nhãn lần 1) và điều kiện 3 (nghỉ ≥3 ngày) là việc của người, script không lo được.

Mồi lấy từ chính pool `gold_test` CHƯA gán (không phải 30 clip pilot) — vừa che giấu
khối cũ, vừa tận dụng công nghe: nhãn của mồi dùng thẳng được cho bước 4 (gán đại trà).
Mồi KHÔNG cần round-robin theo lớp như `select_gold_pilot.chon_pilot` — mục đích của mồi
là che giấu, không phải phủ đều lớp; phủ lớp là việc của bước gán đại trà sau.

⚠️ `--mapping-out` ghi bảng ten_mu → file_id thật + cờ la_pilot. KHÔNG mở file này trong
lúc gán nhãn — mở ra là lộ luôn đâu là 30 clip cũ, hỏng số đo tự-nhất-quán.

⚠️ Vòng SAU lần đầu (ví dụ vòng 3): `--pilot` trỏ tới file pilot MỚI (không phải
`pilot_v1_candidates.csv`), và bắt buộc truyền `--tru-them` gồm cột `file_id` của MỌI
clip đã nghe ở các vòng trước (pilot cũ + mồi cũ) — nếu không, mồi của vòng mới có thể
trùng đúng clip vòng cũ mà người gán đã nghe rồi, và điều kiện mù §8.4 hỏng dù tên đã
băm lại. `chon_moi()` chỉ tự loại pilot của CHÍNH vòng đang chạy, không biết gì về các
vòng trước nếu không được báo qua `--tru-them`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import REPO_ROOT, enable_utf8_output  # noqa: E402
from select_gold_pilot import doc_gold_theo_lop  # noqa: E402

PILOT_CANDIDATES_PATH = REPO_ROOT / "data" / "gold" / "pilot_v1_candidates.csv"
SPLITS_PATH = REPO_ROOT / "data" / "manifests" / "splits.csv"
SEGMENTS_PATH = REPO_ROOT / "data" / "raw" / "audioset_strong" / "segments.jsonl"
RAW_MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "raw_manifest.csv"
EXCLUSIONS_PATH = REPO_ROOT / "data" / "manifests" / "exclusions.csv"
OUT_DIR = REPO_ROOT / "data" / "gold" / "blind_v1_lan2"
MAPPING_PATH = REPO_ROOT / "data" / "gold" / "blind_v1_lan2_mapping.csv"

DEFAULT_N_MOI = 30
DEFAULT_SEED = "kltn-2026-gold-blind-v1-lan2"


def hash_fraction(seed: str, key: str) -> float:
    """Số thực ổn định trong [0,1) từ một khoá — cùng công thức với
    `select_gold_pilot.hash_fraction`."""
    digest = hashlib.sha256(f"{seed}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def hash_ten(seed: str, file_id: str) -> str:
    """Tên file băm, không chứa manh mối nào của `file_id` gốc — tên file cũng là manh
    mối gợi nhớ y như thứ tự, phải giấu cả hai (§8.4 điều kiện 2)."""
    digest = hashlib.sha256(f"{seed}|blind|{file_id}".encode()).hexdigest()
    return f"blind_{digest[:16]}.wav"


def chon_moi(pool_thong_tin: dict[str, dict], da_dung: set[str], n: int, seed: str) -> list[str]:
    """`n` file_id mồi từ `pool_thong_tin`, loại `da_dung` (30 clip pilot). Sắp theo
    hash_fraction rồi lấy đầu — pool ít hơn `n` thì trả về hết, không lỗi, không lặp vô
    hạn (cùng bẫy đã gặp ở `chon_pilot` với lớp hiếm hết file)."""
    con_lai = sorted(
        (fid for fid in pool_thong_tin if fid not in da_dung),
        key=lambda fid: hash_fraction(seed, fid),
    )
    return con_lai[:n]


def xep_lich_mu(pilot_ids: list[str], moi_ids: list[str], seed: str) -> list[str]:
    """Thứ tự hiển thị/copy cuối cùng: hoán vị của pilot ∪ mồi, xáo bằng hash_fraction
    với salt riêng (khác salt dùng để chọn mồi) — khối 30 pilot không được đứng liền
    nhau, nếu không người gán vẫn nhận ra khối cũ dù tên đã băm."""
    ca_hai = list(pilot_ids) + list(moi_ids)
    return sorted(ca_hai, key=lambda fid: hash_fraction(seed, f"order|{fid}"))


def dung_anh_xa(order: list[str], pilot_ids: set[str], seed: str) -> list[dict]:
    """[{file_id, ten_mu, la_pilot}] — một dòng cho mỗi file_id trong `order`, tên băm
    duy nhất trong toàn bộ lượt chạy (salt gồm cả file_id nên không đụng độ thực tế)."""
    return [
        {"file_id": fid, "ten_mu": hash_ten(seed, fid), "la_pilot": fid in pilot_ids}
        for fid in order
    ]


def sao_chep_file(anh_xa: list[dict], thong_tin: dict[str, dict], out_dir: Path,
                  goc: Path = REPO_ROOT) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in anh_xa:
        nguon = goc / thong_tin[row["file_id"]]["path"]
        shutil.copyfile(nguon, out_dir / row["ten_mu"])


def ghi_anh_xa(anh_xa: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ten_mu", "file_id", "la_pilot"])
        writer.writeheader()
        for row in anh_xa:
            writer.writerow(row)


def doc_pilot_ids(pilot_path: Path) -> list[str]:
    with pilot_path.open(encoding="utf-8", newline="") as handle:
        return [row["file_id"] for row in csv.DictReader(handle)]


def run(args: argparse.Namespace) -> int:
    goc = getattr(args, "repo_root", REPO_ROOT)
    if not args.pilot.exists():
        print(f"❌ chưa có {args.pilot}\n   Chạy: scripts/select_gold_pilot.py trước.")
        return 1
    if not args.splits.exists() or not args.segments.exists():
        print("❌ thiếu splits.csv hoặc segments.jsonl.")
        return 1

    pilot_ids = doc_pilot_ids(args.pilot)
    tru_them = getattr(args, "tru_them", None)
    du_lieu = doc_gold_theo_lop(args.splits, args.segments, args.raw_manifest,
                                args.exclusions, tru_them)
    pool_thong_tin = du_lieu["thong_tin"]

    moi_ids = chon_moi(pool_thong_tin, set(pilot_ids), args.n_moi, args.seed)
    if len(moi_ids) < args.n_moi:
        print(f"⚠️ pool chỉ còn {len(moi_ids)} file mồi khả dụng, ít hơn --n-moi={args.n_moi}.")
    print(f"▶ {len(pilot_ids)} clip pilot + {len(moi_ids)} clip mồi = {len(pilot_ids) + len(moi_ids)} clip (seed={args.seed!r})")

    thu_tu = xep_lich_mu(pilot_ids, moi_ids, args.seed)
    anh_xa = dung_anh_xa(thu_tu, set(pilot_ids), args.seed)

    # Mồi cũng cần path — lấy từ pool_thong_tin; pilot lấy từ pilot_v1_candidates.csv
    # phòng khi pilot đã bị loại khỏi pool (không nên xảy ra, nhưng an toàn hơn suy diễn).
    with args.pilot.open(encoding="utf-8", newline="") as handle:
        pilot_thong_tin = {row["file_id"]: {"path": row["path"]} for row in csv.DictReader(handle)}
    thong_tin_day_du = {**pool_thong_tin, **pilot_thong_tin}

    sao_chep_file(anh_xa, thong_tin_day_du, args.out_dir, goc=goc)
    ghi_anh_xa(anh_xa, args.mapping_out)

    print(f"✓ {len(anh_xa)} file WAV → {args.out_dir}")
    print(f"✓ mapping → {args.mapping_out}")
    print("  ⚠️ KHÔNG mở mapping trong lúc gán nhãn — lộ là hỏng số đo tự-nhất-quán.")
    return 0


def main() -> int:
    enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", type=Path, default=PILOT_CANDIDATES_PATH)
    parser.add_argument("--n-moi", type=int, default=DEFAULT_N_MOI)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--splits", type=Path, default=SPLITS_PATH)
    parser.add_argument("--segments", type=Path, default=SEGMENTS_PATH)
    parser.add_argument("--raw-manifest", type=Path, default=RAW_MANIFEST_PATH)
    parser.add_argument("--exclusions", type=Path, default=EXCLUSIONS_PATH)
    parser.add_argument("--tru-them", type=Path, default=None, dest="tru_them",
                        help="CSV cột file_id của MỌI clip đã nghe ở các vòng trước "
                             "(pilot cũ + mồi cũ) — bắt buộc từ vòng thứ hai trở đi")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--mapping-out", type=Path, default=MAPPING_PATH)
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
