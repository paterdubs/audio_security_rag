"""Kiểm định chính bộ sàng lọc tự động — DATA_PLAN §4.4.

    python scripts/validate_screen.py --sample      # rút mẫu, tạo phiếu để duyệt
    python scripts/validate_screen.py --score       # đọc phiếu đã duyệt, tính tỉ lệ lỗi

Không có bước này thì tự động hoá là vô căn cứ. Câu phải viết được trong khoá luận:

    "Foreground bank được sàng lọc tự động với ngưỡng τ = 0.30; kiểm định trên 10%
     mẫu ngẫu nhiên cho tỉ lệ lỗi X%."

Không có X thì không ai có lý do tin bank sạch. Script này sinh ra X, kèm khoảng tin
cậy — vì X đo trên một mẫu, và một con số trần không nói được mẫu ấy đủ lớn hay chưa.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

from common import REPO_ROOT, enable_utf8_output

SCORES_PATH = REPO_ROOT / "data" / "manifests" / "screen_scores.csv"
AUDIT_PATH = REPO_ROOT / "data" / "manifests" / "screen_audit.csv"
NORMALIZED_DIR = REPO_ROOT / "data" / "interim" / "normalized"

SAMPLE_FRACTION = 0.10       # DATA_PLAN §4.4
ERROR_BUDGET = 0.10          # vượt mức này → nâng ngưỡng và chạy lại TOÀN BỘ
SEED = 20260914              # cố định để mẫu tái lập được; ghi vào báo cáo

# CỐ Ý KHÔNG có p_target / p_confusable / decision. Phiếu này dặn người duyệt nghe mù,
# mà in sẵn điểm của máy ngay cạnh thì lời dặn ấy vô nghĩa: biết máy chấm 0.95 rồi mới
# nghe thì kết luận đã bị neo vào đó, và toàn bộ phép kiểm định mất giá trị — nó chỉ
# còn đo được mức độ người duyệt đồng ý với máy khi đã biết máy nghĩ gì.
# Điểm vẫn nằm nguyên trong screen_scores.csv và được ghép lại theo file_id ở --score.
AUDIT_FIELDS = ["file_id", "class_id", "onset", "offset", "path_norm", "verdict", "note"]
# Người duyệt chỉ điền cột `verdict` bằng một trong ba giá trị này.
VERDICTS = {"ok": "máy đúng", "wrong_class": "sai lớp", "bad_audio": "audio không dùng được"}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# ── Rút mẫu ──────────────────────────────────────────────────────────────────


def stratified_sample(accepted: list[dict], fraction: float, seed: int) -> list[dict]:
    """Rút mẫu PHÂN TẦNG THEO LỚP, không rút ngẫu nhiên toàn cục.

    Rút toàn cục thì lớp nhỏ có thể không có clip nào trong mẫu: `fireworks` chiếm 3%
    số clip, nên 10% ngẫu nhiên rất dễ bỏ sót nó hoàn toàn. Mà chính lớp nhỏ và lớp
    dễ nhầm mới là nơi bộ sàng lọc sai nhiều nhất — một tỉ lệ lỗi tổng thể đẹp đẽ
    được đóng góp toàn bởi `speech_normal` thì không nói lên điều gì.

    Mỗi lớp lấy ÍT NHẤT 1 clip nếu lớp đó có clip nào được tự động nhận.
    """
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = {}
    for row in accepted:
        by_class.setdefault(row["class_id"], []).append(row)

    sample = []
    for class_id in sorted(by_class):
        rows = sorted(by_class[class_id], key=lambda r: r["file_id"])   # tất định
        take = max(1, round(len(rows) * fraction))
        sample.extend(rng.sample(rows, min(take, len(rows))))
    return sorted(sample, key=lambda r: (r["class_id"], r["file_id"]))


def build_audit_sheet(scored: list[dict], fraction: float, seed: int) -> list[dict]:
    accepted = [r for r in scored if r["decision"] == "auto_accept"]
    if not accepted:
        raise SystemExit("❌ chưa có clip nào được tự động nhận — chạy scripts/auto_screen.py trước")
    return [{**row, "verdict": "", "note": ""} for row in stratified_sample(accepted, fraction, seed)]


# ── Tính tỉ lệ lỗi ───────────────────────────────────────────────────────────


def wilson_interval(errors: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Khoảng tin cậy 95% cho tỉ lệ lỗi, theo Wilson.

    KHÔNG dùng công thức chuẩn thông thường: cỡ mẫu ở đây nhỏ (vài chục clip) và tỉ
    lệ lỗi gần 0, đúng hai điều kiện làm công thức chuẩn cho ra cận dưới ÂM. Một
    khoảng tin cậy [-0.02, 0.08] in vào khoá luận là tự tố cáo mình dùng sai công cụ.
    """
    if total == 0:
        return 0.0, 1.0
    p = errors / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


def error_rate(audited: list[dict]) -> tuple[int, int, list[dict]]:
    """(số lỗi, số đã duyệt, các dòng chưa điền). `bad_audio` KHÔNG tính là lỗi sàng lọc.

    Bộ sàng lọc trả lời câu hỏi "clip này có đúng lớp không". Một clip đúng lớp nhưng
    thu hỏng là lỗi của bước chuẩn hoá, không phải của ngưỡng τ. Gộp chung sẽ làm tỉ
    lệ lỗi phồng lên và đẩy ta đi nâng ngưỡng để chữa một bệnh khác.
    """
    missing = [row for row in audited if row.get("verdict", "").strip() not in VERDICTS]
    judged = [row for row in audited if row.get("verdict", "").strip() in VERDICTS]
    errors = sum(1 for row in judged if row["verdict"].strip() == "wrong_class")
    return errors, len(judged), missing


def report(errors: int, total: int, audited: list[dict]) -> bool:
    """In kết quả, trả True nếu đạt ngưỡng chấp nhận."""
    rate = errors / total if total else 0.0
    low, high = wilson_interval(errors, total)
    bad_audio = sum(1 for r in audited if r.get("verdict", "").strip() == "bad_audio")

    print(f"\n▶ đã duyệt {total} clip · {errors} sai lớp · {bad_audio} audio hỏng (không tính là lỗi sàng lọc)")
    print(f"\n  tỉ lệ lỗi auto-accept: {rate:.1%}   (KTC 95% Wilson: {low:.1%}–{high:.1%})")

    print(f"\n{'lớp':<22}{'duyệt':>7}{'sai':>6}{'tỉ lệ':>9}")
    print("─" * 44)
    by_class: dict[str, list[dict]] = {}
    for row in audited:
        if row.get("verdict", "").strip() in VERDICTS:
            by_class.setdefault(row["class_id"], []).append(row)
    for class_id in sorted(by_class):
        rows = by_class[class_id]
        sai = sum(1 for r in rows if r["verdict"].strip() == "wrong_class")
        print(f"{class_id:<22}{len(rows):>7}{sai:>6}{sai / len(rows):>8.0%}")

    if rate > ERROR_BUDGET:
        print(f"\n❌ {rate:.1%} > {ERROR_BUDGET:.0%}: phải NÂNG ngưỡng p_target trong auto_screen.py")
        print("   rồi chạy lại TOÀN BỘ bước sàng lọc — DATA_PLAN §4.4 bước 3.")
        return False

    # Cận trên mới là con số phải nhìn: rate = 0/20 vẫn cho cận trên 16%, tức mẫu 20
    # clip KHÔNG đủ để khẳng định dưới 10% dù không thấy lỗi nào.
    if high > ERROR_BUDGET:
        print(f"\n⚠️ {rate:.1%} đạt ngưỡng, NHƯNG cận trên {high:.1%} vẫn vượt {ERROR_BUDGET:.0%}.")
        print("   Mẫu chưa đủ lớn để khẳng định. Duyệt thêm clip rồi chạy lại --score.")
    else:
        print(f"\n✓ {rate:.1%} ≤ {ERROR_BUDGET:.0%} — đạt. Câu viết được vào khoá luận:")
        print(f'   "Foreground bank được sàng lọc tự động với ngưỡng τ = 0.30; kiểm định')
        print(f'    trên {total} clip mẫu ngẫu nhiên phân tầng cho tỉ lệ lỗi {rate:.1%}')
        print(f'    (KTC 95%: {low:.1%}–{high:.1%})."')
    return True


# ── Chạy ─────────────────────────────────────────────────────────────────────


def cmd_sample(args) -> int:
    scored = read_csv(SCORES_PATH)
    if not scored:
        print("❌ chưa có screen_scores.csv — chạy scripts/auto_screen.py trước")
        return 1
    if AUDIT_PATH.exists() and not args.force:
        print(f"❌ {AUDIT_PATH.name} đã tồn tại. Ghi đè sẽ xoá công duyệt đã bỏ ra.")
        print("   Dùng --force nếu thật sự muốn rút mẫu mới.")
        return 1

    sheet = build_audit_sheet(scored, args.fraction, args.seed)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sheet)

    accepted = sum(1 for r in scored if r["decision"] == "auto_accept")
    print(f"▶ rút {len(sheet)}/{accepted} clip tự động nhận ({len(sheet) / accepted:.1%}), "
          f"phân tầng theo lớp · seed {args.seed}")
    print(f"\n✓ đã ghi {AUDIT_PATH.relative_to(REPO_ROOT)}")
    print("\nGiờ mở file đó, nghe từng clip và điền cột `verdict` bằng một trong:")
    for value, meaning in VERDICTS.items():
        print(f"    {value:<12} {meaning}")
    print("\n⚠️ Nghe MÙ: đừng nhìn cột p_target trước khi quyết. Biết máy chấm 0.9 rồi mới")
    print("   nghe thì kết luận của bạn đã bị neo vào đó, và phép kiểm định mất giá trị.")
    print("\nXong thì chạy: python scripts/validate_screen.py --score")
    return 0


def cmd_score() -> int:
    audited = read_csv(AUDIT_PATH)
    if not audited:
        print("❌ chưa có screen_audit.csv — chạy --sample trước")
        return 1

    errors, total, missing = error_rate(audited)
    if missing:
        print(f"⚠️ {len(missing)}/{len(audited)} dòng chưa điền `verdict`, đang bỏ qua:")
        for row in missing[:3]:
            print(f"      {row['file_id']}")
    if total == 0:
        print("❌ chưa dòng nào được duyệt — không tính được gì")
        return 1

    return 0 if report(errors, total, audited) else 1


def main() -> int:
    enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", action="store_true", help="rút mẫu, tạo phiếu duyệt")
    group.add_argument("--score", action="store_true", help="đọc phiếu đã duyệt, tính tỉ lệ lỗi")
    parser.add_argument("--fraction", type=float, default=SAMPLE_FRACTION)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true", help="ghi đè phiếu cũ")
    args = parser.parse_args()

    return cmd_sample(args) if args.sample else cmd_score()


if __name__ == "__main__":
    sys.exit(main())
