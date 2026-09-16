"""Đưa clip đã duyệt vào foreground bank, cắt gọn theo biên — DATA_PLAN §3, §4.3.

    python scripts/promote_to_bank.py --dry-run
    python scripts/promote_to_bank.py
    python scripts/promote_to_bank.py --include-reviewed

Đây là mắt xích giữa sàng lọc và Scaper. `data/interim/normalized/` giữ clip nguyên
vẹn; bank chứa bản ĐÃ CẮT GỌN quanh sự kiện — đó là thứ Scaper cần, vì Scaper đặt cả
file nó nhận được vào soundscape và ghi onset/offset theo đúng độ dài file đó. Đưa
clip chưa cắt vào bank nghĩa là mọi nhãn strong sinh ra sau này đều bao cả phần im
lặng trước và sau sự kiện — sai một cách hệ thống, và không có gì báo.

Cắt là thao tác một chiều, nên nó KHÔNG đụng tới `interim/normalized/`: muốn đổi biên
thì xoá bank và chạy lại.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import soundfile

from common import REPO_ROOT, enable_utf8_output

SCORES_PATH = REPO_ROOT / "data" / "manifests" / "screen_scores.csv"
AUDIT_PATH = REPO_ROOT / "data" / "manifests" / "screen_audit.csv"
SPLITS_PATH = REPO_ROOT / "data" / "manifests" / "splits.csv"
RAW_MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "raw_manifest.csv"
BANK_DIR = REPO_ROOT / "data" / "banks" / "foreground"
MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "foreground_manifest.csv"

FIELDS = ["file_id", "class_id", "source_dataset", "onset", "offset", "duration_bank",
          "provenance", "path_bank"]
# Nới thêm khi cắt, ngoài phần auto_screen đã nới. Biên do máy đề xuất bám sát đỉnh
# năng lượng; cắt đúng sát mép sẽ chặt mất đuôi vang tự nhiên của tiếng kính vỡ hay
# tiếng súng, và cái đuôi ấy là đặc trưng phân biệt chính của chúng.
TRIM_MARGIN_SEC = 0.05
MIN_BANK_DURATION_SEC = 0.1


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def human_verdicts() -> dict[str, str]:
    """{file_id: verdict} từ phiếu kiểm định §4.4, nếu người đã điền.

    Phán xét của người GHI ĐÈ quyết định của máy: clip người đánh `wrong_class` không
    được vào bank dù máy chấm bao nhiêu. Không có bước này thì phiếu kiểm định chỉ
    dùng để tính một con số rồi vứt đi, trong khi nó chứa 245 phán xét có giá trị.
    """
    return {row["file_id"]: row["verdict"].strip()
            for row in read_csv(AUDIT_PATH) if row.get("verdict", "").strip()}


def foreground_train_ids() -> set[str]:
    return {row["file_id"] for row in read_csv(SPLITS_PATH) if row["split"] == "foreground_bank_train"}


def selectable(scored: list[dict], include_reviewed: bool, in_split: set[str] | None = None) -> tuple[list[dict], dict[str, int]]:
    """(clip được đưa vào bank, lý do loại và số lượng).

    `in_split`: tập file_id hiện đang thuộc `foreground_bank_train` trong splits.csv.
    Verdict "ok" KHÔNG được cứu một clip đã rớt khỏi tập này — splits.csv được sinh
    lại từ raw_manifest.csv TRỪ exclusions.csv, nên một clip vắng mặt ở đó nghĩa là nó
    đã bị loại chính thức ở một cổng nào đó (thường là chính auto_screen đã ghi
    wrong_class vào exclusions.csv). Verdict cũ (từ trước khi lớp đổi tên/bị rescan)
    không biết gì về lần loại sau đó — tin nó vô điều kiện là đưa một clip đã bị loại
    chính thức lọt vào bank, đúng lỗi thật đã xảy ra (1 clip `applause_cheering` máy
    chấm p_target=0.0029, top_confusable=scream, vẫn lọt vào vì verdict "ok" cũ).
    """
    verdicts = human_verdicts()
    chosen, skipped = [], {}

    def bo(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for row in scored:
        verdict = verdicts.get(row["file_id"])
        if verdict in ("wrong_class", "bad_audio"):
            bo(f"người loại ({verdict})")
            continue
        if in_split is not None and row["file_id"] not in in_split:
            bo("đã bị loại khỏi foreground_bank_train (exclusions.csv) — verdict cũ không cứu được")
            continue
        if row["decision"] == "auto_accept" or verdict == "ok":
            chosen.append(row)
        elif row["decision"] == "review" and include_reviewed:
            bo("đang chờ người duyệt (--include-reviewed không thay được việc duyệt)")
        elif row["decision"] == "review":
            bo("đang chờ người duyệt")
        else:
            bo("máy loại")
    return chosen, skipped


def trim_bounds(onset: str, offset: str, duration: float) -> tuple[float, float]:
    """Biên cắt cuối cùng, đã nới và kẹp trong file. Biên rỗng nghĩa là giữ cả clip."""
    if not onset or not offset:
        return 0.0, duration
    try:
        start, end = float(onset), float(offset)
    except ValueError:
        return 0.0, duration
    start = max(0.0, start - TRIM_MARGIN_SEC)
    end = min(duration, end + TRIM_MARGIN_SEC)
    if end - start < MIN_BANK_DURATION_SEC:
        # Biên hẹp đến mức vô nghĩa thì tin cả clip hơn tin biên.
        return 0.0, duration
    return start, end


def promote(row: dict, dry_run: bool, source_dataset_by_file: dict[str, str]) -> dict | None:
    source = Path(row["path_norm"])
    if not source.exists():
        return None
    info = soundfile.info(str(source))
    start, end = trim_bounds(row.get("onset", ""), row.get("offset", ""), info.duration)

    destination = BANK_DIR / row["class_id"] / f"{row['file_id']}.wav"
    if not dry_run:
        audio, rate = soundfile.read(str(source), start=int(start * info.samplerate),
                                     stop=int(end * info.samplerate))
        destination.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(destination, audio, rate, subtype="PCM_16")

    return {
        "file_id": row["file_id"],
        "class_id": row["class_id"],
        # KHÔNG suy ra source_dataset bằng file_id.split("_")[0] — sai với mọi nguồn
        # có gạch dưới trong tên (`vehicle_crash_cc` ra "vehicle", `desed_soundbank`
        # ra "desed"). Tra lại raw_manifest.csv, nguồn chân lý duy nhất cho cột này.
        "source_dataset": source_dataset_by_file.get(row["file_id"], "?"),
        "onset": round(start, 3),
        "offset": round(end, 3),
        "duration_bank": round(end - start, 3),
        "provenance": "human" if row.get("_human") else "auto_accept",
        "path_bank": str(destination.relative_to(REPO_ROOT)).replace("\\", "/"),
    }


def report(promoted: list[dict], skipped: dict[str, int], scored: list[dict]) -> None:
    from collections import Counter

    per_class = Counter(row["class_id"] for row in promoted)
    minutes = sum(row["duration_bank"] for row in promoted) / 60
    print(f"\n{'lớp':<22}{'clip':>7}{'phút':>8}")
    print("─" * 38)
    for class_id in sorted(per_class):
        rows = [r for r in promoted if r["class_id"] == class_id]
        print(f"{class_id:<22}{len(rows):>7}{sum(r['duration_bank'] for r in rows) / 60:>8.1f}")
    print("─" * 38)
    print(f"{'TỔNG':<22}{len(promoted):>7}{minutes:>8.1f}")

    if skipped:
        print("\nchưa đưa vào bank:")
        for reason, count in sorted(skipped.items(), key=lambda kv: -kv[1]):
            print(f"    {count:>5}  {reason}")

    da_duyet = len(human_verdicts())
    can_duyet = max(1, sum(1 for r in scored if r["decision"] == "auto_accept") // 10)
    if da_duyet < can_duyet:
        print(f"\n⚠️ Bank này CHƯA được kiểm định. Phiếu §4.4 mới điền {da_duyet}/{can_duyet} dòng.")
        print("   Chưa có tỉ lệ lỗi auto-accept thì không có căn cứ nào nói bank sạch, và")
        print("   mọi con số đo trên dữ liệu sinh từ bank đều thừa hưởng sự thiếu căn cứ đó.")


def main() -> int:
    enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-reviewed", action="store_true",
                        help="thêm cả clip người đã đánh `ok` trong phiếu kiểm định")
    args = parser.parse_args()

    scored = read_csv(SCORES_PATH)
    if not scored:
        print("❌ chưa có screen_scores.csv — chạy scripts/auto_screen.py trước")
        return 1

    chosen, skipped = selectable(scored, args.include_reviewed, foreground_train_ids())
    if not chosen:
        print("❌ không clip nào đủ điều kiện vào bank")
        return 1
    print(f"▶ {len(chosen)}/{len(scored)} clip đủ điều kiện")

    if not args.dry_run:
        # Xoá TOÀN BỘ bank cũ TRƯỚC KHI promote — script này luôn tính lại từ đầu
        # (screen_scores.csv + verdict người), không ghi tăng dần. Không xoá thì thư
        # mục lớp đã đổi tên/xoá (ví dụ `laughter_cheering` sau khi tách lớp 15/09)
        # để lại file .wav mồ côi — không nằm trong foreground_manifest.csv (file đó
        # được ghi lại đúng), nhưng vẫn nằm THẬT trên đĩa, và bất cứ script nào quét
        # trực tiếp thư mục bank (như scaper_generate.py --plan-only) sẽ thấy chúng
        # và báo "rò rỉ" sai — đã xảy ra thật, 250 file `laughter_cheering` còn sót.
        import shutil
        if BANK_DIR.exists():
            shutil.rmtree(BANK_DIR)

    source_dataset_by_file = {r["file_id"]: r["source_dataset"] for r in read_csv(RAW_MANIFEST_PATH)}
    promoted, missing = [], 0
    for number, row in enumerate(chosen, 1):
        if number % 500 == 0:
            print(f"  {number}/{len(chosen)}…", flush=True)
        entry = promote(row, args.dry_run, source_dataset_by_file)
        if entry is None:
            missing += 1
        else:
            promoted.append(entry)
    if missing:
        print(f"  ⚠️ {missing} clip không tìm thấy file đã chuẩn hoá")

    if not args.dry_run:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(sorted(promoted, key=lambda r: (r["class_id"], r["file_id"])))

    report(promoted, skipped, scored)
    print(f"\n{'(--dry-run: chưa ghi gì)' if args.dry_run else f'✓ {BANK_DIR.relative_to(REPO_ROOT)}'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
