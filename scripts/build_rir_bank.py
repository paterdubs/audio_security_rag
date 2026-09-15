"""Dựng bank RIR từ OpenSLR SLR28 — nguyên liệu cho lát cắt `reverb` (DATA_PLAN §7).

    python scripts/build_rir_bank.py --dry-run
    python scripts/build_rir_bank.py

RIR (Room Impulse Response) là "dấu vân tay âm học" của một không gian: bắn một xung
cực ngắn vào phòng rồi thu lại, ta được toàn bộ cách phòng đó phản xạ âm. Tích chập
một clip khô với RIR cho ra đúng clip đó như thể được thu trong phòng ấy.

Vì sao bước này quan trọng hơn vẻ ngoài của nó: mọi clip trong foreground bank đều là
bản thu GẦN và KHÔ (Freesound, phòng thu). Micro giám sát gắn tường thì nghe thấy âm
đã đi qua hành lang, dội tường, mất dải cao. Model học từ audio khô sẽ không nhận ra
cùng một âm thanh khi nó vọng — và với `speech_normal`, không nhận ra nghĩa là giọng
nói bình thường bị xếp nhầm sang `shout_yell`, tức báo động giả.

⚠️ Đây là giải pháp TẠM. RIR của SLR28 đo tại phòng thật nhưng là phòng ở Nhật/Đức/Anh,
không phải hành lang IUH. RIR thật của không gian triển khai chỉ có sau buổi thu 1.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import soundfile

from common import REPO_ROOT, enable_utf8_output

RIR_ROOT = REPO_ROOT / "data" / "raw" / "rir_openslr28" / "RIRS_NOISES"
BANK_DIR = REPO_ROOT / "data" / "banks" / "rir"
MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "rir_manifest.csv"

TARGET_SR = 16000            # trùng sample rate của toàn pipeline
SEED = 20260915
PER_SIMULATED_ROOM = 120     # mỗi cỡ phòng mô phỏng lấy bấy nhiêu, không lấy cả 60k
MAX_RIR_SEC = 2.0            # cắt đuôi dài hơn mức này — xem trim_rir()
PRE_DIRECT_MS = 5.0          # giữ lại bấy nhiêu trước xung trực tiếp

FIELDS = ["rir_id", "space_type", "source", "rt60_sec", "duration_sec", "path_bank"]

# Xếp nhóm theo RT60 ĐO ĐƯỢC, không theo tên thư mục. Ba lý do:
#   1. `simulated_rirs` của SLR28 chỉ có smallroom + mediumroom — KHÔNG có largeroom,
#      dù README mô tả cả ba. Tin tên thư mục thì bank thiếu hẳn nhóm vang dài, mà
#      nhà xe và xưởng chính là nhóm đó.
#   2. Thư mục "real" trộn nhiều bộ khác nhau (REVERB, RWCP) với quy ước tên riêng;
#      không có tên chung nào để suy ra cỡ phòng.
#   3. RT60 mới là thứ quyết định clip nghe ra sao. Hai phòng cùng "cỡ vừa" nhưng một
#      phòng trải thảm, một phòng lát gạch thì cho ra hai thứ hoàn toàn khác nhau.
#
# Ranh giới đặt theo mốc âm học thông dụng, đối chiếu với không gian triển khai:
RT60_BINS = [
    (0.50, "small_room"),     # phòng học, văn phòng
    (1.20, "medium_room"),    # hành lang, sảnh
    (float("inf"), "large_room"),   # nhà xe, xưởng
]
# Dưới mức này coi như phòng câm — RWCP có 36 file đo trong buồng tiêu âm (`_ane_`).
# Đưa chúng vào bank reverb thì lát cắt "có vang" lại chứa clip không vang chút nào,
# và tỉ lệ 40% ghi trong báo cáo sẽ mô tả sai thứ thực sự đã sinh ra.
MIN_RT60_SEC = 0.15


def bin_by_rt60(rt60: float) -> str | None:
    if rt60 < MIN_RT60_SEC:
        return None
    for ceiling, name in RT60_BINS:
        if rt60 < ceiling:
            return name
    return RT60_BINS[-1][1]


# ── Đo thời gian vang ────────────────────────────────────────────────────────


def rt60_from_rir(rir: np.ndarray, sample_rate: int) -> float:
    """Ước lượng RT60 bằng tích phân ngược Schroeder, ngoại suy từ đoạn −5…−25 dB.

    Không đo thẳng tới −60 dB vì đuôi RIR thật chìm dưới sàn nhiễu trước khi tới đó;
    đo ở đoạn còn sạch rồi nhân ba là cách chuẩn (T20). Con số này không dùng để tính
    toán gì trong pipeline — nó để BÁO CÁO, cho biết bank phủ dải vang nào và có thiếu
    khoảng nào không.
    """
    energy = rir.astype(np.float64) ** 2
    if energy.sum() <= 0:
        return 0.0
    # Tích phân ngược: năng lượng còn lại kể từ mỗi thời điểm tới hết.
    schroeder = np.cumsum(energy[::-1])[::-1]
    db = 10 * np.log10(np.maximum(schroeder / schroeder[0], 1e-12))

    start = np.argmax(db <= -5.0)
    end = np.argmax(db <= -25.0)
    if end <= start:
        return 0.0
    slope = (db[end] - db[start]) / ((end - start) / sample_rate)   # dB mỗi giây
    if slope >= 0:
        return 0.0
    return float(-60.0 / slope)


# ── Chuẩn hoá RIR ────────────────────────────────────────────────────────────


def trim_rir(rir: np.ndarray, sample_rate: int) -> np.ndarray:
    """Cắt phần im lặng trước xung trực tiếp, và cắt đuôi quá dài.

    Phần trước xung trực tiếp là ĐỘ TRỄ LAN TRUYỀN từ nguồn tới micro. Giữ nó lại thì
    tích chập sẽ đẩy TOÀN BỘ sự kiện lùi đi vài chục mili giây, trong khi nhãn strong
    trong .jams vẫn ghi mốc cũ. Sai lệch âm thầm giữa audio và nhãn, và nó tích luỹ
    đúng ở thứ khoá luận đang đo: độ chính xác của onset.
    """
    peak = int(np.argmax(np.abs(rir)))
    pre = int(PRE_DIRECT_MS * sample_rate / 1000)
    start = max(0, peak - pre)
    end = min(len(rir), start + int(MAX_RIR_SEC * sample_rate))
    return rir[start:end]


def normalize_rir(rir: np.ndarray) -> np.ndarray:
    """Chuẩn hoá theo NĂNG LƯỢNG, không theo đỉnh.

    Tích chập với RIR năng lượng 1 giữ nguyên mức to tổng thể của clip. Chuẩn theo
    đỉnh thì RIR vang dài (nhiều năng lượng rải đều, đỉnh thấp) sẽ làm clip to vọt lên,
    còn RIR khô thì làm clip nhỏ đi — và SNR mà Scaper vừa đặt cẩn thận sẽ sai lệch
    theo đúng độ vang, tức sai một cách CÓ HỆ THỐNG chứ không ngẫu nhiên.
    """
    energy = float(np.sqrt(np.sum(rir.astype(np.float64) ** 2)))
    return rir / energy if energy > 0 else rir


def load_rir(path: Path) -> np.ndarray | None:
    """Đọc, về mono, resample về 16 kHz. Trả None nếu file không dùng được."""
    import librosa

    try:
        audio, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
    except Exception:
        return None
    if audio.size < TARGET_SR // 100 or not np.isfinite(audio).all():
        return None
    return audio


# ── Liệt kê ứng viên ─────────────────────────────────────────────────────────


def simulated_candidates(rng: random.Random) -> list[Path]:
    """Lấy mẫu tất định từ RIR mô phỏng — 40 000 file, ta chỉ cần vài trăm.

    Lấy N file đầu danh sách sẽ ra một bank rất hẹp: file xếp theo tên nên đầu danh
    sách toàn thuộc cùng một phòng mô phỏng.
    """
    root = RIR_ROOT / "simulated_rirs"
    found: list[Path] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(folder.rglob("*.wav"))
        if files:
            found.extend(sorted(rng.sample(files, min(PER_SIMULATED_ROOM, len(files)))))
    return found


def real_candidates() -> list[Path]:
    """RIR đo tại phòng thật. Thư mục này TRỘN LẪN RIR với file nhiễu đẳng hướng.

    Phân biệt bằng `_rir_` / `_noise_` ở GIỮA tên, không phải ở đầu: tên thật có dạng
    `RVB2014_type1_noise_largeroom1_1.wav`. Lọc theo phần đầu tên chỉ bắt được 1/417
    file, và 92 file nhiễu sẽ lọt vào bank — tích chập với nhiễu biến clip thành tiếng
    ù chứ không phải tiếng vang, mà nghe qua thì vẫn "có vẻ có hiệu ứng gì đó".
    """
    root = RIR_ROOT / "real_rirs_isotropic_noises"
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.wav") if "_rir_" in p.name.lower())


# ── Chạy ─────────────────────────────────────────────────────────────────────


def report(rows: list[dict]) -> None:
    print(f"\n{'không gian':<16}{'RIR':>6}{'RT60 trung vị':>16}{'dải RT60':>20}")
    print("─" * 60)
    for space in sorted({r["space_type"] for r in rows}):
        subset = [float(r["rt60_sec"]) for r in rows if r["space_type"] == space]
        subset.sort()
        median = subset[len(subset) // 2]
        print(f"{space:<16}{len(subset):>6}{median:>15.2f}s{f'{subset[0]:.2f}–{subset[-1]:.2f}s':>20}")
    print("─" * 60)
    print(f"{'TỔNG':<16}{len(rows):>6}")
    print("\nĐối chiếu để biết bank phủ đủ chưa:")
    print("   phòng học / văn phòng  ~ 0.4–0.8 s")
    print("   hành lang, sảnh        ~ 0.8–1.5 s")
    print("   nhà xe, xưởng          ~ 1.5–3.0 s")


def main() -> int:
    enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if not RIR_ROOT.exists():
        print(f"❌ chưa có {RIR_ROOT.relative_to(REPO_ROOT)}")
        print("   chạy: scripts/download_sources.py --source rir_openslr28")
        return 1

    rng = random.Random(SEED)
    candidates = real_candidates() + simulated_candidates(rng)
    if not candidates:
        print("❌ không tìm thấy file RIR nào — cấu trúc thư mục có đúng không?")
        return 1
    candidates = candidates[: args.limit] if args.limit else candidates
    print(f"▶ {len(candidates)} RIR ứng viên "
          f"({len(real_candidates())} đo thật, còn lại mô phỏng)")

    if args.dry_run:
        print("\n(--dry-run: chưa đo RT60, chưa xử lý)")
        return 0

    rows, bad, cam = [], 0, 0
    for number, path in enumerate(candidates, 1):
        if number % 100 == 0:
            print(f"  {number}/{len(candidates)}…", flush=True)
        audio = load_rir(path)
        if audio is None:
            bad += 1
            continue
        trimmed = trim_rir(audio, TARGET_SR)
        rt60 = rt60_from_rir(trimmed, TARGET_SR)
        space = bin_by_rt60(rt60)
        if space is None:
            cam += 1                      # phòng câm — không phải RIR có vang
            continue
        final = normalize_rir(trimmed)

        rir_id = f"rir_{space}_{path.stem}_{number:04d}"
        destination = BANK_DIR / space / f"{rir_id}.wav"
        destination.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(destination, final, TARGET_SR, subtype="FLOAT")
        rows.append({
            "rir_id": rir_id,
            "space_type": space,
            "source": "openslr28_real" if "_rir_" in path.name.lower() else "openslr28_sim",
            "rt60_sec": round(rt60, 3),
            "duration_sec": round(len(final) / TARGET_SR, 3),
            "path_bank": str(destination.relative_to(REPO_ROOT)).replace("\\", "/"),
        })

    if cam:
        print(f"  {cam} RIR có RT60 < {MIN_RT60_SEC}s (phòng câm) — đã loại")
    if bad:
        print(f"  ⚠️ {bad} file không đọc được, đã bỏ qua")
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report(rows)
    print(f"\n✓ {BANK_DIR.relative_to(REPO_ROOT)} · {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
