"""Bảng tra thời lượng hữu ích của bank foreground — STATUS §7 B1.

    python scripts/probe_bank.py
    python scripts/probe_bank.py --bank data/banks/foreground --out data/manifests/bank_trim.csv

Vì sao cần bảng này, và vì sao KHÔNG ghi đè bank:

  `scaper_generate.py` đang đặt `source_time=("const", 0)` và
  `event_duration=("const", 1.0)` cho mọi sự kiện. Cả hai đều sai, nhưng sai theo
  hai kiểu khác nhau:

    · const 0     — với 51/4.267 clip có im lặng đầu > 0.5 s thì lấy phải im lặng
    · const 1.0   — 61.5% clip bị CẮT CỤT (siren mất 65.5% nội dung),
                    30.1% clip lại NGẮN HƠN 1 s nên Scaper âm thầm hạ xuống

  Sửa đúng cần biết từng file dài bao nhiêu và im lặng đầu bao nhiêu. Có hai cách
  biết: cắt sẵn bank rồi ghi đè, hoặc đo một lần rồi tra bảng. Bảng tra thắng —
  bank là DỮ LIỆU NGUỒN, ghi đè là không đảo ngược được, còn CSV thì đọc được,
  sửa được, diff được, và vào git.

`eff` là độ dài phần có tiếng; `lead`/`trail` là im lặng hai đầu. Bất biến
lead + eff + trail == dur luôn đúng, và test canh đúng bất biến đó: lệch nó là
chỉ dấu của lỗi lệch chỉ số frame, loại lỗi không có triệu chứng nào khác.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from common import REPO_ROOT, enable_utf8_output

BANK_MAC_DINH = REPO_ROOT / "data" / "banks" / "foreground"
CSV_MAC_DINH = REPO_ROOT / "data" / "manifests" / "bank_trim.csv"

# 40 dB dưới đỉnh — mặc định của librosa. Nới rộng hơn (60) thì nhiễu nền của các
# clip ghi ngoài trời bị tính là "có tiếng"; siết chặt hơn (30) thì phần đuôi vang
# của glass_breaking bị cắt mất, mà đuôi vang chính là đặc trưng của lớp đó.
TOP_DB = 40.0

COT = ["file_id", "class_id", "path", "dur", "eff", "lead", "trail", "rms_db", "peak", "loi"]

BAN_GHI_RONG = {"dur": 0.0, "eff": 0.0, "lead": 0.0, "trail": 0.0, "rms_db": -120.0, "peak": 0.0}


def _dB(x: float) -> float:
    return round(20.0 * float(np.log10(max(x, 1e-10))), 2)


def _duong_dan_tuong_doi(path: Path) -> str:
    """Đường dẫn tính từ gốc repo, dùng dấu `/`.

    Ghi đường dẫn tuyệt đối vào CSV là phá tính tái lập: file đi theo git sang máy
    khác, ổ khác, hoặc vào container là trỏ sai hết. Ngoài repo (tmp của test) thì
    giữ nguyên, vì lúc đó không có gốc nào để quy chiếu.
    """
    duong_dan = path.resolve()
    if duong_dan.is_relative_to(REPO_ROOT):
        return duong_dan.relative_to(REPO_ROOT).as_posix()
    return duong_dan.as_posix()


def do_mot_clip(path: Path) -> dict:
    """Đo một file. File hỏng trả bản ghi có `loi`, KHÔNG ném ra ngoài.

    Ném lỗi ở đây sẽ làm chết cả lượt quét 4.267 file chỉ vì một file hỏng, và
    lượt quét mất vài phút. Ghi lỗi vào chính CSV là cách để nó không bị bỏ qua.
    """
    import librosa

    ban_ghi = {"file_id": path.stem, "class_id": path.parent.name,
               "path": _duong_dan_tuong_doi(path), "loi": "", **BAN_GHI_RONG}
    try:
        song, sr = librosa.load(str(path), sr=None, mono=True)
    except Exception as loi:                                    # noqa: BLE001
        return {**ban_ghi, "loi": f"{type(loi).__name__}: {loi}"}

    if song.size == 0:
        return {**ban_ghi, "loi": "file rỗng"}

    dur = song.size / sr
    dinh = float(np.abs(song).max())
    if dinh <= 0.0:                                             # im lặng tuyệt đối
        return {**ban_ghi, "dur": round(dur, 4), "lead": round(dur, 4), "peak": 0.0}

    _, chi_so = librosa.effects.trim(song, top_db=TOP_DB)
    dau, cuoi = int(chi_so[0]), int(chi_so[1])
    co_tieng = song[dau:cuoi]
    return {
        **ban_ghi,
        "dur": round(dur, 4),
        "eff": round((cuoi - dau) / sr, 4),
        "lead": round(dau / sr, 4),
        "trail": round((song.size - cuoi) / sr, 4),
        "rms_db": _dB(float(np.sqrt(np.mean(co_tieng**2)))) if co_tieng.size else -120.0,
        "peak": round(dinh, 4),
    }


def quet_bank(root: Path) -> list[dict]:
    """Quét mọi `<lớp>/*.wav` dưới `root`. Thứ tự tất định để dataset tái lập được."""
    if not root.exists():
        raise SystemExit(f"❌ không thấy bank {root}")
    return [do_mot_clip(wav)
            for thu_muc in sorted(p for p in root.iterdir() if p.is_dir())
            for wav in sorted(thu_muc.glob("*.wav"))]


def ghi_csv(dong: list[dict], ra: Path) -> None:
    ra.parent.mkdir(parents=True, exist_ok=True)
    with ra.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COT)
        writer.writeheader()
        writer.writerows(dong)


def tom_tat(dong: list[dict]) -> str:
    """Tóm tắt để đọc ngay trên console — con số nào bất thường phải đập vào mắt."""
    tot = [r for r in dong if not r["loi"]]
    hong = [r for r in dong if r["loi"]]
    eff = [r["eff"] for r in tot]
    lead = [r["lead"] for r in tot]
    dam = [
        f"{len(tot)} clip đo được" + (f", ❌ {len(hong)} clip lỗi" if hong else ""),
        f"eff  p10={np.percentile(eff, 10):.2f}s  p50={np.percentile(eff, 50):.2f}s  "
        f"p90={np.percentile(eff, 90):.2f}s  max={max(eff):.2f}s",
        f"lead p50={np.percentile(lead, 50):.3f}s  p90={np.percentile(lead, 90):.3f}s  "
        f"số clip lead>0.5s: {sum(1 for x in lead if x > 0.5)}",
        "",
        "nguồn cấp được cho sự kiện dài:",
    ]
    for moc in (2.0, 4.0, 8.0):
        du = [r for r in tot if r["eff"] >= moc]
        lop = sorted({r["class_id"] for r in du})
        dam.append(f"  eff≥{moc:g}s: {len(du):4d} clip · {len(lop)}/15 lớp")
    if hong:
        dam += ["", "clip lỗi:"] + [f"  {r['file_id']}: {r['loi']}" for r in hong[:10]]
    return "\n".join(dam)


def main() -> None:
    enable_utf8_output()
    parser = argparse.ArgumentParser(description="Đo thời lượng hữu ích của bank foreground")
    parser.add_argument("--bank", type=Path, default=BANK_MAC_DINH)
    parser.add_argument("--out", type=Path, default=CSV_MAC_DINH)
    args = parser.parse_args()

    dong = quet_bank(args.bank)
    ghi_csv(dong, args.out)
    print(tom_tat(dong))
    print(f"\n✓ {args.out.relative_to(REPO_ROOT) if args.out.is_relative_to(REPO_ROOT) else args.out}")


if __name__ == "__main__":
    main()
