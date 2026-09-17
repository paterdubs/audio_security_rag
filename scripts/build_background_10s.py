"""Dựng bank nền dài TỐI THIỂU 10 giây từ các lát 4 giây — STATUS §7 B2.5.

    python scripts/build_background_10s.py
    python scripts/build_background_10s.py --out data/banks/background_10s --dry-run

VẤN ĐỀ. Clip tổng hợp dài 10 giây, nhưng 874/2461 clip nền dài đúng 4 giây và
351 clip `school` nữa dưới 10 giây. Scaper lặp vòng nguồn cho đủ độ dài, nên
~81% clip train mang một nền lặp đúng chu kỳ 4 giây. Đo trên 300 clip không có sự
kiện: mức dB từng giây của `parking` là

    -21.7 -21.7 -21.6 -21.2 │ -21.6 -21.7 -21.6 -21.2 │ -21.6 -21.7

Lặp lại tới từng chữ số. Bản ghi thật không có tính chất đó.

KHÔNG phải vấn đề click. Cú nhảy biên độ tại chỗ nối (0.083) vẫn THẤP HƠN đỉnh
bình thường của chính clip (0.100), nên không có transient giả. Cái sai là nội
dung lặp, không phải sóng đứt — và vì thế nối hai bản ghi KHÁC NHAU lại cho ra
thứ tệ hơn: một cú đổi cảnh giữa clip.

LỐI RA. UrbanSound8K cắt lát bằng cửa sổ 4 giây, bước 2 giây, nên hai lát liền kề
cùng một occurrence chồng nhau ĐÚNG 2.0 giây — đo trên 560 cặp, tương quan
1.00000 tại 2.0 s và ~0.00x ở mọi độ chồng khác. Ghép chúng lại không phải là nối
hai nguồn, mà là TÁI DỰNG chính bản ghi gốc: n lát liền kề cho 2n+2 giây liên tục
thật.

Mỗi occurrence ra ĐÚNG MỘT file, GIỮ NGUYÊN độ dài (tới 66 giây) chứ không cắt
sẵn thành cửa sổ 10 giây. Cắt sẵn vừa nhân bản nội dung lên đĩa vừa khoá cứng
điểm bắt đầu. Để nguyên rồi cho `scaper_generate.chon_nen` rút `source_time`
ngẫu nhiên thì mỗi lần dùng là một cửa sổ khác, không tốn thêm byte nào. Đơn vị
nào tái dựng xong vẫn dưới 10 giây thì kéo lên đúng 10 giây bằng lặp crossfade.

Phép đo cũng phơi ra một điều khác: 364 clip `factory` thực chất chỉ là 88 bản
ghi nhìn qua các cửa sổ chồng nhau — tổng 744 giây audio duy nhất. Số clip của
bank nền lâu nay nói quá độ đa dạng thật của nó.

GIỚI HẠN CÒN LẠI, nói thẳng. Không phải occurrence nào cũng đủ lát để tái dựng:

    khu vực     file   liên tục   phải lặp   giây lặp tb   tổng giây
    factory       88         20         68        5.19s        1104
    parking       11          3          8        2.18s         156
    residential   18         11          7        0.89s         332
    school      1294       1239         55        0.20s       12948

`residential` và `school` gần như sạch. `factory` vẫn còn 68/88 đơn vị phải lặp
vì phần lớn occurrence của nó chỉ có đúng một lát 4 giây — đó là giới hạn của
nguồn, không phải của cách làm. Bank gốc giữ nguyên, không ghi đè.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from common import REPO_ROOT, enable_utf8_output

BANK_GOC = REPO_ROOT / "data" / "banks" / "background"
BANK_RA = REPO_ROOT / "data" / "banks" / "background_10s"
MANIFEST_GOC = REPO_ROOT / "data" / "manifests" / "background_manifest.csv"
MANIFEST_RA = REPO_ROOT / "data" / "manifests" / "background_10s_manifest.csv"

# Đo được, không phải giả định: 560 cặp lát liền kề, tương quan 1.00000 tại đúng
# 2.0 s. Đặt sai hằng số này thì bản tái dựng vừa lặp vừa nhảy, mà vẫn ra audio
# nghe được và vẫn đúng 10 giây — không có triệu chứng nào.
CHONG_LAN_SEC = 2.0

DAI_TOI_THIEU_SEC = 10.0   # bằng đúng độ dài clip tổng hợp
CROSSFADE_SEC = 0.25

COT_MANIFEST = ["file_id", "area_type", "source_group_id", "occurrence", "so_lat",
                "dai_sec", "kieu", "giay_lap"]


@dataclass(frozen=True)
class DonVi:
    """Một occurrence — tức một bản ghi gốc, nhìn qua các lát chồng nhau."""

    area_type: str
    source_group_id: str
    occurrence: str
    paths: tuple[Path, ...]      # các lát LIỀN KỀ, đã sắp theo slice_id


# ── Tái dựng ─────────────────────────────────────────────────────────────────


def chuoi_lien_tuc(lat: list[tuple[int, object]]) -> list:
    """Đoạn slice_id LIỀN KỀ dài nhất, trả theo thứ tự tăng dần.

    Ghép qua chỗ đứt là nối hai quãng khác nhau của bản ghi — đúng cái đứt gãy mà
    cả bước này sinh ra để tránh. Thà lấy đoạn ngắn hơn mà liên tục thật.
    """
    theo_thu_tu = sorted(lat, key=lambda x: x[0])
    tot_nhat: list = []
    hien_tai: list = []
    truoc: int | None = None
    for chi_so, gia_tri in theo_thu_tu:
        hien_tai = hien_tai + [gia_tri] if truoc is not None and chi_so == truoc + 1 else [gia_tri]
        truoc = chi_so
        if len(hien_tai) > len(tot_nhat):
            tot_nhat = hien_tai
    return tot_nhat


def tai_dung(paths: list[Path], sr: int, chong_lan_sec: float = CHONG_LAN_SEC) -> np.ndarray:
    """Ghép các lát liền kề thành lại bản ghi gốc: lát đầu trọn vẹn, các lát sau bỏ phần chồng."""
    import soundfile as sf

    bo_dau = int(round(chong_lan_sec * sr))
    manh: list[np.ndarray] = []
    for thu_tu, path in enumerate(paths):
        song, _ = sf.read(str(path), dtype="float32", always_2d=False)
        manh.append(song if thu_tu == 0 else song[bo_dau:])
    return np.concatenate(manh) if manh else np.zeros(0, dtype=np.float32)


# ── Cắt thành clip 10 giây ───────────────────────────────────────────────────


def lap_crossfade(song: np.ndarray, sr: int, dai_sec: float = DAI_TOI_THIEU_SEC,
                  fade_sec: float = CROSSFADE_SEC) -> np.ndarray:
    """Lặp nguồn cho đủ `dai_sec`, chỗ nối chuyển tiếp mượt bằng crossfade công suất đều.

    Dùng cho các đơn vị vẫn ngắn hơn 10 giây sau khi tái dựng. Nội dung vẫn lặp —
    điều đó không tránh được khi nguồn chỉ có bấy nhiêu — nhưng chỗ nối không tạo
    thêm transient giả, mà một transient đều đặn thì model rất dễ học thành onset.

    Crossfade theo căn bậc hai (công suất đều) chứ không tuyến tính: với hai đoạn
    KHÔNG tương quan, crossfade tuyến tính làm tụt mức ~3 dB ngay tại chỗ nối, và
    tụt mức thì phá tỉ số SNR mà Scaper sắp đặt ngay sau đó.
    """
    n = int(round(dai_sec * sr))
    if len(song) >= n:
        return song[:n].copy()

    fade = min(int(round(fade_sec * sr)), len(song) // 4)
    if fade <= 0:
        lap = int(np.ceil(n / len(song)))
        return np.tile(song, lap)[:n].copy()

    di_len = np.sqrt(np.linspace(0.0, 1.0, fade, dtype=np.float32))
    di_xuong = np.sqrt(np.linspace(1.0, 0.0, fade, dtype=np.float32))

    ra = song.copy()
    while len(ra) < n:
        noi = ra[-fade:] * di_xuong + song[:fade] * di_len
        ra = np.concatenate([ra[:-fade], noi, song[fade:]])
    return ra[:n].copy()


def clip_tu_don_vi(song: np.ndarray, sr: int) -> np.ndarray:
    """Một file cho mỗi đơn vị — GIỮ NGUYÊN độ dài nếu đã đủ 10 giây.

    Không cắt sẵn thành cửa sổ 10 giây. Cắt sẵn vừa nhân bản nội dung lên đĩa vừa
    khoá cứng điểm bắt đầu: mỗi lần dùng lại nghe đúng một đoạn. Để nguyên đơn vị
    dài rồi cho Scaper rút `source_time` ngẫu nhiên thì mỗi lần dùng là một cửa sổ
    khác, không tốn thêm byte nào.

    `parking` chỉ có 11 bản ghi tổng cộng 132 giây. Cắt ra 84 clip 4 giây không
    làm nó đa dạng hơn — chỉ làm con số trông đẹp hơn sự thật.
    """
    return song if len(song) >= int(round(DAI_TOI_THIEU_SEC * sr)) else lap_crossfade(song, sr)


# ── Gom đơn vị từ bank gốc ───────────────────────────────────────────────────


def doc_manifest(path: Path = MANIFEST_GOC) -> dict[str, dict]:
    if not path.exists():
        raise SystemExit(f"❌ chưa có {path.name}")
    with path.open(encoding="utf-8", newline="") as handle:
        return {r["file_id"]: r for r in csv.DictReader(handle)}


def gom_don_vi(bank: Path, manifest: dict[str, dict]) -> list[DonVi]:
    """Gom các lát về từng occurrence, mỗi occurrence lấy đoạn liền kề dài nhất.

    Nguồn không phải UrbanSound8K (tau2019 dài 10 s sẵn) thành đơn vị một lát —
    chúng vốn đã liên tục, không cần tái dựng gì.
    """
    theo_occ: dict[tuple[str, str, str], list[tuple[int, Path]]] = defaultdict(list)
    for wav in sorted(bank.glob("*/*.wav")):
        ban_ghi = manifest.get(wav.stem)
        if ban_ghi is None:
            raise SystemExit(f"❌ {wav.stem} không có trong background_manifest.csv")
        area, nhom = ban_ghi["area_type"], ban_ghi["source_group_id"]
        if ban_ghi["source_dataset"] != "urbansound8k":
            theo_occ[(area, nhom, wav.stem)].append((0, wav))
            continue
        phan = ban_ghi["source_id"].rsplit(".", 1)[0].split("-")
        theo_occ[(area, nhom, phan[2])].append((int(phan[3]), wav))

    return [DonVi(area, nhom, occ, tuple(chuoi_lien_tuc(lat)))
            for (area, nhom, occ), lat in sorted(theo_occ.items())]


# ── Sinh bank ────────────────────────────────────────────────────────────────


def sinh(don_vi: list[DonVi], ra_dir: Path, sr: int, dry_run: bool = False) -> list[dict]:
    import soundfile as sf

    dong: list[dict] = []
    for dv in don_vi:
        song = tai_dung(list(dv.paths), sr)
        dai_don_vi = len(song) / sr
        mau = clip_tu_don_vi(song, sr)
        file_id = f"bgc_{dv.area_type}_{dv.source_group_id}_{dv.occurrence}"
        if not dry_run:
            dich = ra_dir / dv.area_type / f"{file_id}.wav"
            dich.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(dich), mau, sr, subtype="PCM_16")
        dong.append({
            "file_id": file_id, "area_type": dv.area_type,
            "source_group_id": dv.source_group_id, "occurrence": dv.occurrence,
            "so_lat": len(dv.paths), "dai_sec": round(len(mau) / sr, 3),
            "kieu": "lien_tuc" if dai_don_vi >= DAI_TOI_THIEU_SEC else "lap_crossfade",
            "giay_lap": round(max(0.0, DAI_TOI_THIEU_SEC - dai_don_vi), 3),
        })
    return dong


def tom_tat(dong: list[dict]) -> str:
    theo_khu: dict[str, list[dict]] = defaultdict(list)
    for r in dong:
        theo_khu[r["area_type"]].append(r)
    def hang(ten: str, rs: list[dict]) -> str:
        lien = sum(1 for r in rs if r["kieu"] == "lien_tuc")
        dai = [r["dai_sec"] for r in rs]
        return (f"{ten:<14}{len(rs):>6}{lien:>10}{len(rs) - lien:>6}"
                f"{np.mean([r['giay_lap'] for r in rs]):>12.2f}s"
                f"{np.percentile(dai, 50):>9.1f}{max(dai):>9.1f}{sum(dai):>11.0f}")

    dam = [f"{'khu vực':<14}{'file':>6}{'liên tục':>10}{'lặp':>6}{'giây lặp tb':>13}"
           f"{'dài p50':>9}{'dài max':>9}{'tổng giây':>11}", "─" * 78]
    dam += [hang(khu, theo_khu[khu]) for khu in sorted(theo_khu)]
    dam += ["─" * 78, hang("TỔNG", dong)]
    return "\n".join(dam)


def main() -> None:
    enable_utf8_output()
    parser = argparse.ArgumentParser(description="Dựng bank nền 10 giây từ các lát 4 giây")
    parser.add_argument("--bank", type=Path, default=BANK_GOC)
    parser.add_argument("--out", type=Path, default=BANK_RA)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_RA)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--dry-run", action="store_true", help="chỉ báo cáo, không ghi audio")
    args = parser.parse_args()

    don_vi = gom_don_vi(args.bank, doc_manifest())
    dong = sinh(don_vi, args.out, args.sr, dry_run=args.dry_run)
    print(tom_tat(dong))

    if args.dry_run:
        print("\n(dry-run — chưa ghi file nào)")
        return
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COT_MANIFEST)
        writer.writeheader()
        writer.writerows(dong)
    print(f"\n✓ {args.out.relative_to(REPO_ROOT)}  ·  {args.manifest.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
