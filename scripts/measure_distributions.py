"""Đo phân bố thời lượng sự kiện: synthetic so với AudioSet Strong — STATUS §7 B0.

    python scripts/measure_distributions.py
    python scripts/measure_distributions.py --synthetic data/synthetic_legacy/train
    python scripts/measure_distributions.py --out docs/measurements/dist_sau_sua.md

Vì sao cần script riêng thay vì đo một lần rồi chép số vào tài liệu:

  Hợp đồng `long_event ≥ 8s` từng được ghi "đạt 10.0% · 791 clip" trong log sinh,
  trong khi số sự kiện ≥ 8s thực tế sinh ra là 0. Log báo cáo KẾ HOẠCH, không báo
  cáo KẾT QUẢ. Chênh lệch đó tồn tại suốt hai vòng train mà không có triệu chứng.

  Cách duy nhất bắt được loại lỗi đó là đọc ngược lại chính file nhãn đã sinh và
  so với phân bố đích. Đó là việc của file này, và nó phải chạy lại được sau mỗi
  lần sinh chứ không phải một phép đo dùng một lần.

Độ phủ sóng tính bằng HỢP các khoảng, không phải tổng. Hai sự kiện chồng lấn
nhau 100% mà cộng dồn sẽ cho 2× độ phủ thật — và chồng lấn là thứ ta cố ý ép.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from common import REPO_ROOT, enable_utf8_output

AUDIOSET_SEGMENTS = REPO_ROOT / "data" / "raw" / "audioset_strong" / "segments.jsonl"
DEFAULT_SYNTHETIC = REPO_ROOT / "data" / "synthetic" / "train"
CLIP_DURATION_SEC = 10.0
MOC_DAI = (2.0, 4.0, 8.0)  # các mốc "sự kiện dài" dùng để đối chiếu hợp đồng


# ── Đọc dữ liệu ──────────────────────────────────────────────────────────────


def hop_khoang(khoang: list[tuple[float, float]]) -> float:
    """Tổng độ dài phần HỢP của các khoảng, đã gộp phần chồng lấn.

    Phải lấy khoảng đầu tiên từ bản ĐÃ SẮP XẾP. Lấy từ danh sách gốc rồi mới duyệt
    bản đã sắp là sai — và sai đúng trên dữ liệu thật, vì `.jams` ghi sự kiện theo
    thứ tự nạp (chuỗi nhân quả trước, sự kiện tự do sau) chứ không theo thời gian.
    """
    if not khoang:
        return 0.0
    theo_thu_tu = sorted(khoang)
    tong, dau, cuoi = 0.0, *theo_thu_tu[0]
    for bat_dau, ket_thuc in theo_thu_tu[1:]:
        if bat_dau > cuoi:
            tong += cuoi - dau
            dau, cuoi = bat_dau, ket_thuc
        else:
            cuoi = max(cuoi, ket_thuc)
    return tong + (cuoi - dau)


def doc_audioset(path: Path) -> tuple[dict[str, list[float]], list[float], int]:
    """(thời lượng theo lớp, độ phủ sóng từng clip, số clip) từ segments.jsonl.

    CHỈ lấy phía `train_strong`. Phía `eval_strong` giữ nguyên làm hạt giống cho
    gold_test, kể cả với một công cụ chỉ để đọc: đối chiếu dữ liệu huấn luyện với
    tập sẽ dùng để chấm cũng là một dạng ngó trộm, chỉ là gián tiếp hơn.
    Xem docs/decisions/ADR-0006-phan-bo-dich-chi-tu-train-strong.md.
    """
    from scaper_generate import ytid_eval_strong

    if not path.exists():
        raise SystemExit(f"❌ chưa có {path.relative_to(REPO_ROOT)} — chạy scripts/fetch_audioset_strong.py trước")
    bo_qua = ytid_eval_strong()
    theo_lop: dict[str, list[float]] = defaultdict(list)
    phu_song: list[float] = []
    so_clip = 0
    with path.open(encoding="utf-8") as handle:
        for dong in handle:
            if not dong.strip():
                continue
            ban_ghi = json.loads(dong)
            if ban_ghi.get("ytid") in bo_qua:
                continue
            so_clip += 1
            khoang = []
            for su_kien in ban_ghi["events"]:
                dau, cuoi = float(su_kien["onset"]), min(float(su_kien["offset"]), CLIP_DURATION_SEC)
                theo_lop[su_kien["class_id"]].append(cuoi - dau)
                khoang.append((dau, cuoi))
            phu_song.append(hop_khoang(khoang) / CLIP_DURATION_SEC)
    return dict(theo_lop), phu_song, so_clip


def doc_jams(thu_muc: Path) -> tuple[dict[str, list[float]], list[float], int]:
    """(thời lượng theo lớp, độ phủ sóng từng clip, số clip) từ thư mục jams/.

    Chỉ lấy sự kiện `foreground`. Nền phủ trọn clip nên tính vào sẽ luôn ra 100%.
    """
    jams_dir = thu_muc / "jams"
    if not jams_dir.exists():
        raise SystemExit(f"❌ không thấy {jams_dir.relative_to(REPO_ROOT)}")
    theo_lop: dict[str, list[float]] = defaultdict(list)
    phu_song: list[float] = []
    so_clip = 0
    for jams_path in sorted(jams_dir.glob("*.jams")):
        so_clip += 1
        du_lieu = json.loads(jams_path.read_text(encoding="utf-8"))
        khoang = []
        for muc in du_lieu["annotations"][0]["data"]:
            if muc["value"].get("role") != "foreground":
                continue
            dau, dai = float(muc["time"]), float(muc["duration"])
            theo_lop[muc["value"]["label"]].append(dai)
            khoang.append((dau, min(dau + dai, CLIP_DURATION_SEC)))
        phu_song.append(hop_khoang(khoang) / CLIP_DURATION_SEC)
    if not so_clip:
        raise SystemExit(f"❌ {jams_dir.relative_to(REPO_ROOT)} không có file .jams nào")
    return dict(theo_lop), phu_song, so_clip


# ── Kết xuất markdown ────────────────────────────────────────────────────────


def bang_theo_lop(theo_lop: dict[str, list[float]], tieu_de: str) -> list[str]:
    """Bảng thời lượng per-class, kèm tỉ lệ đạt từng mốc sự kiện dài."""
    cot_moc = "".join(f" ≥{m:g}s |" for m in MOC_DAI)
    dong = [
        f"### {tieu_de}", "",
        f"| lớp | n | p05 | p50 | p95 | max |{cot_moc}",
        f"|---|---:|---:|---:|---:|---:|{'---:|' * len(MOC_DAI)}",
    ]
    for lop in sorted(theo_lop):
        d = theo_lop[lop]
        moc = "".join(f" {100 * np.mean([x >= m for x in d]):.1f}% |" for m in MOC_DAI)
        dong.append(
            f"| {lop} | {len(d)} | {np.percentile(d, 5):.2f} | {np.percentile(d, 50):.2f} |"
            f" {np.percentile(d, 95):.2f} | {max(d):.2f} |{moc}"
        )
    tat_ca = [x for d in theo_lop.values() for x in d]
    moc = "".join(f" **{100 * np.mean([x >= m for x in tat_ca]):.1f}%** |" for m in MOC_DAI)
    dong.append(
        f"| **TỔNG** | **{len(tat_ca)}** | **{np.percentile(tat_ca, 5):.2f}** |"
        f" **{np.percentile(tat_ca, 50):.2f}** | **{np.percentile(tat_ca, 95):.2f}** |"
        f" **{max(tat_ca):.2f}** |{moc}"
    )
    return dong + [""]


def _trong_so_mix_deu(theo_lop: dict[str, list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """(giá trị, trọng số) sao cho mỗi LỚP đóng góp như nhau, bất kể có bao nhiêu sự kiện."""
    gia_tri = np.concatenate([np.asarray(d, dtype=float) for d in theo_lop.values()])
    trong_so = np.concatenate([np.full(len(d), 1.0 / len(d)) for d in theo_lop.values()])
    return gia_tri, trong_so / trong_so.sum()


def _phan_vi_co_trong_so(gia_tri: np.ndarray, trong_so: np.ndarray, p: float) -> float:
    thu_tu = np.argsort(gia_tri)
    x, w = gia_tri[thu_tu], trong_so[thu_tu]
    return float(np.interp(p / 100.0, np.cumsum(w) - 0.5 * w, x))


def bang_doi_chieu(synth: dict, synth_phu: list[float], synth_n: int,
                   gold: dict, gold_phu: list[float], gold_n: int) -> list[str]:
    """Bảng so trực tiếp hai phân bố — đây là bảng nói lên mức lệch train/test.

    Có HAI cột đích, và bỏ cột thứ hai đi là mời một kết luận sai vào khoá luận.

    AudioSet có mix lớp rất lệch: `speech_normal` chiếm 42.5% số sự kiện và
    `running_footsteps` 13.8%, cả hai đều là lớp sự kiện ngắn. Dữ liệu tổng hợp của
    ta thì cân bằng ĐỀU, 6.7% mỗi lớp — cố ý, vì hệ thống an ninh cần lớp hiếm
    (gunshot, scream, glass_breaking) ngang với lớp phổ biến.

    Hệ quả: so tỉ lệ sự kiện dài của ta với cột "mix tự nhiên" là so lệch chuẩn. Ở
    mốc 4 giây, đích đọc ra 8.9% theo mix tự nhiên nhưng 15.5% theo mix đều — gần
    gấp đôi. Nhìn nhầm cột sẽ kết luận "ta sinh sự kiện dài gấp đôi thực tế" trong
    khi thực ra đang bám khá sát.
    """
    a = [x for d in synth.values() for x in d]
    b = [x for d in gold.values() for x in d]
    gv, tso = _trong_so_mix_deu(gold)
    hang: list[tuple[str, float, float, float]] = [
        ("p05 (s)", float(np.percentile(a, 5)), float(np.percentile(b, 5)),
         _phan_vi_co_trong_so(gv, tso, 5)),
        ("p50 (s)", float(np.percentile(a, 50)), float(np.percentile(b, 50)),
         _phan_vi_co_trong_so(gv, tso, 50)),
        ("p95 (s)", float(np.percentile(a, 95)), float(np.percentile(b, 95)),
         _phan_vi_co_trong_so(gv, tso, 95)),
        ("max (s)", float(max(a)), float(max(b)), float(max(b))),
        *[(f"% sự kiện ≥ {m:g}s", 100 * float(np.mean([x >= m for x in a])),
           100 * float(np.mean([x >= m for x in b])),
           100 * float(np.sum(tso[gv >= m]))) for m in MOC_DAI],
        ("sự kiện/clip", len(a) / synth_n, len(b) / gold_n, float("nan")),
        ("độ phủ sóng %", 100 * float(np.mean(synth_phu)), 100 * float(np.mean(gold_phu)),
         float("nan")),
    ]
    dong = [
        "### Đối chiếu trực tiếp", "",
        "Cột **mix đều** là cột đúng để so: dữ liệu tổng hợp cân bằng đều 15 lớp, còn "
        "AudioSet lệch nặng về `speech_normal` (42.5%) và `running_footsteps` (13.8%) — "
        "đều là lớp sự kiện ngắn. So với cột mix tự nhiên là so lệch chuẩn.", "",
        "| chỉ tiêu | SYNTHETIC | AUDIOSET mix tự nhiên | AUDIOSET **mix đều** | lệch so với mix đều |",
        "|---|---:|---:|---:|---:|",
    ]
    for ten, x, y, z in hang:
        if np.isnan(z):
            dong.append(f"| {ten} | {x:.2f} | {y:.2f} | — | — |")
        else:
            dong.append(f"| {ten} | {x:.2f} | {y:.2f} | **{z:.2f}** | {x - z:+.2f} |")
    return dong + [""]


def duong_dan_tuyet_doi(path: Path) -> Path:
    """Phân giải đường dẫn CLI về tuyệt đối, ưu tiên gốc repo.

    Gõ `--synthetic data/synthetic/train` từ bất kỳ thư mục nào cũng phải trỏ đúng
    chỗ; không có bước này thì `relative_to(REPO_ROOT)` ở khâu in báo cáo ném lỗi.
    """
    if path.is_absolute():
        return path
    tu_repo = (REPO_ROOT / path).resolve()
    return tu_repo if tu_repo.exists() else path.resolve()


def dung_bao_cao(synth: tuple, gold: tuple, nguon_synth: Path) -> str:
    from datetime import date

    s_lop, s_phu, s_n = synth
    g_lop, g_phu, g_n = gold
    dong = [
        f"# Phân bố thời lượng sự kiện — {date.today().isoformat()}", "",
        f"Sinh bởi `scripts/measure_distributions.py`. Nguồn synthetic: "
        f"`{nguon_synth.relative_to(REPO_ROOT).as_posix()}` ({s_n} clip). "
        f"Đích: AudioSet Strong ({g_n} clip).", "",
        "Độ phủ sóng = tỉ lệ thời lượng clip nằm trong HỢP các khoảng sự kiện "
        "foreground (đã gộp chồng lấn).", "",
    ]
    dong += bang_doi_chieu(s_lop, s_phu, s_n, g_lop, g_phu, g_n)
    dong += bang_theo_lop(s_lop, "SYNTHETIC — theo lớp")
    dong += bang_theo_lop(g_lop, "AUDIOSET Strong (đích) — theo lớp")
    return "\n".join(dong) + "\n"


def main() -> None:
    enable_utf8_output()
    parser = argparse.ArgumentParser(description="Đo phân bố thời lượng sự kiện synthetic vs AudioSet")
    parser.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTHETIC,
                        help="thư mục split synthetic (chứa jams/)")
    parser.add_argument("--out", type=Path, default=None, help="đường dẫn file markdown xuất ra")
    args = parser.parse_args()

    nguon = duong_dan_tuyet_doi(args.synthetic)
    synth = doc_jams(nguon)
    gold = doc_audioset(AUDIOSET_SEGMENTS)
    bao_cao = dung_bao_cao(synth, gold, nguon)

    if args.out:
        ra = duong_dan_tuyet_doi(args.out)
        ra.parent.mkdir(parents=True, exist_ok=True)
        ra.write_text(bao_cao, encoding="utf-8")
        print(f"✓ {ra.relative_to(REPO_ROOT)}")
    else:
        print(bao_cao)


if __name__ == "__main__":
    main()
