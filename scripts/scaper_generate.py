"""Sinh train/dev synthetic kèm strong label bằng Scaper — DATA_PLAN §7.

    python scripts/scaper_generate.py --split train
    python scripts/scaper_generate.py --split dev --limit 20
    python scripts/scaper_generate.py --split train --plan-only

Scaper lưu thời điểm đặt nguồn vào hỗn hợp. Đây không phải ground truth đã nghe
kiểm chứng: nguồn có khoảng lặng và các slice vẫn cần verify_synthetic.py.

Ba điều script này làm mà gọi thẳng Scaper không làm được:
  1. CHẶN RÒ RỈ ở đầu vào. Chỉ clip được chia vào foreground_bank_train mới được
     dùng làm nguyên liệu. Một clip của gold_test lọt vào đây là hỏng cả bài test,
     và không có triệu chứng nào ngoài việc điểm số đẹp lên.
  2. LẬP KẾ HOẠCH SLICE. Tỉ lệ plan không thay thế QA audio; recipe legacy còn
     chưa ép đúng overlap/long_event, xem verify_synthetic.py và docs/STATUS.md.
  3. CHUỖI NHÂN QUẢ, gồm hai chuỗi ÂM TÍNH (pháo hoa + reo hò, rơi bát đĩa + cười).
     Thiếu chúng model học "cứ có chuỗi sự kiện là nguy hiểm".

Cần SoX binary — xem requirements-data.txt.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from common import REPO_ROOT, enable_utf8_output

CONFIG_PATH = REPO_ROOT / "ml" / "configs" / "scaper_train.yaml"
SPLITS_PATH = REPO_ROOT / "data" / "manifests" / "splits.csv"
FOREGROUND_DIR = REPO_ROOT / "data" / "banks" / "foreground"
# Bank nền LIÊN TỤC do build_background_10s.py dựng, không phải bank 4 giây gốc.
# Bank gốc giữ nguyên làm nguyên liệu cho chính bước dựng đó.
BACKGROUND_DIR = REPO_ROOT / "data" / "banks" / "background_10s"
RIR_DIR = REPO_ROOT / "data" / "banks" / "rir"
OUTPUT_DIR = REPO_ROOT / "data" / "synthetic"

BANG_TRA_PATH = REPO_ROOT / "data" / "manifests" / "bank_trim.csv"
MANIFEST_NEN_PATH = REPO_ROOT / "data" / "manifests" / "background_10s_manifest.csv"

BANK_SPLIT = "foreground_bank_train"
SLICE_NAMES = ["overlap", "low_snr", "reverb", "long_event", "causal_chain"]
# `reverb` là phép tích chập áp lên cả clip nên clip im lặng vẫn mang được;
# bốn lát còn lại định nghĩa trên sự kiện nên không.
EVENT_BASED_SLICES = {"overlap", "low_snr", "long_event", "causal_chain"}


@dataclass
class ClipPlan:
    """Công thức của một clip, quyết định TRƯỚC khi đụng tới audio.

    Tách hẳn khỏi phần sinh audio để kiểm thử được: mọi lỗi tỉ lệ slice hay chuỗi
    nhân quả đều nằm ở đây, mà phần này không cần SoX, không cần bank, chạy trong
    mili giây.
    """

    index: int
    slices: set[str] = field(default_factory=set)
    n_events: int = 0
    chain: str | None = None
    area_type: str = ""
    rir_used: str = ""


# ── Cổng chống rò rỉ ─────────────────────────────────────────────────────────


def bank_eligible_ids() -> set[str]:
    """file_id được phép làm nguyên liệu. Đọc từ splits.csv, không đoán theo thư mục."""
    if not SPLITS_PATH.exists():
        raise SystemExit("❌ chưa có data/manifests/splits.csv — chạy scripts/make_splits.py trước")
    with SPLITS_PATH.open(encoding="utf-8", newline="") as handle:
        return {row["file_id"] for row in csv.DictReader(handle) if row["split"] == BANK_SPLIT}


def check_bank_clean(bank_files: list[Path], eligible: set[str]) -> list[str]:
    """file trong bank foreground mà KHÔNG thuộc foreground_bank_train.

    Trả danh sách vi phạm thay vì ném lỗi, để gọi được từ test và để báo cáo liệt kê
    hết một lượt chứ không dừng ở cái đầu tiên.
    """
    return sorted(p.stem for p in bank_files if p.stem not in eligible)


# ── Chọn nguồn (STATUS §7 B2) ────────────────────────────────────────────────
#
# Trước B2, `source_file=("choose", [])` giao việc chọn file cho Scaper. Tiện,
# nhưng script mất khả năng biết file nào được bốc — mà không biết file nào thì
# không đặt được `source_time` lẫn `event_duration` cho đúng file đó. Đó là gốc
# chung của cả ba lỗi: lấy phải im lặng đầu, cắt cụt sự kiện dài, và ép ngưỡng
# long_event lên những clip vốn không đủ dài.
#
# Giành lại quyền chọn file thì cả ba tan cùng lúc, và bank giữ nguyên bit-for-bit
# vì mọi thứ cần biết đã nằm trong bank_trim.csv.

COT_BANG_TRA = ["file_id", "class_id", "path", "dur", "eff", "lead", "trail", "rms_db", "peak", "loi"]

# Hai hằng số dưới đây cùng chặn một hỏng hóc ĐO ĐƯỢC: khi phân bố đích đòi thời
# lượng dài hơn thứ bank có, mọi lượt rút dài dồn hết vào vài clip dài nhất. Đo
# trên bank thật, 1.100 sự kiện mỗi lớp: MỘT file siren chiếm 57.5% số lượt của cả
# lớp. Model sẽ học thuộc đúng file đó thay vì học lớp siren.
#
# Quét K cho thấy một điểm ngọt rõ ràng (siren: lượt dùng file nhiều nhất · p95 ·
# KS tổng so với phân bố đích):
#
#     K= 1  57.5%  9.48s  KS 0.036      K= 8  10.5%  9.48s  KS 0.049
#     K= 2  36.2%  9.48s  KS 0.036      K=16   5.6%  9.16s  KS 0.064
#     K= 4  18.7%  9.48s  KS 0.037      K=32   3.3%  5.13s  KS 0.078
#
# K=8 cắt mức dùng lại 5,5 lần mà p95 của siren KHÔNG đổi và KS chỉ nhích 0.013.
# Từ K=16 trở lên mới bắt đầu ăn vào đuôi phân bố — đúng cái đuôi cần giữ.
SO_UNG_VIEN_TOI_THIEU = 8

# Sàn chặn hạ cấp vô lý: lớp chỉ có clip 1.0 s và 3.7 s, hỏi 9 s thì phải trả 3.7 s
# chứ không phải 1.0 s. Đo được sàn 0.5 gần như miễn phí (khớp 91.5% → 92.1%, mọi
# chỉ số khác giữ nguyên), còn sàn 0.7 thì siết quá: mức dùng lại vọt lại 36.4%.
SAN_SO_VOI_DAI_NHAT = 0.50


@dataclass(frozen=True)
class ClipNguon:
    """Một clip bank kèm số đo đã biết trước — đọc từ bank_trim.csv."""

    file_id: str
    class_id: str
    path: str        # tương đối gốc repo, để CSV đi được sang máy khác
    dur: float
    eff: float       # độ dài phần CÓ TIẾNG
    lead: float      # im lặng đầu, chính là source_time cần dùng

    @property
    def duong_dan_day_du(self) -> Path:
        return REPO_ROOT / self.path


def doc_bang_tra(path: Path = BANG_TRA_PATH) -> dict[str, tuple[ClipNguon, ...]]:
    """{lớp: clip đã sắp theo file_id}. Bỏ clip lỗi và clip không có tiếng.

    Clip `eff == 0` lọt xuống dưới sẽ thành `event_duration=0`, và Scaper gặp giá
    trị đó thì tự rút một giá trị khác — nghĩa là mất kiểm soát đúng cái ta vừa
    giành lại. Loại ngay tại đây.
    """
    if not path.exists():
        raise SystemExit(f"❌ chưa có {path.name} — chạy scripts/probe_bank.py trước")
    theo_lop: dict[str, list[ClipNguon]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("loi") or float(row["eff"]) <= 0.0:
                continue
            theo_lop.setdefault(row["class_id"], []).append(ClipNguon(
                file_id=row["file_id"], class_id=row["class_id"], path=row["path"],
                dur=float(row["dur"]), eff=float(row["eff"]), lead=float(row["lead"]),
            ))
    return {lop: tuple(sorted(ds, key=lambda c: c.file_id)) for lop, ds in sorted(theo_lop.items())}


def loc_hop_le(bang: dict[str, tuple[ClipNguon, ...]],
               eligible: set[str]) -> tuple[dict[str, tuple[ClipNguon, ...]], list[str]]:
    """(bảng đã lọc, file_id vi phạm). Lớp mất sạch clip bị bỏ hẳn, không để rỗng.

    Lớp rỗng lọt xuống dưới sẽ thành `rng.choice([])` → IndexError giữa chừng một
    lượt sinh 18 giờ, chứ không phải lỗi ngay lúc bắt đầu.
    """
    sach: dict[str, tuple[ClipNguon, ...]] = {}
    vi_pham: list[str] = []
    for lop, ds in bang.items():
        giu = tuple(c for c in ds if c.file_id in eligible)
        vi_pham += [c.file_id for c in ds if c.file_id not in eligible]
        if giu:
            sach[lop] = giu
    return sach, sorted(vi_pham)


def chon_nguon(rng: random.Random, ung_vien: tuple[ClipNguon, ...],
               d_muon: float) -> tuple[ClipNguon, float]:
    """(clip được chọn, thời lượng THỰC SỰ dùng được) — d_thực ≤ eff luôn đúng.

    Không clip nào đủ dài thì hạ `d_thực` xuống đúng `eff` của clip được chọn, và
    trả con số đã hạ ra ngoài. Hạ xuống là chuyện bình thường (siren muốn 8.99 s
    mà bank chỉ tới 9.48 s ở 2 clip); hạ xuống mà vẫn khai con số cũ mới là lỗi —
    đó chính là cách hợp đồng `long_event ≥ 8s` báo đạt 791 clip trong khi số sự
    kiện ≥ 8 s thật sự sinh ra là 0.
    """
    if not ung_vien:
        raise ValueError("không có clip nguồn nào để chọn")
    du_dai = [c for c in ung_vien if c.eff >= d_muon]
    if len(du_dai) >= SO_UNG_VIEN_TOI_THIEU:
        return rng.choice(du_dai), d_muon

    # Chưa đủ ứng viên: nới dần xuống theo eff giảm dần cho tới khi đủ K clip.
    # Clip nào đã đáp ứng `d_muon` thì giữ nguyên `d_muon`; clip nới thêm thì hạ
    # `d_thực` xuống eff của chính nó — hạ bao nhiêu phải NÓI RA bấy nhiêu.
    dai_nhat = max(c.eff for c in ung_vien)
    nhom = list(du_dai)
    for clip in sorted(ung_vien, key=lambda c: (-c.eff, c.file_id)):
        if len(nhom) >= SO_UNG_VIEN_TOI_THIEU:
            break
        if clip not in nhom and clip.eff >= dai_nhat * SAN_SO_VOI_DAI_NHAT:
            nhom.append(clip)
    chon = rng.choice(nhom)
    return chon, min(d_muon, chon.eff)


def tham_so_su_kien(nguon: ClipNguon, d_thuc: float, snr: tuple[float, float],
                    events: dict, rng: random.Random) -> dict:
    """Bộ tham số cho `Scaper.add_event` — THIẾU `event_time`, kèm `thoi_luong_cuoi`.

    Cố ý chưa có `event_time`. Thời lượng cuối của sự kiện là `event_duration ×
    time_stretch`, mà hệ số kéo rút ở đây; đặt vị trí trước rồi mới rút hệ số kéo
    thì sự kiện co lại ±10% SAU khi đã tính chỗ, và cặp bị ép của lát cắt `overlap`
    tụt xuống dưới ngưỡng ở một phần clip — hợp đồng hỏng lại, vẫn không triệu chứng.
    Nơi gọi phải đặt vị trí theo `thoi_luong_cuoi` rồi mới gắn `event_time` vào.

    Hai điều phải tự tính thay vì để Scaper rút:

    1. TIME_STRETCH. Scaper nhân độ dài với hệ số kéo SAU khi cắt. Để nó tự rút
       thì nhãn cuối lệch tới ±10% so với con số ta định — đủ để một sự kiện định
       là 4.00 s ra 3.61 s và trượt hợp đồng long_event mà không có triệu chứng.
       Rút sẵn rồi chia ngược lại thì thời lượng cuối đúng bằng `d_thuc`.

    2. GIỚI HẠN THEO NGUỒN. Chia ngược cho hệ số kéo < 1 làm `event_duration` dài
       ra, có thể vượt quá phần có tiếng của file. Cắt về `eff` và tính lại thời
       lượng cuối, chứ không để Scaper cắt ngầm.

    `thoi_luong_cuoi` KHÔNG phải tham số của Scaper — nơi gọi phải bóc ra trước
    khi truyền đi; nó có mặt để khâu lập kế hoạch biết nhãn thật sẽ dài bao nhiêu.
    """
    pitch_low, pitch_high = events["pitch_shift_semitones"]
    stretch_low, stretch_high = events["time_stretch"]
    keo = rng.uniform(float(stretch_low), float(stretch_high))
    doc_tu_nguon = min(d_thuc / keo, nguon.eff)

    # Chốt chặn, cố ý đặt ở đây chứ không ở chỗ khác: `quyet_dinh_clip` gọi hàm này,
    # nên bộ mô phỏng B7 chạm tới nó — và vì thế nó bắt được cả ràng buộc của Scaper
    # trong 3 giây, thay vì để lượt sinh chết ở clip 7.815 sau 45 phút. Đúng ca đã
    # xảy ra: nhãn AudioSet 0.001 giây → sự kiện 16 mẫu → Scaper vỡ ở phép fade.
    if doc_tu_nguon < TONG_FADE_SCAPER_SEC:
        raise ValueError(
            f"sự kiện `{nguon.class_id}` chỉ dài {doc_tu_nguon:.4f}s, dưới giới hạn "
            f"{TONG_FADE_SCAPER_SEC}s của Scaper (fade 10 ms vào + 10 ms ra). "
            f"Nguồn {nguon.file_id} eff={nguon.eff:.4f}s, thời lượng muốn {d_thuc:.4f}s."
        )
    return {
        "label": ("const", nguon.class_id),
        "source_file": ("const", str(nguon.duong_dan_day_du)),
        "source_time": ("const", nguon.lead),
        "event_duration": ("const", doc_tu_nguon),
        "snr": ("const", rng.uniform(float(snr[0]), float(snr[1]))),
        "pitch_shift": ("const", rng.uniform(float(pitch_low), float(pitch_high))),
        "time_stretch": ("const", keo),
        "thoi_luong_cuoi": doc_tu_nguon * keo,
    }


# ── Rút thời lượng theo miền đích (STATUS §7 B3) ─────────────────────────────
#
# `event_duration=("const", 1.0)` làm toàn bộ 16.545 sự kiện của lô cũ rơi vào
# [0.19 s, 1.10 s] — không một ngoại lệ. Bộ đánh giá thật thì có p95 = 7.94 s và
# 17.5% sự kiện vượt 2 giây. Đó là lệch phân bố train/test, không phải lỗi lặt vặt.
#
# Câu hỏi đúng không phải "nên đặt hằng số bằng bao nhiêu" mà "sự kiện lớp này
# dài bao nhiêu trong thế giới thật". AudioSet Strong trả lời sẵn cho cả 15 lớp,
# nên lấy mẫu lại từ chính nó thay vì đoán một con số.
#
# Mô phỏng trên bank thật: 94.3% phân bố đích là cấp được. Ba lớp hụt vì giới hạn
# vật lý của nguồn — siren 52.7% (UrbanSound8K cắt sẵn 4 s), scream 85.8%,
# explosion 87.4%. Hụt thì `chon_nguon` hạ xuống VÀ trả con số đã hạ ra ngoài.

AUDIOSET_SEGMENTS_PATH = REPO_ROOT / "data" / "raw" / "audioset_strong" / "segments.jsonl"

# Sàn thời lượng sự kiện. AudioSet Strong có nhãn thoái hoá xuống tới 0.001 giây —
# không phải sự kiện âm thanh mà là lỗi gán nhãn. Để lọt thì Scaper NỔ giữa chừng:
# nó fade 10 ms vào và 10 ms ra, nên `event_audio[:160] *= cua_so` vỡ khi sự kiện chỉ
# có 16 mẫu. Đã xảy ra thật lúc 18:0x, làm hỏng lượt sinh ở clip thứ ~7.815/7.920.
#
# 0.05 s là gấp 2,5 lần giới hạn kỹ thuật (20 ms) mà chỉ bỏ 0,20% phân bố đích
# (7/3.482). Nới xuống 0.02 s bỏ 0,14% nhưng sát mép Scaper; siết lên 0.1 s bỏ tới
# 2,10% — lúc đó ta đang nắn lại miền đích chứ không còn là dọn nhãn hỏng.
# Clip ngắn nhất trong bank là 0,1016 s, nên sàn này không chạm tới bank.
TOI_THIEU_SU_KIEN_SEC = 0.05

# Giới hạn CỨNG của Scaper: fade 10 ms vào + 10 ms ra. Ngắn hơn là nó vỡ.
TONG_FADE_SCAPER_SEC = 0.02
AUDIOSET_EVAL_LABELS = (REPO_ROOT / "data" / "raw" / "audioset_strong" / "labels"
                        / "audioset_eval_strong.tsv")


def ytid_eval_strong(path: Path = AUDIOSET_EVAL_LABELS) -> set[str]:
    """ytid thuộc split `eval_strong` của AudioSet — phần GIỮ NGUYÊN cho gold_test.

    `segment_id` có dạng `<ytid>_<start_ms>`, mà ytid của YouTube chứa được cả `-`
    lẫn `_`, nên chỉ được cắt đúng nhóm số cuối cùng.
    """
    if not path.exists():
        raise SystemExit(
            f"❌ chưa có {path.name} — không xác định được đâu là phía eval_strong.\n"
            f"   Chạy scripts/fetch_audioset_strong.py trước. KHÔNG gộp cả hai split:\n"
            f"   xem docs/decisions/ADR-0006-phan-bo-dich-chi-tu-train-strong.md"
        )
    with path.open(encoding="utf-8") as handle:
        next(handle, None)                                   # bỏ dòng tiêu đề
        return {dong.split("\t", 1)[0].rsplit("_", 1)[0] for dong in handle if dong.strip()}


def doc_phan_bo_dich(path: Path = AUDIOSET_SEGMENTS_PATH, do_dai_clip: float = 10.0,
                     loai_tru_ytid: set[str] | None = None,
                     nhan_eval: Path = AUDIOSET_EVAL_LABELS,
                     toi_thieu: float = TOI_THIEU_SU_KIEN_SEC) -> dict[str, tuple[float, ...]]:
    """{lớp: thời lượng các sự kiện có thật}, CHỈ lấy phía `train_strong`.

    Vì sao phải loại phía eval ra:

      `fetch_audioset_strong.py` gộp cả hai split khi dựng segments.jsonl, nên 937
      segment của ta = 825 từ train_strong + 112 từ eval_strong. Lấy phân bố trên
      cả 937 để THIẾT KẾ dữ liệu huấn luyện nghĩa là thống kê của những clip sẽ
      thành gold_test đã chảy ngược vào train. Không phải rò rỉ nhãn theo nghĩa
      nặng, nhưng đủ để một hội đồng hỏi mà ta không có câu trả lời sạch.

      Bỏ 112 clip đó ra gần như không đổi phân bố — đo trên dữ liệu thật,
      KS(937 vs 825) = 0.0083 — nên ta được sạch về phương pháp mà không mất gì.

    Cắt trần ở `do_dai_clip`: AudioSet ghi offset tới đúng 10.0 s, mà sự kiện dài
    hơn clip thì không đặt vào đâu được.

    Cắt SÀN ở `toi_thieu`: AudioSet có nhãn xuống tới 0.001 giây, và đó là lỗi gán
    nhãn chứ không phải sự kiện. Để lọt thì Scaper nổ giữa chừng lượt sinh — xem
    `TOI_THIEU_SU_KIEN_SEC`.
    """
    if not path.exists():
        raise SystemExit(f"❌ chưa có {path.name} — chạy scripts/fetch_audioset_strong.py trước")
    bo_qua = ytid_eval_strong(nhan_eval) if loai_tru_ytid is None else loai_tru_ytid
    theo_lop: dict[str, list[float]] = {}
    with path.open(encoding="utf-8") as handle:
        for dong in handle:
            if not dong.strip():
                continue
            ban_ghi = json.loads(dong)
            if ban_ghi.get("ytid") in bo_qua:
                continue
            for su_kien in ban_ghi["events"]:
                dai = min(float(su_kien["offset"]) - float(su_kien["onset"]), do_dai_clip)
                if dai >= toi_thieu:
                    theo_lop.setdefault(su_kien["class_id"], []).append(round(dai, 4))
    return {lop: tuple(sorted(ds)) for lop, ds in sorted(theo_lop.items())}


def kiem_phu_song(bang: dict[str, object], phan_bo: dict[str, tuple[float, ...]]) -> list[str]:
    """Lớp có trong bank mà KHÔNG có trong phân bố đích — liệt kê hết một lượt."""
    return sorted(lop for lop in bang if lop not in phan_bo)


def lop_cap_duoc_su_kien_dai(bang: dict[str, tuple[ClipNguon, ...]],
                             phan_bo: dict[str, tuple[float, ...]], nguong: float,
                             so_clip_toi_thieu: int = SO_UNG_VIEN_TOI_THIEU) -> tuple[str, ...]:
    """Lớp ép được sự kiện dài — phải đủ CẢ HAI điều kiện, không phải một.

    1. BANK có ít nhất `so_clip_toi_thieu` clip đủ dài. Ít hơn thì mọi sự kiện dài
       của lớp đó dồn vào vài file, và model học thuộc file thay vì học lớp.
    2. MIỀN ĐÍCH có ít nhất một sự kiện thật đạt ngưỡng. Thiếu điều kiện này thì ta
       ép ra một thứ không tồn tại ngoài đời.

    Điều kiện 2 không thừa: `object_drop_dishes` có 22 clip bank ≥ 4 s nhưng AudioSet
    không ghi nhận một tiếng rơi bát đĩa nào kéo dài 4 giây. Ép nó vào là dạy model
    một thứ sẽ không bao giờ gặp, và tệ hơn là làm nó bỏ qua tiếng rơi bát đĩa thật.

    Đo trên bank thật, số lớp đủ cả hai điều kiện: 2s → 14 · 3s → 9 · 4s → 8 ·
    6s → 6 · 8s → 4.
    """
    return tuple(
        lop for lop in sorted(bang)
        if sum(1 for c in bang[lop] if c.eff >= nguong) >= so_clip_toi_thieu
        and any(d >= nguong for d in phan_bo.get(lop, ()))
    )


def nguong_co_bien(nguong: float, events: dict) -> float:
    """Ngưỡng đã cộng biên cho `time_stretch`, dùng khi lọc nguồn cho sự kiện dài.

    Thời lượng ghi vào nhãn là `event_duration × time_stretch`, mà `event_duration`
    bị chặn trên bởi `eff` của file nguồn. Lấy đúng `eff = ngưỡng` rồi gặp hệ số kéo
    0.9 thì nhãn cuối chỉ còn 3.6 s — dưới ngưỡng 4 s đã công bố.

    Bộ mô phỏng B7 bắt được đúng ca này: 2/791 clip `long_event` trượt ngưỡng. Hai
    clip trên bảy trăm chín mốt là thứ không ai phát hiện bằng cách nghe, và nếu để
    lọt thì nó lại thành một hợp đồng "gần đúng" — chính thứ STATUS §7 đang dọn.

    Chia cho hệ số kéo NHỎ NHẤT là đủ: khi đó `eff × keo ≥ ngưỡng` với mọi keo hợp lệ.
    Đo trên bank thật, biên này không làm mất lớp nào — vẫn 8/15 lớp đủ điều kiện.
    """
    keo_nho_nhat = float(events["time_stretch"][0])
    return nguong / keo_nho_nhat if keo_nho_nhat > 0 else nguong


def ung_vien_su_kien_dai(ung_vien: tuple[ClipNguon, ...], nguong: float) -> tuple[ClipNguon, ...]:
    """Chỉ giữ clip đủ dài, để lọc TRƯỚC khi gọi `chon_nguon`.

    `chon_nguon` có nhánh hạ thời lượng khi bank thiếu nguồn. Nếu nhánh đó với tới
    được clip ngắn hơn ngưỡng thì hợp đồng `long_event` lại hỏng đúng như lô cũ, chỉ
    khác là lần này nhãn ghi 4 s thay vì 8 s. Lọc trước mới chặn được tận gốc: mọi
    clip còn lại đều có eff ≥ ngưỡng, nên `min(d_muốn, eff)` không thể tụt xuống dưới.

    Nơi gọi phải truyền `nguong_co_bien(...)`, không phải ngưỡng trần — xem lý do ở đó.
    """
    return tuple(c for c in ung_vien if c.eff >= nguong)


def rut_thoi_luong_dai(rng: random.Random, phan_bo: dict[str, tuple[float, ...]],
                       lop: str, nguong: float, toi_da: float) -> float:
    """Thời lượng lấy mẫu từ phần ĐUÔI của phân bố đích — chỉ các giá trị ≥ `nguong`.

    Vẫn là thời lượng có thật của lớp đó ngoài đời, chỉ là lấy trong nhánh dài. Kẹp
    cứng xuống đúng `nguong` thì mọi sự kiện dài đều dài y hệt nhau, và đó lại là một
    quy luật hình học của bộ sinh chứ không phải của âm thanh.

    Không có giá trị nào đạt ngưỡng thì NỔ, không lặng lẽ trả về giá trị ngắn nhất —
    lặng lẽ chính là hình dạng của lỗi `long_event` ở lô cũ.
    """
    duoi = [d for d in phan_bo.get(lop, ()) if d >= nguong]
    if not duoi:
        raise ValueError(
            f"long_event: lớp `{lop}` không có thời lượng nào ≥ {nguong}s trong miền đích"
        )
    return min(rng.choice(duoi), toi_da)


def rut_thoi_luong(rng: random.Random, phan_bo: dict[str, tuple[float, ...]],
                   lop: str, toi_da: float) -> float:
    """Một thời lượng lấy mẫu lại từ phân bố đích của `lop`, chặn trên bởi `toi_da`.

    Lấy mẫu lại (bootstrap) chứ không khớp tham số hay nội suy: mọi giá trị trả về
    đều là một thời lượng CÓ THẬT của lớp đó trong AudioSet. Nói được "ta khớp phân
    bố thời lượng của miền đích" mà không phải bảo vệ thêm giả định phân phối nào.

    Lớp thiếu thì ném KeyError chứ không lặng lẽ thay bằng mặc định — lặng lẽ thay
    bằng 1.0 s chính là cách lô cũ sinh ra 16.545 sự kiện dài đúng một giây.
    """
    if lop not in phan_bo:
        raise KeyError(f"lớp `{lop}` không có trong phân bố đích AudioSet Strong")
    return min(rng.choice(phan_bo[lop]), toi_da)


# ── Chọn nền (STATUS §7 B2.5) ────────────────────────────────────────────────
#
# Bank nền cũ dài 4 giây trong khi clip là 10 giây, nên Scaper lặp vòng nó 2,5
# lần: ~81% clip train mang một nền lặp đúng chu kỳ 4 giây, thứ không tồn tại
# trong bản ghi thật. `build_background_10s.py` tái dựng lại bản ghi gốc từ các
# lát chồng nhau 2 giây của UrbanSound8K và giữ NGUYÊN độ dài đơn vị.
#
# Giữ nguyên độ dài chỉ có nghĩa khi ta tự rút `source_time`: mỗi lần dùng là một
# cửa sổ 10 giây khác nhau trong cùng bản ghi, không tốn thêm byte nào trên đĩa.

COT_MANIFEST_NEN = ["file_id", "area_type", "dai_sec"]


@dataclass(frozen=True)
class NenNguon:
    file_id: str
    area_type: str
    dai_sec: float

    @property
    def duong_dan_day_du(self) -> Path:
        return BACKGROUND_DIR / self.area_type / f"{self.file_id}.wav"


def doc_bang_nen(path: Path = MANIFEST_NEN_PATH) -> dict[str, tuple[NenNguon, ...]]:
    """{khu vực: nền đã sắp theo file_id} — đọc từ background_10s_manifest.csv."""
    if not path.exists():
        raise SystemExit(f"❌ chưa có {path.name} — chạy scripts/build_background_10s.py trước")
    theo_khu: dict[str, list[NenNguon]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            theo_khu.setdefault(row["area_type"], []).append(
                NenNguon(row["file_id"], row["area_type"], float(row["dai_sec"])))
    return {khu: tuple(sorted(ns, key=lambda n: n.file_id)) for khu, ns in sorted(theo_khu.items())}


def chon_nen(rng: random.Random, ung_vien: tuple[NenNguon, ...],
             duration: float) -> tuple[NenNguon, float]:
    """(nền được chọn, source_time). Cửa sổ rút ngẫu nhiên, không bao giờ đọc quá đuôi.

    Đọc quá đuôi thì Scaper lặp vòng cho đủ — đúng thứ B2.5 vừa bỏ công chữa.
    """
    if not ung_vien:
        raise ValueError("không có nền nào để chọn")
    nen = rng.choice(ung_vien)
    thua = max(0.0, nen.dai_sec - duration)
    return nen, rng.uniform(0.0, thua) if thua > 0 else 0.0


# ── Đặt thời điểm (STATUS §7 B4 · B6) ────────────────────────────────────────
#
# Lô cũ đặt mọi sự kiện bằng `event_time=("uniform", 0, 10)`, kéo theo hai hỏng hóc
# đo được trên chính 7.920 file nhãn đã sinh:
#
#   · CHỒNG LẤN CHỈ LÀ TÌNH CỜ. Kế hoạch đánh dấu 2.387 clip (30.1%) mang lát cắt
#     `overlap`, nhưng chỉ 1.752 clip (22.1%) có chồng lấn thật, và `min_overlap_ratio`
#     chưa bao giờ được dùng tới lúc sinh.
#   · DỒN Ở MÉP. 813 sự kiện khởi đầu sau giây thứ 9 và 1.225 sự kiện (7.4%) bị cắt
#     ở mốc 10 s, sinh ra một đống mẩu cực ngắn không có trong miền đích.
#
# Ép chồng lấn LUÔN khả thi nên không có nhánh dự phòng nào ở đây — xem chứng minh
# trong docstring của `moc_chong_lan`. Không có nhánh dự phòng thì cũng không có
# chỗ cho một thất bại âm thầm.


def do_chong_lan(t_a: float, d_a: float, t_b: float, d_b: float) -> float:
    """Độ dài phần giao của hai sự kiện, 0 nếu rời nhau."""
    return max(0.0, min(t_a + d_a, t_b + d_b) - max(t_a, t_b))


def moc_ngau_nhien(rng: random.Random, d: float, duration: float) -> float:
    """Thời điểm bắt đầu ngẫu nhiên sao cho sự kiện nằm TRỌN trong clip."""
    return rng.uniform(0.0, max(0.0, duration - d))


# Số vị trí thử cho mỗi sự kiện KHÔNG bị ép chồng lấn. Đặt thuần ngẫu nhiên thì các
# sự kiện va vào nhau quá nhiều so với thực tế, và độ phủ sóng BÃO HOÀ: gần gấp đôi
# số sự kiện (1.90 → 3.59) chỉ nâng phủ sóng 35.7% → 47.7% vì phần thêm chui vào
# chỗ đã có tiếng.
#
# Miền đích không như vậy — sự kiện ở đó xếp gần như NỐI TIẾP: chỉ 17.1% thời lượng
# sự kiện bị chồng lấn (hệ số nén hợp/tổng = 0.829). Thử vài vị trí rồi lấy vị trí
# ít va chạm nhất kéo ta về đúng hành vi đó, và nâng phủ sóng mà KHÔNG phải tăng số
# sự kiện. Đo ở bộ trọng số D: 16 vị trí thử đưa phủ sóng 43.8% → 47.0% đồng thời
# hạ chồng lấn 68.2% → 53.0%. Cả hai chỉ số cùng tốt lên, hiếm khi có đánh đổi kiểu đó.
SO_VI_TRI_THU = 16


def khoang_trong(da_dat: list[tuple[float, float]], duration: float,
                 d: float) -> list[tuple[float, float]]:
    """Các khoảng `[thấp, cao]` mà đặt sự kiện dài `d` vào sẽ KHÔNG chồng lấn gì.

    Rỗng nghĩa là clip không còn chỗ trống nào đủ rộng — lúc đó `moc_it_va_cham`
    lo phần còn lại.
    """
    if not da_dat:
        return [(0.0, duration - d)] if duration >= d else []
    gop: list[list[float]] = []
    for dau, cuoi in sorted((t, t + dai) for t, dai in da_dat):
        if gop and dau <= gop[-1][1]:
            gop[-1][1] = max(gop[-1][1], cuoi)
        else:
            gop.append([dau, cuoi])

    trong, truoc = [], 0.0
    for dau, cuoi in gop:
        if dau - truoc >= d:
            trong.append((truoc, dau - d))
        truoc = max(truoc, cuoi)
    if duration - truoc >= d:
        trong.append((truoc, duration - d))
    return trong


def moc_trong_khoang_trong(rng: random.Random, trong: list[tuple[float, float]]) -> float:
    """Một điểm trong các khoảng trống, rút ĐỀU theo độ dài chứ không đều theo số khoảng.

    Đều theo số khoảng sẽ dồn sự kiện vào các khe hẹp, vì khe hẹp cũng nhiều phiếu
    như khe rộng — kết quả là sự kiện bám dính nhau ở vài chỗ thay vì trải ra.
    """
    thap, cao = rng.choices(trong, weights=[c - t + 1e-9 for t, c in trong])[0]
    return rng.uniform(thap, cao)


def moc_it_va_cham(rng: random.Random, d: float, duration: float,
                   da_dat: list[tuple[float, float]],
                   so_thu: int = SO_VI_TRI_THU) -> float:
    """Thời điểm bắt đầu ít chồng lấn nhất trong `so_thu` vị trí rút thử.

    Không phải "tránh chồng lấn bằng mọi giá": hết chỗ trống thì nó chọn vị trí ít
    va chạm nhất trong số đã thử, chứ không bỏ cuộc hay ném lỗi. Chồng lấn CÓ THẬT
    ngoài đời, chỉ là ít hơn nhiều so với đặt ngẫu nhiên đều.

    Luôn rút đủ `so_thu` lần, không dừng sớm khi tìm được vị trí không va chạm. Dừng
    sớm không sai, nhưng làm số lần rút phụ thuộc nội dung clip, và khi đó đọc một
    dãy RNG để dò lỗi trở nên rất khó. `dat_cum` mới là chỗ quyết định có gọi hàm này
    hay không — nó ưu tiên `khoang_trong` trước.
    """
    ung_vien = [moc_ngau_nhien(rng, d, duration) for _ in range(max(1, so_thu))]
    if not da_dat:
        return ung_vien[0]
    return min(ung_vien, key=lambda t: sum(do_chong_lan(t, d, ta, td) for ta, td in da_dat))


def moc_chong_lan(rng: random.Random, t_neo: float, d_neo: float, d: float,
                  duration: float, ty_le: float) -> float:
    """Thời điểm bắt đầu để giao với sự kiện neo đạt ít nhất `ty_le × min(d_neo, d)`.

    Cửa sổ hợp lệ KHÔNG BAO GIỜ RỖNG, nên hàm này không cần nhánh dự phòng. Gọi
    o = ty_le · min(d_neo, d), tập t thoả `giao ≥ o` là [t_neo + o − d, t_neo + d_neo − o];
    giao nó với [0, duration − d] cho cửa sổ cuối. Bốn cách làm cửa sổ đó rỗng đều
    bất khả:

        t_neo + d_neo − o ≥ 0        vì o ≤ d_neo
        t_neo + o − d ≤ duration − d vì t_neo ≤ duration − d_neo và o ≤ d_neo
        t_neo + o − d ≤ t_neo + d_neo − o  vì 2o ≤ d_neo + d
        duration − d ≥ 0             vì mọi sự kiện đều ngắn hơn clip

    Rút ĐỀU trên cửa sổ đó chứ không đặt đúng bằng ngưỡng: luôn chồng đúng 30% thì
    model học được một quy luật hình học của bộ sinh, không phải chồng lấn âm thanh.
    """
    giao_toi_thieu = ty_le * min(d_neo, d)
    thap = max(0.0, t_neo + giao_toi_thieu - d)
    cao = min(max(0.0, duration - d), t_neo + d_neo - giao_toi_thieu)
    return rng.uniform(thap, max(thap, cao))


def dat_cum(rng: random.Random, thoi_luong: list[float], duration: float,
            ty_le_chong_lan: float | None = None,
            moc_co_dinh: list[float] | None = None,
            so_vi_tri_thu: int = SO_VI_TRI_THU) -> list[float]:
    """Thời điểm bắt đầu cho cả cụm sự kiện của một clip.

    `moc_co_dinh` là các mốc đã định trước — thời điểm của chuỗi nhân quả. Thứ tự
    và khoảng cách của chuỗi mới là thứ mang nghĩa (glass_breaking → scream →
    running_footsteps là đột nhập, đảo lại thì không), nên ép chồng lấn không được
    xê dịch chúng; cặp bị ép luôn nằm trong phần sự kiện tự do.
    """
    co_dinh = list(moc_co_dinh or [])
    tu_do = thoi_luong[len(co_dinh):]
    if ty_le_chong_lan is not None and len(tu_do) < 2:
        raise ValueError("ép chồng lấn cần ít nhất hai sự kiện tự do trong clip")

    moc = list(co_dinh)
    da_dat = list(zip(co_dinh, thoi_luong, strict=False))
    for thu_tu, d in enumerate(tu_do):
        if ty_le_chong_lan is not None and thu_tu == 1:
            t = moc_chong_lan(rng, moc[len(co_dinh)], tu_do[0], d, duration, ty_le_chong_lan)
        else:
            # Còn chỗ trống thì đặt hẳn vào đó, không chồng lấn chút nào. Hết chỗ mới
            # chấp nhận va chạm ít nhất có thể. Đo được: bước này nâng tỉ lệ clip
            # KHÔNG chồng lấn từ 10–16% lên 33–51%, và nâng cả độ phủ sóng — hiếm khi
            # có thay đổi cùng lúc tốt cho hai chỉ số đối nghịch.
            trong = khoang_trong(da_dat, duration, d)
            t = (moc_trong_khoang_trong(rng, trong) if trong
                 else moc_it_va_cham(rng, d, duration, da_dat, so_vi_tri_thu))
        moc.append(t)
        da_dat.append((t, d))
    return moc


# ── Lập kế hoạch ─────────────────────────────────────────────────────────────


def sample_event_count(rng: random.Random, weights: dict) -> int:
    """Số sự kiện trong clip. Nhánh 0 sự kiện (~10%) là nhánh dạy model biết im lặng."""
    counts = sorted(int(k) for k in weights)
    return rng.choices(counts, weights=[float(weights[k]) for k in counts])[0]


def silent_probability(config: dict) -> float:
    weights = config["events"]["count_weights"]
    total = sum(float(v) for v in weights.values())
    return float(weights.get(0, weights.get("0", 0.0))) / total


def slice_probabilities(config: dict) -> dict[str, float]:
    """Xác suất rút cho từng lát cắt, đã bù phần clip im lặng.

    Tỉ lệ trong config là tỉ lệ trên TOÀN BỘ split — đó là con số sẽ viết vào báo
    cáo. Nhưng 4 trong 5 lát cắt định nghĩa trên sự kiện nên không gán được cho
    clip im lặng (~10%). Rút thẳng theo tỉ lệ đích sẽ cho kết quả thấp hơn đích
    đúng bằng 10%: đo được 26.9% trong khi tài liệu ghi 30%. Chênh lệch nhỏ và
    không có triệu chứng, nên chia cho phần clip có sự kiện ngay tại đây.
    """
    forced = config["forced_slices"]
    with_events = 1.0 - silent_probability(config)
    probability = {}
    for name in SLICE_NAMES:
        ratio = float(forced[name]["ratio"])
        if name in EVENT_BASED_SLICES:
            if ratio > with_events:
                raise SystemExit(
                    f"❌ lát cắt `{name}` đòi {ratio:.0%} nhưng chỉ {with_events:.0%} clip có sự kiện.\n"
                    f"   Giảm ratio, hoặc giảm trọng số clip 0 sự kiện trong scaper_train.yaml."
                )
            ratio /= with_events
        probability[name] = ratio
    return probability


def so_su_kien_tu_do_can(slices: set[str], forced: dict) -> int:
    """Số sự kiện KHÔNG thuộc chuỗi mà clip cần có, để ép được các lát cắt của nó.

    `max` chứ không cộng dồn: sự kiện dài được phép đồng thời là một nửa của cặp
    chồng lấn, nên `overlap` và `long_event` dùng chung ngân sách.
    """
    return max(
        int(forced["overlap"]["min_events"]) if "overlap" in slices else 0,
        1 if "long_event" in slices else 0,
    )


def plan_clips(config: dict, n_clips: int, seed: int, chains: list[dict] | None = None,
               areas: list[str] | None = None) -> list[ClipPlan]:
    """Kế hoạch cho n_clips, tất định theo seed.

    Mỗi slice rút Bernoulli ĐỘC LẬP theo tỉ lệ của nó, nên một clip có thể vừa
    overlap vừa reverb — đúng ý đồ: các lát cắt trong DATA_PLAN chồng lấn nhau,
    tổng tỉ lệ của chúng vượt 100%. Chia rời thành 5 nhóm loại trừ nhau sẽ làm sai
    tỉ lệ từng lát và không sinh được ca khó nhất (low_snr + overlap cùng lúc).
    """
    forced = config["forced_slices"]
    chains = config["causal_chains"] if chains is None else chains
    rng = random.Random(seed)
    probability = slice_probabilities(config)

    plans = []
    for index in range(n_clips):
        plan = ClipPlan(index=index)
        # Khu vực nền chọn TRƯỚC, vì loại phòng dùng để tích chập phải khớp với nó.
        plan.area_type = rng.choice(areas) if areas else ""
        plan.n_events = sample_event_count(rng, config["events"]["count_weights"])
        silent = plan.n_events == 0
        for name in SLICE_NAMES:
            # Clip im lặng không mang được lát cắt nào định nghĩa trên sự kiện.
            if silent and name in EVENT_BASED_SLICES:
                continue
            if rng.random() < probability[name]:
                plan.slices.add(name)

        if "overlap" in plan.slices:
            plan.n_events = max(plan.n_events, int(forced["overlap"]["min_events"]))
        if "causal_chain" in plan.slices and not chains:
            plan.slices.discard("causal_chain")     # không còn chuỗi nào sinh được
        if "causal_chain" in plan.slices:
            plan.chain = rng.choice(chains)["name"]
            so_mat_xich = len(chain_by_name(chains, plan.chain)["sequence"])
            # Chuỗi ăn hết ngân sách sự kiện thì `overlap` và `long_event` không còn
            # gì để ép. Lô cũ rơi đúng vào đây: clip mang nhãn lát cắt mà 0 sự kiện
            # tự do, và nó vẫn lặng lẽ tính vào tỉ lệ trong báo cáo. Cộng thêm chỗ,
            # chứ không ép vào chính chuỗi — xê dịch mắt xích là đổi nghĩa của chuỗi.
            #
            # Lấy max chứ không cộng dồn: sự kiện dài ĐƯỢC PHÉP là một nửa của cặp
            # chồng lấn, nên hai lát cắt dùng chung ngân sách.
            plan.n_events = max(plan.n_events, so_mat_xich + so_su_kien_tu_do_can(plan.slices, forced))
        plans.append(plan)
    return plans


def usable_chains(chains: list[dict], available: set[str]) -> tuple[list[dict], list[tuple[str, list[str]]]]:
    """(chuỗi dùng được, [(tên chuỗi bỏ, các lớp còn thiếu)]).

    Chuỗi nhân quả chỉ có nghĩa khi ĐỦ mọi mắt xích. Thiếu một lớp thì hoặc Scaper
    chết giữa chừng, hoặc — tệ hơn — ta lặng lẽ sinh ra chuỗi cụt mang nhãn của chuỗi
    đủ: `glass_breaking → scream` vẫn được ghi là kịch bản "đột nhập" dù thiếu hẳn
    tiếng chân chạy, và model học một định nghĩa sai về đột nhập.

    Bỏ hẳn chuỗi và NÓI RA là lựa chọn đúng: thiếu `shout_yell` thì hai kịch bản
    forced_entry và emergency không sinh được, và đó là thông tin phải biết chứ không
    phải chi tiết cần giấu.
    """
    ok, bo = [], []
    for chain in chains:
        thieu = [label for label in chain["sequence"] if label not in available]
        (bo.append((chain["name"], thieu)) if thieu else ok.append(chain))
    return ok, bo


def available_labels() -> set[str]:
    """Các lớp THẬT SỰ có clip trong bank foreground — đọc từ đĩa, không từ config."""
    if not FOREGROUND_DIR.exists():
        return set()
    return {p.name for p in FOREGROUND_DIR.iterdir() if p.is_dir() and any(p.glob("*.wav"))}


def chain_by_name(chains: list[dict], name: str) -> dict:
    for chain in chains:
        if chain["name"] == name:
            return chain
    raise SystemExit(f"❌ không có chuỗi tên `{name}` trong scaper_train.yaml")


def ngan_sach_mat_xich(chain: dict, duration: float) -> float:
    """Thời lượng tối đa cho MỖI mắt xích để cả chuỗi lọt vừa clip.

    Từ B3 mắt xích có thời lượng thật lấy từ miền đích, và `applause_cheering` có
    p50 = 3,5 s. Chuỗi 3 mắt xích cộng khoảng cách vượt 10 giây rất dễ, lúc đó
    `chain_event_times` trả rỗng và nhánh dự phòng đặt chúng TỰ DO — mất thứ tự, mà
    clip vẫn mang nhãn `causal_chain`. Cổng kiểm bắt được 91 clip như vậy.

    Chia đều phần còn lại sau khi trừ các khoảng cách NHỎ NHẤT: `chain_event_times`
    co khoảng cách về mức nhỏ nhất khi chật, nên chuỗi luôn lọt vừa.

    Thời lượng cuối không bao giờ vượt `d_thuc` — vì `min(d/keo, eff) × keo =
    min(d, eff×keo) ≤ d` — nên chặn `d_thuc` ở ngân sách này là đủ, không cần biên.
    """
    gaps_spec = chain.get("gaps_sec") or []
    con_lai = duration - sum(float(low) for low, _ in gaps_spec)
    return max(0.0, con_lai) / max(1, len(chain["sequence"]))


def chain_event_times(chain: dict, rng: random.Random, duration: float,
                      event_duration: float | list[float] = 1.0) -> list[float]:
    """Thời điểm bắt đầu của từng sự kiện trong chuỗi, giữ đúng THỨ TỰ và khoảng cách.

    Thứ tự mới là thứ mang nghĩa: `glass_breaking → scream → running_footsteps` là
    đột nhập, còn đảo ngược thì không. Nếu chuỗi tràn khỏi clip thì co khoảng cách
    về mức nhỏ nhất; vẫn tràn thì trả rỗng để nơi gọi bỏ qua chuỗi này, chứ không
    cắt cụt chuỗi thành một chuỗi khác nghĩa.

    `event_duration` nhận được cả một số lẫn một danh sách một thời lượng cho mỗi
    mắt xích. Từ B3 mỗi mắt xích có thời lượng riêng lấy từ miền đích, nên dùng một
    hằng số chung sẽ tính sai chỗ trống mà chuỗi cần — chuỗi vẫn sinh ra được, chỉ
    là các mắt xích chồng lên nhau hoặc tràn khỏi clip.
    """
    mat_xich = chain["sequence"]
    dai = ([float(event_duration)] * len(mat_xich)
           if isinstance(event_duration, (int, float)) else list(event_duration))
    gaps_spec = chain.get("gaps_sec") or []
    gaps = [rng.uniform(low, high) for low, high in gaps_spec]
    need = lambda gs: sum(dai) + sum(gs)  # noqa: E731

    if need(gaps) > duration:
        gaps = [low for low, _ in gaps_spec]          # co về khoảng cách nhỏ nhất
    if need(gaps) > duration:
        return []

    start = rng.uniform(0, duration - need(gaps))
    times, cursor = [], start
    for position in range(len(mat_xich)):
        times.append(round(cursor, 3))
        if position < len(gaps):
            cursor += dai[position] + gaps[position]
    return times


def slice_report(plans: list[ClipPlan], config: dict) -> str:
    lines = [f"{'lát cắt':<16}{'đích':>8}{'thực tế':>10}{'clip':>8}"]
    lines.append("─" * 42)
    for name in SLICE_NAMES:
        target = float(config["forced_slices"][name]["ratio"])
        hit = sum(1 for p in plans if name in p.slices)
        lines.append(f"{name:<16}{target:>7.0%}{hit / len(plans):>10.1%}{hit:>8}")
    silent = sum(1 for p in plans if p.n_events == 0)
    lines.append(f"\n{silent} clip không có sự kiện ({silent / len(plans):.1%}) — nhánh dạy model biết im lặng")
    return "\n".join(lines)


# ── Quyết định nội dung clip (STATUS §7 B7) ──────────────────────────────────
#
# Bài học đắt nhất của dự án nằm ở đây. Hợp đồng `long_event ≥ 8s` khai đạt 791 clip
# trong khi số sự kiện thật sự sinh ra là 0, và nó sống sót qua hai vòng train.
# Nguyên nhân không phải một phép tính sai mà là KHOẢNG CÁCH giữa thứ được lập kế
# hoạch và thứ được sinh ra: log báo cái đầu, không ai đọc cái sau.
#
# `quyet_dinh_clip` đóng khoảng cách đó bằng CẤU TRÚC chứ không bằng kỷ luật. Nó là
# nơi duy nhất quyết định một clip chứa gì. Bộ mô phỏng đọc thẳng kết quả của nó;
# khâu sinh audio chỉ dịch kết quả đó sang lời gọi Scaper. Hai đường không thể lệch
# nhau vì chúng là một — và nhờ vậy nghiệm thu được phân bố trong vài giây thay vì
# sau 18 giờ sinh audio.


@dataclass(frozen=True)
class SuKienQuyetDinh:
    """Một sự kiện đã chốt xong mọi thứ, trước khi đụng tới audio."""

    label: str
    nguon: ClipNguon
    start: float
    thoi_luong: float            # thời lượng CUỐI, đã nhân time_stretch
    tham_so_scaper: dict         # truyền thẳng vào Scaper.add_event


@dataclass(frozen=True)
class ClipQuyetDinh:
    index: int
    area_type: str
    nen: NenNguon
    nen_source_time: float
    su_kien: tuple[SuKienQuyetDinh, ...]


@dataclass(frozen=True)
class NguyenLieu:
    """Mọi thứ đọc từ đĩa một lần rồi dùng cho cả lượt sinh."""

    bang: dict[str, tuple[ClipNguon, ...]]
    bang_nen: dict[str, tuple[NenNguon, ...]]
    phan_bo: dict[str, tuple[float, ...]]
    lop_dai: tuple[str, ...]
    chains: list[dict]

    @property
    def lop(self) -> tuple[str, ...]:
        return tuple(sorted(self.bang))

    @classmethod
    def nap(cls, config: dict, chains: list[dict] | None = None) -> NguyenLieu:
        bang, vi_pham = loc_hop_le(doc_bang_tra(), bank_eligible_ids())
        if vi_pham:
            raise SystemExit(f"❌ {len(vi_pham)} clip trong bank KHÔNG thuộc {BANK_SPLIT}: "
                             f"{', '.join(vi_pham[:5])}…")
        phan_bo = doc_phan_bo_dich()
        thieu = kiem_phu_song(bang, phan_bo)
        if thieu:
            raise SystemExit(f"❌ lớp không có trong phân bố đích AudioSet: {', '.join(thieu)}")
        nguong = nguong_co_bien(
            float(config["forced_slices"]["long_event"]["min_event_duration_sec"]), config["events"])
        return cls(bang=bang, bang_nen=doc_bang_nen(), phan_bo=phan_bo,
                   lop_dai=lop_cap_duoc_su_kien_dai(bang, phan_bo, nguong),
                   chains=config["causal_chains"] if chains is None else chains)


def quyet_dinh_clip(rng: random.Random, plan: ClipPlan, config: dict,
                    nguyen_lieu: NguyenLieu) -> ClipQuyetDinh:
    """Chốt toàn bộ nội dung một clip: nền, nhãn, nguồn, thời lượng, thời điểm.

    Thứ tự rút RNG là một phần của hợp đồng — đổi thứ tự là đổi cả dataset dù mọi
    phân bố vẫn y nguyên. Trình tự: nền → nhãn → (thời lượng, nguồn) → thời điểm.
    """
    duration = float(config["duration_sec"])
    events = config["events"]
    forced = config["forced_slices"]
    snr_low, snr_high = events["snr_db"]
    if "low_snr" in plan.slices:
        snr_high = float(forced["low_snr"]["max_snr_db"])

    nen, nen_source_time = chon_nen(rng, nguyen_lieu.bang_nen[plan.area_type], duration)
    if plan.n_events == 0:
        return ClipQuyetDinh(plan.index, plan.area_type, nen, nen_source_time, ())

    mat_xich = (chain_by_name(nguyen_lieu.chains, plan.chain)["sequence"] if plan.chain else [])
    ep_dai = "long_event" in plan.slices
    nhan = list(mat_xich)
    for thu_tu in range(plan.n_events - len(mat_xich)):
        # Sự kiện tự do ĐẦU TIÊN gánh hợp đồng long_event, và cũng là neo của cặp
        # chồng lấn — hai lát cắt dùng chung một sự kiện, xem `so_su_kien_tu_do_can`.
        nguon_lop = nguyen_lieu.lop_dai if (ep_dai and thu_tu == 0) else nguyen_lieu.lop
        nhan.append(rng.choice(nguon_lop))

    nguong = float(forced["long_event"]["min_event_duration_sec"])
    ngan_sach = (ngan_sach_mat_xich(chain_by_name(nguyen_lieu.chains, plan.chain), duration)
                 if mat_xich else duration)
    nguon_da_chon: list[ClipNguon] = []
    tham_so_da_rut: list[dict] = []
    thoi_luong: list[float] = []
    for thu_tu, lop in enumerate(nhan):
        la_su_kien_dai = ep_dai and thu_tu == len(mat_xich)
        if la_su_kien_dai:
            d_muon = rut_thoi_luong_dai(rng, nguyen_lieu.phan_bo, lop, nguong, duration)
            ung_vien = ung_vien_su_kien_dai(nguyen_lieu.bang[lop], nguong_co_bien(nguong, events))
        else:
            # Mắt xích bị chặn theo ngân sách để cả chuỗi lọt vừa clip; sự kiện tự do
            # thì chỉ bị chặn bởi độ dài clip.
            toi_da = ngan_sach if thu_tu < len(mat_xich) else duration
            d_muon = rut_thoi_luong(rng, nguyen_lieu.phan_bo, lop, toi_da)
            ung_vien = nguyen_lieu.bang[lop]
        nguon, d_thuc = chon_nguon(rng, ung_vien, d_muon)
        # Rút tham số NGAY, trước khi đặt vị trí: thời lượng cuối là
        # `event_duration × time_stretch`, nên đặt vị trí theo `d_thuc` rồi mới rút
        # hệ số kéo sẽ làm sự kiện co giãn ±10% sau khi đã tính chỗ.
        tham_so = tham_so_su_kien(nguon, d_thuc, (snr_low, snr_high), events, rng)
        nguon_da_chon.append(nguon)
        tham_so_da_rut.append(tham_so)
        thoi_luong.append(tham_so["thoi_luong_cuoi"])

    moc_co_dinh = (chain_event_times(nguyen_lieu.chains and chain_by_name(nguyen_lieu.chains, plan.chain),
                                     rng, duration, thoi_luong[:len(mat_xich)])
                   if mat_xich else None)
    if mat_xich and not moc_co_dinh:
        # Không được lặng lẽ đặt tự do: mất thứ tự chuỗi mà clip vẫn mang nhãn
        # `causal_chain`, đúng kiểu hỏng hóc `long_event` của lô cũ. Với `ngan_sach`
        # ở trên thì nhánh này không thể xảy ra — nên nếu xảy ra, phải nổ để biết.
        raise ValueError(
            f"chuỗi `{plan.chain}` không lọt vừa clip {duration}s dù đã chặn ngân sách "
            f"{ngan_sach:.2f}s/mắt xích; thời lượng thực tế "
            f"{[round(x, 2) for x in thoi_luong[:len(mat_xich)]]}"
        )
    ty_le = float(forced["overlap"]["min_overlap_ratio"]) if "overlap" in plan.slices else None
    moc = dat_cum(rng, thoi_luong, duration, ty_le, moc_co_dinh)

    su_kien = []
    for lop, nguon, tham_so, start in zip(nhan, nguon_da_chon, tham_so_da_rut, moc, strict=True):
        day_du = {**tham_so, "event_time": ("const", round(start, 4))}
        su_kien.append(SuKienQuyetDinh(
            label=lop, nguon=nguon, start=start,
            thoi_luong=day_du.pop("thoi_luong_cuoi"), tham_so_scaper=day_du,
        ))
    return ClipQuyetDinh(plan.index, plan.area_type, nen, nen_source_time, tuple(su_kien))


# ── Sinh audio ───────────────────────────────────────────────────────────────


# Nền của khu vực nào thì phải mang tiếng vang của khu vực đó. Ghép ngẫu nhiên sẽ cho
# ra clip nền nhà xe mà vang như phòng học — model học được rằng vang và bối cảnh
# không liên quan gì nhau, trong khi ngoài đời chúng đi liền.
AREA_TO_RIR_SPACE = {
    "school": "medium_room",        # hành lang, sảnh lớp
    "parking": "large_room",        # nhà xe, vang dài
    "residential": "small_room",    # nhà ở, sân nhỏ
    "factory": "large_room",        # xưởng
}


def apply_rir(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Tích chập clip với RIR, giữ NGUYÊN độ dài và mức to.

    Hai điều bắt buộc, cả hai đều âm thầm nếu làm sai:

    1. CẮT VỀ ĐÚNG ĐỘ DÀI CŨ. Tích chập cho ra tín hiệu dài thêm bằng đuôi RIR (tới 2
       giây). Không cắt thì clip 10 giây thành 12 giây, trong khi .jams vẫn mô tả một
       clip 10 giây — mọi mốc thời gian sau đó lệch.
    2. GIỮ MỨC TO. RIR đã chuẩn hoá theo năng lượng nên về lý thuyết mức giữ nguyên,
       nhưng cộng hưởng có thể đẩy đỉnh vượt 1.0. Hạ xuống theo đỉnh, KHÔNG chuẩn hoá
       lại toàn bộ — chuẩn hoá lại sẽ phá tỉ số SNR mà Scaper vừa đặt.
    """
    # 10 s × RIR ~2 s: convolution trực tiếp mất ~30 s/clip trên máy hiện tại.
    # FFT tính cùng phép tích chập tuyến tính (~0.018 s), sai khác số học rất nhỏ;
    # giữ nguyên crop/peak limit, không thay recipe hay thứ tự rút RNG.
    from scipy.signal import fftconvolve

    wet = fftconvolve(audio, rir, mode="full")[: len(audio)]
    peak = float(np.abs(wet).max())
    if peak > 0.99:
        wet = wet * (0.99 / peak)
    return wet.astype(np.float32)


def pick_rir(rng: random.Random, area_type: str) -> Path | None:
    """Một RIR ngẫu nhiên thuộc loại phòng khớp với khu vực nền."""
    space = AREA_TO_RIR_SPACE.get(area_type)
    folder = RIR_DIR / space if space else None
    if not folder or not folder.exists():
        return None
    files = sorted(folder.glob("*.wav"))
    return rng.choice(files) if files else None



def class_dirs(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and any(p.glob("*.wav"))) if root.exists() else []


def build_scaper(config: dict, seed: int):
    import scaper

    generator = scaper.Scaper(
        duration=float(config["duration_sec"]),
        fg_path=str(FOREGROUND_DIR),
        bg_path=str(BACKGROUND_DIR),
        random_state=seed,
    )
    generator.ref_db = int(config["ref_db"])
    generator.sr = int(config["sample_rate"])
    return generator


def populate(generator, plan: ClipPlan, config: dict, rng: random.Random,
             chains: list[dict] | None = None, nguyen_lieu: NguyenLieu | None = None) -> ClipQuyetDinh:
    """Nạp nền + sự kiện của một clip vào generator, theo đúng quyết định đã chốt.

    Hàm này CỐ Ý không còn quyết định gì. Mọi lựa chọn nằm ở `quyet_dinh_clip`, còn
    đây chỉ dịch kết quả sang lời gọi Scaper. Nhờ vậy bộ mô phỏng `--simulate-labels`
    và lượt sinh audio thật không thể lệch nhau: cả hai đọc cùng một nguồn quyết định.

    Trả về `ClipQuyetDinh` để nơi gọi ghi được vào chỉ mục mà không phải tính lại.
    """
    nguyen_lieu = NguyenLieu.nap(config, chains) if nguyen_lieu is None else nguyen_lieu
    quyet_dinh = quyet_dinh_clip(rng, plan, config, nguyen_lieu)

    # Chỉ đích danh khu vực và đích danh file nền, thay vì để Scaper bốc: kế hoạch đã
    # chọn khu vực rồi, RIR sắp tích chập phải khớp chính khu vực đó, và `source_time`
    # chỉ đặt đúng được khi ta biết nền nào đang dùng (STATUS §7 B2.5).
    generator.add_background(
        label=("const", quyet_dinh.area_type),
        source_file=("const", str(quyet_dinh.nen.duong_dan_day_du)),
        source_time=("const", quyet_dinh.nen_source_time),
    )
    for su_kien in quyet_dinh.su_kien:
        generator.add_event(**su_kien.tham_so_scaper)
    return quyet_dinh


def seed_clip(seed: int, index: int) -> int:
    """Seed RIÊNG cho từng clip — nền của cả bước B8.

    Chừng nào mọi clip còn rút chung một dòng RNG thì clip thứ i phụ thuộc mọi clip
    trước nó, và không song song hoá được. Seed theo clip gỡ nút đó, đồng thời làm
    resume trở nên CHÍNH XÁC thay vì phải replay: cơ chế cũ buộc gọi lại `populate()`
    cho từng clip đã xong chỉ để đẩy dòng RNG tới đúng chỗ.

    Băm chứ không cộng, vì hai lý do:

      · `seed + index` làm dev (seed_offset 1.000.000) clip 0 trùng hệt train clip
        1.000.000 — hai split lẽ ra phải độc lập.
      · `hash()` của Python ngẫu nhiên hoá theo PYTHONHASHSEED, nên dùng nó sẽ cho
        một dataset KHÁC mỗi lần khởi động mà mọi phân bố vẫn y nguyên — không có
        triệu chứng nào.
    """
    import hashlib

    return int.from_bytes(
        hashlib.blake2b(f"{seed}:{index}".encode(), digest_size=8).digest(), "big")


def so_worker_thuc_te(yeu_cau: int | None, so_clip: int) -> int:
    """Số tiến trình thật sự nên mở.

    Mỗi worker nạp lại toàn bộ bank và phân bố đích, nên mở nhiều worker hơn số clip
    là trả phí khởi tạo cho những tiến trình không có việc.
    """
    import os

    toi_da = os.cpu_count() or 1
    muon = toi_da if yeu_cau is None else yeu_cau
    return max(1, min(muon, toi_da, so_clip if so_clip > 0 else 1))


def sinh_mot_clip(plan: ClipPlan, split: str, seed: int, config: dict,
                  chains: list[dict] | None, nguyen_lieu: NguyenLieu,
                  out: Path, reverb_ok: bool) -> tuple[int, str, bool]:
    """Sinh trọn một clip. (chỉ số, tên RIR đã dùng, có thật sự sinh không).

    Không đụng tới trạng thái nào ngoài file của chính clip này, nên gọi được từ một
    tiến trình riêng. `rng` dựng từ `seed_clip`, độc lập hoàn toàn với các clip khác.
    """
    import soundfile

    name = f"{split}_{plan.index:06d}"
    audio_path = out / "audio" / f"{name}.wav"
    jams_path = out / "jams" / f"{name}.jams"

    # Quyết định nội dung TRƯỚC khi kiểm clip đã có. Tốn vài phần nghìn giây mỗi clip
    # và không chạm tới audio, nhưng `rir_used` chỉ suy ra được từ đây — trả về rỗng
    # cho clip bỏ qua sẽ làm `slice_index` khai "gắn nhãn reverb nhưng rir_used rỗng"
    # cho toàn bộ phần đã sinh từ lần chạy trước. Đo được 2.663/7.920 clip như vậy.
    hat_giong = seed_clip(seed, plan.index)
    rng = random.Random(hat_giong)
    # NumPy chỉ nhận seed 32 bit, còn `seed_clip` cố ý trả 64 bit để 9.360 clip không
    # trùng seed nhau (32 bit cho ~1% xác suất có ít nhất một cặp trùng). Thu gọn
    # riêng cho Scaper là an toàn: sau B2 mọi tham số truyền vào đều là `const`, nên
    # dòng RNG của Scaper gần như không được dùng tới.
    da_co = audio_path.exists() and jams_path.exists()
    # Với clip đã có thì bỏ hẳn phần đắt: khởi tạo Scaper và tổng hợp audio.
    generator = _BoQua() if da_co else build_scaper(config, hat_giong % 2**32)
    populate(generator, plan, config, rng, chains, nguyen_lieu)

    rir_path = pick_rir(rng, plan.area_type) if "reverb" in plan.slices and reverb_ok else None
    rir_used = rir_path.stem if rir_path else ""
    if da_co:
        # Clip đã sinh xong ở lần chạy trước bị ngắt (crash, mất điện, TaskStop) — đã
        # xảy ra thật 17/09/2026. Với seed theo clip thì không clip nào phụ thuộc nó.
        return plan.index, rir_used, False

    # Chỉ công bố cặp file cuối sau khi hoàn tất cả RIR. Nếu bị ngắt,
    # file .pending sẽ được ghi lại ở lần resume, không bị coi là clip xong.
    pending_audio = audio_path.with_suffix(".pending.wav")
    pending_jams = jams_path.with_suffix(".pending.jams")
    generator.generate(
        audio_path=str(pending_audio),
        jams_path=str(pending_jams),
        # CỐ Ý để None: `reverb` của Scaper là hiệu ứng vang tổng hợp của SoX, KHÔNG
        # phải tích chập RIR. DATA_PLAN §7 yêu cầu tích chập, và nó khác hẳn — RIR
        # đo tại phòng thật mang cả hình dạng phản xạ sớm lẫn cách phòng hút tần số
        # cao, thứ mà hiệu ứng tổng hợp không tái tạo được. Tích chập làm ở dưới.
        reverb=None,
        # BẮT BUỘC. Nền ở -23 LUFS cộng SNR tối đa +25 dB cho sự kiện ở +2 dBFS,
        # tức méo cứng. Đo thật trên 5 clip đầu: 3/5 clip méo, tới 4016 mẫu chạm
        # trần. Sinh ra train set mà chính cổng chất lượng của ta sẽ loại (>0.1%
        # mẫu cắt phẳng → `clipped`) là tự mâu thuẫn, và méo là thứ model học
        # được rất nhanh: nó sẽ dùng méo làm dấu hiệu nhận biết sự kiện to.
        # fix_clipping hạ đều cả soundscape nên GIỮ NGUYÊN tỉ số SNR đã đặt.
        fix_clipping=True,
    )

    if rir_path:
        audio, rate = soundfile.read(str(pending_audio))
        rir, _ = soundfile.read(str(rir_path))
        soundfile.write(pending_audio, apply_rir(audio, rir), rate, subtype="PCM_16")

    # Scaper ghi đường dẫn lúc generate vào recipe; trỏ về tên cuối để
    # công cụ đọc/tái dựng JAMS không tìm file .pending đã được chuyển đi.
    jam = json.loads(pending_jams.read_text(encoding="utf-8"))
    for annotation in jam.get("annotations", []):
        recipe = annotation.get("sandbox", {}).get("scaper")
        if recipe is not None:
            recipe.update(soundscape_audio_path=str(audio_path),
                          audio_path=str(audio_path), jams_path=str(jams_path))
    pending_jams.write_text(json.dumps(jam, ensure_ascii=False, indent=2), encoding="utf-8")
    pending_audio.replace(audio_path)
    pending_jams.replace(jams_path)
    return plan.index, rir_used, True


class _BoQua:
    """Generator giả cho clip đã sinh xong: nhận lời gọi của `populate` rồi bỏ đi.

    `populate` vẫn phải chạy để RNG tới đúng vị trí mà `pick_rir` cần, nhưng không
    có lý do gì khởi tạo Scaper và tổng hợp lại audio đã có.
    """

    def add_background(self, **kwargs) -> None:
        pass

    def add_event(self, **kwargs) -> None:
        pass


# ── Chạy song song (STATUS §7 B8) ────────────────────────────────────────────
#
# Đo thật trên lô 17/09: 8,4 clip/phút → 9.360 clip mất 18,6 giờ. Khâu này thuần CPU
# và mỗi clip độc lập sau khi có `seed_clip`, nên trải ra 16 nhân là việc hiển nhiên.
#
# Mỗi worker nạp bank + phân bố đích MỘT LẦN trong initializer. Truyền `NguyenLieu`
# qua pickle cho từng clip sẽ gửi lại 4.267 bản ghi nguồn mỗi lần — đắt hơn cả việc
# sinh audio.

_WORKER: dict = {}


def _khoi_tao_worker(config: dict, chains: list[dict] | None, out: str, reverb_ok: bool) -> None:
    _WORKER.update(config=config, chains=chains, out=Path(out), reverb_ok=reverb_ok,
                   nguyen_lieu=NguyenLieu.nap(config, chains))


def _sinh_worker(viec: tuple[ClipPlan, str, int]) -> tuple[int, str, bool]:
    plan, split, seed = viec
    return sinh_mot_clip(plan, split, seed, _WORKER["config"], _WORKER["chains"],
                         _WORKER["nguyen_lieu"], _WORKER["out"], _WORKER["reverb_ok"])


def generate_split(config: dict, split: str, plans: list[ClipPlan], seed: int,
                   chains: list[dict] | None = None,
                   nguyen_lieu: NguyenLieu | None = None,
                   workers: int | None = 1) -> int:
    """Sinh cả split. `workers=1` chạy tuần tự; lớn hơn thì trải ra nhiều tiến trình.

    Kết quả KHÔNG phụ thuộc số worker: mỗi clip có seed riêng (`seed_clip`) và chỉ
    ghi file của chính nó, nên chạy 1 hay 12 tiến trình đều ra cùng một dataset.
    """
    out = OUTPUT_DIR / split
    for sub in ("audio", "jams", "tsv"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    reverb_ok = bool(class_dirs(RIR_DIR))
    if not reverb_ok:
        print("  ⚠️ chưa có bank RIR — bỏ lát cắt `reverb`, sẽ phải sinh lại khi có")

    theo_index = {plan.index: plan for plan in plans}
    n = so_worker_thuc_te(workers, len(plans))
    xong = 0

    if n == 1:
        if nguyen_lieu is None:
            nguyen_lieu = NguyenLieu.nap(config, chains)
        ket_qua = (sinh_mot_clip(plan, split, seed, config, chains, nguyen_lieu, out, reverb_ok)
                   for plan in plans)
    else:
        import multiprocessing as mp

        print(f"  ▸ {n} tiến trình song song")
        pool = mp.Pool(n, initializer=_khoi_tao_worker,
                       initargs=(config, chains, str(out), reverb_ok))
        viec = [(plan, split, seed) for plan in plans]
        ket_qua = pool.imap_unordered(_sinh_worker, viec, chunksize=8)

    try:
        for index, rir_used, _ in ket_qua:
            theo_index[index].rir_used = rir_used
            xong += 1
            if xong % 100 == 0:
                print(f"  {xong}/{len(plans)}…", flush=True)
    finally:
        if n > 1:
            pool.close()
            pool.join()
    return xong


def write_plan_index(split: str, plans: list[ClipPlan]) -> None:
    """Ghi kế hoạch ra JSONL để biết clip nào thuộc lát cắt nào khi báo cáo kết quả."""
    path = OUTPUT_DIR / split / "slice_index.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for plan in plans:
            handle.write(json.dumps({
                "clip_id": f"{split}_{plan.index:06d}",
                "slices": sorted(plan.slices),
                "n_events": plan.n_events,
                "chain": plan.chain,
                "area_type": plan.area_type,
                "rir_used": plan.rir_used,
            }, ensure_ascii=False) + "\n")


# ── Mô phỏng nhãn (STATUS §7 B7) ─────────────────────────────────────────────


def mo_phong_nhan(config: dict, plans: list[ClipPlan], seed: int,
                  chains: list[dict] | None = None,
                  nguyen_lieu: NguyenLieu | None = None) -> list[ClipQuyetDinh]:
    """Chốt nội dung mọi clip mà KHÔNG đụng tới audio.

    Cùng `quyet_dinh_clip`, cùng seed, cùng thứ tự rút RNG như lượt sinh thật — nên
    phân bố nhãn ở đây chính là phân bố sẽ sinh ra, không phải một ước lượng.

    Chạy vài giây thay vì 18 giờ. Đó là khác biệt giữa "phát hiện hợp đồng hỏng
    trước khi sinh" và "phát hiện sau hai vòng train", vốn là chuyện đã xảy ra với
    `long_event`.
    """
    if nguyen_lieu is None:
        nguyen_lieu = NguyenLieu.nap(config, chains)
    # Phải dùng ĐÚNG `seed_clip` như `sinh_mot_clip`. Dùng một dòng RNG chung ở đây
    # thì bảng nghiệm thu mô tả một dataset khác dataset sắp sinh ra — và vì mọi phân
    # bố vẫn hợp lý, không ai nhận ra. Đó đúng là kiểu hỏng hóc B7 sinh ra để chặn.
    return [quyet_dinh_clip(random.Random(seed_clip(seed, plan.index)), plan, config, nguyen_lieu)
            for plan in plans]


def bao_cao_mo_phong(quyet_dinh: list[ClipQuyetDinh], plans: list[ClipPlan], config: dict) -> str:
    """Đối chiếu phân bố mô phỏng với hợp đồng trong config — bảng nghiệm thu B7."""
    forced = config["forced_slices"]
    duration = float(config["duration_sec"])
    ty_le = float(forced["overlap"]["min_overlap_ratio"])
    nguong = float(forced["long_event"]["min_event_duration_sec"])

    chuoi_theo_ten = {c["name"]: c for c in config["causal_chains"]}
    tat_ca = [sk for qd in quyet_dinh for sk in qd.su_kien]
    theo_index = {qd.index: qd for qd in quyet_dinh}
    phu_song, tran_mep = [], 0
    ov_ke = ov_dat = le_ke = le_dat = cc_ke = cc_dat = 0

    for plan in plans:
        sk = theo_index[plan.index].su_kien
        tran_mep += sum(1 for x in sk if x.start + x.thoi_luong > duration + 1e-6)
        khoang = [(x.start, x.start + x.thoi_luong) for x in sk]
        phu_song.append(_hop_khoang(khoang) / duration)
        if "overlap" in plan.slices:
            ov_ke += 1
            ov_dat += any(
                do_chong_lan(sk[i].start, sk[i].thoi_luong, sk[j].start, sk[j].thoi_luong)
                >= ty_le * min(sk[i].thoi_luong, sk[j].thoi_luong) - 1e-9
                for i in range(len(sk)) for j in range(i + 1, len(sk)))
        if "long_event" in plan.slices:
            le_ke += 1
            le_dat += any(x.thoi_luong >= nguong - 1e-9 for x in sk)
        if plan.chain:
            # Thứ tự mới là thứ mang nghĩa: glass_breaking → scream → running_footsteps
            # là đột nhập, đảo lại thì không. Lô đầu 17/09 có 91 clip mang nhãn chuỗi
            # mà mắt xích đảo thứ tự, và chỉ verify_synthetic (sau 47 phút sinh) mới
            # thấy. Kiểm ngay ở đây thì mất 3 giây.
            mat_xich = chuoi_theo_ten[plan.chain]["sequence"]
            dau = sk[:len(mat_xich)]
            cc_ke += 1
            cc_dat += ([x.label for x in dau] == mat_xich
                       and [x.start for x in dau] == sorted(x.start for x in dau))

    dai = [x.thoi_luong for x in tat_ca]
    ti_le = lambda m: 100 * sum(1 for x in dai if x >= m) / len(dai)  # noqa: E731
    dong = [
        "", "── mô phỏng nhãn (chưa sinh audio) " + "─" * 38,
        f"{len(tat_ca)} sự kiện / {len(plans)} clip = {len(tat_ca) / len(plans):.2f} mỗi clip",
        "",
        f"{'hợp đồng':<38}{'clip':>8}{'đạt':>8}{'tỉ lệ':>9}",
        "─" * 63,
        f"{f'overlap ≥ {ty_le:.2f}':<38}{ov_ke:>8}{ov_dat:>8}"
        f"{100 * ov_dat / max(ov_ke, 1):>8.1f}%",
        f"{f'long_event ≥ {nguong:g}s':<38}{le_ke:>8}{le_dat:>8}"
        f"{100 * le_dat / max(le_ke, 1):>8.1f}%",
        f"{'causal_chain đúng thứ tự':<38}{cc_ke:>8}{cc_dat:>8}"
        f"{100 * cc_dat / max(cc_ke, 1):>8.1f}%",
        "",
        f"{'sự kiện tràn khỏi clip':<38}{tran_mep:>8}   (phải là 0)",
        f"{'độ phủ sóng':<38}{100 * sum(phu_song) / len(phu_song):>7.1f}%",
        f"{'% sự kiện ≥ 2s / ≥ 4s / ≥ 8s':<38}"
        f"{ti_le(2.0):>6.1f}% {ti_le(4.0):>5.1f}% {ti_le(8.0):>5.1f}%",
    ]
    hong = []
    if ov_ke and ov_dat < ov_ke:
        hong.append(f"overlap: {ov_ke - ov_dat} clip mang nhãn mà không đạt ngưỡng")
    if le_ke and le_dat < le_ke:
        hong.append(f"long_event: {le_ke - le_dat} clip mang nhãn mà không đạt ngưỡng")
    if cc_ke and cc_dat < cc_ke:
        hong.append(f"causal_chain: {cc_ke - cc_dat} clip có mắt xích sai thứ tự")
    if tran_mep:
        hong.append(f"{tran_mep} sự kiện tràn khỏi clip")
    if hong:
        dong += ["", "❌ HỢP ĐỒNG KHÔNG ĐẠT:"] + [f"   · {x}" for x in hong]
    else:
        dong += ["", "✓ mọi hợp đồng đạt 100% — sinh audio được"]
    return "\n".join(dong)


def _hop_khoang(khoang: list[tuple[float, float]]) -> float:
    """Tổng độ dài phần HỢP. Phải lấy khoảng đầu từ bản ĐÃ SẮP XẾP."""
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


# ── Chạy ─────────────────────────────────────────────────────────────────────


def main() -> int:
    enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "dev"], required=True)
    parser.add_argument("--limit", type=int, help="chỉ sinh N clip đầu (thử nhanh)")
    parser.add_argument("--plan-only", action="store_true", help="chỉ lập kế hoạch + kiểm tra, không sinh audio")
    parser.add_argument("--workers", type=int, default=1,
                        help="số tiến trình sinh song song; 0 = dùng hết số nhân (STATUS §7 B8). "
                             "Kết quả KHÔNG đổi theo số worker — mỗi clip có seed riêng.")
    parser.add_argument("--simulate-labels", action="store_true",
                        help="chốt nhãn của mọi clip và đối chiếu hợp đồng, KHÔNG sinh audio (STATUS §7 B7)")
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    split_config = config["splits"][args.split]
    seed = int(config["seed"]) + int(split_config.get("seed_offset", 0))

    n_clips = int(float(split_config["target_hours"]) * 3600 / float(config["duration_sec"]))
    if args.limit:
        n_clips = min(n_clips, args.limit)

    co_san = available_labels()
    chains, bo_chuoi = usable_chains(config["causal_chains"], co_san)
    for name, thieu in bo_chuoi:
        print(f"⚠️ bỏ chuỗi `{name}` — bank chưa có lớp {thieu}")
    if bo_chuoi:
        print("   Sinh chuỗi cụt mà vẫn mang nhãn của chuỗi đủ sẽ dạy model một định")
        print("   nghĩa sai về kịch bản đó, nên bỏ hẳn và ghi lại ở đây.")
        print()

    areas = class_dirs(BACKGROUND_DIR)
    plans = plan_clips(config, n_clips, seed, chains, areas)
    print(f"▶ {args.split}: {n_clips} clip × {config['duration_sec']}s "
          f"= {n_clips * float(config['duration_sec']) / 3600:.1f} h · seed {seed}\n")
    print(slice_report(plans, config))

    foreground = sorted(FOREGROUND_DIR.rglob("*.wav"))
    if not foreground:
        print(f"\n❌ bank foreground rỗng ({FOREGROUND_DIR.relative_to(REPO_ROOT)}).")
        print("   Cần chạy auto_screen.py + người duyệt trước — DATA_PLAN §4.2–4.3.")
        return 1

    violations = check_bank_clean(foreground, bank_eligible_ids())
    if violations:
        print(f"\n❌ {len(violations)} file trong bank KHÔNG thuộc {BANK_SPLIT}:")
        for name in violations[:5]:
            print(f"      {name}")
        print("   Đây là rò rỉ vào train set. Sửa bank hoặc chạy lại make_splits.py.")
        return 1
    print(f"\n✓ {len(foreground)} clip bank đều thuộc {BANK_SPLIT}")

    if not class_dirs(BACKGROUND_DIR):
        print(f"❌ bank background rỗng ({BACKGROUND_DIR.relative_to(REPO_ROOT)}) — DATA_PLAN §5.")
        return 1

    # Nghiệm thu phân bố TRƯỚC khi sinh audio. Trả mã lỗi khi hợp đồng không đạt, để
    # cắm được vào tiền kiểm của một lượt chạy dài mà không cần ai đọc bảng bằng mắt.
    if args.simulate_labels:
        bao_cao = bao_cao_mo_phong(mo_phong_nhan(config, plans, seed, chains), plans, config)
        print(bao_cao)
        return 1 if "❌" in bao_cao else 0

    if args.plan_only:
        print("\n(--plan-only: chưa sinh audio)")
        return 0

    written = generate_split(config, args.split, plans, seed, chains,
                             workers=(None if args.workers == 0 else args.workers))
    # Ghi SAU khi sinh: `rir_used` chỉ có giá trị sau bước tích chập. Ghi trước thì
    # cột đó luôn rỗng và không truy ngược được clip nào dùng RIR nào khi phân tích lỗi.
    write_plan_index(args.split, plans)
    print(f"\n✓ đã sinh {written} clip → {(OUTPUT_DIR / args.split).relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
