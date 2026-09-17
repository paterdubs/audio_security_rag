"""Kiểm tra bộ synthetic có ĐÚNG như nó tự khai không — TRAINING_OPS_PLAN.md Pha 2.

    .venv/Scripts/python.exe scripts/verify_synthetic.py --split train
    .venv/Scripts/python.exe scripts/verify_synthetic.py --split train --limit 500

Vì sao script này tồn tại: ngày 17/09/2026 tìm ra bốn lỗi IM LẶNG trong cùng một buổi,
không lỗi nào làm chương trình dừng, cả bốn chỉ làm số liệu sai. Nặng nhất:

    forced_slices.long_event khai "sự kiện dài ≥ 8 giây" cho 10% clip,
    trong khi event_duration=("const", 1.0) làm độ dài tối đa thực tế là 1.1 giây.

Một lát cắt đánh giá nguyên vẹn là nhãn rỗng, và không có gì báo. Mỗi hàm `kiem_*` dưới
đây đối chiếu một điều khoản trong scaper_train.yaml với audio/nhãn THẬT đã sinh ra.

Mã thoát: 0 = mọi điều khoản đúng · 1 = có điều khoản bị vi phạm · 2 = không chạy được.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scaper_generate as sg  # noqa: E402
from common import REPO_ROOT, enable_utf8_output  # noqa: E402

enable_utf8_output()

CONFIG_PATH = REPO_ROOT / "ml" / "configs" / "scaper_train.yaml"
ONTOLOGY_PATH = REPO_ROOT / "ml" / "configs" / "ontology_map.yaml"
SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"
BANK_DIR = REPO_ROOT / "data" / "banks" / "foreground"
REPORT_PATH = REPO_ROOT / "data" / "manifests" / "synthetic_contract.csv"
STAGE = "verify_synthetic"

# Sai lệch cho phép khi so tỉ lệ slice thực tế với tỉ lệ khai trong config. Rút Bernoulli
# độc lập nên tỉ lệ thực dao động quanh giá trị khai; ±5 điểm phần trăm là dung sai lấy
# mẫu bình thường, KHÔNG phải lỗi.
TY_LE_DUNG_SAI = 0.05

# Độ dài sự kiện trong clip phải phản ánh được độ dài trong bank nguồn. Nếu phân vị 90
# của MỌI lớp đều kẹp dưới ngưỡng này trong khi bank có clip dài hơn nhiều, gần như chắc
# chắn có một hằng số đang cắt cụt tất cả.
NGUONG_NGHI_NGO_CAT_CUT = 1.5


# ── Đọc dữ liệu ──────────────────────────────────────────────────────────────


def doc_slice_index(split_dir: Path, config: dict, split: str, n_clips: int) -> dict[str, dict]:
    """Lấy lát cắt của từng clip. Dựng lại từ plan_clips nếu file chưa có.

    `slice_index.jsonl` chỉ được ghi ở CUỐI scaper_generate.main(), nên nó biến mất khi
    job bị ngắt giữa chừng (đã xảy ra thật). Dựng lại được vì plan tất định theo seed —
    đã kiểm chứng: kế hoạch cho 1500 clip đầu giống hệt nhau dù định sinh 1500 hay 7920.
    """
    path = split_dir / "slice_index.jsonl"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            return {row["clip_id"]: row for row in map(json.loads, handle) if row}

    print(f"⚠️ chưa có {path.name} — dựng lại từ plan_clips (tất định theo seed)")
    split_config = config["splits"][split]
    seed = int(config["seed"]) + int(split_config.get("seed_offset", 0))
    chains, _ = sg.usable_chains(config["causal_chains"], sg.available_labels())
    areas = sg.class_dirs(sg.BACKGROUND_DIR)
    plans = sg.plan_clips(config, n_clips, seed, chains, areas)
    return {
        f"{split}_{p.index:06d}": {
            "clip_id": f"{split}_{p.index:06d}", "slices": sorted(p.slices),
            "n_events": p.n_events, "chain": p.chain, "area_type": p.area_type,
            # rir_used chỉ biết được sau khi sinh audio; dựng lại thì không có.
            "rir_used": None,
        }
        for p in plans
    }


def doc_su_kien(jams_path: Path) -> list[dict]:
    """Sự kiện foreground của một clip: nhãn, onset, offset, snr."""
    data = json.loads(jams_path.read_text(encoding="utf-8"))
    events = []
    for obs in data["annotations"][0]["data"]:
        value = obs["value"]
        if not isinstance(value, dict) or value.get("role") != "foreground":
            continue
        events.append({
            "label": value["label"],
            "onset": float(obs["time"]),
            "offset": float(obs["time"]) + float(obs["duration"]),
            "snr": float(value.get("snr", 0.0)),
            "source_file": value.get("source_file", ""),
        })
    return sorted(events, key=lambda e: e["onset"])


# ── Từng điều khoản ──────────────────────────────────────────────────────────


def kiem_long_event(events, spec) -> str | None:
    nguong = float(spec["min_event_duration_sec"])
    dai_nhat = max((e["offset"] - e["onset"] for e in events), default=0.0)
    if dai_nhat < nguong:
        return f"dài nhất {dai_nhat:.2f}s < {nguong}s"
    return None


def kiem_low_snr(events, spec) -> str | None:
    nguong = float(spec["max_snr_db"])
    if not events:
        return "không có sự kiện nào để xét SNR"
    to_nhat = max(e["snr"] for e in events)
    if to_nhat > nguong:
        return f"SNR lớn nhất {to_nhat:.1f}dB > {nguong}dB"
    return None


def kiem_overlap(events, spec) -> str | None:
    min_events = int(spec["min_events"])
    ty_le_toi_thieu = float(spec["min_overlap_ratio"])
    if len(events) < min_events:
        return f"chỉ có {len(events)} sự kiện, cần ≥{min_events}"
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            a, b = events[i], events[j]
            chong = min(a["offset"], b["offset"]) - max(a["onset"], b["onset"])
            if chong <= 0:
                continue
            ngan_hon = min(a["offset"] - a["onset"], b["offset"] - b["onset"])
            if ngan_hon > 0 and chong / ngan_hon >= ty_le_toi_thieu:
                return None
    return f"không cặp nào chồng lấn ≥{ty_le_toi_thieu:.0%}"


def kiem_reverb(row) -> str | None:
    # rir_used=None nghĩa là slice_index được DỰNG LẠI chứ không đọc từ file — không kết
    # luận được, và báo lỗi ở đây sẽ là dương tính giả.
    if row.get("rir_used") is None:
        return None
    if not row["rir_used"]:
        return "gắn nhãn reverb nhưng rir_used rỗng"
    return None


def kiem_causal_chain(events, row, chains_by_name) -> str | None:
    ten = row.get("chain")
    if not ten:
        return "gắn nhãn causal_chain nhưng không ghi tên chuỗi"
    chain = chains_by_name.get(ten)
    if chain is None:
        return f"chuỗi {ten!r} không có trong config"
    can = list(chain["sequence"])
    # Phải xuất hiện ĐÚNG THỨ TỰ thời gian. Chuỗi đúng nhãn nhưng sai thứ tự dạy model
    # một định nghĩa sai về kịch bản — `break_in` mà tiếng hét đi trước tiếng kính vỡ thì
    # không còn là đột nhập.
    con_lai = list(can)
    for e in events:
        if con_lai and e["label"] == con_lai[0]:
            con_lai.pop(0)
    if con_lai:
        thay = [e["label"] for e in events]
        return f"chuỗi {ten}: cần {can} theo thứ tự, thực tế {thay}"
    return None


# ── Kiểm tra toàn cục ────────────────────────────────────────────────────────


def kiem_tran_mep(events, duration: float) -> str | None:
    """Không sự kiện nào được kéo dài quá mốc cuối clip — STATUS §7 B4.

    Điều khoản này mới có từ B4. Lô 17/09 đặt `event_time=("uniform", 0, 10)` nên
    1.225 sự kiện (7.4%) bị Scaper cắt ở mốc 10 giây, sinh ra một đống mẩu cực ngắn
    không có trong miền đích. Nhãn vẫn ĐÚNG (Scaper ghi độ dài đã cắt), nên không có
    triệu chứng nào ngoài việc phân bố thời lượng lệch đi — đúng loại lỗi cần cổng
    kiểm tự động, vì mắt người đọc nhãn sẽ không thấy gì sai.
    """
    tran = [e for e in events if e["offset"] > duration + 1e-6]
    if not tran:
        return None
    te_nhat = max(tran, key=lambda e: e["offset"])
    return (f"{len(tran)} sự kiện vượt mốc {duration:g}s "
            f"(tệ nhất: {te_nhat['label']} kết thúc ở {te_nhat['offset']:.2f}s)")


def kiem_dem_file(split_dir: Path, slice_index: dict) -> list[tuple[str, str]]:
    n_audio = len(list((split_dir / "audio").glob("*.wav")))
    n_jams = len(list((split_dir / "jams").glob("*.jams")))
    loi = []
    if n_audio != n_jams:
        loi.append(("count_mismatch", f"audio={n_audio} nhưng jams={n_jams}"))
    if (split_dir / "slice_index.jsonl").exists() and len(slice_index) != n_jams:
        loi.append(("count_mismatch", f"slice_index={len(slice_index)} nhưng jams={n_jams}"))
    return loi


def kiem_do_dai_so_voi_bank(do_dai_theo_lop: dict[str, list[float]]) -> list[tuple[str, str]]:
    """Độ dài sự kiện trong clip phải phản ánh được độ dài của bank nguồn.

    Đây là điều khoản bắt được lỗi `event_duration=("const", 1.0)`: bank có clip siren
    trung vị 4.0s và vehicle_crash tới 30s, nhưng MỌI lớp trong synthetic đều kẹp ở 1.1s.
    """
    loi = []
    tat_ca = [d for v in do_dai_theo_lop.values() for d in v]
    if not tat_ca:
        return [("no_events", "không có sự kiện nào trong bộ dữ liệu")]

    p90_tong = float(np.percentile(tat_ca, 90))
    if p90_tong >= NGUONG_NGHI_NGO_CAT_CUT:
        return loi      # có sự kiện dài — không nghi ngờ cắt cụt

    # Mọi sự kiện đều ngắn. Bank có dài hơn không?
    bank_dai = {}
    for folder in sorted(BANK_DIR.iterdir()):
        if not folder.is_dir():
            continue
        files = sorted(folder.glob("*.wav"))[:60]
        if not files:
            continue
        dur = [soundfile.info(str(f)).duration for f in files]
        bank_dai[folder.name] = float(np.median(dur))

    dai_trong_bank = {k: v for k, v in bank_dai.items() if v > NGUONG_NGHI_NGO_CAT_CUT * 2}
    if dai_trong_bank:
        ten = ", ".join(f"{k} ({v:.1f}s)" for k, v in sorted(dai_trong_bank.items())[:4])
        loi.append((
            "duration_truncated",
            f"p90 độ dài sự kiện toàn bộ = {p90_tong:.2f}s, nhưng bank có lớp dài hơn "
            f"nhiều: {ten}. Kiểm tra event_duration trong scaper_generate.py"
        ))
    return loi


def kiem_nhan_thuoc_ontology(nhan_da_gap: set[str]) -> list[tuple[str, str]]:
    config = yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    hop_le = set(config["classes"])
    la = sorted(nhan_da_gap - hop_le)
    if la:
        return [("unknown_label", f"nhãn không có trong ontology: {la}")]
    return []


def kiem_ty_le_slice(config: dict, dem_slice: dict[str, int], tong: int) -> list[tuple[str, str]]:
    loi = []
    for ten, spec in config["forced_slices"].items():
        khai = float(spec["ratio"])
        thuc = dem_slice.get(ten, 0) / tong if tong else 0.0
        if abs(thuc - khai) > TY_LE_DUNG_SAI:
            loi.append((
                "slice_ratio", f"slice {ten}: khai {khai:.0%}, thực tế {thuc:.0%}"
            ))
    return loi


# ── Điều phối ────────────────────────────────────────────────────────────────


def run(split: str, limit: int | None) -> int:
    split_dir = SYNTHETIC_DIR / split
    if not (split_dir / "jams").exists():
        print(f"❌ không thấy {split_dir / 'jams'}")
        return 2

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    chains_by_name = {c["name"]: c for c in config["causal_chains"]}
    forced = config["forced_slices"]

    jams_files = sorted((split_dir / "jams").glob("*.jams"))
    if limit:
        jams_files = jams_files[:limit]
    n_clips_config = int(float(config["splits"][split]["target_hours"]) * 3600
                         / float(config["duration_sec"]))
    slice_index = doc_slice_index(split_dir, config, split, n_clips_config)

    print(f"▶ kiểm tra {len(jams_files)} clip của split {split!r}\n")

    vi_pham: list[dict] = []
    dem_slice: dict[str, int] = defaultdict(int)
    do_dai_theo_lop: dict[str, list[float]] = defaultdict(list)
    nhan_da_gap: set[str] = set()

    for jams_path in jams_files:
        clip_id = jams_path.stem
        row = slice_index.get(clip_id)
        if row is None:
            vi_pham.append({"file_id": clip_id, "reason_code": "no_slice_info",
                            "detail": "không có dòng trong slice_index"})
            continue

        events = doc_su_kien(jams_path)
        for e in events:
            nhan_da_gap.add(e["label"])
            do_dai_theo_lop[e["label"]].append(e["offset"] - e["onset"])

        slices = set(row.get("slices") or [])
        for ten in slices:
            dem_slice[ten] += 1

        # Điều khoản áp cho MỌI clip, không riêng lát cắt nào.
        if (chi_tiet := kiem_tran_mep(events, float(config["duration_sec"]))) is not None:
            vi_pham.append({"file_id": clip_id, "reason_code": "event_overflow",
                            "detail": chi_tiet})

        kiem_tra = {
            "long_event": lambda: kiem_long_event(events, forced["long_event"]),
            "low_snr": lambda: kiem_low_snr(events, forced["low_snr"]),
            "overlap": lambda: kiem_overlap(events, forced["overlap"]),
            "reverb": lambda: kiem_reverb(row),
            "causal_chain": lambda: kiem_causal_chain(events, row, chains_by_name),
        }
        for ten, ham in kiem_tra.items():
            if ten in slices and (chi_tiet := ham()) is not None:
                vi_pham.append({"file_id": clip_id, "reason_code": f"slice_{ten}",
                                "detail": chi_tiet})

    toan_cuc = (kiem_dem_file(split_dir, slice_index)
                + kiem_do_dai_so_voi_bank(do_dai_theo_lop)
                + kiem_nhan_thuoc_ontology(nhan_da_gap)
                + kiem_ty_le_slice(config, dem_slice, len(jams_files)))
    for ma, chi_tiet in toan_cuc:
        vi_pham.append({"file_id": f"<{split}>", "reason_code": ma, "detail": chi_tiet})

    bao_cao(split, len(jams_files), dem_slice, do_dai_theo_lop, vi_pham)
    ghi_csv(vi_pham)
    return 1 if vi_pham else 0


def bao_cao(split, n_clips, dem_slice, do_dai_theo_lop, vi_pham) -> None:
    print(f"{'lát cắt':<16}{'số clip':>9}{'tỉ lệ':>9}")
    print("-" * 34)
    for ten, n in sorted(dem_slice.items()):
        print(f"{ten:<16}{n:>9}{n / n_clips:>8.0%}")

    print(f"\n{'lớp':<22}{'n':>6}{'p50':>8}{'p90':>8}{'max':>8}")
    print("-" * 52)
    for lop in sorted(do_dai_theo_lop):
        v = np.array(do_dai_theo_lop[lop])
        print(f"{lop:<22}{len(v):>6}{np.percentile(v, 50):>8.2f}"
              f"{np.percentile(v, 90):>8.2f}{v.max():>8.2f}")

    theo_ma: dict[str, int] = defaultdict(int)
    for v in vi_pham:
        theo_ma[v["reason_code"]] += 1

    print()
    if not vi_pham:
        print(f"✅ HỢP ĐỒNG ĐẠT — {n_clips} clip, không điều khoản nào bị vi phạm")
        return

    print(f"❌ HỢP ĐỒNG KHÔNG ĐẠT — {len(vi_pham)} vi phạm / {n_clips} clip\n")
    print(f"{'mã lỗi':<24}{'số lần':>8}")
    print("-" * 32)
    for ma, n in sorted(theo_ma.items(), key=lambda x: -x[1]):
        print(f"{ma:<24}{n:>8}")
    print("\nVí dụ:")
    for v in vi_pham[:6]:
        print(f"  [{v['reason_code']}] {v['file_id']}: {v['detail']}")


def ghi_csv(vi_pham: list[dict]) -> None:
    """Ghi theo đúng schema của exclusions.csv — lỗi dữ liệu nằm cùng một chỗ với mọi
    quyết định loại bỏ khác, không phải một định dạng riêng chỉ script này hiểu."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    with REPORT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["file_id", "stage", "reason_code", "detail",
                                "decided_by", "decided_at"])
        writer.writeheader()
        for v in vi_pham:
            writer.writerow({**v, "stage": STAGE, "decided_by": "auto", "decided_at": now})
    print(f"\n✓ {REPORT_PATH.relative_to(REPO_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, help="chỉ kiểm N clip đầu (chạy thử nhanh)")
    args = parser.parse_args()
    return run(args.split, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
