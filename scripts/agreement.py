"""Tự-nhất-quán (test–retest) pilot lần 1 ↔ lần 2 — DATA_PLAN §8.3 bước 3 / §8.5.

    python scripts/agreement.py --lan1 data/gold/pilot_v1_lan1.tsv \\
        --lan2 data/gold/pilot_v1_lan2.tsv --mapping data/gold/blind_v1_lan2_mapping.csv

KHÔNG phải kappa liên-người (DATA_PLAN §8.1) — một người gán lại chính mình, sau khoảng
nghỉ, trong điều kiện mù (xem `make_blind_set.py`). Gọi sai tên trong báo cáo là lỗi.

Chỉ chấm các file_id có `la_pilot=True` trong mapping. Mồi (`la_pilot=False`) không có
lần 1 để đối chiếu nên bị loại khỏi tự-nhất-quán — nhãn của mồi ở lần 2 vẫn dùng được cho
gán đại trà, nhưng đọc thẳng từ `--lan2` qua mapping, không qua script này.

Hai cổng của §8.5 áp cho batch này (event-F1 tự-nhất-quán ≥ 0.75, lệch onset trung vị ≤
100 ms). Hai cổng còn lại (≥20 event/lớp, ≥20 clip/slice) áp cho `gold_test` ĐẦY ĐỦ sau
khi gán đại trà — KHÔNG áp cho pilot 30 clip, script này không kiểm chúng.

Script luôn trả 0 nếu tạo được báo cáo — cổng KHÔNG ĐẠT là một phát hiện phải ghi lại,
không phải một lỗi chạy chương trình. Chỉ trả khác 0 khi thiếu file đầu vào.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import enable_utf8_output  # noqa: E402

from ml.datasets.synthetic_sed import event_class_ids  # noqa: E402
from ml.evaluation.error_analysis import so, viet_dan  # noqa: E402
from ml.evaluation.error_taxonomy import KetQuaPhanLoai, phan_loai  # noqa: E402
from ml.evaluation.sed_metrics import ONSET_COLLAR_SEC, event_and_segment_f1  # noqa: E402

ONTOLOGY_PATH = REPO_ROOT / "ml" / "configs" / "ontology_map.yaml"
LAN1_PATH = REPO_ROOT / "data" / "gold" / "pilot_v1_lan1.tsv"
LAN2_PATH = REPO_ROOT / "data" / "gold" / "pilot_v1_lan2.tsv"
MAPPING_PATH = REPO_ROOT / "data" / "gold" / "blind_v1_lan2_mapping.csv"
OUT_PATH = REPO_ROOT / "docs" / "measurements" / "agreement_pilot_v1.md"

DEFAULT_DURATION_SEC = 10.0
NGUONG_TU_NHAT_QUAN = 0.75
NGUONG_LECH_ONSET_MS = 100.0


def doc_tsv_gold(path: Path) -> dict[str, list[dict]]:
    """{filename: [{event_label, onset, offset}]}. File không sự kiện nào không có dòng
    → không có mặt trong dict, giống format thật của `pilot_v1_lan1.tsv`. Việc lấp đầy
    theo một universe cố định là việc của `day_du_universe`, không phải của hàm này."""
    ket_qua: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            ket_qua.setdefault(row["filename"], []).append({
                "event_label": row["event_label"],
                "onset": float(row["onset"]),
                "offset": float(row["offset"]),
            })
    return ket_qua


def doc_anh_xa_pilot(mapping_path: Path) -> dict[str, str]:
    """{ten_mu: file_id} CHỈ cho dòng `la_pilot=True` — mồi không có lần 1 để đối chiếu
    nên không thuộc phạm vi tự-nhất-quán."""
    with mapping_path.open(encoding="utf-8", newline="") as handle:
        return {row["ten_mu"]: row["file_id"] for row in csv.DictReader(handle)
                if row["la_pilot"] == "True"}


def doi_ten_theo_anh_xa(events: dict[str, list[dict]],
                        anh_xa: dict[str, str]) -> tuple[dict[str, list[dict]], list[str]]:
    """(events đổi khoá blind→file_id, [ten_mu KHÔNG có trong anh_xa]).

    Tên không map được (mồi, hoặc tên lạ do gõ nhầm project Label Studio) phải báo ra
    chứ không âm thầm rơi mất — im lặng bỏ qua thì một file gán nhầm dự án biến mất mà
    không ai biết.
    """
    da_doi: dict[str, list[dict]] = {}
    chua_map: list[str] = []
    for ten_mu, su_kien in events.items():
        if ten_mu in anh_xa:
            da_doi[anh_xa[ten_mu]] = su_kien
        else:
            chua_map.append(ten_mu)
    return da_doi, sorted(chua_map)


def day_du_universe(events: dict[str, list[dict]], universe: set[str]) -> dict[str, list[dict]]:
    """Đúng các id trong `universe`: thiếu thì lấp `[]` (không sự kiện nào, không phải
    chưa đo), thừa (ngoài universe) thì loại — tự-nhất-quán chỉ tính trên clip có mặt ở
    CẢ HAI lần."""
    return {fid: events.get(fid, []) for fid in universe}


def lech_onset_list(kq: KetQuaPhanLoai) -> list[float]:
    """Độ lệch onset (giây) của MỌI cặp CÓ CẢ HAI onset — dung/bien/thay_the, không chỉ
    `dung`. Giới hạn ở `dung` sẽ tự động loại mọi cặp lệch quá collar 200 ms, làm trung
    vị luôn nhỏ hơn 100 ms một cách giả tạo — đúng cái con số cổng §8.5 muốn phát hiện."""
    return [abs(c.pred_onset - c.ref_onset) for c in kq.cap
            if c.ref_onset is not None and c.pred_onset is not None]


def trung_vi(gia_tri: list[float]) -> float | None:
    """`None` nghĩa là không có cặp nào ghép được — không phải lệch bằng 0.0. Cùng kỷ
    luật đã áp cho F1 ở epoch chưa full-eval."""
    return statistics.median(gia_tri) if gia_tri else None


def dong_bao_cao(p: dict):
    yield "# Tự-nhất-quán pilot gold lần 1 ↔ lần 2 — DATA_PLAN §8.3 bước 3"
    yield ""
    yield ("KHÔNG phải kappa liên-người — một người gán lại chính mình sau khoảng nghỉ, "
           "trong điều kiện mù (DATA_PLAN §8.1). Collar onset "
           f"{ONSET_COLLAR_SEC * 1000:.0f} ms.")
    yield ""
    if p["chua_map_lan2"]:
        yield (f"⚠️ {len(p['chua_map_lan2'])} tên trong `--lan2` không khớp file_id nào "
               f"trong mapping (mồi, hoặc gõ nhầm project): {', '.join(p['chua_map_lan2'])}")
        yield ""

    yield "## Cổng §8.5 (áp cho batch pilot, KHÔNG áp cho gold_test đầy đủ)"
    yield ""
    dat_f1 = p["event_f1"] >= NGUONG_TU_NHAT_QUAN
    yield (f"- Tự-nhất-quán event-F1: **{so(p['event_f1'], 4)}** "
           f"(ngưỡng ≥ {so(NGUONG_TU_NHAT_QUAN, 2)}) → "
           f"{'✅ ĐẠT' if dat_f1 else '❌ KHÔNG ĐẠT'}")
    if p["lech_onset_med_ms"] is None:
        yield "- Lệch onset trung vị: **chưa đo** (không cặp nào ghép được) → ❌ KHÔNG ĐẠT"
        dat_onset = False
    else:
        dat_onset = p["lech_onset_med_ms"] <= NGUONG_LECH_ONSET_MS
        yield (f"- Lệch onset trung vị: **{so(p['lech_onset_med_ms'], 1)} ms** "
               f"(ngưỡng ≤ {so(NGUONG_LECH_ONSET_MS, 0)} ms) → "
               f"{'✅ ĐẠT' if dat_onset else '❌ KHÔNG ĐẠT'}")
    yield ""
    if not (dat_f1 and dat_onset):
        yield ("→ Không đạt cả hai cổng: quay lại bước 3 (DATA_PLAN §8.3), sửa "
               "`annotation_guideline.md`, KHÔNG đi tiếp bước 4.")
        yield ""
    yield ("Hai cổng còn lại của §8.5 (≥20 event/lớp, ≥20 clip/slice) áp cho `gold_test` "
           "đầy đủ sau khi gán đại trà, KHÔNG áp cho batch pilot này.")
    yield ""

    yield "## Tự-nhất-quán event-F1 theo từng lớp (DATA_PLAN §8.2 B3)"
    yield ""
    yield "| Lớp | Event-F1 |"
    yield "|---|---|"
    for lop in sorted(p["per_class_event"]):
        yield f"| {lop} | {so(p['per_class_event'][lop], 4)} |"


def run(args: argparse.Namespace) -> int:
    for path in (args.lan1, args.lan2, args.mapping):
        if not path.exists():
            print(f"❌ chưa có {path}")
            return 1

    anh_xa = doc_anh_xa_pilot(args.mapping)
    universe = set(anh_xa.values())

    lan1_tho = doc_tsv_gold(args.lan1)
    lan1 = day_du_universe(lan1_tho, universe)

    lan2_blind = doc_tsv_gold(args.lan2)
    lan2_doi_ten, chua_map = doi_ten_theo_anh_xa(lan2_blind, anh_xa)
    lan2 = day_du_universe(lan2_doi_ten, universe)

    class_ids = event_class_ids(ONTOLOGY_PATH)
    _, event_f1, per_class_event = event_and_segment_f1(lan1, lan2, class_ids, DEFAULT_DURATION_SEC)

    kq = phan_loai(lan1, lan2, ONSET_COLLAR_SEC)
    lech_med = trung_vi(lech_onset_list(kq))
    lech_med_ms = None if lech_med is None else lech_med * 1000.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    viet_dan(args.out, dong_bao_cao({
        "event_f1": event_f1, "lech_onset_med_ms": lech_med_ms,
        "per_class_event": per_class_event, "chua_map_lan2": chua_map,
    }))
    print(f"✓ {args.out}")
    return 0


def main() -> int:
    enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lan1", type=Path, default=LAN1_PATH)
    parser.add_argument("--lan2", type=Path, default=LAN2_PATH)
    parser.add_argument("--mapping", type=Path, default=MAPPING_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
