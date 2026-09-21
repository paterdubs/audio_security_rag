"""Test cho scripts/agreement.py — tự-nhất-quán (test–retest) pilot lần 1 ↔ lần 2,
DATA_PLAN §8.3 bước 3 / §8.5.

KHÔNG phải kappa liên-người: dự án chỉ có một người gán (§8.1). Test ở đây phải giữ tên
gọi đúng trong mọi thông điệp báo cáo — gọi nhầm là lỗi báo cáo, không phải lỗi code.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from agreement import (  # noqa: E402
    bo_duoi_wav,
    chi_tiet_bat_dong,
    day_du_universe,
    doc_anh_xa_pilot,
    doc_tsv_gold,
    doi_ten_theo_anh_xa,
    lech_onset_list,
    run,
    trung_vi,
)

from ml.evaluation.error_taxonomy import CapGhep, KetQuaPhanLoai  # noqa: E402


def ghi_tsv(path: Path, dong: list[tuple[str, float, float, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["filename", "onset", "offset", "event_label"])
        for row in dong:
            writer.writerow(row)


def test_doc_tsv_gold_doc_dung_dinh_dang(tmp_path):
    path = tmp_path / "a.tsv"
    ghi_tsv(path, [("f1.wav", 1.0, 2.5, "speech_normal"), ("f1.wav", 3.0, 3.5, "door_slam")])
    kq = doc_tsv_gold(path)
    assert kq == {"f1.wav": [
        {"event_label": "speech_normal", "onset": 1.0, "offset": 2.5},
        {"event_label": "door_slam", "onset": 3.0, "offset": 3.5},
    ]}


def test_doc_tsv_gold_file_khong_su_kien_khong_xuat_hien(tmp_path):
    """File không sự kiện nào không có dòng trong TSV — giống pilot_v1_lan1.tsv thật,
    2/30 file legitimately không có dòng nào. Không được suy ra rỗng cho MỌI file khác
    có tên trong dữ liệu — đó là việc của day_du_universe, không phải của hàm đọc."""
    path = tmp_path / "a.tsv"
    ghi_tsv(path, [("f1.wav", 1.0, 2.0, "speech_normal")])
    kq = doc_tsv_gold(path)
    assert "f2.wav" not in kq
    assert list(kq) == ["f1.wav"]


def test_doc_anh_xa_pilot_chi_lay_la_pilot_true(tmp_path):
    path = tmp_path / "mapping.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ten_mu", "file_id", "la_pilot"])
        writer.writeheader()
        writer.writerow({"ten_mu": "blind_a.wav", "file_id": "p0", "la_pilot": "True"})
        writer.writerow({"ten_mu": "blind_b.wav", "file_id": "m0", "la_pilot": "False"})
    anh_xa = doc_anh_xa_pilot(path)
    assert anh_xa == {"blind_a.wav": "p0"}


def test_doi_ten_theo_anh_xa_map_dung_va_bao_chua_map():
    events = {"blind_a.wav": [{"event_label": "x", "onset": 0.0, "offset": 1.0}],
             "blind_z.wav": [{"event_label": "y", "onset": 0.0, "offset": 1.0}]}
    anh_xa = {"blind_a.wav": "p0"}
    da_doi, chua_map = doi_ten_theo_anh_xa(events, anh_xa)
    assert da_doi == {"p0": events["blind_a.wav"]}
    assert chua_map == ["blind_z.wav"]


def test_bo_duoi_wav_bo_dung_duoi():
    kq = bo_duoi_wav({"p0.wav": [1], "p1": [2]})
    assert kq == {"p0": [1], "p1": [2]}


def test_day_du_universe_dien_rong_cho_thieu():
    kq = day_du_universe({"p0": [{"event_label": "x", "onset": 0.0, "offset": 1.0}]},
                         universe={"p0", "p1"})
    assert kq == {"p0": [{"event_label": "x", "onset": 0.0, "offset": 1.0}], "p1": []}


def test_day_du_universe_bo_id_ngoai_universe():
    """Id không thuộc 30 clip pilot (vd mồi lỡ có mặt) bị loại — universe của tự-nhất-
    quán chỉ gồm đúng các clip có cả hai lần gán."""
    kq = day_du_universe({"p0": [], "m9": []}, universe={"p0"})
    assert kq == {"p0": []}


def test_lech_onset_list_chi_lay_cap_co_ca_hai_onset():
    """Chỉ lấy cặp CÓ CẢ HAI onset (dung/bien/thay_the) — bỏ thieu/thua vì một phía
    không tồn tại thì 'độ lệch' vô nghĩa, không phải bằng 0."""
    cap = [
        CapGhep(clip_id="c1", loai="dung", ref_onset=1.0, pred_onset=1.05),
        CapGhep(clip_id="c1", loai="bien", ref_onset=2.0, pred_onset=2.5),
        CapGhep(clip_id="c1", loai="thieu", ref_onset=3.0, pred_onset=None),
        CapGhep(clip_id="c1", loai="thua", ref_onset=None, pred_onset=4.0),
    ]
    kq = KetQuaPhanLoai(cap=cap)
    lech = sorted(lech_onset_list(kq))
    assert len(lech) == 2
    assert lech[0] == pytest.approx(0.05)
    assert lech[1] == pytest.approx(0.5)


def test_chi_tiet_bat_dong_bo_qua_dung():
    """`dung` (khớp trong collar) không phải ca bất đồng — không cần nghe lại lần ba."""
    cap = [
        CapGhep(clip_id="c1", loai="dung", ref_lop="x", pred_lop="x", ref_onset=1.0, pred_onset=1.05),
        CapGhep(clip_id="c1", loai="bien", ref_lop="x", pred_lop="x", ref_onset=2.0, pred_onset=2.5),
        CapGhep(clip_id="c1", loai="thieu", ref_lop="y", ref_onset=3.0),
        CapGhep(clip_id="c1", loai="thua", pred_lop="z", pred_onset=4.0),
    ]
    kq = KetQuaPhanLoai(cap=cap)
    bat_dong = chi_tiet_bat_dong(kq)
    assert len(bat_dong) == 3
    assert all(r["loai"] != "dung" for r in bat_dong)


def test_chi_tiet_bat_dong_sap_theo_clip_id_roi_onset():
    cap = [
        CapGhep(clip_id="c2", loai="bien", ref_lop="x", pred_lop="x", ref_onset=5.0, pred_onset=5.5),
        CapGhep(clip_id="c1", loai="bien", ref_lop="x", pred_lop="x", ref_onset=2.0, pred_onset=2.5),
        CapGhep(clip_id="c1", loai="thieu", ref_lop="y", ref_onset=1.0),
    ]
    kq = KetQuaPhanLoai(cap=cap)
    bat_dong = chi_tiet_bat_dong(kq)
    assert [(r["clip_id"], r["ref_onset"]) for r in bat_dong] == [
        ("c1", 1.0), ("c1", 2.0), ("c2", 5.0),
    ]


def test_trung_vi_rong_tra_none():
    """Danh sách rỗng (không cặp nào ghép được) → None, không phải 0.0 — cùng kỷ luật
    'chưa đo không phải bằng 0' đã áp cho F1 epoch chưa full-eval."""
    assert trung_vi([]) is None


def test_trung_vi_tinh_dung():
    assert trung_vi([0.1, 0.2, 0.3]) == 0.2
    assert trung_vi([0.1, 0.2, 0.3, 0.4]) == 0.25


def test_run_gate_dat_khi_lan_2_gan_giong_lan_1(tmp_path):
    """Lần 2 lệch onset nhỏ (<100ms), cùng lớp, cùng số sự kiện với lần 1 → cả hai cổng
    §8.5 báo ĐẠT, script trả 0."""
    repo = tmp_path
    lan1 = repo / "lan1.tsv"
    lan2 = repo / "lan2.tsv"
    mapping = repo / "mapping.csv"
    out = repo / "report.md"

    ghi_tsv(lan1, [
        ("p0", 1.000, 2.000, "speech_normal"),
        ("p1", 0.500, 1.500, "door_slam"),
    ])
    ghi_tsv(lan2, [
        ("blind_a.wav", 1.040, 2.040, "speech_normal"),
        ("blind_b.wav", 0.560, 1.560, "door_slam"),
    ])
    with mapping.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ten_mu", "file_id", "la_pilot"])
        writer.writeheader()
        writer.writerow({"ten_mu": "blind_a.wav", "file_id": "p0", "la_pilot": "True"})
        writer.writerow({"ten_mu": "blind_b.wav", "file_id": "p1", "la_pilot": "True"})

    import argparse
    args = argparse.Namespace(lan1=lan1, lan2=lan2, mapping=mapping, out=out)
    ma = run(args)
    assert ma == 0
    noi_dung = out.read_text(encoding="utf-8")
    assert "ĐẠT" in noi_dung
    assert "không phải kappa liên-người" in noi_dung.lower()


def test_run_gate_khong_dat_khi_lan_2_khac_han_lan_1(tmp_path):
    repo = tmp_path
    lan1 = repo / "lan1.tsv"
    lan2 = repo / "lan2.tsv"
    mapping = repo / "mapping.csv"
    out = repo / "report.md"

    ghi_tsv(lan1, [("p0", 1.0, 2.0, "speech_normal")])
    ghi_tsv(lan2, [("blind_a.wav", 8.0, 9.0, "door_slam")])
    with mapping.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ten_mu", "file_id", "la_pilot"])
        writer.writeheader()
        writer.writerow({"ten_mu": "blind_a.wav", "file_id": "p0", "la_pilot": "True"})

    import argparse
    args = argparse.Namespace(lan1=lan1, lan2=lan2, mapping=mapping, out=out)
    ma = run(args)
    assert ma == 0
    noi_dung = out.read_text(encoding="utf-8")
    assert "KHÔNG ĐẠT" in noi_dung
    assert "Danh sách ca bất đồng" in noi_dung
    assert "p0" in noi_dung.split("Danh sách ca bất đồng")[1]


def test_run_lan1_co_duoi_wav_van_khop_duoc_voi_file_id_tran(tmp_path):
    """Tái hiện lỗi thật gặp khi chạy production 22/09: `pilot_v1_lan1.tsv` ghi filename
    CÓ đuôi `.wav` (đúng quy ước TSV — tên file WAV thật, ví dụ
    `as_strong_-0TTFAArJ9k_30000.wav`), còn cột `file_id` của mapping.csv KHÔNG có đuôi
    (đúng quy ước file_id xuyên suốt pipeline, ví dụ `as_strong_-0TTFAArJ9k_30000`).
    Trước khi vá, universe (suy từ mapping) không khớp được key nào của lan1 → lan1 rỗng
    tuyệt đối → event_f1/per_class_event ra NaN cho MỌI lớp, không phải một cổng thật sự
    không đạt."""
    repo = tmp_path
    lan1 = repo / "lan1.tsv"
    lan2 = repo / "lan2.tsv"
    mapping = repo / "mapping.csv"
    out = repo / "report.md"

    ghi_tsv(lan1, [("p0.wav", 1.000, 2.000, "speech_normal")])
    ghi_tsv(lan2, [("blind_a.wav", 1.040, 2.040, "speech_normal")])
    with mapping.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ten_mu", "file_id", "la_pilot"])
        writer.writeheader()
        writer.writerow({"ten_mu": "blind_a.wav", "file_id": "p0", "la_pilot": "True"})

    import argparse
    args = argparse.Namespace(lan1=lan1, lan2=lan2, mapping=mapping, out=out)
    ma = run(args)
    assert ma == 0
    noi_dung = out.read_text(encoding="utf-8")
    # 14/15 lớp không xuất hiện trong fixture bé này nên NaN per-class là ĐÚNG (0 ref/0
    # pred không định nghĩa được) — chỉ cổng tổng hợp và lớp CÓ dữ liệu mới cần là số thật.
    assert "event-F1: **1,0000**" in noi_dung
    assert "| speech_normal | 1,0000 |" in noi_dung
    assert "✅ ĐẠT" in noi_dung
