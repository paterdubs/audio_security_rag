"""Test cho `select_gold_pilot.py` — chọn N clip pilot cho gold_test, rải đều theo lớp.

Mục đích: pilot 30 clip (DATA_PLAN §8.3 bước 1) phải phủ đủ 15 lớp để phát hiện lỗ hổng
guideline sớm — chọn ngẫu nhiên thuần tuý có thể ra 30 clip toàn `speech_normal` (lớp có
nhiều file nhất, 210/367) mà bỏ sót `door_slam` (chỉ 11 file) hoàn toàn.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from select_gold_pilot import (  # noqa: E402
    chon_pilot,
    doc_gold_theo_lop,
    file_id_bi_loai,
    gom_theo_lop,
    loc_chi_lop,
    segment_bi_loai,
)


def test_chon_pilot_deterministic():
    """Cùng seed, cùng dữ liệu → cùng kết quả qua hai lần gọi. Không có tính chất này thì
    'gán lại lần 2' (DATA_PLAN §8, bước 2) không tái lập được đúng 30 clip của lần 1."""
    theo_lop = {"a": ["f1", "f2", "f3"], "b": ["f2", "f4", "f5"]}
    kq1 = chon_pilot(theo_lop, n=4, seed="hat-giong-1")
    kq2 = chon_pilot(theo_lop, n=4, seed="hat-giong-1")
    assert kq1 == kq2


def test_chon_pilot_moi_lop_co_it_nhat_mot_file_khi_du_du_lieu():
    """15 lớp, mỗi lớp ≥2 file, n=30 → round-robin phải cho MỌI lớp xuất hiện ít nhất
    một lần. Chọn ngẫu nhiên thuần tuý (không rải theo lớp) có xác suất bỏ sót lớp hiếm
    (11/367 file như door_slam) không hề nhỏ."""
    theo_lop = {chr(65 + i): [f"{chr(65 + i)}_{j}" for j in range(5)] for i in range(15)}
    kq = chon_pilot(theo_lop, n=30, seed="s")
    lop_co_mat = set()
    for fid in kq:
        for lop, files in theo_lop.items():
            if fid in files:
                lop_co_mat.add(lop)
    assert lop_co_mat == set(theo_lop)


def test_chon_pilot_khong_vuot_qua_n():
    theo_lop = {"a": [f"a{i}" for i in range(50)], "b": [f"b{i}" for i in range(50)]}
    kq = chon_pilot(theo_lop, n=10, seed="s")
    assert len(kq) == 10


def test_chon_pilot_khong_chon_trung_file_du_thuoc_nhieu_lop():
    """Một file AudioSet-strong 10s có thể chứa nhiều sự kiện của nhiều lớp khác nhau —
    nếu thuật toán không khử trùng, file đó bị đếm nhiều lần và pilot thực tế < n clip
    thật, dù báo cáo ra đủ n."""
    theo_lop = {"a": ["x", "y"], "b": ["x", "z"], "c": ["x"]}
    kq = chon_pilot(theo_lop, n=3, seed="s")
    assert len(kq) == len(set(kq))


def test_chon_pilot_lop_it_file_khong_lam_chet_vong_lap():
    """Lớp hiếm (door_slam: 11/367 file) hết file trước các lớp khác — vòng lặp phải bỏ
    qua lớp đó và tiếp tục với lớp còn file, không phải dừng hẳn hoặc lặp vô hạn."""
    theo_lop = {"hiem": ["h1"], "thuong": [f"t{i}" for i in range(20)]}
    kq = chon_pilot(theo_lop, n=10, seed="s")
    assert len(kq) == 10
    assert "h1" in kq


# ── Tôn trọng exclusions.csv — audio rác phát hiện lúc gán không được chọn lại ──


def test_gom_theo_lop_bo_qua_segment_ngoai_gold_test():
    segments = [
        {"ytid": "AAA", "file_id": "seg_a", "path": "p/a.wav",
         "events": [{"class_id": "siren"}]},
        {"ytid": "BBB", "file_id": "seg_b", "path": "p/b.wav",
         "events": [{"class_id": "siren"}]},
    ]
    kq = gom_theo_lop(gold_groups={"youtube_AAA"}, segments=segments)
    assert kq["theo_lop"] == {"siren": ["seg_a"]}
    assert "seg_b" not in kq["thong_tin"]


def test_gom_theo_lop_bo_qua_segment_bi_loai():
    """Đây là ca 'audio rác': segment vẫn thuộc gold_test, nhưng đã bị đánh dấu loại —
    phải biến mất khỏi mọi lớp, không chỉ khỏi một lớp."""
    segments = [{"ytid": "AAA", "file_id": "seg_a", "path": "p/a.wav",
                "events": [{"class_id": "siren"}, {"class_id": "alarm_bell"}]}]
    kq = gom_theo_lop(gold_groups={"youtube_AAA"}, segments=segments,
                      segment_loai_tru={"seg_a"})
    assert kq["theo_lop"] == {}
    assert kq["thong_tin"] == {}


def test_file_id_bi_loai_rong_khi_chua_co_exclusions_csv(tmp_path):
    """exclusions.csv chỉ được tạo khi có file đầu tiên bị loại — chưa có file này
    không phải lỗi, và không được làm select_gold_pilot.py chết."""
    assert file_id_bi_loai(tmp_path / "khong_ton_tai.csv") == set()


def test_file_id_bi_loai_doc_dung_cot_file_id(tmp_path):
    path = tmp_path / "exclusions.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_id", "stage", "reason_code", "detail", "decided_by", "decided_at"])
        writer.writerow(["evt_1", "gold_pilot", "corrupted", "toàn tiếng ồn trắng",
                         "nguoi_gan", "2026-09-22T00:00:00+00:00"])
    assert file_id_bi_loai(path) == {"evt_1"}


def test_segment_bi_loai_gom_ca_segment_khi_mot_su_kien_bi_loai(tmp_path):
    """Một segment audioset_strong có NHIỀU dòng sự kiện trong raw_manifest.csv (mỗi sự
    kiện một file_id). Loại một sự kiện của nó vì audio rác phải kéo theo loại CẢ
    segment — âm thanh đã rác thì rác cho mọi lớp, không phải chỉ lớp bị ghi exclusion."""
    path = tmp_path / "raw_manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_id", "source_dataset", "source_id"])
        writer.writerow(["evt_1", "audioset_strong", "seg_a"])
        writer.writerow(["evt_2", "audioset_strong", "seg_a"])
        writer.writerow(["evt_3", "audioset_strong", "seg_b"])
    assert segment_bi_loai(path, loai={"evt_1"}) == {"seg_a"}


def test_segment_bi_loai_rong_khi_khong_co_gi_bi_loai(tmp_path):
    path = tmp_path / "raw_manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_id", "source_dataset", "source_id"])
        writer.writerow(["evt_1", "audioset_strong", "seg_a"])
    assert segment_bi_loai(path, loai=set()) == set()


# ── Thiên về lớp hiếm (vòng pilot sau, DATA_PLAN §8.3 lặp lại) ──────────────────


def test_loc_chi_lop_giu_dung_cac_lop_yeu_cau():
    theo_lop = {"door_slam": ["a"], "speech_normal": ["b"], "siren": ["c"]}
    kq = loc_chi_lop(theo_lop, {"door_slam", "siren"})
    assert kq == {"door_slam": ["a"], "siren": ["c"]}


def test_loc_chi_lop_rong_hoac_none_giu_nguyen_moi_lop():
    """Mặc định (không truyền --chi-lop) phải giữ hành vi cũ — pilot lần 1 gọi hàm này
    với chi_lop=None và không được đổi kết quả."""
    theo_lop = {"a": ["x"], "b": ["y"]}
    assert loc_chi_lop(theo_lop, None) == theo_lop
    assert loc_chi_lop(theo_lop, set()) == theo_lop


def test_loc_chi_lop_lop_khong_ton_tai_khong_bao_loi():
    theo_lop = {"a": ["x"]}
    assert loc_chi_lop(theo_lop, {"lop_khong_co", "a"}) == {"a": ["x"]}


# ── Loại clip đã dùng ở vòng pilot trước — không được chọn lại (--tru-them) ─────


def test_doc_gold_theo_lop_loai_clip_da_dung_o_vong_truoc(tmp_path):
    """Ca thảm hoạ nếu thiếu: vòng pilot 3 vô tình chọn lại đúng 30 clip của vòng 1 —
    người gán nhớ lại nhãn cũ, tự-nhất-quán đo được là trí nhớ, không phải guideline."""
    splits = tmp_path / "splits.csv"
    with splits.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_dataset", "split", "source_group_id"])
        w.writerow(["audioset_strong", "gold_test", "youtube_AAA"])
        w.writerow(["audioset_strong", "gold_test", "youtube_BBB"])

    segments = tmp_path / "segments.jsonl"
    with segments.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"ytid": "AAA", "file_id": "seg_a", "path": "p/a.wav",
                            "events": [{"class_id": "siren"}]}) + "\n")
        f.write(json.dumps({"ytid": "BBB", "file_id": "seg_b", "path": "p/b.wav",
                            "events": [{"class_id": "siren"}]}) + "\n")

    tru_them = tmp_path / "da_dung.csv"
    with tru_them.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_id"])
        w.writerow(["seg_a"])

    kq = doc_gold_theo_lop(splits, segments, tru_them_path=tru_them)
    assert kq["theo_lop"] == {"siren": ["seg_b"]}
    assert "seg_a" not in kq["thong_tin"]


def test_doc_gold_theo_lop_khong_truyen_tru_them_giu_hanh_vi_cu(tmp_path):
    splits = tmp_path / "splits.csv"
    with splits.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_dataset", "split", "source_group_id"])
        w.writerow(["audioset_strong", "gold_test", "youtube_AAA"])

    segments = tmp_path / "segments.jsonl"
    with segments.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"ytid": "AAA", "file_id": "seg_a", "path": "p/a.wav",
                            "events": [{"class_id": "siren"}]}) + "\n")

    kq = doc_gold_theo_lop(splits, segments)
    assert kq["theo_lop"] == {"siren": ["seg_a"]}
