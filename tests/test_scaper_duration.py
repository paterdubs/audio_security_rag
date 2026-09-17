"""Test cho khâu RÚT THỜI LƯỢNG của scaper_generate.py — STATUS §7 B3.

Toàn bộ 16.545 sự kiện của lô cũ nằm trong [0.19 s, 1.10 s] — không một ngoại lệ.
Trần 1.10 s chính là `event_duration=1.0` nhân hệ số kéo tối đa 1.1. Trong khi đó
bộ đánh giá thật (AudioSet Strong, 4.012 sự kiện) có p95 = 7.94 s và 17.5% sự
kiện vượt 2 giây.

Đó không phải "một lỗi dữ liệu" mà là LỆCH PHÂN BỐ train/test: model được dạy mọi
sự kiện dài khoảng một giây, rồi bị chấm trên tập mà còi hụ có p50 = 8.99 s.

B3 bỏ hằng số, thay bằng lấy mẫu lại từ chính phân bố của miền đích. Phép đo quan
trọng nhất vì thế không phải "hàm có chạy không" mà là "phân bố sinh ra có khớp
phân bố đích không" — nên test xương sống ở đây dùng khoảng cách Kolmogorov–Smirnov.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import scaper_generate as sg  # noqa: E402


def _viet_segments(tmp_path: Path, su_kien: dict[str, list[tuple[float, float]]],
                   ytid: str = "x") -> Path:
    """Dựng segments.jsonl tối giản: {lớp: [(onset, offset), ...]}."""
    ra = tmp_path / "segments.jsonl"
    dong = []
    for lop, cac_khoang in su_kien.items():
        for onset, offset in cac_khoang:
            dong.append(json.dumps({
                "file_id": f"as_{lop}_{onset}", "path": "x.wav", "ytid": ytid, "window_start_ms": 0,
                "events": [{"class_id": lop, "onset": onset, "offset": offset}],
            }))
    ra.write_text("\n".join(dong), encoding="utf-8")
    return ra


def _ks(a, b) -> float:
    """Khoảng cách Kolmogorov–Smirnov giữa hai mẫu — 0 là trùng khít."""
    from scipy.stats import ks_2samp

    return float(ks_2samp(a, b).statistic)


# ── Đọc phân bố đích ─────────────────────────────────────────────────────────


def test_doc_phan_bo_dich_gom_thoi_luong_theo_lop(tmp_path):
    path = _viet_segments(tmp_path, {"siren": [(0.0, 9.0), (1.0, 2.0)], "gunshot": [(0.0, 0.4)]})

    phan_bo = sg.doc_phan_bo_dich(path, loai_tru_ytid=set())

    assert sorted(phan_bo) == ["gunshot", "siren"]
    assert sorted(phan_bo["siren"]) == pytest.approx([1.0, 9.0])


def test_doc_phan_bo_dich_cat_thoi_luong_vuot_do_dai_clip(tmp_path):
    """AudioSet ghi offset tới 10.0 s; sự kiện dài hơn clip là vô nghĩa với ta."""
    path = _viet_segments(tmp_path, {"siren": [(0.0, 30.0)]})

    phan_bo = sg.doc_phan_bo_dich(path, do_dai_clip=10.0, loai_tru_ytid=set())

    assert phan_bo["siren"] == (10.0,)


def test_doc_phan_bo_dich_bo_su_kien_do_dai_khong_duong(tmp_path):
    """onset == offset là nhãn hỏng — để lọt sẽ thành event_duration=0."""
    path = _viet_segments(tmp_path, {"siren": [(1.0, 1.0), (2.0, 5.0), (3.0, 2.0)]})

    assert sg.doc_phan_bo_dich(path, loai_tru_ytid=set())["siren"] == (3.0,)


def test_doc_phan_bo_dich_bao_loi_khi_thieu_file(tmp_path):
    with pytest.raises(SystemExit, match="fetch_audioset_strong"):
        sg.doc_phan_bo_dich(tmp_path / "khong_co.jsonl")


# ── Rút thời lượng ───────────────────────────────────────────────────────────


def test_rut_thoi_luong_chi_tra_gia_tri_co_that_trong_phan_bo(tmp_path):
    """Lấy mẫu lại (bootstrap), không nội suy — giá trị nào cũng phải có thật."""
    phan_bo = sg.doc_phan_bo_dich(_viet_segments(tmp_path, {
        "siren": [(0.0, d) for d in (0.5, 2.0, 9.0)]}), loai_tru_ytid=set())
    rng = random.Random(0)

    ra = {sg.rut_thoi_luong(rng, phan_bo, "siren", toi_da=10.0) for _ in range(200)}

    assert ra <= {0.5, 2.0, 9.0}
    assert ra == {0.5, 2.0, 9.0}, "không phủ hết phân bố"


def test_rut_thoi_luong_khop_phan_bo_dich(tmp_path):
    """Test xương sống: mẫu rút ra phải trùng khít phân bố đích.

    Đây là điều B3 hứa hẹn. Hàm chạy không lỗi mà phân bố lệch thì vẫn sinh ra một
    train set lệch miền — đúng cái hỏng đang phải sửa, và cũng không có triệu chứng.
    """
    goc = list(np.random.default_rng(0).lognormal(mean=-0.3, sigma=1.0, size=800).clip(0.05, 10.0))
    phan_bo = sg.doc_phan_bo_dich(_viet_segments(tmp_path, {"siren": [(0.0, d) for d in goc]}), loai_tru_ytid=set())
    rng = random.Random(1)

    mau = [sg.rut_thoi_luong(rng, phan_bo, "siren", toi_da=10.0) for _ in range(4000)]

    assert _ks(mau, goc) < 0.06


def test_rut_thoi_luong_ton_trong_tran_toi_da(tmp_path):
    """`toi_da` chặn trên cứng — dùng cho ca sự kiện phải lọt trong phần còn lại của clip."""
    phan_bo = sg.doc_phan_bo_dich(_viet_segments(tmp_path, {
        "siren": [(0.0, d) for d in (0.5, 4.0, 9.5)]}), loai_tru_ytid=set())
    rng = random.Random(0)

    for _ in range(200):
        assert sg.rut_thoi_luong(rng, phan_bo, "siren", toi_da=3.0) <= 3.0


def test_rut_thoi_luong_tat_dinh_theo_seed(tmp_path):
    phan_bo = sg.doc_phan_bo_dich(_viet_segments(tmp_path, {
        "siren": [(0.0, 0.1 * i) for i in range(1, 60)]}), loai_tru_ytid=set())

    lay = lambda: [sg.rut_thoi_luong(random.Random(9), phan_bo, "siren", 10.0) for _ in range(20)]  # noqa: E731
    assert lay() == lay()


def test_rut_thoi_luong_lop_thieu_trong_phan_bo_bao_loi_ro_rang(tmp_path):
    """Lớp không có trong miền đích phải nổ ngay, không lặng lẽ dùng giá trị mặc định.

    Lặng lẽ thay bằng 1.0 s chính là cách lô cũ sinh ra 16.545 sự kiện dài một giây.
    """
    phan_bo = sg.doc_phan_bo_dich(_viet_segments(tmp_path, {"siren": [(0.0, 2.0)]}), loai_tru_ytid=set())

    with pytest.raises(KeyError, match="gunshot"):
        sg.rut_thoi_luong(random.Random(0), phan_bo, "gunshot", 10.0)


def test_kiem_phu_song_liet_ke_lop_thieu():
    """Thiếu lớp phải được liệt kê MỘT LƯỢT, không dừng ở lớp đầu tiên."""
    thieu = sg.kiem_phu_song({"a": (), "b": (), "c": ()}, {"a": (1.0,)})

    assert thieu == ["b", "c"]


# ── Ghép với khâu chọn nguồn (B2 + B3) ───────────────────────────────────────


def test_ghep_b2_b3_khop_phan_bo_khi_bank_du_nguon(tmp_path):
    """Bank có đủ clip dài → phân bố sinh ra vẫn bám phân bố đích.

    Tách riêng với ca bank thiếu nguồn: lệch do BANK THIẾU là giới hạn vật lý phải
    ghi nhận, còn lệch khi bank ĐỦ là lỗi của mã.
    """
    import csv

    goc = list(np.random.default_rng(2).lognormal(mean=-0.2, sigma=0.9, size=500).clip(0.05, 10.0))
    phan_bo = sg.doc_phan_bo_dich(_viet_segments(tmp_path, {"siren": [(0.0, d) for d in goc]}),
                                  loai_tru_ytid=set())

    csv_path = tmp_path / "bank_trim.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=sg.COT_BANG_TRA)
        w.writeheader()
        for i in range(400):                      # clip trải từ 0.1 s tới 10 s
            w.writerow({"file_id": f"c{i:03d}", "class_id": "siren",
                        "path": f"data/banks/foreground/siren/c{i:03d}.wav",
                        "dur": 0.1 + i * 0.025, "eff": 0.1 + i * 0.025, "lead": 0.0,
                        "trail": 0.0, "rms_db": -20.0, "peak": 0.5, "loi": ""})
    bang = sg.doc_bang_tra(csv_path)
    rng = random.Random(3)

    mau = [sg.chon_nguon(rng, bang["siren"], sg.rut_thoi_luong(rng, phan_bo, "siren", 10.0))[1]
           for _ in range(3000)]

    assert _ks(mau, goc) < 0.08


def test_ghep_b2_b3_bank_thieu_nguon_thi_lech_XUONG_chu_khong_len(tmp_path):
    """Bank không đủ dài thì thời lượng chỉ được HỤT, không bao giờ vượt.

    Vượt nghĩa là ta khai một sự kiện dài hơn phần có tiếng của file — tức là nhãn
    phủ lên cả im lặng, đúng loại nhãn sai mà không nghe ra được.
    """
    import csv

    phan_bo = sg.doc_phan_bo_dich(_viet_segments(tmp_path, {
        "siren": [(0.0, d) for d in (6.0, 7.0, 8.0, 9.0)]}), loai_tru_ytid=set())
    csv_path = tmp_path / "bank_trim.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=sg.COT_BANG_TRA)
        w.writeheader()
        for i in range(10):                        # bank chỉ tới 2.0 s
            w.writerow({"file_id": f"c{i}", "class_id": "siren",
                        "path": f"data/banks/foreground/siren/c{i}.wav",
                        "dur": 1.0 + i * 0.1, "eff": 1.0 + i * 0.1, "lead": 0.0,
                        "trail": 0.0, "rms_db": -20.0, "peak": 0.5, "loi": ""})
    bang = sg.doc_bang_tra(csv_path)
    rng = random.Random(4)

    for _ in range(300):
        muon = sg.rut_thoi_luong(rng, phan_bo, "siren", 10.0)
        nguon, thuc = sg.chon_nguon(rng, bang["siren"], muon)
        assert thuc <= muon + 1e-9
        assert thuc <= nguon.eff + 1e-9


# ── Chặn rò rỉ phân bố: eval_strong giữ nguyên cho gold_test ─────────────────
#
# `fetch_audioset_strong.py` gộp cả hai split khi dựng segments.jsonl:
#     rows = load_strong_rows(train_path) + load_strong_rows(eval_path)
# nên 937 segment của ta = 825 từ train_strong + 112 từ eval_strong.
#
# Lấy phân bố thời lượng trên cả 937 để THIẾT KẾ dữ liệu huấn luyện nghĩa là thống
# kê của những clip sẽ thành gold_test đã chảy ngược vào train. Không phải rò rỉ
# nhãn theo nghĩa nặng, nhưng đủ để một hội đồng hỏi và ta không có câu trả lời sạch.
#
# Bỏ 112 clip eval ra gần như không đổi phân bố (KS = 0.0083 đo trên dữ liệu thật),
# nên ta được sạch về phương pháp mà không mất gì.


def _viet_eval_tsv(tmp_path: Path, ytids: list[str]) -> Path:
    """audioset_eval_strong.tsv tối giản — segment_id là `<ytid>_<start_ms>`."""
    ra = tmp_path / "audioset_eval_strong.tsv"
    dong = ["segment_id\tstart_time_seconds\tend_time_seconds\tlabel"]
    dong += [f"{y}_30000\t0.000\t10.000\t/m/04rlf" for y in ytids]
    ra.write_text("\n".join(dong), encoding="utf-8")
    return ra


def test_ytid_eval_strong_boc_dung_ytid_khoi_segment_id(tmp_path):
    """ytid YouTube chứa được cả `-` và `_`, nên chỉ được cắt nhóm số cuối cùng."""
    path = _viet_eval_tsv(tmp_path, ["--CHY2qO5zc", "s9d-2nhuJCQ", "a_b_c"])

    assert sg.ytid_eval_strong(path) == {"--CHY2qO5zc", "s9d-2nhuJCQ", "a_b_c"}


def test_doc_phan_bo_dich_loai_clip_thuoc_eval_strong(tmp_path):
    """Clip phía eval không được góp một con số nào vào phân bố đích."""
    ra = tmp_path / "segments.jsonl"
    ra.write_text("\n".join([
        json.dumps({"file_id": "a", "path": "x", "ytid": "PHIA_TRAIN", "window_start_ms": 0,
                    "events": [{"class_id": "siren", "onset": 0.0, "offset": 2.0}]}),
        json.dumps({"file_id": "b", "path": "x", "ytid": "PHIA_EVAL", "window_start_ms": 0,
                    "events": [{"class_id": "siren", "onset": 0.0, "offset": 9.0}]}),
    ]), encoding="utf-8")

    phan_bo = sg.doc_phan_bo_dich(ra, loai_tru_ytid={"PHIA_EVAL"})

    assert phan_bo["siren"] == (2.0,), "thời lượng phía eval đã lọt vào phân bố đích"


def test_doc_phan_bo_dich_lop_chi_co_o_phia_eval_thi_bien_han(tmp_path):
    """Lớp chỉ xuất hiện phía eval phải biến mất hẳn, không để lại lớp rỗng.

    Lớp rỗng lọt xuống dưới thì `rng.choice(())` nổ giữa chừng lượt sinh; tệ hơn là
    ai đó thêm nhánh mặc định và ta quay lại đúng hằng số 1.0 giây.
    """
    ra = tmp_path / "segments.jsonl"
    ra.write_text(json.dumps({"file_id": "b", "path": "x", "ytid": "PHIA_EVAL",
                              "window_start_ms": 0,
                              "events": [{"class_id": "siren", "onset": 0.0, "offset": 9.0}]}),
                  encoding="utf-8")

    assert "siren" not in sg.doc_phan_bo_dich(ra, loai_tru_ytid={"PHIA_EVAL"})


def test_doc_phan_bo_dich_thieu_nhan_eval_thi_CHET_chu_khong_gop_ca_hai(tmp_path):
    """Thiếu audioset_eval_strong.tsv phải dừng hẳn.

    Đây là điểm mấu chốt. Nếu thiếu file mà lặng lẽ dùng cả 937 clip thì bản sửa rò
    rỉ này tự vô hiệu hoá đúng vào lúc không ai nhìn — và không có triệu chứng nào.
    """
    ra = _viet_segments(tmp_path, {"siren": [(0.0, 2.0)]})

    with pytest.raises(SystemExit, match="eval_strong"):
        sg.doc_phan_bo_dich(ra, nhan_eval=tmp_path / "khong_co.tsv")


# ── Sự kiện dài (B5) ─────────────────────────────────────────────────────────
#
# Hợp đồng cũ khai `long_event: {ratio: 0.10, min_event_duration_sec: 8.0}` và log
# sinh báo "long_event 10.0% · 791 clip". Số sự kiện ≥ 8 s thật sự sinh ra: 0.
# Nhãn lát cắt có, nội dung không có — mọi kết luận từng rút ra trên lát này đều
# vô hiệu. Đây là lỗi im lặng nặng nhất tìm được trong dự án.
#
# Đo trữ lượng theo ngưỡng (lớp đủ CẢ bank ≥8 clip LẪN có sự kiện thật trong miền đích):
#     2s → 14 lớp    3s → 9 lớp    4s → 8 lớp    6s → 6 lớp    8s → 4 lớp
# Ngưỡng 8 s vừa bất khả thi vừa không khớp thực tế (miền đích chỉ 5.1% sự kiện ≥8s).


def _bang_gia(tmp_path, dac_ta: dict[str, list[float]]):
    """bảng tra giả: {lớp: [eff, ...]}."""
    import csv

    ra = tmp_path / "bank_trim.csv"
    with ra.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=sg.COT_BANG_TRA)
        w.writeheader()
        for lop, effs in dac_ta.items():
            for i, eff in enumerate(effs):
                w.writerow({"file_id": f"{lop}_{i}", "class_id": lop,
                            "path": f"data/banks/foreground/{lop}/{lop}_{i}.wav",
                            "dur": eff, "eff": eff, "lead": 0.0, "trail": 0.0,
                            "rms_db": -20.0, "peak": 0.5, "loi": ""})
    return sg.doc_bang_tra(ra)


def test_lop_cap_duoc_su_kien_dai_doi_HAI_dieu_kien(tmp_path):
    """Lớp phải vừa có đủ clip bank dài, VỪA có sự kiện dài thật trong miền đích."""
    bang = _bang_gia(tmp_path, {
        "siren": [5.0] * 10,              # đủ bank, và miền đích có → nhận
        "object_drop_dishes": [5.0] * 10,  # đủ bank nhưng miền đích KHÔNG có → loại
        "explosion": [5.0, 5.0],           # miền đích có nhưng bank chỉ 2 clip → loại
    })
    phan_bo = {"siren": (1.0, 6.0), "object_drop_dishes": (0.3, 1.2), "explosion": (1.0, 6.0)}

    assert sg.lop_cap_duoc_su_kien_dai(bang, phan_bo, nguong=4.0, so_clip_toi_thieu=8) == ("siren",)


def test_lop_cap_duoc_su_kien_dai_loai_lop_dai_tren_giay_nhung_khong_co_that(tmp_path):
    """Ca `object_drop_dishes`: 22 clip bank ≥ 4 s mà miền đích có 0 sự kiện ≥ 4 s.

    Tiếng rơi bát đĩa kéo dài 4 giây không tồn tại ngoài đời. Ép nó vào là dạy model
    một thứ sẽ không bao giờ gặp, và tệ hơn là làm nó bỏ qua tiếng rơi bát đĩa thật.
    """
    bang = _bang_gia(tmp_path, {"object_drop_dishes": [6.0] * 30})

    assert sg.lop_cap_duoc_su_kien_dai(bang, {"object_drop_dishes": (0.3, 1.38)}, 4.0, 8) == ()


def test_rut_thoi_luong_dai_chi_tra_gia_tri_dat_nguong():
    phan_bo = {"siren": (0.5, 1.0, 4.2, 6.0, 9.0)}
    rng = random.Random(0)

    ra = {sg.rut_thoi_luong_dai(rng, phan_bo, "siren", nguong=4.0, toi_da=10.0) for _ in range(300)}

    assert ra == {4.2, 6.0, 9.0}


def test_rut_thoi_luong_dai_ton_trong_tran_toi_da():
    rng = random.Random(0)

    for _ in range(100):
        assert sg.rut_thoi_luong_dai(rng, {"siren": (4.2, 6.0, 9.0)}, "siren", 4.0, 5.0) <= 5.0


def test_rut_thoi_luong_dai_khong_co_gia_tri_nao_thi_NO_chu_khong_im_lang():
    """Không có thời lượng nào đạt ngưỡng phải nổ.

    Lặng lẽ trả về giá trị ngắn nhất chính là hình dạng của lỗi `long_event`: hợp
    đồng khai đạt, thực tế không có gì, và không ai biết.
    """
    with pytest.raises(ValueError, match="long_event"):
        sg.rut_thoi_luong_dai(random.Random(0), {"siren": (0.5, 1.0)}, "siren", 4.0, 10.0)


def test_ung_vien_su_kien_dai_loc_dung_clip_du_dai(tmp_path):
    bang = _bang_gia(tmp_path, {"siren": [1.0, 3.9, 4.0, 9.0]})

    ung_vien = sg.ung_vien_su_kien_dai(bang["siren"], nguong=4.0)

    assert sorted(c.eff for c in ung_vien) == [4.0, 9.0]


def test_hop_dong_long_event_duoc_BAO_DAM_khong_the_tut_duoi_nguong(tmp_path):
    """Bất biến xương sống B5: đã chọn trong nhóm đủ dài thì d_thực KHÔNG BAO GIỜ < ngưỡng.

    `chon_nguon` có nhánh hạ thời lượng khi bank thiếu nguồn. Nếu nhánh đó chạm được
    vào clip ngắn hơn ngưỡng thì hợp đồng long_event lại hỏng đúng như lô cũ — chỉ
    khác là lần này có nhãn 4 s thay vì 8 s. Lọc ứng viên TRƯỚC khi gọi mới chặn được.
    """
    bang = _bang_gia(tmp_path, {"siren": [0.5] * 50 + [4.1, 4.5, 5.0, 9.4]})
    phan_bo = {"siren": (0.3, 1.0, 4.2, 6.0, 9.9)}
    rng = random.Random(3)
    ung_vien = sg.ung_vien_su_kien_dai(bang["siren"], 4.0)

    for _ in range(500):
        d = sg.rut_thoi_luong_dai(rng, phan_bo, "siren", 4.0, 10.0)
        _, d_thuc = sg.chon_nguon(rng, ung_vien, d)
        assert d_thuc >= 4.0 - 1e-9, f"tụt xuống {d_thuc:.2f}s, dưới ngưỡng hợp đồng"


def test_nguong_co_bien_bu_cho_time_stretch():
    """eff = đúng ngưỡng rồi gặp hệ số kéo 0.9 thì nhãn cuối chỉ còn 90% ngưỡng."""
    assert sg.nguong_co_bien(4.0, {"time_stretch": [0.9, 1.1]}) == pytest.approx(4.0 / 0.9)
    assert sg.nguong_co_bien(4.0, {"time_stretch": [1.0, 1.0]}) == pytest.approx(4.0)


def test_su_kien_dai_van_dat_nguong_SAU_khi_nhan_time_stretch(tmp_path):
    """Bất biến phải đúng trên THỜI LƯỢNG CUỐI, không phải trên eff.

    Bộ mô phỏng B7 bắt được đúng ca này trên dữ liệu thật: 2/791 clip `long_event`
    trượt ngưỡng vì `eff × keo` tụt xuống dưới 4 s. Hai trên bảy trăm chín mốt là
    thứ không ai nghe ra được, và để lọt thì hợp đồng lại thành "gần đúng".
    """
    su_kien = {"pitch_shift_semitones": [-1, 1], "time_stretch": [0.9, 1.1]}
    nguong = 4.0
    bang = _bang_gia(tmp_path, {"siren": [4.05, 4.2, 4.6, 5.0, 6.0, 7.0, 8.0, 9.0, 9.4, 10.0]})
    phan_bo = {"siren": (4.1, 4.5, 5.0, 8.0)}
    ung_vien = sg.ung_vien_su_kien_dai(bang["siren"], sg.nguong_co_bien(nguong, su_kien))
    rng = random.Random(0)

    for _ in range(2000):
        d = sg.rut_thoi_luong_dai(rng, phan_bo, "siren", nguong, 10.0)
        nguon, d_thuc = sg.chon_nguon(rng, ung_vien, d)
        tham_so = sg.tham_so_su_kien(nguon, d_thuc, (5.0, 5.0), su_kien, rng)

        assert tham_so["thoi_luong_cuoi"] >= nguong - 1e-9, \
            f"nhãn cuối {tham_so['thoi_luong_cuoi']:.3f}s < ngưỡng {nguong}s"


def test_doc_phan_bo_dich_bo_nhan_thoai_hoa_cuc_ngan(tmp_path):
    """AudioSet có nhãn xuống tới 0.001 s — đó là lỗi gán nhãn, không phải sự kiện.

    Để lọt thì Scaper NỔ giữa chừng lượt sinh: nó fade 10 ms vào và 10 ms ra, nên
    `event_audio[:160] *= cửa_sổ` vỡ khi sự kiện chỉ có 16 mẫu. Đã xảy ra thật, làm
    hỏng lượt sinh ở clip thứ ~7.815/7.920 — tức là sau 45 phút chạy.
    """
    path = _viet_segments(tmp_path, {"siren": [(0.0, 0.001), (1.0, 1.04), (2.0, 2.5)]})

    phan_bo = sg.doc_phan_bo_dich(path, loai_tru_ytid=set(), toi_thieu=0.05)

    assert phan_bo["siren"] == (0.5,), "nhãn 1 ms và 40 ms vẫn lọt qua sàn"


def test_doc_phan_bo_dich_san_cao_hon_gioi_han_fade_cua_scaper():
    """Sàn phải trên 20 ms — tổng fade vào + fade ra của Scaper."""
    assert sg.TOI_THIEU_SU_KIEN_SEC > 0.02


def test_moi_thoi_luong_rut_ra_deu_dung_duoc_voi_scaper(tmp_path):
    """Bất biến cuối: không giá trị nào rút ra có thể làm Scaper vỡ."""
    path = _viet_segments(tmp_path, {"siren": [(0.0, d) for d in (0.001, 0.03, 0.06, 2.0, 9.0)]})
    phan_bo = sg.doc_phan_bo_dich(path, loai_tru_ytid=set())
    rng = random.Random(0)

    for _ in range(500):
        assert sg.rut_thoi_luong(rng, phan_bo, "siren", 10.0) >= sg.TOI_THIEU_SU_KIEN_SEC
