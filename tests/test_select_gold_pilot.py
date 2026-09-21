"""Test cho `select_gold_pilot.py` — chọn N clip pilot cho gold_test, rải đều theo lớp.

Mục đích: pilot 30 clip (DATA_PLAN §8.3 bước 1) phải phủ đủ 15 lớp để phát hiện lỗ hổng
guideline sớm — chọn ngẫu nhiên thuần tuý có thể ra 30 clip toàn `speech_normal` (lớp có
nhiều file nhất, 210/367) mà bỏ sót `door_slam` (chỉ 11 file) hoàn toàn.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from select_gold_pilot import chon_pilot  # noqa: E402


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
