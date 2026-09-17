"""Test cho scripts/measure_distributions.py — STATUS §7 B0.

Công cụ này là thứ duy nhất đọc ngược lại file nhãn ĐÃ SINH và đối chiếu với miền
đích, nên nó phải đúng hơn mọi thứ nó đo. Một báo cáo sai ở đây không làm hỏng dữ
liệu — nó làm hỏng kết luận, mà kết luận thì đi thẳng vào khoá luận.

Trọng tâm: bảng đối chiếu phải quy về CÙNG MIX LỚP. AudioSet lệch nặng về
`speech_normal` (42.5% số sự kiện) và `running_footsteps` (13.8%), cả hai đều là lớp
sự kiện ngắn; dữ liệu tổng hợp thì cân bằng đều 15 lớp. Ở mốc 4 giây, đích đọc ra
8.9% theo mix tự nhiên nhưng 15.5% theo mix đều. Nhìn nhầm cột là kết luận ngược.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import measure_distributions as md  # noqa: E402


# ── Hợp các khoảng ───────────────────────────────────────────────────────────


def test_hop_khoang_gop_phan_chong_lan_chu_khong_cong_don():
    """Hai sự kiện chồng nhau hoàn toàn phủ 1 giây, không phải 2.

    Cộng dồn sẽ thổi phồng độ phủ sóng đúng ở những clip chồng lấn — mà chồng lấn
    lại là thứ ta cố ý ép lên 30%.
    """
    assert md.hop_khoang([(0.0, 1.0), (0.0, 1.0)]) == pytest.approx(1.0)


def test_hop_khoang_cac_ca_co_ban():
    assert md.hop_khoang([]) == 0.0
    assert md.hop_khoang([(0.0, 2.0), (5.0, 6.0)]) == pytest.approx(3.0)      # rời nhau
    assert md.hop_khoang([(0.0, 2.0), (1.0, 4.0)]) == pytest.approx(4.0)      # chồng một phần
    assert md.hop_khoang([(1.0, 9.0), (3.0, 4.0)]) == pytest.approx(8.0)      # lồng trọn
    assert md.hop_khoang([(5.0, 6.0), (0.0, 2.0)]) == pytest.approx(3.0)      # không sắp sẵn


# ── Quy về cùng mix lớp ──────────────────────────────────────────────────────


def test_trong_so_mix_deu_cho_moi_lop_dong_gop_bang_nhau():
    """Lớp 1000 sự kiện và lớp 10 sự kiện phải nặng như nhau."""
    gia_tri, trong_so = md._trong_so_mix_deu({"nhieu": [1.0] * 1000, "it": [9.0] * 10})

    assert trong_so.sum() == pytest.approx(1.0)
    assert trong_so[gia_tri == 1.0].sum() == pytest.approx(0.5)
    assert trong_so[gia_tri == 9.0].sum() == pytest.approx(0.5)


def test_ty_le_su_kien_dai_doi_han_khi_quy_ve_mix_deu():
    """Chính ca làm lệch kết luận: lớp dài thì hiếm, lớp ngắn thì đông.

    Mix tự nhiên cho 10/1010 ≈ 1.0% sự kiện dài; mix đều cho 50%. Khác biệt này là
    lý do bảng đối chiếu phải có cột thứ hai.
    """
    gia_tri, trong_so = md._trong_so_mix_deu({"ngan": [0.5] * 1000, "dai": [9.0] * 10})

    tu_nhien = float(np.mean(gia_tri >= 4.0))
    mix_deu = float(trong_so[gia_tri >= 4.0].sum())

    assert tu_nhien == pytest.approx(10 / 1010, abs=0.002)
    assert mix_deu == pytest.approx(0.5)


def test_phan_vi_co_trong_so_bang_phan_vi_thuong_khi_trong_so_deu():
    """Trọng số đều thì phải trùng numpy — nếu không, công thức nội suy sai."""
    x = np.arange(1.0, 101.0)
    w = np.full(100, 1 / 100)

    for p in (5, 25, 50, 75, 95):
        assert md._phan_vi_co_trong_so(x, w, p) == pytest.approx(np.percentile(x, p), abs=1.0)


def test_phan_vi_co_trong_so_bi_keo_theo_lop_duoc_can_nang():
    """Cân nặng lớp dài lên thì trung vị phải dịch về phía dài."""
    gia_tri, trong_so = md._trong_so_mix_deu({"ngan": [1.0] * 900, "dai": [9.0] * 100})

    assert md._phan_vi_co_trong_so(gia_tri, trong_so, 50) > 1.0


# ── Bảng đối chiếu ───────────────────────────────────────────────────────────


def test_bang_doi_chieu_hien_ca_hai_cot_dich():
    """Thiếu cột mix đều là mời một kết luận sai vào khoá luận."""
    synth = {"a": [1.0] * 50, "b": [5.0] * 50}
    gold = {"a": [1.0] * 900, "b": [5.0] * 100}

    bang = "\n".join(md.bang_doi_chieu(synth, [0.2] * 100, 100, gold, [0.5] * 1000, 1000))

    assert "mix tự nhiên" in bang and "mix đều" in bang
    assert "speech_normal" in bang, "thiếu lời giải thích vì sao có hai cột"
