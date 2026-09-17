"""Test cho scripts/verify_synthetic.py — cổng chất lượng của dữ liệu tổng hợp.

Cổng này là thứ duy nhất đọc lại audio/nhãn THẬT và đối chiếu với từng điều khoản
trong `scaper_train.yaml`. Một điều khoản nó bỏ sót là một điều khoản không ai gác.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import verify_synthetic as vs  # noqa: E402


def _sk(label: str, onset: float, offset: float) -> dict:
    return {"label": label, "onset": onset, "offset": offset, "snr": 10.0}


def test_kiem_tran_mep_bo_qua_clip_sach():
    events = [_sk("siren", 0.0, 4.0), _sk("gunshot", 8.0, 9.5)]

    assert vs.kiem_tran_mep(events, 10.0) is None


def test_kiem_tran_mep_cham_dung_moc_cuoi_van_hop_le():
    """Kết thúc đúng 10.000s là vừa khít, không phải tràn."""
    assert vs.kiem_tran_mep([_sk("siren", 6.0, 10.0)], 10.0) is None


def test_kiem_tran_mep_bat_duoc_su_kien_vuot_moc():
    """Lô 17/09 có 1.225 sự kiện như thế này và không có gì báo."""
    chi_tiet = vs.kiem_tran_mep([_sk("siren", 9.5, 11.2)], 10.0)

    assert chi_tiet is not None
    assert "siren" in chi_tiet and "11.2" in chi_tiet


def test_kiem_tran_mep_bao_su_kien_TE_NHAT():
    """Nhiều sự kiện tràn thì phải chỉ ra cái tệ nhất, không phải cái đầu tiên."""
    chi_tiet = vs.kiem_tran_mep(
        [_sk("gunshot", 9.9, 10.4), _sk("siren", 9.0, 13.5), _sk("scream", 9.8, 10.1)], 10.0)

    assert "3 sự kiện" in chi_tiet and "siren" in chi_tiet
