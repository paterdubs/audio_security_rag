"""Test cho runner ablation pos_weight chạy không người trông.

Runner này chạy qua đêm, nên mọi lỗi của nó chỉ bị phát hiện sáng hôm sau khi đã mất
nhiều giờ GPU. Hai chỗ nguy hiểm nhất đều thuần logic nên test được:

1. `doc_ket_qua` gom số từ ba file json khác nhau của một run — thiếu file phải trả None
   (CHƯA ĐO) chứ không phải 0, đúng kỷ luật đã dặn: "F1=0 vì chưa đo bị đọc thành F1=0".
2. `train_xong` quyết định có khởi động lượt kế tiếp hay không. Trả True sai một lần là
   lượt sau train trên checkpoint dở dang; trả False sai là chạy chồng hai tiến trình
   train trên cùng một GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from chay_ablation_pos_weight import (  # noqa: E402
    bang_tom_tat,
    doc_ket_qua,
    train_xong,
)


def dung_run(thu_muc: Path, *, ece=None, theta=None, f1=None, pos_weight=None) -> None:
    thu_muc.mkdir(parents=True, exist_ok=True)
    if ece is not None:
        (thu_muc / "calibration.json").write_text(
            json.dumps({"ece_tong": ece}), encoding="utf-8")
    if theta is not None:
        (thu_muc / "analysis.json").write_text(
            json.dumps({"theta_sao": theta, "f1_sao": f1, "f1_05": 0.1}), encoding="utf-8")
    if pos_weight is not None:
        (thu_muc / "manifest.json").write_text(
            json.dumps({"config": {"args": {"pos_weight_max": pos_weight}}}), encoding="utf-8")


# ── Gom kết quả ──────────────────────────────────────────────────────────────


def test_doc_ket_qua_gom_du_ba_nguon(tmp_path):
    dung_run(tmp_path / "r", ece=0.0231, theta=0.35, f1=0.4333, pos_weight=1.0)
    kq = doc_ket_qua(tmp_path / "r")
    assert kq["ece_tong"] == 0.0231
    assert kq["theta_sao"] == 0.35
    assert kq["f1_sao"] == 0.4333
    assert kq["pos_weight_max"] == 1.0


def test_doc_ket_qua_thieu_file_tra_none_chu_khong_phai_0(tmp_path):
    """Run mới train xong nhưng chưa chạy calibration: ECE phải là CHƯA ĐO. Trả 0 ở đây
    sẽ làm bảng tổng kết khoe một run hiệu chuẩn hoàn hảo."""
    dung_run(tmp_path / "r", theta=0.35, f1=0.4333)
    kq = doc_ket_qua(tmp_path / "r")
    assert kq["ece_tong"] is None
    assert kq["theta_sao"] == 0.35


def test_doc_ket_qua_thu_muc_khong_ton_tai_khong_no(tmp_path):
    kq = doc_ket_qua(tmp_path / "chua_co")
    assert kq["ece_tong"] is None and kq["theta_sao"] is None


# ── Cổng "train xong chưa" ───────────────────────────────────────────────────


def test_train_xong_khi_du_so_epoch(tmp_path):
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({"timing": {"epochs": [{"epoch": i} for i in range(1, 26)]}}),
                 encoding="utf-8")
    assert train_xong(tmp_path, 25) is True


def test_train_chua_xong_khi_thieu_epoch(tmp_path):
    """Đây là ca nguy hiểm: trả True sớm thì lượt sau train trên checkpoint dở dang."""
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({"timing": {"epochs": [{"epoch": i} for i in range(1, 20)]}}),
                 encoding="utf-8")
    assert train_xong(tmp_path, 25) is False


def test_train_chua_xong_khi_chua_co_manifest(tmp_path):
    assert train_xong(tmp_path / "trong", 25) is False


# ── Bảng tổng kết ────────────────────────────────────────────────────────────


def test_bang_tom_tat_danh_dau_chua_do(tmp_path):
    ket_qua = {
        "panns_ft_pw1": {"pos_weight_max": 1.0, "ece_tong": 0.0231,
                         "theta_sao": 0.35, "f1_sao": 0.4333, "f1_05": 0.416},
        "panns_ft_pw10": {"pos_weight_max": 10.0, "ece_tong": None,
                          "theta_sao": None, "f1_sao": None, "f1_05": None},
    }
    dong = list(bang_tom_tat(ket_qua))
    noi_dung = "\n".join(dong)
    assert "panns_ft_pw1" in noi_dung
    assert "0,0231" in noi_dung          # dấu phẩy thập phân, quy ước repo
    assert "chưa đo" in noi_dung         # không được in 0 cho ô trống
