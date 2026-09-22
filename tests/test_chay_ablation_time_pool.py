"""Test cho runner ablation time_pool_blocks chạy không người trông.

Anh em của `test_chay_ablation_pos_weight.py`: cùng rủi ro (chạy qua đêm, lỗi chỉ lộ ra
sáng hôm sau khi đã mất giờ GPU), khác biến ablation. Hai chỗ nguy hiểm mới so với runner
`pos_weight`:

1. `doc_ket_qua` giờ phải đọc thêm khối `long_event` từ `analysis.json` (`lat_cat`), vì
   MỤC ĐÍCH của ablation này là so tỉ lệ phân mảnh `long_event` giữa hai mức
   `time_pool_blocks` — không phải ECE/F1 tổng như ablation `pos_weight`. Thiếu `n_ref`
   hoặc `n_ref=0` phải trả `ti_le=None`, không phải chia-cho-0 hay 0.
2. `xay_lenh_train` phải đặt `--pos-weight-max 1.0` CỐ ĐỊNH (bài học đã rút ra từ ablation
   trước) và chỉ đổi `--time-pool-blocks` — trộn hai biến trong cùng lượt sẽ làm không thể
   quy kết chênh lệch fragmentation cho biến nào.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from chay_ablation_time_pool import (  # noqa: E402
    bang_tom_tat,
    doc_ket_qua,
    doc_long_event,
    train_xong,
    xay_lenh_train,
)


def dung_run(thu_muc: Path, *, ece=None, theta=None, f1=None, f1_05=None,
             time_pool_blocks=None, pos_weight_max=None,
             le_phan_manh=None, le_n_ref=None, le_f1=None) -> None:
    thu_muc.mkdir(parents=True, exist_ok=True)
    if ece is not None:
        (thu_muc / "calibration.json").write_text(
            json.dumps({"ece_tong": ece}), encoding="utf-8")
    if theta is not None:
        (thu_muc / "analysis.json").write_text(json.dumps({
            "theta_sao": theta, "f1_sao": f1, "f1_05": f1_05,
            "lat_cat": [{"lat_cat": "long_event", "phan_manh": le_phan_manh,
                         "n_ref": le_n_ref, "event_f1": le_f1}],
        }), encoding="utf-8")
    if time_pool_blocks is not None or pos_weight_max is not None:
        (thu_muc / "manifest.json").write_text(json.dumps({
            "config": {"args": {"time_pool_blocks": time_pool_blocks,
                                 "pos_weight_max": pos_weight_max}}}), encoding="utf-8")


# ── Đọc kết quả `long_event` — mục đích thật của ablation này ────────────────


def test_doc_long_event_tinh_ti_le_tu_phan_manh_va_n_ref():
    ana = {"lat_cat": [{"lat_cat": "long_event", "phan_manh": 167, "n_ref": 559,
                        "event_f1": 0.3182}]}
    kq = doc_long_event(ana)
    assert kq["phan_manh"] == 167
    assert kq["n_ref"] == 559
    assert abs(kq["ti_le"] - 167 / 559) < 1e-9


def test_doc_long_event_n_ref_khong_co_tra_none_khong_phai_0():
    """analysis.json chưa sinh (run mới train xong) → không có key 'lat_cat' → CHƯA ĐO."""
    kq = doc_long_event({})
    assert kq["ti_le"] is None
    assert kq["phan_manh"] is None


def test_doc_long_event_n_ref_bang_0_tra_none_khong_chia_cho_0():
    ana = {"lat_cat": [{"lat_cat": "long_event", "phan_manh": 0, "n_ref": 0,
                        "event_f1": None}]}
    kq = doc_long_event(ana)
    assert kq["ti_le"] is None


def test_doc_ket_qua_gom_ca_long_event_lan_cau_hinh(tmp_path):
    dung_run(tmp_path / "r", ece=0.02, theta=0.35, f1=0.43, f1_05=0.41,
              time_pool_blocks=3, pos_weight_max=1.0,
              le_phan_manh=167, le_n_ref=559, le_f1=0.3182)
    kq = doc_ket_qua(tmp_path / "r")
    assert kq["time_pool_blocks"] == 3
    assert kq["pos_weight_max"] == 1.0
    assert abs(kq["long_event"]["ti_le"] - 167 / 559) < 1e-9


def test_doc_ket_qua_thieu_file_tra_none_chu_khong_phai_0(tmp_path):
    kq = doc_ket_qua(tmp_path / "chua_co")
    assert kq["time_pool_blocks"] is None
    assert kq["long_event"]["ti_le"] is None


# ── Lệnh train phải khoá pos_weight_max=1.0, chỉ đổi time_pool_blocks ────────


def test_xay_lenh_train_khoa_pos_weight_chi_doi_time_pool_blocks():
    lenh = xay_lenh_train("panns_ft_v4", 3, tiep_tuc=False)
    assert "--pos-weight-max" in lenh
    i = lenh.index("--pos-weight-max")
    assert lenh[i + 1] == "1.0"
    j = lenh.index("--time-pool-blocks")
    assert lenh[j + 1] == "3"
    assert "--resume" not in lenh
    assert lenh[lenh.index("--name") + 1] == "panns_ft_v4"


def test_xay_lenh_train_them_resume_khi_tiep_tuc():
    lenh = xay_lenh_train("panns_ft_tpb2", 2, tiep_tuc=True)
    assert "--resume" in lenh


# ── Cổng "train xong chưa" — copy đúng bài học từ ablation pos_weight ────────


def test_train_xong_khi_du_so_epoch(tmp_path):
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({"timing": {"epochs": [{"epoch": i} for i in range(1, 26)]}}),
                 encoding="utf-8")
    assert train_xong(tmp_path, 25) is True


def test_train_chua_xong_khi_thieu_epoch(tmp_path):
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({"timing": {"epochs": [{"epoch": i} for i in range(1, 10)]}}),
                 encoding="utf-8")
    assert train_xong(tmp_path, 25) is False


# ── Bảng tổng kết ────────────────────────────────────────────────────────────


def test_bang_tom_tat_in_ti_le_phan_manh_long_event_va_danh_dau_chua_do():
    ket_qua = {
        "panns_ft_v4": {"time_pool_blocks": 3, "ece_tong": 0.0231, "theta_sao": 0.35,
                        "f1_sao": 0.4333, "f1_05": 0.416,
                        "long_event": {"ti_le": 167 / 559, "event_f1": 0.3182}},
        "panns_ft_tpb2": {"time_pool_blocks": None, "ece_tong": None, "theta_sao": None,
                          "f1_sao": None, "f1_05": None,
                          "long_event": {"ti_le": None, "event_f1": None}},
    }
    dong = list(bang_tom_tat(ket_qua))
    noi_dung = "\n".join(dong)
    assert "panns_ft_v4" in noi_dung
    assert "0,2988" in noi_dung or "0,2987" in noi_dung  # 167/559, dấu phẩy thập phân
    assert "chưa đo" in noi_dung
