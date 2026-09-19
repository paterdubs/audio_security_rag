"""Test cho Pha 5 — bảng đối chiếu nhiều lần chạy.

Công cụ so sánh nguy hiểm hơn công cụ đo: nó đứng ở cuối chuỗi và người đọc tin thẳng
vào cột "khác nhau ở đâu". Ba lời khai sai mà bảng này có thể đưa ra, cả ba đều đã có
sẵn dữ liệu để đưa ra ngay hôm nay:

1. v1/v2 không có vân tay dữ liệu (audio legacy bị xoá 18/09) → "giống nhau" là bịa.
2. Cả ba manifest lập HỒI CỨU cùng một lúc nên băm mã trùng nhau → "cùng mã nguồn" là
   bịa; ta chỉ biết mã ở thời điểm lập manifest.
3. `ket_qua_cuoi` trong manifest chấm trên val split RIÊNG của từng run. Xếp hạng theo
   nó ra v2 > v1 > v3, ngược hẳn dev chung (v3 > v2 > v1).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.tracking.compare_runs import (  # noqa: E402
    KHONG_RO,
    bang_metric,
    diff_config,
    diff_ma,
    diff_van_tay,
    doc_run,
)


def lam_run(tmp_path: Path, ten: str, *, van_tay, args: dict, hoi_cuu: bool = True,
            tree: str = "aaa", ket_qua: dict | None = None,
            analysis: dict | None = None) -> Path:
    run_dir = tmp_path / ten
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "run": ten, "hoi_cuu": hoi_cuu,
        "config": {"args": args},
        "data": {"split": "train", "van_tay": van_tay},
        "code": {"tree_sha256": tree},
        "ket_qua_cuoi": ket_qua or {"event_f1": 0.1, "segment_f1": 0.3, "clip_map": 0.8},
    }, ensure_ascii=False), encoding="utf-8")
    if analysis is not None:
        (run_dir / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False),
                                               encoding="utf-8")
    return run_dir


# ── Vân tay dữ liệu ──────────────────────────────────────────────────────────


def test_van_tay_null_la_khong_ro_chu_khong_phai_giong_nhau(tmp_path):
    """Audio legacy của v1/v2 đã bị xoá 18/09, vân tay hai run đó là null VĨNH VIỄN.
    So `None == None` rồi kết luận "cùng dữ liệu" là đúng kiểu lỗi run_manifest từng mắc
    (khai PASSED khi thiếu file báo cáo hợp đồng)."""
    a = doc_run(lam_run(tmp_path, "v1", van_tay=None, args={"lr": 1e-3}))
    b = doc_run(lam_run(tmp_path, "v2", van_tay=None, args={"lr": 1e-3}))

    kq = diff_van_tay([a, b])
    assert kq["ket_luan"] == KHONG_RO
    assert kq["theo_run"] == {"v1": KHONG_RO, "v2": KHONG_RO}


def test_mot_run_co_van_tay_mot_run_khong_van_la_khong_ro(tmp_path):
    """v3 có vân tay, v2 thì không. Biết một nửa không đủ kết luận "khác nhau" — cũng
    không đủ kết luận "giống nhau". Chỉ `KHONG_RO` là lời khai đúng."""
    a = doc_run(lam_run(tmp_path, "v2", van_tay=None, args={}))
    b = doc_run(lam_run(tmp_path, "v3", van_tay={"nhan_sha256": "abc"}, args={}))

    kq = diff_van_tay([a, b])
    assert kq["ket_luan"] == KHONG_RO
    assert kq["theo_run"]["v3"] == "abc"


def test_hai_van_tay_day_du_thi_moi_ket_luan_giong_hay_khac(tmp_path):
    """Khi cả hai run đều khai vân tay thì mới được phép nói giống/khác."""
    a = doc_run(lam_run(tmp_path, "x", van_tay={"nhan_sha256": "abc"}, args={}))
    b = doc_run(lam_run(tmp_path, "y", van_tay={"nhan_sha256": "abc"}, args={}))
    c = doc_run(lam_run(tmp_path, "z", van_tay={"nhan_sha256": "xyz"}, args={}))

    assert diff_van_tay([a, b])["ket_luan"] == "giong"
    assert diff_van_tay([a, c])["ket_luan"] == "khac"


# ── Vân tay mã nguồn ─────────────────────────────────────────────────────────


def test_manifest_hoi_cuu_thi_bam_ma_la_khong_ro(tmp_path):
    """Ba manifest của v1/v2/v3 đều `hoi_cuu: true` và được lập trong cùng một phút, nên
    `code.tree_sha256` của chúng TRÙNG NHAU. Băm đó là mã lúc lập manifest, không phải mã
    lúc train. Khai "cùng mã nguồn" từ đó là chứng thực một điều không ai biết."""
    a = doc_run(lam_run(tmp_path, "v1", van_tay=None, args={}, hoi_cuu=True, tree="same"))
    b = doc_run(lam_run(tmp_path, "v3", van_tay=None, args={}, hoi_cuu=True, tree="same"))

    kq = diff_ma([a, b])
    assert kq["ket_luan"] == KHONG_RO
    assert "hồi cứu" in kq["ly_do"].lower()


def test_manifest_ghi_luc_train_thi_bam_ma_dung_duoc(tmp_path):
    """Run ghi manifest ngay lúc train thì băm mã là bằng chứng thật."""
    a = doc_run(lam_run(tmp_path, "v4", van_tay=None, args={}, hoi_cuu=False, tree="same"))
    b = doc_run(lam_run(tmp_path, "v5", van_tay=None, args={}, hoi_cuu=False, tree="same"))

    assert diff_ma([a, b])["ket_luan"] == "giong"


# ── Diff cấu hình ────────────────────────────────────────────────────────────


def test_co_thieu_khac_han_co_gia_tri_khac(tmp_path):
    """v1 KHÔNG CÓ khoá `mixup_alpha` vì cờ đó ra đời sau v1; v2 có và đặt 0.2. Gộp hai
    trường hợp thành "khác giá trị" làm mất thông tin quan trọng nhất: v1 chạy bằng một
    phiên bản code khác, nên nó không so ngang được chứ không chỉ là chỉnh tham số."""
    a = doc_run(lam_run(tmp_path, "v1", van_tay=None, args={"lr": 1e-3}))
    b = doc_run(lam_run(tmp_path, "v2", van_tay=None, args={"lr": 1e-3, "mixup_alpha": 0.2}))
    c = doc_run(lam_run(tmp_path, "v3", van_tay=None, args={"lr": 5e-4, "mixup_alpha": 0.2}))

    theo_khoa = {h["khoa"]: h for h in diff_config([a, b, c])}
    assert theo_khoa["mixup_alpha"]["gia_tri"]["v1"] == "<không có>"
    assert theo_khoa["lr"]["gia_tri"]["v3"] == "0.0005"
    assert "name" not in theo_khoa  # tên run khác nhau là hiển nhiên, không phải diff


def test_config_giong_het_nhau_van_phai_bao_cho_nguoi_doc_biet(tmp_path):
    """v2 và v3 khác nhau ĐÚNG mỗi `name`: thứ phân biệt chúng là dữ liệu, mà vân tay của
    v2 lại null. Trả bảng rỗng khiến người đọc kết luận "cùng cấu hình, khác kết quả →
    khác kiến trúc". Phải nói rõ là không có khác biệt nào ĐƯỢC GHI LẠI."""
    a = doc_run(lam_run(tmp_path, "v2", van_tay=None, args={"lr": 1e-3, "name": "v2"}))
    b = doc_run(lam_run(tmp_path, "v3", van_tay={"nhan_sha256": "z"},
                        args={"lr": 1e-3, "name": "v3"}))

    assert diff_config([a, b]) == []
    assert diff_van_tay([a, b])["ket_luan"] == KHONG_RO


# ── Metric ───────────────────────────────────────────────────────────────────


def test_metric_tu_manifest_phai_deo_nhan_val_rieng(tmp_path):
    """`ket_qua_cuoi` chấm trên val split riêng của từng run (val_ratio=0.1 trên chính
    tập train của nó). Số thật: 0,1282 / 0,1897 / 0,1077 → xếp v2 > v1 > v3, NGƯỢC hẳn
    dev chung (0,2953 / 0,3518 / 0,3979). Bảng không đeo nhãn thì xếp hạng lộn ngược."""
    a = doc_run(lam_run(tmp_path, "v1", van_tay=None, args={},
                        ket_qua={"event_f1": 0.1282, "segment_f1": 0.412, "clip_map": 0.83}))
    b = doc_run(lam_run(tmp_path, "v2", van_tay=None, args={},
                        ket_qua={"event_f1": 0.1897, "segment_f1": 0.298, "clip_map": 0.81}))

    bang = bang_metric([a, b])
    assert bang["nguon"] == "val_rieng"
    assert not bang["so_ngang_duoc"]
    assert "val" in bang["canh_bao"].lower()


def test_co_analysis_thi_uu_tien_dev_chung_va_in_kem_theta(tmp_path):
    """analysis.json chấm cả ba run trên CÙNG data/synthetic/dev nên so ngang được — với
    điều kiện in kèm θ, vì F1 ở θ=0,50 và ở θ* chênh nhau gấp ba lần."""
    a = doc_run(lam_run(tmp_path, "v1", van_tay=None, args={},
                        analysis={"f1_sao": 0.2953, "seg_sao": 0.41, "theta_sao": 0.95,
                                  "f1_05": 0.1147, "split": "dev", "subset": "all"}))
    b = doc_run(lam_run(tmp_path, "v3", van_tay=None, args={},
                        analysis={"f1_sao": 0.3979, "seg_sao": 0.44, "theta_sao": 0.90,
                                  "f1_05": 0.0820, "split": "dev", "subset": "all"}))

    bang = bang_metric([a, b])
    assert bang["nguon"] == "dev_chung"
    assert bang["so_ngang_duoc"]
    assert bang["hang"][0]["theta_sao"] == 0.95


def test_tron_run_co_va_khong_co_analysis_thi_ha_xuong_val_rieng(tmp_path):
    """Một run chưa chạy Pha 4 thì không có gì để so trên dev chung. Lấy dev chung cho
    run này, val riêng cho run kia rồi xếp chung một bảng là so hai thứ khác nhau."""
    a = doc_run(lam_run(tmp_path, "v1", van_tay=None, args={},
                        analysis={"f1_sao": 0.2953, "seg_sao": 0.41, "theta_sao": 0.95,
                                  "f1_05": 0.1147, "split": "dev", "subset": "all"}))
    b = doc_run(lam_run(tmp_path, "vx", van_tay=None, args={}))

    bang = bang_metric([a, b])
    assert bang["nguon"] == "val_rieng"
    assert not bang["so_ngang_duoc"]


def test_run_thieu_manifest_bao_loi_chu_khong_bo_qua(tmp_path):
    """Bỏ qua im lặng một run không có manifest thì bảng vẫn in ra đẹp đẽ với ít cột hơn,
    và không ai để ý run bị thiếu."""
    (tmp_path / "trong").mkdir()
    try:
        doc_run(tmp_path / "trong")
    except FileNotFoundError as e:
        assert "manifest.json" in str(e)
    else:
        raise AssertionError("phải ném FileNotFoundError")
