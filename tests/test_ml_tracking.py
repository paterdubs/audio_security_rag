"""Test cho Pha 1 — vân tay dữ liệu và sổ ghi lần chạy.

Vân tay tồn tại để trả lời "hai run này có cùng dữ liệu không". Một vân tay nhạy quá
(đổi khi dữ liệu không đổi) thì bị bỏ qua sau vài lần báo động giả; một vân tay điếc
(không đổi khi audio đổi) thì tệ hơn cả không có, vì nó CHỨNG THỰC một điều sai.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.tracking.fingerprint import (  # noqa: E402
    _chuan_hoa_nhan,
    bam_file,
    thong_ke_do_dai,
    van_tay_chia_tap,
    van_tay_du_lieu,
    van_tay_ma,
)
from ml.tracking.run_manifest import doc_hop_dong, ghi, ghi_epoch  # noqa: E402

CONTRACT_FIELDS = ["file_id", "stage", "reason_code", "detail", "decided_by", "decided_at"]


# ── Vân tay nhãn ─────────────────────────────────────────────────────────────


def test_van_tay_nhan_khong_doi_khi_chi_thu_tu_su_kien_doi():
    """Thứ tự sự kiện trong một clip là chi tiết triển khai của Scaper, không phải dữ
    liệu. Vân tay nhảy theo nó sẽ báo động giả mỗi lần sinh lại cùng một recipe."""
    a = [{"clip_id": "c0", "events": [[1, 2.0, 3.0], [0, 5.0, 6.0]]}]
    b = [{"clip_id": "c0", "events": [[0, 5.0, 6.0], [1, 2.0, 3.0]]}]
    assert _chuan_hoa_nhan(a) == _chuan_hoa_nhan(b)


def test_van_tay_nhan_doi_khi_onset_doi_mot_phan_nghin_giay():
    a = [{"clip_id": "c0", "events": [[0, 2.000, 3.0]]}]
    b = [{"clip_id": "c0", "events": [[0, 2.001, 3.0]]}]
    assert _chuan_hoa_nhan(a) != _chuan_hoa_nhan(b)


def test_van_tay_nhan_doi_khi_doi_lop():
    a = [{"clip_id": "c0", "events": [[0, 2.0, 3.0]]}]
    b = [{"clip_id": "c0", "events": [[1, 2.0, 3.0]]}]
    assert _chuan_hoa_nhan(a) != _chuan_hoa_nhan(b)


# ── Thống kê độ dài — tầng người đọc được ────────────────────────────────────


def test_thong_ke_do_dai_bat_duoc_event_duration_const():
    """Lỗi thật 17/09/2026: `event_duration=("const", 1.0)` làm MỌI sự kiện của MỌI lớp
    kẹt trần ở 1,10 s (1,0 × hệ số kéo 1,1), trong khi nguồn dài tới hàng chục giây.

    Hai tầng băm chỉ nói 'khác lần trước'. Chính bảng này là thứ nói khác ở ĐÂU: p05 và
    max bằng nhau ở mọi lớp là triệu chứng không thể đọc nhầm.
    """
    clips = [{"clip_id": f"c{i}", "events": [[0, 0.0, 1.10], [1, 2.0, 3.10]]}
             for i in range(50)]
    thong_ke = thong_ke_do_dai(clips, ["gunshot", "siren"])

    assert thong_ke["siren"]["max"] == 1.10
    assert thong_ke["siren"]["p05"] == thong_ke["siren"]["max"]


def test_lop_vang_mat_ghi_n_events_bang_khong_chu_khong_bien_mat():
    """Lớp không có sự kiện nào phải HIỆN ra trong manifest. Vắng mặt khỏi bảng trông
    giống 'chưa kiểm' chứ không giống 'đã kiểm, bằng 0'."""
    thong_ke = thong_ke_do_dai([{"clip_id": "c0", "events": [[0, 0.0, 1.0]]}], ["a", "b"])
    assert thong_ke["b"] == {"n_events": 0}


# ── Vân tay waveform và meta ─────────────────────────────────────────────────


def _viet_features(tmp_path: Path, clips: list[dict], waves: np.ndarray) -> Path:
    features = tmp_path / "features"
    features.mkdir()
    (features / "train_meta.json").write_text(json.dumps({
        "split": "train", "sample_rate": 32000, "duration": 10.0,
        "class_ids": ["gunshot", "siren"], "clips": clips,
    }, ensure_ascii=False), encoding="utf-8")
    np.save(features / "train_wave32k.npy", waves)
    return features


def test_van_tay_doi_khi_audio_doi_du_nhan_y_nguyen():
    """Yêu cầu R2 nói rõ: đếm clip hoặc băm nhãn KHÔNG đủ. Lô 17/09 và lô B0–B9 có cùng
    7.920 clip và cùng 15 lớp — thứ phân biệt chúng nằm trong audio."""
    import tempfile

    clips = [{"clip_id": "c0", "events": [[0, 1.0, 2.0]]}]
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        a = van_tay_du_lieu(_viet_features(Path(d1), clips, np.zeros((1, 100), np.int16)), "train")
        b = van_tay_du_lieu(_viet_features(Path(d2), clips, np.ones((1, 100), np.int16)), "train")

    assert a["nhan_sha256"] == b["nhan_sha256"]
    assert a["waveform_sha256"] != b["waveform_sha256"]


def test_bo_bam_waveform_thi_ghi_null_chu_khong_im_lang(tmp_path):
    """`waveform_sha256: null` nói thẳng rằng vân tay lần này KHÔNG phát hiện được audio
    đổi. Bỏ trường đi sẽ khiến người đọc tưởng đã kiểm."""
    features = _viet_features(tmp_path, [{"clip_id": "c0", "events": []}],
                              np.zeros((1, 100), np.int16))
    assert van_tay_du_lieu(features, "train", bam_waveform=False)["waveform_sha256"] is None


# ── Vân tay mã nguồn và phép chia ────────────────────────────────────────────


def test_van_tay_ma_doi_khi_mot_file_py_doi(tmp_path):
    goc = tmp_path / "ml"
    goc.mkdir()
    (goc / "a.py").write_text("x = 1", encoding="utf-8")
    truoc = van_tay_ma(tmp_path, ("ml",))
    (goc / "a.py").write_text("x = 2", encoding="utf-8")

    assert van_tay_ma(tmp_path, ("ml",))["tree_sha256"] != truoc["tree_sha256"]


def test_van_tay_ma_bo_qua_pycache(tmp_path):
    goc = tmp_path / "ml"
    (goc / "__pycache__").mkdir(parents=True)
    (goc / "a.py").write_text("x = 1", encoding="utf-8")
    (goc / "__pycache__" / "a.py").write_text("rác", encoding="utf-8")

    assert van_tay_ma(tmp_path, ("ml",))["n_files"] == 1


def test_van_tay_chia_tap_khong_phu_thuoc_thu_tu_xao():
    """Khẳng định 'tập val không đổi' phải đúng với TẬP, không phải với thứ tự xáo."""
    a = van_tay_chia_tap(np.array([3, 1, 2]), np.array([5, 4]))
    b = van_tay_chia_tap(np.array([1, 2, 3]), np.array([4, 5]))
    assert a == b


# ── Hợp đồng dữ liệu ─────────────────────────────────────────────────────────


def _viet_hop_dong(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTRACT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_thieu_bao_cao_la_KHONG_RO_chu_khong_phai_PASSED(tmp_path):
    """'Chưa ai kiểm' và 'đã kiểm, sạch' là hai trạng thái khác nhau. Gộp chúng lại chính
    là cách một lô dữ liệu hỏng đi lọt vào train mà manifest vẫn xanh."""
    assert doc_hop_dong(tmp_path / "khong-ton-tai.csv")["ket_qua"] == "KHONG_RO"


def test_bao_cao_chi_co_header_la_PASSED(tmp_path):
    path = tmp_path / "hd.csv"
    _viet_hop_dong(path, [])
    ket_qua = doc_hop_dong(path)
    assert ket_qua["ket_qua"] == "PASSED"
    assert ket_qua["n_vi_pham"] == 0


def test_bao_cao_co_vi_pham_la_FAILED(tmp_path):
    path = tmp_path / "hd.csv"
    _viet_hop_dong(path, [{"file_id": "train_000000", "stage": "verify_synthetic",
                           "reason_code": "slice_long_event", "detail": "dài nhất 1.10s < 8.0s",
                           "decided_by": "auto", "decided_at": "2026-09-17T05:24:22+00:00"}])
    ket_qua = doc_hop_dong(path)
    assert ket_qua["ket_qua"] == "FAILED"
    assert ket_qua["n_vi_pham"] == 1
    assert ket_qua["sha256"] == bam_file(path)


# ── Ghi manifest ─────────────────────────────────────────────────────────────


def test_manifest_co_mat_ngay_chu_khong_doi_cuoi_run(tmp_path):
    """Cùng bài học với `segments.jsonl` ghi ở cuối `run()`: 375 file .wav trên đĩa với
    đúng 3 dòng metadata. Một run 25 epoch bị ngắt ở epoch 18 mà không manifest thì
    checkpoint còn đó cũng vô dụng."""
    manifest = {"run": "thu", "timing": {"bat_dau": "t0", "ket_thuc": None, "epochs": []}}
    ghi(tmp_path, manifest)
    assert (tmp_path / "manifest.json").exists()

    ghi_epoch(tmp_path, manifest, {"epoch": 1, "ket_thuc": "t1", "seconds": 12.0})
    tren_dia = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert tren_dia["timing"]["epochs"] == [{"epoch": 1, "ket_thuc": "t1", "seconds": 12.0}]
    assert tren_dia["timing"]["ket_thuc"] == "t1"


def test_hoi_cuu_khong_muon_hop_dong_cua_lo_khac(tmp_path, monkeypatch):
    """Lỗi thật 19/09/2026, bắt được ngay lần chạy đầu của chính module này.

    `hoi_cuu` mặc định đọc `data/manifests/synthetic_contract.csv`. File đó hiện ghi lô
    B0-B9 PASSED. Manifest hồi cứu cho v1/v2 — train tren lo legacy da FAILED — vi the
    khai PASSED: dung loai loi khai sai ma Pha 1 sinh ra de loai bo. Khong co bao cao cua
    chinh lo da train thi phai la KHONG_RO.
    """
    from ml.tracking import run_manifest

    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "history.json").write_text('{"args": {"split": "train"}, "history": []}',
                                          encoding="utf-8")
    hop_dong_lo_khac = tmp_path / "hd.csv"
    _viet_hop_dong(hop_dong_lo_khac, [])          # 0 vi pham = PASSED

    monkeypatch.setattr(run_manifest, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(run_manifest, "CONTRACT_PATH", hop_dong_lo_khac)

    assert run_manifest.hoi_cuu("r1", "khong-con", pip_freeze=False) == 0
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["data"]["hop_dong"]["ket_qua"] == "KHONG_RO"
    assert any("KHONG_RO" in c for c in manifest["canh_bao"])
