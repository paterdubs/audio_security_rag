"""Test cho scripts/make_blind_set.py — bộ gán lại mù pilot lần 2, DATA_PLAN §8.3 bước 2.

§8.4 đòi 4 điều kiện mù; script chỉ chịu trách nhiệm cho điều kiện 2 (băm tên + xáo thứ
tự) và điều kiện 4 (trộn clip mồi). Điều kiện 1 (dự án Label Studio mới) và 3 (nghỉ ≥3
ngày) là thao tác của người, không phải code.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_blind_set import (  # noqa: E402
    chon_moi,
    dung_anh_xa,
    ghi_anh_xa,
    hash_ten,
    run,
    sao_chep_file,
    xep_lich_mu,
)


def thong_tin(*ids: str) -> dict[str, dict]:
    return {i: {"path": f"data/raw/audioset_strong/audio/{i}.wav", "classes": ["speech_normal"]}
            for i in ids}


def test_hash_ten_deterministic():
    """Cùng seed + file_id → cùng tên băm qua hai lần gọi — không có tính chất này thì
    lần 2 không copy lại đúng cùng bộ tên mỗi lần chạy `run()` lại (ví dụ khi cần sửa
    danh sách mồi và chạy lại script)."""
    a = hash_ten("seed1", "as_strong_abc_30000")
    b = hash_ten("seed1", "as_strong_abc_30000")
    assert a == b


def test_hash_ten_khac_id_khac_ten():
    a = hash_ten("seed1", "as_strong_abc_30000")
    b = hash_ten("seed1", "as_strong_xyz_10000")
    assert a != b


def test_hash_ten_khong_lo_file_id():
    """Tên băm không được chứa nguyên văn file_id gốc — lộ file_id qua tên file là lộ
    luôn manh mối 'đây là clip nào', phá điều kiện mù dù đã băm."""
    file_id = "as_strong_-glN59TmfME_30000"
    ten = hash_ten("seed1", file_id)
    assert file_id not in ten
    assert "glN59TmfME" not in ten


def test_hash_ten_co_duoi_wav():
    assert hash_ten("seed1", "as_strong_abc_30000").endswith(".wav")


def test_chon_moi_loai_da_dung():
    """Mồi không bao giờ trùng với clip đã dùng (30 clip pilot) — trùng thì cùng một
    clip vừa là pilot vừa là mồi, vô nghĩa."""
    pool = thong_tin(*[f"c{i}" for i in range(10)])
    da_dung = {"c0", "c1", "c2"}
    moi = chon_moi(pool, da_dung, n=5, seed="s")
    assert set(moi).isdisjoint(da_dung)
    assert len(moi) == 5


def test_chon_moi_deterministic():
    pool = thong_tin(*[f"c{i}" for i in range(20)])
    a = chon_moi(pool, set(), n=5, seed="s")
    b = chon_moi(pool, set(), n=5, seed="s")
    assert a == b


def test_chon_moi_khong_du_pool_khong_loi():
    """Pool nhỏ hơn n cầu — trả về hết pool còn lại, không lỗi, không lặp vô hạn. Cùng
    loại bẫy đã gặp ở chon_pilot() với lớp hiếm hết file trước các lớp khác."""
    pool = thong_tin("c0", "c1", "c2")
    moi = chon_moi(pool, set(), n=10, seed="s")
    assert set(moi) == {"c0", "c1", "c2"}


def test_xep_lich_mu_la_hoan_vi():
    """Kết quả là một hoán vị đúng của pilot ∪ mồi — không rơi mất, không nhân đôi."""
    pilot = [f"p{i}" for i in range(5)]
    moi = [f"m{i}" for i in range(5)]
    thu_tu = xep_lich_mu(pilot, moi, seed="s")
    assert set(thu_tu) == set(pilot) | set(moi)
    assert len(thu_tu) == 10


def test_xep_lich_mu_deterministic():
    pilot = [f"p{i}" for i in range(5)]
    moi = [f"m{i}" for i in range(5)]
    a = xep_lich_mu(pilot, moi, seed="s")
    b = xep_lich_mu(pilot, moi, seed="s")
    assert a == b


def test_xep_lich_mu_thuc_su_xao_tron():
    """Thứ tự đầu ra không được giữ nguyên khối pilot đứng liền nhau ở đầu — nếu vậy
    người gán vẫn nhận ra '10 clip đầu là block cũ', phá điều kiện mù dù đã băm tên."""
    pilot = [f"p{i}" for i in range(15)]
    moi = [f"m{i}" for i in range(15)]
    thu_tu = xep_lich_mu(pilot, moi, seed="s")
    assert thu_tu != pilot + moi
    assert thu_tu[:15] != pilot


def test_dung_anh_xa_gan_dung_co_la_pilot():
    pilot = ["p0", "p1"]
    moi = ["m0"]
    thu_tu = xep_lich_mu(pilot, moi, seed="s")
    anh_xa = dung_anh_xa(thu_tu, set(pilot), seed="s")
    theo_id = {r["file_id"]: r for r in anh_xa}
    assert theo_id["p0"]["la_pilot"] is True
    assert theo_id["p1"]["la_pilot"] is True
    assert theo_id["m0"]["la_pilot"] is False


def test_dung_anh_xa_ten_mu_duy_nhat():
    """Không hai file_id nào được gán cùng một tên băm trong một lượt chạy — trùng tên
    thì copy file sau đè lên file trước, mất một clip mà không có triệu chứng nào."""
    ids = [f"c{i}" for i in range(50)]
    anh_xa = dung_anh_xa(ids, set(), seed="s")
    ten_list = [r["ten_mu"] for r in anh_xa]
    assert len(ten_list) == len(set(ten_list))


def test_sao_chep_file_copy_dung_ten(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "a.wav").write_bytes(b"RIFF-fake-audio")
    thong_tin_map = {"a": {"path": "src/a.wav"}}
    anh_xa = [{"file_id": "a", "ten_mu": "blind_xyz.wav", "la_pilot": True}]
    out_dir = tmp_path / "out"
    sao_chep_file(anh_xa, thong_tin_map, out_dir, goc=tmp_path)
    assert (out_dir / "blind_xyz.wav").read_bytes() == b"RIFF-fake-audio"


def test_ghi_anh_xa_ghi_dung_cot(tmp_path):
    anh_xa = [
        {"file_id": "p0", "ten_mu": "blind_a.wav", "la_pilot": True},
        {"file_id": "m0", "ten_mu": "blind_b.wav", "la_pilot": False},
    ]
    out = tmp_path / "mapping.csv"
    ghi_anh_xa(anh_xa, out)
    with out.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {r["ten_mu"] for r in rows} == {"blind_a.wav", "blind_b.wav"}
    assert {r["file_id"] for r in rows} == {"p0", "m0"}


def test_run_tao_du_60_clip_va_mapping(tmp_path, monkeypatch):
    """Kiểm tra end-to-end run(): 30 pilot + 30 mồi từ pool giả → thư mục out_dir có đủ
    60 file, mapping.csv có đủ 60 dòng, không trùng tên."""
    repo = tmp_path
    (repo / "data" / "raw" / "audioset_strong" / "audio").mkdir(parents=True)
    (repo / "data" / "manifests").mkdir(parents=True)
    (repo / "data" / "gold").mkdir(parents=True)

    pilot_ids = [f"as_strong_p{i}_10000" for i in range(30)]
    pool_ids = pilot_ids + [f"as_strong_m{i}_10000" for i in range(40)]

    for fid in pool_ids:
        (repo / "data" / "raw" / "audioset_strong" / "audio" / f"{fid}.wav").write_bytes(b"x")

    segments_path = repo / "data" / "raw" / "audioset_strong" / "segments.jsonl"
    with segments_path.open("w", encoding="utf-8") as handle:
        import json
        for fid in pool_ids:
            handle.write(json.dumps({
                "file_id": fid, "path": f"data/raw/audioset_strong/audio/{fid}.wav",
                "ytid": fid, "events": [{"class_id": "speech_normal", "onset": 0.0, "offset": 1.0}],
            }) + "\n")

    splits_path = repo / "data" / "manifests" / "splits.csv"
    with splits_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_id", "source_dataset", "source_group_id", "split"])
        for fid in pool_ids:
            writer.writerow([fid, "audioset_strong", f"youtube_{fid}", "gold_test"])

    pilot_path = repo / "data" / "gold" / "pilot_v1_candidates.csv"
    with pilot_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_id", "path", "classes_tham_khao"])
        for fid in pilot_ids:
            writer.writerow([fid, f"data/raw/audioset_strong/audio/{fid}.wav", "speech_normal"])

    out_dir = repo / "data" / "gold" / "blind_v1_lan2"
    mapping_out = repo / "data" / "gold" / "blind_v1_lan2_mapping.csv"

    import argparse
    args = argparse.Namespace(
        pilot=pilot_path, n_moi=30, seed="test-seed",
        splits=splits_path, segments=segments_path,
        raw_manifest=repo / "data" / "manifests" / "raw_manifest.csv",
        exclusions=repo / "data" / "manifests" / "exclusions.csv",
        out_dir=out_dir, mapping_out=mapping_out, repo_root=repo,
    )
    ma = run(args)
    assert ma == 0
    assert len(list(out_dir.glob("*.wav"))) == 60
    with mapping_out.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 60
    assert sum(1 for r in rows if r["la_pilot"] == "True") == 30


def test_run_tru_them_loai_moi_da_dung_o_vong_truoc(tmp_path):
    """Vòng pilot SAU lần đầu: nếu quên --tru-them, mồi có thể trùng đúng clip vòng
    trước mà người gán đã nghe — điều kiện mù §8.4 hỏng dù tên file đã băm lại. Ca này
    dựng pool chỉ đủ 30 pilot mới + 30 clip 'đã dùng ở vòng trước' + 10 mồi thật sự mới,
    rồi kiểm KHÔNG có mồi nào trùng danh sách đã dùng."""
    repo = tmp_path
    (repo / "data" / "raw" / "audioset_strong" / "audio").mkdir(parents=True)
    (repo / "data" / "manifests").mkdir(parents=True)
    (repo / "data" / "gold").mkdir(parents=True)

    pilot_ids = [f"as_strong_p{i}_10000" for i in range(30)]
    da_dung_truoc = [f"as_strong_cu{i}_10000" for i in range(30)]
    moi_that = [f"as_strong_moi{i}_10000" for i in range(10)]
    pool_ids = pilot_ids + da_dung_truoc + moi_that

    for fid in pool_ids:
        (repo / "data" / "raw" / "audioset_strong" / "audio" / f"{fid}.wav").write_bytes(b"x")

    segments_path = repo / "data" / "raw" / "audioset_strong" / "segments.jsonl"
    with segments_path.open("w", encoding="utf-8") as handle:
        import json
        for fid in pool_ids:
            handle.write(json.dumps({
                "file_id": fid, "path": f"data/raw/audioset_strong/audio/{fid}.wav",
                "ytid": fid, "events": [{"class_id": "speech_normal", "onset": 0.0, "offset": 1.0}],
            }) + "\n")

    splits_path = repo / "data" / "manifests" / "splits.csv"
    with splits_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_id", "source_dataset", "source_group_id", "split"])
        for fid in pool_ids:
            writer.writerow([fid, "audioset_strong", f"youtube_{fid}", "gold_test"])

    pilot_path = repo / "data" / "gold" / "pilot_v3_candidates.csv"
    with pilot_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_id", "path", "classes_tham_khao"])
        for fid in pilot_ids:
            writer.writerow([fid, f"data/raw/audioset_strong/audio/{fid}.wav", "speech_normal"])

    tru_them_path = repo / "data" / "gold" / "da_dung_truoc.csv"
    with tru_them_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_id"])
        for fid in da_dung_truoc:
            writer.writerow([fid])

    out_dir = repo / "data" / "gold" / "blind_v3_lan2"
    mapping_out = repo / "data" / "gold" / "blind_v3_lan2_mapping.csv"

    import argparse
    args = argparse.Namespace(
        pilot=pilot_path, n_moi=10, seed="test-seed-v3",
        splits=splits_path, segments=segments_path,
        raw_manifest=repo / "data" / "manifests" / "raw_manifest.csv",
        exclusions=repo / "data" / "manifests" / "exclusions.csv",
        tru_them=tru_them_path,
        out_dir=out_dir, mapping_out=mapping_out, repo_root=repo,
    )
    ma = run(args)
    assert ma == 0
    with mapping_out.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    moi_da_chon = {r["file_id"] for r in rows if r["la_pilot"] == "False"}
    assert moi_da_chon == set(moi_that)
    assert moi_da_chon.isdisjoint(da_dung_truoc)
