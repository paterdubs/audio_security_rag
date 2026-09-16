"""Test cho scripts/fetch_audioset_strong.py.

Nguồn này khác biệt: một ổ 10 giây có thể chứa NHIỀU sự kiện, và audio KHÔNG được
phát hành lại. Các lỗi im lặng đáng lo nhất là parse sai segment_id (ytid có thể
chứa dấu gạch dưới) và lẫn nhãn UNCERTAIN vào ground truth dùng để tính điểm model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fetch_audioset_strong as fas  # noqa: E402


# ── Tách segment_id ──────────────────────────────────────────────────────────


def test_tach_ytid_thuong():
    assert fas.parse_segment_id("abc123_5000") == ("abc123", 5000)


def test_ytid_chua_dau_gach_duoi_khong_bi_cat_nham():
    """YTID của YouTube có thể chứa '_' — phải tách từ bên PHẢI, không phải bên trái."""
    assert fas.parse_segment_id("a_b_c_20000") == ("a_b_c", 20000)


# ── Ánh xạ mid → class, giới hạn theo vocab strong thật có ──────────────────


def test_chi_giu_mid_co_trong_vocab_strong():
    ontology = {"classes": {
        "gunshot": {"audioset_ids": ["/m/032s66", "/m/04zjc"]},
        "vehicle_crash": {"audioset_ids": ["/m/07rknqz"]},  # weak-only, KHÔNG có trong strong
    }}
    mapping = fas.mid_to_class_map(ontology, known_mids={"/m/032s66"})
    assert mapping == {"/m/032s66": "gunshot"}


# ── Chọn segment ─────────────────────────────────────────────────────────────
#
# File TSV thật của AudioSet strong CHỈ có 4 cột (segment_id, start_time_seconds,
# end_time_seconds, label) — không có cột present/uncertain. Đã kiểm tra thật 16/09
# sau khi select_segments() luôn trả rỗng vì giả định sai có cột đó.


def _row(segment_id, label, start="1.0", end="3.0"):
    return {"segment_id": segment_id, "label": label,
            "start_time_seconds": start, "end_time_seconds": end}


def test_gop_nhieu_su_kien_cung_mot_o():
    rows = [
        _row("v1_0", "/m/032s66", "1.0", "2.0"),
        _row("v1_0", "/m/07p6fty", "4.0", "6.0"),
    ]
    mid_to_class = {"/m/032s66": "gunshot", "/m/07p6fty": "shout_yell"}
    segments = fas.select_segments(rows, mid_to_class)
    assert len(segments[("v1", 0)]) == 2
    classes = {e["class_id"] for e in segments[("v1", 0)]}
    assert classes == {"gunshot", "shout_yell"}


def test_mid_khong_khop_lop_nao_thi_bo_qua():
    rows = [_row("v1_0", "/m/lung_tung")]
    segments = fas.select_segments(rows, {"/m/032s66": "gunshot"})
    assert segments == {}


# ── Ghi segments.jsonl, không đè kết quả cũ ─────────────────────────────────


def test_them_dong_moi_khong_xoa_dong_cu(tmp_path, monkeypatch):
    monkeypatch.setattr(fas, "RAW_ROOT", tmp_path)
    monkeypatch.setattr(fas, "SEGMENTS_PATH", tmp_path / "segments.jsonl")

    fas.append_segments([{"file_id": "a", "events": []}])
    fas.append_segments([{"file_id": "b", "events": []}])

    with fas.SEGMENTS_PATH.open(encoding="utf-8") as handle:
        ids = [json.loads(line)["file_id"] for line in handle]
    assert ids == ["a", "b"]


def test_them_dong_da_co_khong_bi_nhan_doi(tmp_path, monkeypatch):
    monkeypatch.setattr(fas, "RAW_ROOT", tmp_path)
    monkeypatch.setattr(fas, "SEGMENTS_PATH", tmp_path / "segments.jsonl")

    fas.append_segments([{"file_id": "a", "events": []}])
    fas.append_segments([{"file_id": "a", "events": []}])

    with fas.SEGMENTS_PATH.open(encoding="utf-8") as handle:
        assert len(handle.readlines()) == 1


# ── Tải audio: gọi đúng cửa sổ, không tải nguyên video ──────────────────────


def test_tai_audio_goi_dung_khoang_thoi_gian(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        dest = Path(cmd[cmd.index("-o") + 1].replace(".%(ext)s", ".wav"))
        dest.write_bytes(b"RIFF")

        class Result:
            returncode = 0
        return Result()

    monkeypatch.setattr(fas.subprocess, "run", fake_run)
    dest = tmp_path / "clip.wav"
    ok = fas.download_window("abc123", 5000, dest)

    assert ok
    section = captured["cmd"][captured["cmd"].index("--download-sections") + 1]
    assert section == "*5.0-15.0"


def test_tai_audio_that_bai_tra_ve_false(tmp_path, monkeypatch):
    def fake_run(cmd, **kwargs):
        class Result:
            returncode = 1
        return Result()

    monkeypatch.setattr(fas.subprocess, "run", fake_run)
    ok = fas.download_window("abc123", 0, tmp_path / "clip.wav")
    assert not ok


# ── Đọc vocab mid → display name ────────────────────────────────────────────


def test_doc_vocab_bo_qua_dong_khong_phai_mid(tmp_path):
    path = tmp_path / "vocab.tsv"
    path.write_text("MID\tdisplay_name\n/m/032s66\tGunshot, gunfire\n", encoding="utf-8")
    mapping = fas.load_mid_to_display_name(path)
    assert mapping == {"/m/032s66": "Gunshot, gunfire"}


# ── Chọn mẫu phân tầng theo lớp ──────────────────────────────────────────────


def test_lop_hiem_khong_bi_lop_dong_lan_at():
    """66229 ổ tổng nhưng lệch rất mạnh (speech_normal >> fireworks) — lấy N ổ đầu
    theo thứ tự bất kỳ sẽ toàn speech_normal, bỏ đói lớp hiếm."""
    segments = {(f"v{i}", 0): [{"class_id": "speech_normal"}] for i in range(1000)}
    segments[("rare", 0)] = [{"class_id": "fireworks"}]
    keys = fas.stratified_keys(segments, per_class_limit=5)
    assert ("rare", 0) in keys


def test_moi_lop_toi_da_per_class_limit_o():
    segments = {(f"v{i}", 0): [{"class_id": "siren"}] for i in range(50)}
    keys = fas.stratified_keys(segments, per_class_limit=10)
    assert len(keys) == 10


def test_mot_o_thoa_nhieu_lop_khong_bi_dem_hai_lan():
    segments = {("v1", 0): [{"class_id": "siren"}, {"class_id": "alarm_bell"}]}
    keys = fas.stratified_keys(segments, per_class_limit=5)
    assert keys == [("v1", 0)]
