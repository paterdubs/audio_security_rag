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


# ── Ghi segments.jsonl NGAY từng dòng, không đè kết quả cũ ──────────────────


def test_them_dong_moi_khong_xoa_dong_cu(tmp_path, monkeypatch):
    monkeypatch.setattr(fas, "RAW_ROOT", tmp_path)
    monkeypatch.setattr(fas, "SEGMENTS_PATH", tmp_path / "segments.jsonl")

    seen = fas.load_existing_segment_ids()
    fas.append_one_segment({"file_id": "a", "events": []}, seen)
    fas.append_one_segment({"file_id": "b", "events": []}, seen)

    with fas.SEGMENTS_PATH.open(encoding="utf-8") as handle:
        ids = [json.loads(line)["file_id"] for line in handle]
    assert ids == ["a", "b"]


def test_them_dong_da_co_khong_bi_nhan_doi(tmp_path, monkeypatch):
    monkeypatch.setattr(fas, "RAW_ROOT", tmp_path)
    monkeypatch.setattr(fas, "SEGMENTS_PATH", tmp_path / "segments.jsonl")

    seen = fas.load_existing_segment_ids()
    fas.append_one_segment({"file_id": "a", "events": []}, seen)
    fas.append_one_segment({"file_id": "a", "events": []}, seen)

    with fas.SEGMENTS_PATH.open(encoding="utf-8") as handle:
        assert len(handle.readlines()) == 1


def test_khong_nhan_doi_ke_ca_qua_lan_goi_load_lai(tmp_path, monkeypatch):
    """Mô phỏng đúng kịch bản đã gây mất dữ liệu: tiến trình bị giết giữa `run()`, rồi
    chạy lại — `load_existing_segment_ids()` đọc lại từ đĩa phải thấy dòng đã ghi trước đó."""
    monkeypatch.setattr(fas, "RAW_ROOT", tmp_path)
    monkeypatch.setattr(fas, "SEGMENTS_PATH", tmp_path / "segments.jsonl")

    seen_1 = fas.load_existing_segment_ids()
    fas.append_one_segment({"file_id": "a", "events": []}, seen_1)
    # "Tiến trình mới" — nạp lại seen từ đĩa, không dùng chung biến seen_1 trong RAM.
    seen_2 = fas.load_existing_segment_ids()
    fas.append_one_segment({"file_id": "a", "events": []}, seen_2)

    with fas.SEGMENTS_PATH.open(encoding="utf-8") as handle:
        assert len(handle.readlines()) == 1


def test_ghi_ngay_khong_doi_toi_cuoi(tmp_path, monkeypatch):
    """Đây chính là bug đã xảy ra thật (17/09/2026): bản cũ dồn kết quả vào RAM (`new_rows`)
    rồi chỉ ghi một lần ở cuối `run()`. Nếu tiến trình chết giữa đường (mất điện, crash,
    bị kill), mọi file đã tải thành công KHÔNG có dòng nào trong segments.jsonl trỏ tới —
    375 file .wav trên đĩa mà chỉ 3 dòng ghi lại. Test này khẳng định dòng ghi xong ngay
    sau lệnh gọi, không phụ thuộc bước nào tiếp theo."""
    monkeypatch.setattr(fas, "RAW_ROOT", tmp_path)
    monkeypatch.setattr(fas, "SEGMENTS_PATH", tmp_path / "segments.jsonl")

    seen = fas.load_existing_segment_ids()
    fas.append_one_segment({"file_id": "a", "events": []}, seen)

    # Đọc lại file NGAY, giả lập một tiến trình khác (hoặc lần chạy lại sau crash) mở
    # file trong khi tiến trình ghi còn "sống" — nếu dữ liệu còn kẹt trong buffer chưa
    # flush, dòng này sẽ không thấy được.
    with fas.SEGMENTS_PATH.open(encoding="utf-8") as handle:
        assert len(handle.readlines()) == 1


# ── Tải audio: gọi đúng cửa sổ, không tải nguyên video ──────────────────────


class _FakeInfo:
    def __init__(self, duration):
        self.duration = duration


def _fake_download_ok(cmd, timeout):
    """Giả lập yt-dlp chạy thành công: ghi file đích rồi trả mã 0."""
    Path(cmd[cmd.index("-o") + 1].replace(".%(ext)s", ".wav")).write_bytes(b"RIFF")
    return 0


def test_tai_audio_goi_dung_khoang_thoi_gian(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, timeout):
        captured["cmd"], captured["timeout"] = cmd, timeout
        return _fake_download_ok(cmd, timeout)

    monkeypatch.setattr(fas, "run_with_hard_timeout", fake_run)
    monkeypatch.setattr(fas.soundfile, "info", lambda path: _FakeInfo(10.0))
    dest = tmp_path / "clip.wav"
    ok = fas.download_window("abc123", 5000, dest)

    assert ok
    section = captured["cmd"][captured["cmd"].index("--download-sections") + 1]
    assert section == "*5.0-15.0"


def test_ep_keyframe_de_cat_dung_vi_tri(tmp_path, monkeypatch):
    """Không có --force-keyframes-at-cuts, ffmpeg cắt tại keyframe gần nhất — đã đo
    thật một clip ra 19.994s thay vì 10s, làm nhãn onset/offset lệch khỏi audio."""
    captured = {}

    def fake_run(cmd, timeout):
        captured["cmd"] = cmd
        return _fake_download_ok(cmd, timeout)

    monkeypatch.setattr(fas, "run_with_hard_timeout", fake_run)
    monkeypatch.setattr(fas.soundfile, "info", lambda path: _FakeInfo(10.0))
    fas.download_window("abc123", 0, tmp_path / "clip.wav")
    assert "--force-keyframes-at-cuts" in captured["cmd"]


def test_co_tran_thoi_gian_cho_moi_clip(tmp_path, monkeypatch):
    """Thiếu trần thời gian thì một video treo khoá cả job vĩnh viễn — đã xảy ra thật
    17/09/2026 (job đứng 56 phút). Đây là lỗi im lặng nên phải có test canh."""
    captured = {}

    def fake_run(cmd, timeout):
        captured["timeout"] = timeout
        return _fake_download_ok(cmd, timeout)

    monkeypatch.setattr(fas, "run_with_hard_timeout", fake_run)
    monkeypatch.setattr(fas.soundfile, "info", lambda path: _FakeInfo(10.0))
    fas.download_window("abc123", 0, tmp_path / "clip.wav")

    assert captured.get("timeout"), "phải truyền trần thời gian xuống"
    assert captured["timeout"] <= 600, "trần quá lớn thì coi như không có trần"


def test_qua_han_thi_bo_file_do_va_tra_ve_false(tmp_path, monkeypatch):
    """File tải dở vẫn `exists()`, nên lần chạy sau sẽ bỏ qua nó và nhận một clip hỏng
    vào manifest. Phải xoá hẳn."""
    dest = tmp_path / "clip.wav"

    def fake_run(cmd, timeout):
        dest.write_bytes(b"RIFF....phan tai do")
        return None      # None = quá hạn

    monkeypatch.setattr(fas, "run_with_hard_timeout", fake_run)

    assert not fas.download_window("abc123", 0, dest)
    assert not dest.exists()


# ── Trần thời gian phải THOÁT RA ĐƯỢC, không chỉ tồn tại ────────────────────


def test_khong_mo_pipe_vi_pipe_khoa_duong_thoat_cua_timeout(monkeypatch):
    """Bản vá TRƯỚC (subprocess.run + capture_output + timeout) KHÔNG đủ: job vẫn treo
    34,7 phút với ffmpeg chỉ tốn 0,64 s CPU.

    Cơ chế: yt-dlp sinh ffmpeg làm tiến trình CHÁU, cháu thừa hưởng hai đầu pipe. Hết giờ,
    Python giết yt-dlp rồi communicate() chờ pipe ĐÓNG, mà pipe chỉ đóng khi ffmpeg chết —
    chính đường thoát của timeout bị khoá. Điều kiện cần để thoát được: không mở pipe nào.
    """
    ghi_nhan = {}

    class _FakePopen:
        def __init__(self, cmd, **kwargs):
            ghi_nhan.update(kwargs)
            self.pid = 1

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(fas.subprocess, "Popen", _FakePopen)
    fas.run_with_hard_timeout(["yt-dlp"], 10)

    assert ghi_nhan.get("stdout") is fas.subprocess.DEVNULL
    assert ghi_nhan.get("stderr") is fas.subprocess.DEVNULL


def test_qua_han_thi_giet_ca_cay_roi_bao_none(monkeypatch):
    """Hết giờ phải (a) giết cả cây tiến trình và (b) TRẢ VỀ ĐƯỢC, không treo ở bước dọn."""
    da_giet = []

    class _FakeHangingPopen:
        def __init__(self, cmd, **kwargs):
            self.pid = 4242
            self.lan_goi = 0

        def wait(self, timeout=None):
            # Lần đầu: treo quá hạn. Lần sau (sau khi bị giết): trả về ngay.
            self.lan_goi += 1
            if self.lan_goi == 1:
                raise fas.subprocess.TimeoutExpired("yt-dlp", timeout)
            return -9

    monkeypatch.setattr(fas.subprocess, "Popen", _FakeHangingPopen)
    monkeypatch.setattr(fas, "kill_process_tree", da_giet.append)

    assert fas.run_with_hard_timeout(["yt-dlp"], 1) is None
    assert da_giet == [4242]


def test_giet_ca_cay_tien_trinh_tren_windows(monkeypatch):
    """Giết mỗi tiến trình cha để lại ffmpeg mồ côi, và nó vẫn giữ file đích."""
    goi = {}
    monkeypatch.setattr(fas.sys, "platform", "win32")
    monkeypatch.setattr(fas.subprocess, "run", lambda cmd, **kw: goi.setdefault("cmd", cmd))

    fas.kill_process_tree(4242)

    assert goi["cmd"][:1] == ["taskkill"]
    assert "/T" in goi["cmd"], "thiếu /T thì tiến trình cháu sống sót"
    assert "4242" in goi["cmd"]


def test_tai_audio_that_bai_tra_ve_false(tmp_path, monkeypatch):
    monkeypatch.setattr(fas, "run_with_hard_timeout", lambda cmd, timeout: 1)
    ok = fas.download_window("abc123", 0, tmp_path / "clip.wav")
    assert not ok


def test_do_dai_lech_qua_xa_10s_thi_loai(tmp_path, monkeypatch):
    """Exit code 0 không đảm bảo cắt đúng vị trí — chỉ đảm bảo ffmpeg không crash.
    Phải tự đo lại độ dài thật, không tin exit code."""
    monkeypatch.setattr(fas, "run_with_hard_timeout", _fake_download_ok)
    monkeypatch.setattr(fas.soundfile, "info", lambda path: _FakeInfo(19.994))
    dest = tmp_path / "clip.wav"
    ok = fas.download_window("abc123", 200000, dest)
    assert not ok
    assert not dest.exists()      # file sai độ dài phải bị xoá, không để lại rác


def test_do_dai_gan_dung_van_duoc_giu(tmp_path, monkeypatch):
    """Cho phép lệch nhỏ do làm tròn container, không đòi đúng tuyệt đối 10.000s."""
    monkeypatch.setattr(fas, "run_with_hard_timeout", _fake_download_ok)
    monkeypatch.setattr(fas.soundfile, "info", lambda path: _FakeInfo(10.008))
    dest = tmp_path / "clip.wav"
    assert fas.download_window("abc123", 0, dest)
    assert dest.exists()


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
