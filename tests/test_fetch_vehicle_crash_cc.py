"""Test cho scripts/fetch_vehicle_crash_cc.py.

Lỗi im lặng đáng lo nhất với nguồn này: quên dùng `video_id` làm `source_group_id`.
Vài video góp nhiều clip (hậu tố `_00`, `_01`... trong `file_name`) — chia theo file
tên khác nhau nhưng cùng video là rò rỉ (N2), y hệt lỗi đã bắt được ở DESED và FSD50K.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fetch_vehicle_crash_cc as fcc  # noqa: E402


def test_tai_clip_khong_tai_lai_neu_da_co(tmp_path, monkeypatch):
    monkeypatch.setattr(fcc, "RAW_DIR", tmp_path)
    dest = tmp_path / "clips" / "abc_00.wav"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"da co san")

    called = []
    monkeypatch.setattr(fcc.urllib.request, "urlopen", lambda *a, **k: called.append(a) or None)

    result = fcc.download_clip("clips/abc_00.wav")
    assert result == dest
    assert called == []          # không gọi mạng vì file đã tồn tại
    assert dest.read_bytes() == b"da co san"


def test_tai_clip_moi_thi_goi_mang(tmp_path, monkeypatch):
    monkeypatch.setattr(fcc, "RAW_DIR", tmp_path)

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"RIFF..."

    captured = {}

    def fake_urlopen(url, timeout=60):
        captured["url"] = url
        return FakeResponse()

    monkeypatch.setattr(fcc.urllib.request, "urlopen", fake_urlopen)
    dest = fcc.download_clip("clips/xyz_00.wav")

    assert dest.read_bytes() == b"RIFF..."
    assert captured["url"] == fcc.BASE_URL + "clips/xyz_00.wav"


def test_download_metadata_ghi_file_va_parse_dung_cot(tmp_path, monkeypatch):
    monkeypatch.setattr(fcc, "RAW_DIR", tmp_path)
    monkeypatch.setattr(fcc, "METADATA_PATH", tmp_path / "metadata.csv")

    csv_text = (
        "file_name,label,start_sec,end_sec,duration_sec,title,channel,video_id,source_url,license\n"
        "clips/abc_00.wav,Car Crash,0.0,9.96,9.96,Some Title,Some Channel,abc,"
        "https://youtube.com/watch?v=abc,Creative Commons Attribution license (reuse allowed)\n"
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return csv_text.encode("utf-8")

    monkeypatch.setattr(fcc.urllib.request, "urlopen", lambda *a, **k: FakeResponse())

    rows = fcc.download_metadata()
    assert len(rows) == 1
    assert rows[0]["video_id"] == "abc"
    assert rows[0]["file_name"] == "clips/abc_00.wav"
    assert fcc.METADATA_PATH.read_text(encoding="utf-8") == csv_text


def test_hai_clip_cung_video_duoc_dem_mot_lan(tmp_path, monkeypatch, capsys):
    """`n_videos` (in trong báo cáo) phải đếm theo video_id, không theo số clip — nếu
    không sẽ báo sai số video gốc, làm người đọc tưởng đa dạng hơn thực tế."""
    monkeypatch.setattr(fcc, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(fcc, "RAW_DIR", tmp_path)
    monkeypatch.setattr(fcc, "METADATA_PATH", tmp_path / "metadata.csv")

    csv_text = (
        "file_name,label,start_sec,end_sec,duration_sec,title,channel,video_id,source_url,license\n"
        "clips/abc_00.wav,Car Crash,0.0,5.0,5.0,T1,C1,abc,u1,lic\n"
        "clips/abc_01.wav,Car Crash,5.0,10.0,5.0,T1,C1,abc,u1,lic\n"
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return csv_text.encode("utf-8")

    monkeypatch.setattr(fcc.urllib.request, "urlopen", lambda *a, **k: FakeResponse())
    monkeypatch.setattr(fcc, "download_clip", lambda file_name: tmp_path / file_name)

    assert fcc.run() == 0
    assert "1 video gốc" in capsys.readouterr().out
