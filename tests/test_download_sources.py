"""Test cho scripts/download_sources.py.

Ba lỗi dưới đây đều đã XẢY RA THẬT ngày 14/09/2026 và đều thuộc loại tệ nhất: file
tải về hỏng mà không có triệu chứng nào, chỉ lộ ra ở MD5 sau khi đã tốn hàng GB
băng thông — hoặc tệ hơn, không lộ ra nếu nguồn không công bố checksum.
"""

from __future__ import annotations

import http.client
import os
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import download_sources as ds  # noqa: E402


# ── Khoá: chặn hai tiến trình cùng ghi một file ──────────────────────────────


def test_khoa_chan_tien_trinh_thu_hai(tmp_path: Path):
    part = tmp_path / "x.zip.part"
    with ds.DownloadLock(part):
        with pytest.raises(SystemExit) as err:
            with ds.DownloadLock(part):
                pass
    assert str(os.getpid()) in str(err.value)      # nói rõ AI đang giữ khoá


def test_khoa_duoc_tra_lai_sau_khi_xong(tmp_path: Path):
    part = tmp_path / "x.zip.part"
    with ds.DownloadLock(part):
        pass
    with ds.DownloadLock(part):                    # vào lại được
        pass


def test_khoa_duoc_tra_lai_ca_khi_co_loi(tmp_path: Path):
    # 504 của Zenodo làm chết lệnh tải giữa chừng. Không trả khoá thì mọi lần chạy
    # sau đều bị chặn bởi một khoá mồ côi, và cách sửa duy nhất là xoá tay.
    part = tmp_path / "x.zip.part"
    with pytest.raises(RuntimeError):
        with ds.DownloadLock(part):
            raise RuntimeError("mạng đứt")
    with ds.DownloadLock(part):
        pass


def test_hai_file_khac_nhau_khong_chan_nhau(tmp_path: Path):
    with ds.DownloadLock(tmp_path / "a.part"), ds.DownloadLock(tmp_path / "b.part"):
        pass


# ── Phân loại lỗi: cái nào đáng thử lại ──────────────────────────────────────


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_loi_may_chu_tam_thoi_thi_thu_lai(code: int):
    assert ds.is_transient(urllib.error.HTTPError("u", code, "", {}, None))


@pytest.mark.parametrize("code", [401, 403, 404, 410])
def test_loi_thuoc_ve_yeu_cau_thi_khong_thu_lai(code: int):
    # Thử lại 404 tám lần chỉ làm chậm việc báo cho người biết URL đã sai.
    assert not ds.is_transient(urllib.error.HTTPError("u", code, "", {}, None))


@pytest.mark.parametrize("error", [
    urllib.error.URLError("mạng đứt"),
    TimeoutError(),
    ConnectionResetError(),
    http.client.IncompleteRead(b""),
])
def test_su_co_mang_thi_thu_lai(error: Exception):
    assert ds.is_transient(error)


def test_loi_lap_trinh_thi_khong_nuot(tmp_path: Path):
    assert not ds.is_transient(ValueError("lỗi của chính ta"))


# ── Thử lại: phải TẢI TIẾP, không làm lại từ đầu ─────────────────────────────


def test_thu_lai_doc_lai_kich_thuoc_that_cua_part(tmp_path: Path, monkeypatch):
    """Một cú 504 ở phút thứ 40 của file 6 GB không được làm mất 40 phút đó."""
    part = tmp_path / "x.part"
    part.write_bytes(b"a" * 5_000_000)
    offsets = []

    def gia_lap(url, p, done, name):
        offsets.append(done)
        if len(offsets) < 3:
            raise urllib.error.HTTPError(url, 504, "Gateway Time-out", {}, None)
        return done + 100

    monkeypatch.setattr(ds, "fetch_one_pass", gia_lap)
    monkeypatch.setattr(ds.time, "sleep", lambda _: None)

    assert ds.fetch_with_retry("u", part, 5_000_000, "x") == 5_000_100
    # Mọi lần thử đều bắt đầu từ 5 MB đã có, không lần nào quay về 0.
    assert offsets == [5_000_000, 5_000_000, 5_000_000]


def test_het_luot_thu_thi_nem_loi_chu_khong_bao_thanh_cong(tmp_path: Path, monkeypatch):
    def luon_hong(url, p, done, name):
        raise urllib.error.HTTPError(url, 503, "", {}, None)

    monkeypatch.setattr(ds, "fetch_one_pass", luon_hong)
    monkeypatch.setattr(ds.time, "sleep", lambda _: None)
    with pytest.raises(urllib.error.HTTPError):
        ds.fetch_with_retry("u", tmp_path / "x.part", 0, "x")


def test_loi_khong_tam_thoi_thi_dung_ngay_lan_dau(tmp_path: Path, monkeypatch):
    lan = []

    def hong_404(url, p, done, name):
        lan.append(1)
        raise urllib.error.HTTPError(url, 404, "", {}, None)

    monkeypatch.setattr(ds, "fetch_one_pass", hong_404)
    monkeypatch.setattr(ds.time, "sleep", lambda _: None)
    with pytest.raises(urllib.error.HTTPError):
        ds.fetch_with_retry("u", tmp_path / "x.part", 0, "x")
    assert len(lan) == 1


# ── Kiểm tra kích thước THẬT trên đĩa ────────────────────────────────────────


def test_bo_dem_khop_nhung_file_thua_byte_thi_van_bi_bat(tmp_path: Path, monkeypatch):
    """Đúng lỗi FSD50K: bộ đếm báo 2.31 GB khớp Zenodo từng byte, file 2.51 GB.

    So bộ đếm với chính nó thì luôn đúng. Chỉ hỏi hệ thống tệp mới phát hiện được
    tiến trình thứ hai đã nối thêm gì vào file.
    """
    dest = tmp_path / "x.zip"
    part = tmp_path / "x.zip.part"
    part.write_bytes(b"a" * 1500)                  # trên đĩa 1500 byte

    monkeypatch.setattr(ds, "fetch_with_retry", lambda *a: 1000)   # bộ đếm nói 1000
    with pytest.raises(OSError) as err:
        ds._download_locked("u", dest, part, expected_size=1000)
    assert "1500" in str(err.value) and "1000" in str(err.value)
    assert not dest.exists()                       # không được đổi tên file hỏng


def test_kich_thuoc_khop_thi_di_tiep(tmp_path: Path, monkeypatch):
    dest = tmp_path / "x.zip"
    part = tmp_path / "x.zip.part"
    part.write_bytes(b"a" * 1000)
    monkeypatch.setattr(ds, "fetch_with_retry", lambda *a: 1000)
    ds._download_locked("u", dest, part, expected_size=1000)
    assert dest.exists() and not part.exists()


def test_thieu_byte_so_voi_zenodo_thi_bat(tmp_path: Path, monkeypatch):
    dest = tmp_path / "x.zip"
    part = tmp_path / "x.zip.part"
    part.write_bytes(b"a" * 900)
    monkeypatch.setattr(ds, "fetch_with_retry", lambda *a: 900)
    with pytest.raises(OSError):
        ds._download_locked("u", dest, part, expected_size=1000)
