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


# ── Dọn kho nén chia nhiều phần ──────────────────────────────────────────────


def test_xoa_ca_cac_phan_z0x_chu_khong_rieng_file_zip(tmp_path: Path):
    """Xoá mỗi `.zip` là bỏ sót 5 phần + file ghép — 34 GB nằm lì ở riêng FSD50K."""
    archive = tmp_path / "A.dev_audio.zip"
    joined = tmp_path / "A.dev_audio.joined.zip"
    parts = [tmp_path / f"A.dev_audio.z0{i}" for i in range(1, 6)]
    for path in [archive, joined, *parts]:
        path.write_bytes(b"x" * 100)
    khac = tmp_path / "B.khac.zip"           # kho của nguồn khác, KHÔNG được đụng
    khac.write_bytes(b"x" * 100)

    freed = ds.cleanup_archive(archive)

    assert freed == 700                      # 1 zip + 1 joined + 5 phần
    assert not any(p.exists() for p in [archive, joined, *parts])
    assert khac.exists()


def test_xoa_cac_phan_roi_nhung_GIU_file_ghep_cho_buoc_giai_nen(tmp_path: Path):
    """Đỉnh dung lượng thật của FSD50K là 60 GB, không phải 37 GB như config từng ghi.

    Xoá các phần rời ngay sau khi ghép xong kéo đỉnh ấy xuống 42 GB. Đo thật 14/09:
    đĩa tụt còn 25 GB giữa lúc giải nén, suýt không đủ.
    """
    archive = tmp_path / "A.dev_audio.zip"
    joined = tmp_path / "A.dev_audio.joined.zip"
    parts = [tmp_path / f"A.dev_audio.z0{i}" for i in range(1, 6)]
    for path in [archive, joined, *parts]:
        path.write_bytes(b"x" * 100)

    freed = ds.cleanup_archive(archive, keep=joined)

    assert freed == 600                      # 1 zip + 5 phần, KHÔNG tính file ghép
    assert joined.exists()                   # còn cần để giải nén
    assert not any(p.exists() for p in [archive, *parts])


def test_kho_khong_chia_phan_thi_chi_xoa_chinh_no(tmp_path: Path):
    archive = tmp_path / "x.tar.gz"
    archive.write_bytes(b"x" * 50)
    assert ds.cleanup_archive(archive) == 50
    assert not archive.exists()


def test_file_da_bi_xoa_tu_truoc_khong_lam_no_loi(tmp_path: Path):
    archive = tmp_path / "x.zip"
    assert ds.cleanup_archive(archive) == 0


def test_kiem_tra_file_ghep_truoc_khi_xoa_cac_phan(tmp_path: Path):
    """Xoá xong mới phát hiện file ghép hỏng thì phải tải lại 18 GB."""
    import zipfile as zf
    tot = tmp_path / "tot.zip"
    with zf.ZipFile(tot, "w") as handle:
        handle.writestr("a.wav", b"x")
    assert ds.verify_zip_readable(tot) == 1

    hong = tmp_path / "hong.zip"
    hong.write_bytes(b"khong phai zip")
    with pytest.raises(Exception):
        ds.verify_zip_readable(hong)


# ── Tìm công cụ zip ──────────────────────────────────────────────────────────


def test_uu_tien_zip_tren_path(monkeypatch):
    monkeypatch.setattr(ds.shutil, "which", lambda name: "/usr/bin/zip")
    assert ds.find_zip_tool() == "/usr/bin/zip"


def test_tim_duoc_noi_winget_cai_du_khong_co_tren_path(tmp_path: Path, monkeypatch):
    # Bộ cài GnuWin32 KHÔNG thêm vào PATH. Chỉ dựa vào which() sẽ báo "chưa cài" cho
    # một máy đã cài xong — bảo người dùng làm lại đúng việc họ vừa làm.
    gia_lap = tmp_path / "zip.exe"
    gia_lap.write_bytes(b"")
    monkeypatch.setattr(ds.shutil, "which", lambda name: None)
    monkeypatch.setattr(ds, "ZIP_FALLBACKS", [tmp_path / "khong_co.exe", gia_lap])
    assert ds.find_zip_tool() == str(gia_lap)


def test_that_su_chua_cai_thi_bao_cach_cai(monkeypatch):
    monkeypatch.setattr(ds.shutil, "which", lambda name: None)
    monkeypatch.setattr(ds, "ZIP_FALLBACKS", [])
    with pytest.raises(SystemExit) as err:
        ds.find_zip_tool()
    assert "winget" in str(err.value) and "apt install" in str(err.value)


# ── Luồng kết thúc sớm mà không ném lỗi ──────────────────────────────────────


def test_luong_ket_thuc_som_thi_tai_tiep_chu_khong_chet(tmp_path: Path, monkeypatch):
    """FSD50K.dev_audio.z05: thiếu 17 MB trên 3 GB, im lặng hoàn toàn.

    response.read() trả b"" giữa chừng nên không có lỗi nào được ném; mọi thứ trông
    như đã xong. Không có nhánh này thì lỗi chỉ lộ ra ở phép so kích thước cuối cùng,
    ném OSError không-thể-thử-lại và giết cả lệnh tải — dù việc cần làm chỉ là tải tiếp.
    """
    part = tmp_path / "x.part"
    luot = []

    def gia_lap(url, p, done, name):
        luot.append(done)
        moi = [2_000_000, 2_900_000, 3_000_000][len(luot) - 1]
        p.write_bytes(b"a" * moi)
        return moi

    monkeypatch.setattr(ds, "fetch_one_pass", gia_lap)
    monkeypatch.setattr(ds.time, "sleep", lambda _: None)

    assert ds.fetch_with_retry("u", part, 0, "x", expected_size=3_000_000) == 3_000_000
    assert luot == [0, 2_000_000, 2_900_000]      # mỗi lượt tải tiếp, không làm lại


def test_khong_tien_them_duoc_byte_nao_thi_dung(tmp_path: Path, monkeypatch):
    # Thử lại 8 lần một máy chủ luôn trả đúng chừng ấy byte chỉ làm mất thời gian.
    part = tmp_path / "x.part"

    def kep(url, p, done, name):
        p.write_bytes(b"a" * 1000)
        return 1000

    monkeypatch.setattr(ds, "fetch_one_pass", kep)
    monkeypatch.setattr(ds.time, "sleep", lambda _: None)
    with pytest.raises(OSError, match="kẹt"):
        ds.fetch_with_retry("u", part, 0, "x", expected_size=2000)


def test_du_kich_thuoc_thi_tra_ve_ngay(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ds, "fetch_one_pass", lambda *a: 1000)
    assert ds.fetch_with_retry("u", tmp_path / "x.part", 0, "x", expected_size=1000) == 1000


def test_khong_biet_kich_thuoc_dich_thi_khong_kiem_tra(tmp_path: Path, monkeypatch):
    # Nguồn không công bố kích thước: tin vào luồng, không có gì để đối chiếu.
    monkeypatch.setattr(ds, "fetch_one_pass", lambda *a: 500)
    assert ds.fetch_with_retry("u", tmp_path / "x.part", 0, "x", expected_size=0) == 500
