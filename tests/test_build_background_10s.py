"""Test cho scripts/build_background_10s.py — STATUS §7 B2.5.

Nền của bank hiện tại dài 4 giây trong khi clip là 10 giây, nên Scaper lặp vòng
nó 2,5 lần. Đo trên 300 clip không có sự kiện: mức dB từng giây lặp đúng chu kỳ 4
giây, và ở `parking`/`residential` giá trị tại mốc 4 s và 8 s trùng nhau tới từng
chữ số. Đó là dấu vết tổng hợp không tồn tại trong bản ghi thật.

Điều cứu được tình thế nằm ở cách UrbanSound8K cắt lát: các lát cùng một
occurrence chồng nhau ĐÚNG 2.0 giây (đo trên 560 cặp, tương quan 1.00000 tại
2.0 s và ~0.00x ở mọi độ chồng khác). Nghĩa là ghép chúng lại TÁI DỰNG được chính
bản ghi gốc liên tục, chứ không phải nối hai bản ghi khác nhau.

Phân biệt này là toàn bộ giá trị của bước B2.5, nên test nặng nhất ở đây là test
tái dựng phải ra lại đúng tín hiệu gốc từng mẫu một.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_background_10s as bb  # noqa: E402

SR = 16000


def _tin_hieu(giay: float, seed: int = 0) -> np.ndarray:
    """Nhiễu tất định — mỗi mẫu một giá trị khác nhau, nên ghép sai là lộ ngay."""
    return np.random.default_rng(seed).standard_normal(int(giay * SR)).astype(np.float32) * 0.1


def _ghi(path: Path, y: np.ndarray) -> Path:
    """Ghi ở float32, KHÔNG phải PCM_16.

    Bank thật là PCM_16, nhưng test ở đây kiểm thuật toán ghép — mà PCM_16 làm
    tròn tới 1 LSB = 3.05e-5, đủ để một test đòi khớp từng mẫu đỏ vì codec chứ
    không vì lỗi ghép. Tách bạch hai thứ đó ra.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, SR, subtype="FLOAT")
    return path


def _cat_lat(y: np.ndarray, tmp: Path, dai: float = 4.0, buoc: float = 2.0) -> list[Path]:
    """Cắt y thành các lát chồng nhau đúng kiểu UrbanSound8K."""
    n, b = int(dai * SR), int(buoc * SR)
    return [_ghi(tmp / f"lat_{k}.wav", y[i:i + n])
            for k, i in enumerate(range(0, len(y) - n + 1, b))]


# ── Tái dựng bản ghi gốc ─────────────────────────────────────────────────────


def test_tai_dung_khoi_phuc_dung_tin_hieu_goc(tmp_path):
    """Ghép các lát chồng 2 s phải ra lại ĐÚNG tín hiệu gốc, từng mẫu một.

    Đây là test quan trọng nhất file này. Sai một chỗ chồng lấn thì kết quả vẫn là
    audio nghe được, vẫn đúng 10 giây, chỉ có nội dung là lặp hoặc nhảy — loại lỗi
    không có triệu chứng nào ngoài việc model học phải một dấu vết không có thật.
    """
    goc = _tin_hieu(12.0, seed=1)
    lat = _cat_lat(goc, tmp_path)
    assert len(lat) == 5                                # 12s → lát tại 0,2,4,6,8

    dung = bb.tai_dung(lat, SR)

    assert len(dung) == len(goc)
    np.testing.assert_allclose(dung, goc, atol=1e-6)


def test_tai_dung_mot_lat_tra_nguyen_lat_do(tmp_path):
    """Occurrence chỉ có một lát thì không ghép được gì — trả nguyên 4 s."""
    goc = _tin_hieu(4.0, seed=2)
    _ghi(tmp_path / "a.wav", goc)

    dung = bb.tai_dung([tmp_path / "a.wav"], SR)

    np.testing.assert_allclose(dung, goc, atol=1e-6)


def test_tai_dung_do_dai_dung_cong_thuc_2n_cong_2(tmp_path):
    """n lát liền kề → 2n + 2 giây. Công thức này quyết định trữ lượng của cả bank."""
    lat = _cat_lat(_tin_hieu(14.0, seed=3), tmp_path)

    for n in range(1, len(lat) + 1):
        dung = bb.tai_dung(lat[:n], SR)
        assert len(dung) == pytest.approx((2 * n + 2) * SR, abs=1)


# ── Chọn đoạn liền kề ────────────────────────────────────────────────────────


def test_chuoi_lien_tuc_lay_doan_dai_nhat():
    """Lát 0,1,2 rồi nhảy sang 5,6 → phải lấy đoạn [0,1,2], không ghép qua chỗ đứt.

    Ghép qua chỗ đứt là nối hai quãng KHÁC NHAU của bản ghi lại với nhau — đúng cái
    đứt gãy mà cả bước này sinh ra để tránh.
    """
    assert bb.chuoi_lien_tuc([(0, "a"), (1, "b"), (2, "c"), (5, "e"), (6, "f")]) == ["a", "b", "c"]


def test_chuoi_lien_tuc_uu_tien_doan_dai_hon_du_o_cuoi():
    assert bb.chuoi_lien_tuc([(0, "a"), (3, "c"), (4, "d"), (5, "e")]) == ["c", "d", "e"]


def test_chuoi_lien_tuc_lat_le_loi():
    assert bb.chuoi_lien_tuc([(7, "x")]) == ["x"]


# ── Lặp có crossfade cho đơn vị ngắn ─────────────────────────────────────────


def test_lap_crossfade_ra_dung_do_dai_muc_tieu():
    ra = bb.lap_crossfade(_tin_hieu(4.0, seed=7), SR, dai_sec=10.0, fade_sec=0.25)

    assert len(ra) == 10 * SR


def test_lap_crossfade_khong_tao_cu_nhay_tai_cho_noi():
    """Chỗ nối phải mượt hơn nối thô — đây là lý do dùng crossfade thay vì cắt dán.

    So với chính tín hiệu: cú nhảy lớn nhất sau khi lặp không được vượt cú nhảy lớn
    nhất vốn có của nguồn. Vượt nghĩa là ta vừa tạo ra một transient giả, mà một
    transient đều đặn mỗi vài giây thì model rất dễ học thành onset.
    """
    nguon = _tin_hieu(4.0, seed=8)
    ra = bb.lap_crossfade(nguon, SR, dai_sec=10.0, fade_sec=0.25)

    assert np.abs(np.diff(ra)).max() <= np.abs(np.diff(nguon)).max() + 1e-6


def test_lap_crossfade_giu_nguyen_muc_to():
    """Crossfade sai luật sẽ làm tụt mức tại chỗ nối — phá tỉ số SNR Scaper sắp đặt."""
    nguon = _tin_hieu(4.0, seed=9)
    ra = bb.lap_crossfade(nguon, SR, dai_sec=10.0, fade_sec=0.25)

    rms_nguon = float(np.sqrt(np.mean(nguon**2)))
    rms_ra = float(np.sqrt(np.mean(ra**2)))
    assert rms_ra == pytest.approx(rms_nguon, rel=0.15)


def test_lap_crossfade_dai_hon_muc_tieu_thi_cat_bot():
    ra = bb.lap_crossfade(_tin_hieu(14.0, seed=10), SR, dai_sec=10.0, fade_sec=0.25)

    assert len(ra) == 10 * SR


# ── Bất biến của cả bank sinh ra ─────────────────────────────────────────────


def test_moi_clip_sinh_ra_deu_dat_it_nhat_10_giay(tmp_path):
    """Không clip nào ngắn hơn 10 s — ngắn hơn là Scaper lại lặp vòng, đúng thứ đang chữa."""
    for giay, seed in [(4.0, 11), (10.0, 12), (26.0, 13)]:
        lat = _cat_lat(_tin_hieu(giay, seed), tmp_path / f"u{seed}")

        clip = bb.clip_tu_don_vi(bb.tai_dung(lat, SR), SR)

        assert len(clip) >= 10 * SR


def test_don_vi_du_dai_duoc_giu_NGUYEN_do_dai(tmp_path):
    """Đơn vị 26 s phải ra file 26 s, không bị cắt về 10 s.

    Giữ nguyên độ dài chính là thứ cho phép Scaper rút `source_time` ngẫu nhiên:
    mỗi lần dùng là một cửa sổ 10 s khác nhau trong cùng bản ghi. Cắt sẵn về 10 s
    thì mỗi lần dùng lại nghe đúng một đoạn, và ta mất đa dạng mà chẳng đổi lại gì.
    """
    lat = _cat_lat(_tin_hieu(26.0, seed=14), tmp_path / "dai")
    goc = bb.tai_dung(lat, SR)

    clip = bb.clip_tu_don_vi(goc, SR)

    assert len(clip) == len(goc)
    np.testing.assert_allclose(clip, goc, atol=1e-6)


def test_don_vi_ngan_duoc_keo_len_dung_10_giay(tmp_path):
    """Đơn vị 4 s → đúng 10 s bằng lặp crossfade, không hơn không kém."""
    lat = _cat_lat(_tin_hieu(4.0, seed=15), tmp_path / "ngan")

    clip = bb.clip_tu_don_vi(bb.tai_dung(lat, SR), SR)

    assert len(clip) == 10 * SR
