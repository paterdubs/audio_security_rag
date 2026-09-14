"""Test cho scripts/normalize_audio.py.

Hai nhóm quan trọng nhất:
  · `flat_top_ratio` — quyết định clip nào bị loại vĩnh viễn khỏi dataset.
  · `normalize_loudness` — sai ở đây làm lệch toàn bộ phân bố độ to của bank, và
    không có thông báo lỗi nào cả.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import normalize_audio as na  # noqa: E402

SAMPLE_RATE = 16000
MIN_RUN = 3
CEILING = 0.999


def sine(seconds: float = 1.0, freq: float = 440.0, amplitude: float = 0.5) -> np.ndarray:
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t)


# ── Phát hiện đỉnh bị cắt phẳng ──────────────────────────────────────────────


def test_tin_hieu_sach_khong_bi_bao_meo():
    assert na.flat_top_ratio(sine(), MIN_RUN, CEILING) == 0.0


def test_phat_hien_doan_bi_cat_phang():
    audio = sine()
    audio[100:110] = 1.0                       # 10 mẫu liên tiếp chạm trần
    ratio = na.flat_top_ratio(audio, MIN_RUN, CEILING)
    assert ratio == pytest.approx(10 / len(audio))


def test_dinh_nhon_don_le_khong_tinh_la_meo():
    """Tiếng súng và kính vỡ vốn có đỉnh nhọn chạm trần — đó KHÔNG phải méo."""
    audio = sine()
    audio[100] = 1.0
    audio[500] = -1.0                          # hai mẫu rời rạc, không thành đoạn
    assert na.flat_top_ratio(audio, MIN_RUN, CEILING) == 0.0


def test_doan_ngan_hon_nguong_thi_bo_qua():
    audio = sine()
    audio[100:102] = 1.0                       # 2 mẫu < min_run=3
    assert na.flat_top_ratio(audio, MIN_RUN, CEILING) == 0.0


def test_cong_don_nhieu_doan_cat_phang():
    audio = sine()
    audio[100:105] = 1.0
    audio[700:703] = -1.0
    assert na.flat_top_ratio(audio, MIN_RUN, CEILING) == pytest.approx(8 / len(audio))


def test_dinh_am_cung_duoc_tinh():
    """Méo xảy ra ở cả hai chiều biên độ — chỉ xét chiều dương là bỏ sót một nửa."""
    audio = sine()
    audio[100:110] = -1.0
    assert na.flat_top_ratio(audio, MIN_RUN, CEILING) > 0


# ── Chuẩn hoá độ to ──────────────────────────────────────────────────────────

LOUDNESS_SETTINGS = {"target_lufs": -23.0, "min_duration_for_lufs": 0.4, "peak_ceiling_dbfs": -1.0}


def test_hai_clip_to_nho_khac_nhau_ve_cung_mot_muc():
    """Đây chính là mục đích của bước này: bank không được lẫn lộn to nhỏ."""
    quiet, _ = na.normalize_loudness(sine(3.0, amplitude=0.02), SAMPLE_RATE, LOUDNESS_SETTINGS)
    loud, _ = na.normalize_loudness(sine(3.0, amplitude=0.8), SAMPLE_RATE, LOUDNESS_SETTINGS)

    import pyloudnorm

    meter = pyloudnorm.Meter(SAMPLE_RATE)
    assert meter.integrated_loudness(quiet) == pytest.approx(-23.0, abs=0.5)
    assert meter.integrated_loudness(loud) == pytest.approx(-23.0, abs=0.5)


def test_clip_qua_ngan_thi_giu_nguyen_khong_chuan_hoa():
    """pyloudnorm cần ≥0.4s cho một block 400ms; ngắn hơn thì số đo vô nghĩa."""
    short = sine(0.2)
    result, measured = na.normalize_loudness(short, SAMPLE_RATE, LOUDNESS_SETTINGS)
    assert measured is None
    np.testing.assert_array_equal(result, short)


def test_khong_bao_gio_vuot_tran_sau_khi_chuan_hoa():
    result, _ = na.normalize_loudness(sine(3.0, amplitude=0.01), SAMPLE_RATE, LOUDNESS_SETTINGS)
    ceiling = 10 ** (LOUDNESS_SETTINGS["peak_ceiling_dbfs"] / 20)
    assert np.max(np.abs(result)) <= ceiling + 1e-6


def test_tran_khong_lam_to_them_tin_hieu_nho():
    """apply_peak_ceiling chỉ được HẠ gain — nâng lên sẽ phá kết quả chuẩn hoá độ to."""
    quiet = sine(amplitude=0.05)
    result = na.apply_peak_ceiling(quiet, -1.0)
    np.testing.assert_array_equal(result, quiet)


# ── Cổng chất lượng ──────────────────────────────────────────────────────────

GATES = {
    "min_duration_sec": 0.3,
    "max_duration_sec": 30.0,
    "min_original_sample_rate": 16000,
    "silence_rms_threshold": 0.001,
    "max_clipped_ratio": 0.01,
    "clipping_review_ratio": 0.001,
    "clipping_min_run_samples": 3,
    "clipping_ceiling": 0.999,
}


def check(audio: np.ndarray, original_rate: int = 44100, clipped: float = 0.0):
    return na.check_quality(audio, SAMPLE_RATE, original_rate, clipped, GATES)


def test_clip_dat_thi_khong_loai_khong_gan_co():
    assert check(sine()) == (None, None)


@pytest.mark.parametrize(
    "audio,original_rate,clipped,expected_code",
    [
        (sine(0.1), 44100, 0.0, "too_short"),
        (sine(31.0), 44100, 0.0, "too_long"),
        (sine(), 8000, 0.0, "upsampled"),
        (np.zeros(SAMPLE_RATE), 44100, 0.0, "silent"),
        (sine(), 44100, 0.05, "clipped"),
    ],
)
def test_cac_ly_do_loai(audio, original_rate, clipped, expected_code):
    rejection, flag = check(audio, original_rate, clipped)
    assert rejection is not None and rejection.reason_code == expected_code
    assert flag is None, "clip đã bị loại thì không cần gắn cờ nữa"


def test_vung_xam_thi_gan_co_chu_khong_loai():
    """Điểm mấu chốt: dữ liệu an ninh khan hiếm, máy không được tự loại ca mơ hồ."""
    rejection, flag = check(sine(), clipped=0.005)
    assert rejection is None, "0.5% méo chưa đủ để máy tự loại"
    assert flag is not None and flag.flag == "clipping"


def test_thu_tu_uu_tien_do_dai_truoc_meo():
    """Clip vừa quá ngắn vừa méo phải báo lý do gốc rễ, không báo lý do thứ cấp."""
    rejection, _ = check(sine(0.1), clipped=0.9)
    assert rejection.reason_code == "too_short"


# ── Kiểm tra trên dữ liệu thật ───────────────────────────────────────────────


def test_file_da_chuan_hoa_dung_dinh_dang():
    import soundfile

    files = list((na.NORMALIZED_DIR).rglob("*.wav"))
    if not files:
        pytest.skip("chưa chạy normalize_audio.py")

    for path in files[:20]:
        info = soundfile.info(str(path))
        assert info.samplerate == 16000, f"{path.name}: {info.samplerate} Hz"
        assert info.channels == 1, f"{path.name}: {info.channels} kênh"


def test_khong_clip_nao_vua_bi_loai_vua_duoc_giu():
    """Mâu thuẫn này nghĩa là dataset không tái lập được."""
    import csv

    import common
    if not common.EXCLUSIONS_PATH.exists():
        pytest.skip("chưa có exclusions.csv")

    with common.EXCLUSIONS_PATH.open(encoding="utf-8", newline="") as handle:
        excluded = {r["file_id"] for r in csv.DictReader(handle) if r["stage"] == na.STAGE}

    kept = {p.stem for p in na.NORMALIZED_DIR.rglob("*.wav")}
    assert not (excluded & kept), f"vừa loại vừa giữ: {sorted(excluded & kept)[:5]}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
