"""Test cho scripts/probe_bank.py — bảng tra thời lượng hữu ích của bank.

Bảng này là đầu vào của MỌI bản sửa trong STATUS §7: `source_time` và
`event_duration` của từng sự kiện đều tính từ nó. Đo sai ở đây thì toàn bộ 9.360
clip sinh ra sau đó đều sai, và sai một cách im lặng — Scaper vẫn chạy trơn tru,
jams vẫn hợp lệ, chỉ có nội dung là lệch.

Nên test ở đây dựng tín hiệu ĐÃ BIẾT TRƯỚC đáp án thay vì kiểm tra trên audio
thật: chỉ tín hiệu tự dựng mới nói được "đáp án đúng phải là 0.5 giây".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import probe_bank as pb  # noqa: E402

SR = 16000

# librosa.effects.trim quyết định theo FRAME (frame_length=2048, hop=512), nên mốc
# nó trả về lệch tối đa nửa frame = 1024/16000 = 0.064 s so với mốc thật. Lệch này
# LUÔN theo hướng giữ thêm chứ không cắt vào — xem test_trim_thien_ve_giu_lai.
# Đặt dung sai chặt hơn nửa frame là test sẽ đỏ vì lượng tử hoá, không vì lỗi.
# `eff` nới ra ở CẢ HAI biên nên sai số của nó gấp đôi sai số một biên.
NUA_FRAME_SEC = 1024 / SR
DUNG_SAI_SEC = NUA_FRAME_SEC + 0.01          # cho lead, trail, dur — một biên
DUNG_SAI_EFF = 2 * NUA_FRAME_SEC + 0.01      # cho eff — hai biên


def _ghi(path: Path, song: np.ndarray, sr: int = SR) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), song.astype(np.float32), sr)
    return path


def _am(giay: float, bien_do: float = 0.5, sr: int = SR) -> np.ndarray:
    """Sóng sin 440 Hz — tín hiệu 'có tiếng' rõ ràng, không phải nhiễu ngẫu nhiên."""
    t = np.arange(int(giay * sr)) / sr
    return bien_do * np.sin(2 * np.pi * 440.0 * t)


def _im(giay: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(giay * sr), dtype=np.float32)


# ── Ba dạng tín hiệu nền tảng ────────────────────────────────────────────────


def test_im_lang_am_im_lang_do_dung_ba_doan(tmp_path):
    """0.5s im · 1.0s tiếng · 0.8s im → lead=0.5, eff=1.0, trail=0.8.

    Đây chính là ca mà `source_time=("const", 0)` làm hỏng: nó lấy 1 giây đầu,
    mà nửa giây đầu là im lặng.
    """
    wav = _ghi(tmp_path / "a.wav", np.concatenate([_im(0.5), _am(1.0), _im(0.8)]))

    do = pb.do_mot_clip(wav)

    assert do["lead"] == pytest.approx(0.5, abs=DUNG_SAI_SEC)
    assert do["eff"] == pytest.approx(1.0, abs=DUNG_SAI_EFF)
    assert do["trail"] == pytest.approx(0.8, abs=DUNG_SAI_SEC)
    assert do["dur"] == pytest.approx(2.3, abs=DUNG_SAI_SEC)


def test_am_thuan_khong_co_im_lang_dau_cuoi(tmp_path):
    """Tiếng phủ kín clip → lead=trail=0, eff=dur. Đây là dạng của 98.8% bank."""
    wav = _ghi(tmp_path / "b.wav", _am(1.5))

    do = pb.do_mot_clip(wav)

    assert do["lead"] == pytest.approx(0.0, abs=DUNG_SAI_SEC)
    assert do["trail"] == pytest.approx(0.0, abs=DUNG_SAI_SEC)
    assert do["eff"] == pytest.approx(do["dur"], abs=DUNG_SAI_EFF)


def test_im_lang_thuan_cho_eff_bang_khong(tmp_path):
    """Clip toàn im lặng → eff=0, và KHÔNG được ném lỗi.

    Bank có clip gần như im lặng (vehicle_crash −25.7 dB). Ném lỗi ở đây sẽ làm
    chết cả lượt quét 4.267 file vì một file hỏng.
    """
    wav = _ghi(tmp_path / "c.wav", _im(1.0))

    do = pb.do_mot_clip(wav)

    assert do["eff"] == 0.0
    assert do["dur"] == pytest.approx(1.0, abs=DUNG_SAI_SEC)


# ── Bất biến phải đúng với mọi clip ──────────────────────────────────────────


def test_trim_thien_ve_giu_lai_chu_khong_cat_vao_dau_song(tmp_path):
    """lead đo được ≤ lead thật, và eff đo được ≥ phần có tiếng thật.

    Hướng của sai số quan trọng hơn độ lớn. Trim cắt lố vào đầu sóng sẽ ăn mất
    transient tấn công của gunshot/glass_breaking — đúng cái đặc trưng nhận dạng
    lớp đó. Giữ thừa vài chục mili giây im lặng thì vô hại.
    """
    wav = _ghi(tmp_path / "huong.wav", np.concatenate([_im(0.5), _am(1.0), _im(0.8)]))

    do = pb.do_mot_clip(wav)

    assert do["lead"] <= 0.5 + 1e-6, "trim đã cắt lấn vào đầu sóng"
    assert do["trail"] <= 0.8 + 1e-6, "trim đã cắt lấn vào đuôi sóng"
    assert do["eff"] >= 1.0 - 1e-6, "phần có tiếng bị đo thiếu"


def test_bat_bien_lead_cong_eff_cong_trail_bang_dur(tmp_path):
    """lead + eff + trail == dur. Sai bất biến này là chỉ dấu lệch chỉ số frame."""
    wav = _ghi(tmp_path / "d.wav", np.concatenate([_im(0.3), _am(0.7), _im(0.2), _am(0.4), _im(0.6)]))

    do = pb.do_mot_clip(wav)

    assert do["lead"] + do["eff"] + do["trail"] == pytest.approx(do["dur"], abs=1e-6)


def test_eff_khong_bao_gio_vuot_dur(tmp_path):
    """eff ≤ dur luôn đúng — `event_duration` lấy từ eff nên vượt là Scaper cắt ngầm."""
    for i, song in enumerate([_am(0.2), _am(3.0), np.concatenate([_im(1.0), _am(0.1)])]):
        do = pb.do_mot_clip(_ghi(tmp_path / f"e{i}.wav", song))
        assert do["eff"] <= do["dur"] + 1e-6


def test_file_hong_bao_loi_chu_khong_lam_chet_luot_quet(tmp_path):
    """File không đọc được phải trả bản ghi có `loi`, không ném ra ngoài."""
    hong = tmp_path / "hong.wav"
    hong.write_bytes(b"day khong phai wav")

    do = pb.do_mot_clip(hong)

    assert do.get("loi")
    assert do["eff"] == 0.0


# ── Quét cả thư mục ──────────────────────────────────────────────────────────


def test_quet_bank_gan_dung_lop_theo_thu_muc_cha(tmp_path):
    """Lớp đọc từ tên thư mục cha, và file_id là stem — khớp với splits.csv."""
    _ghi(tmp_path / "gunshot" / "esc50_gunshot_abc.wav", _am(0.5))
    _ghi(tmp_path / "siren" / "us8k_siren_xyz.wav", np.concatenate([_im(0.2), _am(3.8)]))

    dong = pb.quet_bank(tmp_path)

    assert len(dong) == 2
    theo_id = {r["file_id"]: r for r in dong}
    assert theo_id["esc50_gunshot_abc"]["class_id"] == "gunshot"
    assert theo_id["us8k_siren_xyz"]["class_id"] == "siren"
    assert theo_id["us8k_siren_xyz"]["lead"] == pytest.approx(0.2, abs=DUNG_SAI_SEC)


def test_quet_bank_tat_dinh_theo_thu_tu(tmp_path):
    """Hai lượt quét cho đúng một thứ tự — dataset phải tái lập được."""
    _ghi(tmp_path / "gunshot" / "b.wav", _am(0.5))
    _ghi(tmp_path / "gunshot" / "a.wav", _am(0.5))
    _ghi(tmp_path / "siren" / "c.wav", _am(0.5))

    assert [r["file_id"] for r in pb.quet_bank(tmp_path)] == [r["file_id"] for r in pb.quet_bank(tmp_path)]


def test_ghi_csv_du_cot_va_doc_lai_duoc(tmp_path):
    """CSV ghi ra phải đọc lại đúng bằng csv.DictReader — đây là hợp đồng với scaper_generate."""
    import csv

    _ghi(tmp_path / "bank" / "gunshot" / "g.wav", np.concatenate([_im(0.4), _am(0.9)]))
    ra = tmp_path / "bank_trim.csv"

    pb.ghi_csv(pb.quet_bank(tmp_path / "bank"), ra)

    with ra.open(encoding="utf-8", newline="") as handle:
        dong = list(csv.DictReader(handle))
    assert len(dong) == 1
    assert set(pb.COT) <= set(dong[0])
    assert float(dong[0]["lead"]) == pytest.approx(0.4, abs=DUNG_SAI_SEC)
    assert float(dong[0]["eff"]) == pytest.approx(0.9, abs=DUNG_SAI_EFF)


def test_path_ghi_tuong_doi_tu_goc_repo():
    """Đường dẫn trong CSV phải tương đối — tuyệt đối là phá tính tái lập.

    CSV này đi theo git. Ghi `D:/IUH_K18/...` vào thì sang máy khác, sang container,
    hay đổi ổ đĩa là trỏ sai hết — mà trỏ sai sẽ biểu hiện thành "không tìm thấy
    file nguồn" giữa chừng một lượt sinh 18 giờ, chứ không phải lúc bắt đầu.
    """
    that = pb.REPO_ROOT / "data" / "banks" / "foreground" / "gunshot" / "x.wav"

    assert pb._duong_dan_tuong_doi(that) == "data/banks/foreground/gunshot/x.wav"
    assert not Path(pb._duong_dan_tuong_doi(that)).is_absolute()
