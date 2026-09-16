"""Test cho scripts/promote_to_bank.py.

Hai lỗi thật bắt được ngày 16/09 khi tách lớp `laughter_cheering`:
  1. Thư mục bank không được xoá trước khi ghi lại, nên 249 file .wav cũ nằm mồ côi
     trong `data/banks/foreground/laughter_cheering/` — không có trong
     `foreground_manifest.csv` (file đó được ghi lại đúng) nhưng vẫn nằm THẬT trên
     đĩa, và `scaper_generate.py --plan-only` (quét trực tiếp thư mục bank) báo nhầm
     đó là rò rỉ dữ liệu.
  2. Verdict "ok" cũ (điền trước khi lớp đổi tên) đưa được một clip ĐÃ BỊ LOẠI CHÍNH
     THỨC (auto_screen ghi wrong_class vào exclusions.csv sau khi rescan) lọt vào
     bank — verdict cũ không biết gì về lần loại sau đó.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import soundfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import promote_to_bank as ptb  # noqa: E402


def _score_row(file_id: str, class_id: str, decision: str = "auto_accept") -> dict:
    return {"file_id": file_id, "class_id": class_id, "decision": decision,
            "p_target": "0.9", "p_confusable": "0.05", "onset": "", "offset": "",
            "path_norm": ""}


def _write_minimal_wav(path: Path) -> None:
    soundfile.write(str(path), np.zeros(1600, dtype=np.float32), 16000)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _patch_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ptb, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ptb, "SCORES_PATH", tmp_path / "screen_scores.csv")
    monkeypatch.setattr(ptb, "AUDIT_PATH", tmp_path / "screen_audit.csv")
    monkeypatch.setattr(ptb, "SPLITS_PATH", tmp_path / "splits.csv")
    monkeypatch.setattr(ptb, "BANK_DIR", tmp_path / "data" / "banks" / "foreground")
    monkeypatch.setattr(ptb, "MANIFEST_PATH", tmp_path / "foreground_manifest.csv")


def _setup_one_clip(tmp_path: Path, file_id: str, class_id: str) -> None:
    normalized = tmp_path / f"{file_id}.wav"
    _write_minimal_wav(normalized)
    _write_csv(ptb.SCORES_PATH, list(_score_row(file_id, class_id).keys()),
               [{**_score_row(file_id, class_id), "path_norm": str(normalized)}])
    _write_csv(ptb.SPLITS_PATH, ["file_id", "class_id", "source_dataset", "source_group_id",
                                 "merged_group_id", "split"],
               [{"file_id": file_id, "class_id": class_id, "source_dataset": "x",
                 "source_group_id": "g1", "merged_group_id": "g1", "split": "foreground_bank_train"}])


# ── main(): xoá bank cũ trước khi ghi lại ────────────────────────────────────


def test_thu_muc_lop_da_xoa_bi_don_khi_chay_lai(tmp_path, monkeypatch):
    _patch_paths(tmp_path, monkeypatch)
    _setup_one_clip(tmp_path, "a", "laughter")

    # Mô phỏng file mồ côi của một lớp đã đổi tên — vẫn còn thật trên đĩa từ lần chạy
    # trước, dù screen_scores.csv hiện tại không còn dòng nào thuộc lớp đó.
    stale_dir = ptb.BANK_DIR / "laughter_cheering"
    stale_dir.mkdir(parents=True)
    (stale_dir / "orphan.wav").write_bytes(b"RIFF")

    monkeypatch.setattr(sys, "argv", ["promote_to_bank.py"])
    assert ptb.main() == 0

    assert not stale_dir.exists()                          # dòng mồ côi bị xoá sạch
    assert (ptb.BANK_DIR / "laughter" / "a.wav").exists()   # clip mới vẫn được ghi


def test_dry_run_khong_dong_gi_ca(tmp_path, monkeypatch):
    _patch_paths(tmp_path, monkeypatch)
    _setup_one_clip(tmp_path, "a", "laughter")

    stale_dir = ptb.BANK_DIR / "old_class"
    stale_dir.mkdir(parents=True)
    (stale_dir / "orphan.wav").write_bytes(b"RIFF")

    monkeypatch.setattr(sys, "argv", ["promote_to_bank.py", "--dry-run"])
    assert ptb.main() == 0
    assert stale_dir.exists()      # --dry-run không đụng tới đĩa, kể cả xoá


# ── selectable(): verdict cũ không cứu được clip đã bị loại chính thức ──────


def test_verdict_ok_khong_cuu_duoc_clip_da_bi_loai_khoi_split(tmp_path, monkeypatch):
    """Đúng lỗi thật: clip đã bị auto_screen loại (ghi wrong_class vào
    exclusions.csv, nên make_splits.py không còn xếp nó vào split nào) vẫn lọt vào
    bank vì verdict "ok" cũ (điền trước khi lớp đổi tên) ghi đè vô điều kiện."""
    monkeypatch.setattr(ptb, "AUDIT_PATH", tmp_path / "screen_audit.csv")
    _write_csv(ptb.AUDIT_PATH, ["file_id", "class_id", "verdict"],
               [{"file_id": "a", "class_id": "applause_cheering", "verdict": "ok"}])

    scored = [{**_score_row("a", "applause_cheering", decision="auto_reject")}]
    chosen, skipped = ptb.selectable(scored, include_reviewed=False, in_split=set())  # "a" KHÔNG còn trong split

    assert chosen == []
    assert any("loại khỏi foreground_bank_train" in reason for reason in skipped)


def test_verdict_ok_van_duoc_nhan_neu_con_trong_split(tmp_path, monkeypatch):
    monkeypatch.setattr(ptb, "AUDIT_PATH", tmp_path / "screen_audit.csv")
    _write_csv(ptb.AUDIT_PATH, ["file_id", "class_id", "verdict"],
               [{"file_id": "a", "class_id": "applause_cheering", "verdict": "ok"}])

    scored = [{**_score_row("a", "applause_cheering", decision="auto_reject")}]
    chosen, skipped = ptb.selectable(scored, include_reviewed=False, in_split={"a"})

    assert [r["file_id"] for r in chosen] == ["a"]


def test_khong_truyen_in_split_thi_giu_hanh_vi_cu():
    """in_split=None (mặc định) — không kiểm tra split, giữ hành vi trước khi sửa."""
    scored = [_score_row("a", "siren", decision="auto_accept")]
    chosen, skipped = ptb.selectable(scored, include_reviewed=False)
    assert [r["file_id"] for r in chosen] == ["a"]


# ── promote(): source_dataset tra từ raw_manifest, không suy từ file_id ────


def test_source_dataset_tra_dung_khong_doan_tu_ten_file(tmp_path, monkeypatch):
    """file_id.split('_')[0] sai với mọi nguồn có gạch dưới trong tên: đo thật,
    "vehicle_crash_cc_xxx" ra "vehicle" thay vì "vehicle_crash_cc"."""
    normalized = tmp_path / "clip.wav"
    _write_minimal_wav(normalized)
    row = {**_score_row("vehicle_crash_cc_a1b2", "vehicle_crash"), "path_norm": str(normalized)}
    monkeypatch.setattr(ptb, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ptb, "BANK_DIR", tmp_path / "bank")

    entry = ptb.promote(row, dry_run=True, source_dataset_by_file={"vehicle_crash_cc_a1b2": "vehicle_crash_cc"})
    assert entry["source_dataset"] == "vehicle_crash_cc"


def test_source_dataset_thieu_thi_dau_hoi_khong_chet(tmp_path, monkeypatch):
    normalized = tmp_path / "clip.wav"
    _write_minimal_wav(normalized)
    row = {**_score_row("a", "siren"), "path_norm": str(normalized)}
    monkeypatch.setattr(ptb, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ptb, "BANK_DIR", tmp_path / "bank")

    entry = ptb.promote(row, dry_run=True, source_dataset_by_file={})
    assert entry["source_dataset"] == "?"
