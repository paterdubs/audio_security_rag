"""Test cho scripts/data_inventory.py.

Điểm khác biệt có chủ đích với coverage_report.py: script này đo số CLIP THẬT SỰ
ĐÃ VÀO BANK (foreground_manifest.csv, sau sàng lọc + duyệt người), không phải mọi
clip đủ điều kiện (interim/normalized/). Test phải bắt được nếu ai đó vô tình đổi
nguồn dữ liệu sang manifest sai, vì hai con số đó luôn khác nhau và không thể thay
nhau được trong khoá luận.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import data_inventory as di  # noqa: E402


ONTOLOGY = {"classes": {
    "siren": {"group": "A", "target_foreground": 100, "min_acceptable": 40},
    "explosion": {"group": "B", "target_foreground": 50, "min_acceptable": 20},
    "ambient_noise": {"group": "B", "target_foreground": 0, "min_acceptable": 0},
}}


def _fg_row(file_id, class_id, source, duration_bank):
    return {"file_id": file_id, "class_id": class_id, "source_dataset": source,
            "duration_bank": str(duration_bank)}


# ── Foreground bank ──────────────────────────────────────────────────────────


def test_lop_chua_co_clip_nao_van_xuat_hien_voi_0():
    """Lớp trong ontology nhưng chưa có clip nào — vẫn phải liệt kê, không bỏ qua,
    vì đây chính là điều cổng D7 cần thấy."""
    table = di.foreground_table(ONTOLOGY, [])
    by_class = {r["class_id"]: r for r in table}
    assert by_class["explosion"]["n_clips"] == 0
    assert by_class["explosion"]["status"] == "⛔ chưa có"


def test_tinh_dung_so_phut_tu_giay():
    rows = [_fg_row("a", "siren", "urbansound8k", 30.0), _fg_row("b", "siren", "urbansound8k", 30.0)]
    table = di.foreground_table(ONTOLOGY, rows)
    siren = next(r for r in table if r["class_id"] == "siren")
    assert siren["minutes"] == 1.0


def test_trang_thai_theo_dung_bac_muc_tieu_toi_thieu():
    rows = [_fg_row(f"e{i}", "explosion", "fsd50k", 1.0) for i in range(25)]  # 25 >= min(20), < target(50)
    table = di.foreground_table(ONTOLOGY, rows)
    explosion = next(r for r in table if r["class_id"] == "explosion")
    assert explosion["status"] == "🟡 trên tối thiểu"


def test_lop_target_0_la_background_khong_co_trang_thai_thieu():
    """`ambient_noise` (target=0) không dùng con đường foreground — không được báo
    là "⛔ chưa có" chỉ vì không có clip nào, nó vốn không thuộc bank này."""
    table = di.foreground_table(ONTOLOGY, [])
    ambient = next(r for r in table if r["class_id"] == "ambient_noise")
    assert ambient["status"] == "—"


def test_gop_nguon_khong_trung_lap():
    rows = [_fg_row("a", "siren", "esc50", 5.0), _fg_row("b", "siren", "esc50", 5.0),
            _fg_row("c", "siren", "fsd50k", 5.0)]
    table = di.foreground_table(ONTOLOGY, rows)
    siren = next(r for r in table if r["class_id"] == "siren")
    assert siren["sources"] == ["esc50", "fsd50k"]


# ── Background bank ──────────────────────────────────────────────────────────


def test_background_bo_qua_clip_flagged():
    """`flagged` là clip máy nghi có sự kiện Nhóm A lẫn vào — không được tính vào
    số giờ nền sạch, kể cả khi chưa ai xác nhận loại nó."""
    rows = [
        {"area_type": "school", "duration": "60.0", "verdict": "clean"},
        {"area_type": "school", "duration": "60.0", "verdict": "flagged"},
    ]
    table = di.background_table(rows)
    school = next(r for r in table if r["area_type"] == "school")
    assert school["n_clips"] == 1


# ── RIR bank ─────────────────────────────────────────────────────────────────


def test_rir_trung_vi_khong_bi_keo_boi_gia_tri_ngoai_lai():
    rows = [{"space_type": "small_room", "rt60_sec": v} for v in ["0.2", "0.3", "5.0"]]
    table = di.rir_table(rows)
    small = next(r for r in table if r["space_type"] == "small_room")
    assert small["median_rt60"] == 0.3


# ── Synthetic train/dev ──────────────────────────────────────────────────────


def test_synthetic_summary_khong_co_file_thi_tra_none(tmp_path, monkeypatch):
    monkeypatch.setattr(di, "SYNTHETIC_DIR", tmp_path)
    assert di.synthetic_summary("train") is None


def test_synthetic_summary_tinh_dung_gio_va_dem_im_lang(tmp_path, monkeypatch):
    monkeypatch.setattr(di, "SYNTHETIC_DIR", tmp_path)
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    (split_dir / "slice_index.jsonl").write_text(
        '{"clip_id": "a", "slices": ["overlap"], "n_events": 2, "chain": null}\n'
        '{"clip_id": "b", "slices": [], "n_events": 0, "chain": null}\n',
        encoding="utf-8",
    )
    summary = di.synthetic_summary("train")
    assert summary["n_clips"] == 2
    assert summary["n_silent"] == 1
    assert summary["hours"] == round(2 * 10.0 / 3600, 1)
    assert summary["slice_counts"] == {"overlap": 1}


# ── dev / gold_test thật ─────────────────────────────────────────────────────


def test_real_audio_table_tinh_dung_theo_split():
    splits_rows = [
        {"file_id": "a", "split": "dev"},
        {"file_id": "b", "split": "gold_test"},
        {"file_id": "c", "split": "foreground_bank_train"},   # KHÔNG được tính vào dev/gold_test
    ]
    raw_index = {
        "a": {"duration": "600.0", "claimed_class": "siren"},
        "b": {"duration": "300.0", "claimed_class": "gunshot"},
        "c": {"duration": "999.0", "claimed_class": "laughter"},
    }
    table = di.real_audio_table(splits_rows, raw_index)
    by_split = {r["split"]: r for r in table}
    assert by_split["dev"]["n_clips"] == 1
    assert by_split["gold_test"]["n_clips"] == 1
    assert by_split["dev"]["minutes"] == 10.0
