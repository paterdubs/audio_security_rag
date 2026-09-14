"""Test cho scripts/make_splits.py và scripts/check_leakage.py.

Rò rỉ audio không làm hỏng gì lộ liễu — nó chỉ làm điểm số đẹp lên. Nên test ở đây
phải DỰNG RA rò rỉ rồi bắt script phát hiện, chứ chạy trên dữ liệu sạch thì mọi
kiểm tra đều đạt kể cả khi chúng không kiểm tra gì cả.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import check_leakage as cl  # noqa: E402
import make_splits as ms  # noqa: E402

CONFIG = {
    "seed": "test-seed",
    "eligibility": {
        "desed_soundbank": ["foreground_bank_train"],
        "esc50": ["foreground_bank_train"],
        "audioset_strong": ["dev", "gold_test"],
        "iuh_field": ["gold_test"],
    },
    "ratios": {"dev": 0.6, "gold_test": 0.4},
}


def _row(file_id: str, group: str, dataset: str = "esc50", class_id: str = "siren") -> dict:
    return {
        "file_id": file_id,
        "source_group_id": group,
        "source_dataset": dataset,
        "claimed_class": class_id,
        "checksum_sha256": f"cs_{file_id}",
    }


def _dedup(checksum: str, *groups: str) -> dict:
    return {"checksum_sha256": checksum, "source_group_ids": "|".join(groups)}


# ── Gộp nhóm bắc cầu: chính là lỗi đã đo được ở DESED ────────────────────────


def test_hai_id_freesound_cung_mot_audio_thi_ve_mot_nhom():
    rows = [_row("a", "freesound_13613"), _row("b", "freesound_34853")]
    dedup = [_dedup("x", "freesound_13613", "freesound_34853")]
    merged = ms.merge_groups(rows, dedup)
    assert merged["freesound_13613"] == merged["freesound_34853"]


def test_gop_bac_cau_qua_nhieu_chang():
    # A≡B qua bản trùng này, B≡C qua bản trùng khác → cả ba phải cùng một tập.
    # Gộp từng cặp mà không truy vết bắc cầu sẽ để C ở nhóm riêng.
    rows = [_row("a", "g_a"), _row("b", "g_b"), _row("c", "g_c")]
    dedup = [_dedup("x", "g_a", "g_b"), _dedup("y", "g_b", "g_c")]
    merged = ms.merge_groups(rows, dedup)
    assert len(set(merged.values())) == 1


def test_id_nhom_gop_khong_phu_thuoc_thu_tu_doc_file():
    rows = [_row("a", "g_a"), _row("b", "g_b")]
    xuoi = ms.merge_groups(rows, [_dedup("x", "g_a", "g_b")])
    nguoc = ms.merge_groups(rows[::-1], [_dedup("x", "g_b", "g_a")])
    assert xuoi == nguoc


def test_nhom_khong_lien_quan_thi_khong_bi_gop():
    rows = [_row("a", "g_a"), _row("b", "g_b")]
    merged = ms.merge_groups(rows, [])
    assert merged["g_a"] != merged["g_b"]


# ── Ràng buộc theo nguồn ─────────────────────────────────────────────────────


def test_mau_su_kien_cat_roi_khong_duoc_vao_gold_test():
    rows = [_row(f"f{i}", f"g{i}", "desed_soundbank") for i in range(50)]
    out = ms.build_splits(rows, [], CONFIG)
    assert {r["split"] for r in out} == {"foreground_bank_train"}


def test_nguon_la_thi_dung_chu_khong_gan_bua():
    rows = [_row("a", "g_a", "mivia")]
    with pytest.raises(SystemExit) as err:
        ms.build_splits(rows, [], CONFIG)
    assert "mivia" in str(err.value)


def test_bank_trung_byte_voi_clip_danh_gia_thi_dung():
    # Không cách gán nào an toàn: để cả nhóm vào train thì mất clip đánh giá,
    # để vào gold_test thì mẩu bank rơi vào bài test. Phải người quyết định.
    rows = [_row("a", "g_a", "desed_soundbank"), _row("b", "g_b", "audioset_strong")]
    with pytest.raises(SystemExit) as err:
        ms.build_splits(rows, [_dedup("x", "g_a", "g_b")], CONFIG)
    assert "trùng byte" in str(err.value)


# ── Gán theo băm: ổn định khi thêm dữ liệu ───────────────────────────────────


def test_them_du_lieu_moi_khong_lam_xao_lai_nhom_cu():
    cu = [_row(f"f{i}", f"g{i}", "audioset_strong") for i in range(40)]
    moi = cu + [_row(f"n{i}", f"gn{i}", "audioset_strong") for i in range(40)]
    truoc = {r["file_id"]: r["split"] for r in ms.build_splits(cu, [], CONFIG)}
    sau = {r["file_id"]: r["split"] for r in ms.build_splits(moi, [], CONFIG)}
    assert all(sau[fid] == split for fid, split in truoc.items())


def test_nguon_hai_tap_thi_chia_ra_ca_hai():
    rows = [_row(f"f{i}", f"g{i}", "audioset_strong") for i in range(200)]
    out = ms.build_splits(rows, [], CONFIG)
    assert {r["split"] for r in out} == {"dev", "gold_test"}


def test_moi_clip_cung_nhom_gop_luon_cung_mot_tap():
    rows = [_row(f"f{i}", f"g{i % 20}", "audioset_strong") for i in range(200)]
    out = ms.build_splits(rows, [], CONFIG)
    theo_nhom = {}
    for r in out:
        theo_nhom.setdefault(r["merged_group_id"], set()).add(r["split"])
    assert all(len(s) == 1 for s in theo_nhom.values())


# ── check_leakage: phải BẮT được rò rỉ dựng sẵn ──────────────────────────────


def _split_row(file_id: str, group: str, split: str) -> dict:
    return {"file_id": file_id, "source_group_id": group, "split": split, "merged_group_id": group}


def test_bat_duoc_nhom_nguon_nam_hai_ben():
    rows = [_split_row("a", "g1", "dev"), _split_row("b", "g1", "gold_test")]
    assert cl.check_group_not_split(rows)


def test_khong_bao_dong_gia_khi_nhom_nam_mot_ben():
    rows = [_split_row("a", "g1", "dev"), _split_row("b", "g1", "dev")]
    assert cl.check_group_not_split(rows) == []


def test_bat_duoc_audio_giong_het_o_hai_ben_du_khac_nhom():
    # Đúng tình huống DESED: khác id nhóm nên kiểm tra 1 không thấy gì,
    # nhưng hai file giống nhau từng byte.
    split_rows = [_split_row("a", "g1", "dev"), _split_row("b", "g2", "gold_test")]
    manifest = [
        {"file_id": "a", "checksum_sha256": "GIONG_NHAU"},
        {"file_id": "b", "checksum_sha256": "GIONG_NHAU"},
    ]
    assert cl.check_group_not_split(split_rows) == []      # kiểm tra 1 mù ở đây
    assert cl.check_no_shared_audio(split_rows, manifest)  # kiểm tra 2 bắt được


def test_bat_duoc_nhom_trung_lap_bac_cau_hai_ben():
    split_rows = [_split_row("a", "g1", "dev"), _split_row("b", "g2", "gold_test")]
    assert cl.check_dedup_groups_intact(split_rows, [_dedup("x", "g1", "g2")])


def test_nhom_trung_lap_cung_mot_ben_la_vo_hai():
    split_rows = [_split_row("a", "g1", "dev"), _split_row("b", "g2", "dev")]
    assert cl.check_dedup_groups_intact(split_rows, [_dedup("x", "g1", "g2")]) == []


def test_thanh_vien_da_bi_loai_khoi_manifest_khong_gay_bao_dong_gia():
    # g2 không còn trong splits.csv (đã bị loại) → chỉ còn g1, không thể bắc cầu.
    split_rows = [_split_row("a", "g1", "dev")]
    assert cl.check_dedup_groups_intact(split_rows, [_dedup("x", "g1", "g2")]) == []
