"""Test cho `scripts/check_leakage.py` — cổng chặn rò rỉ nguồn giữa các tập.

File này CHẶN cả pipeline nhưng trước 22/09 không có test nào. Rò rỉ nguồn **không có
triệu chứng nào ngoài việc điểm số đẹp lên** (docstring `scaper_generate.py`), nên một
cổng chặn hỏng âm thầm còn tệ hơn không có cổng: nó phát giấy chứng nhận sạch.

Ba kiểm tra cũ (1–3) được phủ test hồi cứu. Hai kiểm tra mới (4–5) thêm 22/09:

4. Nguồn dùng THẬT trong JAMS synthetic phải thuộc `foreground_bank_train`. Kiểm tra
   1–3 chỉ soi `splits.csv` — tức soi Ý ĐỊNH. Kiểm tra 4 soi ĐẦU RA. Dự án đã bị
   `run_manifest.doc_hop_dong` khai PASSED trong khi thiếu file báo cáo, nên cổng tự
   khai báo không thay được cổng đo lại.
5. Mức dùng chung nguồn giữa synthetic train và dev — KHÔNG phải cổng (cả hai sinh từ
   cùng một bank nên trùng là theo thiết kế), mà là số phải khai báo trong phần Hạn chế.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_leakage import (  # noqa: E402
    check_dedup_groups_intact,
    check_group_not_split,
    check_no_shared_audio,
    check_synthetic_sources_in_bank,
    clips_touching,
    collect_sources,
    read_jams_sources,
    shared_source_stats,
    source_id_from_path,
)


def viet_jams(path: Path, nguon: list[tuple[str, str]]) -> None:
    """nguon = [(role, source_file)]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"annotations": [{
        "namespace": "scaper",
        "data": [{"value": {"role": r, "source_file": s, "label": "x"}} for r, s in nguon],
    }]}), encoding="utf-8")


# ── Kiểm tra 1–3: phủ hồi cứu cho cổng đang chạy mà chưa có test ─────────────


def test_check_group_not_split_bat_duoc_nhom_nam_o_hai_tap():
    """Hai đoạn cắt từ CÙNG một video YouTube, một vào bank train một vào gold_test:
    `file_id` khác nhau nên so theo clip sẽ không thấy gì."""
    rows = [
        {"source_group_id": "youtube_AAA", "split": "foreground_bank_train"},
        {"source_group_id": "youtube_AAA", "split": "gold_test"},
        {"source_group_id": "youtube_BBB", "split": "gold_test"},
    ]
    loi = check_group_not_split(rows)
    assert len(loi) == 1 and "youtube_AAA" in loi[0]


def test_check_group_not_split_sach_tra_rong():
    rows = [
        {"source_group_id": "youtube_AAA", "split": "foreground_bank_train"},
        {"source_group_id": "youtube_BBB", "split": "gold_test"},
    ]
    assert check_group_not_split(rows) == []


def test_check_no_shared_audio_bat_duoc_hai_file_khac_ten_cung_noi_dung():
    """Đúng tình huống đã đo được ở DESED: khác tên, khác nhóm, nội dung giống hệt."""
    manifest = [{"file_id": "a", "checksum_sha256": "deadbeef" * 8},
                {"file_id": "b", "checksum_sha256": "deadbeef" * 8}]
    rows = [{"file_id": "a", "split": "foreground_bank_train"},
            {"file_id": "b", "split": "gold_test"}]
    assert len(check_no_shared_audio(rows, manifest)) == 1


def test_check_no_shared_audio_bo_qua_file_khong_co_trong_manifest():
    """file_id lạ không được làm cổng nổ — nó là việc của kiểm tra khác."""
    rows = [{"file_id": "khong_co", "split": "gold_test"}]
    assert check_no_shared_audio(rows, []) == []


def test_check_dedup_groups_intact_bat_duoc_nhom_bac_cau():
    dedup = [{"checksum_sha256": "c" * 64, "source_group_ids": "g1|g2"}]
    rows = [{"source_group_id": "g1", "split": "foreground_bank_train"},
            {"source_group_id": "g2", "split": "gold_test"}]
    assert len(check_dedup_groups_intact(rows, dedup)) == 1


def test_check_dedup_groups_intact_sach_tra_rong():
    dedup = [{"checksum_sha256": "c" * 64, "source_group_ids": "g1|g2"}]
    rows = [{"source_group_id": "g1", "split": "gold_test"},
            {"source_group_id": "g2", "split": "gold_test"}]
    assert check_dedup_groups_intact(rows, dedup) == []


# ── Kiểm tra 4: nguồn dùng thật trong JAMS ──────────────────────────────────


def test_source_id_from_path_cat_duoc_duong_dan_windows_tuyet_doi():
    """JAMS ghi đường dẫn TUYỆT ĐỐI của máy sinh dữ liệu. Không đổi `\\` thành `/`
    trước khi cắt thì trên POSIX cả chuỗi bị coi là một tên file và mọi phép so trượt."""
    duong = "D:\\IUH_K18\\KLTN\\data\\banks\\foreground\\gunshot\\fsd50k_12345.wav"
    assert source_id_from_path(duong) == "fsd50k_12345"


def test_source_id_from_path_cat_duoc_duong_dan_posix():
    assert source_id_from_path("/home/x/banks/foreground/gunshot/fsd50k_1.wav") \
        == "fsd50k_1"


def test_read_jams_sources_tach_foreground_va_background(tmp_path):
    viet_jams(tmp_path / "c.jams", [
        ("background", "D:\\banks\\background_10s\\parking\\bgc_parking_1.wav"),
        ("foreground", "D:\\banks\\foreground\\gunshot\\fsd50k_111.wav"),
        ("foreground", "D:\\banks\\foreground\\siren\\fsd50k_222.wav"),
    ])
    kq = read_jams_sources(tmp_path / "c.jams")
    assert kq["foreground"] == {"fsd50k_111", "fsd50k_222"}
    assert kq["background"] == {"bgc_parking_1"}


def test_read_jams_sources_file_hong_tra_rong_khong_no(tmp_path):
    """JAMS cắt dở vì tiến trình sinh bị kill — một file hỏng không được làm sập cả
    lượt kiểm 9.360 file, nhưng cũng không được im lặng thành 'sạch'."""
    (tmp_path / "hong.jams").write_text("{ khong phai json", encoding="utf-8")
    kq = read_jams_sources(tmp_path / "hong.jams")
    assert kq["foreground"] == set() and kq["background"] == set()


def test_collect_sources_gom_theo_clip(tmp_path):
    viet_jams(tmp_path / "dev_1.jams", [("foreground", "/b/fsd50k_111.wav")])
    viet_jams(tmp_path / "dev_2.jams", [("foreground", "/b/fsd50k_222.wav")])
    kq = collect_sources(tmp_path)
    assert set(kq) == {"dev_1", "dev_2"}
    assert kq["dev_1"]["foreground"] == {"fsd50k_111"}


def test_check_synthetic_sources_in_bank_bat_duoc_nguon_gold_test():
    """Ca thảm hoạ: một clip gold_test lọt vào nguyên liệu synthetic."""
    loi = check_synthetic_sources_in_bank({"fsd50k_111", "as_strong_gold_999"},
                                          {"fsd50k_111"}, "dev")
    assert len(loi) == 1 and "as_strong_gold_999" in loi[0]


def test_check_synthetic_sources_in_bank_sach_tra_rong():
    assert check_synthetic_sources_in_bank({"a"}, {"a", "b"}, "train") == []


# ── Kiểm tra 5: số phải khai báo, KHÔNG phải cổng ───────────────────────────


def test_shared_source_stats_dem_dung_va_tinh_ti_le():
    kq = shared_source_stats({"a", "b", "c"}, {"b", "c", "d"})
    assert kq["n_train"] == 3 and kq["n_dev"] == 3 and kq["n_chung"] == 2
    assert abs(kq["ti_le_dev"] - 2 / 3) < 1e-9


def test_shared_source_stats_mau_so_0_tra_none_khong_phai_0():
    """dev rỗng là CHƯA ĐO. Trả 0 ở đây sẽ khoe 'dev không dùng lại nguồn nào'."""
    assert shared_source_stats({"a"}, set())["ti_le_dev"] is None


def test_clips_touching_dem_clip_co_it_nhat_mot_nguon_dung_lai():
    clip_nguon = {
        "dev_1": {"fsd50k_111"},
        "dev_2": {"fsd50k_999"},
        "dev_3": {"fsd50k_999", "fsd50k_111"},
    }
    kq = clips_touching(clip_nguon, {"fsd50k_111"})
    assert kq["n_clip"] == 3 and kq["n_dinh"] == 2
    assert abs(kq["ti_le"] - 2 / 3) < 1e-9


def test_clips_touching_khong_co_clip_tra_none():
    assert clips_touching({}, {"x"})["ti_le"] is None
