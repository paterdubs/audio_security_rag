"""Chứng minh verify_ontology.py thực sự bắt lỗi.

Một script kiểm tra lúc nào cũng báo ĐẠT thì tệ hơn là không có, vì nó tạo cảm giác
an toàn giả. Mỗi test ở đây cố tình làm hỏng một thứ và đòi script phải phát hiện.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import verify_ontology as vo  # noqa: E402

ONTOLOGY = {
    "/m/032s66": "Gunshot, gunfire",
    "/m/0g6b5": "Fireworks",
    "/m/07rknqz": "Skidding",
}
STRONG_IDS = {"/m/032s66", "/m/0g6b5"}  # Skidding cố ý không có strong label


def make_class(**overrides) -> dict:
    """Một lớp hợp lệ tối thiểu; test ghi đè đúng thứ nó muốn làm hỏng."""
    spec = {
        "group": "A",
        "audioset_ids": ["/m/032s66"],
        "audioset_names": ["Gunshot, gunfire"],
        "fsd50k_labels": [],
        "esc50_labels": [],
        "urbansound_labels": [],
        "desed_labels": [],
        "mivia_labels": [],
        "confusable_with": [],
        "target_foreground": 200,
        "min_acceptable": 80,
    }
    spec.update(overrides)
    return spec


def codes(issues, level=vo.ERROR) -> list[str]:
    return [i.code for i in issues if i.level == level]


# ── C02/C03: id và tên phải khớp ontology gốc ────────────────────────────────


def test_phat_hien_id_khong_ton_tai():
    classes = {"gunshot": make_class(audioset_ids=["/m/khongcothat"], audioset_names=["Bịa"])}
    assert "C02" in codes(vo.check_audioset_ids(classes, ONTOLOGY))


def test_phat_hien_ten_khong_khop_id():
    """Đây là lỗi nguy hiểm nhất: id đúng định dạng, tên nghe hợp lý, nhưng tải về sai lớp."""
    classes = {"gunshot": make_class(audioset_names=["Fireworks"])}
    assert "C03" in codes(vo.check_audioset_ids(classes, ONTOLOGY))


def test_phat_hien_so_luong_id_va_ten_lech_nhau():
    classes = {"gunshot": make_class(audioset_ids=["/m/032s66", "/m/0g6b5"])}
    assert "C03" in codes(vo.check_audioset_ids(classes, ONTOLOGY))


def test_chap_nhan_anh_xa_dung():
    assert vo.check_audioset_ids({"gunshot": make_class()}, ONTOLOGY) == []


# ── C05: một id không được thuộc hai lớp ─────────────────────────────────────


def test_phat_hien_id_dung_chung_boi_hai_lop():
    classes = {"gunshot": make_class(), "fireworks": make_class()}
    assert "C05" in codes(vo.check_id_uniqueness(classes))


# ── C04: strong label phải được khai báo trung thực ──────────────────────────


def test_canh_bao_id_thieu_strong_label_ma_khong_khai():
    classes = {"vehicle_crash": make_class(audioset_ids=["/m/07rknqz"], audioset_names=["Skidding"])}
    issues = vo.check_strong_availability(classes, STRONG_IDS)
    assert "C04" in codes(issues, vo.WARN)


def test_khai_weak_only_dung_thi_khong_canh_bao():
    classes = {
        "vehicle_crash": make_class(
            audioset_ids=["/m/07rknqz"],
            audioset_names=["Skidding"],
            audioset_weak_only=["/m/07rknqz"],
        )
    }
    assert vo.check_strong_availability(classes, STRONG_IDS) == []


def test_bao_loi_khi_khai_weak_only_da_loi_thoi():
    """Nếu AudioSet bổ sung strong label cho lớp đó, khai báo cũ thành sai — phải sửa."""
    classes = {"gunshot": make_class(audioset_weak_only=["/m/032s66"])}
    assert "C04" in codes(vo.check_strong_availability(classes, STRONG_IDS))


# ── C06: quan hệ nhầm lẫn phải đối xứng ──────────────────────────────────────


def test_phat_hien_quan_he_nham_lan_mot_chieu():
    classes = {
        "gunshot": make_class(confusable_with=["fireworks"]),
        "fireworks": make_class(group="B", confusable_with=[]),
    }
    assert "C06" in codes(vo.check_confusable(classes))


def test_phat_hien_nham_voi_lop_khong_ton_tai():
    classes = {"gunshot": make_class(confusable_with=["lop_ma"])}
    assert "C06" in codes(vo.check_confusable(classes))


def test_phat_hien_tu_nham_voi_chinh_minh():
    classes = {"gunshot": make_class(confusable_with=["gunshot"])}
    assert "C06" in codes(vo.check_confusable(classes))


def test_quan_he_doi_xung_thi_dat():
    classes = {
        "gunshot": make_class(confusable_with=["fireworks"]),
        "fireworks": make_class(group="B", confusable_with=["gunshot"]),
    }
    assert vo.check_confusable(classes) == []


# ── C07/C01/C11: ràng buộc cấu trúc ──────────────────────────────────────────


def test_phat_hien_muc_tieu_nho_hon_nguong_toi_thieu():
    classes = {"gunshot": make_class(target_foreground=50, min_acceptable=80)}
    assert "C07" in codes(vo.check_structure(classes))


def test_phat_hien_thieu_khoa_bat_buoc():
    spec = make_class()
    del spec["confusable_with"]
    assert "C11" in codes(vo.check_structure({"gunshot": spec}))


def test_phat_hien_sai_so_luong_lop():
    assert "C01" in codes(vo.check_structure({"gunshot": make_class()}))


# ── C08: nhãn ESC-50 phải có thật ────────────────────────────────────────────


def test_phat_hien_nhan_esc50_bia():
    classes = {"door_slam": make_class(esc50_labels=["door_slam"])}
    assert "C08" in codes(vo.check_esc50(classes, {"door_wood_knock", "glass_breaking"}))


# ── Kiểm tra ngược lại trên file thật ────────────────────────────────────────


def test_file_that_phai_dat():
    """Bảo vệ ml/configs/ontology_map.yaml khỏi bị sửa hỏng."""
    import yaml

    config = yaml.safe_load(vo.CONFIG_PATH.read_text(encoding="utf-8"))
    errors = [i for i in vo.run_checks(config) if i.level == vo.ERROR]
    assert errors == [], f"ontology_map.yaml có lỗi: {errors}"



# ── taxonomy.md và ontology_map.yaml phải luôn khớp nhau ─────────────────────


def _taxonomy_class_ids() -> set[str]:
    import re

    path = vo.REPO_ROOT / "docs" / "taxonomy.md"
    text = path.read_text(encoding="utf-8")
    # Mỗi lớp có một mục dạng "### A1. `gunshot` — ..."
    return set(re.findall(r"^### [AB]\d+\.\s+`([a-z_]+)`", text, re.MULTILINE))


def test_taxonomy_va_ontology_khop_nhau():
    """taxonomy.md nói với người, ontology_map.yaml nói với máy — lệch nhau là hỏng cả hai."""
    import yaml

    config = yaml.safe_load(vo.CONFIG_PATH.read_text(encoding="utf-8"))
    in_config = set(config["classes"])
    in_docs = _taxonomy_class_ids()

    assert in_docs, "không bóc được lớp nào từ taxonomy.md — kiểm tra lại định dạng tiêu đề"
    assert in_config == in_docs, (
        f"chỉ có trong yaml: {sorted(in_config - in_docs)} · "
        f"chỉ có trong taxonomy.md: {sorted(in_docs - in_config)}"
    )


def test_taxonomy_co_du_bang_tra_cap_nham_lan():
    """Mọi cặp confusable_with phải xuất hiện ở bảng tra nhanh §3 của taxonomy.md."""
    import yaml

    config = yaml.safe_load(vo.CONFIG_PATH.read_text(encoding="utf-8"))
    text = (vo.REPO_ROOT / "docs" / "taxonomy.md").read_text(encoding="utf-8")
    quick_table = text.split("## 3. Bảng tra nhanh")[1].split("## 4.")[0]

    missing = [
        f"{name}↔{other}"
        for name, spec in config["classes"].items()
        for other in spec.get("confusable_with", [])
        if name < other and not (f"`{name}` ↔ `{other}`" in quick_table or f"`{other}` ↔ `{name}`" in quick_table)
    ]
    assert not missing, f"cặp nhầm lẫn thiếu trong bảng tra nhanh: {missing}"


# ── C12: fsd50k_labels phải suy ra từ mã AudioSet ────────────────────────────

FSD50K_VOCAB = {
    "/m/032s66": "Gunshot_and_gunfire",
    "/m/07pbtc8": "Walk_and_footsteps",
    "/m/06h7j": "Run",
}


def test_phat_hien_nhan_fsd50k_doan_sai():
    """FSD50K đặt tên theo quy ước riêng nên đoán tên là sai lặng lẽ."""
    classes = {
        "running_footsteps": make_class(
            audioset_ids=["/m/07pbtc8", "/m/06h7j"],
            audioset_names=["Walk, footsteps", "Run"],
            fsd50k_labels=["Footsteps", "Running"],   # tên bịa, nghe hợp lý
        )
    }
    assert "C12" in codes(vo.check_fsd50k(classes, FSD50K_VOCAB))


def test_nhan_fsd50k_suy_dung_tu_ma_thi_dat():
    classes = {
        "running_footsteps": make_class(
            audioset_ids=["/m/07pbtc8", "/m/06h7j"],
            audioset_names=["Walk, footsteps", "Run"],
            fsd50k_labels=["Walk_and_footsteps", "Run"],
        )
    }
    assert vo.check_fsd50k(classes, FSD50K_VOCAB) == []


def test_ma_khong_co_trong_fsd50k_thi_bo_qua_chu_khong_bao_loi():
    """FSD50K chỉ có 200 lớp trên 632 của AudioSet — thiếu là bình thường."""
    classes = {"vehicle_crash": make_class(audioset_ids=["/m/07pjjrj"], audioset_names=["Smash, crash"],
                                           fsd50k_labels=[])}
    assert vo.check_fsd50k(classes, FSD50K_VOCAB) == []


def test_khai_thua_nhan_fsd50k_cung_bi_bat():
    classes = {"vehicle_crash": make_class(audioset_ids=["/m/07pjjrj"], audioset_names=["Smash, crash"],
                                           fsd50k_labels=["Smash_and_crash"])}
    assert "C12" in codes(vo.check_fsd50k(classes, FSD50K_VOCAB))
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
