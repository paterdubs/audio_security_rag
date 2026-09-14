"""Test cho scripts/audioset_ontology.py.

Bẫy mà module này tồn tại để tránh: coi nhãn phân cấp là nhiều sự kiện. Bỏ qua nó thì
`gunshot`, `siren`, `fireworks` đều ra 0 clip dùng được từ FSD50K — một kết luận sai
mà không có thông báo lỗi nào, và ta sẽ bỏ mất nguồn tốt nhất cho ba lớp đó.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import audioset_ontology as ao  # noqa: E402

# Sounds of things ─┬─ Explosion ─┬─ Gunshot
#                   │             └─ Fireworks
#                   └─ Alarm ─────── Siren
PARENTS = {
    "explosion": ["things"],
    "alarm": ["things"],
    "gunshot": ["explosion"],
    "fireworks": ["explosion"],
    "siren": ["alarm"],
}


def test_tim_du_to_tien_qua_nhieu_tang():
    assert ao.ancestors_of("gunshot", PARENTS) == {"explosion", "things"}


def test_nut_goc_khong_co_to_tien():
    assert ao.ancestors_of("things", PARENTS) == set()


def test_ma_la_khong_co_to_tien():
    assert ao.ancestors_of("khong_ton_tai", PARENTS) == set()


# ── Thu gọn về nhãn cụ thể nhất ──────────────────────────────────────────────


def test_bo_nhan_cha_khi_co_nhan_con():
    """Ca thật: mọi clip Gunshot của FSD50K đều kèm nhãn Explosion."""
    assert ao.collapse_to_most_specific({"gunshot", "explosion"}, PARENTS) == {"gunshot"}


def test_giu_nguyen_hai_nhan_anh_em():
    """Gunshot và Fireworks cùng là con của Explosion — đây mới thật sự là hai sự kiện."""
    assert ao.collapse_to_most_specific({"gunshot", "fireworks"}, PARENTS) == {"gunshot", "fireworks"}


def test_bo_duoc_to_tien_cach_nhieu_tang():
    assert ao.collapse_to_most_specific({"gunshot", "explosion", "things"}, PARENTS) == {"gunshot"}


def test_mot_nhan_thi_giu_nguyen():
    assert ao.collapse_to_most_specific({"explosion"}, PARENTS) == {"explosion"}


def test_tap_rong():
    assert ao.collapse_to_most_specific(set(), PARENTS) == set()


def test_khong_lam_mat_nhan_o_nhanh_khac():
    assert ao.collapse_to_most_specific({"siren", "alarm", "gunshot"}, PARENTS) == {"siren", "gunshot"}


def test_khong_treo_khi_ontology_co_chu_trinh():
    """Ontology là DAG do người biên tập; chu trình do lỗi dữ liệu không được làm treo script."""
    cyclic = {"a": ["b"], "b": ["a"]}
    assert ao.ancestors_of("a", cyclic) == {"a", "b"}


# ── Kiểm tra trên ontology thật ──────────────────────────────────────────────

REAL_ONTOLOGY = Path(__file__).resolve().parent.parent / "data" / "reference" / "audioset_ontology.json"


@pytest.mark.parametrize(
    "child,ancestor,child_name,ancestor_name",
    [
        ("/m/032s66", "/m/014zdl", "Gunshot", "Explosion"),
        ("/m/0g6b5", "/m/014zdl", "Fireworks", "Explosion"),
        ("/m/03kmc9", "/m/07pp_mv", "Siren", "Alarm"),
        ("/m/07rn7sz", "/m/039jq", "Shatter", "Glass"),
        ("/m/07rjzl8", "/m/02dgv", "Slam", "Door"),
    ],
)
def test_quan_he_phan_cap_that_van_dung(child, ancestor, child_name, ancestor_name):
    """Nếu AudioSet đổi cấu trúc ontology, phép chọn clip FSD50K phải được xem lại."""
    if not REAL_ONTOLOGY.exists():
        pytest.skip("chưa tải ontology")
    parents = ao.load_parents(REAL_ONTOLOGY)
    assert ancestor in ao.ancestors_of(child, parents), f"{child_name} không còn nằm dưới {ancestor_name}"


def test_thu_gon_tren_ontology_that():
    if not REAL_ONTOLOGY.exists():
        pytest.skip("chưa tải ontology")
    parents = ao.load_parents(REAL_ONTOLOGY)
    # Gunshot + Explosion → chỉ Gunshot
    assert ao.collapse_to_most_specific({"/m/032s66", "/m/014zdl"}, parents) == {"/m/032s66"}
    # Gunshot + Fireworks là hai anh em → giữ cả hai
    assert ao.collapse_to_most_specific({"/m/032s66", "/m/0g6b5"}, parents) == {"/m/032s66", "/m/0g6b5"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
