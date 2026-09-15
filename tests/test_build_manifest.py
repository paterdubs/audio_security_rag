"""Test cho scripts/build_manifest.py.

Trọng tâm là ánh xạ license: lỗi ở đây KHÔNG làm script chết, nó chỉ âm thầm ghi
"UNKNOWN" vào 100% số dòng — đúng loại lỗi dễ lọt nhất và tốn kém nhất khi công bố.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_manifest as bm  # noqa: E402


# ── Khoá tra license: đây là lỗi đã thực sự xảy ra ở D1 ──────────────────────


@pytest.mark.parametrize(
    "audio_stem,expected",
    [
        ("1-100032-A-0", "1-100032-A"),     # tên file audio có hậu tố mã lớp
        ("5-9032-B-40", "5-9032-B"),
        ("1-100210-B-36", "1-100210-B"),
    ],
)
def test_cat_dung_hau_to_ma_lop(audio_stem: str, expected: str):
    assert bm.esc50_license_key(audio_stem) == expected


# ESC-50 có đúng MỘT clip lỗi trong file LICENSE của chính nó: `5-141683-A-35.wav`
# được ghi thành `5-141683-Ad.ogg` (thừa chữ "d" ở phần take). Đây là lỗi thượng nguồn,
# không phải lỗi của ta, và clip đó thuộc lớp `washing_machine` — ngoài 15 lớp của dự án.
ESC50_UPSTREAM_LICENSE_TYPOS = {"5-141683-A-35"}


def test_khoa_license_khop_voi_file_license_that():
    """Chặn tái diễn: trước khi sửa, phép tra này trượt 100% và mọi clip mất license."""
    root = bm.RAW_DIR / "esc50" / "ESC-50-master"
    if not (root / "LICENSE").exists():
        pytest.skip("chưa tải ESC-50")

    licenses = bm.parse_esc50_licenses(root / "LICENSE")
    stems = [p.stem for p in (root / "audio").glob("*.wav")]
    unmatched = {s for s in stems if bm.esc50_license_key(s) not in licenses}
    assert unmatched <= ESC50_UPSTREAM_LICENSE_TYPOS, f"tra trượt ngoài dự kiến: {unmatched}"


def test_moi_clip_thuoc_lop_cua_ta_deu_co_license():
    """Điều thực sự quan trọng: 15 lớp của ta phải truy vết được 100%."""
    root = bm.RAW_DIR / "esc50" / "ESC-50-master"
    if not (root / "LICENSE").exists():
        pytest.skip("chưa tải ESC-50")

    import csv

    import yaml

    ontology = yaml.safe_load(
        (bm.REPO_ROOT / "ml" / "configs" / "ontology_map.yaml").read_text(encoding="utf-8")
    )
    label_map = bm.reverse_label_map(ontology, "esc50_labels")
    licenses = bm.parse_esc50_licenses(root / "LICENSE")

    with (root / "meta" / "esc50.csv").open(encoding="utf-8", newline="") as handle:
        ours = [r for r in csv.DictReader(handle) if r["category"] in label_map]

    assert ours, "không có clip nào khớp lớp của ta — kiểm tra lại esc50_labels"
    missing = [r["filename"] for r in ours if bm.esc50_license_key(Path(r["filename"]).stem) not in licenses]
    assert not missing, f"{len(missing)} clip thuộc lớp của ta thiếu license: {missing[:5]}"


# ── Bóc tách file LICENSE ────────────────────────────────────────────────────


def test_boc_tach_dong_license(tmp_path: Path):
    content = (
        "101 - Dog:\n"
        "- [1-100032-A.ogg]: clip derived from rose_bark.wav "
        "(http://www.freesound.org/people/nfrae/sounds/100032/) by nfrae [CC0]\n"
        "dòng rác không theo định dạng\n"
    )
    path = tmp_path / "LICENSE"
    path.write_text(content, encoding="utf-8")

    entries = bm.parse_esc50_licenses(path)
    assert entries["1-100032-A"] == {
        "url": "http://www.freesound.org/people/nfrae/sounds/100032/",
        "author": "nfrae",
        "license": "CC0",
    }


def test_bo_qua_dong_khong_dung_dinh_dang(tmp_path: Path):
    path = tmp_path / "LICENSE"
    path.write_text("## Dataset license\nvăn bản tự do\n", encoding="utf-8")
    assert bm.parse_esc50_licenses(path) == {}


# ── Quyền phát hành lại ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "license_name,expected",
    [
        ("CC0", "yes"),
        ("CC-BY", "yes"),
        ("CC-BY-NC", "labels_only"),       # phi thương mại → chỉ công bố nhãn
        ("CC-Sampling+", "labels_only"),
        ("UNKNOWN", "labels_only"),        # không rõ thì mặc định là chặt nhất
    ],
)
def test_mac_dinh_an_toan_khi_license_la_lop_moi(license_name: str, expected: str):
    assert bm.REDISTRIBUTABLE.get(license_name, bm.DEFAULT_REDISTRIBUTABLE) == expected


# ── Ánh xạ nhãn nguồn → lớp của ta ───────────────────────────────────────────


def test_gop_nhieu_nhan_nguon_ve_mot_lop():
    ontology = {
        "classes": {
            "alarm_bell": {"esc50_labels": ["clock_alarm", "church_bells"]},
            "fireworks": {"esc50_labels": ["fireworks"]},
            "gunshot": {"esc50_labels": []},
        }
    }
    mapping = bm.reverse_label_map(ontology, "esc50_labels")
    assert mapping == {
        "clock_alarm": "alarm_bell",
        "church_bells": "alarm_bell",
        "fireworks": "fireworks",
    }


def test_lop_khong_co_nhan_nguon_thi_khong_xuat_hien():
    ontology = {"classes": {"vehicle_crash": {"esc50_labels": []}}}
    assert bm.reverse_label_map(ontology, "esc50_labels") == {}


# ── Manifest đã sinh ─────────────────────────────────────────────────────────


def test_manifest_that_khong_con_license_unknown():
    """Nguyên tắc N3: dữ liệu không truy vết được nguồn gốc thì không vào dataset."""
    if not bm.MANIFEST_PATH.exists():
        pytest.skip("chưa sinh manifest")

    import csv

    with bm.MANIFEST_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    unknown = [r["file_id"] for r in rows if r["license"] == "UNKNOWN"]
    assert not unknown, f"{len(unknown)} dòng thiếu license"
    assert all(r["source_group_id"] for r in rows), "có dòng thiếu source_group_id (vi phạm N2)"
    assert all(r["checksum_sha256"] for r in rows), "có dòng thiếu checksum"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# ── Phát hiện trùng khớp từng byte ───────────────────────────────────────────


def row(file_id: str, path: str, checksum: str, group: str) -> dict:
    return {"file_id": file_id, "path_raw": path, "checksum_sha256": checksum, "source_group_id": group}


def test_file_khac_nhau_thi_khong_bao_trung():
    rows = [row("a", "p/a.wav", "h1", "g1"), row("b", "p/b.wav", "h2", "g2")]
    assert bm.find_exact_duplicates(rows, []) == []


def test_trung_trong_cung_nhom_nguon_la_vo_hai():
    rows = [row("a", "p/a.wav", "h1", "g1"), row("b", "p/b.wav", "h1", "g1")]
    groups = bm.find_exact_duplicates(rows, [])
    assert len(groups) == 1 and groups[0]["risk"] == "same_group"


def test_trung_bac_cau_giua_hai_nhom_nguon_bi_danh_dau_nguy_hiem():
    """Ca làm hỏng phép chia tập: cùng audio, hai nhóm nguồn → hai bên train/test."""
    rows = [row("a", "p/a.wav", "h1", "freesound_13613"), row("b", "p/b.wav", "h1", "freesound_34853")]
    groups = bm.find_exact_duplicates(rows, [])
    assert groups[0]["risk"] == "cross_group"
    assert groups[0]["source_group_ids"] == "freesound_13613|freesound_34853"


def test_ban_sao_bi_manifest_gop_mat_van_duoc_phat_hien():
    """Chặn tái diễn: manifest chỉ giữ một dòng/checksum, nên nhóm nguồn của bản
    bị gộp phải lấy lại từ danh sách vừa quét — thiếu bước này thì báo nhầm là vô hại."""
    kept = [row("desed_alarm_bell_h1", "p/13613_0.wav", "h1", "freesound_13613")]
    scanned = [
        bm.RawEntry(
            file_id="desed_alarm_bell_h1", path_raw="p/34853_0.wav", source_dataset="desed_soundbank",
            source_id="train/34853_0", source_group_id="freesound_34853", claimed_class="alarm_bell",
            label_type_orig="isolated_event", checksum_sha256="h1",
        )
    ]
    groups = bm.find_exact_duplicates(kept, scanned)
    assert len(groups) == 1
    assert groups[0]["risk"] == "cross_group", "bản sao bị gộp mất đã không được tính"
    assert groups[0]["n_copies"] == 2


def test_bo_qua_file_rac_appledouble(tmp_path):
    """DESED đóng gói trên macOS nên tarball có kèm `._*.wav` vài trăm byte."""
    (tmp_path / "that.wav").write_bytes(b"x")
    (tmp_path / "._that.wav").write_bytes(b"y")
    assert [p.name for p in bm.audio_files_in(tmp_path)] == ["that.wav"]


# ── Chuẩn hoá license ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://creativecommons.org/publicdomain/zero/1.0/", "CC0"),
        ("http://creativecommons.org/licenses/by/3.0/", "CC-BY"),
        ("http://creativecommons.org/licenses/by-nc/3.0/", "CC-BY-NC"),
        ("http://creativecommons.org/licenses/sampling+/1.0/", "CC-Sampling+"),
        ("https://example.com/giay-phep-la", "UNKNOWN"),
    ],
)
def test_quy_url_license_ve_ten_ngan(url: str, expected: str):
    assert bm.normalize_license_url(url) == expected


def test_boc_duoc_license_trong_link_chuyen_huong_youtube():
    """DESED có vài dòng là link YouTube bọc URL creativecommons đã mã hoá phần trăm."""
    url = ("https://www.youtube.com/redirect?v=em8En_Sz6yo&event=video_description"
           "&q=https%3A%2F%2Fcreativecommons.org%2Flicenses%2Fby%2F4.0%2F")
    assert bm.normalize_license_url(url) == "CC-BY"


def test_by_nc_khong_bi_nhan_nham_thanh_by():
    """`licenses/by-nc` chứa cả chuỗi `licenses/by` — thứ tự khớp phải đúng."""
    assert bm.normalize_license_url("http://creativecommons.org/licenses/by-nc/3.0/") != "CC-BY"


# ── UrbanSound8K ─────────────────────────────────────────────────────────────
#
# Nguồn chính cho `siren` (FSD50K chỉ cho 27 clip), nên hai lỗi ở đây tốn kém:
# nhãn gõ sai → 0 clip mà không báo gì; quên lọc salience → bank foreground lẫn
# clip mà sự kiện đích chỉ là tiếng nền.

US8K_HEADER = ["slice_file_name", "fsID", "start", "end", "salience", "fold", "classID", "class"]


def _us8k_row(name: str, fs_id: str, salience: str, class_name: str, fold: str = "1") -> dict:
    return dict(zip(US8K_HEADER, [name, fs_id, "0.0", "4.0", salience, fold, "0", class_name]))


def test_nhan_gõ_sai_thi_dung_ngay_chu_khong_ra_0_clip():
    rows = [_us8k_row("a.wav", "1", "1", "siren")]
    with pytest.raises(SystemExit) as err:
        bm.verify_us8k_labels(rows, {"sirens": "siren"})
    assert "sirens" in str(err.value)
    assert "siren" in str(err.value)      # nêu luôn nhãn thật để sửa được ngay


def test_nhan_dung_thi_khong_can_gi():
    rows = [_us8k_row("a.wav", "1", "1", "siren"), _us8k_row("b.wav", "2", "1", "gun_shot")]
    bm.verify_us8k_labels(rows, {"siren": "siren", "gun_shot": "gunshot"})


def test_chi_salience_1_vao_foreground_bank(tmp_path, monkeypatch):
    rows = [
        _us8k_row("fg.wav", "10", "1", "siren"),
        _us8k_row("bg.wav", "11", "2", "siren"),
        _us8k_row("khac.wav", "12", "1", "jackhammer"),   # lớp ngoài ontology
    ]
    monkeypatch.setattr(bm, "load_us8k_rows", lambda: rows)
    ghi = {}
    monkeypatch.setattr(bm, "write_exclusions", lambda stage, rejected, processed: ghi.update(
        stage=stage, rejected=rejected, processed=processed))
    monkeypatch.setattr(bm, "US8K_AUDIO", tmp_path)       # không file nào tồn tại

    ontology = {"classes": {"siren": {"urbansound_labels": ["siren"]}}}
    list(bm.adapt_urbansound8k(ontology, None))

    assert list(ghi["rejected"]) == ["urbansound8k_bg.wav"]
    assert ghi["rejected"]["urbansound8k_bg.wav"][0] == "background_salience"
    # `khac.wav` ngoài ontology: không chọn, cũng KHÔNG ghi exclusion —
    # exclusions.csv chỉ nói về clip ta thực sự cân nhắc.
    assert ghi["processed"] == {"urbansound8k_fg.wav", "urbansound8k_bg.wav"}


def test_fsid_lam_nhom_chong_ro_ri(tmp_path, monkeypatch):
    audio = tmp_path / "fold1"
    audio.mkdir()
    (audio / "a.wav").write_bytes(b"x")
    monkeypatch.setattr(bm, "US8K_AUDIO", tmp_path)
    monkeypatch.setattr(bm, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bm, "sha256_of", lambda p: "deadbeef" * 8)
    monkeypatch.setattr(bm, "audio_properties", lambda p: (4.0, 44100, 2))

    entry = bm._us8k_entry({**_us8k_row("a.wav", "27724", "1", "siren"), "class_id": "siren"}, audio / "a.wav")
    # Nhiều lát cắt chung một fsID; gom theo fsID mới chặn được rò rỉ train/test (N2).
    assert entry.source_group_id == "freesound_27724"
    assert entry.orig_split == "fold1"
    # Lát 4 giây chỉ CHỨA sự kiện — không được khai cả file là sự kiện.
    assert entry.orig_onset == "" and entry.orig_offset == ""
    assert entry.label_type_orig == "weak_with_salience"
    assert entry.redistributable == "labels_only"     # CC BY-NC không cho phát hành lại audio


# ── Gộp nhóm trùng lặp: ghi đè là xoá bằng chứng ─────────────────────────────
#
# Manifest chỉ giữ MỘT dòng cho mỗi checksum, nên bản trùng chỉ còn tồn tại trong
# dedup_groups.csv. Ghi đè file đó khi quét nguồn khác sẽ xoá sạch nhóm của nguồn cũ,
# và check_leakage vẫn báo xanh vì nó đọc chính file đã rỗng. Đã xảy ra thật.


def _nhom(checksum: str, groups: str = "g1|g2") -> dict:
    return {"checksum_sha256": checksum, "n_copies": "2", "representative_file_id": f"f_{checksum}",
            "source_group_ids": groups, "members": "a|b", "risk": "cross_group"}


def test_quet_nguon_moi_khong_lam_mat_nhom_cua_nguon_cu():
    cu = [_nhom("desed_aa"), _nhom("desed_bb")]
    moi = [_nhom("us8k_cc")]
    ket_qua = {r["checksum_sha256"] for r in bm.merge_dedup_groups(cu, moi)}
    assert ket_qua == {"desed_aa", "desed_bb", "us8k_cc"}


def test_quet_lai_cung_checksum_thi_lan_moi_thang():
    cu = [_nhom("aa", groups="g1|g2")]
    moi = [_nhom("aa", groups="g1|g2|g3")]
    ket_qua = bm.merge_dedup_groups(cu, moi)
    assert len(ket_qua) == 1
    assert ket_qua[0]["source_group_ids"] == "g1|g2|g3"


def test_thu_tu_on_dinh_de_git_diff_doc_duoc():
    xuoi = bm.merge_dedup_groups([_nhom("bb")], [_nhom("aa")])
    nguoc = bm.merge_dedup_groups([_nhom("aa")], [_nhom("bb")])
    assert [r["checksum_sha256"] for r in xuoi] == [r["checksum_sha256"] for r in nguoc] == ["aa", "bb"]


def test_lan_quet_dau_tien_khong_co_gi_de_giu():
    assert bm.merge_dedup_groups([], [_nhom("aa")]) == [_nhom("aa")]


# ── Adapter: AudioSet-strong ──────────────────────────────────────────────────


def test_mot_o_nhieu_su_kien_sinh_nhieu_dong_cung_mot_file(tmp_path, monkeypatch):
    """Khác mọi adapter khác: một file .wav sinh NHIỀU dòng manifest, mỗi dòng
    một sự kiện. Thiếu điều này thì gold_test không đo được EOR (nhiều sự kiện
    chồng lấn trong một clip)."""
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(bm, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bm, "sha256_of", lambda p: "cafebabe" * 8)
    monkeypatch.setattr(bm, "audio_properties", lambda p: (10.0, 16000, 1))
    monkeypatch.setattr(bm, "load_audioset_strong_segments", lambda: [{
        "file_id": "as_strong_abc_0", "path": "clip.wav", "ytid": "abc",
        "events": [
            {"class_id": "gunshot", "onset": 1.0, "offset": 2.0},
            {"class_id": "shout_yell", "onset": 4.0, "offset": 6.0},
        ],
    }])

    rows = list(bm.adapt_audioset_strong({}, None))
    assert len(rows) == 2
    assert {r.claimed_class for r in rows} == {"gunshot", "shout_yell"}
    assert {r.path_raw for r in rows} == {"clip.wav"}
    # Cùng video → cùng nhóm chống rò rỉ, dù khác sự kiện/khác dòng manifest.
    assert {r.source_group_id for r in rows} == {"youtube_abc"}


# ── Xoá dòng cũ khi quét lại một nguồn ───────────────────────────────────────


def test_doi_ontology_khong_de_lai_dong_cu_mang_class_da_mat():
    """Đúng lỗi thật gặp khi tách laughter_cheering: file_id đổi theo class_id, nên
    dòng cũ không bị .update() ghi đè và nằm lại vĩnh viễn, trùng checksum."""
    old_rows = {
        "esc50_laughter_cheering_aaaa": {"source_dataset": "esc50", "checksum_sha256": "aaaa"},
        "esc50_alarm_bell_bbbb": {"source_dataset": "esc50", "checksum_sha256": "bbbb"},
        "fsd50k_gunshot_cccc": {"source_dataset": "fsd50k", "checksum_sha256": "cccc"},
    }
    ket_qua = bm.drop_stale_rows(old_rows, "esc50", limit=None)
    assert set(ket_qua) == {"fsd50k_gunshot_cccc"}


def test_chay_thu_co_limit_khong_xoa_gi():
    """--limit là chạy thử, chỉ quét một phần — xoá lúc đó sẽ mất phần chưa quét tới."""
    old_rows = {"esc50_a": {"source_dataset": "esc50"}, "esc50_b": {"source_dataset": "esc50"}}
    assert bm.drop_stale_rows(old_rows, "esc50", limit=50) == old_rows


def test_thieu_audio_thi_bo_qua_khong_chet(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bm, "load_audioset_strong_segments", lambda: [{
        "file_id": "as_strong_abc_0", "path": "khong_ton_tai.wav", "ytid": "abc",
        "events": [{"class_id": "gunshot", "onset": 1.0, "offset": 2.0}],
    }])
    assert list(bm.adapt_audioset_strong({}, None)) == []
