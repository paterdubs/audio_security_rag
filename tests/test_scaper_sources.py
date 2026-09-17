"""Test cho khâu CHỌN NGUỒN của scaper_generate.py — STATUS §7 B2.

Trước B2, việc chọn file giao cho Scaper: `source_file=("choose", [])`. Hệ quả là
script không biết Scaper bốc file nào, nên không đặt được `source_time` hay
`event_duration` cho đúng file đó — và đó là gốc của cả ba lỗi ①②④.

B2 giành lại quyền chọn file. Đổi lại, script phải tự gánh hai thứ Scaper đang lo
hộ, và cả hai đều hỏng KHÔNG TRIỆU CHỨNG nếu làm sai:

  · cổng chống rò rỉ — một clip gold_test lọt vào bank làm hỏng cả bài test, mà
    biểu hiện duy nhất là điểm số đẹp lên
  · tính tất định — cùng seed phải cho cùng dãy file, nếu không thì dataset không
    tái lập được và mọi con số trong luận văn mất chỗ dựa

Hai điều đó là trọng tâm của file test này.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import scaper_generate as sg  # noqa: E402

SU_KIEN = {"pitch_shift_semitones": [-1, 1], "time_stretch": [0.9, 1.1]}


def _viet_csv(tmp_path: Path, dong: list[dict]) -> Path:
    ra = tmp_path / "bank_trim.csv"
    with ra.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sg.COT_BANG_TRA)
        writer.writeheader()
        for r in dong:
            writer.writerow({**{c: "" for c in sg.COT_BANG_TRA}, **r})
    return ra


def _clip(file_id: str, lop: str, eff: float, lead: float = 0.0, **thay_doi) -> dict:
    return {"file_id": file_id, "class_id": lop, "path": f"data/banks/foreground/{lop}/{file_id}.wav",
            "dur": lead + eff, "eff": eff, "lead": lead, "trail": 0.0,
            "rms_db": -20.0, "peak": 0.5, "loi": "", **thay_doi}


# ── Đọc bảng tra ─────────────────────────────────────────────────────────────


def test_doc_bang_tra_gom_theo_lop_va_sap_xep_tat_dinh(tmp_path):
    """Nhóm theo lớp, trong lớp sắp theo file_id — nền của tính tái lập."""
    path = _viet_csv(tmp_path, [_clip("g_b", "gunshot", 0.8), _clip("s_a", "siren", 4.0),
                                _clip("g_a", "gunshot", 0.5)])

    bang = sg.doc_bang_tra(path)

    assert set(bang) == {"gunshot", "siren"}
    assert [c.file_id for c in bang["gunshot"]] == ["g_a", "g_b"]


def test_doc_bang_tra_bo_clip_loi_va_clip_khong_co_tieng(tmp_path):
    """Clip lỗi hoặc eff=0 không chở nổi sự kiện nào — phải loại từ đầu.

    Để lọt vào thì Scaper nhận `event_duration=0`, và nó sẽ tự rút một giá trị
    khác mà ta không kiểm soát được.
    """
    path = _viet_csv(tmp_path, [
        _clip("tot", "gunshot", 0.8),
        _clip("hong", "gunshot", 0.0, loi="LibsndfileError: hỏng"),
        _clip("im_lang", "gunshot", 0.0),
    ])

    bang = sg.doc_bang_tra(path)

    assert [c.file_id for c in bang["gunshot"]] == ["tot"]


def test_doc_bang_tra_bao_loi_khi_thieu_file(tmp_path):
    """Thiếu bảng tra phải chết ngay với thông điệp chỉ đúng việc cần làm."""
    with pytest.raises(SystemExit, match="probe_bank"):
        sg.doc_bang_tra(tmp_path / "khong_co.csv")


# ── Cổng chống rò rỉ ─────────────────────────────────────────────────────────


def test_loc_hop_le_bo_clip_ngoai_foreground_bank_train(tmp_path):
    """Clip không thuộc bank_train bị loại VÀ được liệt kê ra, không loại âm thầm."""
    bang = sg.doc_bang_tra(_viet_csv(tmp_path, [
        _clip("hop_le", "gunshot", 0.8), _clip("ro_ri", "gunshot", 0.9),
    ]))

    sach, vi_pham = sg.loc_hop_le(bang, {"hop_le"})

    assert [c.file_id for c in sach["gunshot"]] == ["hop_le"]
    assert vi_pham == ["ro_ri"]


def test_loc_hop_le_bo_han_lop_khong_con_clip_nao(tmp_path):
    """Lớp mất sạch clip phải biến khỏi bảng, không để lại lớp rỗng.

    Lớp rỗng lọt xuống dưới sẽ thành `rng.choice([])` → IndexError giữa chừng một
    lượt sinh 18 giờ, chứ không phải lỗi lúc bắt đầu.
    """
    bang = sg.doc_bang_tra(_viet_csv(tmp_path, [
        _clip("a", "gunshot", 0.8), _clip("b", "siren", 4.0),
    ]))

    sach, vi_pham = sg.loc_hop_le(bang, {"a"})

    assert set(sach) == {"gunshot"}
    assert vi_pham == ["b"]


# ── Chọn nguồn theo thời lượng mong muốn ─────────────────────────────────────


def test_chon_nguon_chi_lay_clip_du_dai(tmp_path):
    """Muốn 3.0 s thì chỉ được chọn clip có eff ≥ 3.0 s, và trả đúng 3.0 s."""
    bang = sg.doc_bang_tra(_viet_csv(tmp_path, [
        _clip("ngan", "siren", 0.5), _clip("vua", "siren", 3.2), _clip("dai", "siren", 8.0),
    ]))
    rng = random.Random(0)

    for _ in range(50):
        nguon, d_thuc = sg.chon_nguon(rng, bang["siren"], 3.0)
        assert nguon.file_id in {"vua", "dai"}
        assert d_thuc == pytest.approx(3.0)


def test_chon_nguon_ha_xuong_clip_dai_nhat_khi_khong_du(tmp_path):
    """Không clip nào đủ dài → lấy clip dài nhất, và d_thuc hạ theo eff của nó.

    Đây chính là ca `siren` muốn 8.99 s mà bank chỉ có tới 9.48 s ở 2 clip. Hạ
    xuống là đúng; hạ xuống mà vẫn khai 8.99 s mới là lỗi — nên assert d_thuc.
    """
    bang = sg.doc_bang_tra(_viet_csv(tmp_path, [
        _clip("a", "scream", 1.0), _clip("b", "scream", 3.7),
    ]))
    rng = random.Random(0)

    nguon, d_thuc = sg.chon_nguon(rng, bang["scream"], 9.0)

    assert nguon.file_id == "b"
    assert d_thuc == pytest.approx(3.7)


def test_chon_nguon_khong_bao_gio_tra_qua_eff(tmp_path):
    """Bất biến: d_thuc ≤ eff. Vượt là Scaper âm thầm cắt xuống — đúng lỗi ②."""
    bang = sg.doc_bang_tra(_viet_csv(tmp_path, [
        _clip(f"c{i}", "gunshot", 0.2 + 0.3 * i) for i in range(8)
    ]))
    rng = random.Random(1)

    for muon in (0.1, 0.5, 1.0, 2.0, 5.0, 20.0):
        nguon, d_thuc = sg.chon_nguon(rng, bang["gunshot"], muon)
        assert d_thuc <= nguon.eff + 1e-9


def test_chon_nguon_tat_dinh_theo_seed(tmp_path):
    """Cùng seed → cùng dãy file. Thiếu điều này là dataset không tái lập được."""
    bang = sg.doc_bang_tra(_viet_csv(tmp_path, [_clip(f"c{i}", "gunshot", 1.0 + i) for i in range(20)]))

    lay = lambda: [sg.chon_nguon(random.Random(42), bang["gunshot"], 1.0)[0].file_id  # noqa: E731
                   for _ in range(10)]
    assert lay() == lay()


def test_chon_nguon_trai_deu_chu_khong_dinh_mot_file(tmp_path):
    """Phải dùng nhiều clip khác nhau, kể cả nhánh 'không đủ dài'.

    Luôn lấy clip dài nhất khi thiếu sẽ khiến 2 clip siren dài xuất hiện hàng nghìn
    lần — model học thuộc đúng hai file đó thay vì học lớp siren.
    """
    bang = sg.doc_bang_tra(_viet_csv(tmp_path, [_clip(f"c{i}", "siren", 1.0 + 0.1 * i) for i in range(30)]))
    rng = random.Random(3)

    dung = {sg.chon_nguon(rng, bang["siren"], 99.0)[0].file_id for _ in range(200)}

    assert len(dung) >= 3


# ── Tham số truyền cho Scaper ────────────────────────────────────────────────


def test_tham_so_su_kien_lay_source_time_bang_lead(tmp_path):
    """source_time = lead của chính file đó — đây là bản sửa ① mà không đụng bank."""
    bang = sg.doc_bang_tra(_viet_csv(tmp_path, [_clip("x", "vehicle_crash", 3.0, lead=1.2)]))
    nguon = bang["vehicle_crash"][0]

    tham_so = sg.tham_so_su_kien(nguon, 2.0, snr=(5.0, 5.0),
                                 events=SU_KIEN, rng=random.Random(0))

    assert tham_so["source_time"] == ("const", pytest.approx(1.2))
    assert tham_so["source_file"][0] == "const"
    assert tham_so["label"] == ("const", "vehicle_crash")
    assert "event_time" not in tham_so, "vị trí phải do nơi gọi gắn SAU khi đã biết thời lượng cuối"


def test_tham_so_su_kien_bu_time_stretch_de_ra_dung_thoi_luong(tmp_path):
    """event_duration × time_stretch phải ra đúng thời lượng đã định.

    Scaper nhân độ dài với time_stretch SAU khi cắt. Đặt event_duration=d rồi để
    Scaper tự rút stretch thì nhãn cuối lệch tới ±10% — đủ để một sự kiện định là
    4.0 s ra 3.6 s và trượt hợp đồng long_event mà không ai thấy.
    """
    bang = sg.doc_bang_tra(_viet_csv(tmp_path, [_clip("x", "siren", 9.0)]))
    nguon = bang["siren"][0]

    for seed in range(30):
        tham_so = sg.tham_so_su_kien(nguon, 4.0, snr=(5.0, 5.0),
                                     events=SU_KIEN, rng=random.Random(seed))
        keo = tham_so["time_stretch"][1]
        assert tham_so["event_duration"][1] * keo == pytest.approx(tham_so["thoi_luong_cuoi"])
        assert tham_so["thoi_luong_cuoi"] == pytest.approx(4.0)


def test_tham_so_su_kien_khong_doc_qua_duoi_file(tmp_path):
    """source_time + event_duration ≤ dur. Vượt là Scaper lấy phải im lặng đuôi."""
    bang = sg.doc_bang_tra(_viet_csv(tmp_path, [_clip("x", "gunshot", 1.0, lead=0.3)]))
    nguon = bang["gunshot"][0]

    for seed in range(30):
        tham_so = sg.tham_so_su_kien(nguon, 1.0, snr=(5.0, 5.0),
                                     events=SU_KIEN, rng=random.Random(seed))
        assert tham_so["source_time"][1] + tham_so["event_duration"][1] <= nguon.dur + 1e-9


def test_tham_so_su_kien_giu_snr_va_pitch_trong_dai_cau_hinh(tmp_path):
    bang = sg.doc_bang_tra(_viet_csv(tmp_path, [_clip("x", "gunshot", 1.0)]))
    nguon = bang["gunshot"][0]

    for seed in range(30):
        tham_so = sg.tham_so_su_kien(nguon, 0.8, snr=(-5.0, 25.0),
                                     events=SU_KIEN, rng=random.Random(seed))
        assert -5.0 <= tham_so["snr"][1] <= 25.0
        assert -1 <= tham_so["pitch_shift"][1] <= 1
        assert 0.9 <= tham_so["time_stretch"][1] <= 1.1


# ── Nền liên tục (B2.5) ──────────────────────────────────────────────────────


def _viet_manifest_nen(tmp_path: Path, dong: list[dict]) -> Path:
    ra = tmp_path / "background_10s_manifest.csv"
    with ra.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sg.COT_MANIFEST_NEN)
        writer.writeheader()
        writer.writerows(dong)
    return ra


def _nen(file_id: str, area: str, dai: float) -> dict:
    return {"file_id": file_id, "area_type": area, "dai_sec": dai}


def test_doc_bang_nen_gom_theo_khu_vuc(tmp_path):
    path = _viet_manifest_nen(tmp_path, [_nen("b", "parking", 32.0), _nen("a", "parking", 10.0),
                                         _nen("c", "factory", 10.0)])

    bang = sg.doc_bang_nen(path)

    assert set(bang) == {"parking", "factory"}
    assert [n.file_id for n in bang["parking"]] == ["a", "b"]


def test_chon_nen_rut_cua_so_ngau_nhien_trong_ban_ghi_dai(tmp_path):
    """Đơn vị 32 s, clip 10 s → source_time phải nằm trong [0, 22].

    Đây là lý do B2.5 giữ nguyên độ dài đơn vị thay vì cắt sẵn: mỗi lần dùng là một
    cửa sổ 10 giây khác nhau trong cùng bản ghi, không tốn thêm byte nào trên đĩa.
    """
    bang = sg.doc_bang_nen(_viet_manifest_nen(tmp_path, [_nen("dai", "parking", 32.0)]))
    rng = random.Random(0)

    moc = {sg.chon_nen(rng, bang["parking"], 10.0)[1] for _ in range(200)}

    assert all(0.0 <= t <= 22.0 + 1e-9 for t in moc)
    assert len(moc) > 50, "không rút ngẫu nhiên, đang trả về cùng một mốc"


def test_chon_nen_dung_10s_thi_source_time_bang_0(tmp_path):
    """Đơn vị vừa đúng 10 s không còn chỗ nào để trượt — phải là 0, không âm."""
    bang = sg.doc_bang_nen(_viet_manifest_nen(tmp_path, [_nen("vua", "school", 10.0)]))
    rng = random.Random(0)

    for _ in range(20):
        _, moc = sg.chon_nen(rng, bang["school"], 10.0)
        assert moc == 0.0


def test_chon_nen_khong_bao_gio_doc_qua_duoi_file(tmp_path):
    """source_time + 10 ≤ dài. Vượt là Scaper lặp vòng — đúng thứ B2.5 vừa chữa."""
    bang = sg.doc_bang_nen(_viet_manifest_nen(tmp_path, [
        _nen(f"n{i}", "factory", d) for i, d in enumerate([10.0, 11.0, 26.0, 66.0])
    ]))
    rng = random.Random(7)

    for _ in range(300):
        nen, moc = sg.chon_nen(rng, bang["factory"], 10.0)
        assert moc + 10.0 <= nen.dai_sec + 1e-9


def test_chon_nen_tat_dinh_theo_seed(tmp_path):
    bang = sg.doc_bang_nen(_viet_manifest_nen(tmp_path, [
        _nen(f"n{i}", "factory", 10.0 + i) for i in range(20)
    ]))

    lay = lambda: [sg.chon_nen(random.Random(5), bang["factory"], 10.0) for _ in range(10)]  # noqa: E731
    assert [(n.file_id, t) for n, t in lay()] == [(n.file_id, t) for n, t in lay()]


def test_chon_nguon_khong_don_het_luot_vao_vai_clip_dai_nhat(tmp_path):
    """Lớp thiếu nguồn dài không được dồn hết lượt vào 1-2 clip.

    Đo trên bank thật trước khi có chặn này: MỘT file siren chiếm 57.5% số lượt của
    cả lớp (1.100 sự kiện). Model sẽ học thuộc đúng file đó thay vì học lớp siren,
    và triệu chứng duy nhất là điểm trên tập dev đẹp hơn thực lực.

    Dựng đúng hình dạng của siren: nhiều clip ngắn, rất ít clip dài, còn phân bố
    đích thì toàn đòi dài.
    """
    bang = sg.doc_bang_tra(_viet_csv(tmp_path, [_clip(f"ngan{i}", "siren", 3.0 + 0.05 * i) for i in range(60)]
                                    + [_clip("dai_a", "siren", 9.4), _clip("dai_b", "siren", 9.5)]))
    rng = random.Random(11)

    dem: dict[str, int] = {}
    for _ in range(1000):
        nguon, _ = sg.chon_nguon(rng, bang["siren"], 9.0)
        dem[nguon.file_id] = dem.get(nguon.file_id, 0) + 1

    nhieu_nhat = max(dem.values()) / 1000
    assert nhieu_nhat < 0.20, f"một file chiếm {nhieu_nhat:.1%} số lượt"
    assert len(dem) >= sg.SO_UNG_VIEN_TOI_THIEU


def test_chon_nguon_van_giu_duoc_duoi_dai_khi_noi_nhom(tmp_path):
    """Nới nhóm để đa dạng KHÔNG được làm mất hẳn các sự kiện dài.

    Đây là nửa còn lại của đánh đổi: siết quá thì hết dùng lại file, nhưng p95 của
    siren tụt từ 9.48 s xuống 5.13 s và ta mất đúng cái đuôi phân bố cần giữ.
    """
    bang = sg.doc_bang_tra(_viet_csv(tmp_path, [_clip(f"ngan{i}", "siren", 3.0 + 0.05 * i) for i in range(60)]
                                    + [_clip("dai_a", "siren", 9.4), _clip("dai_b", "siren", 9.5)]))
    rng = random.Random(12)

    ra = [sg.chon_nguon(rng, bang["siren"], 9.0)[1] for _ in range(1000)]

    assert max(ra) >= 9.0 - 1e-9, "mất hẳn sự kiện dài"
    assert sum(1 for x in ra if x >= 9.0) >= 100, "sự kiện dài quá hiếm"


def test_tham_so_su_kien_CHUA_co_event_time_de_dat_vi_tri_dung_thoi_luong_cuoi(tmp_path):
    """Vị trí phải được đặt SAU khi biết thời lượng cuối, không phải trước.

    Thời lượng cuối là `event_duration × time_stretch`, mà hệ số kéo rút bên trong
    hàm này. Đặt vị trí theo thời lượng TRƯỚC khi kéo thì sự kiện co giãn ±10% sau
    khi đã tính chỗ — và cặp bị ép của lát cắt `overlap` tụt xuống dưới ngưỡng ở một
    phần clip. Đúng kiểu hỏng hóc không triệu chứng mà cả STATUS §7 sinh ra để chặn;
    test `test_hop_dong_overlap_dat_100_phan_tram` đã bắt được nó một lần.
    """
    bang = sg.doc_bang_tra(_viet_csv(tmp_path, [_clip("x", "siren", 6.0)]))

    tham_so = sg.tham_so_su_kien(bang["siren"][0], 4.0, snr=(5.0, 5.0),
                                 events=SU_KIEN, rng=random.Random(0))

    assert "event_time" not in tham_so
    assert tham_so["thoi_luong_cuoi"] == pytest.approx(4.0)


def test_tham_so_su_kien_NO_khi_su_kien_ngan_hon_gioi_han_scaper(tmp_path):
    """Sự kiện ngắn hơn tổng fade của Scaper phải nổ NGAY, có thông điệp đọc được.

    Đặt chốt chặn ở đây là có chủ ý: `quyet_dinh_clip` gọi hàm này, nên bộ mô phỏng
    B7 chạm tới nó và bắt được ràng buộc của Scaper trong 3 giây — thay vì để lượt
    sinh chết ở clip 7.815 sau 45 phút, đúng như đã xảy ra thật.
    """
    bang = sg.doc_bang_tra(_viet_csv(tmp_path, [_clip("ti_hon", "gunshot", 0.005)]))

    with pytest.raises(ValueError, match="Scaper"):
        sg.tham_so_su_kien(bang["gunshot"][0], 0.005, snr=(5.0, 5.0),
                           events=SU_KIEN, rng=random.Random(0))


def test_tham_so_su_kien_khong_no_voi_su_kien_ngan_nhat_hop_le(tmp_path):
    """Sàn 0.05 s chia cho hệ số kéo lớn nhất vẫn phải trên giới hạn Scaper."""
    bang = sg.doc_bang_tra(_viet_csv(tmp_path, [_clip("ngan", "gunshot", 0.1016)]))

    for seed in range(50):
        tham_so = sg.tham_so_su_kien(bang["gunshot"][0], sg.TOI_THIEU_SU_KIEN_SEC,
                                     snr=(5.0, 5.0), events=SU_KIEN, rng=random.Random(seed))
        assert tham_so["event_duration"][1] >= sg.TONG_FADE_SCAPER_SEC
