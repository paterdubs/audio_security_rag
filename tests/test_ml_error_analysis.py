"""Test cho phân tích lỗi — Pha 4 của docs/TRAINING_OPS_PLAN.md.

Phép ghép cặp ref↔pred ở đây là code TỰ VIẾT nằm cạnh một thư viện tham chiếu, vì
`sed_eval` chỉ trả tỉ lệ S/D/I chứ không phơi ra cặp nào khớp cặp nào. Đó đúng là chỗ dễ
ra số đẹp mà không ai kiểm được: một sự kiện bị đếm hai lần chỉ hiện ra dưới dạng "tỉ lệ
Deletion hơi thấp", không có triệu chứng nào khác. Nên mọi bất biến đếm được đều có test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import ml.evaluation.error_analysis as ea  # noqa: E402
from ml.evaluation.error_taxonomy import (  # noqa: E402
    LOAI_TANG_MOT,
    ghep_mot_clip,
    ma_tran_nham,
    phan_loai,
)
from ml.evaluation.sed_metrics import (  # noqa: E402
    ONSET_COLLAR_SEC,
    frames_to_events,
    thong_ke_su_kien,
)


def sk(lop: str, on: float, off: float) -> dict:
    return {"event_label": lop, "onset": on, "offset": off}


def khoa(events) -> list:
    return sorted((e["event_label"], round(e["onset"], 6)) for e in events)


# ── Bất biến của phép ghép tầng 1 ────────────────────────────────────────────


def test_tong_tang_mot_bang_so_su_kien_tham_chieu():
    """Mỗi sự kiện tham chiếu phải nhận ĐÚNG MỘT nhãn trong {đúng, biên, thay_thế, thiếu}.

    Không có bất biến này thì một ref vừa khớp ở vòng 'đúng' vừa được vét lại ở vòng
    'thay thế' sẽ làm tổng lớn hơn 4.873 sự kiện của dev mà bảng vẫn in ra bình thường.
    """
    ref = [sk("gunshot", 1.0, 1.5), sk("scream", 3.0, 4.0), sk("siren", 6.0, 8.0)]
    pred = [sk("gunshot", 1.05, 1.6), sk("explosion", 3.1, 3.9)]
    cap, _, _ = ghep_mot_clip("c1", ref, pred, ONSET_COLLAR_SEC)

    tang_mot = [c for c in cap if c.loai in LOAI_TANG_MOT]
    assert len(tang_mot) == len(ref)
    assert sorted(c.loai for c in tang_mot) == ["dung", "thay_the", "thieu"]


def test_moi_du_bao_dung_nhieu_nhat_mot_lan():
    """Một dự báo bị hai ref cùng tiêu thụ sẽ làm Insertion tụt xuống và Deletion tụt
    theo — hai lỗi ngược chiều nhau, triệt tiêu trong tổng, không ai nhìn ra."""
    ref = [sk("gunshot", 1.0, 2.0), sk("gunshot", 1.1, 2.1)]
    pred = [sk("gunshot", 1.02, 2.0)]
    cap, _, _ = ghep_mot_clip("c1", ref, pred, ONSET_COLLAR_SEC)

    da_dung = [c for c in cap if c.pred_onset is not None]
    assert len(da_dung) == 1
    assert sum(1 for c in cap if c.loai == "thieu") == 1


def test_bien_khong_bi_nuot_thanh_thieu_cong_thua():
    """Đúng lớp, chồng lấn, nhưng onset lệch 0,5 s > collar 0,2 s → Boundary.

    sed_eval tính ca này thành MỘT Deletion + MỘT Insertion. Đó không sai theo định nghĩa
    của nó, nhưng gộp chung với lỗi 'không nghe ra gì cả' thì mất hẳn thông tin cần để
    quyết định: lệch biên sửa bằng hậu xử lý, không nghe ra thì phải sửa dữ liệu.
    """
    cap, _, _ = ghep_mot_clip("c1", [sk("siren", 2.0, 5.0)], [sk("siren", 2.5, 5.2)],
                              ONSET_COLLAR_SEC)
    assert [c.loai for c in cap] == ["bien"]


def test_dung_chon_du_bao_gan_onset_nhat():
    """Hai dự báo cùng lớp đều trong collar → phải lấy cái gần onset nhất, không phải cái
    đứng trước trong danh sách. Lấy nhầm thì cái còn lại thành Insertion và ma trận nhầm
    lẫn ghi sai một ô."""
    ref = [sk("gunshot", 1.00, 1.50)]
    pred = [sk("gunshot", 1.18, 1.60), sk("gunshot", 1.01, 1.49)]
    cap, _, _ = ghep_mot_clip("c1", ref, pred, ONSET_COLLAR_SEC)

    dung = [c for c in cap if c.loai == "dung"]
    assert len(dung) == 1
    assert dung[0].pred_onset == pytest.approx(1.01)


def test_thay_the_khi_khac_lop_cung_thoi_diem():
    cap, _, _ = ghep_mot_clip("c1", [sk("scream", 3.0, 4.0)], [sk("shout_yell", 3.05, 4.1)],
                              ONSET_COLLAR_SEC)
    assert [c.loai for c in cap] == ["thay_the"]
    assert cap[0].ref_lop == "scream"
    assert cap[0].pred_lop == "shout_yell"


def test_thua_khi_khong_chong_lan_ref_nao():
    cap, _, _ = ghep_mot_clip("c1", [sk("gunshot", 1.0, 2.0)], [sk("gunshot", 7.0, 8.0)],
                              ONSET_COLLAR_SEC)
    assert sorted(c.loai for c in cap) == ["thieu", "thua"]


def test_clip_khong_su_kien_tham_chieu_van_dem_du_bao():
    """Clip im lặng bị bỏ qua là lỗi ĐÃ XẢY RA ở khâu chấm điểm: thiếu clip ở một phía
    làm sed_eval hiểu là 'không đánh giá' chứ không phải 'đoán rỗng', và mọi dương tính
    giả trong clip đó biến mất khỏi mẫu số. Phép ghép không được lặp lại lỗi đó."""
    cap, _, _ = ghep_mot_clip("im_lang", [], [sk("dog_bark", 1.0, 2.0)], ONSET_COLLAR_SEC)
    assert [c.loai for c in cap] == ["thua"]

    cap_rong, _, _ = ghep_mot_clip("rong", [], [], ONSET_COLLAR_SEC)
    assert cap_rong == []


# ── Tầng 2: cấu trúc ─────────────────────────────────────────────────────────


def test_phan_manh_dem_rieng_khong_tru_vao_tang_mot():
    """1 sự kiện thật → 3 mảnh dự báo. Tầng 1 vẫn phải đủ 1 nhãn cho ref đó; phân mảnh là
    con số ĐẾM RIÊNG. Trừ chéo hai tầng vào nhau thì tổng không còn kiểm được."""
    ref = [sk("siren", 1.0, 5.0)]
    pred = [sk("siren", 1.0, 2.0), sk("siren", 2.5, 3.5), sk("siren", 4.0, 5.0)]
    cap, n_manh, n_gop = ghep_mot_clip("c1", ref, pred, ONSET_COLLAR_SEC)

    assert n_manh == 1
    assert n_gop == 0
    assert sum(1 for c in cap if c.loai in LOAI_TANG_MOT) == 1
    assert sum(1 for c in cap if c.loai == "thua") == 2


def test_gop_dem_rieng():
    """3 phát súng thật → 1 đoạn dự báo dài. Đây là ca mà docstring của sed_metrics đã
    nêu: gộp hai phát thành một là đủ để mất điểm cả hai."""
    ref = [sk("gunshot", 1.0, 1.2), sk("gunshot", 2.0, 2.2), sk("gunshot", 3.0, 3.2)]
    pred = [sk("gunshot", 0.9, 3.5)]
    cap, n_manh, n_gop = ghep_mot_clip("c1", ref, pred, ONSET_COLLAR_SEC)

    assert n_gop == 1
    assert n_manh == 0
    assert sum(1 for c in cap if c.loai in LOAI_TANG_MOT) == 3


def test_phan_loai_cong_don_qua_nhieu_clip():
    tham_chieu = {"a": [sk("gunshot", 1.0, 2.0)], "b": [], "c": [sk("scream", 1.0, 2.0)]}
    du_bao = {"a": [sk("gunshot", 1.05, 2.0)], "b": [sk("dog_bark", 5.0, 6.0)], "c": []}
    kq = phan_loai(tham_chieu, du_bao, ONSET_COLLAR_SEC)

    assert kq.dem["dung"] == 1
    assert kq.dem["thua"] == 1
    assert kq.dem["thieu"] == 1
    assert kq.n_ref == 2
    assert sum(kq.dem[loai] for loai in LOAI_TANG_MOT) == kq.n_ref


# ── Đối chiếu với sed_eval ───────────────────────────────────────────────────


def test_thong_ke_su_kien_tra_so_dem_khong_phai_ti_le():
    """sed_eval chỉ công bố S/D/I dưới dạng tỉ lệ chuẩn hoá theo Nref. Quên nhân ngược
    lại thì bảng in ra 0,5 sự kiện Deletion — vô nghĩa nhưng vẫn in được."""
    tham_chieu = {"f1": [sk("gunshot", 1.0, 2.0), sk("scream", 5.0, 6.0)]}
    du_bao = {"f1": [sk("scream", 1.05, 2.1), sk("gunshot", 8.0, 8.5)]}
    tk = thong_ke_su_kien(tham_chieu, du_bao, ["gunshot", "scream"])

    assert tk["n_ref"] == 2
    assert tk["n_sys"] == 2
    assert (tk["substitution"], tk["deletion"], tk["insertion"]) == (1, 1, 1)


def test_dung_cua_ghep_greedy_khop_ntp_cua_sed_eval_o_ca_de():
    """Ca không nhập nhằng thì hai phép ghép PHẢI ra cùng số khớp đúng. Lệch ở đây nghĩa
    là phép ghép tự viết sai từ gốc, không phải 'greedy vs tối ưu'."""
    tham_chieu = {"f1": [sk("gunshot", 1.0, 2.0)], "f2": [sk("scream", 3.0, 4.0)]}
    du_bao = {"f1": [sk("gunshot", 1.05, 2.0)], "f2": [sk("scream", 7.0, 8.0)]}
    kq = phan_loai(tham_chieu, du_bao, ONSET_COLLAR_SEC)
    tk = thong_ke_su_kien(tham_chieu, du_bao, ["gunshot", "scream"])

    assert kq.dem["dung"] == sum(v["n_tp"] for v in tk["theo_lop"].values()) == 1


# ── Ma trận nhầm lẫn ─────────────────────────────────────────────────────────


def test_ma_tran_nham_tong_bang_so_cap():
    """Tổng mọi ô phải bằng số cặp đã sinh. Lệch đi nghĩa là có cặp rơi ra ngoài ma trận
    — và một ô thiếu trong ma trận 16×16 thì không ai phát hiện bằng mắt."""
    lop = ["gunshot", "scream", "shout_yell"]
    tham_chieu = {"a": [sk("scream", 1.0, 2.0)], "b": [sk("gunshot", 1.0, 2.0)]}
    du_bao = {"a": [sk("shout_yell", 1.1, 2.0)], "b": [sk("gunshot", 8.0, 9.0)]}
    kq = phan_loai(tham_chieu, du_bao, ONSET_COLLAR_SEC)
    mt = ma_tran_nham(kq, lop)

    assert mt.shape == (len(lop) + 1, len(lop) + 1)
    assert mt.sum() == len(kq.cap)
    assert mt[lop.index("scream"), lop.index("shout_yell")] == 1
    assert mt[lop.index("gunshot"), len(lop)] == 1      # thiếu → cột ∅
    assert mt[len(lop), lop.index("gunshot")] == 1      # thừa → hàng ∅


def test_ma_tran_nham_duong_cheo_gom_ca_bien():
    """Biên là 'đúng lớp, sai thời điểm' nên nó nằm trên đường chéo. Báo cáo phải nói
    điều đó, nếu không người đọc tưởng đường chéo = số đúng."""
    lop = ["siren"]
    kq = phan_loai({"a": [sk("siren", 2.0, 5.0)]}, {"a": [sk("siren", 2.5, 5.2)]},
                   ONSET_COLLAR_SEC)
    mt = ma_tran_nham(kq, lop)
    assert mt[0, 0] == 1
    assert kq.dem["dung"] == 0
    assert kq.dem["bien"] == 1


# ── Ngưỡng theo từng lớp ─────────────────────────────────────────────────────


def test_nguong_vector_cho_ket_qua_giong_quet_tung_lop_rieng():
    """θ dạng mảng (C,) phải broadcast đúng trong `frames_to_events`. Nếu numpy so sánh
    nhầm trục thì hàm VẪN chạy và VẪN trả sự kiện — sai một cách im lặng, đúng kiểu lỗi
    đã bắt được nhiều lần ở repo này."""
    rng = np.random.default_rng(0)
    prob = rng.random((100, 3)).astype(np.float32)
    lop = ["a", "b", "c"]
    nguong = np.array([0.3, 0.6, 0.9])

    gop = frames_to_events(prob, lop, 10.0, nguong, median_size=1)
    rieng = []
    for i, lop_i in enumerate(lop):
        rieng += [e for e in frames_to_events(prob, lop, 10.0, float(nguong[i]), median_size=1)
                  if e["event_label"] == lop_i]

    assert khoa(gop) == khoa(rieng)
    assert len(gop) > 0


def test_nguong_toi_uu_theo_lop_lay_argmax_cua_tung_lop():
    """θ toàn cục 0,9x không tối ưu cho lớp hiếm — đó là lý do tồn tại của Pha 4 phần 4.
    Mỗi lớp phải lấy argmax của CHÍNH nó, không phải θ của điểm tổng."""
    quet = [
        {"threshold": 0.30, "event_f1_theo_lop": {"gunshot": 0.10, "siren": 0.40}},
        {"threshold": 0.90, "event_f1_theo_lop": {"gunshot": 0.35, "siren": 0.05}},
    ]
    tot = ea.nguong_toi_uu_theo_lop(quet, ["gunshot", "siren", "vang_mat"])

    assert tot["gunshot"]["threshold"] == 0.90
    assert tot["siren"]["threshold"] == 0.30
    assert tot["vang_mat"]["event_f1"] == 0.0


# ── Lát cắt ──────────────────────────────────────────────────────────────────


def test_doc_lat_cat_gan_sach_cho_clip_khong_lat_cat(tmp_path):
    """357/1.440 clip dev không thuộc lát cắt nào. Bỏ chúng ra khỏi bảng thì cột 'sạch'
    biến mất và không còn mốc đối chiếu cho năm lát cắt khó."""
    path = tmp_path / "slice_index.jsonl"
    path.write_text(
        '{"clip_id": "a", "slices": ["overlap", "reverb"]}\n'
        '{"clip_id": "b", "slices": []}\n',
        encoding="utf-8",
    )
    lat_cat = ea.doc_lat_cat(path)
    assert lat_cat["a"] == ["overlap", "reverb"]
    assert lat_cat["b"] == [ea.LAT_CAT_SACH]


def test_lat_cat_chong_nhau_nen_tong_clip_lon_hon_so_clip(tmp_path):
    """Một clip vừa overlap vừa reverb được đếm ở CẢ HAI hàng. Trên dev thật tổng là
    2.088 trên 1.440 clip. Báo cáo không nói câu đó thì người đọc cộng cột lại và kết
    luận file hỏng."""
    path = tmp_path / "slice_index.jsonl"
    path.write_text(
        '{"clip_id": "a", "slices": ["overlap", "reverb"]}\n'
        '{"clip_id": "b", "slices": ["reverb"]}\n',
        encoding="utf-8",
    )
    nhom = ea.nhom_theo_lat_cat(ea.doc_lat_cat(path), ["a", "b"])
    assert sum(len(v) for v in nhom.values()) > 2
    assert nhom["reverb"] == ["a", "b"]
    assert nhom["overlap"] == ["a"]
    assert nhom[ea.LAT_CAT_SACH] == []


def test_nhom_theo_lat_cat_bo_qua_clip_khong_co_trong_du_doan(tmp_path):
    """slice_index.jsonl là của TOÀN split; file dự đoán có thể chỉ chứa một subset.
    Lấy nhầm clip ngoài subset vào bảng thì mẫu số sai mà không có lỗi nào bắn ra."""
    path = tmp_path / "slice_index.jsonl"
    path.write_text(
        '{"clip_id": "a", "slices": ["overlap"]}\n'
        '{"clip_id": "ngoai_subset", "slices": ["overlap"]}\n',
        encoding="utf-8",
    )
    nhom = ea.nhom_theo_lat_cat(ea.doc_lat_cat(path), ["a"])
    assert nhom["overlap"] == ["a"]


# ── Nguồn chân lý và ghi bền ─────────────────────────────────────────────────


def test_doc_confusable_lay_tu_yaml_khong_hardcode():
    """ontology_map.yaml là nguồn chân lý duy nhất. Chép danh sách lớp dễ nhầm vào code
    nghĩa là sửa YAML xong báo cáo vẫn in bảng cũ."""
    bang = ea.doc_confusable(ea.ONTOLOGY_PATH)
    assert "gunshot" in bang
    assert "fireworks" in bang["gunshot"]
    assert ea.BACKGROUND_CLASS not in bang


def test_viet_dan_ghi_tung_dong_khong_don_toi_cuoi(tmp_path):
    """Dồn output tới cuối hàm rồi mới ghi đã làm MẤT 372 dòng segments.jsonl khi tiến
    trình chết giữa chừng. Ghi từng dòng + flush thì phần đã sinh vẫn còn trên đĩa."""
    path = tmp_path / "bao_cao.md"

    def dong_loi():
        yield "dòng 1"
        yield "dòng 2"
        raise RuntimeError("chết giữa chừng")

    with pytest.raises(RuntimeError):
        ea.viet_dan(path, dong_loi())

    assert path.read_text(encoding="utf-8").splitlines() == ["dòng 1", "dòng 2"]


def test_bang_lat_cat_tach_rieng_sau_loai_loi_cho_tung_lat_cat():
    """Pha 4 chỉ xuất F1 theo lát cắt, nên 927 sự kiện phân mảnh của v3 là một con số
    GỘP: không trả lời được bao nhiêu trong đó rơi vào `long_event` — đúng câu hỏi cần
    trả lời. Mỗi hàng phải mang phân loại lỗi của riêng tập con clip đó."""
    class_ids = ["gunshot", "siren"]
    tham_chieu = {
        "dai": [{"event_label": "siren", "onset": 0.0, "offset": 6.0}],
        "ngan": [{"event_label": "gunshot", "onset": 1.0, "offset": 1.2}],
    }
    du_bao = {
        # Một sự kiện dài bị cắt làm ba mảnh — kiểu lỗi nghi cho long_event.
        "dai": [{"event_label": "siren", "onset": 0.0, "offset": 1.5},
                {"event_label": "siren", "onset": 2.5, "offset": 3.5},
                {"event_label": "siren", "onset": 4.5, "offset": 6.0}],
        "ngan": [{"event_label": "gunshot", "onset": 1.0, "offset": 1.2}],
    }
    nhom = {"long_event": ["dai"], ea.LAT_CAT_SACH: ["ngan"]}
    hang = {h["lat_cat"]: h for h in
            ea.bang_lat_cat(tham_chieu, du_bao, class_ids, 10.0, nhom)}

    assert hang["long_event"]["phan_manh"] == 1
    assert hang[ea.LAT_CAT_SACH]["phan_manh"] == 0
    assert hang["long_event"]["dem"]["thua"] == 2
    assert hang[ea.LAT_CAT_SACH]["dem"] == {"dung": 1, "bien": 0, "thay_the": 0,
                                            "thieu": 0, "thua": 0}
    for h in hang.values():
        tang_mot = sum(h["dem"][k] for k in ("dung", "bien", "thay_the", "thieu"))
        assert tang_mot == h["n_ref"]


def test_bang_lat_cat_dem_lai_tren_tap_con_chu_khong_chia_ti_le_tu_tong():
    """Lát cắt CHỒNG NHAU: một clip vừa `overlap` vừa `reverb` đóng góp lỗi cho cả hai
    hàng. Suy ra số lỗi của lát cắt bằng cách nhân tỉ lệ với tổng toàn tập là sai lặng
    lẽ — tổng theo hàng phải LỚN HƠN tổng toàn tập, không bằng."""
    class_ids = ["gunshot"]
    tham_chieu = {"a": [{"event_label": "gunshot", "onset": 0.0, "offset": 0.5}]}
    du_bao = {"a": []}
    nhom = {"overlap": ["a"], "reverb": ["a"]}
    hang = ea.bang_lat_cat(tham_chieu, du_bao, class_ids, 10.0, nhom)

    assert sum(h["dem"]["thieu"] for h in hang) == 2  # một Deletion thật, đếm ở hai hàng


def test_bang_lat_cat_lat_cat_rong_khong_lam_no_phan_loai():
    """`sach` có thể rỗng khi chấm trên subset nhỏ. phan_loai() ném AssertionError nếu
    tổng tầng 1 lệch số sự kiện, nên hàng rỗng phải được xử lý riêng chứ không gọi vào."""
    hang = ea.bang_lat_cat({}, {}, ["gunshot"], 10.0, {"long_event": []})
    assert hang[0]["n_clip"] == 0
    assert hang[0]["dem"] == {"dung": 0, "bien": 0, "thay_the": 0, "thieu": 0, "thua": 0}
    assert hang[0]["phan_manh"] == 0


def test_ablation_khong_ghi_de_len_bao_cao_mac_dinh():
    """Chạy `--adaptive-postproc` để làm ablation mà vẫn ghi vào `analysis.json` thì lượt
    sau xoá mất lượt trước, và bảng so sánh hai cấu hình không còn gì để so. Mất 3 phút
    tính lại mỗi lần, và không có lỗi nào bắn ra."""
    assert ea.ten_bao_cao(False) != ea.ten_bao_cao(True)
    assert ea.ten_bao_cao(False) == "analysis"


def test_ten_bao_cao_tran_51_giu_ten_cu():
    """max_frames=51 là mặc định của MAX_MEDIAN_FRAMES — phải ra đúng tên cũ
    'analysis_adaptive', không thì mọi lệnh --adaptive-postproc chạy trước khi tham số
    này tồn tại sẽ đọc nhầm sang file khác."""
    assert ea.ten_bao_cao(True, 51) == "analysis_adaptive"


def test_ten_bao_cao_moi_tran_ra_ten_rieng():
    """Ablation quét MAX_MEDIAN_FRAMES ∈ {101, 201, 401} — ghi đè lẫn nhau thì bảng so
    sánh theo trần không còn dữ liệu để so, giống lỗi đã né ở test phía trên."""
    ten = {ea.ten_bao_cao(True, m) for m in (101, 201, 401)}
    assert len(ten) == 3
    assert ea.ten_bao_cao(False) not in ten
    assert "analysis_adaptive" not in ten


def test_ten_bao_cao_nguon_train_khac_ten_nguon_dev():
    """`--adaptive-source train` (không rò rỉ) và mặc định `dev` (rò rỉ) phải ra hai file
    khác nhau — gộp chung thì bảng 3 cột 7-khung/theo-lớp-từ-dev/theo-lớp-từ-train mất
    một cột, và không ai biết con số nào rò rỉ, con số nào không."""
    assert ea.ten_bao_cao(True, nguon_cua_so="train") == "analysis_adaptive_train"
    assert ea.ten_bao_cao(True, nguon_cua_so="dev") == "analysis_adaptive"
    assert ea.ten_bao_cao(True) == ea.ten_bao_cao(True, nguon_cua_so="dev")
