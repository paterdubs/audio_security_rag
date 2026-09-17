"""Test cho khâu QUYẾT ĐỊNH NỘI DUNG CLIP — STATUS §7 B7.

Bài học đắt nhất của dự án: hợp đồng `long_event ≥ 8s` khai đạt 791 clip trong khi
số sự kiện thật sự sinh ra là 0, và điều đó tồn tại suốt hai vòng train. Nguyên nhân
không phải một phép tính sai mà là KHOẢNG CÁCH giữa thứ được lập kế hoạch và thứ
được sinh ra — log báo cái đầu, không ai đọc cái sau.

B7 đóng khoảng cách đó bằng cấu trúc chứ không bằng kỷ luật: `quyet_dinh_clip` là
nơi DUY NHẤT quyết định clip chứa gì. Bộ mô phỏng đọc thẳng kết quả của nó, còn khâu
sinh audio chỉ dịch kết quả đó sang lời gọi Scaper. Hai đường không thể lệch nhau,
vì chúng là một.

Nhờ vậy nghiệm thu được phân bố trong vài giây thay vì sau 18 giờ sinh audio.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import scaper_generate as sg  # noqa: E402

CONFIG = {
    "duration_sec": 10.0,
    "ref_db": -23,
    "sample_rate": 16000,
    "seed": 1,
    "events": {
        "count_weights": {0: 0.10, 1: 0.20, 2: 0.30, 3: 0.25, 4: 0.15},
        "snr_db": [-5, 25],
        "pitch_shift_semitones": [-1, 1],
        "time_stretch": [0.9, 1.1],
        "rir_probability": 0.40,
    },
    "forced_slices": {
        "overlap": {"ratio": 0.30, "min_events": 2, "min_overlap_ratio": 0.30},
        "low_snr": {"ratio": 0.25, "max_snr_db": 5},
        "reverb": {"ratio": 0.40},
        "long_event": {"ratio": 0.10, "min_event_duration_sec": 4.0},
        "causal_chain": {"ratio": 0.15},
    },
    "causal_chains": [
        {"name": "break_in", "sequence": ["glass_breaking", "scream", "running_footsteps"],
         "gaps_sec": [[0.5, 2.0], [0.3, 1.5]]},
    ],
}

LOP = ["glass_breaking", "scream", "running_footsteps", "siren", "gunshot"]


def _nguyen_lieu() -> sg.NguyenLieu:
    """Nguyên liệu giả, đủ hình dạng thật: lớp dài, lớp ngắn, nhiều clip mỗi lớp."""
    bang = {}
    for lop in LOP:
        dai_co_the = [6.0, 5.0, 4.5] if lop in {"siren", "running_footsteps"} else []
        effs = dai_co_the + [0.4 + 0.1 * i for i in range(20)]
        bang[lop] = tuple(sorted(
            (sg.ClipNguon(f"{lop}_{i}", lop, f"data/banks/foreground/{lop}/{lop}_{i}.wav",
                          dur=e + 0.2, eff=e, lead=0.2) for i, e in enumerate(effs)),
            key=lambda c: c.file_id))
    phan_bo = {lop: tuple([0.3, 0.8, 1.2, 2.0] + ([5.0, 6.0] if lop in {"siren", "running_footsteps"} else []))
               for lop in LOP}
    bang_nen = {khu: tuple(sg.NenNguon(f"bg_{khu}_{i}", khu, 10.0 + 4 * i) for i in range(5))
                for khu in ("school", "parking", "residential", "factory")}
    return sg.NguyenLieu(
        bang=bang, bang_nen=bang_nen, phan_bo=phan_bo,
        lop_dai=sg.lop_cap_duoc_su_kien_dai(bang, phan_bo, 4.0, so_clip_toi_thieu=3),
        chains=CONFIG["causal_chains"],
    )


def _ke_hoach(n=400, seed=1):
    return sg.plan_clips(CONFIG, n_clips=n, seed=seed, areas=["school", "parking"])


# ── Bất biến trên mọi clip ───────────────────────────────────────────────────


def test_moi_su_kien_nam_tron_trong_clip():
    """Không sự kiện nào tràn qua mốc 10 s. Lô cũ có 1.225 sự kiện như vậy."""
    nl = _nguyen_lieu()
    rng = random.Random(0)

    for plan in _ke_hoach():
        qd = sg.quyet_dinh_clip(rng, plan, CONFIG, nl)
        for sk in qd.su_kien:
            assert sk.start >= -1e-9
            assert sk.start + sk.thoi_luong <= 10.0 + 1e-6, f"{sk.label} tràn khỏi clip"


def test_so_su_kien_dung_bang_ke_hoach():
    nl = _nguyen_lieu()
    rng = random.Random(0)

    for plan in _ke_hoach():
        assert len(sg.quyet_dinh_clip(rng, plan, CONFIG, nl).su_kien) == plan.n_events


def test_nen_khong_bao_gio_doc_qua_duoi_file():
    """source_time + 10 ≤ dài nền. Vượt là Scaper lặp vòng — đúng thứ B2.5 đã chữa."""
    nl = _nguyen_lieu()
    rng = random.Random(0)

    for plan in _ke_hoach():
        qd = sg.quyet_dinh_clip(rng, plan, CONFIG, nl)
        assert qd.nen_source_time + 10.0 <= qd.nen.dai_sec + 1e-9


def test_su_kien_khong_doc_qua_duoi_file_nguon():
    """source_time + event_duration ≤ dur của chính file nguồn."""
    nl = _nguyen_lieu()
    rng = random.Random(0)

    for plan in _ke_hoach():
        for sk in sg.quyet_dinh_clip(rng, plan, CONFIG, nl).su_kien:
            ts = sk.tham_so_scaper
            assert ts["source_time"][1] + ts["event_duration"][1] <= sk.nguon.dur + 1e-9


def test_thoi_luong_cuoi_khop_event_duration_nhan_time_stretch():
    """Nhãn sẽ ghi `event_duration × time_stretch`, nên `thoi_luong` phải bằng đúng thế."""
    nl = _nguyen_lieu()
    rng = random.Random(0)

    for plan in _ke_hoach():
        for sk in sg.quyet_dinh_clip(rng, plan, CONFIG, nl).su_kien:
            ts = sk.tham_so_scaper
            assert sk.thoi_luong == pytest.approx(ts["event_duration"][1] * ts["time_stretch"][1])


# ── Hợp đồng lát cắt ─────────────────────────────────────────────────────────


def test_hop_dong_overlap_dat_100_phan_tram():
    """Lô cũ: nhãn có, nội dung không. Ở đây phải 100%, không ngoại lệ nào."""
    nl = _nguyen_lieu()
    rng = random.Random(0)
    ty_le = CONFIG["forced_slices"]["overlap"]["min_overlap_ratio"]
    so_clip = 0

    for plan in _ke_hoach(800):
        if "overlap" not in plan.slices:
            continue
        so_clip += 1
        sk = sg.quyet_dinh_clip(rng, plan, CONFIG, nl).su_kien
        assert any(
            sg.do_chong_lan(sk[i].start, sk[i].thoi_luong, sk[j].start, sk[j].thoi_luong)
            >= ty_le * min(sk[i].thoi_luong, sk[j].thoi_luong) - 1e-9
            for i in range(len(sk)) for j in range(i + 1, len(sk))
        ), f"clip {plan.index} mang nhãn overlap nhưng không có cặp nào đạt"
    assert so_clip > 50


def test_hop_dong_long_event_dat_100_phan_tram():
    """Hợp đồng từng hỏng 100%: 791 clip khai đạt, 0 sự kiện ≥ 8 s thật sự tồn tại."""
    nl = _nguyen_lieu()
    rng = random.Random(0)
    nguong = CONFIG["forced_slices"]["long_event"]["min_event_duration_sec"]
    so_clip = 0

    for plan in _ke_hoach(800):
        if "long_event" not in plan.slices:
            continue
        so_clip += 1
        sk = sg.quyet_dinh_clip(rng, plan, CONFIG, nl).su_kien
        assert any(x.thoi_luong >= nguong - 1e-9 for x in sk), \
            f"clip {plan.index} mang nhãn long_event nhưng sự kiện dài nhất chỉ {max(x.thoi_luong for x in sk):.2f}s"
    assert so_clip > 20


def test_hop_dong_low_snr_khong_vuot_tran():
    nl = _nguyen_lieu()
    rng = random.Random(0)
    tran = CONFIG["forced_slices"]["low_snr"]["max_snr_db"]

    for plan in _ke_hoach(600):
        if "low_snr" not in plan.slices:
            continue
        for sk in sg.quyet_dinh_clip(rng, plan, CONFIG, nl).su_kien:
            assert sk.tham_so_scaper["snr"][1] <= tran + 1e-9


def test_chuoi_nhan_qua_giu_dung_thu_tu():
    """`glass_breaking → scream → running_footsteps` là đột nhập; đảo lại thì không."""
    nl = _nguyen_lieu()
    rng = random.Random(0)
    so_clip = 0

    for plan in _ke_hoach(800):
        if not plan.chain:
            continue
        so_clip += 1
        sk = sg.quyet_dinh_clip(rng, plan, CONFIG, nl).su_kien
        mat_xich = sg.chain_by_name(CONFIG["causal_chains"], plan.chain)["sequence"]
        assert [x.label for x in sk[:len(mat_xich)]] == mat_xich
        moc = [x.start for x in sk[:len(mat_xich)]]
        assert moc == sorted(moc), "mắt xích bị đảo thứ tự thời gian"
    assert so_clip > 30


def test_lop_cua_su_kien_dai_nam_trong_danh_sach_kha_thi():
    """Sự kiện bị ép dài phải thuộc lớp cấp được — không ép lên lớp không có nguyên liệu."""
    nl = _nguyen_lieu()
    rng = random.Random(0)

    for plan in _ke_hoach(600):
        if "long_event" not in plan.slices:
            continue
        sk = sg.quyet_dinh_clip(rng, plan, CONFIG, nl).su_kien
        nguong = CONFIG["forced_slices"]["long_event"]["min_event_duration_sec"]
        assert any(x.label in nl.lop_dai and x.thoi_luong >= nguong - 1e-9 for x in sk)


# ── Tính tất định ────────────────────────────────────────────────────────────


def test_tat_dinh_hoan_toan_theo_seed():
    """Cùng seed → cùng từng con số. Thiếu điều này thì dataset không tái lập được."""
    nl = _nguyen_lieu()

    def chay():
        rng = random.Random(7)
        return [(sk.label, sk.nguon.file_id, round(sk.start, 6), round(sk.thoi_luong, 6))
                for plan in _ke_hoach(200) for sk in sg.quyet_dinh_clip(rng, plan, CONFIG, nl).su_kien]

    assert chay() == chay()


def test_clip_khong_su_kien_van_co_nen():
    """Nhánh 10% clip im lặng vẫn phải có nền — im lặng nghĩa là không có SỰ KIỆN."""
    nl = _nguyen_lieu()
    rng = random.Random(0)
    im_lang = [p for p in _ke_hoach(400) if p.n_events == 0]

    assert im_lang, "seed này không sinh ra clip im lặng nào"
    for plan in im_lang:
        qd = sg.quyet_dinh_clip(rng, plan, CONFIG, nl)
        assert qd.su_kien == ()
        assert qd.nen is not None


def test_chi_dung_lop_co_trong_nguyen_lieu():
    """Không được bịa ra lớp nào ngoài bảng tra — bịa sẽ thành KeyError lúc sinh audio."""
    nl = _nguyen_lieu()
    rng = random.Random(0)

    for plan in _ke_hoach(400):
        for sk in sg.quyet_dinh_clip(rng, plan, CONFIG, nl).su_kien:
            assert sk.label in nl.bang
            assert sk.nguon.class_id == sk.label


# ── Báo cáo mô phỏng phải BIẾT NÓI KHÔNG ──────────────────────────────────────


def _qd_gia(index, su_kien):
    nen = sg.NenNguon("bg", "school", 10.0)
    sk = tuple(sg.SuKienQuyetDinh(lop, sg.ClipNguon(f"{lop}_0", lop, "x.wav", d + 0.2, d, 0.2),
                                  t, d, {}) for lop, t, d in su_kien)
    return sg.ClipQuyetDinh(index, "school", nen, 0.0, sk)


def test_bao_cao_bao_DAT_khi_moi_hop_dong_deu_du():
    plan = sg.ClipPlan(index=0, slices={"overlap", "long_event"}, n_events=2)
    qd = _qd_gia(0, [("siren", 0.0, 5.0), ("gunshot", 1.0, 2.0)])

    bao_cao = sg.bao_cao_mo_phong([qd], [plan], CONFIG)

    assert "✓" in bao_cao and "❌" not in bao_cao


def test_bao_cao_BAT_duoc_overlap_khong_dat():
    """Hai sự kiện rời hẳn nhau mà clip vẫn mang nhãn overlap."""
    plan = sg.ClipPlan(index=0, slices={"overlap"}, n_events=2)
    qd = _qd_gia(0, [("siren", 0.0, 1.0), ("gunshot", 5.0, 1.0)])

    bao_cao = sg.bao_cao_mo_phong([qd], [plan], CONFIG)

    assert "❌" in bao_cao and "overlap" in bao_cao


def test_bao_cao_BAT_duoc_long_event_khong_dat():
    """Đây chính xác là hỏng hóc lô cũ: nhãn long_event, sự kiện dài nhất 1 giây."""
    plan = sg.ClipPlan(index=0, slices={"long_event"}, n_events=1)
    qd = _qd_gia(0, [("gunshot", 0.0, 1.0)])

    bao_cao = sg.bao_cao_mo_phong([qd], [plan], CONFIG)

    assert "❌" in bao_cao and "long_event" in bao_cao


def test_bao_cao_BAT_duoc_su_kien_tran_khoi_clip():
    plan = sg.ClipPlan(index=0, slices=set(), n_events=1)
    qd = _qd_gia(0, [("siren", 9.5, 2.0)])

    bao_cao = sg.bao_cao_mo_phong([qd], [plan], CONFIG)

    assert "❌" in bao_cao and "tràn" in bao_cao


def test_mo_phong_khop_chinh_xac_voi_lua_chon_cua_quyet_dinh_clip():
    """Mô phỏng và lượt sinh thật đọc CÙNG một nguồn quyết định, nên phải trùng khít.

    Đây là điều làm B7 có giá trị: bảng nghiệm thu chạy trong vài giây nói đúng thứ
    sẽ sinh ra sau 18 giờ, chứ không phải một ước lượng gần đúng.
    """
    nl = _nguyen_lieu()
    plans = _ke_hoach(150)

    mo_phong = sg.mo_phong_nhan(CONFIG, plans, seed=3, nguyen_lieu=nl)

    # Từ B8, mỗi clip rút từ seed riêng của nó — xem `seed_clip`.
    truc_tiep = [sg.quyet_dinh_clip(random.Random(sg.seed_clip(3, plan.index)), plan, CONFIG, nl)
                 for plan in plans]
    assert [[(x.label, x.start, x.thoi_luong) for x in qd.su_kien] for qd in mo_phong] == \
           [[(x.label, x.start, x.thoi_luong) for x in qd.su_kien] for qd in truc_tiep]


# ── Chuỗi nhân quả phải LỌT VỪA clip (B9) ────────────────────────────────────


def test_ngan_sach_mat_xich_chia_deu_phan_con_lai():
    """10 s trừ khoảng cách nhỏ nhất (0.5 + 0.3), chia cho 3 mắt xích."""
    chuoi = CONFIG["causal_chains"][0]

    assert sg.ngan_sach_mat_xich(chuoi, 10.0) == pytest.approx((10.0 - 0.8) / 3)


def test_ngan_sach_mat_xich_chuoi_hai_mat_xich():
    chuoi = {"sequence": ["a", "b"], "gaps_sec": [[0.3, 2.0]]}

    assert sg.ngan_sach_mat_xich(chuoi, 10.0) == pytest.approx((10.0 - 0.3) / 2)


def test_chuoi_nhan_qua_LUON_dung_thu_tu_thoi_gian():
    """Vi phạm đã đo trên lô thật: 91/7.920 clip có mắt xích đảo thứ tự.

    Từ B3 mắt xích có thời lượng thật (`applause_cheering` p50 = 3,5 s), nên chuỗi
    tràn khỏi clip rất dễ; lúc đó `chain_event_times` trả rỗng và nhánh dự phòng đặt
    chúng tự do — mất thứ tự, mà clip vẫn mang nhãn `causal_chain`. Đúng hình dạng
    của lỗi `long_event`: nhãn có, nội dung không.
    """
    nl = _nguyen_lieu()
    rng = random.Random(0)
    so_clip = 0

    for plan in _ke_hoach(1500):
        if not plan.chain:
            continue
        so_clip += 1
        sk = sg.quyet_dinh_clip(rng, plan, CONFIG, nl).su_kien
        mat_xich = sg.chain_by_name(CONFIG["causal_chains"], plan.chain)["sequence"]
        moc = [x.start for x in sk[:len(mat_xich)]]

        assert [x.label for x in sk[:len(mat_xich)]] == mat_xich
        assert moc == sorted(moc), (
            f"clip {plan.index}: mắt xích lệch thứ tự thời gian {[round(m, 2) for m in moc]}")
    assert so_clip > 100


def test_mat_xich_khong_vuot_ngan_sach():
    nl = _nguyen_lieu()
    rng = random.Random(1)

    for plan in _ke_hoach(800):
        if not plan.chain:
            continue
        chuoi = sg.chain_by_name(CONFIG["causal_chains"], plan.chain)
        ngan_sach = sg.ngan_sach_mat_xich(chuoi, 10.0)
        sk = sg.quyet_dinh_clip(rng, plan, CONFIG, nl).su_kien

        for x in sk[:len(chuoi["sequence"])]:
            assert x.thoi_luong <= ngan_sach + 1e-9


def test_bao_cao_BAT_duoc_chuoi_sai_thu_tu():
    """91 clip lô đầu sai thứ tự mà bảng nghiệm thu vẫn báo '✓ mọi hợp đồng đạt'.

    Chỉ `verify_synthetic` — chạy SAU 47 phút sinh audio — mới thấy. Bịt lỗ này thì
    cùng lỗi đó bị bắt trong 3 giây.
    """
    plan = sg.ClipPlan(index=0, slices={"causal_chain"}, n_events=3, chain="break_in")
    dao = _qd_gia(0, [("glass_breaking", 5.0, 1.0), ("scream", 1.0, 1.0),
                      ("running_footsteps", 8.0, 1.0)])

    bao_cao = sg.bao_cao_mo_phong([dao], [plan], CONFIG)

    assert "❌" in bao_cao and "causal_chain" in bao_cao


def test_bao_cao_chap_nhan_chuoi_dung_thu_tu():
    plan = sg.ClipPlan(index=0, slices={"causal_chain"}, n_events=3, chain="break_in")
    dung = _qd_gia(0, [("glass_breaking", 1.0, 1.0), ("scream", 3.0, 1.0),
                       ("running_footsteps", 6.0, 1.0)])

    assert "❌" not in sg.bao_cao_mo_phong([dung], [plan], CONFIG)
