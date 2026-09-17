"""Test cho khâu ĐẶT THỜI ĐIỂM của scaper_generate.py — STATUS §7 B4 và B6.

Hợp đồng khai `overlap: {ratio: 0.30, min_events: 2, min_overlap_ratio: 0.30}`, và
log sinh báo "overlap 30% · 2387 clip". Nhưng đếm lại trên chính 7.920 file nhãn
đã sinh: chỉ 1.752 clip (22.1%) có chồng lấn THẬT, và trong đó 74.0% số cặp đạt
ngưỡng 0.30. `min_overlap_ratio` chưa bao giờ được dùng tới lúc sinh — chồng lấn
hiện là TÌNH CỜ, do `event_time=("uniform", 0, 10)` thỉnh thoảng va nhau.

Một suy luận quan trọng cho thiết kế: ép chồng lấn LUÔN khả thi, không cần nhánh
dự phòng. Với sự kiện neo tại t₀ dài d₀ và sự kiện thứ hai dài d₁, giao cần
o = r·min(d₀,d₁), tập t₁ hợp lệ là

    [t₀ + o − d₁, t₀ + d₀ − o]   giao   [0, D − d₁]

và cả bốn điều kiện làm nó rỗng đều bất khả vì o ≤ min(d₀,d₁) và t₀ ≤ D − d₀.
Điều đó đáng khoá bằng test, vì một nhánh dự phòng âm thầm chính là cách hợp đồng
`long_event ≥ 8s` báo đạt 791 clip trong khi thực tế sinh ra 0 sự kiện như vậy.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import scaper_generate as sg  # noqa: E402

D = 10.0


# ── Đo chồng lấn ─────────────────────────────────────────────────────────────


def test_do_chong_lan_cac_ca_co_ban():
    assert sg.do_chong_lan(0.0, 2.0, 1.0, 2.0) == pytest.approx(1.0)      # giao một nửa
    assert sg.do_chong_lan(0.0, 1.0, 5.0, 1.0) == 0.0                     # rời hẳn
    assert sg.do_chong_lan(0.0, 1.0, 1.0, 1.0) == 0.0                     # chạm đầu đuôi
    assert sg.do_chong_lan(1.0, 2.0, 0.0, 9.0) == pytest.approx(2.0)      # lồng trọn


# ── Đặt ngẫu nhiên (B6) ──────────────────────────────────────────────────────


def test_moc_ngau_nhien_khong_bao_gio_tran_qua_cuoi_clip():
    """start + d ≤ D. Lô cũ dùng uniform(0, 10) nên 1.225 sự kiện bị cắt ở mốc 10 s
    và 813 sự kiện khởi đầu sau giây thứ 9 — một đống mẩu cực ngắn không có thật."""
    rng = random.Random(0)

    for d in (0.2, 1.0, 4.0, 9.9, 10.0):
        for _ in range(200):
            moc = sg.moc_ngau_nhien(rng, d, D)
            assert 0.0 <= moc <= D - d + 1e-9


def test_moc_ngau_nhien_su_kien_dai_bang_clip_thi_bat_dau_tu_0():
    rng = random.Random(0)

    assert sg.moc_ngau_nhien(rng, 10.0, D) == pytest.approx(0.0)


def test_moc_ngau_nhien_trai_deu_tren_khoang_hop_le():
    rng = random.Random(1)

    moc = [sg.moc_ngau_nhien(rng, 2.0, D) for _ in range(2000)]

    assert min(moc) < 0.5 and max(moc) > 7.5, "không phủ hết khoảng hợp lệ"


# ── Ép chồng lấn (B4) ────────────────────────────────────────────────────────


def test_moc_chong_lan_luon_dat_nguong_tren_dai_tham_so_rong():
    """Test xương sống B4: mọi tổ hợp đều đạt ngưỡng, KHÔNG có ca thất bại.

    Quét cả các ca ngặt nhất — sự kiện dài gần bằng clip, sự kiện cực ngắn, neo sát
    mép — vì đó là chỗ một cài đặt cẩu thả sẽ lặng lẽ trả về mức chồng lấn thấp hơn.
    """
    rng = random.Random(2)

    for d_neo in (0.2, 1.0, 3.0, 7.0, 9.9):
        for d in (0.2, 1.0, 3.0, 7.0, 9.9):
            for ty_le in (0.3, 0.5, 1.0):
                for _ in range(20):
                    t_neo = sg.moc_ngau_nhien(rng, d_neo, D)

                    moc = sg.moc_chong_lan(rng, t_neo, d_neo, d, D, ty_le)

                    giao = sg.do_chong_lan(t_neo, d_neo, moc, d)
                    can = ty_le * min(d_neo, d)
                    assert giao >= can - 1e-9, f"{giao:.4f} < {can:.4f}"
                    assert 0.0 <= moc <= D - d + 1e-9


def test_moc_chong_lan_ty_le_1_cho_su_kien_ngan_nam_tron_trong_su_kien_dai():
    """r = 1.0 nghĩa là sự kiện ngắn phải lọt HẲN vào trong sự kiện dài."""
    rng = random.Random(3)

    for _ in range(200):
        t_neo = sg.moc_ngau_nhien(rng, 4.0, D)

        moc = sg.moc_chong_lan(rng, t_neo, 4.0, 1.0, D, 1.0)

        assert t_neo - 1e-9 <= moc and moc + 1.0 <= t_neo + 4.0 + 1e-9


def test_moc_chong_lan_van_bien_thien_chu_khong_dinh_dung_nguong():
    """Chồng lấn phải trải trên khoảng hợp lệ, không phải luôn đúng bằng mức tối thiểu.

    Luôn đúng bằng ngưỡng thì model học được một quy luật hình học của bộ sinh chứ
    không học chồng lấn âm thanh.
    """
    rng = random.Random(4)

    giao = [sg.do_chong_lan(2.0, 3.0, sg.moc_chong_lan(rng, 2.0, 3.0, 3.0, D, 0.30), 3.0)
            for _ in range(2000)]

    assert min(giao) < 1.2 and max(giao) > 2.5, "chồng lấn không biến thiên"


def test_moc_chong_lan_tat_dinh_theo_seed():
    lay = lambda: [sg.moc_chong_lan(random.Random(5), 1.0, 2.0, 2.0, D, 0.3) for _ in range(10)]  # noqa: E731
    assert lay() == lay()


def test_moc_chong_lan_hai_su_kien_dai_hon_nua_clip():
    """d₀ = d₁ = 7 s trong clip 10 s: buộc phải chồng ít nhất 4 s dù không ai ép."""
    rng = random.Random(6)

    for _ in range(100):
        t_neo = sg.moc_ngau_nhien(rng, 7.0, D)

        moc = sg.moc_chong_lan(rng, t_neo, 7.0, 7.0, D, 0.30)

        assert sg.do_chong_lan(t_neo, 7.0, moc, 7.0) >= 4.0 - 1e-9


# ── Đặt cả cụm sự kiện của một clip ──────────────────────────────────────────


def test_dat_cum_khong_ep_thi_moi_su_kien_deu_nam_tron_trong_clip():
    rng = random.Random(7)

    moc = sg.dat_cum(rng, [0.5, 2.0, 4.0], D, ty_le_chong_lan=None)

    assert len(moc) == 3
    assert all(0.0 <= t and t + d <= D + 1e-9 for t, d in zip(moc, [0.5, 2.0, 4.0]))


def test_dat_cum_co_ep_thi_luon_ton_tai_mot_cap_dat_nguong():
    """Đây là điều hợp đồng hứa và lô cũ không giữ: 100% clip overlap phải có cặp thật."""
    rng = random.Random(8)

    for _ in range(300):
        dai = [rng.uniform(0.2, 5.0) for _ in range(rng.randint(2, 4))]

        moc = sg.dat_cum(rng, dai, D, ty_le_chong_lan=0.30)

        cap_dat = [
            sg.do_chong_lan(moc[i], dai[i], moc[j], dai[j]) >= 0.30 * min(dai[i], dai[j]) - 1e-9
            for i in range(len(dai)) for j in range(i + 1, len(dai))
        ]
        assert any(cap_dat), "không cặp nào đạt ngưỡng chồng lấn"


def test_dat_cum_ep_chong_lan_can_it_nhat_hai_su_kien():
    with pytest.raises(ValueError, match="hai sự kiện"):
        sg.dat_cum(random.Random(0), [1.0], D, ty_le_chong_lan=0.30)


def test_dat_cum_giu_nguyen_moc_da_co_dinh_cua_chuoi_nhan_qua():
    """Chuỗi nhân quả đã định sẵn thời điểm; ép chồng lấn không được xê dịch chúng.

    Thứ tự mới là thứ mang nghĩa: glass_breaking → scream → running_footsteps là
    đột nhập, đảo lại thì không.
    """
    rng = random.Random(9)
    co_dinh = [1.0, 3.5]

    moc = sg.dat_cum(rng, [1.0, 1.0, 2.0, 2.0], D, ty_le_chong_lan=0.30, moc_co_dinh=co_dinh)

    assert moc[:2] == co_dinh
    assert len(moc) == 4


# ── Kế hoạch phải chừa chỗ cho cặp bị ép ─────────────────────────────────────


def test_plan_clip_vua_co_chuoi_vua_overlap_van_con_su_kien_tu_do():
    """Chuỗi 3 mắt xích + overlap → phải có ≥ 5 sự kiện, không phải 3.

    Lô cũ đặt n_events = max(n_events, len(chuỗi)) nên chuỗi ăn hết ngân sách: clip
    mang nhãn `overlap` mà không còn sự kiện tự do nào để ép, và nó vẫn được tính
    vào tỉ lệ 30% trong báo cáo. Đúng kiểu hỏng hóc không có triệu chứng.
    """
    from test_scaper_generate import CONFIG  # tái dùng cấu hình mẫu đã có

    ke_hoach = sg.plan_clips(CONFIG, n_clips=4000, seed=1)
    ca_kho = [p for p in ke_hoach if p.chain and "overlap" in p.slices]

    assert ca_kho, "seed này không sinh ra ca vừa có chuỗi vừa overlap"
    for p in ca_kho:
        so_mat_xich = len(sg.chain_by_name(CONFIG["causal_chains"], p.chain)["sequence"])
        assert p.n_events >= so_mat_xich + 2, f"chỉ {p.n_events} sự kiện cho chuỗi {so_mat_xich} mắt xích"


def test_so_su_kien_tu_do_can_dung_chung_ngan_sach():
    """overlap cần 2, long_event cần 1, cả hai cùng lúc vẫn là 2 — không cộng thành 3.

    Sự kiện dài được phép đồng thời là một nửa của cặp chồng lấn. Cộng dồn sẽ phình
    số sự kiện của những clip khó nhất mà không đổi lại gì.
    """
    forced = {"overlap": {"min_events": 2}}

    assert sg.so_su_kien_tu_do_can(set(), forced) == 0
    assert sg.so_su_kien_tu_do_can({"long_event"}, forced) == 1
    assert sg.so_su_kien_tu_do_can({"overlap"}, forced) == 2
    assert sg.so_su_kien_tu_do_can({"overlap", "long_event"}, forced) == 2
    assert sg.so_su_kien_tu_do_can({"reverb", "low_snr"}, forced) == 0


def test_plan_clip_co_long_event_luon_con_it_nhat_mot_su_kien_tu_do():
    """Chuỗi 3 mắt xích + long_event → phải có ≥ 4 sự kiện.

    Cùng một lỗ hổng như overlap: chuỗi ăn hết ngân sách thì không còn sự kiện tự do
    nào để ép cho dài, mà clip vẫn được tính vào tỉ lệ 10% trong báo cáo.
    """
    from test_scaper_generate import CONFIG

    ke_hoach = sg.plan_clips(CONFIG, n_clips=4000, seed=2)
    ca_kho = [p for p in ke_hoach if p.chain and "long_event" in p.slices]

    assert ca_kho, "seed này không sinh ra ca vừa có chuỗi vừa long_event"
    for p in ca_kho:
        so_mat_xich = len(sg.chain_by_name(CONFIG["causal_chains"], p.chain)["sequence"])
        assert p.n_events >= so_mat_xich + 1


# ── Đặt vào chỗ trống (B6) ───────────────────────────────────────────────────
#
# Đặt thuần ngẫu nhiên làm độ phủ sóng BÃO HOÀ: gần gấp đôi số sự kiện (1.90 → 3.59
# trung bình) chỉ nâng phủ sóng 35.7% → 47.7%, vì phần thêm chui vào chỗ đã có tiếng,
# trong khi chồng lấn phình 55% → 74%.
#
# Miền đích không như vậy. Đo trên AudioSet train_strong: chỉ 17.1% thời lượng sự
# kiện bị chồng lấn — hệ số nén hợp/tổng = 0.829. Sự kiện ở đó xếp gần như NỐI TIẾP.


def _he_so_nen(moc: list[float], dai: list[float]) -> float:
    """hợp / tổng. 1.000 = không chồng lấn chút nào; càng nhỏ càng chồng nhiều."""
    khoang = sorted((t, t + d) for t, d in zip(moc, dai))
    tong, a, b = 0.0, *khoang[0]
    for x, y in khoang[1:]:
        if x > b:
            tong += b - a
            a, b = x, y
        else:
            b = max(b, y)
    return (tong + b - a) / sum(dai)


def test_moc_it_va_cham_tim_duoc_cho_trong():
    """Nửa đầu clip đã kín thì sự kiện mới phải rơi vào nửa sau."""
    rng = random.Random(0)
    da_dat = [(0.0, 5.0)]

    moc = [sg.moc_it_va_cham(rng, 1.0, D, da_dat, so_thu=16) for _ in range(50)]

    assert all(t >= 5.0 - 1e-9 for t in moc), "vẫn đâm vào chỗ đã có tiếng"


def test_moc_it_va_cham_het_cho_thi_chon_it_va_cham_nhat_chu_khong_bo_cuoc():
    """Clip kín đặc vẫn phải trả về một vị trí hợp lệ.

    Chồng lấn CÓ THẬT ngoài đời. Ném lỗi hay bỏ cuộc ở đây sẽ biến một tình huống
    bình thường thành lỗi giữa chừng lượt sinh 18 giờ.
    """
    moc = sg.moc_it_va_cham(random.Random(0), 2.0, D, [(0.0, 10.0)], so_thu=16)

    assert 0.0 <= moc <= D - 2.0 + 1e-9


def test_moc_it_va_cham_khong_bao_gio_tran_qua_cuoi_clip():
    rng = random.Random(1)

    for d in (0.2, 3.0, 9.9, 10.0):
        for _ in range(100):
            t = sg.moc_it_va_cham(rng, d, D, [(1.0, 2.0), (6.0, 1.0)], so_thu=8)
            assert 0.0 <= t <= D - d + 1e-9


def test_moc_it_va_cham_tat_dinh_va_so_lan_goi_rng_khong_doi():
    """Số lần rút RNG chỉ phụ thuộc `so_thu`, không phụ thuộc dữ liệu.

    Dừng sớm khi gặp vị trí không va chạm sẽ làm số lần rút phụ thuộc nội dung clip,
    và bước song song hoá B8 mất khả năng tái lập từng clip một cách độc lập.
    """
    def lay(da_dat):
        rng = random.Random(5)
        sg.moc_it_va_cham(rng, 1.0, D, da_dat, so_thu=8)
        return rng.random()

    assert lay([]) == lay([(0.0, 9.0)]) == lay([(0.0, 1.0), (5.0, 1.0)])


def test_dat_cum_giam_chong_lan_so_voi_dat_thuan_ngau_nhien():
    """Số đo xương sống B6: hệ số nén phải nhích hẳn về phía miền đích (0.829).

    So với cách đặt CŨ — thuần ngẫu nhiên đều trên [0, D−d] — chứ không so hai mức
    `so_vi_tri_thu`, vì cả hai mức giờ đều đi qua nhánh khoảng trống trước.

    Hàm chạy không lỗi mà hệ số nén không đổi thì ta vẫn sinh ra một train set chồng
    lấn dày hơn thực tế nhiều, và đó lại là thứ không có triệu chứng.
    """
    dai = [1.5, 2.0, 1.0, 2.5]
    cu, moi = [], []
    for seed in range(300):
        rng = random.Random(seed)
        cu.append(_he_so_nen([sg.moc_ngau_nhien(rng, d, D) for d in dai], dai))
        moi.append(_he_so_nen(sg.dat_cum(random.Random(seed), dai, D), dai))

    assert sum(moi) / 300 > sum(cu) / 300 + 0.10


def test_dat_cum_van_giu_bao_dam_chong_lan_khi_bi_ep():
    """B6 không được phá B4: clip bị ép vẫn phải có cặp đạt ngưỡng."""
    rng = random.Random(7)

    for _ in range(300):
        dai = [rng.uniform(0.2, 4.0) for _ in range(rng.randint(2, 5))]

        moc = sg.dat_cum(rng, dai, D, ty_le_chong_lan=0.30)

        assert any(sg.do_chong_lan(moc[i], dai[i], moc[j], dai[j]) >= 0.30 * min(dai[i], dai[j]) - 1e-9
                   for i in range(len(dai)) for j in range(i + 1, len(dai)))


def test_khoang_trong_tim_dung_khe_con_lai():
    """Đã chiếm [2,4] và [6,7], sự kiện 1 s còn ba khe: [0,1], [4,5], [7,9]."""
    trong = sg.khoang_trong([(2.0, 2.0), (6.0, 1.0)], D, d=1.0)

    assert trong == [(0.0, 1.0), (4.0, 5.0), (7.0, 9.0)]


def test_khoang_trong_bo_khe_hep_hon_su_kien():
    """Khe 1 s không chứa nổi sự kiện 2 s — phải loại, không trả khoảng âm."""
    trong = sg.khoang_trong([(0.0, 2.0), (3.0, 7.0)], D, d=2.0)

    assert trong == [] or all(cao >= thap for thap, cao in trong)
    assert not any(thap <= 2.0 <= cao for thap, cao in trong), "trả về khe 1 s không đủ rộng"


def test_khoang_trong_gop_cac_su_kien_chong_nhau():
    """Hai sự kiện chồng nhau là MỘT vùng bị chiếm, không phải hai.

    [1,4] và [2,5] gộp thành [1,5]. Sự kiện 1 s vì thế chỉ đặt được tại đúng t=0
    (khe [0,1] vừa khít) hoặc trong [5,9] — nên khoảng đầu là (0.0, 0.0), không
    phải (0.0, 1.0): mốc trả về là mốc BẮT ĐẦU, đã trừ sẵn độ dài sự kiện.
    """
    assert sg.khoang_trong([(1.0, 3.0), (2.0, 3.0)], D, d=1.0) == [(0.0, 0.0), (5.0, 9.0)]


def test_khoang_trong_clip_rong_va_clip_kin():
    assert sg.khoang_trong([], D, d=2.0) == [(0.0, 8.0)]
    assert sg.khoang_trong([(0.0, 10.0)], D, d=1.0) == []


def test_moc_trong_khoang_trong_rut_deu_theo_DO_DAI_khe():
    """Khe rộng phải được chọn nhiều hơn khe hẹp, tỉ lệ theo độ dài.

    Rút đều theo SỐ khe sẽ dồn sự kiện vào các khe hẹp — khe hẹp cũng nhiều phiếu
    như khe rộng — và sự kiện bám dính nhau ở vài chỗ thay vì trải ra.
    """
    rng = random.Random(0)
    trong = [(0.0, 0.1), (5.0, 9.0)]

    ra = [sg.moc_trong_khoang_trong(rng, trong) for _ in range(2000)]

    assert sum(1 for t in ra if t >= 5.0) / 2000 > 0.90


def test_dat_cum_dung_cho_trong_khi_con_cho():
    """Ba sự kiện ngắn trong clip 10 s thì KHÔNG được chồng lấn chút nào."""
    for seed in range(100):
        dai = [0.5, 0.8, 0.6]

        moc = sg.dat_cum(random.Random(seed), dai, D)

        assert all(sg.do_chong_lan(moc[i], dai[i], moc[j], dai[j]) == 0.0
                   for i in range(3) for j in range(i + 1, 3)), f"seed {seed} vẫn chồng lấn"


def test_dat_cum_het_cho_thi_van_dat_duoc_khong_ne_m_loi():
    """Bốn sự kiện 3 s trong clip 10 s buộc phải chồng — không được ném lỗi."""
    dai = [3.0] * 4

    moc = sg.dat_cum(random.Random(0), dai, D)

    assert len(moc) == 4
    assert all(0.0 <= t <= D - 3.0 + 1e-9 for t in moc)
