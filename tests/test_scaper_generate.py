"""Test cho scripts/scaper_generate.py.

Phần lập kế hoạch tách hẳn khỏi phần sinh audio chính là để test được chỗ này: sai
tỉ lệ lát cắt hay sai thứ tự chuỗi nhân quả đều KHÔNG có triệu chứng nào khi chạy,
chúng chỉ làm các con số trong báo cáo mô tả sai thứ thực sự đã sinh ra.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import scaper_generate as sg  # noqa: E402

CONFIG = {
    "duration_sec": 10.0,
    "events": {
        "count_weights": {0: 0.10, 1: 0.30, 2: 0.30, 3: 0.20, 4: 0.10},
        "snr_db": [-5, 25],
        "pitch_shift_semitones": [-1, 1],
        "time_stretch": [0.9, 1.1],
    },
    "forced_slices": {
        "overlap": {"ratio": 0.30, "min_events": 2, "min_overlap_ratio": 0.30},
        "low_snr": {"ratio": 0.25, "max_snr_db": 5},
        "reverb": {"ratio": 0.40},
        "long_event": {"ratio": 0.10, "min_event_duration_sec": 8.0},
        "causal_chain": {"ratio": 0.15},
    },
    "causal_chains": [
        {"name": "break_in", "sequence": ["glass_breaking", "scream", "running_footsteps"],
         "gaps_sec": [[0.5, 2.0], [0.3, 1.5]]},
        {"name": "false_alarm_fireworks", "sequence": ["fireworks", "laughter_cheering"],
         "gaps_sec": [[0.3, 2.0]]},
    ],
}

N = 20000      # đủ lớn để sai số lấy mẫu nhỏ hơn ngưỡng kiểm tra


# ── Tỉ lệ lát cắt: con số này đi thẳng vào báo cáo ───────────────────────────


@pytest.mark.parametrize("name", sg.SLICE_NAMES)
def test_ti_le_lat_cat_dat_dung_dich_tren_toan_bo_split(name: str):
    """Tỉ lệ trong config là tỉ lệ trên TOÀN BỘ split, kể cả clip im lặng.

    Không bù phần clip im lặng thì mọi lát cắt định nghĩa trên sự kiện đều thấp hơn
    đích đúng 10% — đo được 26.9% trong khi tài liệu ghi 30%. Đủ nhỏ để không ai để ý.
    """
    plans = sg.plan_clips(CONFIG, N, seed=1)
    thuc_te = sum(1 for p in plans if name in p.slices) / N
    dich = CONFIG["forced_slices"][name]["ratio"]
    assert abs(thuc_te - dich) < 0.015, f"{name}: {thuc_te:.1%} vs đích {dich:.0%}"


def test_ti_le_clip_im_lang_dat_dung_trong_so():
    plans = sg.plan_clips(CONFIG, N, seed=1)
    assert abs(sum(1 for p in plans if p.n_events == 0) / N - 0.10) < 0.015


def test_clip_im_lang_khong_mang_lat_cat_dinh_nghia_tren_su_kien():
    for plan in sg.plan_clips(CONFIG, N, seed=1):
        if plan.n_events == 0:
            assert not (plan.slices & sg.EVENT_BASED_SLICES)


def test_clip_im_lang_van_mang_duoc_reverb():
    # reverb là tích chập áp lên cả clip, không cần sự kiện nào.
    plans = sg.plan_clips(CONFIG, N, seed=1)
    assert any("reverb" in p.slices for p in plans if p.n_events == 0)


def test_dat_ti_le_khong_the_dat_thi_dung_chu_khong_im_lang_lam_sai():
    config = {**CONFIG, "forced_slices": {**CONFIG["forced_slices"],
                                          "overlap": {"ratio": 0.95, "min_events": 2}}}
    with pytest.raises(SystemExit) as err:
        sg.plan_clips(config, 10, seed=1)
    assert "overlap" in str(err.value)


# ── Ràng buộc nội tại của kế hoạch ───────────────────────────────────────────


def test_clip_overlap_luon_co_it_nhat_hai_su_kien():
    for plan in sg.plan_clips(CONFIG, N, seed=2):
        if "overlap" in plan.slices:
            assert plan.n_events >= 2


def test_clip_chuoi_nhan_qua_du_cho_moi_su_kien_cua_chuoi():
    chains = {c["name"]: c for c in CONFIG["causal_chains"]}
    for plan in sg.plan_clips(CONFIG, N, seed=2):
        if "causal_chain" in plan.slices:
            assert plan.chain is not None
            assert plan.n_events >= len(chains[plan.chain]["sequence"])


def test_khong_mang_chuoi_thi_khong_gan_ten_chuoi():
    for plan in sg.plan_clips(CONFIG, N, seed=2):
        if "causal_chain" not in plan.slices:
            assert plan.chain is None


def test_ke_hoach_tat_dinh_theo_seed():
    a = sg.plan_clips(CONFIG, 500, seed=7)
    b = sg.plan_clips(CONFIG, 500, seed=7)
    assert [(p.slices, p.n_events, p.chain) for p in a] == [(p.slices, p.n_events, p.chain) for p in b]


def test_doi_seed_thi_ke_hoach_doi():
    a = sg.plan_clips(CONFIG, 500, seed=7)
    b = sg.plan_clips(CONFIG, 500, seed=8)
    assert [(p.slices, p.n_events) for p in a] != [(p.slices, p.n_events) for p in b]


# ── Chuỗi nhân quả: THỨ TỰ mới là thứ mang nghĩa ─────────────────────────────


def test_thoi_diem_cac_su_kien_giu_dung_thu_tu_chuoi():
    chain = CONFIG["causal_chains"][0]
    for seed in range(200):
        times = sg.chain_event_times(chain, random.Random(seed), 10.0)
        assert times == sorted(times), "đảo thứ tự là đổi nghĩa: đột nhập ≠ ngẫu nhiên"
        assert len(times) == len(chain["sequence"])


def test_chuoi_nam_tron_trong_clip():
    chain = CONFIG["causal_chains"][0]
    for seed in range(200):
        times = sg.chain_event_times(chain, random.Random(seed), 10.0, event_duration=1.0)
        assert times[0] >= 0 and times[-1] + 1.0 <= 10.0 + 1e-9


def test_khoang_cach_nam_trong_dai_da_khai():
    chain = CONFIG["causal_chains"][0]
    for seed in range(200):
        times = sg.chain_event_times(chain, random.Random(seed), 10.0, event_duration=1.0)
        for position, (low, high) in enumerate(chain["gaps_sec"]):
            gap = times[position + 1] - times[position] - 1.0
            assert low - 1e-6 <= gap <= high + 1e-6


def test_chuoi_khong_vua_clip_thi_bo_han_chu_khong_cat_cut():
    # Cắt cụt `glass_breaking → scream → running_footsteps` thành hai sự kiện đầu
    # sẽ tạo ra một clip mang nhãn "chuỗi đột nhập" nhưng nội dung là chuỗi khác.
    chain = CONFIG["causal_chains"][0]
    assert sg.chain_event_times(chain, random.Random(0), duration=2.0) == []


def test_co_khoang_cach_ve_muc_nho_nhat_truoc_khi_bo():
    chain = CONFIG["causal_chains"][0]
    # 3 sự kiện × 1 s + khoảng cách nhỏ nhất (0.5 + 0.3) = 3.8 s → vừa đủ.
    assert sg.chain_event_times(chain, random.Random(0), duration=3.8) != []


def test_ten_chuoi_la_thi_dung():
    with pytest.raises(SystemExit):
        sg.chain_by_name(CONFIG["causal_chains"], "khong_ton_tai")


# ── Cổng chống rò rỉ vào train set ───────────────────────────────────────────


def test_bat_duoc_clip_ngoai_bank_lot_vao_foreground():
    files = [Path("bank/siren/a.wav"), Path("bank/siren/b.wav")]
    assert sg.check_bank_clean(files, eligible={"a"}) == ["b"]


def test_bank_sach_thi_khong_bao_gi():
    files = [Path("bank/siren/a.wav")]
    assert sg.check_bank_clean(files, eligible={"a", "b"}) == []


# ── Chuỗi nhân quả thiếu mắt xích ────────────────────────────────────────────


def test_bo_chuoi_khi_bank_thieu_mot_lop():
    """Sinh chuỗi CỤT mà vẫn mang nhãn của chuỗi đủ là dạy model định nghĩa sai.

    `glass_breaking → scream` vẫn được ghi là kịch bản "đột nhập" dù thiếu hẳn tiếng
    chân chạy. Xảy ra thật: bank không có `shout_yell` (0 clip auto-accept), làm hai
    kịch bản forced_entry và emergency không sinh được.
    """
    chains = [
        {"name": "break_in", "sequence": ["glass_breaking", "scream", "running_footsteps"]},
        {"name": "forced_entry", "sequence": ["door_slam", "shout_yell", "running_footsteps"]},
    ]
    ok, bo = sg.usable_chains(chains, {"glass_breaking", "scream", "running_footsteps", "door_slam"})
    assert [c["name"] for c in ok] == ["break_in"]
    assert bo == [("forced_entry", ["shout_yell"])]


def test_bank_du_moi_lop_thi_khong_bo_chuoi_nao():
    chains = [{"name": "x", "sequence": ["a", "b"]}]
    ok, bo = sg.usable_chains(chains, {"a", "b", "c"})
    assert len(ok) == 1 and bo == []


def test_neu_ten_moi_lop_con_thieu_de_sua_duoc_ngay():
    chains = [{"name": "x", "sequence": ["a", "b", "c"]}]
    _, bo = sg.usable_chains(chains, {"a"})
    assert bo[0][1] == ["b", "c"]


def test_khong_con_chuoi_nao_thi_khong_clip_nao_mang_lat_cat_do():
    # Không bỏ lát cắt thì plan_clips chọn chuỗi từ danh sách rỗng và nổ ở rng.choice.
    plans = sg.plan_clips(CONFIG, 500, seed=3, chains=[])
    assert all("causal_chain" not in p.slices and p.chain is None for p in plans)


def test_chi_chon_trong_cac_chuoi_con_dung_duoc():
    chi_mot = [CONFIG["causal_chains"][0]]
    plans = sg.plan_clips(CONFIG, 500, seed=3, chains=chi_mot)
    ten = {p.chain for p in plans if p.chain}
    assert ten <= {"break_in"}


# ── Tích chập RIR ────────────────────────────────────────────────────────────


def test_tich_chap_giu_nguyen_do_dai():
    """Tích chập cho ra tín hiệu dài thêm bằng đuôi RIR (tới 2 giây). Không cắt về độ
    dài cũ thì clip 10 giây thành 12 giây trong khi .jams vẫn mô tả clip 10 giây."""
    audio = np.random.default_rng(0).normal(0, 0.1, 16000).astype(np.float32)
    rir = np.random.default_rng(1).normal(0, 0.05, 8000).astype(np.float32)
    assert len(sg.apply_rir(audio, rir)) == len(audio)


def test_tich_chap_khong_de_vuot_tran():
    # Cộng hưởng có thể đẩy đỉnh vượt 1.0; vượt trần là méo cứng đi thẳng vào train set.
    audio = np.ones(4000, dtype=np.float32) * 0.9
    rir = np.ones(500, dtype=np.float32)
    assert np.abs(sg.apply_rir(audio, rir)).max() <= 1.0


def test_tich_chap_khong_chuan_hoa_lai_toan_bo():
    """Chỉ hạ khi vượt trần, KHÔNG chuẩn hoá lại: chuẩn hoá lại sẽ phá tỉ số SNR mà
    Scaper vừa đặt cẩn thận, và SNR là tham số định nghĩa lát cắt low_snr."""
    audio = (np.random.default_rng(0).normal(0, 0.01, 4000)).astype(np.float32)
    rir = np.zeros(100, dtype=np.float32)
    rir[0] = 1.0                       # RIR đơn vị: đầu ra phải gần y hệt đầu vào
    ra = sg.apply_rir(audio, rir)
    assert np.abs(ra).max() == pytest.approx(np.abs(audio).max(), rel=1e-5)


def test_clip_im_lang_tich_chap_khong_no():
    assert len(sg.apply_rir(np.zeros(1000, dtype=np.float32), np.zeros(100, dtype=np.float32))) == 1000


# ── Khu vực nền quyết định loại phòng ────────────────────────────────────────


def test_moi_khu_vuc_deu_co_loai_phong_tuong_ung():
    """Ghép ngẫu nhiên sẽ cho clip nền nhà xe mà vang như phòng học — model học được
    rằng vang và bối cảnh không liên quan gì nhau, trong khi ngoài đời chúng đi liền."""
    import yaml
    khu_vuc = yaml.safe_load((sg.REPO_ROOT / "ml/configs/background_map.yaml").read_text(encoding="utf-8"))
    thieu = set(khu_vuc["area_types"]) - set(sg.AREA_TO_RIR_SPACE)
    assert not thieu, f"khu vực chưa có loại phòng: {sorted(thieu)}"


def test_nha_xe_va_xuong_dung_phong_lon():
    # Hai không gian này vang dài nhất trong bốn khu vực triển khai.
    assert sg.AREA_TO_RIR_SPACE["parking"] == "large_room"
    assert sg.AREA_TO_RIR_SPACE["factory"] == "large_room"


def test_ke_hoach_gan_khu_vuc_cho_moi_clip():
    plans = sg.plan_clips(CONFIG, 200, seed=5, areas=["school", "parking"])
    assert all(p.area_type in {"school", "parking"} for p in plans)


def test_khong_co_bank_nen_thi_de_trong_chu_khong_bia():
    assert all(p.area_type == "" for p in sg.plan_clips(CONFIG, 20, seed=5, areas=[]))
