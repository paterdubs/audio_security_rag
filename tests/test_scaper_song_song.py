"""Test cho seed theo clip và khâu song song — STATUS §7 B8.

Đo thật trên lô 17/09: **8,4 clip/phút**, tức 9.360 clip mất **18,6 giờ**. Máy có 16
nhân logic và khâu này thuần CPU, mỗi clip độc lập — nhưng KHÔNG song song hoá được
chừng nào mọi clip còn rút chung một dòng RNG.

Đổi sang seed riêng cho từng clip gỡ nút đó, và kéo theo một thứ nữa: resume trở nên
CHÍNH XÁC thay vì phải replay. Cơ chế cũ buộc phải gọi lại `populate()` cho từng clip
đã xong chỉ để đẩy dòng RNG tới đúng vị trí; bỏ sót là mọi clip sinh sau đó rút trúng
số khác hẳn một lần chạy sạch — phá đúng lời hứa ở đầu `scaper_train.yaml`.

Bất biến phải giữ: clip thứ i ra kết quả y hệt nhau dù chạy tuần tự, chạy song song,
hay chạy lại sau khi bị ngắt.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import scaper_generate as sg  # noqa: E402
from test_scaper_quyet_dinh import CONFIG, _ke_hoach, _nguyen_lieu  # noqa: E402


# ── Seed theo clip ───────────────────────────────────────────────────────────


def test_seed_clip_tat_dinh():
    """Cùng (seed, index) → cùng số, mọi lần chạy, mọi máy."""
    assert sg.seed_clip(20260914, 42) == sg.seed_clip(20260914, 42)


def test_seed_clip_khac_nhau_giua_cac_clip():
    seeds = {sg.seed_clip(20260914, i) for i in range(5000)}

    assert len(seeds) == 5000, "có clip trùng seed"


def test_seed_clip_khac_nhau_giua_cac_split():
    """train và dev lệch nhau bằng seed_offset — không được trùng công thức.

    dev seed_offset = 1_000_000, nên cộng thẳng `seed + index` sẽ khiến dev clip 0
    trùng hệt train clip 1.000.000. Băm mới tách được hẳn hai dãy.
    """
    train = {sg.seed_clip(20260914, i) for i in range(2000)}
    dev = {sg.seed_clip(20260914 + 1_000_000, i) for i in range(2000)}

    assert not (train & dev)


def test_seed_clip_khong_phu_thuoc_hash_ngau_nhien_cua_python():
    """Giá trị phải cố định qua các lần khởi động Python.

    `hash()` của Python được ngẫu nhiên hoá theo PYTHONHASHSEED, nên dùng nó sẽ cho
    ra một dataset KHÁC mỗi lần khởi động — và không ai nhận ra, vì mọi phân bố vẫn
    y nguyên. Đó là lý do hàm này dùng băm ổn định.
    """
    import subprocess

    ma = ("import sys; sys.path.insert(0, r'scripts'); "
          "import scaper_generate as sg; print(sg.seed_clip(1, 7))")
    ra = {subprocess.run([sys.executable, "-c", ma], capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parent.parent),
                         env={"PYTHONHASHSEED": str(s), "PATH": ""}).stdout.strip()
          for s in (0, 1, 12345)}

    assert len(ra) == 1 and ra != {""}, f"seed_clip đổi theo PYTHONHASHSEED: {ra}"


# ── Độc lập giữa các clip ────────────────────────────────────────────────────


def test_clip_cho_cung_ket_qua_du_xu_ly_theo_THU_TU_NAO():
    """Nền của song song hoá: clip thứ i không phụ thuộc clip nào trước nó.

    Nếu còn phụ thuộc thì chạy 10 worker sẽ ra một dataset khác chạy tuần tự, mà cả
    hai đều "chạy xong bình thường" — không có triệu chứng nào.
    """
    nl = _nguyen_lieu()
    plans = _ke_hoach(60)

    xuoi = {p.index: sg.quyet_dinh_clip(random.Random(sg.seed_clip(5, p.index)), p, CONFIG, nl)
            for p in plans}
    nguoc = {p.index: sg.quyet_dinh_clip(random.Random(sg.seed_clip(5, p.index)), p, CONFIG, nl)
             for p in reversed(plans)}

    lay = lambda qd: [(x.label, x.nguon.file_id, round(x.start, 9), round(x.thoi_luong, 9))  # noqa: E731
                      for x in qd.su_kien]
    assert {i: lay(q) for i, q in xuoi.items()} == {i: lay(q) for i, q in nguoc.items()}


def test_bo_qua_mot_clip_khong_lam_lech_cac_clip_con_lai():
    """Đây chính là thứ resume kiểu cũ phải replay RNG mới giữ được.

    Với seed theo clip, bỏ qua clip nào cũng không ảnh hưởng clip khác — resume trở
    thành chính xác thay vì gần đúng.
    """
    nl = _nguyen_lieu()
    plans = _ke_hoach(40)

    day_du = [sg.quyet_dinh_clip(random.Random(sg.seed_clip(9, p.index)), p, CONFIG, nl)
              for p in plans]
    thieu = [sg.quyet_dinh_clip(random.Random(sg.seed_clip(9, p.index)), p, CONFIG, nl)
             for p in plans if p.index % 3]

    theo_index = {qd.index: [(x.label, x.start) for x in qd.su_kien] for qd in day_du}
    for qd in thieu:
        assert [(x.label, x.start) for x in qd.su_kien] == theo_index[qd.index]


def test_mo_phong_dung_dung_seed_theo_clip():
    """Bộ mô phỏng B7 phải rút y hệt lượt sinh thật — nếu không, bảng nghiệm thu nói dối."""
    nl = _nguyen_lieu()
    plans = _ke_hoach(50)

    mo_phong = sg.mo_phong_nhan(CONFIG, plans, seed=11, nguyen_lieu=nl)
    truc_tiep = [sg.quyet_dinh_clip(random.Random(sg.seed_clip(11, p.index)), p, CONFIG, nl)
                 for p in plans]

    assert [[(x.label, x.start, x.thoi_luong) for x in qd.su_kien] for qd in mo_phong] == \
           [[(x.label, x.start, x.thoi_luong) for x in qd.su_kien] for qd in truc_tiep]


# ── Chia việc cho worker ─────────────────────────────────────────────────────


def test_so_worker_khong_bao_gio_vuot_so_clip():
    """8 clip mà mở 16 worker là phí công khởi tạo — mỗi worker nạp lại cả bank."""
    assert sg.so_worker_thuc_te(yeu_cau=16, so_clip=8) == 8
    assert sg.so_worker_thuc_te(yeu_cau=16, so_clip=1000) == 16


def test_so_worker_toi_thieu_la_mot():
    assert sg.so_worker_thuc_te(yeu_cau=0, so_clip=100) == 1
    assert sg.so_worker_thuc_te(yeu_cau=-3, so_clip=100) == 1
    assert sg.so_worker_thuc_te(yeu_cau=None, so_clip=100) >= 1


def test_so_worker_khong_vuot_so_nhan_may():
    import os

    assert sg.so_worker_thuc_te(yeu_cau=None, so_clip=10_000) <= (os.cpu_count() or 1)


@pytest.mark.parametrize("so_clip", [0, 1])
def test_so_worker_voi_it_clip(so_clip):
    assert sg.so_worker_thuc_te(yeu_cau=8, so_clip=so_clip) == max(1, so_clip)


def test_clip_bo_qua_van_bao_duoc_rir_da_dung(tmp_path, monkeypatch):
    """Clip đã sinh xong vẫn phải trả về `rir_used`, không phải chuỗi rỗng.

    Đây là khiếm khuyết do chính B8 gây ra và cổng kiểm bắt được: lượt sinh lại bỏ qua
    6.759 clip đã có, trả `rir_used=""`, và `slice_index` thành ra khai "gắn nhãn
    reverb nhưng rir_used rỗng" cho 2.663 clip. Audio thì vẫn đúng — RIR đã được áp
    lúc sinh lần đầu — nên đây là lỗi SỔ SÁCH, loại chỉ cổng kiểm mới thấy.
    """
    nl = _nguyen_lieu()
    plan = next(p for p in _ke_hoach(300) if "reverb" in p.slices)
    out = tmp_path / "train"
    (out / "audio").mkdir(parents=True)
    (out / "jams").mkdir(parents=True)
    (out / "audio" / f"train_{plan.index:06d}.wav").write_bytes(b"da-co")
    (out / "jams" / f"train_{plan.index:06d}.jams").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(sg, "pick_rir", lambda rng, area: Path(f"rir/{area}/gia_lap.wav"))

    index, rir_used, da_sinh = sg.sinh_mot_clip(plan, "train", 7, CONFIG, None, nl, out, True)

    assert index == plan.index
    assert da_sinh is False, "clip đã có mà vẫn sinh lại"
    assert rir_used == "gia_lap", "mất thông tin RIR khi bỏ qua clip"
