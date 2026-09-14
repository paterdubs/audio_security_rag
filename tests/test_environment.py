"""Kiểm tra môi trường khớp mức ghim — chặn loại lỗi đã lọt hai lần trong ngày 14/09.

Trôi phiên bản thư viện không làm gì hỏng ngay và không để lại dấu vết. Lần một:
môi trường chạy numpy 2.5.3 trong khi requirements ghim 1.26.4, và 852 clip đầu tiên
được chuẩn hoá trong môi trường sai đó — chỉ lộ ra khi thử chạy scaper. Lần hai: cài
bộ sàng lọc PANNs kéo theo matplotlib, matplotlib gỡ numpy 1.26 và cài numpy 2.0.2,
scaper chết lại — không có gì nối hai sự kiện ấy với nhau.

Vì vậy mức ghim phải được kiểm tra bằng test, chứ không bằng trí nhớ của người cài.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = [REPO_ROOT / "requirements-data.txt", REPO_ROOT / "requirements-screen.txt"]

# Chỉ kiểm tra những gói mà sai phiên bản sẽ làm SAI DỮ LIỆU hoặc làm chết một bước,
# chứ không kiểm tra mọi gói — pin toàn bộ rồi test toàn bộ chỉ tạo ra việc bảo trì.
CRITICAL = ["numpy", "scipy", "librosa", "soundfile", "pyloudnorm", "scaper"]

PIN_RE = re.compile(r"^([A-Za-z0-9_.\-]+)==([0-9][^\s#]*)", re.MULTILINE)


def pinned_versions() -> dict[str, str]:
    """{tên gói viết thường: phiên bản} gộp từ mọi file requirements.

    Gói nào xuất hiện ở nhiều file thì mức ghim PHẢI trùng nhau — chính chỗ này là
    nơi numpy bị hai file khai hai đằng và pip âm thầm chọn một.
    """
    pins: dict[str, str] = {}
    for path in REQUIREMENTS:
        if not path.exists():
            continue
        for name, version in PIN_RE.findall(path.read_text(encoding="utf-8")):
            key = name.lower().replace("_", "-")
            if key in pins and pins[key] != version:
                pytest.fail(f"`{name}` bị ghim hai mức khác nhau: {pins[key]} và {version}")
            pins[key] = version
    return pins


def installed_version(package: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(package)
    except PackageNotFoundError:
        return None


@pytest.mark.parametrize("package", CRITICAL)
def test_phien_ban_dang_cai_khop_muc_ghim(package: str):
    pins = pinned_versions()
    expected = pins.get(package)
    if expected is None:
        pytest.skip(f"{package} không được ghim ở file requirements nào")
    actual = installed_version(package)
    if actual is None:
        pytest.skip(f"{package} chưa cài (bình thường nếu chưa cài requirements-screen.txt)")
    assert actual == expected, (
        f"{package}: đang dùng {actual} nhưng requirements ghim {expected}.\n"
        f"Chạy: .venv/Scripts/python.exe -m pip install -r requirements-data.txt"
    )


def test_numpy_duoi_2_vi_scaper_van_dung_np_Inf():
    """scaper 1.6.5 là bản MỚI NHẤT và vẫn dùng np.Inf, đã bị gỡ ở numpy 2.0.

    Trên numpy 2.x mọi lệnh sinh soundscape chết. Không có bản scaper nào sửa, nên
    đây là ràng buộc cứng chứ không phải sở thích.
    """
    numpy = pytest.importorskip("numpy")
    major = int(numpy.__version__.split(".")[0])
    assert major < 2, (
        f"numpy {numpy.__version__} sẽ làm chết scaper ở np.Inf.\n"
        f"Thủ phạm thường gặp: cài panns-inference kéo theo matplotlib mới, "
        f"matplotlib đòi numpy>=2.0 và pip lặng lẽ gỡ numpy 1.26."
    )


def test_moi_file_requirements_deu_ghim_chinh_xac():
    """Không chấp nhận `>=` hay để trống: dataset phải tái lập được, kể cả 6 tháng sau."""
    long = re.compile(r"^([A-Za-z0-9_.\-]+)\s*(>=|<=|>|<|~=)", re.MULTILINE)
    for path in REQUIREMENTS:
        if not path.exists():
            continue
        loose = long.findall(path.read_text(encoding="utf-8"))
        assert not loose, f"{path.name} ghim lỏng: {loose} — dùng `==`"


def test_numpy_duoc_ghim_cung_mot_muc_o_moi_file():
    # Hai file khai hai mức thì pip chọn một, và lựa chọn đó phụ thuộc THỨ TỰ cài.
    pins = pinned_versions()          # pinned_versions() tự fail nếu lệch
    assert "numpy" in pins
