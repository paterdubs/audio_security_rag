"""Tải các dataset công khai theo đăng ký trong ml/configs/sources.yaml.

    python scripts/download_sources.py --list
    python scripts/download_sources.py --verify-sizes
    python scripts/download_sources.py --source esc50
    python scripts/download_sources.py --priority 1

Ba điều script này làm mà tải tay không làm được:
  1. CHẶN TRƯỚC khi ổ đĩa không đủ chỗ, thay vì tải 5 tiếng rồi chết ở phút cuối.
  2. Đối chiếu checksum MD5 do Zenodo công bố — file tải hỏng là lỗi âm thầm,
     phải bắt ngay ở đây chứ không phải lúc train.
  3. Tải tiếp được chỗ dở dang (HTTP Range) — mạng đứt giữa file 6 GB là chuyện thường.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import yaml

from common import REPO_ROOT, enable_utf8_output

CONFIG_PATH = REPO_ROOT / "ml" / "configs" / "sources.yaml"
RAW_DIR = REPO_ROOT / "data" / "raw"
ARCHIVE_DIR = RAW_DIR / "_archives"
ZENODO_API = "https://zenodo.org/api/records/{record}"
CHUNK = 1 << 20  # 1 MB
DISK_SAFETY_MARGIN_GB = 2.0


# ── Tiện ích ─────────────────────────────────────────────────────────────────


def free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def http_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def zenodo_files(record: int) -> dict[str, dict]:
    """{tên file: {url, size, md5}} lấy trực tiếp từ API Zenodo."""
    data = http_json(ZENODO_API.format(record=record))
    return {
        f["key"]: {
            "url": f["links"]["self"],
            "size": f.get("size", 0),
            "md5": f.get("checksum", "").removeprefix("md5:"),
        }
        for f in data.get("files", [])
    }


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


# ── Tải ──────────────────────────────────────────────────────────────────────


def finalize(part: Path, dest: Path, attempts: int = 6) -> None:
    """Đổi tên .part → tên thật, có thử lại.

    Windows hay khoá file ngay sau khi ghi xong một file lớn (trình quét virus mở nó
    để kiểm tra), làm os.replace ném WinError 32. Đây là lỗi TẠM THỜI: nội dung đã
    tải đủ và đúng. Không thử lại thì mất công tải lại vài GB chỉ vì một cú đổi tên.
    """
    for attempt in range(attempts):
        try:
            part.replace(dest)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            delay = 2 ** attempt
            print(f"    file đang bị khoá, thử lại sau {delay}s…", flush=True)
            time.sleep(delay)


class DownloadLock:
    """Một người ghi cho mỗi file. Vào được thì giữ, không vào được thì báo rõ ai giữ.

    Chạy hai lệnh tải cùng một nguồn là chuyện rất dễ xảy ra — mở phiên mới rồi
    "nối lại" lệnh tải mà lệnh cũ vẫn còn sống. Hai tiến trình cùng nối thêm vào một
    .part cho ra file thừa byte, và mỗi tiến trình đều thấy bộ đếm của mình hoàn
    toàn đúng. Chỉ MD5 bắt được, sau khi đã tốn hàng giờ băng thông.
    """

    def __init__(self, part: Path) -> None:
        self.path = part.with_suffix(part.suffix + ".lock")

    def __enter__(self) -> "DownloadLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # O_EXCL: tạo được thì thôi, đã có thì ném FileExistsError. Nguyên tử,
            # nên không có khe hở giữa "kiểm tra tồn tại" và "tạo".
            handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            owner = self.path.read_text(encoding="utf-8").strip() or "?"
            raise SystemExit(
                f"❌ {self.path.stem} đang được tiến trình khác tải (PID {owner}).\n"
                f"   Chạy song song hai lệnh tải cùng file sẽ làm hỏng file.\n"
                f"   Nếu tiến trình đó đã chết: xoá {self.path.name} rồi chạy lại."
            ) from None
        os.write(handle, str(os.getpid()).encode())
        os.close(handle)
        return self

    def __exit__(self, *exc) -> None:
        self.path.unlink(missing_ok=True)


# Lỗi máy chủ tạm thời, KHÔNG phải lỗi của file: thử lại được.
# 504 của Zenodo đã làm chết một lệnh tải 6 GB đang chạy dở (14/09/2026).
TRANSIENT_HTTP = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 8


def is_transient(error: Exception) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code in TRANSIENT_HTTP
    return isinstance(error, (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead))


def fetch_one_pass(url: str, part: Path, done: int, name: str) -> int:
    """Một lượt tải, trả số byte đã có sau lượt đó. Ném lỗi để nơi gọi quyết định thử lại."""
    request = urllib.request.Request(url)
    if done:
        request.add_header("Range", f"bytes={done}-")
        print(f"    tải tiếp từ {done / 1e6:.0f} MB")

    with urllib.request.urlopen(request, timeout=120) as response:
        # BẮT BUỘC kiểm tra máy chủ có CHẤP NHẬN Range không. Gửi Range mà nhận 200
        # nghĩa là máy chủ phớt lờ và trả về TOÀN BỘ file; nối thêm vào .part sẽ tạo
        # ra một file gồm hai bản chồng nhau — hỏng âm thầm, chỉ lộ ra ở bước MD5
        # sau khi đã tải thừa hàng GB. Đã xảy ra thật với Zenodo (14/09/2026).
        resumed = done > 0 and response.status == 206
        if done and not resumed:
            print(f"    máy chủ không nhận Range (HTTP {response.status}) → tải lại từ đầu")
            done = 0

        total = done + int(response.headers.get("Content-Length", 0))
        with part.open("ab" if resumed else "wb") as handle:
            return _stream(response, handle, done, total, name)


def fetch_with_retry(url: str, part: Path, done: int, name: str) -> int:
    """Tải, thử lại khi máy chủ lỗi tạm thời hoặc mạng đứt.

    Mỗi lần thử lại đọc lại kích thước THẬT của .part rồi tải tiếp từ đó, nên không
    lượt nào phải làm lại từ đầu. Đây là điểm khác biệt so với chạy lại cả lệnh: một
    cú 504 ở phút thứ 40 của file 6 GB chỉ mất vài giây, không mất 40 phút.
    """
    for attempt in range(MAX_ATTEMPTS):
        try:
            return fetch_one_pass(url, part, done, name)
        except Exception as error:                       # noqa: BLE001 — phân loại ngay bên dưới
            if not is_transient(error) or attempt == MAX_ATTEMPTS - 1:
                raise
            done = part.stat().st_size if part.exists() else 0
            delay = min(60, 2 ** attempt)
            print(f"    ⚠️ {type(error).__name__}: {error} → thử lại sau {delay}s "
                  f"(lần {attempt + 2}/{MAX_ATTEMPTS}, đang có {done / 1e6:.0f} MB)", flush=True)
            time.sleep(delay)
    raise OSError(f"{name}: hết {MAX_ATTEMPTS} lần thử")


def download(url: str, dest: Path, expected_size: int = 0) -> None:
    """Tải có hỗ trợ tải tiếp. File .part chỉ được đổi tên khi đã tải xong."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    with DownloadLock(part):
        _download_locked(url, dest, part, expected_size)


def _download_locked(url: str, dest: Path, part: Path, expected_size: int) -> None:
    done = part.stat().st_size if part.exists() else 0

    if dest.exists() and (not expected_size or dest.stat().st_size == expected_size):
        print(f"    đã có, bỏ qua: {dest.name}")
        return

    # .part đã đủ kích thước: chỉ còn thiếu bước đổi tên (lần chạy trước chết ở đó).
    # Không có nhánh này thì ta sẽ gửi Range bắt đầu từ cuối file và nhận về 416.
    if expected_size and done == expected_size:
        print(f"    đã tải đủ từ lần trước, chỉ cần đổi tên: {dest.name}")
        finalize(part, dest)
        return

    done = fetch_with_retry(url, part, done, dest.name)

    # Kiểm tra KÍCH THƯỚC THẬT TRÊN ĐĨA, không phải bộ đếm của chính mình. Hai con số
    # này lệch nhau khi có tiến trình khác cùng ghi vào .part: bộ đếm nội bộ vẫn khớp
    # hoàn hảo với Zenodo trong khi file đã thừa 200 MB. Đã xảy ra thật (14/09/2026,
    # FSD50K) — bộ đếm báo "xong 2.31 GB" đúng từng byte, file trên đĩa 2.51 GB.
    actual = part.stat().st_size
    if actual != done:
        raise OSError(
            f"{dest.name}: bộ đếm nói {done} byte nhưng file có {actual} byte.\n"
            f"    Gần như chắc chắn có tiến trình tải khác cùng ghi vào {part.name}.\n"
            f"    Xoá file .part đó rồi tải lại — nội dung hiện tại không cứu được."
        )
    if expected_size and done != expected_size:
        raise OSError(f"{dest.name}: tải về {done} byte, Zenodo công bố {expected_size} byte")

    finalize(part, dest)
    print(f"    xong: {dest.name} ({done / 1e9:.2f} GB)")


def _stream(response, handle, done: int, total: int, name: str) -> int:
    last_report = done
    while chunk := response.read(CHUNK):
        handle.write(chunk)
        done += len(chunk)
        if done - last_report >= 50 * CHUNK:
            pct = f"{100 * done / total:.0f}%" if total else "?"
            print(f"    {name}: {done / 1e9:.2f} GB ({pct})", flush=True)
            last_report = done
    return done


def verify_checksum(path: Path, expected_md5: str) -> bool:
    if not expected_md5:
        print("    ⚠️ nguồn không công bố checksum — bỏ qua kiểm tra")
        return True
    print("    đang tính MD5…", flush=True)
    actual = md5_of(path)
    if actual == expected_md5:
        print("    ✓ checksum khớp")
        return True
    print(f"    ✗ CHECKSUM SAI: nhận {actual}, mong đợi {expected_md5}")
    return False


# ── Giải nén ─────────────────────────────────────────────────────────────────

# Nơi winget đặt Info-ZIP trên Windows. Bộ cài KHÔNG thêm vào PATH, nên chỉ dựa vào
# shutil.which() sẽ báo "chưa cài" cho một máy đã cài xong — người dùng đứng trước
# một thông báo bảo họ làm lại đúng việc vừa làm.
ZIP_FALLBACKS = [
    Path(r"C:\Program Files (x86)\GnuWin32\bin\zip.exe"),
    Path(r"C:\Program Files\GnuWin32\bin\zip.exe"),
]


def find_zip_tool() -> str:
    """Đường dẫn tới `zip` (Info-ZIP), tìm trên PATH trước rồi tới nơi cài mặc định."""
    found = shutil.which("zip")
    if found:
        return found
    for candidate in ZIP_FALLBACKS:
        if candidate.exists():
            return str(candidate)
    raise SystemExit(
        "❌ cần công cụ `zip` (Info-ZIP) để ghép kho zip chia nhiều phần.\n"
        "   Windows : winget install --id GnuWin32.Zip   (bấm Yes ở cửa sổ UAC)\n"
        "   Debian  : sudo apt install zip\n"
        "   macOS   : đã có sẵn"
    )


def join_split_zip(archive: Path) -> Path:
    """Ghép kho zip chia nhiều phần (.z01….zNN + .zip) thành một file đọc được.

    zipfile của Python không đọc được kho chia phần — nó ném thẳng
    `BadZipFile: zipfiles that span multiple disks are not supported`. Nối tay các
    phần cũng không đủ: bản ghi thư mục trung tâm mang số hiệu đĩa cần sửa lại.
    `zip -s 0` làm đúng việc đó và là cách chính FSD50K hướng dẫn.
    """
    joined = archive.with_name(archive.stem + ".joined.zip")
    if joined.exists():
        print(f"    đã ghép sẵn: {joined.name}")
        return joined

    tool = find_zip_tool()
    print(f"    ghép {archive.name} + các phần .z01… → {joined.name}")
    result = subprocess.run([tool, "-s", "0", str(archive), "--out", str(joined)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        joined.unlink(missing_ok=True)     # đừng để lại file ghép dở, lần sau sẽ tin nhầm
        raise SystemExit(f"❌ ghép thất bại: {result.stderr.strip()[:300]}")
    return joined


def extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"    giải nén → {dest.relative_to(REPO_ROOT)}")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest, filter="data")
    else:
        raise ValueError(f"Không biết cách giải nén {archive.name}")


# ── Điều phối một nguồn ──────────────────────────────────────────────────────


def check_disk(spec: dict, name: str) -> bool:
    needed = spec.get("peak_disk_gb") or spec.get("approx_size_gb", 0) * 2
    available = free_gb(RAW_DIR)
    if available - DISK_SAFETY_MARGIN_GB >= needed:
        return True
    print(f"  ⛔ {name}: cần ~{needed:.1f} GB (nén + giải nén), chỉ còn {available:.1f} GB")
    print("     Dọn chỗ hoặc xoá data/raw/_archives/ rồi chạy lại.")
    return False


def resolve_targets(spec: dict) -> list[tuple[str, dict]]:
    """Trả về [(tên file, {url, size, md5})] cho một nguồn."""
    if "zenodo_record" in spec:
        catalog = zenodo_files(spec["zenodo_record"])
        missing = [f for f in spec["files"] if f not in catalog]
        if missing:
            raise KeyError(f"Zenodo record không có file: {missing}")
        return [(f, catalog[f]) for f in spec["files"]]
    return [(spec["archive_name"], {"url": spec["url"], "size": 0, "md5": ""})]


def fetch_source(name: str, spec: dict, keep_archive: bool) -> bool:
    access = spec.get("access", "")
    if access == "application_required":
        print(f"  ⏭  {name}: cần nộp đơn — {spec.get('url', '')}")
        return True
    if access == "youtube_scrape":
        print(f"  ⏭  {name}: cần yt-dlp, xử lý bằng script riêng (chưa viết)")
        return True
    if spec.get("status") == "BLOCKED":
        print(f"  ⛔ {name}: đang bị chặn, xem `notes` trong sources.yaml")
        return False
    if not check_disk(spec, name):
        return False

    print(f"\n▶ {name}  ({spec.get('approx_size_gb', '?')} GB, {spec.get('license', '?')})")
    dest = RAW_DIR / name

    # TẢI HẾT rồi mới giải nén. Giải nén ngay sau từng file là sai với kho chia nhiều
    # phần: phần `.zip` (đoạn CUỐI của bộ) lại đứng ĐẦU danh sách, nên lần giải nén
    # đầu tiên nổ ra trước khi 5 phần còn lại kịp tải. Đã xảy ra với FSD50K 14/09.
    archives = []
    for filename, info in resolve_targets(spec):
        archive = ARCHIVE_DIR / name / filename
        download(info["url"], archive, info["size"])
        if not verify_checksum(archive, info["md5"]):
            return False
        archives.append(archive)

    for archive in archives:
        if archive.suffix not in (".zip", ".gz", ".tgz"):
            continue                       # .z01…: là phần của bộ, không giải nén riêng
        source = join_split_zip(archive) if spec.get("split_archive") else archive
        extract(source, dest)
        if not keep_archive:
            freed = cleanup_archive(archive, source)
            print(f"    đã xoá file nén, giải phóng {freed / 1e9:.1f} GB (cleanup_policy)")
    return True


def cleanup_archive(archive: Path, source: Path) -> int:
    """Xoá kho nén và MỌI phần của nó, trả về số byte giải phóng.

    Với kho chia nhiều phần, xoá mỗi file `.zip` là bỏ sót 5 phần `.z0x` cộng file
    ghép — 34 GB nằm lì trên đĩa ở riêng FSD50K, trong khi đỉnh dung lượng của bước
    kế tiếp đã tính trên giả định chúng biến mất.
    """
    doomed = {archive, source, *archive.parent.glob(f"{archive.stem}.z[0-9][0-9]")}
    freed = 0
    for path in doomed:
        if path.exists():
            freed += path.stat().st_size
            path.unlink()
    return freed


# ── Các lệnh ─────────────────────────────────────────────────────────────────


def cmd_list(config: dict) -> int:
    print(f"{'nguồn':<22}{'ưu tiên':<9}{'GB':>7}  {'truy cập':<32}lớp cung cấp")
    print("─" * 110)
    for name, spec in config["sources"].items():
        classes = ", ".join(spec.get("provides_classes", [])) or "—"
        print(f"{name:<22}{spec.get('priority', '?'):<9}{spec.get('approx_size_gb', 0):>7.2f}  "
              f"{spec.get('access', '?'):<32}{classes[:45]}")
    print(f"\nCòn trống: {free_gb(RAW_DIR):.1f} GB")
    return 0


def cmd_verify_sizes(config: dict) -> int:
    """Đối chiếu số liệu trong sources.yaml với Zenodo — bắt trường hợp file ghi sai."""
    problems = 0
    for name, spec in config["sources"].items():
        if "zenodo_record" not in spec:
            continue
        catalog = zenodo_files(spec["zenodo_record"])
        actual = sum(catalog[f]["size"] for f in spec["files"] if f in catalog) / 1e9
        declared = spec.get("approx_size_gb", 0)
        ok = abs(actual - declared) < max(0.05, 0.02 * actual)
        print(f"{'✓' if ok else '✗'} {name:<22} yaml={declared:>6.2f} GB  thực tế={actual:>6.2f} GB")
        problems += 0 if ok else 1
    print(f"\n{problems} nguồn lệch số liệu")
    return 1 if problems else 0


def cmd_download(config: dict, args) -> int:
    selected = {
        name: spec
        for name, spec in config["sources"].items()
        if (args.source and name == args.source)
        or (args.priority and spec.get("priority") == args.priority)
    }
    if not selected:
        print("Không có nguồn nào khớp. Dùng --list để xem danh sách.")
        return 1

    failed = [name for name, spec in selected.items() if not fetch_source(name, spec, args.keep_archives)]
    print(f"\n{'✗' if failed else '✓'} xong — {len(selected) - len(failed)}/{len(selected)} nguồn"
          + (f", thất bại: {failed}" if failed else ""))
    return 1 if failed else 0


def main() -> int:
    enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="liệt kê các nguồn đã đăng ký")
    parser.add_argument("--verify-sizes", action="store_true", help="đối chiếu dung lượng với Zenodo")
    parser.add_argument("--source", help="tải một nguồn theo tên")
    parser.add_argument("--priority", type=int, help="tải mọi nguồn ở mức ưu tiên này")
    parser.add_argument("--keep-archives", action="store_true", help="giữ lại file nén sau khi giải nén")
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.list:
        return cmd_list(config)
    if args.verify_sizes:
        return cmd_verify_sizes(config)
    if args.source or args.priority:
        return cmd_download(config, args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
