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
import json
import shutil
import sys
import tarfile
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


def download(url: str, dest: Path, expected_size: int = 0) -> None:
    """Tải có hỗ trợ tải tiếp. File .part chỉ được đổi tên khi đã tải xong."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0

    if dest.exists() and (not expected_size or dest.stat().st_size == expected_size):
        print(f"    đã có, bỏ qua: {dest.name}")
        return

    request = urllib.request.Request(url)
    if done:
        request.add_header("Range", f"bytes={done}-")
        print(f"    tải tiếp từ {done / 1e6:.0f} MB")

    with urllib.request.urlopen(request, timeout=120) as response:
        total = done + int(response.headers.get("Content-Length", 0))
        with part.open("ab" if done else "wb") as handle:
            done = _stream(response, handle, done, total, dest.name)

    part.replace(dest)
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
    for filename, info in resolve_targets(spec):
        archive = ARCHIVE_DIR / name / filename
        download(info["url"], archive, info["size"])
        if not verify_checksum(archive, info["md5"]):
            return False
        if archive.suffix in (".zip", ".gz", ".tgz") and not filename.endswith((".z01", ".z02", ".z03", ".z04", ".z05")):
            extract(archive, dest)
            if not keep_archive:
                archive.unlink()
                print("    đã xoá file nén (cleanup_policy)")
    return True


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
