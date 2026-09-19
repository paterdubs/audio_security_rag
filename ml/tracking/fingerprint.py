"""Vân tay dữ liệu / mã nguồn / môi trường — nền của Pha 1 (TRAINING_OPS_PLAN §4).

Yêu cầu R2 nói rõ: **đếm clip hoặc hash nhãn là KHÔNG đủ**. Lô 17/09 và lô B0–B9 có cùng
số clip (7.920) và cùng 15 lớp; thứ phân biệt chúng nằm trong audio và trong phân bố độ
dài sự kiện. Vì vậy vân tay ở đây gồm ba tầng độc lập:

    nhan_sha256        — clip_id + (lớp, onset, offset) đã chuẩn hoá
    waveform_sha256    — băm TOÀN BỘ memmap .npy, không phải kích thước file
    thong_ke_do_dai    — p05/p50/p95/max theo lớp

Tầng ba là tầng duy nhất con người đọc được. Chính nó sẽ bắt ngay một lỗi kiểu
`event_duration=("const", 1.0)`: max của mọi lớp kẹt ở 1,10 s là thứ đập vào mắt trong
manifest, trong khi hai tầng băm chỉ nói "khác lần trước" mà không nói khác chỗ nào.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

# Băm theo khối: file waveform train hiện là 5,07 GB, đọc nguyên vào RAM là tự chuốc
# MemoryError trên máy đang train.
CHUNK_BYTES = 8 * 1024 * 1024

# YAML dài nhất (`ontology_map.yaml`) là 16 KB — nhúng nguyên văn vẫn rẻ. Trần này để
# một file config phình to bất ngờ không biến manifest thành file vài megabyte.
MAX_CONFIG_BYTES = 256 * 1024

DURATION_PERCENTILES = (5, 25, 50, 75, 95)


def bam_file(path: Path, chunk_bytes: int = CHUNK_BYTES) -> str:
    """SHA-256 của nội dung file, đọc theo khối."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def bam_van_ban(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chuan_hoa_nhan(clips: list[dict]) -> str:
    """Chuỗi chuẩn tắc của nhãn — thứ tự cố định, số làm tròn 6 chữ số.

    Làm tròn CỐ Ý: onset/offset đi qua JSON rồi qua float nên chữ số cuối có thể nhảy
    giữa hai lần đọc; không làm tròn thì vân tay đổi mà dữ liệu thì không.
    """
    dong = []
    for clip in clips:
        events = sorted((int(c), round(float(on), 6), round(float(off), 6))
                        for c, on, off in clip["events"])
        dong.append(clip["clip_id"] + "|" + ";".join(f"{c},{on:.6f},{off:.6f}" for c, on, off in events))
    return "\n".join(sorted(dong))


def thong_ke_do_dai(clips: list[dict], class_ids: list[str]) -> dict[str, dict]:
    """Số sự kiện và phân vị độ dài theo lớp — tầng người đọc được của vân tay."""
    gom: list[list[float]] = [[] for _ in class_ids]
    for clip in clips:
        for class_idx, onset, offset in clip["events"]:
            gom[int(class_idx)].append(float(offset) - float(onset))

    ket_qua: dict[str, dict] = {}
    for class_id, values in zip(class_ids, gom, strict=False):
        if not values:
            ket_qua[class_id] = {"n_events": 0}
            continue
        array = np.asarray(values, dtype=np.float64)
        percentiles = np.percentile(array, DURATION_PERCENTILES)
        ket_qua[class_id] = {
            "n_events": len(values),
            **{f"p{p:02d}": round(float(v), 4) for p, v in zip(DURATION_PERCENTILES, percentiles, strict=False)},
            "max": round(float(array.max()), 4),
            "tong_giay": round(float(array.sum()), 2),
        }
    return ket_qua


def van_tay_du_lieu(features_dir: Path, split: str, sample_rate: int = 32000,
                    bam_waveform: bool = True) -> dict:
    """Vân tay của một split đã precompute.

    `bam_waveform=False` chỉ dùng khi đang test hoặc khi đã biết chắc file không đổi —
    bỏ băm waveform thì vân tay KHÔNG còn phát hiện được việc audio bị sinh lại, nên
    manifest ghi rõ `waveform_sha256: null` thay vì im lặng.
    """
    meta_path = features_dir / f"{split}_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    class_ids: list[str] = meta["class_ids"]
    clips: list[dict] = meta["clips"]
    wave_path = features_dir / f"{split}_wave{sample_rate // 1000}k.npy"

    return {
        "split": split,
        "n_clips": len(clips),
        "class_ids": class_ids,
        "duration_sec": float(meta["duration"]),
        "sample_rate": int(meta["sample_rate"]),
        "nhan_sha256": bam_van_ban(_chuan_hoa_nhan(clips)),
        "meta_sha256": bam_file(meta_path),
        "waveform_file": wave_path.name,
        "waveform_bytes": wave_path.stat().st_size if wave_path.exists() else None,
        "waveform_sha256": bam_file(wave_path) if (bam_waveform and wave_path.exists()) else None,
        "thong_ke_do_dai": thong_ke_do_dai(clips, class_ids),
    }


def van_tay_chia_tap(train_idx: np.ndarray, val_idx: np.ndarray) -> dict:
    """Vân tay của phép chia train/val — khẳng định tập val không đổi giữa các run."""
    return {
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "train_sha256": bam_van_ban(",".join(map(str, sorted(int(i) for i in train_idx)))),
        "val_sha256": bam_van_ban(",".join(map(str, sorted(int(i) for i in val_idx)))),
    }


def van_tay_ma(repo_root: Path, thu_muc: tuple[str, ...] = ("ml", "scripts")) -> dict:
    """Băm nội dung mọi file .py trong `ml/` và `scripts/`.

    Không dùng git SHA thay thế: worktree của dự án thường dirty lúc train (các run
    v1/v2/v3 đều thế), và một SHA sạch gắn lên mã đã sửa là lời khai sai.
    """
    tung_file: dict[str, str] = {}
    for ten in thu_muc:
        for path in sorted((repo_root / ten).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            tung_file[path.relative_to(repo_root).as_posix()] = bam_file(path)
    tong = bam_van_ban("\n".join(f"{k} {v}" for k, v in sorted(tung_file.items())))
    return {"n_files": len(tung_file), "tree_sha256": tong, "files": tung_file}


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=repo_root, capture_output=True,
                             text=True, timeout=30, encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def trang_thai_git(repo_root: Path) -> dict:
    """SHA + tình trạng dirty + băm của chính phần diff chưa commit.

    Chỉ lưu SHA là không đủ tái lập: rủi ro "Git có nhưng worktree dirty" được ghi thành
    một dòng riêng trong TRAINING_OPS_PLAN §5 vì nó đã xảy ra với cả ba run đầu.
    """
    sha = _git(repo_root, "rev-parse", "HEAD")
    status = _git(repo_root, "status", "--porcelain")
    diff = _git(repo_root, "diff", "HEAD")
    if sha is None:
        return {"co_git": False}
    return {
        "co_git": True,
        "sha": sha.strip(),
        "dirty": bool(status and status.strip()),
        "dirty_files": sorted(line[3:] for line in (status or "").splitlines() if line.strip()),
        "diff_sha256": bam_van_ban(diff) if diff else None,
        "diff_bytes": len(diff.encode("utf-8")) if diff else 0,
    }


def noi_dung_config(paths: list[Path], repo_root: Path) -> dict[str, dict]:
    """Nhúng NỘI DUNG file config, không phải đường dẫn.

    Đường dẫn không tái lập được gì: `scaper_train.yaml` đã đổi bốn lần trong hai ngày
    (B2, B3, B5, B6) mà tên file thì không đổi lần nào.
    """
    ket_qua: dict[str, dict] = {}
    for path in paths:
        key = path.relative_to(repo_root).as_posix() if path.is_relative_to(repo_root) else path.name
        if not path.exists():
            ket_qua[key] = {"ton_tai": False}
            continue
        size = path.stat().st_size
        ghi = {"ton_tai": True, "bytes": size, "sha256": bam_file(path)}
        if size <= MAX_CONFIG_BYTES:
            ghi["noi_dung"] = path.read_text(encoding="utf-8")
        ket_qua[key] = ghi
    return ket_qua


def moi_truong(pip_freeze: bool = True) -> dict:
    """Phiên bản thư viện, GPU và (tuỳ chọn) toàn bộ `pip freeze`."""
    import torch

    thong_tin = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if pip_freeze:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                             text=True, timeout=120, encoding="utf-8", errors="replace")
        thong_tin["pip_freeze"] = sorted(out.stdout.split()) if out.returncode == 0 else None
    return thong_tin
