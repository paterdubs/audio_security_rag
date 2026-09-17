"""Bảo đảm dữ liệu PANNs có sẵn TRƯỚC khi import panns_inference.

Vì sao cần file này: `panns_inference/config.py` đọc `~/panns_data/class_labels_indices.csv`
ngay lúc **import module**, và nếu thiếu thì nó gọi `os.system("wget ...")`. Trong image
slim không có `wget`, nên lệnh đó im lặng không làm gì rồi `open()` chết với
FileNotFoundError — thông báo không hề nhắc tới wget, và người đọc đi tìm nhầm chỗ.
(Đúng lỗi này đã gặp trên Windows ở scripts/auto_screen.py, cùng nguyên nhân.)

Tải bằng urllib thay vì wget: chạy được trên mọi nền, và kiểm được kích thước thật.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# panns_inference hardcode ~/panns_data — không đọc biến môi trường nào. Dockerfile
# symlink chỗ đó sang /models/panns để rơi vào named volume.
PANNS_DIR = Path.home() / "panns_data"

LABELS_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
LABELS_NAME = "class_labels_indices.csv"

CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
CHECKPOINT_NAME = "Cnn14_mAP=0.431.pth"
CHECKPOINT_BYTES = 327428481   # đã đo trên bản đã dùng cho toàn bộ bước sàng lọc dữ liệu


def _download(url: str, destination: Path, expected_bytes: int | None = None) -> None:
    """Tải qua file .part rồi mới đổi tên: tiến trình chết giữa chừng sẽ KHÔNG để lại
    một file trông như đã tải xong — đúng loại lỗi im lặng đã làm hỏng 2.4 GB FSD50K."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    logger.info("tai %s ...", destination.name)
    with urllib.request.urlopen(url, timeout=300) as response, part.open("wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)

    actual = part.stat().st_size
    if expected_bytes is not None and actual != expected_bytes:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{destination.name}: tai ve {actual} byte, mong doi {expected_bytes}")
    part.replace(destination)
    logger.info("xong %s (%d byte)", destination.name, actual)


def ensure_panns_data() -> Path:
    """Bảo đảm CSV nhãn + checkpoint có mặt. Trả về đường dẫn checkpoint."""
    labels = PANNS_DIR / LABELS_NAME
    checkpoint = PANNS_DIR / CHECKPOINT_NAME

    if not labels.exists():
        _download(LABELS_URL, labels)

    if not (checkpoint.exists() and checkpoint.stat().st_size == CHECKPOINT_BYTES):
        _download(CHECKPOINT_URL, checkpoint, CHECKPOINT_BYTES)
        # Tác giả PANNs KHÔNG công bố checksum. Tự ghi lại để lần sau còn đối chiếu —
        # không có nó thì không biết mình đang chạy đúng bộ trọng số của lần trước.
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        checkpoint.with_suffix(".pth.sha256").write_text(digest, encoding="utf-8")
        logger.info("SHA-256 checkpoint: %s", digest[:16])

    return checkpoint
