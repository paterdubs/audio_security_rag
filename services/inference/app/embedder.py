"""Nhúng văn bản bằng BGE-M3 — SYSTEM.md §7.1, §7.4.

Đặt embedder ở service `inference` (không phải `api`) để mọi thứ cần torch nằm gọn trong
một image: `api` nhẹ đi ~2 GB và model chỉ nạp một lần ở một chỗ.

⚠️ Nhúng `caption_vi`, KHÔNG nhúng `caption_en`. Người dùng hỏi bằng tiếng Việt; nhúng
caption tiếng Anh rồi truy vấn tiếng Việt làm retrieval sụt nghiêm trọng (§7.4).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
EXPECTED_DIM = int(os.environ.get("EMBED_DIM", "1024"))

_model = None


def load_model() -> None:
    """Nạp model một lần lúc khởi động (warmup), không nạp lười ở request đầu.

    Nạp lười làm request đầu tiên mất hàng chục giây và thường bị timeout — tệ hơn nữa,
    healthcheck sẽ báo xanh trước khi service thực sự phục vụ được.
    """
    global _model
    if _model is not None:
        return
    from sentence_transformers import SentenceTransformer

    logger.info("nap embedder %s ...", EMBED_MODEL)
    _model = SentenceTransformer(EMBED_MODEL, device="cpu")

    dim = int(_model.get_sentence_embedding_dimension())
    if dim != EXPECTED_DIM:
        # Sai số chiều là lỗi KHÔNG được bỏ qua: cột VECTOR(n) trong DB cố định, ghi
        # vector sai chiều sẽ lỗi ở tận lúc INSERT, xa chỗ gây ra lỗi.
        raise RuntimeError(
            f"{EMBED_MODEL} cho {dim} chiều nhưng EMBED_DIM={EXPECTED_DIM}. "
            "Đổi model khác số chiều thì phải viết migration mới cho cột embedding."
        )
    logger.info("embedder san sang: %d chieu", dim)


def is_ready() -> bool:
    return _model is not None


def embed(texts: list[str]) -> list[list[float]]:
    if _model is None:
        raise RuntimeError("Embedder chưa nạp xong")
    # normalize_embeddings=True để cosine distance trong pgvector so được trực tiếp;
    # không chuẩn hoá thì `<=>` vẫn chạy nhưng khoảng cách lệch theo độ dài vector.
    vectors = _model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [[float(x) for x in vector] for vector in vectors]
