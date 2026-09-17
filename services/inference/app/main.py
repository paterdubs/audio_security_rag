"""Service `inference` — mọi thứ cần model nằm ở đây (SYSTEM.md §4.2).

Walking skeleton W1: PANNs CNN14 (SED tạm) + caption template (baseline B0) + BGE-M3
(embedder cho RAG). Model đề xuất thật là W4–W5.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app import embedder, panns_model
from app.captioner import caption_en, caption_vi

logger = logging.getLogger(__name__)

MODEL_VERSIONS = {
    # Hậu tố `-w1baseline` nói thẳng đây là bản tạm, để không ai đọc nhầm số của nó
    # thành số của đóng góp nghiên cứu. Model thật đổi giá trị này khi W4-W5 xong.
    #
    # Phần `mAP0.431` KHÔNG thừa: chuỗi này được ghi vào cột model_versions của
    # security_events, nên nó phải định danh được BỘ TRỌNG SỐ đã sinh ra sự kiện.
    # "placeholder" trần thì sáu tháng sau không ai biết hàng nghìn sự kiện trong DB
    # do checkpoint nào tạo ra — đúng loại thông tin không thể dựng lại về sau.
    "sed": "panns-cnn14-mAP0.431-w1baseline",
    "aac": "template-b0",
    "embed": os.environ.get("EMBED_MODEL", "BAAI/bge-m3"),
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    # Nạp model TRƯỚC khi nhận request: healthcheck chỉ xanh sau khi xong, nên
    # `depends_on: service_healthy` của api chờ đúng thời điểm.
    panns_model.load_model()
    embedder.load_model()
    logger.info("inference san sang")
    yield


app = FastAPI(title="Audio Security Inference", version="0.1.0", lifespan=lifespan)


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)


@app.get("/health")
async def health() -> dict:
    ready = panns_model.is_ready() and embedder.is_ready()
    if not ready:
        # 503 chứ không phải 200: healthcheck của Docker phải THẤY được là chưa sẵn sàng.
        raise HTTPException(status_code=503, detail="Model chua nap xong")
    return {"status": "healthy", "models": MODEL_VERSIONS}


@app.post("/infer")
async def infer(file: UploadFile = File(...)) -> dict:
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="File rong")

    try:
        detections, duration = panns_model.detect(audio_bytes)
    except Exception as error:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Khong doc duoc audio: {type(error).__name__}") from error

    payload = [
        {"class_id": d.class_id, "onset": d.onset, "offset": d.offset, "confidence": d.confidence}
        for d in detections
    ]
    return {
        "detections": payload,
        "caption_en": caption_en(payload),
        "caption_vi": caption_vi(payload),
        "duration": round(duration, 3),
        "model_versions": MODEL_VERSIONS,
    }


@app.post("/embed")
async def embed(request: EmbedRequest) -> dict:
    return {"vectors": embedder.embed(request.texts), "model": MODEL_VERSIONS["embed"]}
