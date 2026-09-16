"""Client gọi service `inference` — mọi thứ cần model đều nằm bên đó.

Nhờ tách như vậy, image `api` không cần torch (nhẹ đi ~2 GB) và model chỉ nạp một lần ở
một chỗ. Cái giá là một chặng HTTP nội bộ — chấp nhận được vì cả hai container cùng node.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.config import get_settings

# Suy luận CPU cho clip 10 giây có thể mất vài giây; timeout mặc định 5s của httpx sẽ
# cắt giữa chừng và biến một request chậm thành một lỗi khó hiểu.
INFER_TIMEOUT_SEC = 120.0
EMBED_TIMEOUT_SEC = 60.0


@dataclass(frozen=True)
class InferenceResult:
    detections: list[dict]
    caption_en: str
    caption_vi: str
    duration: float
    model_versions: dict[str, str]


class InferenceClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or get_settings().inference_url).rstrip("/")

    async def infer(self, audio_bytes: bytes, filename: str) -> InferenceResult:
        async with httpx.AsyncClient(timeout=INFER_TIMEOUT_SEC) as client:
            response = await client.post(
                f"{self.base_url}/infer",
                files={"file": (filename, audio_bytes, "audio/wav")},
            )
            response.raise_for_status()
            payload = response.json()
        return InferenceResult(
            detections=payload["detections"],
            caption_en=payload["caption_en"],
            caption_vi=payload["caption_vi"],
            duration=payload["duration"],
            model_versions=payload["model_versions"],
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=EMBED_TIMEOUT_SEC) as client:
            response = await client.post(f"{self.base_url}/embed", json={"texts": texts})
            response.raise_for_status()
            return response.json()["vectors"]

    async def healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except httpx.HTTPError:
            return False
