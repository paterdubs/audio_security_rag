"""WebSocket cảnh báo — đường NÓNG (SYSTEM.md §4.1, §6.5).

Nguyên tắc kiến trúc số một: cảnh báo tức thời đi THẲNG từ risk scoring ra WebSocket,
**không đi qua LLM**. Dùng RAG để phát cảnh báo thì vừa chậm (LLM latency) vừa không tin
cậy (LLM có thể bịa).

Bản W1 phát trong tiến trình. W7 đổi sang Redis pub/sub để nhiều worker cùng phát được.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter(tags=["alerts"])

_clients: set[WebSocket] = set()
_lock = asyncio.Lock()


async def broadcast_event(payload: dict) -> int:
    """Đẩy cảnh báo tới mọi client đang mở. Trả về số client nhận được.

    Client chết KHÔNG được làm hỏng luồng ingest: gỡ nó ra và đi tiếp. Một dashboard bị
    đóng tab không có lý do gì làm mất sự kiện đang ghi.
    """
    async with _lock:
        targets = list(_clients)

    delivered = 0
    for websocket in targets:
        try:
            await websocket.send_text(json.dumps(payload, ensure_ascii=False))
            delivered += 1
        except Exception:  # noqa: BLE001
            async with _lock:
                _clients.discard(websocket)
    return delivered


@router.websocket("/alerts/stream")
async def alerts_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    async with _lock:
        _clients.add(websocket)
    logger.info("alert client ket noi, tong=%d", len(_clients))
    try:
        while True:
            # Giữ kết nối sống; client không cần gửi gì, nhưng đọc để phát hiện ngắt kết nối.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        async with _lock:
            _clients.discard(websocket)
