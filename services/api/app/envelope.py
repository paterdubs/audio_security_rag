"""Envelope thống nhất cho mọi response — SYSTEM.md §4.4.

    {"success": bool, "data": ..., "error": ..., "meta": ...}

Lý do bắt buộc envelope thay vì trả thẳng object: frontend chỉ cần MỘT chỗ xử lý lỗi,
và `meta` là nơi phân trang sống mà không phải nhét vào thân dữ liệu. Trả lẫn lộn lúc có
lúc không là thứ khiến client phải đoán.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ErrorBody(BaseModel):
    code: str
    message: str


class Envelope(BaseModel, Generic[T]):
    success: bool
    data: T | None = None
    error: ErrorBody | None = None
    meta: dict[str, Any] | None = None


def ok(data: Any, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"success": True, "data": data, "error": None, "meta": meta}


def fail(code: str, message: str, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Lỗi cũng đi qua envelope: HTTP status nói chuyện tầng giao vận, `code` nói
    chuyện tầng nghiệp vụ. Client bắt `code`, không parse chuỗi message."""
    return {"success": False, "data": None, "error": {"code": code, "message": message}, "meta": meta}
