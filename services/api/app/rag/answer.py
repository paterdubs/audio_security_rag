"""Sinh câu trả lời CÓ RÀNG BUỘC — SYSTEM.md §7.3.

Hai ràng buộc cứng, áp cho MỌI provider:
  1. Không có bằng chứng ⇒ trả lời "không có sự kiện nào", **không bịa**. Một hệ RAG tốt
     phải biết từ chối (§3.8 dành hẳn 5% câu hỏi không có đáp án để đo đúng điều này).
  2. Mọi câu trả lời kèm `citations[]` là event_id thật. Câu trả lời không citation bị
     coi là **lỗi hệ thống**, không phải câu trả lời hợp lệ (§4.4).

`TemplateProvider` là mặc định: không gọi LLM ngoài nên không thể hallucinate, chạy được
trong CI không cần API key, và là đường cơ sở trung thực để so với provider LLM sau này.
"""

from __future__ import annotations

from typing import Protocol

from app.captions import SEVERITY_VI
from app.rag.retrieval import RetrievedEvent

NO_EVIDENCE_VI = (
    "Không tìm thấy sự kiện nào phù hợp với câu hỏi trong khoảng thời gian được hỏi. "
    "Hệ thống không suy đoán khi không có bằng chứng."
)


class AnswerProvider(Protocol):
    name: str

    def answer(self, question: str, events: list[RetrievedEvent]) -> str: ...


class TemplateProvider:
    """Trích xuất thuần: mọi câu chữ đều lấy từ caption đã lưu, không thêm thông tin mới."""

    name = "template"

    def answer(self, question: str, events: list[RetrievedEvent]) -> str:
        if not events:
            return NO_EVIDENCE_VI

        lines = [f"Tìm thấy {len(events)} sự kiện liên quan:"]
        for event in events:
            moment = event.window_start.strftime("%H:%M ngày %d/%m/%Y")
            muc = SEVERITY_VI.get(event.severity, event.severity)
            lines.append(f"- {moment}: {event.caption_vi} (mức {muc}, {event.event_id})")
        return "\n".join(lines)


def get_provider(name: str) -> AnswerProvider:
    """Chọn provider theo cấu hình. Tên lạ ⇒ dừng ngay, KHÔNG âm thầm rơi về template:
    một hệ thống tưởng đang chạy Claude mà thực ra chạy template sẽ cho ra số liệu đánh
    giá sai hoàn toàn mà không ai biết."""
    providers: dict[str, AnswerProvider] = {"template": TemplateProvider()}
    if name not in providers:
        raise ValueError(
            f"RAG_ANSWER_PROVIDER={name!r} chưa được hiện thực. Có: {sorted(providers)}. "
            "AnthropicProvider/OllamaProvider cắm vào đây (xem SYSTEM.md §7.3)."
        )
    return providers[name]
