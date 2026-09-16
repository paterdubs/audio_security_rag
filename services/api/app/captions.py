"""Từ điển lớp → tiếng Việt, và tiện ích mô tả sự kiện.

⚠️ Glossary cố định là BẮT BUỘC (SYSTEM.md §7.4): nếu cùng một lớp lúc gọi là "tiếng kính
vỡ", lúc "tiếng vỡ kính", lúc "kính bị đập", thì embedding phân mảnh và retrieval sụt.
Đây là bản dùng chung cho cả caption template lẫn câu trả lời RAG.
"""

from __future__ import annotations

CLASS_VI: dict[str, str] = {
    "gunshot": "tiếng súng",
    "explosion": "tiếng nổ",
    "scream": "tiếng hét hoảng loạn",
    "glass_breaking": "tiếng kính vỡ",
    "vehicle_crash": "tiếng va chạm xe",
    "shout_yell": "tiếng quát tháo",
    "door_slam": "tiếng đóng sập cửa",
    "running_footsteps": "tiếng bước chân chạy",
    "siren": "tiếng còi hú",
    "alarm_bell": "tiếng chuông báo động",
    "fireworks": "tiếng pháo",
    "object_drop_dishes": "tiếng rơi vỡ đồ vật",
    "laughter": "tiếng cười",
    "applause_cheering": "tiếng vỗ tay, reo hò",
    "speech_normal": "tiếng nói chuyện",
    "ambient_noise": "tiếng ồn nền",
}

SEVERITY_VI: dict[str, str] = {
    "LOW": "thấp",
    "MEDIUM": "trung bình",
    "HIGH": "cao",
    "CRITICAL": "nghiêm trọng",
}


def class_vi(class_id: str) -> str:
    # Lớp lạ: trả lại nguyên class_id thay vì bịa tên tiếng Việt — người đọc cần thấy
    # đúng thứ hệ thống đang nói, kể cả khi đó là một id chưa dịch.
    return CLASS_VI.get(class_id, class_id)
