"""Glossary cố định 16 lớp — SYSTEM.md §7.4.

⚠️ BẮT BUỘC cố định: nếu cùng một lớp lúc gọi "tiếng kính vỡ", lúc "tiếng vỡ kính", lúc
"kính bị đập", embedding sẽ phân mảnh và retrieval sụt. Đây là chi tiết nhỏ nhưng bỏ qua
thì hỏng cả tầng RAG.

Bản EN dùng để benchmark với AudioCaps/Clotho; bản VI dùng để embed và hiển thị.
"""

from __future__ import annotations

CLASS_EN: dict[str, str] = {
    "gunshot": "a gunshot",
    "explosion": "an explosion",
    "scream": "a person screaming",
    "glass_breaking": "glass breaking",
    "vehicle_crash": "a vehicle crash",
    "shout_yell": "a person shouting",
    "door_slam": "a door slamming",
    "running_footsteps": "running footsteps",
    "siren": "a siren",
    "alarm_bell": "an alarm bell",
    "fireworks": "fireworks",
    "object_drop_dishes": "objects or dishes falling",
    "laughter": "laughter",
    "applause_cheering": "applause and cheering",
    "speech_normal": "people talking",
    "ambient_noise": "ambient background noise",
}

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
