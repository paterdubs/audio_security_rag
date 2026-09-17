"""Caption theo template từ timeline sự kiện.

Đây KHÔNG phải code vứt đi: nó chính là **baseline B0** trong SYSTEM.md §5.5 —
"Structured captioner: timeline SED → template". B0 là cận trên về grounding (EHR ≈ 0,
vì mọi từ đều sinh từ detection có thật) và cận dưới về độ tự nhiên. Model đề xuất phải
chứng minh nó tốt hơn B0 ở CIDEr mà không tệ hơn nhiều ở EHR — không có B0 thì không có
gì để so.

Sinh cả EN (để benchmark với AudioCaps/Clotho) lẫn VI (để embed cho RAG) — §7.4.
"""

from __future__ import annotations

from app.vocab import CLASS_EN, CLASS_VI

NO_EVENT_EN = "No salient sound event was detected."
NO_EVENT_VI = "Không phát hiện sự kiện âm thanh đáng chú ý."


def _join(parts: list[str], last_word: str) -> str:
    if len(parts) == 1:
        return parts[0]
    return f"{', '.join(parts[:-1])} {last_word} {parts[-1]}"


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def caption_en(detections: list[dict]) -> str:
    if not detections:
        return NO_EVENT_EN
    ordered = sorted(detections, key=lambda d: d["onset"])
    names = [CLASS_EN.get(d["class_id"], d["class_id"]) for d in ordered]

    if len(ordered) == 1:
        return f"{names[0].capitalize()} is heard.".replace("  ", " ")

    # Chồng lấn ⇒ "cùng lúc"; nối tiếp ⇒ "sau đó". Phân biệt hai trường hợp này là điều
    # kiện cần để đo TOA (Temporal Order Accuracy, §8.2) — một caption nói sai thứ tự
    # thời gian là một dạng hallucination riêng, không phải lỗi từ ngữ.
    first, second = (ordered[0]["onset"], ordered[0]["offset"]), (ordered[1]["onset"], ordered[1]["offset"])
    if _overlaps(first, second):
        return f"{_join(names, 'and').capitalize()} are heard at the same time."
    return f"{names[0].capitalize()} is followed by {_join(names[1:], 'and')}."


def caption_vi(detections: list[dict]) -> str:
    if not detections:
        return NO_EVENT_VI
    ordered = sorted(detections, key=lambda d: d["onset"])
    names = [CLASS_VI.get(d["class_id"], d["class_id"]) for d in ordered]

    if len(ordered) == 1:
        return f"Nghe thấy {names[0]}."

    first, second = (ordered[0]["onset"], ordered[0]["offset"]), (ordered[1]["onset"], ordered[1]["offset"])
    if _overlaps(first, second):
        return f"Nghe thấy {_join(names, 'và')} cùng lúc."
    return f"Nghe thấy {names[0]}, sau đó là {_join(names[1:], 'và')}."
