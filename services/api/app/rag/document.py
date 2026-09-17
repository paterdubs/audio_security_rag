"""Văn bản đại diện cho một sự kiện khi đem đi nhúng — SYSTEM.md §7.2, §7.4.

VÌ SAO TỒN TẠI FILE NÀY (đo được, không phải suy đoán):

Bản đầu của `audio/upload` nhúng thẳng `caption_vi`, tức một câu template trần như
"Nghe thấy tiếng súng." Đo trên 9 sự kiện thật thì độ tương đồng GIỮA CÁC SỰ KIỆN
KHÁC HẲN NHAU nằm ở 0.83–0.92 — "tiếng súng" và "tiếng pháo" giống nhau 0.88. Lý do:
mọi caption dùng chung khuôn "Nghe thấy tiếng … .", nên phần khuôn chi phối vector còn
từ phân biệt gần như không có tiếng nói.

Hậu quả đo được: 8/8 câu hỏi — kể cả "công thức nấu phở bò" và "cài đặt máy in canon" —
đều trả về CÙNG một sự kiện, và câu lạc đề còn ghi điểm cao hơn câu đúng đề
(phở 0.5048 so với tiếng súng 0.4036). Truy hồi ngữ nghĩa coi như không hoạt động.

Cách chữa: lập chỉ mục trên một văn bản mang đủ thông tin phân biệt — tên lớp, địa
điểm, thời gian, mức độ — thay vì mỗi câu caption. Đây cũng đúng là những thứ người
dùng hỏi tới ("ở nhà xe", "tối qua", "có nghiêm trọng không").

Glossary cố định ở `captions.py` vẫn là bắt buộc: cùng một lớp phải luôn gọi bằng một
tên, nếu không embedding lại phân mảnh theo cách khác.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.captions import SEVERITY_VI, class_vi

# Giờ Việt Nam. Sự kiện LƯU trong DB bằng UTC (đúng chuẩn, không đổi), nhưng văn bản
# đem đi nhúng phải nói theo giờ địa phương: người dùng hỏi "tối qua", "lúc 11 giờ đêm"
# theo đồng hồ của họ, không theo UTC.
LOCAL_TZ = timezone(timedelta(hours=7))

AREA_TYPE_VI: dict[str, str] = {
    "school": "trường học",
    "parking": "nhà xe",
    "residential": "khu dân cư",
    "factory": "nhà xưởng",
}


def part_of_day(hour: int) -> str:
    """Buổi trong ngày theo giờ địa phương.

    Có mặt để câu hỏi dạng "tối qua có gì không" bắt được sự kiện, thay vì phải khớp
    đúng con số giờ.
    """
    if 5 <= hour < 11:
        return "buổi sáng"
    if 11 <= hour < 14:
        return "buổi trưa"
    if 14 <= hour < 18:
        return "buổi chiều"
    if 18 <= hour < 23:
        return "buổi tối"
    return "ban đêm"


def build_document(
    *,
    caption_vi: str,
    class_ids: list[str],
    window_start: datetime,
    location_name: str | None = None,
    area_type: str | None = None,
    severity: str | None = None,
) -> str:
    """Văn bản sẽ được nhúng và lập chỉ mục cho một sự kiện.

    Tên lớp đứng RIÊNG một dòng chứ không chỉ nằm trong caption: đó là từ phân biệt
    chính, để nó lặp lại là cố ý — nhằm kéo vector ra khỏi phần khuôn mẫu chung.
    """
    if window_start.tzinfo is None:
        # Ngây thơ thì coi là UTC: cột window_start là timestamptz và mọi chỗ ghi đều
        # dùng UTC. Đoán nhầm sang giờ địa phương sẽ lệch 7 tiếng một cách âm thầm.
        window_start = window_start.replace(tzinfo=timezone.utc)
    local = window_start.astimezone(LOCAL_TZ)

    # Nhãn trường ("Sự kiện an ninh:", "Mức độ nguy hiểm:") đã bị bỏ CÓ CHỦ Ý. Đo thật:
    # thêm chúng vào thì hai sự kiện khác hẳn nhau vẫn dùng chung 56% token, gần như
    # không hơn gì caption trần (60%) — vì nhãn trường lặp y hệt ở MỌI tài liệu, đúng
    # loại khuôn mẫu đã gây ra lỗi ban đầu. Giữ phần khuôn ở mức tối thiểu.
    if class_ids:
        # Tên lớp đứng đầu và lặp lại ở cả hai dạng (tiếng Việt + class_id tiếng Anh):
        # đây là từ phân biệt chính, và class_id giúp bắt câu hỏi lẫn tiếng Anh.
        names = ", ".join(class_vi(c) for c in class_ids)
        lines = [names, ", ".join(class_ids), caption_vi]
    else:
        # Nói thẳng là không có, thay vì bỏ trống. Một sự kiện "không nghe thấy gì" cũng
        # phải truy xuất được bằng chính câu hỏi ấy.
        lines = ["không có sự kiện đáng chú ý", caption_vi]

    if location_name:
        where = location_name
        if area_type:
            where += f" ({AREA_TYPE_VI.get(area_type, area_type)})"
        lines.append(where)

    lines.append(f"{local:%H:%M} ngày {local:%d/%m/%Y}, {part_of_day(local.hour)}")

    if severity:
        lines.append(f"mức {SEVERITY_VI.get(severity, severity)}")

    return "\n".join(lines)
