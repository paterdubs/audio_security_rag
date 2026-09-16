# ADR-0005 — Giữ `shout_yell` dưới mức tối thiểu, không bổ sung thêm

**Trạng thái:** Đã chốt · 2026-09-16
**Cổng:** D7 — phân tích thiếu hụt (DATA_PLAN §11)

## Bối cảnh

Sau khi duyệt toàn bộ hàng đợi người (2077 clip trong `screen_audit.csv`), foreground
bank đạt 4268 clip, 15/16 lớp vượt mức tối thiểu. Riêng `shout_yell`:

| | Có | Mục tiêu | Tối thiểu |
|---|---:|---:|---:|
| `shout_yell` | 56 | 200 | 80 |

Nguyên nhân: PANNs CNN14 không có nhãn AudioSet riêng cho "quát tháo có lời" — tagger
nhận nhầm sang Speech/Groan/Gasp (đã ghi nhận ở `guard_low_resolution_classes()`,
`auto_screen.py`). Vì vậy cả sàng lọc tự động lẫn nguồn tự nhiên đều ít cho lớp này,
không phải do bank thiếu clip khai thác được — mà do đặc trưng âm học của lớp trùng
lấp nhiều với `speech_normal`/`scream`.

## Quyết định

Chọn **Phương án C** (DATA_PLAN §11): **giữ lớp `shout_yell` trong taxonomy, không bổ
sung thêm dữ liệu**, chấp nhận cỡ mẫu 56/80. Người dùng xác nhận trực tiếp: *"không
cần bổ sung thêm"*.

Không chọn:
- **A (bổ sung có chủ đích)** — không còn nguồn công khai nào chưa khai thác cho lớp
  này trong 5 dataset hiện có; tìm nguồn mới tốn thời gian không tương xứng lợi ích.
- **B (gộp lớp)** — `shout_yell` là lớp Nhóm A (sinh cảnh báo), gộp với `scream` hay
  `speech_normal` sẽ xoá mất phân biệt "có lời/không lời" mà taxonomy dựa vào để tính
  FAR ở khu dân cư — đổi lấy 24 clip không đáng.
- **D (bỏ lớp)** — mất hẳn khả năng phát hiện "quát tháo, cãi vã to tiếng", một tín
  hiệu an ninh thật ở khu dân cư/trường học.

## Hệ quả

- `shout_yell` **phải nêu rõ trong mọi bảng kết quả** (Event-F1, PSDS theo lớp) là cỡ
  mẫu dưới mức tối thiểu — số liệu ở lớp này có phương sai cao hơn các lớp khác.
- Không loại khỏi macro-F1 (khác khuyến nghị mặc định của DATA_PLAN cho lớp thiếu dữ
  liệu) — quyết định vẫn tính lớp này vào macro-F1, nhưng ghi chú rõ cỡ mẫu nhỏ trong
  phần Hạn chế của khoá luận.
- Nếu buổi thu thực địa tại IUH (thu thực địa, kịch bản `media_playback`) tình cờ ghi
  được thêm mẫu `shout_yell`, có thể bổ sung sau — quyết định này không đóng cửa vĩnh
  viễn, chỉ là không chủ động tìm thêm ở giai đoạn hiện tại.
