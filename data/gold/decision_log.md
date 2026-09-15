# decision_log.md — Nhật ký quyết định gán nhãn

> Mọi ca không có sẵn trong `taxonomy.md` / `annotation_guideline.md` đều ghi vào đây,
> kèm quyết định và lý do. Vừa là công cụ chống trôi tiêu chí giữa các phiên, vừa là
> vật liệu trực tiếp cho chương phân tích lỗi của khoá luận.
>
> Ba câu hỏi phải trả lời mỗi mục: *tình huống gì · quyết thế nào · vì sao chọn thế*.

---

### 2026-09-15 · kiểm định §4.4 · lớp `alarm_bell`

**Tình huống:** Trong 25 clip `alarm_bell` của phiếu kiểm định xuất hiện nhiều loại
chuông khác nhau: chuông nhà thờ, chuông cửa, và **chuông quầy phục vụ** (loại gõ một
tiếng "ting" để gọi nhân viên, thường thấy ở quầy thức ăn nhanh).

Chuông nhà thờ và chuông cửa đã được taxonomy A10 liệt kê rõ là **thuộc** lớp này.
Chuông quầy phục vụ thì **không có** trong danh sách bao gồm lẫn danh sách loại trừ.

**Quyết định:** ⚠️ CHỜ XÁC NHẬN — đề xuất xếp chuông quầy phục vụ **ngoài** `alarm_bell`
(điền `wrong_class` khi gặp trong phiếu kiểm định).

**Lý do:** Câu định nghĩa mở đầu A10 là "thiết bị phát tín hiệu **cảnh báo lặp lại**".
Chuông quầy là một tiếng đơn để gọi người, không lặp lại và không mang nghĩa cảnh báo.

Lý do vận hành quan trọng hơn: `alarm_bell` thuộc Nhóm A, tức là lớp **sinh cảnh báo**.
Đưa chuông quầy vào bank nghĩa là dạy model rằng một tiếng "ting" là sự kiện an ninh —
và một trong bốn khu vực triển khai là trường học, nơi căng tin có đúng loại chuông đó.
Hệ quả là báo động giả đều đặn mỗi giờ ăn trưa, đúng kiểu làm người dùng tắt hệ thống.

**Áp dụng chung:** mọi chuông gọi phục vụ / chuông quầy / chuông bàn về sau xử lý y hệt.
Nếu xác nhận, phải bổ sung dòng này vào `taxonomy.md` A10 mục ❌ **Không bao gồm**.

---
