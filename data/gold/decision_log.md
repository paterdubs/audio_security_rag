# decision_log.md — Nhật ký quyết định gán nhãn

> Đối soát **18/09/2026**: chưa có gold annotation artifact ngoài nhật ký này.
> Mục chuông quầy phục vụ bên dưới vẫn **CHỜ XÁC NHẬN**; chưa có quyết định mới của người
> gán nhãn nên không tự đổi taxonomy hay verdict. Tình trạng dự án: [STATUS](../../docs/STATUS.md).

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

### 2026-09-22 · pilot lần 1 · `as_strong_0QlQLIvLpNI_170000.wav`

**Tình huống:** Nghe file này (thuộc `gold_test`, chọn thay thế cho 1 trong 6 file "rác"
của vòng trước), gần như toàn bộ nội dung là âm thanh trò chơi điện tử (hiệu ứng game),
chỉ đúng một đoạn `speech_normal` là giọng người thật. AudioSet gán nhãn `glass_breaking`
cho clip này vì có một hiệu ứng vỡ kính TRONG GAME nghe tương tự âm thật — nhãn không sai
theo định nghĩa "có nghe thấy loại âm thanh này", nhưng đây không phải một sự kiện xảy ra
trong môi trường vật lý thật.

**Quyết định:** Loại cả clip, không gán bất kỳ dòng nào (kể cả `speech_normal` hợp lệ) —
coi như "audio rác" theo nghĩa rộng, ghi vào `exclusions.csv` với mã riêng
`synthetic_or_game_audio` (khác `unusable` của 6 file trước, để sau này đếm riêng được
tỉ lệ nhiễm game/phim ảnh trong nguồn AudioSet-strong).

**Lý do:** `DATA_PLAN.md` §8 định nghĩa gold_test là *"audio THẬT"* — yêu cầu này chưa
từng được thao tác hoá thành tiêu chí cụ thể trong `annotation_guideline.md`. AudioSet
lấy nhãn từ audio track của video YouTube; quy trình kiểm định của AudioSet chỉ xác nhận
"loại âm thanh X có xuất hiện", không phân biệt bản ghi âm thật ngoài đời với hiệu ứng
game/phim/dàn dựng — hai loại có thể cho ra sóng âm gần giống hệt nhau nhưng KHÔNG mang
đặc trưng ngữ cảnh thật (dội âm phòng thật, tạp âm nền thật, động lực học tự nhiên) mà
mô hình cần học để hoạt động đúng trong triển khai thực tế.

**Áp dụng chung:** mọi clip mà nội dung nghe rõ ràng là trò chơi điện tử / phim ảnh / hiệu
ứng âm thanh dàn dựng (không phải ghi âm một sự kiện thật) → loại cả clip, dùng
`reason_code=synthetic_or_game_audio`. Đã bổ sung thành quy tắc tường minh ở
`annotation_guideline.md` §0.1 — áp dụng ngay từ pilot, trước khi đóng băng guideline.

---
