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

### 2026-09-22 (cập nhật) · game/phim tập trung ở nhóm lớp bạo lực — đo được, không còn là nghi ngờ

**Tình huống:** Sau khi loại `0QlQLIvLpNI_170000`, `select_gold_pilot.py` chọn 6 file
thay thế (vòng 2), cả 6 đều mang ít nhất một trong các lớp `glass_breaking` / `scream` /
`explosion` / `gunshot`. Người gán nghe hết cả 6 — **6/6 (100%) là game/phim**, không có
file nào là audio thật.

Tính gộp cả hai vòng: **12/12 file bị loại (6 "rác" vòng 1 + 6 game vòng 2) đều chạm ít
nhất một trong 5 lớp {glass_breaking, scream, explosion, gunshot, siren}.** Không một file
nào bị loại mà chỉ thuộc các lớp "đời thường" (`applause_cheering`, `door_slam`,
`laughter`, `object_drop_dishes`, `running_footsteps`, `alarm_bell`, `vehicle_crash`,
`fireworks`).

**Quyết định:** Xác nhận đây là **mẫu hình cấu trúc của nguồn**, không phải trùng hợp:
súng nổ / bom / kính vỡ / tiếng thét **thật** hiếm khi được quay và đăng công khai lên
YouTube (nguy hiểm, bất hợp pháp, hoặc hiếm xảy ra trước ống kính), nên phần lớn nội dung
AudioSet gán các nhãn "kịch tính" này thực ra đến từ game/phim/dàn dựng. Tiếp tục xử lý
theo cơ chế đã có (loại + `select_gold_pilot.py` chọn thay thế) — pool chưa cạn (lớp tệ
nhất `scream` còn 19/22 file tính đến 22/09), nhưng cần lường trước: **các lớp bạo lực sẽ
tốn nhiều lượt nghe hơn hẳn mức trung bình** để đủ ≥20 sự kiện mỗi lớp ở cổng §8.5.

**Lý do:** Con số 12/12 = 100% không thể coi là nhiễu thống kê ở cỡ mẫu này. Đây là phát
hiện thật về nguồn dữ liệu, cần ghi vào phần Hạn chế/Phương pháp của khoá luận — không chỉ
ảnh hưởng riêng pilot mà còn cảnh báo trước cho lần gán đại trà (bước 4 của §8.3).

**Áp dụng chung:** khi gặp file thuộc `glass_breaking`/`scream`/`explosion`/`gunshot`/
`siren`, tăng cảnh giác với khả năng là game/phim — không có nghĩa loại thẳng mà không
nghe, nhưng chuẩn bị tinh thần tỉ lệ hỏng cao hơn các lớp khác nhiều lần.

---
