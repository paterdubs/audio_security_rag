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

### 2026-09-22 (tiếp) · `as_strong_0yZGysisqY0_12000.wav` · âm thanh đồ chơi

**Tình huống:** Clip có tiếng nổ nghe gần giống súng, nhưng người gán xác định đó là
**âm thanh đồ chơi** (đồ chơi mô phỏng, không phải súng thật) — khác trường hợp
game/phim ở các mục trên: đây có thể là bản ghi âm THẬT (ai đó quay cảnh chơi đồ chơi),
chỉ là nội dung không phải sự kiện `gunshot` thật.

**Quyết định:** **Không gán** dòng `gunshot` cho clip này. Không loại cả file khỏi
`gold_test` (không có bằng chứng đây là game/rác/nhạc) — xử lý như trường hợp "không có
sự kiện thuộc 16 lớp", giống `as_strong_-ewBC-6gLSo_30000.wav` ở pilot lần 1: **không ghi
dòng nào vào TSV**, file vẫn ở trong `gold_test`.

**Lý do:** Cùng nguyên tắc đã áp dụng ở mục **2026-09-15 · chuông quầy phục vụ**: âm
thanh acoustically giống lớp mục tiêu nhưng KHÔNG mang đúng ý nghĩa của lớp đó (thiết bị
gọi phục vụ ≠ báo động; đồ chơi ≠ vũ khí thật) thì không được gán vào lớp mục tiêu, dù
sóng âm nghe tương tự. Gán nhầm sẽ dạy model báo động giả với đồ chơi trẻ em — hậu quả
thực tế tương tự ca chuông quầy: khu dân cư/trường học là nơi trẻ em chơi đồ chơi phổ
biến, false alarm liên tục sẽ khiến người dùng tắt hệ thống.

**Áp dụng chung:** mọi âm thanh đồ chơi/mô phỏng mô tả *chức năng* khác lớp mục tiêu
(dù *âm thanh* giống) → không gán vào lớp mục tiêu; nếu clip không còn sự kiện nào khác
đáng gán thì coi như "không có sự kiện", giữ nguyên trong `gold_test`, không dùng
`exclusions.csv`. Nếu việc này lặp lại ≥3 lần, cần bổ sung một dòng vào `taxonomy.md` mục
`gunshot` phần ❌ Không bao gồm, tương tự đã làm với chuông quầy ở `alarm_bell`.

---

### 2026-09-22 (tổng kết) · Pilot lần 1 hoàn tất — 30/30, số liệu đầy đủ

**Tình huống:** Pilot lần 1 (DATA_PLAN §8.3 bước 1) đã gán đủ 30 clip. Quá trình phải
loại và thay thế nhiều lần vì lẫn game/phim/nhạc nền trong nguồn AudioSet-strong.

**Số liệu cuối cùng:**

| Lý do loại | Số file (segment) |
|---|---|
| `synthetic_or_game_audio` (game/phim) | 7 |
| `unusable` (rác/không liên quan, chưa xác định rõ loại) | 6 |
| `music_no_event` (nhạc nền, không có sự kiện) | 3 |
| **Tổng loại** | **16** |

**Tổng lượt nghe để có đủ 30 clip hợp lệ: 46 (30 dùng được + 16 loại) → tỉ lệ thất bại
34,8%.** Tức là cứ 3 clip audioset_strong thuộc diện `gold_test` thì trung bình hỏng hơn
1 clip — một con số cần báo cáo nguyên văn trong phần Hạn chế của khoá luận, không làm
tròn nhẹ đi.

**52 dòng sự kiện hợp lệ**, phân bố theo lớp: `speech_normal` 24 · `applause_cheering` 6
· `alarm_bell` 5 · `siren` 4 · `object_drop_dishes` 3 · `gunshot` 3 · `door_slam` 2 ·
`scream` 2 · `laughter` 1 · `glass_breaking` 1 · `explosion` 1. **4 lớp vắng mặt hoàn
toàn trong pilot: `fireworks`, `running_footsteps`, `shout_yell`, `vehicle_crash`** —
hệ quả của việc `select_gold_pilot.py` chọn thay thế dựa trên nhãn AudioSet gốc (đã biết
không đáng tin cho các lớp kịch tính), nên vòng thay thế dồn về đúng những lớp hay hỏng
nhất chứ không giữ đều 15 lớp như lần chọn đầu.

**Quyết định:** Không coi việc thiếu 4 lớp là lỗi cần sửa ngay — mục đích của PILOT là
kiểm tra `annotation_guideline.md` có đủ rõ không (đo tự-nhất-quán ở bước 2-3), không
phải phủ đều 15 lớp. Yêu cầu phủ lớp (≥20 sự kiện/lớp) chỉ áp dụng cho `gold_test` đầy đủ
ở cổng §8.5, không áp dụng cho 30 clip pilot.

**Lý do:** Ba mã lý do tách riêng (`unusable`/`synthetic_or_game_audio`/`music_no_event`)
cho phép đo được CHÍNH XÁC bức tranh nhiễm của nguồn thay vì một con số "rác" mơ hồ —
đúng tinh thần N3 và mục đích ban đầu khi tạo mã riêng.

**Áp dụng chung:** Khi gán đại trà (bước 4 của §8.3) trên phần còn lại của `gold_test`
(367 − 30 hợp lệ − 16 loại = 321 file chưa xét), tiếp tục dùng đúng ba mã lý do này. Tính
lại tỉ lệ thất bại theo lớp định kỳ — nếu một lớp cụ thể (nghi ngờ nhất: `gunshot`,
`explosion`, `glass_breaking`, `scream`, `siren`) có tỉ lệ hỏng > 50% khi gán đại trà,
cần cân nhắc bổ sung nguồn khác (MIVIA, buổi thu thực địa IUH) thay vì tiếp tục rút cạn
AudioSet-strong cho riêng các lớp đó.

---
