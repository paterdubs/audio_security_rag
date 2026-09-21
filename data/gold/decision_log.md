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

### 2026-09-22 (nghe lần ba) · Pilot lần 2 KHÔNG đạt cổng §8.5 — 13/30 clip có ca bất đồng

Bối cảnh chung cho 9 mục dưới đây: `agreement.py` chạy thật trên `pilot_v1_lan1.tsv` ↔
`pilot_v1_lan2.tsv` cho ra event-F1 tự-nhất-quán **0,7339** (ngưỡng ≥0,75 — hụt 0,0161,
KHÔNG ĐẠT) và lệch onset trung vị 10,0 ms (ĐẠT). Theo DATA_PLAN §8.2 B2, mọi ca lần 1 ≠
lần 2 phải nghe lại lần ba, chốt, ghi lý do — chín mục dưới đây là kết quả của việc đó.

---

### 2026-09-22 (nghe lần ba) · Nhóm lệch biên nhẹ, đúng lớp, nghe lại chỉ chỉnh mốc thời gian

**File:** `as_strong_-0TTFAArJ9k_30000.wav`, `as_strong_-99MQ06mPYE_30000.wav`,
`as_strong_-FiHp5DFJmY_30000.wav`.

**Tình huống:** Cả ba đúng lớp và đúng cấu trúc sự kiện ở cả hai lần, chỉ lệch onset/offset
vài trăm ms tới dưới 1s — không phải bất đồng về NỘI DUNG, mà về ĐỘ CHÍNH XÁC đặt biên.

**Quyết định:**
- `-0TTFAArJ9k_30000`: `speech_normal` onset = **0,000s** (theo lần 2; lần 1 đặt 0,280s,
  bỏ sót đoạn đầu rất khẽ).
- `-99MQ06mPYE_30000`: giữ cấu trúc 5 đoạn (speech/object_drop_dishes xen kẽ) của cả hai
  lần, mốc thời gian nghe lần ba cho ra một bộ số thứ ba nằm giữa lần 1 và lần 2 (ví dụ
  đoạn speech giữa: onset 3,27s) — dao động đo bình thường của một chuyển tiếp mềm
  (giọng nói nhỏ dần trước khi có tiếng đồ vật rơi), không phải lỗi.
- `-FiHp5DFJmY_30000`: `laughter` onset = **2,68s** (theo lần 2; lần 1 đặt 3,12s, trễ hơn
  0,44s so với tiếng cười thật).

**Lý do:** Cả ba là bẫy "âm thanh bắt đầu rất khẽ, dễ đặt onset trễ hơn thực tế" — không
cần sửa quy tắc chung, ghi lại để có dữ liệu tham chiếu cho phần phân tích độ nhạy biên
của khoá luận.

---

### 2026-09-22 (nghe lần ba) · Nhóm sự kiện thật bị bỏ sót ở lần 1, lần 2 (mù) nghe đúng

**File:** `as_strong_-glN59TmfME_30000.wav`, `as_strong_-gqN3xq05ns_10000.wav`,
`as_strong_0BCXcbaLeoA_30000.wav`.

**Tình huống:**
- `-glN59TmfME_30000`: lần 1 chỉ gán `speech_normal` hai đoạn giữa clip (2,47–5,18s và
  6,60–7,17s), để trống đầu/cuối clip. Nghe lần ba xác nhận giọng nói chạy liên tục suốt
  10s, kể cả đầu/cuối bị bỏ sót.
- `-gqN3xq05ns_10000`: lần 1 gán **một** khối `gunshot` 0–10s duy nhất. Nghe lần ba xác
  nhận đây là **ba** phát súng rời rạc (0,00–1,55s · 2,33–3,95s · 4,23–5,50s) — lần 1 gán
  quá thô, không tách sự kiện.
- `0BCXcbaLeoA_30000`: lần 1 chỉ gán 2 sự kiện (`object_drop_dishes`, `glass_breaking`).
  Nghe lần ba xác nhận có **thêm một** `glass_breaking` ở cuối clip (9,69–10,00s) mà lần 1
  bỏ sót hoàn toàn.

**Quyết định:** Chốt theo cấu trúc của lần 2 (mù) cho cả ba file — lần 2 nghe đầy đủ hơn
lần 1 ở cả ba trường hợp.

**Lý do:** Không có mẫu hình chung rõ ràng giữa ba ca này (một do bỏ sót đầu/cuối clip,
một do gán quá thô không tách sự kiện, một do bỏ sót hoàn toàn một sự kiện) — ghi nhận
như bằng chứng cho thấy **lần 1 không phải lúc nào cũng đúng hơn lần 2**: điều kiện mù
(không nhớ lần trước, nghe với tâm thế mới) đôi khi bắt được sự kiện lần 1 đã bỏ lỡ.

---

### 2026-09-22 (nghe lần ba) · `as_strong_-AIHL5EIKUg_30000.wav` — 2 sự kiện "thừa" của lần 2 bị loại

**Tình huống:** Lần 2 (mù) thêm hai sự kiện lần 1 không có: `speech_normal` ở đầu clip
(0,00–0,71s) và một `door_slam` ngắn ở giữa (6,63–6,72s, chỉ 0,09s). Sự kiện `door_slam`
cuối clip cũng lệch: lần 1 đặt onset 9,06s, lần 2 đặt 9,70s.

**Quyết định:** Nghe lần ba **loại bỏ** cả hai sự kiện "thừa" của lần 2 — không có giọng
nói ở đầu clip, và tiếng "cạch" ở 6,63–6,72s không đủ rõ để coi là `door_slam` (quá ngắn,
có thể là tạp âm). Giữ cấu trúc 4 sự kiện của lần 1, nhưng cập nhật onset `door_slam` cuối
theo lần 2 (**9,65s**, gần 9,70s hơn 9,06s — nghe lần ba xác nhận tiếng cửa đóng xảy ra
muộn hơn lần 1 ghi).

**Lý do:** Case ngược với nhóm trên — ở đây lần 2 (mù) là bên **sai dương tính** (nghe ra
sự kiện không có thật), có thể do kỳ vọng "phải tìm thấy nhiều sự kiện hơn" khi gán mù lần
thứ hai trên cùng một dạng clip nhiều `door_slam`. Ghi lại để cân bằng lại nhận định ở
nhóm trên: không phải lúc nào lần 2 cũng đáng tin hơn lần 1, phải nghe lần ba mới quyết
được, không suy diễn "lần mù luôn đúng hơn".

---

### 2026-09-22 (nghe lần ba) · `as_strong_13ApmSOcZpU_30000.wav` — lần 2 tự nhân đôi dòng, không phải bất đồng thính giác

**Tình huống:** `agreement.py` báo "thay_the: speech_normal(L1) → siren(L2)" ở mốc ~3,4s.
Nhưng xem đầy đủ dữ liệu lần 2: `siren` bị ghi **hai lần chồng lấn** (1,24–6,18s VÀ
3,46–5,53s là tập con của đoạn trước) — không phải một sự kiện `siren` mới thay thế
`speech_normal`, mà là lỗi cấu trúc khi gán (ghi nhầm cùng một tiếng còi thành hai dòng).

**Quyết định:** Chốt theo cấu trúc lần 1 (5 đoạn, siren MỘT đoạn liên tục 1,08–6,50s chồng
lấn với speech_normal ở giữa 3,40–5,50s) — nghe lần ba xác nhận không có sự kiện `siren`
thứ hai độc lập nào.

**Lý do:** Đây KHÔNG phải một ca bất đồng thính giác thật — là lỗi thao tác lúc gán mù
(có thể do quên đã ghi đoạn `siren` trước, ghi lại một phần của nó). Phân biệt rõ với các
ca khác: `agreement.py` không phân biệt được "bất đồng thật" với "lỗi cấu trúc dữ liệu";
người gán phải tự nhận ra khi nghe lần ba, script không làm thay được.

---

### 2026-09-22 (nghe lần ba) · `as_strong_0BEtPXdDQrs_250000.wav` — đổi lớp thật: `siren`, không phải `alarm_bell`

**Tình huống:** Lần 1 gán `alarm_bell` nguyên clip (0–10s). Lần 2 (mù) gán `siren`. Đây là
ca "thay_the" nghiêm trọng nhất trong 13 ca — đổi hẳn lớp trên toàn bộ clip, không phải
lệch biên.

**Quyết định:** Chốt là **`siren`**, không phải `alarm_bell`. Lần 1 gán sai.

**Lý do:** Nghe lần ba: âm thanh có độ lên-xuống cao độ tuần hoàn kiểu còi xe cấp cứu, không
phải tiếng chuông cộng hưởng của `alarm_bell`. `alarm_bell` và `siren` là cặp dễ nhầm về
mặt cảm nhận (cả hai đều là "âm thanh cảnh báo lặp đi lặp lại") — nhầm lẫn NGAY CẢ với
cùng một người gán, không chỉ giữa mô hình và người.

**Áp dụng chung:** Thêm `siren` ↔ `alarm_bell` vào danh sách cặp dễ nhầm cần chú ý đặc
biệt khi gán (kiểm tra `confusable_with` của cả hai lớp trong `ontology_map.yaml`, bổ
sung nếu chưa có). Gợi ý phân biệt: `siren` có độ lên/xuống cao độ rõ (dải tần trượt liên
tục kiểu còi xe), `alarm_bell` là tiếng cộng hưởng rời rạc/lặp nhịp (kiểu chuông/còi báo
cháy).

---

### 2026-09-22 (nghe lần ba) · `as_strong_-3MV6T3u9Ig_30000.wav` và `as_strong_-HIozxABvUQ_30000.wav` — phát hiện mẫu hình: `speech_normal` chạy nền dưới `applause_cheering`

**Tình huống:** Cả hai lần gán trước (lần 1 và lần 2) đều mặc định coi `speech_normal` và
`applause_cheering` là TUẦN TỰ — giọng nói kết thúc ngay khi tiếng vỗ tay bắt đầu. Nghe
lần ba cho cả hai file phát hiện giọng nói (đám đông/MC) thực ra vẫn tồn tại NỀN, đồng
thời với tiếng vỗ tay, suốt phần còn lại của clip:
- `-3MV6T3u9Ig_30000`: chốt = `speech_normal` 0,00–1,24s; `applause_cheering` VÀ
  `speech_normal` cùng chồng lấn 4,00–10,00s (khác cả lần 1 lẫn lần 2, không lần nào ghi
  chồng lấn ở đây).
- `-HIozxABvUQ_30000`: chốt = `applause_cheering` 0,00–3,05s; `speech_normal` chạy suốt
  **0,00–10,00s** (chồng lấn hoàn toàn với applause, không phải bắt đầu sau khi applause
  dứt như cả lần 1 [3,21s] lẫn lần 2 [7,23s] đều ghi).

**Quyết định:** Chốt theo bộ số ở trên cho cả hai file — một đáp án thứ ba, không trùng
hoàn toàn với lần 1 hay lần 2.

**Lý do:** §4 của `annotation_guideline.md` đã có quy tắc chung "gán tất cả các lớp cùng
xảy ra", nhưng cả hai lần gán trước đều KHÔNG áp dụng đúng quy tắc này cho riêng cặp
`speech_normal`/`applause_cheering` — có thể vì tiếng vỗ tay to át hẳn cảm nhận "còn nghe
được giọng nói", dẫn tới quán tính coi chúng loại trừ nhau dù quy tắc đã ghi rõ không phải
vậy.

**Áp dụng chung:** Thêm một dòng nhắc cụ thể vào §4 — khi gán `applause_cheering` (hoặc
bất kỳ sự kiện to nào), chủ động nghe lại xem còn giọng nói nền hay không trước khi coi
`speech_normal` đã kết thúc, thay vì mặc định tuần tự. Đã thêm vào
`annotation_guideline.md` §4 (xem file, mục cập nhật 22/09).

---

### 2026-09-22 (nghe lần ba) · `as_strong_03bZUQDFJU8_30000.wav` — chốt nguyên vẹn theo lần 1

**Tình huống:** Lần 2 (mù) gộp mất một khoảng lặng: chỉ gán `speech_normal` 0,00–1,25s rồi
`applause_cheering` liên tục 1,44–10,00s. Lần 1 có thêm một đoạn `speech_normal` xen giữa
(7,45–8,05s) và tách `applause_cheering` thành hai đợt (1,47–7,00s và 8,72–10,00s).

**Quyết định:** Chốt theo **nguyên vẹn lần 1** (4 sự kiện, đủ cả khoảng lặng giữa hai đợt
vỗ tay).

**Lý do:** Nghe lần ba xác nhận có một khoảng ngắn giọng nói xen giữa hai đợt vỗ tay —
lần 2 (mù) đã gộp nhầm thành một khối liên tục, bỏ sót đoạn giọng nói xen giữa.

---

### 2026-09-22 (nghe lần ba) · `as_strong_0yZGysisqY0_12000.wav` (tiếp) — phán quyết "đồ chơi" ĐƯỢC CỦNG CỐ, lần 2 (mù) bị đánh lừa

**Tình huống:** Case đã ghi ở mục **2026-09-22 (tiếp) · âm thanh đồ chơi**: lần 1 nghe ra
đây là đồ chơi mô phỏng, không gán sự kiện nào. Lần 2 (mù, không còn nhớ ngữ cảnh) lại gán
`alarm_bell` + `explosion` + `alarm_bell` — ba sự kiện "thừa" tuyệt đối. Nghe lần ba, người
gán mô tả lại: 0,00–6,47s và 8,12–10,00s là **"tiếng đồ chơi trẻ em"**, đoạn giữa
7,06–8,10s là **"tiếng nổ gì đó"** — không chắc chắn, nhưng nằm giữa hai đoạn đã xác định
là đồ chơi.

**Quyết định:** **Giữ nguyên phán quyết của lần 1** — không gán bất kỳ dòng nào vào TSV
cho cả ba đoạn, kể cả đoạn giữa. File vẫn ở trong `gold_test`, coi là "không có sự kiện
thuộc 16 lớp".

**Lý do:** Áp dụng nhất quán nguyên tắc đã lập ở case này (và ở ca chuông quầy phục vụ
15/09): khi đã xác định một clip là đồ chơi/mô phỏng, MỌI đoạn âm thanh trong cùng clip đó
— kể cả đoạn nghe "giống" sự kiện thật hơn — vẫn thuộc cùng một nguồn mô phỏng trừ khi có
bằng chứng NGƯỢC LẠI rõ ràng. "Tiếng nổ gì đó" (mô tả không chắc chắn) không đủ là bằng
chứng ngược lại — tách riêng một đoạn giữa hai đoạn đồ chơi đã xác định để gán `explosion`
thật sẽ là suy diễn, không phải nghe thấy.

**Áp dụng chung — phát hiện quan trọng nhất của cả đợt nghe lần ba:** đồ chơi/âm thanh mô
phỏng CÓ THỂ đánh lừa ngay cả CHÍNH người gán đã từng nhận ra nó, khi nghe lại trong điều
kiện MÙ (không còn ngữ cảnh video/lần gán trước). Đây là một giới hạn thật của quy trình
gán mù bằng tai, cần ghi vào phần Hạn chế của khoá luận: quy tắc "đồ chơi mô phỏng" dựa
một phần vào ngữ cảnh (video, trực giác lúc xem lần đầu) mà điều kiện mù audio-only không
tái tạo được — tự-nhất-quán đo được ở DATA_PLAN §8 vì vậy có thể ĐÁNH GIÁ THẤP độ tin cậy
thật của các quyết định "đồ chơi", không phải đánh giá cao.

---
