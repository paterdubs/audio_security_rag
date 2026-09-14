# taxonomy.md — Định nghĩa 15 lớp âm thanh

> **Vai trò:** định nghĩa chuẩn của 15 lớp. Mọi tranh cãi "clip này thuộc lớp nào" đều tra ở đây.
>
> Liên quan: [ontology_map.yaml](../ml/configs/ontology_map.yaml) (ánh xạ sang nhãn dataset — **file máy đọc**) · [annotation_guideline.md](annotation_guideline.md) (cách gán biên thời gian) · [DATA_PLAN.md](DATA_PLAN.md)
>
> ⚠️ File này và `ontology_map.yaml` **phải luôn khớp nhau**. Sửa một bên thì sửa bên kia trong cùng một commit.

---

## 0. Cách đọc

Mỗi lớp có bốn phần, và phần **❌ Không bao gồm** mới là phần quan trọng nhất — định nghĩa một lớp chủ yếu là nói nó *không* phải cái gì.

Mỗi lớp còn có **dấu hiệu phân biệt**: câu trả lời ngắn cho cặp dễ nhầm nhất của nó. Khi gán nhãn mà lưỡng lự quá 10 giây, tra dòng đó rồi quyết, đừng nghe đi nghe lại — nghe lại nhiều lần chỉ làm quyết định kém nhất quán hơn.

**Vì sao chia hai nhóm.** Nhóm A sinh cảnh báo, Nhóm B thì không. Nhóm B tồn tại **chính xác vì** nó dễ bị nhầm thành Nhóm A: pháo hoa nghe như súng, chén vỡ nghe như kính vỡ, reo hò nghe như la hét. Không có Nhóm B thì hệ thống báo động nhầm liên tục, và ta cũng không đo được tỉ lệ báo nhầm.

**Nguyên tắc bao trùm khi lưỡng lự:** nếu không chắc một clip có thuộc lớp Nhóm A hay không, **loại nó khỏi foreground bank** (mã `wrong_class`). Bank nhỏ mà sạch tốt hơn bank to mà lẫn. Ngoại lệ: ở gold test set thì không được "loại cho chắc" — ở đó phải quyết, vì bỏ sót sự kiện làm sai chỉ số Recall.

---

## 1. Nhóm A — Sự kiện an ninh (10 lớp)

### A1. `gunshot` — Tiếng súng

Tiếng nổ đầu nòng của vũ khí có nòng. Đặc trưng: xung cực nhọn, thời gian lên gần như tức thời, thường kèm đuôi vọng.

| | |
|---|---|
| ✅ **Bao gồm** | Súng ngắn, súng trường, súng săn · Loạt bắn tự động · Tiếng súng có vọng trong không gian kín |
| ❌ **Không bao gồm** | Pháo, pháo hoa → `fireworks` · Nổ bom, nổ bình gas → `explosion` · Cửa đóng sầm → `door_slam` · Súng đồ chơi nhựa, súng nước → loại (`wrong_class`) · Tiếng súng trong phim/game phát qua loa → loại khỏi bank (`wrong_class`), vì bank cần sự kiện thật; nhưng ở **gold test** thì vẫn gán `gunshot` kèm cờ slice S4 |
| 🔍 **Dấu hiệu phân biệt** | **so với `fireworks`:** pháo hoa hầu như luôn đi **thành chuỗi nhiều tiếng** rải rác 1–3 giây, có tiếng rít bay lên trước, và đuôi vang dài hơn. Súng đơn phát thì gọn, khô, dứt điểm. **Một tiếng nổ đơn độc, khô, không có tiếng rít trước → `gunshot`.** |
| 📌 **Ví dụ** | (1) Một phát súng ngắn trong bãi đỗ xe, có vọng ~0.4 s (2) Loạt 5 phát liên thanh cách nhau ~0.1 s (3) Tiếng súng săn ngoài trời, không vọng |

> **Loạt bắn gán thế nào?** Xem [annotation_guideline.md §3](annotation_guideline.md) — quy ước thống nhất là **gộp thành một event** nếu khoảng cách giữa các phát < 0.5 s, tách nếu ≥ 0.5 s. Đừng tự quyết khác đi giữa chừng.

---

### A2. `explosion` — Tiếng nổ

Tiếng nổ thể tích lớn: áp suất giải phóng đột ngột, năng lượng dồn ở tần số thấp, đuôi ầm ì kéo dài.

| | |
|---|---|
| ✅ **Bao gồm** | Nổ bom, nổ mìn · Nổ bình gas, nổ bình khí · Nổ do cháy · Pháo cỡ lớn dùng như vũ khí |
| ❌ **Không bao gồm** | Súng → `gunshot` · Pháo hoa lễ hội → `fireworks` · **Bóng bay nổ, nắp chai bật, túi ni lông nổ** → loại (`wrong_class`) · Sấm → loại (nền thời tiết) |
| 🔍 **Dấu hiệu phân biệt** | **so với `gunshot`:** tiếng nổ có **năng lượng tần số thấp rõ rệt** (nghe "ầm" chứ không "chát") và đuôi dài hơn nhiều, thường > 1 s. Súng nghe nhọn và cao hơn. |
| 📌 **Ví dụ** | (1) Nổ bình gas trong nhà, rung cả micro (2) Nổ ngoài trời xa, ầm kéo dài ~2 s (3) Nổ kèm tiếng vỡ kính theo sau — **gán hai event chồng nhau**, không gán một |

> ⚠️ **Lớp khan hiếm.** Đây là một trong ba lớp đỏ. Đừng nới định nghĩa để gom thêm clip cho đủ số — thà báo cáo `explosion` chỉ có 60 clip còn hơn có 120 clip mà một nửa là bóng bay nổ.

---

### A3. `scream` — Tiếng thét

Tiếng phát thanh của người ở cường độ cực đại, **không thành lời**, mang sắc thái hoảng sợ hoặc đau đớn. Thanh quản bị ép, phổ âm méo và giàu hài bậc cao.

| | |
|---|---|
| ✅ **Bao gồm** | Thét vì sợ hãi · Thét vì đau · Thét chói không rõ lời của người lớn hoặc trẻ em |
| ❌ **Không bao gồm** | Quát tháo, gọi to **có lời** → `shout_yell` · Reo hò, cười to, hét vì vui → `laughter_cheering` · Trẻ con nô đùa la hét ở sân trường → `laughter_cheering` · Trẻ sơ sinh khóc → loại (`wrong_class`) |
| 🔍 **Dấu hiệu phân biệt** | **so với `shout_yell`:** hỏi *"có nghe ra từ ngữ không?"* — nghe ra từ thì là `shout_yell`, chỉ là âm thanh thuần thì là `scream`. **so với `laughter_cheering`:** hỏi *"sắc thái là sợ hay vui?"* — nếu nghe thấy tiếng cười xen kẽ hoặc nhiều người cùng lúc thì gần như chắc chắn là vui. |
| 📌 **Ví dụ** | (1) Một tiếng thét nữ chói, ~1.2 s, không thành lời (2) Thét đau ngắn sau tiếng va đập (3) Thét kéo dài rồi chuyển thành khóc — gán phần thét, dừng ở chỗ chuyển |

> Đây là ranh giới **bất đồng nhiều nhất**. Vì dự án chỉ có một người gán, cặp `scream` ↔ `shout_yell` gần như chắc chắn sẽ là cặp có tự-nhất-quán thấp nhất — dự kiến trước và báo cáo riêng theo lớp ([DATA_PLAN §8.2](DATA_PLAN.md) biện pháp B3).

---

### A4. `glass_breaking` — Kính vỡ

Tiếng vật liệu giòn vỡ vụn: chuỗi xung ngắn, năng lượng tập trung ở tần số cao, thường có đuôi lách tách khi mảnh rơi.

| | |
|---|---|
| ✅ **Bao gồm** | Cửa kính, cửa sổ vỡ · Chai lọ thuỷ tinh vỡ · Kính xe vỡ · Tiếng mảnh vỡ rơi tiếp sau (cùng một event) |
| ❌ **Không bao gồm** | Chén đĩa sứ rơi vỡ → `object_drop_dishes` · Rót nước vào ly, ly chạm nhau → loại · Bẻ gỗ, bẻ nhựa → loại · Kính vỡ **trong** một vụ va chạm xe → gán **cả hai** event chồng nhau |
| 🔍 **Dấu hiệu phân biệt** | **so với `object_drop_dishes`:** kính vỡ có **đuôi tần số cao rất dài** (mảnh nhỏ tiếp tục rơi và trượt), còn sứ vỡ thì "cộp" hơn, tắt nhanh hơn, ít lách tách. Nếu nghe rõ tiếng "leng keng" kéo dài > 0.5 s thì là kính. |
| 📌 **Ví dụ** | (1) Cửa sổ bị đập vỡ, mảnh rơi trong 1.5 s (2) Chai bia vỡ trên sàn bê tông (3) Kính xe vỡ khi va chạm — gán cùng lúc với `vehicle_crash` |

---

### A5. `vehicle_crash` — Va chạm xe

Tiếng va chạm giữa phương tiện với phương tiện hoặc vật cản: xung kim loại biến dạng, thường kèm tiếng phanh gấp, kính vỡ, còi.

| | |
|---|---|
| ✅ **Bao gồm** | Va chạm xe hơi, xe máy · Xe đâm vào vật cản cố định · Tiếng phanh rít **dẫn ngay đến** va chạm |
| ❌ **Không bao gồm** | Phanh rít **không** dẫn đến va chạm → loại · Đóng cửa xe → `door_slam` · Tiếng động cơ, còi xe đơn thuần → `ambient_noise` · Vật kim loại rơi → `object_drop_dishes` |
| 🔍 **Dấu hiệu phân biệt** | **so với `glass_breaking` và `door_slam`:** va chạm xe là **sự kiện phức hợp** — hiếm khi chỉ có một tiếng. Nếu chỉ nghe thấy đúng một tiếng "rầm" đơn lẻ không có bối cảnh giao thông thì **không** gán lớp này. |
| 📌 **Ví dụ** | (1) Phanh rít 0.8 s rồi va chạm rồi kính vỡ (2) Xe máy đổ, kim loại trượt trên đường (3) Va chạm nhẹ ở bãi đỗ, chỉ một tiếng "cộp" kim loại **và có tiếng động cơ nền** |

> 🔴 **Lớp rủi ro cao nhất.** Đã xác minh: AudioSet **không có** lớp nào nghĩa là va chạm xe, nên lớp này phụ thuộc hoàn toàn vào MIVIA Road. Nếu đơn không được duyệt, lớp này ra [cổng quyết định D7](DATA_PLAN.md#11-cổng-quyết-định-d7--phân-tích-thiếu-hụt).

---

### A6. `shout_yell` — Quát, gọi to

Tiếng nói của người ở cường độ cao **có nội dung lời nói**: cãi vã, ra lệnh, gọi từ xa, doạ nạt.

| | |
|---|---|
| ✅ **Bao gồm** | Cãi nhau to tiếng · Quát tháo, ra lệnh · Gọi to từ xa · Doạ nạt |
| ❌ **Không bao gồm** | Thét không thành lời → `scream` · Reo hò cổ vũ → `laughter_cheering` · Trẻ con la hét khi chơi → `laughter_cheering` · Nói chuyện bình thường, kể cả to → `speech_normal` |
| 🔍 **Dấu hiệu phân biệt** | **so với `speech_normal`:** ranh giới là **cường độ và độ căng của giọng**, không phải nội dung. Người nói to trong phòng ồn vẫn là `speech_normal`; giọng căng, vỡ, gấp gáp mới là `shout_yell`. **so với `scream`:** nghe ra từ ngữ → `shout_yell`. |
| 📌 **Ví dụ** | (1) Hai người cãi nhau, giọng căng, ngắt quãng (2) Một người gọi to tên ai đó qua bãi xe (3) Giọng ra lệnh gắt: "Đứng lại!" |

> 🔒 **Quyền riêng tư:** lớp này chỉ ghi nhận *có tiếng quát*, **tuyệt đối không** phiên âm nội dung, không nhận dạng người nói. Xem [SYSTEM.md §1.4](SYSTEM.md).

---

### A7. `door_slam` — Cửa đóng sầm

Cửa đóng mạnh: một xung duy nhất, năng lượng thấp-trung, kèm cộng hưởng của khung cửa và tường.

| | |
|---|---|
| ✅ **Bao gồm** | Cửa gỗ, cửa sắt đóng mạnh · Cửa xe đóng mạnh · Cửa bị gió đập |
| ❌ **Không bao gồm** | **Gõ cửa** → loại (`wrong_class`) · **Cửa kêu cọt kẹt** → loại · Cửa đóng nhẹ, nghe được nhưng không "sầm" → loại · Đồ vật rơi → `object_drop_dishes` |
| 🔍 **Dấu hiệu phân biệt** | **so với `gunshot`:** cửa đóng sầm có **đuôi cộng hưởng tần số thấp** (khung cửa rung) và thời gian lên chậm hơn rõ rệt. Súng thì nhọn và khô. **so với `object_drop_dishes`:** cửa là một xung to, đồ rơi thường có tiếng nảy tiếp theo. |
| 📌 **Ví dụ** | (1) Cửa sắt cầu thang đóng sầm, vọng hành lang (2) Cửa xe hơi đóng mạnh ngoài bãi (3) Gió đập cửa sổ |

> ⚠️ **ESC-50 không dùng được cho lớp này.** Hai lớp cửa của ESC-50 là gõ cửa và cọt kẹt, cả hai đều bị loại theo định nghĩa trên.

---

### A8. `running_footsteps` — Tiếng chạy

Chuỗi bước chân **nhịp nhanh, lực mạnh** — dấu hiệu của người chạy, không phải người đi.

| | |
|---|---|
| ✅ **Bao gồm** | Chạy trên sàn cứng, hành lang, cầu thang · Nhiều người cùng chạy · Chạy rồi dừng đột ngột |
| ❌ **Không bao gồm** | **Đi bộ bình thường** → loại (`wrong_class`) · Đi giày cao gót đi chậm → loại · Vật kéo lê trên sàn → loại |
| 🔍 **Dấu hiệu phân biệt** | Đếm nhịp: **chạy ≈ trên 2.5 bước/giây**, đi bộ ≈ 1.5–2 bước/giây. Kèm theo: bước chạy có lực va đập mạnh hơn và tiếng "bịch" rõ hơn. Khi lưỡng lự, đếm số bước trong 2 giây. |
| 📌 **Ví dụ** | (1) Một người chạy qua hành lang gạch, ~8 bước trong 2.5 s (2) Hai người chạy xuống cầu thang (3) Chạy nhanh rồi phanh gấp bằng giày thể thao (tiếng rít cuối) |

> ⚠️ **Tỉ lệ loại ở lớp này sẽ cao bất thường** — các nguồn công khai đặt nhãn "footsteps" cho cả đi lẫn chạy, mà phần lớn là **đi**. Thấy loại nhiều thì đó là quy trình đang chạy đúng, không phải hỏng.

---

### A9. `siren` — Còi hú

Còi hú của xe ưu tiên hoặc báo động dân sự: **âm cao, biến thiên tần số theo chu kỳ**, kéo dài nhiều giây.

| | |
|---|---|
| ✅ **Bao gồm** | Còi cứu thương, cứu hoả, cảnh sát · Còi báo động dân sự · Còi hú xa, nhỏ nhưng nhận ra được |
| ❌ **Không bao gồm** | Còi xe thông thường (bấm còi) → `ambient_noise` · Chuông báo cháy trong nhà → `alarm_bell` · Còi báo động xe (car alarm) → `alarm_bell` |
| 🔍 **Dấu hiệu phân biệt** | **so với `alarm_bell`:** còi hú **quét tần số lên xuống liên tục** (nghe "u-u-u" trượt), còn chuông báo thì **lặp một cao độ cố định** hoặc đổi qua lại giữa hai cao độ rời rạc. Nghe 3 giây là phân biệt được. |
| 📌 **Ví dụ** | (1) Xe cứu thương chạy qua, hiệu ứng Doppler rõ (2) Còi cảnh sát đứng yên, chu kỳ đều (3) Còi hú xa, lẫn trong ồn giao thông |

---

### A10. `alarm_bell` — Chuông, báo động

Thiết bị phát tín hiệu cảnh báo lặp lại: chuông báo cháy, báo khói, báo động xe, chuông cửa, đồng hồ báo thức, chuông nhà thờ.

| | |
|---|---|
| ✅ **Bao gồm** | Chuông báo cháy, báo khói · Báo động xe · Chuông cửa · Đồng hồ báo thức · Chuông nhà thờ, chuông trường |
| ❌ **Không bao gồm** | Còi hú xe ưu tiên → `siren` · Chuông điện thoại → loại · Nhạc chuông có giai điệu → loại (`music_overlay`) |
| 🔍 **Dấu hiệu phân biệt** | **so với `siren`:** cao độ **cố định hoặc nhảy rời rạc**, không trượt liên tục. Ngoài ra chuông thường có chu kỳ bật/tắt đều đặn. |
| 📌 **Ví dụ** | (1) Báo cháy toà nhà, bíp đều 1 Hz (2) Báo động xe kêu trong bãi đỗ (3) Chuông trường báo hết tiết |

---

## 2. Nhóm B — Lớp gây nhầm lẫn và nền (5 lớp)

> **Đọc kỹ:** các lớp này **không sinh cảnh báo**. Chúng có mặt trong taxonomy để model học cách *không* báo động. Gán nhãn cẩu thả ở Nhóm B sẽ trực tiếp làm tăng tỉ lệ báo động nhầm — đây không phải phần phụ.

### B1. `fireworks` — Pháo, pháo hoa

| | |
|---|---|
| ✅ **Bao gồm** | Pháo hoa lễ hội · Pháo dây, pháo tép · Pháo bông · Tiếng rít bay lên kèm tiếng nổ |
| ❌ **Không bao gồm** | Súng → `gunshot` · Nổ bom, nổ gas → `explosion` |
| 🔍 **Dấu hiệu phân biệt** | **Ba dấu hiệu, có một là đủ:** (a) tiếng **rít bay lên** trước khi nổ (b) **nhiều tiếng rải rác** trong vài giây, không đều nhịp (c) tiếng **vọng dội kéo dài** đặc trưng ngoài trời. Súng thường thiếu cả ba. |
| 📌 **Ví dụ** | (1) Pháo hoa giao thừa, nhiều tiếng chồng nhau (2) Pháo dây nổ lẹt đẹt liên tục 4 s (3) Một quả pháo đơn có tiếng rít trước |

> 🟡 Đây là **cặp nhầm lẫn nguy hiểm nhất của cả hệ thống**: `fireworks` bị nhận nhầm thành `gunshot` sẽ tạo báo động giả, còn ngược lại thì bỏ lọt sự kiện thật. Dữ liệu quốc tế có thể khác pháo Việt Nam về phổ → đã xếp vào buổi thu thực địa số 2.

### B2. `object_drop_dishes` — Đồ vật rơi, chén đĩa

| | |
|---|---|
| ✅ **Bao gồm** | Chén đĩa sứ rơi, vỡ · Xoong nồi va đập · Đồ vật cứng rơi xuống sàn · Ghế đổ |
| ❌ **Không bao gồm** | Kính vỡ → `glass_breaking` · Cửa đóng sầm → `door_slam` · Bước chân → `running_footsteps` |
| 🔍 **Dấu hiệu phân biệt** | Thường có **tiếng nảy tiếp theo** (vật rơi nảy lên rơi lại), và đuôi tắt nhanh hơn kính vỡ nhiều. |
| 📌 **Ví dụ** | (1) Chồng đĩa sứ rơi trong bếp (2) Cái xoong rơi xuống sàn gạch, nảy hai lần (3) Ghế nhựa đổ |

### B3. `laughter_cheering` — Cười, reo hò

| | |
|---|---|
| ✅ **Bao gồm** | Cười một người hoặc nhiều người · Reo hò, cổ vũ · Vỗ tay · **Trẻ con la hét khi nô đùa** |
| ❌ **Không bao gồm** | Thét sợ hãi → `scream` · Quát tháo → `shout_yell` · **Đám đông chung chung không rõ sắc thái** → `ambient_noise` |
| 🔍 **Dấu hiệu phân biệt** | Hỏi *"nhiều người hay một người, và sắc thái vui hay sợ?"* — **nhiều giọng chồng nhau + có nhịp lặp lại (cười thành tràng, hò thành đợt) → `laughter_cheering`**. Thét sợ hãi thường là một giọng và không lặp nhịp. |
| 📌 **Ví dụ** | (1) Tràng cười của nhóm người (2) Đám đông reo hò sau bàn thắng (3) Trẻ con chạy nhảy la hét ở sân trường |

> 📌 **Quyết định taxonomy có chủ ý:** "trẻ con la hét" xếp vào đây chứ không vào `scream`. Sân trường là một trong bốn khu vực triển khai; nếu trẻ con nô đùa bị tính là tiếng thét thì hệ thống sẽ báo động cả ngày và vô dụng.

### B4. `speech_normal` — Tiếng nói bình thường

| | |
|---|---|
| ✅ **Bao gồm** | Hội thoại · Nói một mình · Nói qua điện thoại · Nói to trong môi trường ồn |
| ❌ **Không bao gồm** | Quát tháo, giọng căng → `shout_yell` · Thét → `scream` · Hát → loại (`music_overlay`) · Tiếng nói từ TV/loa → `media_playback` (slice S4) |
| 🔍 **Dấu hiệu phân biệt** | Xem A6. Tiêu chí là **độ căng của giọng**, không phải âm lượng. |
| 📌 **Ví dụ** | (1) Hai người trò chuyện bình thường (2) Một người nói to để át tiếng máy (3) Nói chuyện điện thoại khi đi bộ |

> 🔒 **Quyền riêng tư — ràng buộc cứng:** chỉ phát hiện *có tiếng nói*. Không ASR, không nhận dạng người nói, không lưu nội dung dưới dạng văn bản. Khi thu thực địa tại IUH phải có đồng ý và ghi vào `data_inventory.md`.
>
> Ưu tiên clip **đa ngôn ngữ** để model không học "tiếng nói = tiếng Anh".

### B5. `ambient_noise` — Nền

| | |
|---|---|
| ✅ **Bao gồm** | Giao thông xa · Gió, mưa · Tiếng máy móc đều đều · Đám đông rì rầm không rõ sắc thái · Im lặng có nhiễu nền |
| ❌ **Không bao gồm** | **Bất kỳ sự kiện Nhóm A nào**, dù nhỏ hay ở xa |
| 🔍 **Dấu hiệu phân biệt** | Đây là lớp **mặc định**: cái gì không thuộc 14 lớp trên thì là nền. Nhưng chiều ngược lại phải chặt — một đoạn nền có lẫn dù chỉ một tiếng còi hú xa thì **không** được dùng làm nền. |
| 📌 **Ví dụ** | (1) Bãi đỗ xe lúc vắng, chỉ có ù nền (2) Hành lang trường giờ học (3) Nhà máy, tiếng máy đều |

> Lớp này **không lấy từ ontology** mà lấy từ background bank đã qua sàng lọc tự động với ngưỡng thấp (0.15) — thà loại nhầm nền sạch còn hơn để lọt một tiếng súng vào nền rồi Scaper dán nhãn "không có sự kiện" lên nó. Xem [DATA_PLAN §5](DATA_PLAN.md).

---

## 3. Bảng tra nhanh các cặp dễ nhầm

Dán bảng này cạnh màn hình khi gán nhãn.

| Cặp | Câu hỏi quyết định | Trả lời → lớp |
|---|---|---|
| `gunshot` ↔ `fireworks` | Có tiếng rít bay lên, hoặc nhiều tiếng rải rác, hoặc vọng dội dài? | Có → `fireworks` · Không → `gunshot` |
| `gunshot` ↔ `explosion` | Năng lượng dồn ở tần số thấp và đuôi > 1 s? | Có → `explosion` · Không → `gunshot` |
| `gunshot` ↔ `door_slam` | Có đuôi cộng hưởng khung cửa, thời gian lên chậm? | Có → `door_slam` · Không → `gunshot` |
| `scream` ↔ `shout_yell` | Nghe ra **từ ngữ** không? | Có → `shout_yell` · Không → `scream` |
| `scream` ↔ `laughter_cheering` | Nhiều giọng chồng nhau, có nhịp lặp? | Có → `laughter_cheering` · Không → `scream` |
| `shout_yell` ↔ `speech_normal` | Giọng **căng/vỡ/gấp**? (không xét âm lượng) | Có → `shout_yell` · Không → `speech_normal` |
| `glass_breaking` ↔ `object_drop_dishes` | Đuôi lách tách tần số cao kéo dài > 0.5 s? | Có → `glass_breaking` · Không → `object_drop_dishes` |
| `siren` ↔ `alarm_bell` | Cao độ **trượt liên tục** hay **cố định/nhảy rời rạc**? | Trượt → `siren` · Cố định → `alarm_bell` |
| `running_footsteps` ↔ loại | Trên 2.5 bước/giây? | Có → giữ · Không → loại (đi bộ) |
| `vehicle_crash` ↔ khác | Có bối cảnh giao thông và là sự kiện **phức hợp**? | Không → **không** gán lớp này |
| `explosion` ↔ `fireworks` | Có tiếng rít bay lên, hoặc nhiều tiếng rải rác không đều? | Có → `fireworks` · Không → `explosion` |
| `glass_breaking` ↔ `vehicle_crash` | Có tiếng động cơ/phanh làm bối cảnh không? | Có → gán **cả hai** · Không → chỉ `glass_breaking` |
| `door_slam` ↔ `object_drop_dishes` | Có tiếng **nảy lại** sau xung đầu tiên? | Có → `object_drop_dishes` · Không → `door_slam` |
| `object_drop_dishes` ↔ `running_footsteps` | Một-hai xung rời rạc, hay **chuỗi đều nhịp**? | Chuỗi đều → `running_footsteps` · Rời rạc → `object_drop_dishes` |
| `shout_yell` ↔ `laughter_cheering` | Nhiều giọng chồng nhau, có nhịp lặp, sắc thái vui? | Có → `laughter_cheering` · Không → `shout_yell` |

---

## 4. Quy ước xuyên suốt

1. **Sự kiện chồng nhau thì gán tất cả.** Một vụ va chạm có thể đồng thời là `vehicle_crash` + `glass_breaking` + `scream`. Strong label cho phép chồng lấn — đừng chọn một.
2. **Không có lớp "khác".** Clip không thuộc 15 lớp thì bị loại khỏi foreground bank kèm mã lý do, chứ không gán bừa.
3. **Âm thanh phát lại từ loa/TV** (`media_playback`) là **slice đánh giá S4**, không phải một lớp. Clip loại này được đánh dấu bằng `slice_flags`, và lớp âm thanh vẫn gán bình thường.
4. **Khoảng cách giữa hai event cùng lớp < 0.5 s → gộp một event.** Quy ước này áp dụng cho *mọi* lớp, không riêng `gunshot`.
5. **Nghi ngờ ở bank → loại. Nghi ngờ ở gold test → phải quyết.** Hai chỗ này có logic ngược nhau; nhầm lẫn giữa chúng sẽ làm hỏng chỉ số Recall.

---

## 5. Mục tiêu số lượng

| Lớp | Nhóm | Mục tiêu | Tối thiểu | Trạng thái |
|---|---|---:|---:|---|
| `gunshot` | A | 200 | 80 | ⏳ chờ MIVIA |
| `explosion` | A | 120 | 50 | 🔴 khan hiếm |
| `scream` | A | 200 | 80 | ⏳ chờ MIVIA |
| `glass_breaking` | A | 250 | 100 | 🟢 |
| `vehicle_crash` | A | 80 | 30 | 🔴 chờ MIVIA Road |
| `shout_yell` | A | 200 | 80 | 🟢 |
| `door_slam` | A | 250 | 100 | 🟢 |
| `running_footsteps` | A | 250 | 100 | 🟡 tỉ lệ loại cao |
| `siren` | A | 300 | 120 | 🟢 |
| `alarm_bell` | A | 250 | 100 | 🟢 |
| `fireworks` | B | 200 | 80 | 🟡 |
| `object_drop_dishes` | B | 250 | 100 | 🟢 |
| `laughter_cheering` | B | 250 | 100 | 🟢 |
| `speech_normal` | B | 300 | 120 | 🟢 |
| `ambient_noise` | B | — | — | dùng background bank |

Con số cập nhật tự động: `python scripts/coverage_report.py`. Lớp nào dưới mức tối thiểu ở mốc D7 sẽ ra [cổng quyết định](DATA_PLAN.md#11-cổng-quyết-định-d7--phân-tích-thiếu-hụt).
