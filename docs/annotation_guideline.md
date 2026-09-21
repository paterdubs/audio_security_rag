# annotation_guideline.md — Quy ước gán nhãn

> Đối soát **18/09/2026**: lô synthetic mới đã PASS hợp đồng nhưng chưa có gold annotations/pilot được lưu trong repo;
> chưa coi guideline đã nghiệm thu và đóng băng. Trạng thái ở [STATUS.md](STATUS.md).
> JAMS Scaper ghi biên đặt nguồn, không mặc định trùng onset/offset nghe được.
> Lô legacy còn cắt event quanh 1 s và bắt đầu nguồn tại 0; không lấy các biên đó làm gold.
> Biên của lô mới tốt hơn theo hợp đồng Scaper, nhưng vẫn là biên tổng hợp — không thay thế onset/offset
> do người nghe gán mù trên gold.
> 1.831 lượt bulk-accept foreground không thay thế nghe duyệt hoặc gán gold mù.
> Ruling chuông quầy phục vụ trong [decision_log](../data/gold/decision_log.md) vẫn chờ người xác nhận.

> **Vai trò:** trả lời câu hỏi *"đặt onset/offset ở đâu"*. Câu hỏi *"clip này thuộc lớp nào"* tra ở [taxonomy.md](taxonomy.md).
>
> Liên quan: [DATA_PLAN §4](DATA_PLAN.md) (quy trình bank) · [DATA_PLAN §8](DATA_PLAN.md) (gold test set)
>
> ⚠️ **File này là tài liệu sống trong giai đoạn pilot, và ĐÓNG BĂNG sau đó.** Mọi sửa đổi sau khi bắt đầu gán đại trà đều làm hỏng tính nhất quán của phần đã gán. Sửa sau pilot → phải gán lại từ đầu.

---

## 0. Vì sao file này quan trọng hơn bình thường

Dự án chỉ có **một người gán nhãn**. Không có người thứ hai để tranh luận khi gặp ca mơ hồ, và cũng không có ai phát hiện khi tiêu chí của bạn **trôi dần** qua các buổi làm việc.

File này là phòng tuyến duy nhất chống trôi tiêu chí. Nên:

- Khi gặp ca không có trong file này → **ghi vào [§8 Nhật ký quyết định](#8-nhật-ký-quyết-định)**, quyết một lần, rồi áp dụng y hệt về sau.
- Không "cảm thấy" khác đi giữa chừng. Nếu thật sự cần đổi tiêu chí, dừng lại, sửa file này, và **gán lại phần đã làm** theo tiêu chí mới.

---

## 0.1 Điều kiện tiên quyết — audio có phải ghi âm THẬT không

Phát hiện ở pilot lần 1 (22/09), xem `data/gold/decision_log.md` mục cùng ngày.

`DATA_PLAN.md` §8 định nghĩa gold_test là *"audio THẬT"*, nhưng chưa từng nói rõ **thế
nào là không thật**. AudioSet lấy nhãn từ audio track của video YouTube; quy trình kiểm
định của AudioSet chỉ xác nhận *"loại âm thanh X có nghe thấy"*, **không** phân biệt bản
ghi âm một sự kiện thật ngoài đời với hiệu ứng âm thanh của trò chơi điện tử, phim ảnh,
hay dàn dựng — hai loại có thể cho sóng âm gần giống hệt nhau (game làm hiệu ứng vỡ kính
nghe y như kính vỡ thật) nhưng **không** mang đặc trưng ngữ cảnh thật (dội âm phòng thật,
tạp âm nền thật, động lực học tự nhiên) mà mô hình cần học để hoạt động đúng lúc triển
khai.

**Trước khi định nghĩa onset/offset của bất kỳ sự kiện nào, tự hỏi: nội dung TOÀN CLIP có
rõ ràng là trò chơi điện tử / phim ảnh / hiệu ứng dàn dựng không?**

| Dấu hiệu (nghe được, không cần xem hình) | |
|---|---|
| Giọng bình luận kiểu người chơi game, tiếng UI/menu, nhạc nền đặc trưng game | Loại |
| Động lực học âm thanh nén phẳng bất thường, thiếu tạp âm môi trường xung quanh | Loại |
| Âm thanh có ngữ cảnh vật lý hợp lý (phòng, ngoài trời, tiếng người thật xen kẽ) | Giữ |

**Nếu đúng là game/phim/dàn dựng:** loại **CẢ CLIP**, không gán bất kỳ dòng nào (kể cả
những đoạn `speech_normal` nghe có vẻ thật xen giữa — cả clip đã mất tư cách "audio thật"
của gold_test). Ghi vào `data/manifests/exclusions.csv`, `reason_code=synthetic_or_game_audio`
(khác `unusable` — để đếm riêng được tỉ lệ nhiễm loại này trong nguồn).

**Nếu chỉ một phần clip là game/dàn dựng, phần còn lại rõ ràng là thật:** ghi vào
[§8 Nhật ký quyết định](#8-nhật-ký-quyết-định) trước khi gán — đây chưa có tiền lệ, không
tự quyết ngầm.

---

## 1. Định nghĩa onset và offset

| | Định nghĩa |
|---|---|
| **onset** | Thời điểm **đầu tiên nghe được** sự kiện, kể cả còn rất nhỏ |
| **offset** | Thời điểm sự kiện **chìm xuống ngang mức nền**, tính cả đuôi vọng |

**Đuôi vọng tính vào event.** Tiếng súng trong nhà để xe có thể vọng 0.5 s sau xung chính — offset đặt ở cuối đuôi vọng, không phải cuối xung. Lý do: mô hình học từ phổ, và đuôi vọng là một phần của cái mà mô hình nhìn thấy.

**Đừng cắt theo "chỗ nhìn đẹp trên waveform".** Quyết định bằng tai, dùng waveform và spectrogram để tinh chỉnh biên, không để chọn biên.

---

## 2. Độ chính xác cần đạt

| Yêu cầu | Giá trị |
|---|---|
| Sai số biên chấp nhận được | **± 100 ms** |
| Collar dùng khi chấm điểm | 200 ms |
| Thời lượng event tối thiểu | **100 ms** — ngắn hơn thì bỏ qua |

Sai số 100 ms là nửa collar. Cố gắng chính xác hơn mức đó là lãng phí thời gian — collar sẽ nuốt phần chênh lệch. Nhưng lệch **quá** 100 ms thì chính collar đang che lỗi gán nhãn, và mọi chỉ số sẽ đẹp hơn thực tế.

**Cách làm thực tế:** phóng to đủ để thấy khoảng 1 giây trên màn hình, nghe đi nghe lại **tối đa ba lần**, chốt. Nghe lần thứ tư trở đi không làm bạn chính xác hơn, chỉ làm bạn kém nhất quán hơn.

---

## 3. Gộp hay tách — quy tắc 0.5 giây

**Hai sự kiện cùng lớp cách nhau dưới 0.5 s → gộp thành MỘT event. Từ 0.5 s trở lên → tách.**

Áp dụng cho **mọi lớp**, không riêng tiếng súng. Đo khoảng cách từ **offset của sự kiện trước** đến **onset của sự kiện sau**.

| Tình huống | Cách gán |
|---|---|
| Loạt bắn tự động, các phát cách ~0.1 s | **Một** event `gunshot` kéo dài cả loạt |
| Ba phát súng cách nhau ~1.5 s | **Ba** event `gunshot` riêng |
| Pháo dây nổ lẹt đẹt liên tục 4 s | **Một** event `fireworks` |
| Chuỗi bước chạy | **Một** event `running_footsteps` cho cả chuỗi, không gán từng bước |
| Còi hú ngắt quãng theo chu kỳ < 0.5 s | **Một** event `siren` |
| Chuông báo cháy bíp 1 Hz (nghỉ ~0.9 s) | **Một** event — xem ngoại lệ bên dưới |

> **Ngoại lệ cho thiết bị lặp theo chu kỳ** (`alarm_bell`, `siren`): nếu đó rõ ràng là **một thiết bị đang kêu liên tục**, gán **một** event từ tiếng đầu đến tiếng cuối, kể cả khoảng nghỉ giữa các tiếng bíp vượt 0.5 s. Tách chỉ khi thiết bị thực sự **ngừng hẳn** rồi kêu lại (nghỉ > 3 s).

---

## 4. Sự kiện chồng nhau

**Gán tất cả các lớp cùng xảy ra.** Strong label cho phép chồng lấn, và việc ép chọn một lớp sẽ dạy mô hình sai.

Ví dụ một vụ va chạm điển hình:

```
   0.0        0.8              2.1        3.4      4.6
   |───────────|                |          |        |
   vehicle_crash ────────────────────┤              ← 0.8 – 2.6
              glass_breaking ────┤                   ← 1.4 – 2.6
                     scream ─────────────────┤       ← 1.9 – 3.4
```

Ba event, ba dòng trong TSV, biên thời gian độc lập nhau.

**Khi một sự kiện bị che lấp hoàn toàn** bởi sự kiện to hơn và bạn không còn nghe thấy nó nữa: **không gán**. Chỉ gán cái nghe được — mô hình cũng chỉ có chừng đó thông tin.

> ⚠️ **Bẫy thường gặp (phát hiện từ pilot lần 2, 22/09): `speech_normal` bị bỏ sót khi có
> `applause_cheering`.** Test–retest cho thấy xu hướng mặc định coi giọng nói KẾT THÚC ngay
> khi tiếng vỗ tay bắt đầu, dù giọng nói (đám đông, MC) vẫn tồn tại NỀN suốt bên dưới —
> đúng quy tắc "gán tất cả các lớp cùng xảy ra" ở trên nhưng dễ quên áp dụng khi một lớp to
> hơn hẳn. Khi gán `applause_cheering` (hoặc bất kỳ sự kiện to nào), chủ động nghe lại xem
> còn giọng nói nền hay không trước khi coi `speech_normal` đã kết thúc, thay vì mặc định
> tuần tự. Xem `decision_log.md` 22/09 cho ví dụ cụ thể.

---

## 5. Định dạng nhãn

TSV chuẩn DCASE, **đúng bốn cột, không thêm cột nào**:

```
filename	onset	offset	event_label
gold_0042.wav	1.412	2.605	glass_breaking
gold_0042.wav	1.903	3.387	scream
```

| Quy tắc | |
|---|---|
| Phân tách | Ký tự **tab**, không phải dấu phẩy |
| Đơn vị | Giây, **3 chữ số thập phân** |
| Sắp xếp | Theo `filename`, rồi theo `onset` |
| Clip không có sự kiện | **Không** có dòng nào trong TSV (không dùng dòng rỗng, không dùng nhãn `none`) |

> ⛔ **Tuyệt đối không thêm cột.** `sed_eval` và `psds_eval` đọc đúng bốn cột này. Metadata (người gán, lần gán, độ tự tin, cờ slice) để ở **file riêng**, khớp bằng `filename`.

---

## 6. Cờ slice (chỉ cho gold test set)

Mỗi clip gold được gắn các cờ mô tả *điều kiện* của nó, phục vụ báo cáo theo lát cắt. Ghi ở `data/gold/g1_metadata.csv`, **không** ghi vào TSV.

| Cờ | Gắn khi |
|---|---|
| `S1_overlap` | Có ≥ 2 event chồng lấn về thời gian |
| `S2_low_snr` | Sự kiện chỉ nhỉnh hơn nền một chút, phải chú ý mới nghe ra |
| `S3_reverb` | Vọng rõ rệt (hành lang, gara, giếng trời) |
| `S4_media_playback` | Âm thanh phát từ TV, loa, điện thoại |
| `S5_long` | Clip > 30 s |
| `S6_chain` | Có chuỗi sự kiện nhân quả (vd. kính vỡ → thét → bước chạy) |
| `S7_confusable` | Có mặt lớp Nhóm B dễ nhầm với Nhóm A trong cùng clip |
| `S8_multisource` | Nhiều nguồn âm ở các hướng/khoảng cách khác nhau |

Một clip có thể mang nhiều cờ. Cờ gắn **sau khi** đã gán xong nhãn sự kiện — đừng để việc nghĩ về slice ảnh hưởng tới biên thời gian.

---

## 7. Hai chế độ làm việc — đừng lẫn lộn

Đây là chỗ dễ sai nhất trong cả quy trình.

### 7.1 Chế độ BANK (foreground bank) — máy hỗ trợ ✅

Máy đề xuất lớp và biên; bạn duyệt. Bốn phím:

| Phím | Nghĩa |
|---|---|
| `A` | Nhận đề xuất y nguyên |
| `S` | Nhận lớp, **sửa biên** |
| `D` | Loại — **bắt buộc chọn mã lý do** |
| `F` | Đổi sang lớp khác |

**Khi lưỡng lự → loại.** Bank nhỏ mà sạch tốt hơn bank to mà lẫn.

### 7.2 Chế độ GOLD (test set) — máy bị CẤM ❌

Không đề xuất lớp, không đề xuất biên, không hiện kết quả mô hình, không hiện nhãn gốc của dataset nguồn.

**Khi lưỡng lự → phải quyết.** Ở đây "loại cho chắc" là sai: bỏ sót một sự kiện có thật sẽ làm chỉ số Recall sai lệch.

> **Thoả hiệp duy nhất được phép:** máy được tô vùng *"có năng lượng vượt nền"* — chỉ năng lượng, không phân lớp, không đề xuất biên chính xác. Nếu dùng thì **phải ghi vào báo cáo**.

### 7.3 Lần gán lại (test–retest)

Điều kiện mù bắt buộc, thiếu một điều là số đo mất giá trị:

1. Dự án Label Studio **mới hoàn toàn**, không import gì từ lần 1.
2. **Xáo thứ tự** clip và **đổi tên file thành mã băm**.
3. Khoảng nghỉ **≥ 3 ngày** (pilot) / **≥ 7 ngày** (đại trà).
4. Trộn clip gán lại vào giữa các clip khác, không để thành một khối.

Sau đó: nghe lại lần ba mọi ca bất đồng, chốt, và ghi vào §8.

---

## 8. Nhật ký quyết định

**Mọi ca không có sẵn trong tài liệu đều phải ghi vào đây**, kèm quyết định và lý do. Đây vừa là công cụ chống trôi tiêu chí, vừa là vật liệu trực tiếp cho chương phân tích lỗi của khoá luận.

File: `data/gold/decision_log.md`. Mỗi mục một đoạn:

```markdown
### 2026-09-22 · gold_0117.wav @ 3.2s
**Tình huống:** Tiếng thét chuyển dần thành tiếng khóc, không có ranh giới rõ.
**Quyết định:** offset đặt ở chỗ mất sắc thái thét (3.95s); phần khóc không gán.
**Lý do:** taxonomy A3 loại tiếng khóc; chọn điểm mất sắc thái vì đó là dấu hiệu
nghe được, còn "chỗ biên độ giảm" thì tuỳ ý.
**Áp dụng chung:** mọi ca thét→khóc về sau xử lý y hệt.
```

Ba câu hỏi phải trả lời trong mỗi mục: *tình huống gì · quyết thế nào · vì sao chọn thế*.

---

## 9. Kỷ luật làm việc

Gán nhãn strong là việc dễ bị trôi chất lượng theo mệt mỏi, và chất lượng giảm thì **không có ai phát hiện giúp**.

| Quy tắc | Lý do |
|---|---|
| **Tối đa 45 phút mỗi phiên**, nghỉ ≥ 10 phút | Tai mỏi thì biên bắt đầu lệch hệ thống |
| **Không quá 3 giờ/ngày** cho việc gán gold | Quá mức này thì chất lượng giảm nhanh hơn số lượng tăng |
| **Đầu mỗi phiên: đọc lại §3 và bảng tra nhanh** của taxonomy | Chống trôi giữa các ngày |
| **Dùng tai nghe kín, cùng một cặp, cùng một mức âm lượng** | Loa ngoài làm mất chi tiết tần số cao (đuôi kính vỡ) và mất event nhỏ |
| **Đầu mỗi phiên gán lại 3 clip của phiên trước** | Cảnh báo sớm: lệch nhiều nghĩa là tiêu chí đang trôi |

Ba clip kiểm tra đầu phiên **không** tính vào thống kê test–retest chính thức — đó là công cụ tự theo dõi, không phải số đo báo cáo.

---

## 10. Checklist trước khi bắt đầu gán đại trà

- [ ] Đã đọc hết [taxonomy.md](taxonomy.md), đặc biệt bảng tra nhanh §3
- [ ] Đã làm pilot 30 clip **lần 1**
- [ ] Đã nghỉ ≥ 3 ngày và làm pilot **lần 2** trong điều kiện mù
- [ ] Tự-nhất-quán trên pilot **≥ 0.75**, onset lệch trung vị **≤ 100 ms**
- [ ] Đã sửa tài liệu theo những gì pilot phát hiện, và **ghi lại đã sửa gì**
- [ ] `data/gold/decision_log.md` đã tồn tại và có các ca từ pilot
- [ ] Đã chốt cấu hình tai nghe và mức âm lượng
- [ ] Dự án Label Studio chế độ GOLD đã tắt mọi gợi ý của máy

Không đạt mục nào thì **chưa** bắt đầu. Gán 1 giờ audio bằng tài liệu mơ hồ rồi phát hiện tự-nhất-quán 0.5 là mất trắng 6 giờ công.
