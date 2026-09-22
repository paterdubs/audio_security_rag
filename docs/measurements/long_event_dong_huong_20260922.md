# Đóng hướng điều tra `long_event` — sáu giả thuyết bị loại, và vì sao dừng ở đây — 2026-09-22

Trang tổng kết, không phải phép đo mới. Gom lại một tuần điều tra
([long_event_postproc_20260919.md](long_event_postproc_20260919.md) →
[max_median_ceiling_20260919.md](max_median_ceiling_20260919.md) →
[long_event_gap_20260921.md](long_event_gap_20260921.md) →
[long_event_crosscut_20260921.md](long_event_crosscut_20260921.md) →
[low_snr_co_che_20260921.md](low_snr_co_che_20260921.md) →
[time_pool_ablation_20260922.md](time_pool_ablation_20260922.md)) và giải thích quyết định
**dừng đào sâu hướng này**.

---

## 1. Sáu giả thuyết, sáu lần bác bỏ

| # | Giả thuyết | Đo bằng | Kết quả |
|---|---|---|---|
| 1 | Cửa sổ lọc trung vị 7 khung quá hẹp cho sự kiện dài | ablation cửa sổ | Bác bỏ — nới cửa sổ không đóng khoảng cách |
| 2 | Trần `MAX_MEDIAN_FRAMES=51` chặn cửa sổ thích ứng | quét trần {51,101,201,401} | Bác bỏ — gap hẹp lại tối đa 6,4%, có lúc **rộng ra** |
| 3 | Khe hở năng lượng thật bên trong nhãn (Scaper chồng foreground có khoảng lặng) | so nhóm vỡ/không vỡ cùng lát cắt | Bác bỏ — khe hở giống nhau ở cả hai nhóm |
| 4 | `long_event` khó vì trùng lát cắt khó khác (`low_snr`/`reverb`/`overlap`/`causal_chain`) | bảng 2×2 cắt chéo | Bác bỏ — chênh lệch còn nguyên ở mọi ô, độ dài là biến độc lập |
| 5 | Cơ chế hỏng là Deletion (giống sự kiện thường) | phân rã 6 loại lỗi, v2+v3 | Tinh chỉnh — thật ra là **lệch biên**, không phải Deletion; và biên độ "gần như không đóng góp" của Deletion từng ghi là phóng đại (xác nhận lại trên θ\* của `pw1`, xem [phan_tich_lai_tren_pw1_20260922.md](phan_tich_lai_tren_pw1_20260922.md)) |
| 6 | Trường tiếp nhận thời gian của CNN14 quá ngắn — tăng độ phân giải (`time_pool_blocks` 3→2) sẽ giảm phân mảnh | ablation `time_pool_blocks`, đo lại ở hai chế độ hậu xử lý | **Bác bỏ** — `tpb2` không vỡ ít hơn `pw1` ở chế độ nào, thua thuần về F1 |

## 2. Phát hiện cuối cùng đảo ngược cả cách đo — đây là lý do thật sự để dừng

Ablation #6 đo lại ở chế độ hậu xử lý thích ứng-từ-train
([chuan_bao_cao_20260922.md](chuan_bao_cao_20260922.md)) cho một kết quả bất ngờ:

| | `pw1` (tpb=3) | `tpb2` (tpb=2) |
|---|---:|---:|
| `long_event` tỉ lệ vỡ | 0,2469 | 0,2540 |
| `long_event` F1 | 0,3602 | 0,3066 |

Hai model có tỉ lệ vỡ **gần như bằng nhau** (chênh 0,007) nhưng F1 `long_event` vẫn chênh
lớn (0,054, ~15% tương đối). Nếu phân mảnh là thứ giới hạn F1 trên sự kiện dài, hai con số
F1 đó phải gần nhau khi tỉ lệ vỡ gần nhau. **Chúng không gần nhau.**

Kết luận: **tỉ lệ phân mảnh — chỉ tiêu dự án dùng suốt tuần qua để đo mức độ hỏng của
`long_event` — không nắm được phần chính của vấn đề.** Một model có thể vỡ ít mà F1 vẫn
kém, tức có một thành phần lỗi khác (khả năng là gọi sai lớp, hoặc lệch biên nhẹ trong
collar cho phép nhưng vẫn đủ làm giảm overlap) không được tỉ lệ vỡ ghi nhận.

## 3. Vì sao dừng ở đây thay vì đi tìm chỉ tiêu mới

Ba lý do, cộng dồn:

1. **Chi phí đã trả không nhỏ.** Sáu giả thuyết, hai lượt train riêng cho ablation này
   (~4,5 giờ GPU thực tế, một phần chạy trong điều kiện GPU bị thảo nhiệt chặn xuống
   210/2100 MHz), cộng nhiều giờ phân tích lỗi CPU.
2. **Đóng góp chính của khoá luận chưa bắt đầu.** Deadline 09/11/2026, còn 7 tuần. W4/W5
   (Grounded AAC + bộ metric hallucination — đóng góp C1/C2) hiện ở ngày 0. `long_event`
   là một lát cắt phụ trong bảng báo cáo theo slice (SYSTEM.md §8.3), không phải mục tiêu
   của khoá luận.
3. **Cái đã có đủ để viết một chương phân tích lỗi tốt**, không cần thêm số. Sáu giả
   thuyết bị loại có hệ thống, mỗi cái có số đo và trang riêng, và kết luận trung thực —
   "độ dài sự kiện là biến khó độc lập; cơ chế gồm cả lệch biên lẫn một thành phần chưa
   xác định mà tỉ lệ phân mảnh không đo được" — là một phát biểu khoa học thật, có giá trị
   ngang một cải thiện F1 nhỏ, và không phóng đại quá mức đã đo.

## 4. Việc CHƯA làm, ghi lại để không mất dấu

- **Nghe trực tiếp clip `long_event` vỡ nhiều nhất** — quan sát định tính, cần tai người,
  chưa làm. Đây là hướng rẻ nhất nếu sau này có thời gian quay lại.
- **Đổi hàm mất mát để phạt phân mảnh trực tiếp** (BCE theo khung hiện không phạt việc
  một dự báo vỡ đôi) — cần train lại, không thử.
- **`tpb=4`** (giữa 3 và 2, ít gây rủi ro trúng collar hơn) — không chạy, vì hướng
  `time_pool_blocks` đã bị bác bỏ theo chiều đã thử và chiều ngược lại (tăng trường tiếp
  nhận) mâu thuẫn với nhu cầu phân giải mịn để trúng collar 200 ms — ngõ cụt kiến trúc với
  PANNs, xem cảnh báo ở `tests/test_ml_sed.py::test_bot_gop_thoi_gian_thi_do_phan_giai_min_hon`.
- **Thành phần lỗi thứ hai** làm F1 kém dù tỉ lệ vỡ không đổi — chưa xác định là gì. Nếu
  `long_event` được quay lại trong tương lai, đây là câu hỏi mở đầu tiên nên hỏi, và bảng
  6 loại lỗi (`error_taxonomy.py`) đã có sẵn công cụ để trả lời mà không cần code mới.

## 5. Trạng thái cuối cùng của lát cắt `long_event` trong báo cáo

Đưa vào bảng theo slice (SYSTEM.md §8.3) như mọi lát cắt khác, không có xử lý đặc biệt.
Phần Hạn chế của khoá luận nên ghi ngắn gọn: *độ dài sự kiện là một biến khó độc lập với
F1 thấp hơn lát cắt `sạch` khoảng 0,15 điểm F1 ở mọi cấu hình đã thử; sáu giả thuyết về
nguyên nhân đã được kiểm và loại; cơ chế đầy đủ chưa được xác định.*
