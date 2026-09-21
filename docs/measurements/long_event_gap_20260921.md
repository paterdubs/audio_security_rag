# `long_event`: ứng viên cuối cùng cũng bị bác bỏ — 2026-09-21

Tiếp theo [max_median_ceiling_20260919.md](max_median_ceiling_20260919.md) §4, trang đó để
lại đúng một ứng viên chưa kiểm: *"Scaper có thể tạo khe hở năng lượng THẬT bên trong sự
kiện dài"*. Trang này đo trực tiếp, bằng `ml/evaluation/long_event_gap.py` — không train
lại, không GPU, đọc dự đoán đã lưu (Pha 3) + audio gốc trong `data/synthetic/dev/audio/`.

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với train của `v3` → thiên vị v3 theo
> thiết kế. θ\* lấy từ `analysis.json` của chính run — vẫn chọn trên tập đang chấm.

---

## 1. Phương pháp — bắt buộc có nhóm đối chứng

"Có khe hở" một mình không chứng minh gì — âm thanh thực nào cũng có lúc trầm lúc bổng.
Chỉ **chênh lệch giữa nhóm bị phân mảnh và nhóm không bị phân mảnh** mới là bằng chứng,
nên phép đo tách hai nhóm ngay trong CÙNG lát cắt `long_event` (144 clip, `v3` @θ\*=0,90):

1. `danh_dau_phan_manh()` — với mỗi sự kiện tham chiếu, đánh dấu có bị ≥2 dự báo cùng lớp
   chồng lấn hay không (cùng định nghĩa phân mảnh với `error_taxonomy._dem_cau_truc`).
2. Bỏ sự kiện ngắn hơn **1,0 s** — một sự kiện 0,3 giây không thể chứa khe hở nhiều trăm ms
   có ý nghĩa; giữ nó lại sẽ kéo trung vị nhóm "nguyên" xuống giả tạo vì độ dài, không phải
   vì Scaper hiền lành hơn ở đó. Còn lại **315/559 sự kiện**.
3. Với mỗi sự kiện còn lại: tính RMS từng cửa sổ 20 ms trên audio GỐC (không qua model),
   tìm chuỗi liên tục dài nhất có RMS < 20% trung vị RMS của chính sự kiện đó.

## 2. Kết quả — hai nhóm gần như giống hệt nhau

| nhóm | n | khe hở trung vị | khe hở p75 | khe hở trung bình | khe hở p90 |
|---|---|---|---|---|---|
| bị phân mảnh | 178 | 0,000s | 0,000s | 0,012s | 0,000s |
| không phân mảnh | 137 | 0,000s | 0,000s | 0,012s | 0,020s |

Trung bình **bằng nhau tuyệt đối** (0,012s cả hai nhóm). Ở p90, nhóm **không** phân mảnh
còn có khe hở lớn hơn nhóm bị phân mảnh (0,020s so với 0,000s) — ngược hướng giả thuyết.
Chỉ 17/178 sự kiện vỡ và 16/137 sự kiện nguyên có khe hở > 0 giây ở ngưỡng "im" 20%; phần
lớn tuyệt đối của cả hai nhóm không có khoảng lặng nào đáng kể.

**Kiểm độ nhạy với ngưỡng "im" nới ra 40%** (thay vì 20%, để chắc không phải do ngưỡng quá
khắt): mean vẫn **bằng nhau tuyệt đối** ở cả hai nhóm (0,046s), p75 bằng nhau (0,040s), và
p90 của nhóm nguyên vẫn cao hơn nhóm vỡ (0,168s so với 0,106s). Không phải hiện tượng biến
mất khi đổi ngưỡng — nó không tồn tại ở cả hai mức đã thử.

## 3. Kết luận — hết ứng viên

**Giả thuyết khe hở năng lượng thật bị bác bỏ.** `long_event` giờ đã loại BA ứng viên liên
tiếp qua ba lượt đo:

| Ứng viên | Kết quả | Nguồn |
|---|---|---|
| Cửa sổ lọc trung vị 7 khung quá hẹp | Bác bỏ một phần — nâng mọi lát cắt như nhau | `long_event_postproc_20260919.md` |
| Trần `MAX_MEDIAN_FRAMES=51` quá thấp | Bác bỏ phần lớn — `v2` còn tệ hơn | `max_median_ceiling_20260919.md` |
| Khe hở năng lượng thật bên trong nhãn | **Bác bỏ** — hai nhóm không khác nhau | trang này |

**Nguyên nhân riêng của `long_event` vẫn chưa xác định.** Không có ứng viên thứ tư đang
chờ sẵn — phải nói thẳng điều đó thay vì đi tìm một giả thuyết mới cho có. Hướng còn lại,
chưa được đo ở đề tài này: nghe trực tiếp một mẫu clip bị phân mảnh để tìm lý do bằng tai
(ví dụ chồng lấn với sự kiện khác cùng lớp làm model tách nhầm), hoặc so `long_event` với
các lát cắt khác theo SNR/reverb — cả hai đều ngoài phạm vi của phép đo hôm nay.

## 4. Điều trang này KHÔNG chứng minh

- θ\* chọn trên chính tập đang chấm — chặn trên lạc quan, không phải điểm báo cáo được.
- Ngưỡng "im" chỉ thử 20% và 40% trung vị; không quét toàn dải.
- `gold_test` vẫn rỗng — đo trên dev tổng hợp dùng chung foreground bank với train.
- Chỉ đo trên `v3`; không lặp lại trên `v2` (chữ ký phân mảnh đã xác nhận giống nhau giữa
  hai run ở `long_event_postproc_20260919.md`, nên không kỳ vọng năng lượng khác biệt).
