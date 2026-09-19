# Hiệu chuẩn xác suất — vì sao θ* của cả ba run đỉnh ở 0,85–0,95 — 2026-09-19

Việc 3 trong ba việc của buổi làm 19/09. Sinh bởi `ml/evaluation/calibration.py`
(`--run <tên> --split dev`) — không train lại, đọc thẳng `predictions/dev_all.npz`, mỗi
lượt chạy ~5–6 s. Reliability diagram + ECE (Expected Calibration Error) từng lớp, dựng
nhãn khung bằng đúng `frame_targets()` mà lúc train dùng để tạo nhãn (không tự viết lại
công thức làm tròn onset/offset).

> ⚠️ `data/synthetic/dev` sinh cùng recipe B0–B9 với train của `v3` → dev thiên vị v3
> theo thiết kế. Trang này đo **triệu chứng** (lệch bao nhiêu, lệch ở đâu) — **không kết
> luận nguyên nhân**. `pos_weight` là ứng viên hợp lý nhưng ablation nó cần train lại,
> ngoài phạm vi trang này (xem §4).

---

## 1. ECE tổng tăng dần v1 → v2 → v3 — không đọc thẳng thành "v3 hiệu chuẩn tệ nhất"

| run | ECE tổng (gộp mọi lớp, 10 bin) |
|---|---|
| panns_ft_v1 | 0,1041 |
| panns_ft_v2 | 0,2579 |
| panns_ft_v3 | 0,3332 |

Đọc con số này một mình rất dễ hiểu sai. `v1` có xác suất gần như **hằng số trên từng
mảng ~32 khung** (đã đo ở `long_event_postproc_20260919.md` §3, hệ quả của lưới onset
319,7 ms) — điều đó dồn phần lớn khung vào các bin thấp, ít khi tới bin cao, nên ECE thấp
phản ánh **dải giá trị hẹp**, không phải hiệu chuẩn tốt. `v2`/`v3` học được nhiều hơn nên
đẩy xác suất ra hai đầu thái cực rõ hơn — và chính việc đẩy ra thái cực đó bộc lộ độ lệch
hiệu chuẩn mà `v1` không có cơ hội bộc lộ. Không so ECE giữa ba run như so chất lượng
model, giống lý do đã nêu với event-F1 của `v1` (`error_analysis_20260919.md` §2).

## 2. Chữ ký chung: khối lượng khung khổng lồ ở vùng dự báo giữa, tỉ lệ dương thật gần 0

Hai lớp lệch nhiều nhất ở hai run khác nhau, cùng một chữ ký:

**`v3` / `vehicle_crash`** (ECE = 0,4752, 60.556 khung dương thật trong dev):

| bin dự báo | dự báo trung bình | tỉ lệ dương thật | số khung |
|---|---|---|---|
| 0,40–0,50 | 0,4578 | 0,0169 | 593.773 |
| 0,50–0,60 | 0,5419 | 0,0304 | 519.221 |
| 0,60–0,70 | 0,6393 | 0,0668 | 159.653 |
| 0,80–0,90 | 0,8418 | 0,4582 | 14.426 |
| 0,90–1,00 | 0,9631 | 0,9008 | 9.193 |

**`v2` / `running_footsteps`** (ECE = 0,3400, 27.247 khung dương thật):

| bin dự báo | dự báo trung bình | tỉ lệ dương thật | số khung |
|---|---|---|---|
| 0,10–0,20 | 0,1657 | 0,0004 | 336.950 |
| 0,20–0,30 | 0,2447 | 0,0007 | 366.619 |
| 0,50–0,60 | 0,5462 | 0,0142 | 99.129 |
| 0,90–1,00 | 0,9512 | 0,4407 | 31.325 |

Hai điểm giống nhau ở cả hai run:

- **Vùng dự báo 0,4–0,6 mang khối lượng khung lớn nhất** (nửa triệu đến hơn nửa triệu khung
  ở `vehicle_crash`) mà tỉ lệ dương thật chỉ 1,7–3%. Model nói "50/50" cho một vùng gần
  như chắc chắn là âm.
- **Tỉ lệ dương thật chỉ bắt kịp dự báo ở bin 0,90–1,00**, và ngay cả ở đó vẫn có thể còn
  cách xa (`running_footsteps` của `v2`: dự báo 0,95 nhưng thật chỉ 0,44).

Đây là lời giải thích hợp lý cho việc θ\* của cả ba run đều rơi vào 0,85–0,95 thay vì gần
0,5: phải vượt qua khối lượng khung khổng lồ ở vùng giữa — gần như toàn bộ là âm tính giả
— độ chính xác mới hồi phục.

## 3. Bảng đầy đủ theo lớp

`v3` (lệch nhiều nhất → ít nhất): vehicle_crash 0,4752 · shout_yell 0,4188 ·
speech_normal 0,4126 · explosion 0,4097 · fireworks 0,3627 · scream 0,3453 ·
alarm_bell 0,3421 · glass_breaking 0,3373 · laughter 0,3264 · applause_cheering 0,2890 ·
running_footsteps 0,2799 · siren 0,2743 · door_slam 0,2647 · gunshot 0,2548 ·
object_drop_dishes 0,2143.

Bảng đầy đủ + reliability diagram từng lớp: `ml/runs/{run}/calibration.{json,md}`.

## 4. Điều trang này KHÔNG chứng minh

- **Không kết luận `pos_weight` là nguyên nhân.** Đây là ứng viên hợp lý (loss có trọng số
  dương cao có thể đẩy xác suất lên quá mức ở vùng biên quyết định), nhưng xác nhận nó cần
  ablation `pos_weight` — train lại, chi phí khác hẳn phép đo này. Trang này dừng ở "lệch
  bao nhiêu, lệch theo hướng nào" (luôn quá tự tin ở vùng giữa, hướng dương).
- Số ECE của lớp có ít khung dương (không lớp nào ở bảng trên rơi vào diện đó — mọi lớp có
  từ 13 nghìn khung dương trở lên) — cảnh báo này áp dụng nếu dùng bảng cho lớp hiếm hơn,
  không áp dụng cho các số đã báo cáo ở đây.
- ECE tổng (`ece_tong`) gộp mọi lớp vào cùng một tập bin — bị chi phối bởi lớp có nhiều
  khung nhất, không phải trung bình có trọng số đều giữa các lớp.
- `gold_test` vẫn rỗng; toàn bộ trang này đo trên dev tổng hợp thiên vị v3.
