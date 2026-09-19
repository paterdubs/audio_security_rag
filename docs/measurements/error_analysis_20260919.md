# Phân tích lỗi trên dev tổng hợp — 2026-09-19

Sinh bởi `ml/evaluation/error_analysis.py` (Pha 4 của [TRAINING_OPS_PLAN](../TRAINING_OPS_PLAN.md))
từ dự đoán mức đoạn đã lưu ở Pha 3. Không train lại, không dùng GPU. Báo cáo đầy đủ của
từng run nằm ở `ml/runs/{run}/analysis.md` + `analysis.json`.

**Giao thức chung:** cả ba run chấm trên cùng `data/synthetic/dev` — 1.440 clip, 4.873 sự
kiện tham chiếu, cùng cửa sổ lọc trung vị mặc định 7 khung.

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với tập train của `v3`, nên dev
> **thiên vị v3 theo thiết kế**. Mọi so sánh v1/v2/v3 dưới đây đều mang thiên vị đó và
> không phải bằng chứng v3 là kiến trúc tốt hơn.
>
> ⚠️ θ\* và θ theo từng lớp đều chọn **trên chính tập đang chấm** → rò rỉ. Chúng là chặn
> trên lạc quan. `gold_test` hiện **rỗng** nên chưa có con số nào đủ tư cách báo cáo.

---

## 1. θ = 0,5 hỏng theo kiểu nào

Pha 3 đã cho thấy θ = 0,5 làm tụt điểm. Pha 4 trả lời *tụt vì cái gì* — và câu trả lời
không phải "model không nghe ra sự kiện":

| θ = 0,50 | v1 | v2 | v3 |
|---|---|---|---|
| Đúng | 1.130 | 3.120 | 2.901 |
| Boundary | 2.732 | 1.315 | 1.667 |
| Substitution | 677 | 356 | 296 |
| **Deletion** | **334** | **82** | **9** |
| **Insertion** | **10.290** | **45.895** | **61.020** |
| Insertion / sự kiện thật | 2,11 | 9,42 | **12,52** |

v3 bỏ sót **9 trên 4.873** sự kiện (0,2%) ở θ = 0,5. Nó nghe ra gần như mọi thứ; vấn đề
là nó còn nghe ra thêm 61.020 thứ không có. Đây là lỗi **ngưỡng và hậu xử lý**, không
phải lỗi năng lực phát hiện — và điều đó đổi hẳn việc cần làm tiếp: không phải thêm dữ
liệu cho lớp hiếm, mà là hiệu chỉnh ngưỡng rồi mới đánh giá lại.

Ở θ\* riêng của từng run, cán cân lật ngược:

| θ\* | v1 (0,95) | v2 (0,90) | v3 (0,90) |
|---|---|---|---|
| Đúng | 1.269 | 2.100 | 2.362 |
| Boundary | 1.246 | 895 | 971 |
| Substitution | 550 | 794 | 725 |
| Deletion | 1.808 | 1.084 | 815 |
| Insertion | 656 | 3.278 | 2.940 |
| Fragmentation | 368 | 919 | 927 |
| Merge | 122 | 134 | 146 |

### Boundary không phải rổ độc lập với sed_eval

Một dự báo **đúng lớp** nhưng lệch onset quá collar 200 ms bị `sed_eval` tính **một
Deletion + một Insertion**. Bảng trên tách nó ra vì hai lỗi cần hai cách sửa khác nhau:
lệch biên sửa ở khâu giải mã/hậu xử lý, còn không-nghe-ra-gì thì phải sửa ở dữ liệu. Hai
cách đếm **không cộng ra cùng một con số**; số S/D/I gốc của `sed_eval` in nguyên trong
`analysis.md` của từng run.

Phép ghép cặp ở `error_taxonomy.py` là greedy, `sed_eval` dùng khớp tối ưu trong file.
Đối chiếu số khớp đúng giữa hai cách trên **cả ba run, ở cả hai ngưỡng: lệch +0 ở cả sáu
phép đo** (v3 chẳng hạn: 2.901 vs 2.901 ở θ = 0,50 và 2.362 vs 2.362 ở θ = 0,90). Bảng
phân loại lỗi đứng được là nhờ phép đối chiếu này, không phải nhờ tin vào code tự viết.

---

## 2. Boundary của v1 phần lớn là trần cứng, không phải lỗi model

`TRAINING_OPS_PLAN` §4 đã cảnh báo đúng rằng "bước 323 ms không tự chứng minh bất khả thi
với collar 200 ms". Đo thật thì kết luận sắc hơn cảnh báo đó:

| | v1 | v2 | v3 |
|---|---|---|---|
| số đoạn / bước lưới onset | 31 · **319,7 ms** | 125 · 79,9 ms | 125 · 79,9 ms |
| onset dự báo rơi đúng trên lưới | 100,0% | 100,0% | 100,0% |
| trung vị \|Δonset\| (cặp cùng lớp) | 197 ms | 102 ms | 99 ms |
| \|Δonset\| > collar 200 ms | **49,5%** | 29,9% | 29,1% |
| trần do lượng tử hoá | **37,4%** | 0,0% | 0,0% |

Onset của một sự kiện được lấy tại **đầu đoạn hoạt động đầu tiên**, tức lượng tử hoá một
phía: sai số phân bố đều trong `[0, bước)` chứ không phải `[−bước/2, +bước/2]`. Với bước
319,7 ms thì `(319,7 − 200) / 319,7 = 37,4%` số cặp **không thể** lọt collar dù model đoán
hoàn hảo. Quan sát được 49,5%, tức khoảng ba phần tư lỗi biên của v1 đến từ độ phân giải
chứ không từ chất lượng model.

Hệ quả cho việc đọc bảng cũ: **so event-F1 của v1 với v2/v3 là so một phần độ phân giải
bộ giải mã, không thuần chất lượng model.** mAP mức clip không dính vấn đề này vì nó
không xét vị trí.

v2/v3 có trần 0% nhưng vẫn 29% cặp vượt collar — phần đó là lỗi thật, còn địa chỉ để sửa.

---

## 3. Theo lát cắt

F1 mức sự kiện ở θ\* của từng run. **Lát cắt chồng nhau** — một clip vừa `overlap` vừa
`reverb` nằm ở cả hai hàng, nên tổng cột *clip* (2.088) lớn hơn 1.440 clip của tập.
`sach` là 357 clip không thuộc lát cắt khó nào; nó là mốc đối chiếu, không phải lát cắt.

| lát cắt | clip | v1 @0,95 | v2 @0,90 | v3 @0,90 |
|---|---|---|---|---|
| sach (mốc) | 357 | 0,3324 | 0,3901 | 0,4487 |
| causal_chain | 203 | 0,3107 | 0,4034 | 0,4417 |
| overlap | 437 | 0,2851 | 0,3345 | 0,3799 |
| reverb | 557 | 0,2862 | 0,3321 | 0,3738 |
| low_snr | 390 | 0,2690 | 0,3008 | 0,3369 |
| **long_event** | 144 | 0,2592 | **0,2391** | **0,2685** |

Thứ tự khó giống nhau ở cả ba run, nên nó là tính chất của **dữ liệu**, không phải của một
lần chạy. `long_event` là lát cắt tệ nhất ở v2 và v3 — dưới mốc `sach` khoảng 0,15 F1.
`causal_chain` thì **không khó hơn** clip sạch bao nhiêu, ngược với giả định lúc thiết kế
lát cắt.

Nguyên nhân `long_event` đã được đo tiếp ở
[long_event_postproc_20260919.md](long_event_postproc_20260919.md): chữ ký của nó là **phân
mảnh gấp 1,9 lần** mốc sạch, không phải bỏ sót. Ứng viên "cửa sổ lọc 7 khung quá hẹp" bị
ablation **bác bỏ một phần** — cửa sổ rộng hơn nâng mọi lát cắt lên xấp xỉ cùng một lượng,
nên khoảng cách `long_event` ↔ `sach` gần như không đổi (0,180 → 0,178). Thủ phạm riêng của
lát cắt này vẫn chưa tìm ra.

---

## 4. Ma trận nhầm lẫn so với `confusable_with`

Trên v3 ở θ = 0,90, chỉ **222/725 (31%)** lượt nhầm lớp rơi vào cặp đã khai trong
`ml/configs/ontology_map.yaml`. 69% còn lại là nhầm lẫn mà ontology **chưa dự đoán**.

Các cặp nhầm nhiều nhất chưa được khai:

| thật | đoán | số |
|---|---|---|
| fireworks | running_footsteps | 26 |
| door_slam | running_footsteps | 22 |
| gunshot | running_footsteps | 19 |
| running_footsteps | gunshot | 15 |
| vehicle_crash | door_slam | 12 |

`running_footsteps` là **trung tâm nhầm lẫn**: nó hút nhầm từ bốn lớp xung lực
(`fireworks`, `door_slam`, `gunshot`, `object_drop_dishes`) và bắn nhầm ngược lại. Giả
thuyết chưa kiểm: chuỗi bước chân là một dãy xung ngắn lặp lại, nên nó trùng cấu trúc thời
gian với loạt bắn và với tiếng vật rơi. Giả thuyết này **chưa** kiểm được bằng dữ liệu
hiện có.

> ✅ **Đã làm (19/09, cùng ngày).** Tám cặp qua được tiêu chí "≥ 10 lượt ở v3 **và** ≥ 10 ở
> ít nhất một run khác" đã vào `confusable_with` (đối xứng hai chiều) và bảng tra §3 của
> [taxonomy.md](../taxonomy.md). Tỉ lệ lượt nhầm đã khai của v3 lên **392/725 (54,1%)**.
> Bảy cặp chỉ chạm ngưỡng ở đúng một run thì **không** khai — `confusable_with` điều khiển
> luật định tuyến sang người gán nhãn, mà nhầm-của-model không đồng nghĩa nhầm-của-người;
> chúng được liệt kê trong `taxonomy.md` §3 như nhầm lẫn của model chưa đủ bằng chứng.
> Đây là sửa **tài liệu cho khớp thực tế đo được**, không phải sửa model.

---

## 5. Ngưỡng theo từng lớp

| cấu hình | v1 | v2 | v3 |
|---|---|---|---|
| θ = 0,50 (mặc định lúc train) | 0,1147 | 0,1123 | 0,0820 |
| θ\* toàn cục | 0,2953 (0,95) | 0,3518 (0,90) | 0,3979 (0,90) |
| θ riêng từng lớp | **0,2897** | 0,3581 | 0,4157 |

**θ riêng từng lớp làm v1 TỆ ĐI** (0,2953 → 0,2897). Đó không phải lỗi cài đặt. F1 mức sự
kiện tổng của `sed_eval` là **micro-average**: TP/FP/FN gộp chung qua mọi lớp. Chọn θ tối
đa hoá F1 của *từng lớp riêng* không tối đa hoá con số gộp — một lớp nhiều sự kiện đánh
đổi thêm Insertion có thể nuốt hết phần lợi của vài lớp ít sự kiện. Với v2/v3 phần lợi
thắng (+0,006 và +0,018), với v1 thì không.

Ghi lại để không ai lặp lại: **không dùng θ riêng từng lớp như một cải tiến mặc định.** Nó
là công cụ chẩn đoán khoảng cách theo lớp, và ngay cả phần lợi ở v2/v3 cũng là chặn trên
chọn trên chính tập chấm.

Khoảng cách θ\* giữa các lớp (v3): `siren` tối ưu ở 0,60 còn `explosion` ở 0,95 — chênh
0,35 trên thang ngưỡng. Một θ toàn cục không phục vụ được cả hai.

---

## 6. Điều chưa trả lời được

- **Không có tập độc lập.** `gold_test` rỗng, chưa có real dev. Mọi con số trên trang này
  đo trên tập tổng hợp dùng chung foreground bank với train.
- **Confound v3 chưa gỡ.** dev cùng recipe B0–B9 với train của v3.
- **Nguyên nhân `long_event`** vẫn chưa xác định — đã loại được một ứng viên, xem
  [long_event_postproc_20260919.md](long_event_postproc_20260919.md).
- **`running_footsteps` là hub nhầm lẫn** — đã đo, chưa giải thích.
