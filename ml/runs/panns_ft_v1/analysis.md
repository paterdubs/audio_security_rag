# Phân tích lỗi — panns_ft_v1 trên dev/all

Sinh bởi `ml/evaluation/error_analysis.py` từ `ml/runs/panns_ft_v1/predictions/dev_all.npz` (1440 clip, 4873 sự kiện tham chiếu). Không train lại, không dùng GPU.

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với tập train của `v3`, nên dev **thiên vị v3 theo thiết kế**. Mọi so sánh v1/v2/v3 đọc trên trang này đều mang thiên vị đó.
> ⚠️ θ\* và θ theo từng lớp đều chọn **trên chính tập đang chấm** → rò rỉ. Chúng là chặn trên lạc quan, không phải kết quả báo cáo được.

## 1. Sáu loại lỗi

Tầng 1 gán cho mỗi sự kiện tham chiếu **đúng một** nhãn (Đúng/Boundary/Substitution/Deletion); dự báo không được tiêu thụ là Insertion. Fragmentation và Merge là **tầng 2**, đếm riêng và có thể chồng lên tầng 1.

Collar onset 200 ms, chỉ khớp onset (offset của tiếng vang quá mơ hồ để chấm).

**θ = 0,50** — 4873 sự kiện thật, 14829 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 1130 | 0,232 |
| Boundary | 2732 | 0,561 |
| Substitution | 677 | 0,139 |
| Deletion | 334 | 0,069 |
| Insertion | 10290 | 2,112 |
| Fragmentation (1 thật → n đoán) | 308 | — |
| Merge (n thật → 1 đoán) | 183 | — |

**θ = 0,95** — 4873 sự kiện thật, 3721 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 1269 | 0,260 |
| Boundary | 1246 | 0,256 |
| Substitution | 550 | 0,113 |
| Deletion | 1808 | 0,371 |
| Insertion | 656 | 0,135 |
| Fragmentation (1 thật → n đoán) | 368 | — |
| Merge (n thật → 1 đoán) | 122 | — |

**Boundary không phải một rổ độc lập với sed_eval.** Một dự báo đúng lớp nhưng lệch onset quá collar bị sed_eval tính **một Deletion + một Insertion**. Bảng trên tách nó ra vì hai lỗi cần hai cách sửa khác nhau: lệch biên sửa ở hậu xử lý, còn không-nghe-ra-gì phải sửa ở dữ liệu. Số S/D/I gốc của sed_eval:

| θ | Nref | Nsys | Substitution | Deletion | Insertion | error rate |
|---|---|---|---|---|---|---|
| θ = 0,50 | 4873 | 14829 | 1264 | 2479 | 12435 | 3,320 |
| θ = 0,95 | 4873 | 3721 | 318 | 3286 | 2134 | 1,178 |

Đối chiếu số khớp đúng giữa hai phép ghép (greedy ở đây vs khớp tối ưu trong sed_eval) — lệch lớn nghĩa là phép ghép này không đáng tin và bảng trên phải bỏ:

| θ | ghép greedy | sed_eval Ntp | lệch |
|---|---|---|---|
| θ = 0,50 | 1130 | 1130 | +0 |
| θ = 0,95 | 1269 | 1269 | +0 |

## 2. Theo lát cắt

Chấm ở θ = 0,95. **Lát cắt chồng nhau** — một clip vừa `overlap` vừa `reverb` nằm ở cả hai hàng, nên tổng cột *clip* lớn hơn 1440 clip của tập. `sach` là clip không thuộc lát cắt khó nào; nó là mốc đối chiếu, không phải một lát cắt.

| lát cắt | clip | sự kiện thật | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|---|---|
| overlap | 437 | 1723 | 0.2851 | 0.6279 |
| low_snr | 390 | 1449 | 0.2690 | 0.5767 |
| reverb | 557 | 1888 | 0.2862 | 0.5955 |
| long_event | 144 | 559 | 0.2592 | 0.6017 |
| causal_chain | 203 | 845 | 0.3107 | 0.6175 |
| sach | 357 | 967 | 0.3324 | 0.6876 |

## 3. Ma trận nhầm lẫn

Hàng = lớp thật, cột = lớp đoán, `∅` = không có đối tác (Deletion ở cột, Insertion ở hàng). **Đường chéo gồm cả Đúng lẫn Boundary** — cả hai đều đúng lớp, chỉ khác thời điểm; đọc đường chéo như 'số đúng' là đọc sai.

```
         0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
  0    182    1    3    1    4    2    0    0    2    0    1    1   16    2    0   93   alarm_bell
  1      4  270    1    0    1    1    2    0    0    4    3    1    5    2    0   79   applause_cheering
  2      2    2  100    4    5    2    5    1    1    9    0    0    1    2    0  183   door_slam
  3      1    4    4  144    6    2    1    0    1    6    0    3    1    0    4   99   explosion
  4      1    6    1    3  216    2    3    0    0   21    1    0    4    2    1  111   fireworks
  5      5    1    1    2    2  181    4    0    6    3    0    0    1    1    1  120   glass_breaking
  6      0    0    7    9   10    0   83    0    0    8    0    1    1    1    0  148   gunshot
  7      2    2    0    1    2    0    0  186    1    1    2    7    6    0    1  102   laughter
  8      6    2   16    0    7   12    3    2   76    4    0    3    4    0    1  190   object_drop_dishes
  9      2    5   14    3   23    0    6    1    1   98    0    0    4    2    0  241   running_footsteps
 10      3    0    0    1    1    1    0    0    0    1  216   17    8    0    0   67   scream
 11      2    2    2    1    4    0    0    6    0    4    0  239    3   10    0   98   shout_yell
 12      4    2    0    0    0    0    0    0    0    1    1    0  238    0    0   28   siren
 13      3    3    0    0    2    1    0    1    1    3    1   16    1  177    0  104   speech_normal
 14      1    4    9    3    5    4    1    8    4    9    4    2    8    3  109  145   vehicle_crash
 15     34   76   10   13   76   27   23   22   12   98   22   92   88   51   12    0   ∅
```

Các cặp nhầm nhiều nhất, đối chiếu `confusable_with` khai trong `ontology_map.yaml`:

| thật | đoán | số | đã khai trong ontology |
|---|---|---|---|
| running_footsteps | fireworks | 23 | **chưa** |
| fireworks | running_footsteps | 21 | **chưa** |
| scream | shout_yell | 17 | có |
| alarm_bell | siren | 16 | có |
| object_drop_dishes | door_slam | 16 | có |
| speech_normal | shout_yell | 16 | có |
| running_footsteps | door_slam | 14 | **chưa** |
| object_drop_dishes | glass_breaking | 12 | có |
| gunshot | fireworks | 10 | có |
| shout_yell | speech_normal | 10 | có |
| door_slam | running_footsteps | 9 | **chưa** |
| gunshot | explosion | 9 | có |
| vehicle_crash | door_slam | 9 | **chưa** |
| vehicle_crash | running_footsteps | 9 | **chưa** |
| gunshot | running_footsteps | 8 | **chưa** |

173/550 (31%) lượt nhầm rơi vào cặp đã khai. Phần còn lại là nhầm lẫn mà ontology **chưa dự đoán** — nếu tỉ lệ này cao thì thứ cần sửa là `confusable_with`, không phải model.

## 4. Ngưỡng theo từng lớp

θ tốt nhất **toàn cục** là 0,95. Bảng dưới cho thấy θ đó không tối ưu cho phần lớn lớp:

| lớp | sự kiện thật | θ* riêng | F1 @θ* riêng | F1 @θ=0,95 | chênh |
|---|---|---|---|---|---|
| alarm_bell | 308 | 0,95 | 0,3214 | 0,3214 | +0,0000 |
| applause_cheering | 373 | 0,90 | 0,3808 | 0,3772 | +0,0037 |
| door_slam | 317 | 0,90 | 0,1891 | 0,1773 | +0,0117 |
| explosion | 276 | 0,95 | 0,3774 | 0,3774 | +0,0000 |
| fireworks | 372 | 0,95 | 0,2826 | 0,2826 | +0,0000 |
| glass_breaking | 328 | 0,95 | 0,3055 | 0,3055 | +0,0000 |
| gunshot | 268 | 0,95 | 0,2406 | 0,2406 | +0,0000 |
| laughter | 313 | 0,95 | 0,3963 | 0,3963 | +0,0000 |
| object_drop_dishes | 326 | 0,90 | 0,1948 | 0,1392 | +0,0555 |
| running_footsteps | 400 | 0,80 | 0,1303 | 0,1284 | +0,0020 |
| scream | 315 | 0,95 | 0,4170 | 0,4170 | +0,0000 |
| shout_yell | 371 | 0,95 | 0,2869 | 0,2869 | +0,0000 |
| siren | 274 | 0,95 | 0,4103 | 0,4103 | +0,0000 |
| speech_normal | 313 | 0,95 | 0,2827 | 0,2827 | +0,0000 |
| vehicle_crash | 319 | 0,95 | 0,2098 | 0,2098 | +0,0000 |

| cấu hình ngưỡng | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|
| θ = 0,50 (mặc định lúc train) | 0,1147 | 0,4657 |
| θ = 0,95 toàn cục | 0,2953 | 0,6326 |
| θ riêng từng lớp | 0,2897 | 0,6238 |

Hàng cuối chọn 15 ngưỡng trên chính tập đang chấm, nên nó là **chặn trên**, không phải hiệu năng mong đợi trên dữ liệu mới. Muốn dùng số này làm kết quả thì ngưỡng phải khoá trên dev rồi đo một lần trên tập test độc lập — mà `gold_test` hiện **rỗng**.
