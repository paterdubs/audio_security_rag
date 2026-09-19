# Phân tích lỗi — panns_ft_v2 trên dev/all

Sinh bởi `ml/evaluation/error_analysis.py` từ `ml/runs/panns_ft_v2/predictions/dev_all.npz` (1440 clip, 4873 sự kiện tham chiếu). Không train lại, không dùng GPU.

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với tập train của `v3`, nên dev **thiên vị v3 theo thiết kế**. Mọi so sánh v1/v2/v3 đọc trên trang này đều mang thiên vị đó.
> ⚠️ θ\* và θ theo từng lớp đều chọn **trên chính tập đang chấm** → rò rỉ. Chúng là chặn trên lạc quan, không phải kết quả báo cáo được.

## 1. Sáu loại lỗi

Tầng 1 gán cho mỗi sự kiện tham chiếu **đúng một** nhãn (Đúng/Boundary/Substitution/Deletion); dự báo không được tiêu thụ là Insertion. Fragmentation và Merge là **tầng 2**, đếm riêng và có thể chồng lên tầng 1.

Collar onset 200 ms, chỉ khớp onset (offset của tiếng vang quá mơ hồ để chấm).

**θ = 0,50** — 4873 sự kiện thật, 50686 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 3120 | 0,640 |
| Boundary | 1315 | 0,270 |
| Substitution | 356 | 0,073 |
| Deletion | 82 | 0,017 |
| Insertion | 45895 | 9,418 |
| Fragmentation (1 thật → n đoán) | 912 | — |
| Merge (n thật → 1 đoán) | 171 | — |

**θ = 0,90** — 4873 sự kiện thật, 7067 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 2100 | 0,431 |
| Boundary | 895 | 0,184 |
| Substitution | 794 | 0,163 |
| Deletion | 1084 | 0,223 |
| Insertion | 3278 | 0,673 |
| Fragmentation (1 thật → n đoán) | 919 | — |
| Merge (n thật → 1 đoán) | 134 | — |

**Boundary không phải một rổ độc lập với sed_eval.** Một dự báo đúng lớp nhưng lệch onset quá collar bị sed_eval tính **một Deletion + một Insertion**. Bảng trên tách nó ra vì hai lỗi cần hai cách sửa khác nhau: lệch biên sửa ở hậu xử lý, còn không-nghe-ra-gì phải sửa ở dữ liệu. Số S/D/I gốc của sed_eval:

| θ | Nref | Nsys | Substitution | Deletion | Insertion | error rate |
|---|---|---|---|---|---|---|
| θ = 0,50 | 4873 | 50686 | 1275 | 478 | 46291 | 9,859 |
| θ = 0,90 | 4873 | 7067 | 683 | 2090 | 4284 | 1,448 |

Đối chiếu số khớp đúng giữa hai phép ghép (greedy ở đây vs khớp tối ưu trong sed_eval) — lệch lớn nghĩa là phép ghép này không đáng tin và bảng trên phải bỏ:

| θ | ghép greedy | sed_eval Ntp | lệch |
|---|---|---|---|
| θ = 0,50 | 3120 | 3120 | +0 |
| θ = 0,90 | 2100 | 2100 | +0 |

## 2. Theo lát cắt

Chấm ở θ = 0,90. **Lát cắt chồng nhau** — một clip vừa `overlap` vừa `reverb` nằm ở cả hai hàng, nên tổng cột *clip* lớn hơn 1440 clip của tập. `sach` là clip không thuộc lát cắt khó nào; nó là mốc đối chiếu, không phải một lát cắt.

| lát cắt | clip | sự kiện thật | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|---|---|
| overlap | 437 | 1723 | 0.3345 | 0.6548 |
| low_snr | 390 | 1449 | 0.3008 | 0.6231 |
| reverb | 557 | 1888 | 0.3321 | 0.6152 |
| long_event | 144 | 559 | 0.2391 | 0.6446 |
| causal_chain | 203 | 845 | 0.4034 | 0.6561 |
| sach | 357 | 967 | 0.3901 | 0.6914 |

Phân loại lỗi **đếm lại trên từng tập con clip**, không chia tỉ lệ từ tổng — lát cắt chồng nhau nên cộng cột sẽ lớn hơn tổng toàn tập. Cột *phân mảnh/thật* là số sự kiện thật bị cắt thành nhiều mảnh, chia cho số sự kiện thật của chính lát cắt đó, nên so ngang giữa các hàng được.

| lát cắt | thật | Đúng | Boundary | Substitution | Deletion | Insertion | phân mảnh | phân mảnh/thật | gộp |
|---|---|---|---|---|---|---|---|---|---|
| overlap | 1723 | 686 | 330 | 293 | 414 | 1070 | 323 | 0,187 | 73 |
| low_snr | 1449 | 518 | 313 | 240 | 378 | 924 | 290 | 0,200 | 37 |
| reverb | 1888 | 738 | 339 | 317 | 494 | 1163 | 345 | 0,183 | 55 |
| long_event | 559 | 198 | 144 | 107 | 110 | 648 | 166 | 0,297 | 28 |
| causal_chain | 845 | 375 | 152 | 100 | 218 | 387 | 128 | 0,151 | 18 |
| sach | 967 | 483 | 162 | 150 | 172 | 714 | 172 | 0,178 | 17 |

## 3. Ma trận nhầm lẫn

Hàng = lớp thật, cột = lớp đoán, `∅` = không có đối tác (Deletion ở cột, Insertion ở hàng). **Đường chéo gồm cả Đúng lẫn Boundary** — cả hai đều đúng lớp, chỉ khác thời điểm; đọc đường chéo như 'số đúng' là đọc sai.

```
         0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
  0    222    3    2    0    2    3    1    1    1    1    0    2   20    1    0   49   alarm_bell
  1      3  310    0    0    1    1    1    0    0    5    3    1    8    0    0   40   applause_cheering
  2      3    3  140    6   11    3    5    0    2   17    0    1    3    2    0  121   door_slam
  3      2    6    7  169    4    2    4    0    0    7    1    1   10    1    0   62   explosion
  4      5    7    2    7  250    1    4    0    1   22    1    1    6    1    0   64   fireworks
  5      8    2    1    3    5  223    9    2    3    5    3    1    5    0    1   57   glass_breaking
  6      1    4   12   10   12    2  110    0    0   16    1    3    5    0    2   90   gunshot
  7      1    8    0    0    6    1    1  203    2    5    2    9    8    1    0   66   laughter
  8      7    6   24    0   13   26    4    1   91   16    1    4    3    0    1  129   object_drop_dishes
  9      5    8   13   11   26    2    5    1    1  164    0    1    3    4    0  156   running_footsteps
 10      2    3    0    0    2    0    0    0    0    1  251   15    7    0    1   33   scream
 11      4    4    1    0    1    1    0    3    1    4    2  290    1    8    1   50   shout_yell
 12      2    3    0    0    0    0    0    0    0    2    2    0  255    0    0   10   siren
 13      3    7    1    1    3    4    0    5    0    4    0   22    6  198    0   59   speech_normal
 14      6    7    9    9    7    8    2   10    3    8    6    7   15    5  119   98   vehicle_crash
 15    350  391   88   63  450  148   78  132   61  435  214  286  322  204   56    0   ∅
```

Các cặp nhầm nhiều nhất, đối chiếu `confusable_with` khai trong `ontology_map.yaml`:

| thật | đoán | số | đã khai trong ontology |
|---|---|---|---|
| object_drop_dishes | glass_breaking | 26 | có |
| running_footsteps | fireworks | 26 | có |
| object_drop_dishes | door_slam | 24 | có |
| fireworks | running_footsteps | 22 | có |
| speech_normal | shout_yell | 22 | có |
| alarm_bell | siren | 20 | có |
| door_slam | running_footsteps | 17 | có |
| gunshot | running_footsteps | 16 | có |
| object_drop_dishes | running_footsteps | 16 | có |
| scream | shout_yell | 15 | có |
| vehicle_crash | siren | 15 | có |
| object_drop_dishes | fireworks | 13 | **chưa** |
| running_footsteps | door_slam | 13 | có |
| gunshot | door_slam | 12 | có |
| gunshot | fireworks | 12 | có |

404/794 (51%) lượt nhầm rơi vào cặp đã khai. Phần còn lại là nhầm lẫn mà ontology **chưa dự đoán** — nếu tỉ lệ này cao thì thứ cần sửa là `confusable_with`, không phải model.

## 4. Ngưỡng theo từng lớp

θ tốt nhất **toàn cục** là 0,90. Bảng dưới cho thấy θ đó không tối ưu cho phần lớn lớp:

| lớp | sự kiện thật | θ* riêng | F1 @θ* riêng | F1 @θ=0,90 | chênh |
|---|---|---|---|---|---|
| alarm_bell | 308 | 0,90 | 0,3927 | 0,3927 | +0,0000 |
| applause_cheering | 373 | 0,80 | 0,3177 | 0,3022 | +0,0155 |
| door_slam | 317 | 0,85 | 0,3069 | 0,3047 | +0,0022 |
| explosion | 276 | 0,95 | 0,5216 | 0,5153 | +0,0062 |
| fireworks | 372 | 0,85 | 0,2201 | 0,2077 | +0,0124 |
| glass_breaking | 328 | 0,85 | 0,4486 | 0,4462 | +0,0024 |
| gunshot | 268 | 0,85 | 0,3796 | 0,3699 | +0,0097 |
| laughter | 313 | 0,90 | 0,4620 | 0,4620 | +0,0000 |
| object_drop_dishes | 326 | 0,85 | 0,3663 | 0,3374 | +0,0289 |
| running_footsteps | 400 | 0,90 | 0,2446 | 0,2446 | +0,0000 |
| scream | 315 | 0,85 | 0,4700 | 0,4688 | +0,0011 |
| shout_yell | 371 | 0,90 | 0,3665 | 0,3665 | +0,0000 |
| siren | 274 | 0,80 | 0,4295 | 0,3828 | +0,0467 |
| speech_normal | 313 | 0,85 | 0,3534 | 0,3469 | +0,0065 |
| vehicle_crash | 319 | 0,85 | 0,2903 | 0,2760 | +0,0143 |

| cấu hình ngưỡng | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|
| θ = 0,50 (mặc định lúc train) | 0,1123 | 0,3058 |
| θ = 0,90 toàn cục | 0,3518 | 0,6547 |
| θ riêng từng lớp | 0,3581 | 0,6541 |

Hàng cuối chọn 15 ngưỡng trên chính tập đang chấm, nên nó là **chặn trên**, không phải hiệu năng mong đợi trên dữ liệu mới. Muốn dùng số này làm kết quả thì ngưỡng phải khoá trên dev rồi đo một lần trên tập test độc lập — mà `gold_test` hiện **rỗng**.
