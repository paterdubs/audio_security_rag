# Phân tích lỗi — panns_ft_v2 trên dev/all

Sinh bởi `ml/evaluation/error_analysis.py` từ `ml/runs/panns_ft_v2/predictions/dev_all.npz` (1440 clip, 4873 sự kiện tham chiếu). Không train lại, không dùng GPU.

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với tập train của `v3`, nên dev **thiên vị v3 theo thiết kế**. Mọi so sánh v1/v2/v3 đọc trên trang này đều mang thiên vị đó.
> ⚠️ θ\* và θ theo từng lớp đều chọn **trên chính tập đang chấm** → rò rỉ. Chúng là chặn trên lạc quan, không phải kết quả báo cáo được.

## 1. Sáu loại lỗi

Tầng 1 gán cho mỗi sự kiện tham chiếu **đúng một** nhãn (Đúng/Boundary/Substitution/Deletion); dự báo không được tiêu thụ là Insertion. Fragmentation và Merge là **tầng 2**, đếm riêng và có thể chồng lên tầng 1.

Collar onset 200 ms, chỉ khớp onset (offset của tiếng vang quá mơ hồ để chấm).

**θ = 0,50** — 4873 sự kiện thật, 34546 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 2949 | 0,605 |
| Boundary | 1352 | 0,277 |
| Substitution | 464 | 0,095 |
| Deletion | 108 | 0,022 |
| Insertion | 29781 | 6,111 |
| Fragmentation (1 thật → n đoán) | 587 | — |
| Merge (n thật → 1 đoán) | 155 | — |

**θ = 0,85** — 4873 sự kiện thật, 6664 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 2256 | 0,463 |
| Boundary | 790 | 0,162 |
| Substitution | 803 | 0,165 |
| Deletion | 1024 | 0,210 |
| Insertion | 2815 | 0,578 |
| Fragmentation (1 thật → n đoán) | 656 | — |
| Merge (n thật → 1 đoán) | 122 | — |

**Boundary không phải một rổ độc lập với sed_eval.** Một dự báo đúng lớp nhưng lệch onset quá collar bị sed_eval tính **một Deletion + một Insertion**. Bảng trên tách nó ra vì hai lỗi cần hai cách sửa khác nhau: lệch biên sửa ở hậu xử lý, còn không-nghe-ra-gì phải sửa ở dữ liệu. Số S/D/I gốc của sed_eval:

| θ | Nref | Nsys | Substitution | Deletion | Insertion | error rate |
|---|---|---|---|---|---|---|
| θ = 0,50 | 4873 | 34546 | 1205 | 719 | 30392 | 6,632 |
| θ = 0,85 | 4873 | 6664 | 692 | 1925 | 3716 | 1,300 |

Đối chiếu số khớp đúng giữa hai phép ghép (greedy ở đây vs khớp tối ưu trong sed_eval) — lệch lớn nghĩa là phép ghép này không đáng tin và bảng trên phải bỏ:

| θ | ghép greedy | sed_eval Ntp | lệch |
|---|---|---|---|
| θ = 0,50 | 2949 | 2949 | +0 |
| θ = 0,85 | 2256 | 2256 | +0 |

## 2. Theo lát cắt

Chấm ở θ = 0,85. **Lát cắt chồng nhau** — một clip vừa `overlap` vừa `reverb` nằm ở cả hai hàng, nên tổng cột *clip* lớn hơn 1440 clip của tập. `sach` là clip không thuộc lát cắt khó nào; nó là mốc đối chiếu, không phải một lát cắt.

| lát cắt | clip | sự kiện thật | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|---|---|
| overlap | 437 | 1723 | 0.3791 | 0.6525 |
| low_snr | 390 | 1449 | 0.3467 | 0.6193 |
| reverb | 557 | 1888 | 0.3692 | 0.6167 |
| long_event | 144 | 559 | 0.2881 | 0.6426 |
| causal_chain | 203 | 845 | 0.4581 | 0.6639 |
| sach | 357 | 967 | 0.4203 | 0.6829 |

Phân loại lỗi **đếm lại trên từng tập con clip**, không chia tỉ lệ từ tổng — lát cắt chồng nhau nên cộng cột sẽ lớn hơn tổng toàn tập. Cột *phân mảnh/thật* là số sự kiện thật bị cắt thành nhiều mảnh, chia cho số sự kiện thật của chính lát cắt đó, nên so ngang giữa các hàng được.

| lát cắt | thật | Đúng | Boundary | Substitution | Deletion | Insertion | phân mảnh | phân mảnh/thật | gộp |
|---|---|---|---|---|---|---|---|---|---|
| overlap | 1723 | 753 | 286 | 280 | 404 | 931 | 238 | 0,138 | 59 |
| low_snr | 1449 | 578 | 269 | 244 | 358 | 794 | 204 | 0,141 | 34 |
| reverb | 1888 | 800 | 304 | 335 | 449 | 1007 | 249 | 0,132 | 48 |
| long_event | 559 | 225 | 119 | 113 | 102 | 546 | 139 | 0,249 | 20 |
| causal_chain | 845 | 418 | 125 | 104 | 198 | 333 | 87 | 0,103 | 14 |
| sach | 967 | 502 | 148 | 149 | 168 | 623 | 114 | 0,118 | 18 |

## 3. Ma trận nhầm lẫn

Hàng = lớp thật, cột = lớp đoán, `∅` = không có đối tác (Deletion ở cột, Insertion ở hàng). **Đường chéo gồm cả Đúng lẫn Boundary** — cả hai đều đúng lớp, chỉ khác thời điểm; đọc đường chéo như 'số đúng' là đọc sai.

```
         0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
  0    230    2    2    0    2    4    1    2    1    1    2    1    9    0    0   51   alarm_bell
  1      6  296    1    0    0    1    1    0    0    7    2    1    5    0    0   53   applause_cheering
  2      3    1  163    5    7    4    7    1    2   30    1    1    3    1    0   88   door_slam
  3      6    6   11  157    7    2    6    0    1   10    1    2    6    1    0   60   explosion
  4      6    4    3    7  251    1    6    0    1   27    1    2    2    1    0   60   fireworks
  5     10    1    3    1    3  228   10    0    3   10    1    2    4    0    1   51   glass_breaking
  6      3    2   15    7   11    0  133    0    0   20    0    2    1    0    0   74   gunshot
  7      1    6    0    0    3    1    3  198    2    6    6   10    4    1    0   72   laughter
  8      7    3   22    0   12   22    6    1  121   19    0    2    5    0    0  106   object_drop_dishes
  9      4    6   20    9   20    1   10    0    3  198    1    3    3    1    0  121   running_footsteps
 10      1    3    0    0    1    2    0    0    0    3  249   12    5    0    0   39   scream
 11      3    1    3    0    4    1    1    3    1    4    1  276    1    8    0   64   shout_yell
 12      4    1    0    0    0    0    0    0    0    1    2    0  251    0    0   15   siren
 13      4    5    1    1    3    4    0    2    1   10    0   26    7  184    0   65   speech_normal
 14      8    7    9    7    9    7    5    8    5   14    3    7   11    3  111  105   vehicle_crash
 15    369  188  164   37  309  143  144   95  110  619  106  225  166  115   25    0   ∅
```

Các cặp nhầm nhiều nhất, đối chiếu `confusable_with` khai trong `ontology_map.yaml`:

| thật | đoán | số | đã khai trong ontology |
|---|---|---|---|
| door_slam | running_footsteps | 30 | có |
| fireworks | running_footsteps | 27 | có |
| speech_normal | shout_yell | 26 | có |
| object_drop_dishes | door_slam | 22 | có |
| object_drop_dishes | glass_breaking | 22 | có |
| gunshot | running_footsteps | 20 | có |
| running_footsteps | door_slam | 20 | có |
| running_footsteps | fireworks | 20 | có |
| object_drop_dishes | running_footsteps | 19 | có |
| gunshot | door_slam | 15 | có |
| vehicle_crash | running_footsteps | 14 | **chưa** |
| object_drop_dishes | fireworks | 12 | **chưa** |
| scream | shout_yell | 12 | có |
| explosion | door_slam | 11 | có |
| gunshot | fireworks | 11 | có |

427/803 (53%) lượt nhầm rơi vào cặp đã khai. Phần còn lại là nhầm lẫn mà ontology **chưa dự đoán** — nếu tỉ lệ này cao thì thứ cần sửa là `confusable_with`, không phải model.

## 4. Ngưỡng theo từng lớp

θ tốt nhất **toàn cục** là 0,85. Bảng dưới cho thấy θ đó không tối ưu cho phần lớn lớp:

| lớp | sự kiện thật | θ* riêng | F1 @θ* riêng | F1 @θ=0,85 | chênh |
|---|---|---|---|---|---|
| alarm_bell | 308 | 0,90 | 0,4027 | 0,3905 | +0,0121 |
| applause_cheering | 373 | 0,80 | 0,4111 | 0,3867 | +0,0243 |
| door_slam | 317 | 0,85 | 0,3215 | 0,3215 | +0,0000 |
| explosion | 276 | 0,85 | 0,5404 | 0,5404 | +0,0000 |
| fireworks | 372 | 0,85 | 0,2781 | 0,2781 | +0,0000 |
| glass_breaking | 328 | 0,85 | 0,4887 | 0,4887 | +0,0000 |
| gunshot | 268 | 0,90 | 0,3879 | 0,3794 | +0,0086 |
| laughter | 313 | 0,85 | 0,5008 | 0,5008 | +0,0000 |
| object_drop_dishes | 326 | 0,85 | 0,3605 | 0,3605 | +0,0000 |
| running_footsteps | 400 | 0,90 | 0,2413 | 0,2350 | +0,0064 |
| scream | 315 | 0,85 | 0,5702 | 0,5702 | +0,0000 |
| shout_yell | 371 | 0,80 | 0,4111 | 0,4051 | +0,0060 |
| siren | 274 | 0,75 | 0,5414 | 0,4888 | +0,0526 |
| speech_normal | 313 | 0,85 | 0,4108 | 0,4108 | +0,0000 |
| vehicle_crash | 319 | 0,85 | 0,3246 | 0,3246 | +0,0000 |

| cấu hình ngưỡng | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|
| θ = 0,50 (mặc định lúc train) | 0,1496 | 0,3499 |
| θ = 0,85 toàn cục | 0,3911 | 0,6521 |
| θ riêng từng lớp | 0,4047 | 0,6655 |

Hàng cuối chọn 15 ngưỡng trên chính tập đang chấm, nên nó là **chặn trên**, không phải hiệu năng mong đợi trên dữ liệu mới. Muốn dùng số này làm kết quả thì ngưỡng phải khoá trên dev rồi đo một lần trên tập test độc lập — mà `gold_test` hiện **rỗng**.
