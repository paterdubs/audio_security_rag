# Phân tích lỗi — panns_ft_pw10 trên dev/all

Sinh bởi `ml/evaluation/error_analysis.py` từ `ml/runs/panns_ft_pw10/predictions/dev_all.npz` (1440 clip, 4873 sự kiện tham chiếu). Không train lại, không dùng GPU.

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với tập train của `v3`, nên dev **thiên vị v3 theo thiết kế**. Mọi so sánh v1/v2/v3 đọc trên trang này đều mang thiên vị đó.
> ⚠️ θ\* và θ theo từng lớp đều chọn **trên chính tập đang chấm** → rò rỉ. Chúng là chặn trên lạc quan, không phải kết quả báo cáo được.

## 1. Sáu loại lỗi

Tầng 1 gán cho mỗi sự kiện tham chiếu **đúng một** nhãn (Đúng/Boundary/Substitution/Deletion); dự báo không được tiêu thụ là Insertion. Fragmentation và Merge là **tầng 2**, đếm riêng và có thể chồng lên tầng 1.

Collar onset 200 ms, chỉ khớp onset (offset của tiếng vang quá mơ hồ để chấm).

**θ = 0,50** — 4873 sự kiện thật, 16039 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 3187 | 0,654 |
| Boundary | 920 | 0,189 |
| Substitution | 536 | 0,110 |
| Deletion | 230 | 0,047 |
| Insertion | 11396 | 2,339 |
| Fragmentation (1 thật → n đoán) | 526 | — |
| Merge (n thật → 1 đoán) | 135 | — |

**θ = 0,75** — 4873 sự kiện thật, 5824 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 2449 | 0,503 |
| Boundary | 827 | 0,170 |
| Substitution | 664 | 0,136 |
| Deletion | 933 | 0,192 |
| Insertion | 1884 | 0,387 |
| Fragmentation (1 thật → n đoán) | 616 | — |
| Merge (n thật → 1 đoán) | 125 | — |

**Boundary không phải một rổ độc lập với sed_eval.** Một dự báo đúng lớp nhưng lệch onset quá collar bị sed_eval tính **một Deletion + một Insertion**. Bảng trên tách nó ra vì hai lỗi cần hai cách sửa khác nhau: lệch biên sửa ở hậu xử lý, còn không-nghe-ra-gì phải sửa ở dữ liệu. Số S/D/I gốc của sed_eval:

| θ | Nref | Nsys | Substitution | Deletion | Insertion | error rate |
|---|---|---|---|---|---|---|
| θ = 0,50 | 4873 | 16039 | 754 | 932 | 12098 | 2,829 |
| θ = 0,75 | 4873 | 5824 | 574 | 1850 | 2801 | 1,072 |

Đối chiếu số khớp đúng giữa hai phép ghép (greedy ở đây vs khớp tối ưu trong sed_eval) — lệch lớn nghĩa là phép ghép này không đáng tin và bảng trên phải bỏ:

| θ | ghép greedy | sed_eval Ntp | lệch |
|---|---|---|---|
| θ = 0,50 | 3187 | 3187 | +0 |
| θ = 0,75 | 2449 | 2449 | +0 |

## 2. Theo lát cắt

Chấm ở θ = 0,75. **Lát cắt chồng nhau** — một clip vừa `overlap` vừa `reverb` nằm ở cả hai hàng, nên tổng cột *clip* lớn hơn 1440 clip của tập. `sach` là clip không thuộc lát cắt khó nào; nó là mốc đối chiếu, không phải một lát cắt.

| lát cắt | clip | sự kiện thật | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|---|---|
| overlap | 437 | 1723 | 0.4387 | 0.7289 |
| low_snr | 390 | 1449 | 0.3972 | 0.7136 |
| reverb | 557 | 1888 | 0.4331 | 0.7098 |
| long_event | 144 | 559 | 0.3415 | 0.7427 |
| causal_chain | 203 | 845 | 0.4997 | 0.7304 |
| sach | 357 | 967 | 0.5048 | 0.7572 |

Phân loại lỗi **đếm lại trên từng tập con clip**, không chia tỉ lệ từ tổng — lát cắt chồng nhau nên cộng cột sẽ lớn hơn tổng toàn tập. Cột *phân mảnh/thật* là số sự kiện thật bị cắt thành nhiều mảnh, chia cho số sự kiện thật của chính lát cắt đó, nên so ngang giữa các hàng được.

| lát cắt | thật | Đúng | Boundary | Substitution | Deletion | Insertion | phân mảnh | phân mảnh/thật | gộp |
|---|---|---|---|---|---|---|---|---|---|
| overlap | 1723 | 804 | 289 | 244 | 386 | 605 | 223 | 0,129 | 57 |
| low_snr | 1449 | 621 | 288 | 207 | 333 | 562 | 200 | 0,138 | 34 |
| reverb | 1888 | 881 | 319 | 272 | 416 | 708 | 239 | 0,127 | 47 |
| long_event | 559 | 243 | 134 | 81 | 101 | 406 | 139 | 0,249 | 16 |
| causal_chain | 845 | 443 | 128 | 98 | 176 | 259 | 80 | 0,095 | 17 |
| sach | 967 | 548 | 158 | 128 | 133 | 370 | 100 | 0,103 | 21 |

## 3. Ma trận nhầm lẫn

Hàng = lớp thật, cột = lớp đoán, `∅` = không có đối tác (Deletion ở cột, Insertion ở hàng). **Đường chéo gồm cả Đúng lẫn Boundary** — cả hai đều đúng lớp, chỉ khác thời điểm; đọc đường chéo như 'số đúng' là đọc sai.

```
         0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
  0    248    2    1    0    1    4    0    1    1    2    0    1    0    1    0   46   alarm_bell
  1      3  302    1    0    3    0    3    0    2    7    1    0    1    0    0   50   applause_cheering
  2      3    0  182    4   11    2    8    0    4   23    0    1    2    1    2   74   door_slam
  3      2    4    9  173    7    1    8    1    0    6    0    4    1    1    2   57   explosion
  4      2    3    5    5  272    1    8    0    1   16    1    2    2    0    0   54   fireworks
  5      3    1    5    2    3  248   11    0    2    8    0    0    1    0    1   43   glass_breaking
  6      2    4   13    4   13    1  145    0    0   15    0    1    0    0    1   69   gunshot
  7      1    5    2    0    3    1    0  209    2    3    2    6    3    1    0   75   laughter
  8      9    1   31    0   10   22    6    1  115   22    0    2    3    0    0  104   object_drop_dishes
  9      1    6   10    4   20    2   14    0    2  232    0    1    2    2    0  104   running_footsteps
 10      5    2    0    0    2    1    0    0    0    1  250    7    4    0    2   41   scream
 11      2    0    0    0    1    0    1    5    3    3    3  296    1    2    1   53   shout_yell
 12      5    1    0    0    1    0    0    1    0    3    0    0  247    0    1   15   siren
 13      1    5    1    1    3    1    0    4    2    7    1   25    0  205    0   57   speech_normal
 14      7    7   12    6    8    3    5    2    2    8    3    7    4    2  152   91   vehicle_crash
 15    207  164  119   29  269  106  124   38   76  313   47  151  106  103   32    0   ∅
```

Các cặp nhầm nhiều nhất, đối chiếu `confusable_with` khai trong `ontology_map.yaml`:

| thật | đoán | số | đã khai trong ontology |
|---|---|---|---|
| object_drop_dishes | door_slam | 31 | có |
| speech_normal | shout_yell | 25 | có |
| door_slam | running_footsteps | 23 | có |
| object_drop_dishes | glass_breaking | 22 | có |
| object_drop_dishes | running_footsteps | 22 | có |
| running_footsteps | fireworks | 20 | có |
| fireworks | running_footsteps | 16 | có |
| gunshot | running_footsteps | 15 | có |
| running_footsteps | gunshot | 14 | có |
| gunshot | door_slam | 13 | có |
| gunshot | fireworks | 13 | có |
| vehicle_crash | door_slam | 12 | **chưa** |
| door_slam | fireworks | 11 | có |
| glass_breaking | gunshot | 11 | **chưa** |
| object_drop_dishes | fireworks | 10 | **chưa** |

366/664 (55%) lượt nhầm rơi vào cặp đã khai. Phần còn lại là nhầm lẫn mà ontology **chưa dự đoán** — nếu tỉ lệ này cao thì thứ cần sửa là `confusable_with`, không phải model.

## 4. Ngưỡng theo từng lớp

θ tốt nhất **toàn cục** là 0,75. Bảng dưới cho thấy θ đó không tối ưu cho phần lớn lớp:

| lớp | sự kiện thật | θ* riêng | F1 @θ* riêng | F1 @θ=0,75 | chênh |
|---|---|---|---|---|---|
| alarm_bell | 308 | 0,80 | 0,4645 | 0,4549 | +0,0096 |
| applause_cheering | 373 | 0,55 | 0,4673 | 0,3841 | +0,0833 |
| door_slam | 317 | 0,80 | 0,3675 | 0,3616 | +0,0060 |
| explosion | 276 | 0,70 | 0,5985 | 0,5754 | +0,0231 |
| fireworks | 372 | 0,75 | 0,3504 | 0,3504 | +0,0000 |
| glass_breaking | 328 | 0,75 | 0,5603 | 0,5603 | +0,0000 |
| gunshot | 268 | 0,80 | 0,4559 | 0,4226 | +0,0333 |
| laughter | 313 | 0,75 | 0,5739 | 0,5739 | +0,0000 |
| object_drop_dishes | 326 | 0,65 | 0,4132 | 0,3903 | +0,0229 |
| running_footsteps | 400 | 0,80 | 0,3933 | 0,3798 | +0,0135 |
| scream | 315 | 0,70 | 0,6828 | 0,6742 | +0,0086 |
| shout_yell | 371 | 0,70 | 0,5188 | 0,5097 | +0,0091 |
| siren | 274 | 0,60 | 0,6074 | 0,5684 | +0,0391 |
| speech_normal | 313 | 0,65 | 0,4512 | 0,4279 | +0,0233 |
| vehicle_crash | 319 | 0,75 | 0,3626 | 0,3626 | +0,0000 |

| cấu hình ngưỡng | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|
| θ = 0,50 (mặc định lúc train) | 0,3048 | 0,5596 |
| θ = 0,75 toàn cục | 0,4579 | 0,7366 |
| θ riêng từng lớp | 0,4795 | 0,7471 |

Hàng cuối chọn 15 ngưỡng trên chính tập đang chấm, nên nó là **chặn trên**, không phải hiệu năng mong đợi trên dữ liệu mới. Muốn dùng số này làm kết quả thì ngưỡng phải khoá trên dev rồi đo một lần trên tập test độc lập — mà `gold_test` hiện **rỗng**.
