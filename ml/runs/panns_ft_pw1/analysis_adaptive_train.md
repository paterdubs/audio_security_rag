# Phân tích lỗi — panns_ft_pw1 trên dev/all

Sinh bởi `ml/evaluation/error_analysis.py` từ `ml/runs/panns_ft_pw1/predictions/dev_all.npz` (1440 clip, 4873 sự kiện tham chiếu). Không train lại, không dùng GPU.

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với tập train của `v3`, nên dev **thiên vị v3 theo thiết kế**. Mọi so sánh v1/v2/v3 đọc trên trang này đều mang thiên vị đó.
> ⚠️ θ\* và θ theo từng lớp đều chọn **trên chính tập đang chấm** → rò rỉ. Chúng là chặn trên lạc quan, không phải kết quả báo cáo được.

## 1. Sáu loại lỗi

Tầng 1 gán cho mỗi sự kiện tham chiếu **đúng một** nhãn (Đúng/Boundary/Substitution/Deletion); dự báo không được tiêu thụ là Insertion. Fragmentation và Merge là **tầng 2**, đếm riêng và có thể chồng lên tầng 1.

Collar onset 200 ms, chỉ khớp onset (offset của tiếng vang quá mơ hồ để chấm).

**θ = 0,50** — 4873 sự kiện thật, 4155 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 1957 | 0,402 |
| Boundary | 789 | 0,162 |
| Substitution | 440 | 0,090 |
| Deletion | 1687 | 0,346 |
| Insertion | 969 | 0,199 |
| Fragmentation (1 thật → n đoán) | 600 | — |
| Merge (n thật → 1 đoán) | 109 | — |

**θ = 0,25** — 4873 sự kiện thật, 6371 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 2632 | 0,540 |
| Boundary | 815 | 0,167 |
| Substitution | 647 | 0,133 |
| Deletion | 779 | 0,160 |
| Insertion | 2277 | 0,467 |
| Fragmentation (1 thật → n đoán) | 615 | — |
| Merge (n thật → 1 đoán) | 125 | — |

**Boundary không phải một rổ độc lập với sed_eval.** Một dự báo đúng lớp nhưng lệch onset quá collar bị sed_eval tính **một Deletion + một Insertion**. Bảng trên tách nó ra vì hai lỗi cần hai cách sửa khác nhau: lệch biên sửa ở hậu xử lý, còn không-nghe-ra-gì phải sửa ở dữ liệu. Số S/D/I gốc của sed_eval:

| θ | Nref | Nsys | Substitution | Deletion | Insertion | error rate |
|---|---|---|---|---|---|---|
| θ = 0,50 | 4873 | 4155 | 266 | 2650 | 1932 | 0,995 |
| θ = 0,25 | 4873 | 6371 | 592 | 1649 | 3147 | 1,106 |

Đối chiếu số khớp đúng giữa hai phép ghép (greedy ở đây vs khớp tối ưu trong sed_eval) — lệch lớn nghĩa là phép ghép này không đáng tin và bảng trên phải bỏ:

| θ | ghép greedy | sed_eval Ntp | lệch |
|---|---|---|---|
| θ = 0,50 | 1957 | 1957 | +0 |
| θ = 0,25 | 2632 | 2632 | +0 |

## 2. Theo lát cắt

Chấm ở θ = 0,25. **Lát cắt chồng nhau** — một clip vừa `overlap` vừa `reverb` nằm ở cả hai hàng, nên tổng cột *clip* lớn hơn 1440 clip của tập. `sach` là clip không thuộc lát cắt khó nào; nó là mốc đối chiếu, không phải một lát cắt.

| lát cắt | clip | sự kiện thật | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|---|---|
| overlap | 437 | 1723 | 0.4496 | 0.7344 |
| low_snr | 390 | 1449 | 0.4027 | 0.7124 |
| reverb | 557 | 1888 | 0.4412 | 0.7120 |
| long_event | 144 | 559 | 0.3602 | 0.7494 |
| causal_chain | 203 | 845 | 0.5171 | 0.7350 |
| sach | 357 | 967 | 0.5140 | 0.7606 |

Phân loại lỗi **đếm lại trên từng tập con clip**, không chia tỉ lệ từ tổng — lát cắt chồng nhau nên cộng cột sẽ lớn hơn tổng toàn tập. Cột *phân mảnh/thật* là số sự kiện thật bị cắt thành nhiều mảnh, chia cho số sự kiện thật của chính lát cắt đó, nên so ngang giữa các hàng được.

| lát cắt | thật | Đúng | Boundary | Substitution | Deletion | Insertion | phân mảnh | phân mảnh/thật | gộp |
|---|---|---|---|---|---|---|---|---|---|
| overlap | 1723 | 864 | 287 | 237 | 335 | 732 | 220 | 0,128 | 55 |
| low_snr | 1449 | 674 | 291 | 206 | 278 | 727 | 197 | 0,136 | 35 |
| reverb | 1888 | 947 | 321 | 269 | 351 | 868 | 237 | 0,126 | 46 |
| long_event | 559 | 268 | 128 | 79 | 84 | 454 | 138 | 0,247 | 17 |
| causal_chain | 845 | 475 | 126 | 93 | 151 | 298 | 76 | 0,090 | 17 |
| sach | 967 | 580 | 151 | 122 | 114 | 437 | 105 | 0,109 | 23 |

## 3. Ma trận nhầm lẫn

Hàng = lớp thật, cột = lớp đoán, `∅` = không có đối tác (Deletion ở cột, Insertion ở hàng). **Đường chéo gồm cả Đúng lẫn Boundary** — cả hai đều đúng lớp, chỉ khác thời điểm; đọc đường chéo như 'số đúng' là đọc sai.

```
         0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
  0    257    2    1    0    1    3    0    1    2    2    0    1    0    0    0   38   alarm_bell
  1      3  304    1    0    2    0    2    0    2    7    1    1    2    0    0   48   applause_cheering
  2      4    1  200    5   11    2    8    0    5   23    0    1    2    0    1   54   door_slam
  3      1    3    9  176    5    1    9    1    0   12    0    2    1    1    2   53   explosion
  4      2    4    6    3  275    1    6    0    1   18    1    2    1    0    2   50   fireworks
  5      2    0    5    2    4  266    5    1    3    7    0    0    1    1    2   29   glass_breaking
  6      3    4   15    6   14    0  147    0    0   19    0    1    0    0    0   59   gunshot
  7      0    3    3    0    4    1    0  215    5    5    1    7    1    1    1   66   laughter
  8     11    1   27    0   10   22    7    1  140   21    0    3    2    0    0   81   object_drop_dishes
  9      1    3    7    3   15    1   11    0    1  269    0    1    1    2    0   85   running_footsteps
 10      6    2    0    0    1    2    0    0    0    1  258    8    3    0    0   34   scream
 11      2    0    2    1    0    0    2    5    2    3    2  311    2    1    0   38   shout_yell
 12      4    0    1    0    1    1    0    0    0    3    0    0  252    0    1   11   siren
 13      1    2    0    1    3    1    0    4    2    9    1   23    1  214    0   51   speech_normal
 14      7    5   10    3    7    6    4    2    5   11    2    5    3    4  163   82   vehicle_crash
 15    258  149  189   28  289  143  149   38  105  480   49  164  100  100   36    0   ∅
```

Các cặp nhầm nhiều nhất, đối chiếu `confusable_with` khai trong `ontology_map.yaml`:

| thật | đoán | số | đã khai trong ontology |
|---|---|---|---|
| object_drop_dishes | door_slam | 27 | có |
| door_slam | running_footsteps | 23 | có |
| speech_normal | shout_yell | 23 | có |
| object_drop_dishes | glass_breaking | 22 | có |
| object_drop_dishes | running_footsteps | 21 | có |
| gunshot | running_footsteps | 19 | có |
| fireworks | running_footsteps | 18 | có |
| gunshot | door_slam | 15 | có |
| running_footsteps | fireworks | 15 | có |
| gunshot | fireworks | 14 | có |
| explosion | running_footsteps | 12 | có |
| door_slam | fireworks | 11 | có |
| object_drop_dishes | alarm_bell | 11 | **chưa** |
| running_footsteps | gunshot | 11 | có |
| vehicle_crash | running_footsteps | 11 | **chưa** |

361/647 (56%) lượt nhầm rơi vào cặp đã khai. Phần còn lại là nhầm lẫn mà ontology **chưa dự đoán** — nếu tỉ lệ này cao thì thứ cần sửa là `confusable_with`, không phải model.

## 4. Ngưỡng theo từng lớp

θ tốt nhất **toàn cục** là 0,25. Bảng dưới cho thấy θ đó không tối ưu cho phần lớn lớp:

| lớp | sự kiện thật | θ* riêng | F1 @θ* riêng | F1 @θ=0,25 | chênh |
|---|---|---|---|---|---|
| alarm_bell | 308 | 0,60 | 0,4754 | 0,4552 | +0,0202 |
| applause_cheering | 373 | 0,15 | 0,4589 | 0,4182 | +0,0407 |
| door_slam | 317 | 0,25 | 0,3632 | 0,3632 | +0,0000 |
| explosion | 276 | 0,20 | 0,6182 | 0,6111 | +0,0071 |
| fireworks | 372 | 0,35 | 0,3418 | 0,3412 | +0,0006 |
| glass_breaking | 328 | 0,30 | 0,5841 | 0,5758 | +0,0083 |
| gunshot | 268 | 0,35 | 0,4790 | 0,4272 | +0,0519 |
| laughter | 313 | 0,20 | 0,5927 | 0,5886 | +0,0041 |
| object_drop_dishes | 326 | 0,20 | 0,4275 | 0,4240 | +0,0035 |
| running_footsteps | 400 | 0,45 | 0,4234 | 0,3628 | +0,0606 |
| scream | 315 | 0,20 | 0,7043 | 0,6825 | +0,0217 |
| shout_yell | 371 | 0,20 | 0,5466 | 0,5372 | +0,0094 |
| siren | 274 | 0,20 | 0,6068 | 0,5975 | +0,0093 |
| speech_normal | 313 | 0,20 | 0,4654 | 0,4553 | +0,0101 |
| vehicle_crash | 319 | 0,25 | 0,3833 | 0,3833 | +0,0000 |

| cấu hình ngưỡng | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|
| θ = 0,50 (mặc định lúc train) | 0,4335 | 0,7125 |
| θ = 0,25 toàn cục | 0,4682 | 0,7395 |
| θ riêng từng lớp | 0,4921 | 0,7528 |

Hàng cuối chọn 15 ngưỡng trên chính tập đang chấm, nên nó là **chặn trên**, không phải hiệu năng mong đợi trên dữ liệu mới. Muốn dùng số này làm kết quả thì ngưỡng phải khoá trên dev rồi đo một lần trên tập test độc lập — mà `gold_test` hiện **rỗng**.
