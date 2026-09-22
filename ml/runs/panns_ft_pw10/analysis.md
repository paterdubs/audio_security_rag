# Phân tích lỗi — panns_ft_pw10 trên dev/all

Sinh bởi `ml/evaluation/error_analysis.py` từ `ml/runs/panns_ft_pw10/predictions/dev_all.npz` (1440 clip, 4873 sự kiện tham chiếu). Không train lại, không dùng GPU.

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với tập train của `v3`, nên dev **thiên vị v3 theo thiết kế**. Mọi so sánh v1/v2/v3 đọc trên trang này đều mang thiên vị đó.
> ⚠️ θ\* và θ theo từng lớp đều chọn **trên chính tập đang chấm** → rò rỉ. Chúng là chặn trên lạc quan, không phải kết quả báo cáo được.

## 1. Sáu loại lỗi

Tầng 1 gán cho mỗi sự kiện tham chiếu **đúng một** nhãn (Đúng/Boundary/Substitution/Deletion); dự báo không được tiêu thụ là Insertion. Fragmentation và Merge là **tầng 2**, đếm riêng và có thể chồng lên tầng 1.

Collar onset 200 ms, chỉ khớp onset (offset của tiếng vang quá mơ hồ để chấm).

**θ = 0,50** — 4873 sự kiện thật, 23711 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 3345 | 0,686 |
| Boundary | 918 | 0,188 |
| Substitution | 451 | 0,093 |
| Deletion | 159 | 0,033 |
| Insertion | 18997 | 3,898 |
| Fragmentation (1 thật → n đoán) | 857 | — |
| Merge (n thật → 1 đoán) | 143 | — |

**θ = 0,80** — 4873 sự kiện thật, 6635 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 2412 | 0,495 |
| Boundary | 927 | 0,190 |
| Substitution | 687 | 0,141 |
| Deletion | 847 | 0,174 |
| Insertion | 2609 | 0,535 |
| Fragmentation (1 thật → n đoán) | 894 | — |
| Merge (n thật → 1 đoán) | 141 | — |

**Boundary không phải một rổ độc lập với sed_eval.** Một dự báo đúng lớp nhưng lệch onset quá collar bị sed_eval tính **một Deletion + một Insertion**. Bảng trên tách nó ra vì hai lỗi cần hai cách sửa khác nhau: lệch biên sửa ở hậu xử lý, còn không-nghe-ra-gì phải sửa ở dữ liệu. Số S/D/I gốc của sed_eval:

| θ | Nref | Nsys | Substitution | Deletion | Insertion | error rate |
|---|---|---|---|---|---|---|
| θ = 0,50 | 4873 | 23711 | 782 | 746 | 19584 | 4,332 |
| θ = 0,80 | 4873 | 6635 | 608 | 1853 | 3615 | 1,247 |

Đối chiếu số khớp đúng giữa hai phép ghép (greedy ở đây vs khớp tối ưu trong sed_eval) — lệch lớn nghĩa là phép ghép này không đáng tin và bảng trên phải bỏ:

| θ | ghép greedy | sed_eval Ntp | lệch |
|---|---|---|---|
| θ = 0,50 | 3345 | 3345 | +0 |
| θ = 0,80 | 2412 | 2412 | +0 |

## 2. Theo lát cắt

Chấm ở θ = 0,80. **Lát cắt chồng nhau** — một clip vừa `overlap` vừa `reverb` nằm ở cả hai hàng, nên tổng cột *clip* lớn hơn 1440 clip của tập. `sach` là clip không thuộc lát cắt khó nào; nó là mốc đối chiếu, không phải một lát cắt.

| lát cắt | clip | sự kiện thật | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|---|---|
| overlap | 437 | 1723 | 0.3973 | 0.7339 |
| low_snr | 390 | 1449 | 0.3524 | 0.7139 |
| reverb | 557 | 1888 | 0.3942 | 0.7133 |
| long_event | 144 | 559 | 0.2992 | 0.7441 |
| causal_chain | 203 | 845 | 0.4623 | 0.7330 |
| sach | 357 | 967 | 0.4726 | 0.7652 |

Phân loại lỗi **đếm lại trên từng tập con clip**, không chia tỉ lệ từ tổng — lát cắt chồng nhau nên cộng cột sẽ lớn hơn tổng toàn tập. Cột *phân mảnh/thật* là số sự kiện thật bị cắt thành nhiều mảnh, chia cho số sự kiện thật của chính lát cắt đó, nên so ngang giữa các hàng được.

| lát cắt | thật | Đúng | Boundary | Substitution | Deletion | Insertion | phân mảnh | phân mảnh/thật | gộp |
|---|---|---|---|---|---|---|---|---|---|
| overlap | 1723 | 789 | 338 | 246 | 350 | 876 | 332 | 0,193 | 74 |
| low_snr | 1449 | 598 | 321 | 226 | 304 | 800 | 287 | 0,198 | 41 |
| reverb | 1888 | 866 | 356 | 276 | 390 | 1008 | 355 | 0,188 | 60 |
| long_event | 559 | 237 | 147 | 81 | 94 | 560 | 169 | 0,302 | 18 |
| causal_chain | 845 | 429 | 148 | 99 | 169 | 335 | 120 | 0,142 | 16 |
| sach | 967 | 548 | 170 | 137 | 112 | 497 | 156 | 0,161 | 20 |

## 3. Ma trận nhầm lẫn

Hàng = lớp thật, cột = lớp đoán, `∅` = không có đối tác (Deletion ở cột, Insertion ở hàng). **Đường chéo gồm cả Đúng lẫn Boundary** — cả hai đều đúng lớp, chỉ khác thời điểm; đọc đường chéo như 'số đúng' là đọc sai.

```
         0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
  0    246    6    1    0    1    4    0    1    1    1    0    1    2    1    0   43   alarm_bell
  1      4  314    1    0    2    0    1    0    1    2    2    0    1    0    1   44   applause_cheering
  2      3    3  169    6    9    3    9    0    3   24    0    2    2    1    2   81   door_slam
  3      1    4    8  184    9    1    7    1    0    3    0    1    3    0    3   51   explosion
  4      2    3    4    5  283    1    7    0    1   11    0    2    3    0    1   49   fireworks
  5      2    1    3    2    7  245   10    1    5    9    1    1    1    2    3   35   glass_breaking
  6      2    4   14    8   14    0  136    0    0   17    0    1    2    1    3   66   gunshot
  7      1    9    1    1    5    1    0  220    2    2    2    8    3    2    2   54   laughter
  8      9    3   26    0   14   24    5    1  109   20    0    2    4    0    0  109   object_drop_dishes
  9      2    9    8    4   26    1   13    0    2  214    1    2    3    1    2  112   running_footsteps
 10      3    3    0    0    3    1    0    0    0    0  265    6    3    0    2   29   scream
 11      1    1    0    0    1    0    0    3    1    4    3  310    1    4    1   41   shout_yell
 12      3    2    0    0    1    0    0    0    0    2    0    0  254    0    1   11   siren
 13      2    7    0    2    3    1    0    3    0    5    1   23    2  221    0   43   speech_normal
 14      6    6    8    3    8    2    3    3    1    8    4    8    7    4  169   79   vehicle_crash
 15    241  329   84   74  441  139   89   83   67  246  119  237  211  174   75    0   ∅
```

Các cặp nhầm nhiều nhất, đối chiếu `confusable_with` khai trong `ontology_map.yaml`:

| thật | đoán | số | đã khai trong ontology |
|---|---|---|---|
| object_drop_dishes | door_slam | 26 | có |
| running_footsteps | fireworks | 26 | có |
| door_slam | running_footsteps | 24 | có |
| object_drop_dishes | glass_breaking | 24 | có |
| speech_normal | shout_yell | 23 | có |
| object_drop_dishes | running_footsteps | 20 | có |
| gunshot | running_footsteps | 17 | có |
| gunshot | door_slam | 14 | có |
| gunshot | fireworks | 14 | có |
| object_drop_dishes | fireworks | 14 | **chưa** |
| running_footsteps | gunshot | 13 | có |
| fireworks | running_footsteps | 11 | có |
| glass_breaking | gunshot | 10 | **chưa** |
| door_slam | fireworks | 9 | có |
| door_slam | gunshot | 9 | có |

370/687 (54%) lượt nhầm rơi vào cặp đã khai. Phần còn lại là nhầm lẫn mà ontology **chưa dự đoán** — nếu tỉ lệ này cao thì thứ cần sửa là `confusable_with`, không phải model.

## 4. Ngưỡng theo từng lớp

θ tốt nhất **toàn cục** là 0,80. Bảng dưới cho thấy θ đó không tối ưu cho phần lớn lớp:

| lớp | sự kiện thật | θ* riêng | F1 @θ* riêng | F1 @θ=0,80 | chênh |
|---|---|---|---|---|---|
| alarm_bell | 308 | 0,80 | 0,4450 | 0,4450 | +0,0000 |
| applause_cheering | 373 | 0,60 | 0,3676 | 0,3027 | +0,0649 |
| door_slam | 317 | 0,80 | 0,3634 | 0,3634 | +0,0000 |
| explosion | 276 | 0,85 | 0,5421 | 0,5416 | +0,0005 |
| fireworks | 372 | 0,80 | 0,2819 | 0,2819 | +0,0000 |
| glass_breaking | 328 | 0,80 | 0,5113 | 0,5113 | +0,0000 |
| gunshot | 268 | 0,80 | 0,4489 | 0,4489 | +0,0000 |
| laughter | 313 | 0,75 | 0,5327 | 0,5278 | +0,0049 |
| object_drop_dishes | 326 | 0,70 | 0,4079 | 0,3738 | +0,0341 |
| running_footsteps | 400 | 0,85 | 0,3920 | 0,3822 | +0,0098 |
| scream | 315 | 0,80 | 0,6115 | 0,6115 | +0,0000 |
| shout_yell | 371 | 0,80 | 0,4595 | 0,4595 | +0,0000 |
| siren | 274 | 0,60 | 0,5154 | 0,4820 | +0,0335 |
| speech_normal | 313 | 0,80 | 0,3895 | 0,3895 | +0,0000 |
| vehicle_crash | 319 | 0,85 | 0,3302 | 0,3116 | +0,0186 |

| cấu hình ngưỡng | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|
| θ = 0,50 (mặc định lúc train) | 0,2340 | 0,4968 |
| θ = 0,80 toàn cục | 0,4192 | 0,7407 |
| θ riêng từng lớp | 0,4316 | 0,7420 |

Hàng cuối chọn 15 ngưỡng trên chính tập đang chấm, nên nó là **chặn trên**, không phải hiệu năng mong đợi trên dữ liệu mới. Muốn dùng số này làm kết quả thì ngưỡng phải khoá trên dev rồi đo một lần trên tập test độc lập — mà `gold_test` hiện **rỗng**.
