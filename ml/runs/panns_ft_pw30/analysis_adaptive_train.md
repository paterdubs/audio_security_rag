# Phân tích lỗi — panns_ft_pw30 trên dev/all

Sinh bởi `ml/evaluation/error_analysis.py` từ `ml/runs/panns_ft_pw30/predictions/dev_all.npz` (1440 clip, 4873 sự kiện tham chiếu). Không train lại, không dùng GPU.

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với tập train của `v3`, nên dev **thiên vị v3 theo thiết kế**. Mọi so sánh v1/v2/v3 đọc trên trang này đều mang thiên vị đó.
> ⚠️ θ\* và θ theo từng lớp đều chọn **trên chính tập đang chấm** → rò rỉ. Chúng là chặn trên lạc quan, không phải kết quả báo cáo được.

## 1. Sáu loại lỗi

Tầng 1 gán cho mỗi sự kiện tham chiếu **đúng một** nhãn (Đúng/Boundary/Substitution/Deletion); dự báo không được tiêu thụ là Insertion. Fragmentation và Merge là **tầng 2**, đếm riêng và có thể chồng lên tầng 1.

Collar onset 200 ms, chỉ khớp onset (offset của tiếng vang quá mơ hồ để chấm).

**θ = 0,50** — 4873 sự kiện thật, 42714 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 2741 | 0,562 |
| Boundary | 1737 | 0,356 |
| Substitution | 379 | 0,078 |
| Deletion | 16 | 0,003 |
| Insertion | 37857 | 7,769 |
| Fragmentation (1 thật → n đoán) | 455 | — |
| Merge (n thật → 1 đoán) | 167 | — |

**θ = 0,85** — 4873 sự kiện thật, 6535 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 2508 | 0,515 |
| Boundary | 866 | 0,178 |
| Substitution | 670 | 0,138 |
| Deletion | 829 | 0,170 |
| Insertion | 2491 | 0,511 |
| Fragmentation (1 thật → n đoán) | 661 | — |
| Merge (n thật → 1 đoán) | 123 | — |

**Boundary không phải một rổ độc lập với sed_eval.** Một dự báo đúng lớp nhưng lệch onset quá collar bị sed_eval tính **một Deletion + một Insertion**. Bảng trên tách nó ra vì hai lỗi cần hai cách sửa khác nhau: lệch biên sửa ở hậu xử lý, còn không-nghe-ra-gì phải sửa ở dữ liệu. Số S/D/I gốc của sed_eval:

| θ | Nref | Nsys | Substitution | Deletion | Insertion | error rate |
|---|---|---|---|---|---|---|
| θ = 0,50 | 4873 | 42714 | 1499 | 633 | 38474 | 8,333 |
| θ = 0,85 | 4873 | 6535 | 579 | 1786 | 3448 | 1,193 |

Đối chiếu số khớp đúng giữa hai phép ghép (greedy ở đây vs khớp tối ưu trong sed_eval) — lệch lớn nghĩa là phép ghép này không đáng tin và bảng trên phải bỏ:

| θ | ghép greedy | sed_eval Ntp | lệch |
|---|---|---|---|
| θ = 0,50 | 2741 | 2741 | +0 |
| θ = 0,85 | 2508 | 2508 | +0 |

## 2. Theo lát cắt

Chấm ở θ = 0,85. **Lát cắt chồng nhau** — một clip vừa `overlap` vừa `reverb` nằm ở cả hai hàng, nên tổng cột *clip* lớn hơn 1440 clip của tập. `sach` là clip không thuộc lát cắt khó nào; nó là mốc đối chiếu, không phải một lát cắt.

| lát cắt | clip | sự kiện thật | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|---|---|
| overlap | 437 | 1723 | 0.4180 | 0.7062 |
| low_snr | 390 | 1449 | 0.3800 | 0.6851 |
| reverb | 557 | 1888 | 0.4116 | 0.6759 |
| long_event | 144 | 559 | 0.3130 | 0.7131 |
| causal_chain | 203 | 845 | 0.4820 | 0.7019 |
| sach | 357 | 967 | 0.4931 | 0.7414 |

Phân loại lỗi **đếm lại trên từng tập con clip**, không chia tỉ lệ từ tổng — lát cắt chồng nhau nên cộng cột sẽ lớn hơn tổng toàn tập. Cột *phân mảnh/thật* là số sự kiện thật bị cắt thành nhiều mảnh, chia cho số sự kiện thật của chính lát cắt đó, nên so ngang giữa các hàng được.

| lát cắt | thật | Đúng | Boundary | Substitution | Deletion | Insertion | phân mảnh | phân mảnh/thật | gộp |
|---|---|---|---|---|---|---|---|---|---|
| overlap | 1723 | 819 | 315 | 258 | 331 | 804 | 243 | 0,141 | 60 |
| low_snr | 1449 | 634 | 310 | 220 | 285 | 724 | 211 | 0,146 | 37 |
| reverb | 1888 | 902 | 330 | 283 | 373 | 980 | 260 | 0,138 | 51 |
| long_event | 559 | 236 | 143 | 94 | 86 | 476 | 148 | 0,265 | 18 |
| causal_chain | 845 | 455 | 139 | 96 | 155 | 353 | 83 | 0,098 | 17 |
| sach | 967 | 569 | 157 | 122 | 119 | 493 | 106 | 0,110 | 19 |

## 3. Ma trận nhầm lẫn

Hàng = lớp thật, cột = lớp đoán, `∅` = không có đối tác (Deletion ở cột, Insertion ở hàng). **Đường chéo gồm cả Đúng lẫn Boundary** — cả hai đều đúng lớp, chỉ khác thời điểm; đọc đường chéo như 'số đúng' là đọc sai.

```
         0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
  0    243    3    1    0    1    5    1    1    2    2    0    1    0    1    0   47   alarm_bell
  1      3  282    2    1    1    0    5    0    1    9    2    0    2    0    1   64   applause_cheering
  2      2    1  207    6    6    2    7    0    4   17    0    1    1    0    1   62   door_slam
  3      1    3    7  193    3    1    9    1    0    7    0    2    3    0    2   44   explosion
  4      3    3    6    7  245    1    9    0    1   31    1    2    2    0    0   61   fireworks
  5      0    0   11    4    1  259    7    1    3    8    1    0    1    1    1   30   glass_breaking
  6      2    3   12    4    3    1  164    0    0   19    0    1    0    1    1   57   gunshot
  7      1   10    2    1    2    1    1  217    1    4    2    6    2    1    0   62   laughter
  8      7    2   24    2    5   21    6    0  141   22    0    2    3    1    0   90   object_drop_dishes
  9      1    4    8    5   12    4   11    0    0  265    1    0    3    3    1   82   running_footsteps
 10      3    1    0    0    0    2    0    0    0    1  261    8    1    0    1   37   scream
 11      1    0    3    0    3    1    1    9    2    2    7  285    1    5    0   51   shout_yell
 12      7    0    0    1    1    0    0    2    0    2    2    0  245    1    0   13   siren
 13      1    3    1    1    4    2    0    3    0    6    2   20    2  217    0   51   speech_normal
 14      5    5   18    7    7    7    7    4    2    9    4    6    6    4  150   78   vehicle_crash
 15    213  170  255   64  231  163  229   65  127  477   78  154  105  129   31    0   ∅
```

Các cặp nhầm nhiều nhất, đối chiếu `confusable_with` khai trong `ontology_map.yaml`:

| thật | đoán | số | đã khai trong ontology |
|---|---|---|---|
| fireworks | running_footsteps | 31 | có |
| object_drop_dishes | door_slam | 24 | có |
| object_drop_dishes | running_footsteps | 22 | có |
| object_drop_dishes | glass_breaking | 21 | có |
| speech_normal | shout_yell | 20 | có |
| gunshot | running_footsteps | 19 | có |
| vehicle_crash | door_slam | 18 | **chưa** |
| door_slam | running_footsteps | 17 | có |
| gunshot | door_slam | 12 | có |
| running_footsteps | fireworks | 12 | có |
| glass_breaking | door_slam | 11 | **chưa** |
| running_footsteps | gunshot | 11 | có |
| laughter | applause_cheering | 10 | **chưa** |
| applause_cheering | running_footsteps | 9 | có |
| explosion | gunshot | 9 | có |

357/670 (53%) lượt nhầm rơi vào cặp đã khai. Phần còn lại là nhầm lẫn mà ontology **chưa dự đoán** — nếu tỉ lệ này cao thì thứ cần sửa là `confusable_with`, không phải model.

## 4. Ngưỡng theo từng lớp

θ tốt nhất **toàn cục** là 0,85. Bảng dưới cho thấy θ đó không tối ưu cho phần lớn lớp:

| lớp | sự kiện thật | θ* riêng | F1 @θ* riêng | F1 @θ=0,85 | chênh |
|---|---|---|---|---|---|
| alarm_bell | 308 | 0,85 | 0,4644 | 0,4644 | +0,0000 |
| applause_cheering | 373 | 0,55 | 0,4506 | 0,2990 | +0,1516 |
| door_slam | 317 | 0,90 | 0,3526 | 0,3410 | +0,0116 |
| explosion | 276 | 0,85 | 0,5769 | 0,5769 | +0,0000 |
| fireworks | 372 | 0,80 | 0,3422 | 0,3278 | +0,0144 |
| glass_breaking | 328 | 0,90 | 0,5534 | 0,5263 | +0,0271 |
| gunshot | 268 | 0,90 | 0,4372 | 0,4083 | +0,0289 |
| laughter | 313 | 0,85 | 0,5552 | 0,5552 | +0,0000 |
| object_drop_dishes | 326 | 0,85 | 0,4230 | 0,4230 | +0,0000 |
| running_footsteps | 400 | 0,90 | 0,3848 | 0,3638 | +0,0210 |
| scream | 315 | 0,85 | 0,6746 | 0,6746 | +0,0000 |
| shout_yell | 371 | 0,80 | 0,5052 | 0,4889 | +0,0162 |
| siren | 274 | 0,60 | 0,5886 | 0,5161 | +0,0725 |
| speech_normal | 313 | 0,85 | 0,4313 | 0,4313 | +0,0000 |
| vehicle_crash | 319 | 0,90 | 0,3509 | 0,3504 | +0,0005 |

| cấu hình ngưỡng | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|
| θ = 0,50 (mặc định lúc train) | 0,1152 | 0,2911 |
| θ = 0,85 toàn cục | 0,4397 | 0,7124 |
| θ riêng từng lớp | 0,4693 | 0,7380 |

Hàng cuối chọn 15 ngưỡng trên chính tập đang chấm, nên nó là **chặn trên**, không phải hiệu năng mong đợi trên dữ liệu mới. Muốn dùng số này làm kết quả thì ngưỡng phải khoá trên dev rồi đo một lần trên tập test độc lập — mà `gold_test` hiện **rỗng**.
