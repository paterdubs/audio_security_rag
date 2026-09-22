# Phân tích lỗi — panns_ft_tpb2 trên dev/all

Sinh bởi `ml/evaluation/error_analysis.py` từ `ml/runs/panns_ft_tpb2/predictions/dev_all.npz` (1440 clip, 4873 sự kiện tham chiếu). Không train lại, không dùng GPU.

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với tập train của `v3`, nên dev **thiên vị v3 theo thiết kế**. Mọi so sánh v1/v2/v3 đọc trên trang này đều mang thiên vị đó.
> ⚠️ θ\* và θ theo từng lớp đều chọn **trên chính tập đang chấm** → rò rỉ. Chúng là chặn trên lạc quan, không phải kết quả báo cáo được.

## 1. Sáu loại lỗi

Tầng 1 gán cho mỗi sự kiện tham chiếu **đúng một** nhãn (Đúng/Boundary/Substitution/Deletion); dự báo không được tiêu thụ là Insertion. Fragmentation và Merge là **tầng 2**, đếm riêng và có thể chồng lên tầng 1.

Collar onset 200 ms, chỉ khớp onset (offset của tiếng vang quá mơ hồ để chấm).

**θ = 0,50** — 4873 sự kiện thật, 7004 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 1931 | 0,396 |
| Boundary | 1047 | 0,215 |
| Substitution | 578 | 0,119 |
| Deletion | 1317 | 0,270 |
| Insertion | 3448 | 0,708 |
| Fragmentation (1 thật → n đoán) | 1255 | — |
| Merge (n thật → 1 đoán) | 184 | — |

**θ = 0,30** — 4873 sự kiện thật, 9530 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 2496 | 0,512 |
| Boundary | 1031 | 0,212 |
| Substitution | 714 | 0,146 |
| Deletion | 632 | 0,130 |
| Insertion | 5289 | 1,085 |
| Fragmentation (1 thật → n đoán) | 1392 | — |
| Merge (n thật → 1 đoán) | 184 | — |

**Boundary không phải một rổ độc lập với sed_eval.** Một dự báo đúng lớp nhưng lệch onset quá collar bị sed_eval tính **một Deletion + một Insertion**. Bảng trên tách nó ra vì hai lỗi cần hai cách sửa khác nhau: lệch biên sửa ở hậu xử lý, còn không-nghe-ra-gì phải sửa ở dữ liệu. Số S/D/I gốc của sed_eval:

| θ | Nref | Nsys | Substitution | Deletion | Insertion | error rate |
|---|---|---|---|---|---|---|
| θ = 0,50 | 4873 | 7004 | 460 | 2482 | 4613 | 1,550 |
| θ = 0,30 | 4873 | 9530 | 712 | 1665 | 6322 | 1,785 |

Đối chiếu số khớp đúng giữa hai phép ghép (greedy ở đây vs khớp tối ưu trong sed_eval) — lệch lớn nghĩa là phép ghép này không đáng tin và bảng trên phải bỏ:

| θ | ghép greedy | sed_eval Ntp | lệch |
|---|---|---|---|
| θ = 0,50 | 1931 | 1931 | +0 |
| θ = 0,30 | 2496 | 2496 | +0 |

## 2. Theo lát cắt

Chấm ở θ = 0,30. **Lát cắt chồng nhau** — một clip vừa `overlap` vừa `reverb` nằm ở cả hai hàng, nên tổng cột *clip* lớn hơn 1440 clip của tập. `sach` là clip không thuộc lát cắt khó nào; nó là mốc đối chiếu, không phải một lát cắt.

| lát cắt | clip | sự kiện thật | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|---|---|
| overlap | 437 | 1723 | 0.3336 | 0.7229 |
| low_snr | 390 | 1449 | 0.2913 | 0.7028 |
| reverb | 557 | 1888 | 0.3191 | 0.6916 |
| long_event | 144 | 559 | 0.2422 | 0.7373 |
| causal_chain | 203 | 845 | 0.3819 | 0.7166 |
| sach | 357 | 967 | 0.4007 | 0.7508 |

Phân loại lỗi **đếm lại trên từng tập con clip**, không chia tỉ lệ từ tổng — lát cắt chồng nhau nên cộng cột sẽ lớn hơn tổng toàn tập. Cột *phân mảnh/thật* là số sự kiện thật bị cắt thành nhiều mảnh, chia cho số sự kiện thật của chính lát cắt đó, nên so ngang giữa các hàng được.

| lát cắt | thật | Đúng | Boundary | Substitution | Deletion | Insertion | phân mảnh | phân mảnh/thật | gộp |
|---|---|---|---|---|---|---|---|---|---|
| overlap | 1723 | 824 | 369 | 277 | 253 | 1747 | 501 | 0,291 | 100 |
| low_snr | 1449 | 615 | 367 | 223 | 244 | 1569 | 417 | 0,288 | 60 |
| reverb | 1888 | 872 | 411 | 295 | 310 | 2000 | 509 | 0,270 | 86 |
| long_event | 559 | 257 | 146 | 93 | 63 | 1067 | 219 | 0,392 | 26 |
| causal_chain | 845 | 434 | 175 | 105 | 131 | 714 | 214 | 0,253 | 15 |
| sach | 967 | 580 | 173 | 132 | 82 | 1043 | 271 | 0,280 | 20 |

## 3. Ma trận nhầm lẫn

Hàng = lớp thật, cột = lớp đoán, `∅` = không có đối tác (Deletion ở cột, Insertion ở hàng). **Đường chéo gồm cả Đúng lẫn Boundary** — cả hai đều đúng lớp, chỉ khác thời điểm; đọc đường chéo như 'số đúng' là đọc sai.

```
         0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
  0    255    7    2    1    1    1    1    2    2    0    0    1    4    1    0   30   alarm_bell
  1      3  322    2    1    2    0    0    0    1    6    1    0    4    0    0   31   applause_cheering
  2      4    3  188    4    9    5    7    0    4   18    0    2    2    1    3   67   door_slam
  3      3    8    4  188    5    3    5    2    1    9    0    3    6    3    1   35   explosion
  4      3    4    4    5  287    3    7    0    1   11    0    1    2    0    4   40   fireworks
  5      2    1    2    2    5  270    4    1    2   10    1    1    3    0    0   24   glass_breaking
  6      1    4   13    8   10    2  150    0    0   24    0    0    3    0    1   52   gunshot
  7      4   11    2    0    4    1    0  227    0    5    2    9    4    1    1   42   laughter
  8     10    8   19    0   11   21    2    1  143   31    1    2    3    1    1   72   object_drop_dishes
  9      2   10    7    6   18    2   11    0    0  248    1    2    3    2    2   86   running_footsteps
 10      4    5    0    0    4    1    0    0    0    2  266    4    3    0    2   24   scream
 11      2    0    2    1    7    1    1    1    1    3    0  315    2    1    0   34   shout_yell
 12      3    3    0    0    1    0    0    0    0    1    0    0  260    0    0    6   siren
 13      4    8    1    1    4    2    0    3    0    3    2   19    3  225    3   35   speech_normal
 14      8   10   11    2   11    2    5    2    5   10    4    3    4    5  183   54   vehicle_crash
 15    472  743  188  128  839  293  155  150  120  621  259  399  490  299  133    0   ∅
```

Các cặp nhầm nhiều nhất, đối chiếu `confusable_with` khai trong `ontology_map.yaml`:

| thật | đoán | số | đã khai trong ontology |
|---|---|---|---|
| object_drop_dishes | running_footsteps | 31 | có |
| gunshot | running_footsteps | 24 | có |
| object_drop_dishes | glass_breaking | 21 | có |
| object_drop_dishes | door_slam | 19 | có |
| speech_normal | shout_yell | 19 | có |
| door_slam | running_footsteps | 18 | có |
| running_footsteps | fireworks | 18 | có |
| gunshot | door_slam | 13 | có |
| fireworks | running_footsteps | 11 | có |
| laughter | applause_cheering | 11 | **chưa** |
| object_drop_dishes | fireworks | 11 | **chưa** |
| running_footsteps | gunshot | 11 | có |
| vehicle_crash | door_slam | 11 | **chưa** |
| vehicle_crash | fireworks | 11 | **chưa** |
| glass_breaking | running_footsteps | 10 | **chưa** |

333/714 (47%) lượt nhầm rơi vào cặp đã khai. Phần còn lại là nhầm lẫn mà ontology **chưa dự đoán** — nếu tỉ lệ này cao thì thứ cần sửa là `confusable_with`, không phải model.

## 4. Ngưỡng theo từng lớp

θ tốt nhất **toàn cục** là 0,30. Bảng dưới cho thấy θ đó không tối ưu cho phần lớn lớp:

| lớp | sự kiện thật | θ* riêng | F1 @θ* riêng | F1 @θ=0,30 | chênh |
|---|---|---|---|---|---|
| alarm_bell | 308 | 0,40 | 0,3958 | 0,3952 | +0,0006 |
| applause_cheering | 373 | 0,20 | 0,2477 | 0,2211 | +0,0267 |
| door_slam | 317 | 0,35 | 0,3444 | 0,3281 | +0,0163 |
| explosion | 276 | 0,35 | 0,4957 | 0,4912 | +0,0046 |
| fireworks | 372 | 0,25 | 0,1949 | 0,1912 | +0,0037 |
| glass_breaking | 328 | 0,30 | 0,4492 | 0,4492 | +0,0000 |
| gunshot | 268 | 0,35 | 0,4526 | 0,4351 | +0,0175 |
| laughter | 313 | 0,30 | 0,4046 | 0,4046 | +0,0000 |
| object_drop_dishes | 326 | 0,30 | 0,4257 | 0,4257 | +0,0000 |
| running_footsteps | 400 | 0,40 | 0,3415 | 0,3181 | +0,0233 |
| scream | 315 | 0,30 | 0,5000 | 0,5000 | +0,0000 |
| shout_yell | 371 | 0,25 | 0,3541 | 0,3481 | +0,0060 |
| siren | 274 | 0,25 | 0,3801 | 0,3776 | +0,0025 |
| speech_normal | 313 | 0,35 | 0,3304 | 0,3216 | +0,0088 |
| vehicle_crash | 319 | 0,30 | 0,2940 | 0,2940 | +0,0000 |

| cấu hình ngưỡng | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|
| θ = 0,50 (mặc định lúc train) | 0,3252 | 0,7092 |
| θ = 0,30 toàn cục | 0,3466 | 0,7275 |
| θ riêng từng lớp | 0,3517 | 0,7340 |

Hàng cuối chọn 15 ngưỡng trên chính tập đang chấm, nên nó là **chặn trên**, không phải hiệu năng mong đợi trên dữ liệu mới. Muốn dùng số này làm kết quả thì ngưỡng phải khoá trên dev rồi đo một lần trên tập test độc lập — mà `gold_test` hiện **rỗng**.
