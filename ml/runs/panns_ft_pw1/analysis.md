# Phân tích lỗi — panns_ft_pw1 trên dev/all

Sinh bởi `ml/evaluation/error_analysis.py` từ `ml/runs/panns_ft_pw1/predictions/dev_all.npz` (1440 clip, 4873 sự kiện tham chiếu). Không train lại, không dùng GPU.

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với tập train của `v3`, nên dev **thiên vị v3 theo thiết kế**. Mọi so sánh v1/v2/v3 đọc trên trang này đều mang thiên vị đó.
> ⚠️ θ\* và θ theo từng lớp đều chọn **trên chính tập đang chấm** → rò rỉ. Chúng là chặn trên lạc quan, không phải kết quả báo cáo được.

## 1. Sáu loại lỗi

Tầng 1 gán cho mỗi sự kiện tham chiếu **đúng một** nhãn (Đúng/Boundary/Substitution/Deletion); dự báo không được tiêu thụ là Insertion. Fragmentation và Merge là **tầng 2**, đếm riêng và có thể chồng lên tầng 1.

Collar onset 200 ms, chỉ khớp onset (offset của tiếng vang quá mơ hồ để chấm).

**θ = 0,50** — 4873 sự kiện thật, 5309 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 2118 | 0,435 |
| Boundary | 888 | 0,182 |
| Substitution | 550 | 0,113 |
| Deletion | 1317 | 0,270 |
| Insertion | 1753 | 0,360 |
| Fragmentation (1 thật → n đoán) | 862 | — |
| Merge (n thật → 1 đoán) | 136 | — |

**θ = 0,35** — 4873 sự kiện thật, 6555 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 2476 | 0,508 |
| Boundary | 900 | 0,185 |
| Substitution | 676 | 0,139 |
| Deletion | 821 | 0,169 |
| Insertion | 2503 | 0,514 |
| Fragmentation (1 thật → n đoán) | 886 | — |
| Merge (n thật → 1 đoán) | 134 | — |

**Boundary không phải một rổ độc lập với sed_eval.** Một dự báo đúng lớp nhưng lệch onset quá collar bị sed_eval tính **một Deletion + một Insertion**. Bảng trên tách nó ra vì hai lỗi cần hai cách sửa khác nhau: lệch biên sửa ở hậu xử lý, còn không-nghe-ra-gì phải sửa ở dữ liệu. Số S/D/I gốc của sed_eval:

| θ | Nref | Nsys | Substitution | Deletion | Insertion | error rate |
|---|---|---|---|---|---|---|
| θ = 0,50 | 4873 | 5309 | 377 | 2378 | 2814 | 1,143 |
| θ = 0,35 | 4873 | 6555 | 591 | 1806 | 3488 | 1,208 |

Đối chiếu số khớp đúng giữa hai phép ghép (greedy ở đây vs khớp tối ưu trong sed_eval) — lệch lớn nghĩa là phép ghép này không đáng tin và bảng trên phải bỏ:

| θ | ghép greedy | sed_eval Ntp | lệch |
|---|---|---|---|
| θ = 0,50 | 2118 | 2118 | +0 |
| θ = 0,35 | 2476 | 2476 | +0 |

## 2. Theo lát cắt

Chấm ở θ = 0,35. **Lát cắt chồng nhau** — một clip vừa `overlap` vừa `reverb` nằm ở cả hai hàng, nên tổng cột *clip* lớn hơn 1440 clip của tập. `sach` là clip không thuộc lát cắt khó nào; nó là mốc đối chiếu, không phải một lát cắt.

| lát cắt | clip | sự kiện thật | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|---|---|
| overlap | 437 | 1723 | 0.4053 | 0.7411 |
| low_snr | 390 | 1449 | 0.3680 | 0.7201 |
| reverb | 557 | 1888 | 0.4085 | 0.7205 |
| long_event | 144 | 559 | 0.3182 | 0.7543 |
| causal_chain | 203 | 845 | 0.4783 | 0.7349 |
| sach | 357 | 967 | 0.4919 | 0.7726 |

Phân loại lỗi **đếm lại trên từng tập con clip**, không chia tỉ lệ từ tổng — lát cắt chồng nhau nên cộng cột sẽ lớn hơn tổng toàn tập. Cột *phân mảnh/thật* là số sự kiện thật bị cắt thành nhiều mảnh, chia cho số sự kiện thật của chính lát cắt đó, nên so ngang giữa các hàng được.

| lát cắt | thật | Đúng | Boundary | Substitution | Deletion | Insertion | phân mảnh | phân mảnh/thật | gộp |
|---|---|---|---|---|---|---|---|---|---|
| overlap | 1723 | 800 | 337 | 244 | 342 | 844 | 325 | 0,189 | 70 |
| low_snr | 1449 | 619 | 311 | 216 | 303 | 769 | 275 | 0,190 | 40 |
| reverb | 1888 | 887 | 350 | 266 | 385 | 952 | 347 | 0,184 | 55 |
| long_event | 559 | 249 | 136 | 85 | 89 | 536 | 167 | 0,299 | 18 |
| causal_chain | 845 | 440 | 143 | 102 | 160 | 310 | 118 | 0,140 | 12 |
| sach | 967 | 561 | 157 | 138 | 111 | 458 | 157 | 0,162 | 19 |

## 3. Ma trận nhầm lẫn

Hàng = lớp thật, cột = lớp đoán, `∅` = không có đối tác (Deletion ở cột, Insertion ở hàng). **Đường chéo gồm cả Đúng lẫn Boundary** — cả hai đều đúng lớp, chỉ khác thời điểm; đọc đường chéo như 'số đúng' là đọc sai.

```
         0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
  0    252    4    1    0    1    2    0    1    1    0    0    1    2    0    0   43   alarm_bell
  1      3  312    0    0    1    0    0    0    2    6    1    0    1    0    0   47   applause_cheering
  2      4    3  175    6    8    3    8    0    4   31    0    2    2    1    1   69   door_slam
  3      0    5    5  184    5    1    5    1    0    6    0    1    4    1    2   56   explosion
  4      2    4    4    6  279    2    5    0    0   18    0    2    3    0    2   45   fireworks
  5      2    1    2    2    4  256    5    1    7    9    1    1    0    2    3   32   glass_breaking
  6      2    7   14    8   12    0  134    0    0   16    0    1    1    0    3   70   gunshot
  7      2    7    1    0    5    1    0  221    2    2    1    8    3    1    0   59   laughter
  8      9    6   24    0   15   24    5    1  117   23    1    2    4    0    0   95   object_drop_dishes
  9      2    9   12    4   23    1    7    0    4  221    1    2    3    2    3  106   running_footsteps
 10      4    4    0    0    3    1    0    0    0    1  267    5    1    0    1   28   scream
 11      2    1    2    0    1    0    1    4    2    3    2  316    1    1    1   34   shout_yell
 12      3    1    0    0    1    0    0    0    0    2    0    0  256    0    1   10   siren
 13      3    6    1    1    3    1    0    3    0    4    2   24    2  217    0   46   speech_normal
 14      6    9    8    2    8    5    2    2    2    9    2    6    5    3  169   81   vehicle_crash
 15    277  301   87   65  413  141   85   65   59  257  110  225  198  156   64    0   ∅
```

Các cặp nhầm nhiều nhất, đối chiếu `confusable_with` khai trong `ontology_map.yaml`:

| thật | đoán | số | đã khai trong ontology |
|---|---|---|---|
| door_slam | running_footsteps | 31 | có |
| object_drop_dishes | door_slam | 24 | có |
| object_drop_dishes | glass_breaking | 24 | có |
| speech_normal | shout_yell | 24 | có |
| object_drop_dishes | running_footsteps | 23 | có |
| running_footsteps | fireworks | 23 | có |
| fireworks | running_footsteps | 18 | có |
| gunshot | running_footsteps | 16 | có |
| object_drop_dishes | fireworks | 15 | **chưa** |
| gunshot | door_slam | 14 | có |
| gunshot | fireworks | 12 | có |
| running_footsteps | door_slam | 12 | có |
| glass_breaking | running_footsteps | 9 | **chưa** |
| object_drop_dishes | alarm_bell | 9 | **chưa** |
| running_footsteps | applause_cheering | 9 | có |

374/676 (55%) lượt nhầm rơi vào cặp đã khai. Phần còn lại là nhầm lẫn mà ontology **chưa dự đoán** — nếu tỉ lệ này cao thì thứ cần sửa là `confusable_with`, không phải model.

## 4. Ngưỡng theo từng lớp

θ tốt nhất **toàn cục** là 0,35. Bảng dưới cho thấy θ đó không tối ưu cho phần lớn lớp:

| lớp | sự kiện thật | θ* riêng | F1 @θ* riêng | F1 @θ=0,35 | chênh |
|---|---|---|---|---|---|
| alarm_bell | 308 | 0,55 | 0,4592 | 0,4518 | +0,0074 |
| applause_cheering | 373 | 0,20 | 0,3761 | 0,3191 | +0,0571 |
| door_slam | 317 | 0,30 | 0,3559 | 0,3553 | +0,0006 |
| explosion | 276 | 0,35 | 0,5704 | 0,5704 | +0,0000 |
| fireworks | 372 | 0,30 | 0,2878 | 0,2860 | +0,0019 |
| glass_breaking | 328 | 0,30 | 0,5361 | 0,5352 | +0,0009 |
| gunshot | 268 | 0,40 | 0,4715 | 0,4648 | +0,0068 |
| laughter | 313 | 0,35 | 0,5490 | 0,5490 | +0,0000 |
| object_drop_dishes | 326 | 0,25 | 0,4301 | 0,3992 | +0,0309 |
| running_footsteps | 400 | 0,45 | 0,4089 | 0,3849 | +0,0240 |
| scream | 315 | 0,30 | 0,6246 | 0,6174 | +0,0073 |
| shout_yell | 371 | 0,25 | 0,4801 | 0,4716 | +0,0086 |
| siren | 274 | 0,20 | 0,5302 | 0,4974 | +0,0328 |
| speech_normal | 313 | 0,30 | 0,4160 | 0,4075 | +0,0085 |
| vehicle_crash | 319 | 0,50 | 0,3614 | 0,3515 | +0,0100 |

| cấu hình ngưỡng | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|
| θ = 0,50 (mặc định lúc train) | 0,4160 | 0,7338 |
| θ = 0,35 toàn cục | 0,4333 | 0,7486 |
| θ riêng từng lớp | 0,4476 | 0,7516 |

Hàng cuối chọn 15 ngưỡng trên chính tập đang chấm, nên nó là **chặn trên**, không phải hiệu năng mong đợi trên dữ liệu mới. Muốn dùng số này làm kết quả thì ngưỡng phải khoá trên dev rồi đo một lần trên tập test độc lập — mà `gold_test` hiện **rỗng**.
