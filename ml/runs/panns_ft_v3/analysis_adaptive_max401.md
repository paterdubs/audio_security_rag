# Phân tích lỗi — panns_ft_v3 trên dev/all

Sinh bởi `ml/evaluation/error_analysis.py` từ `ml/runs/panns_ft_v3/predictions/dev_all.npz` (1440 clip, 4873 sự kiện tham chiếu). Không train lại, không dùng GPU.

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với tập train của `v3`, nên dev **thiên vị v3 theo thiết kế**. Mọi so sánh v1/v2/v3 đọc trên trang này đều mang thiên vị đó.
> ⚠️ θ\* và θ theo từng lớp đều chọn **trên chính tập đang chấm** → rò rỉ. Chúng là chặn trên lạc quan, không phải kết quả báo cáo được.

## 1. Sáu loại lỗi

Tầng 1 gán cho mỗi sự kiện tham chiếu **đúng một** nhãn (Đúng/Boundary/Substitution/Deletion); dự báo không được tiêu thụ là Insertion. Fragmentation và Merge là **tầng 2**, đếm riêng và có thể chồng lên tầng 1.

Collar onset 200 ms, chỉ khớp onset (offset của tiếng vang quá mơ hồ để chấm).

**θ = 0,50** — 4873 sự kiện thật, 42446 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 2643 | 0,542 |
| Boundary | 1766 | 0,362 |
| Substitution | 451 | 0,093 |
| Deletion | 13 | 0,003 |
| Insertion | 37586 | 7,713 |
| Fragmentation (1 thật → n đoán) | 338 | — |
| Merge (n thật → 1 đoán) | 164 | — |

**θ = 0,85** — 4873 sự kiện thật, 6313 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 2422 | 0,497 |
| Boundary | 842 | 0,173 |
| Substitution | 698 | 0,143 |
| Deletion | 911 | 0,187 |
| Insertion | 2351 | 0,482 |
| Fragmentation (1 thật → n đoán) | 490 | — |
| Merge (n thật → 1 đoán) | 109 | — |

**Boundary không phải một rổ độc lập với sed_eval.** Một dự báo đúng lớp nhưng lệch onset quá collar bị sed_eval tính **một Deletion + một Insertion**. Bảng trên tách nó ra vì hai lỗi cần hai cách sửa khác nhau: lệch biên sửa ở hậu xử lý, còn không-nghe-ra-gì phải sửa ở dữ liệu. Số S/D/I gốc của sed_eval:

| θ | Nref | Nsys | Substitution | Deletion | Insertion | error rate |
|---|---|---|---|---|---|---|
| θ = 0,50 | 4873 | 42446 | 1548 | 682 | 38255 | 8,308 |
| θ = 0,85 | 4873 | 6313 | 601 | 1850 | 3290 | 1,178 |

Đối chiếu số khớp đúng giữa hai phép ghép (greedy ở đây vs khớp tối ưu trong sed_eval) — lệch lớn nghĩa là phép ghép này không đáng tin và bảng trên phải bỏ:

| θ | ghép greedy | sed_eval Ntp | lệch |
|---|---|---|---|
| θ = 0,50 | 2643 | 2643 | +0 |
| θ = 0,85 | 2422 | 2422 | +0 |

## 2. Theo lát cắt

Chấm ở θ = 0,85. **Lát cắt chồng nhau** — một clip vừa `overlap` vừa `reverb` nằm ở cả hai hàng, nên tổng cột *clip* lớn hơn 1440 clip của tập. `sach` là clip không thuộc lát cắt khó nào; nó là mốc đối chiếu, không phải một lát cắt.

| lát cắt | clip | sự kiện thật | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|---|---|
| overlap | 437 | 1723 | 0.4101 | 0.6833 |
| low_snr | 390 | 1449 | 0.3691 | 0.6540 |
| reverb | 557 | 1888 | 0.4081 | 0.6552 |
| long_event | 144 | 559 | 0.3115 | 0.6890 |
| causal_chain | 203 | 845 | 0.4830 | 0.6830 |
| sach | 357 | 967 | 0.4799 | 0.7191 |

Phân loại lỗi **đếm lại trên từng tập con clip**, không chia tỉ lệ từ tổng — lát cắt chồng nhau nên cộng cột sẽ lớn hơn tổng toàn tập. Cột *phân mảnh/thật* là số sự kiện thật bị cắt thành nhiều mảnh, chia cho số sự kiện thật của chính lát cắt đó, nên so ngang giữa các hàng được.

| lát cắt | thật | Đúng | Boundary | Substitution | Deletion | Insertion | phân mảnh | phân mảnh/thật | gộp |
|---|---|---|---|---|---|---|---|---|---|
| overlap | 1723 | 786 | 307 | 265 | 365 | 752 | 178 | 0,103 | 50 |
| low_snr | 1449 | 592 | 295 | 228 | 334 | 644 | 150 | 0,104 | 31 |
| reverb | 1888 | 873 | 330 | 281 | 404 | 906 | 182 | 0,096 | 42 |
| long_event | 559 | 228 | 140 | 95 | 96 | 442 | 116 | 0,208 | 17 |
| causal_chain | 845 | 454 | 125 | 90 | 176 | 366 | 69 | 0,082 | 13 |
| sach | 967 | 549 | 151 | 132 | 135 | 489 | 85 | 0,088 | 18 |

## 3. Ma trận nhầm lẫn

Hàng = lớp thật, cột = lớp đoán, `∅` = không có đối tác (Deletion ở cột, Insertion ở hàng). **Đường chéo gồm cả Đúng lẫn Boundary** — cả hai đều đúng lớp, chỉ khác thời điểm; đọc đường chéo như 'số đúng' là đọc sai.

```
         0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
  0    245    1    1    0    1    4    1    1    1    5    0    0    0    1    0   47   alarm_bell
  1      4  242    6    1    2    3    6    0    2   11    3    1    0    1    1   90   applause_cheering
  2      3    2  208    4    6    5   11    0    1   19    0    1    0    0    1   56   door_slam
  3      1    1   12  168    3    3   15    1    2    8    0    2    0    0    1   59   explosion
  4      3    3   10    6  231    1   12    0    0   40    1    0    0    1    0   64   fireworks
  5      2    0    9    2    0  265    9    1    3    9    0    0    0    1    1   26   glass_breaking
  6      2    1   13    2    4    0  175    0    0   15    0    1    0    1    0   54   gunshot
  7      0    4    3    0    1    1    1  221    2    3    1    6    0    1    1   68   laughter
  8      8    0   27    1    7   21    6    2  153   21    0    2    1    1    0   76   object_drop_dishes
  9      1    1    8    3   11    4    9    0    0  262    1    1    0    3    1   95   running_footsteps
 10      3    1    1    1    2    1    0    0    0    1  243    7    0    0    1   54   scream
 11      0    0    3    0    2    0    1    6    3    2    2  291    0    7    2   52   shout_yell
 12     12    3    3    1    2    5    3    2    0    6    3    5  189    4    1   35   siren
 13      0    2    2    1    2    2    1    3    1    7    1   21    0  223    0   47   speech_normal
 14      5    2   22    3    6    8    8    5    2    7    3    6    0    6  148   88   vehicle_crash
 15    208   57  273   19  197  185  287   67  162  511   38  164    5  145   33    0   ∅
```

Các cặp nhầm nhiều nhất, đối chiếu `confusable_with` khai trong `ontology_map.yaml`:

| thật | đoán | số | đã khai trong ontology |
|---|---|---|---|
| fireworks | running_footsteps | 40 | có |
| object_drop_dishes | door_slam | 27 | có |
| vehicle_crash | door_slam | 22 | **chưa** |
| object_drop_dishes | glass_breaking | 21 | có |
| object_drop_dishes | running_footsteps | 21 | có |
| speech_normal | shout_yell | 21 | có |
| door_slam | running_footsteps | 19 | có |
| explosion | gunshot | 15 | có |
| gunshot | running_footsteps | 15 | có |
| gunshot | door_slam | 13 | có |
| explosion | door_slam | 12 | có |
| fireworks | gunshot | 12 | có |
| siren | alarm_bell | 12 | có |
| applause_cheering | running_footsteps | 11 | có |
| door_slam | gunshot | 11 | có |

373/698 (53%) lượt nhầm rơi vào cặp đã khai. Phần còn lại là nhầm lẫn mà ontology **chưa dự đoán** — nếu tỉ lệ này cao thì thứ cần sửa là `confusable_with`, không phải model.

## 4. Ngưỡng theo từng lớp

θ tốt nhất **toàn cục** là 0,85. Bảng dưới cho thấy θ đó không tối ưu cho phần lớn lớp:

| lớp | sự kiện thật | θ* riêng | F1 @θ* riêng | F1 @θ=0,85 | chênh |
|---|---|---|---|---|---|
| alarm_bell | 308 | 0,85 | 0,4547 | 0,4547 | +0,0000 |
| applause_cheering | 373 | 0,50 | 0,5074 | 0,3001 | +0,2072 |
| door_slam | 317 | 0,90 | 0,3514 | 0,3290 | +0,0224 |
| explosion | 276 | 0,85 | 0,5902 | 0,5902 | +0,0000 |
| fireworks | 372 | 0,80 | 0,3387 | 0,3110 | +0,0277 |
| glass_breaking | 328 | 0,90 | 0,5444 | 0,5335 | +0,0109 |
| gunshot | 268 | 0,90 | 0,4422 | 0,3911 | +0,0510 |
| laughter | 313 | 0,85 | 0,5788 | 0,5788 | +0,0000 |
| object_drop_dishes | 326 | 0,85 | 0,4043 | 0,4043 | +0,0000 |
| running_footsteps | 400 | 0,90 | 0,3866 | 0,3557 | +0,0309 |
| scream | 315 | 0,85 | 0,6841 | 0,6841 | +0,0000 |
| shout_yell | 371 | 0,80 | 0,5047 | 0,4892 | +0,0155 |
| siren | 274 | 0,50 | 0,6548 | 0,4776 | +0,1771 |
| speech_normal | 313 | 0,85 | 0,4294 | 0,4294 | +0,0000 |
| vehicle_crash | 319 | 0,85 | 0,3490 | 0,3490 | +0,0000 |

| cấu hình ngưỡng | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|
| θ = 0,50 (mặc định lúc train) | 0,1117 | 0,2850 |
| θ = 0,85 toàn cục | 0,4330 | 0,6894 |
| θ riêng từng lớp | 0,4719 | 0,7349 |

Hàng cuối chọn 15 ngưỡng trên chính tập đang chấm, nên nó là **chặn trên**, không phải hiệu năng mong đợi trên dữ liệu mới. Muốn dùng số này làm kết quả thì ngưỡng phải khoá trên dev rồi đo một lần trên tập test độc lập — mà `gold_test` hiện **rỗng**.
