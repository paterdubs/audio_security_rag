# Phân tích lỗi — panns_ft_v3 trên dev/all

Sinh bởi `ml/evaluation/error_analysis.py` từ `ml/runs/panns_ft_v3/predictions/dev_all.npz` (1440 clip, 4873 sự kiện tham chiếu). Không train lại, không dùng GPU.

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với tập train của `v3`, nên dev **thiên vị v3 theo thiết kế**. Mọi so sánh v1/v2/v3 đọc trên trang này đều mang thiên vị đó.
> ⚠️ θ\* và θ theo từng lớp đều chọn **trên chính tập đang chấm** → rò rỉ. Chúng là chặn trên lạc quan, không phải kết quả báo cáo được.

## 1. Sáu loại lỗi

Tầng 1 gán cho mỗi sự kiện tham chiếu **đúng một** nhãn (Đúng/Boundary/Substitution/Deletion); dự báo không được tiêu thụ là Insertion. Fragmentation và Merge là **tầng 2**, đếm riêng và có thể chồng lên tầng 1.

Collar onset 200 ms, chỉ khớp onset (offset của tiếng vang quá mơ hồ để chấm).

**θ = 0,50** — 4873 sự kiện thật, 42676 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 2689 | 0,552 |
| Boundary | 1763 | 0,362 |
| Substitution | 408 | 0,084 |
| Deletion | 13 | 0,003 |
| Insertion | 37816 | 7,760 |
| Fragmentation (1 thật → n đoán) | 374 | — |
| Merge (n thật → 1 đoán) | 162 | — |

**θ = 0,85** — 4873 sự kiện thật, 6471 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 2479 | 0,509 |
| Boundary | 839 | 0,172 |
| Substitution | 692 | 0,142 |
| Deletion | 863 | 0,177 |
| Insertion | 2461 | 0,505 |
| Fragmentation (1 thật → n đoán) | 567 | — |
| Merge (n thật → 1 đoán) | 114 | — |

**Boundary không phải một rổ độc lập với sed_eval.** Một dự báo đúng lớp nhưng lệch onset quá collar bị sed_eval tính **một Deletion + một Insertion**. Bảng trên tách nó ra vì hai lỗi cần hai cách sửa khác nhau: lệch biên sửa ở hậu xử lý, còn không-nghe-ra-gì phải sửa ở dữ liệu. Số S/D/I gốc của sed_eval:

| θ | Nref | Nsys | Substitution | Deletion | Insertion | error rate |
|---|---|---|---|---|---|---|
| θ = 0,50 | 4873 | 42676 | 1513 | 671 | 38474 | 8,344 |
| θ = 0,85 | 4873 | 6471 | 601 | 1793 | 3391 | 1,187 |

Đối chiếu số khớp đúng giữa hai phép ghép (greedy ở đây vs khớp tối ưu trong sed_eval) — lệch lớn nghĩa là phép ghép này không đáng tin và bảng trên phải bỏ:

| θ | ghép greedy | sed_eval Ntp | lệch |
|---|---|---|---|
| θ = 0,50 | 2689 | 2689 | +0 |
| θ = 0,85 | 2479 | 2479 | +0 |

## 2. Theo lát cắt

Chấm ở θ = 0,85. **Lát cắt chồng nhau** — một clip vừa `overlap` vừa `reverb` nằm ở cả hai hàng, nên tổng cột *clip* lớn hơn 1440 clip của tập. `sach` là clip không thuộc lát cắt khó nào; nó là mốc đối chiếu, không phải một lát cắt.

| lát cắt | clip | sự kiện thật | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|---|---|
| overlap | 437 | 1723 | 0.4166 | 0.6928 |
| low_snr | 390 | 1449 | 0.3728 | 0.6684 |
| reverb | 557 | 1888 | 0.4151 | 0.6650 |
| long_event | 144 | 559 | 0.3123 | 0.6986 |
| causal_chain | 203 | 845 | 0.4836 | 0.6897 |
| sach | 357 | 967 | 0.4836 | 0.7267 |

Phân loại lỗi **đếm lại trên từng tập con clip**, không chia tỉ lệ từ tổng — lát cắt chồng nhau nên cộng cột sẽ lớn hơn tổng toàn tập. Cột *phân mảnh/thật* là số sự kiện thật bị cắt thành nhiều mảnh, chia cho số sự kiện thật của chính lát cắt đó, nên so ngang giữa các hàng được.

| lát cắt | thật | Đúng | Boundary | Substitution | Deletion | Insertion | phân mảnh | phân mảnh/thật | gộp |
|---|---|---|---|---|---|---|---|---|---|
| overlap | 1723 | 809 | 306 | 264 | 344 | 782 | 207 | 0,120 | 53 |
| low_snr | 1449 | 611 | 297 | 224 | 317 | 697 | 184 | 0,127 | 35 |
| reverb | 1888 | 900 | 319 | 280 | 389 | 949 | 217 | 0,115 | 45 |
| long_event | 559 | 233 | 139 | 97 | 90 | 464 | 130 | 0,233 | 17 |
| causal_chain | 845 | 457 | 126 | 87 | 175 | 375 | 73 | 0,086 | 13 |
| sach | 967 | 559 | 154 | 128 | 126 | 504 | 95 | 0,098 | 17 |

## 3. Ma trận nhầm lẫn

Hàng = lớp thật, cột = lớp đoán, `∅` = không có đối tác (Deletion ở cột, Insertion ở hàng). **Đường chéo gồm cả Đúng lẫn Boundary** — cả hai đều đúng lớp, chỉ khác thời điểm; đọc đường chéo như 'số đúng' là đọc sai.

```
         0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
  0    245    1    1    0    1    4    1    1    1    5    0    1    0    1    0   46   alarm_bell
  1      5  257    5    1    3    3    6    0    2    8    3    0    1    1    1   77   applause_cheering
  2      3    0  208    4    6    5   11    0    1   20    0    1    1    0    1   56   door_slam
  3      1    1   12  168    3    3   15    1    2    8    0    2    2    0    1   57   explosion
  4      3    4   10    5  231    1   12    0    0   39    1    0    2    1    0   63   fireworks
  5      2    0    9    2    0  265    9    1    3    9    0    0    0    1    1   26   glass_breaking
  6      2    2   13    2    4    1  175    0    0   15    0    1    1    1    0   51   gunshot
  7      0    5    3    0    1    1    1  221    2    3    1    6    0    1    1   67   laughter
  8      7    2   28    1    7   21    6    2  153   20    0    2    3    1    0   73   object_drop_dishes
  9      1    1    8    3   10    4    9    0    0  262    1    1    1    3    1   95   running_footsteps
 10      3    2    1    1    2    1    0    0    0    1  243    7    1    0    1   52   scream
 11      0    0    3    0    2    0    1    6    3    2    2  291    1    7    2   51   shout_yell
 12     10    0    0    1    2    1    1    1    0    4    2    1  228    2    1   20   siren
 13      0    3    2    1    2    2    1    3    1    7    1   21    0  223    0   46   speech_normal
 14      5    3   22    3    6    8    8    5    2    6    3    6    5    6  148   83   vehicle_crash
 15    210   95  276   20  197  188  289   68  162  518   39  168   51  147   33    0   ∅
```

Các cặp nhầm nhiều nhất, đối chiếu `confusable_with` khai trong `ontology_map.yaml`:

| thật | đoán | số | đã khai trong ontology |
|---|---|---|---|
| fireworks | running_footsteps | 39 | có |
| object_drop_dishes | door_slam | 28 | có |
| vehicle_crash | door_slam | 22 | **chưa** |
| object_drop_dishes | glass_breaking | 21 | có |
| speech_normal | shout_yell | 21 | có |
| door_slam | running_footsteps | 20 | có |
| object_drop_dishes | running_footsteps | 20 | có |
| explosion | gunshot | 15 | có |
| gunshot | running_footsteps | 15 | có |
| gunshot | door_slam | 13 | có |
| explosion | door_slam | 12 | có |
| fireworks | gunshot | 12 | có |
| door_slam | gunshot | 11 | có |
| fireworks | door_slam | 10 | có |
| running_footsteps | fireworks | 10 | có |

371/692 (54%) lượt nhầm rơi vào cặp đã khai. Phần còn lại là nhầm lẫn mà ontology **chưa dự đoán** — nếu tỉ lệ này cao thì thứ cần sửa là `confusable_with`, không phải model.

## 4. Ngưỡng theo từng lớp

θ tốt nhất **toàn cục** là 0,85. Bảng dưới cho thấy θ đó không tối ưu cho phần lớn lớp:

| lớp | sự kiện thật | θ* riêng | F1 @θ* riêng | F1 @θ=0,85 | chênh |
|---|---|---|---|---|---|
| alarm_bell | 308 | 0,85 | 0,4547 | 0,4547 | +0,0000 |
| applause_cheering | 373 | 0,55 | 0,5019 | 0,3178 | +0,1841 |
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
| siren | 274 | 0,55 | 0,6533 | 0,5394 | +0,1139 |
| speech_normal | 313 | 0,85 | 0,4294 | 0,4294 | +0,0000 |
| vehicle_crash | 319 | 0,85 | 0,3490 | 0,3490 | +0,0000 |

| cấu hình ngưỡng | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|
| θ = 0,50 (mặc định lúc train) | 0,1131 | 0,2862 |
| θ = 0,85 toàn cục | 0,4371 | 0,6988 |
| θ riêng từng lớp | 0,4731 | 0,7383 |

Hàng cuối chọn 15 ngưỡng trên chính tập đang chấm, nên nó là **chặn trên**, không phải hiệu năng mong đợi trên dữ liệu mới. Muốn dùng số này làm kết quả thì ngưỡng phải khoá trên dev rồi đo một lần trên tập test độc lập — mà `gold_test` hiện **rỗng**.
