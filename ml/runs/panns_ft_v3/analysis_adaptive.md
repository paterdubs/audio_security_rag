# Phân tích lỗi — panns_ft_v3 trên dev/all

Sinh bởi `ml/evaluation/error_analysis.py` từ `ml/runs/panns_ft_v3/predictions/dev_all.npz` (1440 clip, 4873 sự kiện tham chiếu). Không train lại, không dùng GPU.

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với tập train của `v3`, nên dev **thiên vị v3 theo thiết kế**. Mọi so sánh v1/v2/v3 đọc trên trang này đều mang thiên vị đó.
> ⚠️ θ\* và θ theo từng lớp đều chọn **trên chính tập đang chấm** → rò rỉ. Chúng là chặn trên lạc quan, không phải kết quả báo cáo được.

## 1. Sáu loại lỗi

Tầng 1 gán cho mỗi sự kiện tham chiếu **đúng một** nhãn (Đúng/Boundary/Substitution/Deletion); dự báo không được tiêu thụ là Insertion. Fragmentation và Merge là **tầng 2**, đếm riêng và có thể chồng lên tầng 1.

Collar onset 200 ms, chỉ khớp onset (offset của tiếng vang quá mơ hồ để chấm).

**θ = 0,50** — 4873 sự kiện thật, 44340 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 2735 | 0,561 |
| Boundary | 1754 | 0,360 |
| Substitution | 374 | 0,077 |
| Deletion | 10 | 0,002 |
| Insertion | 39477 | 8,101 |
| Fragmentation (1 thật → n đoán) | 441 | — |
| Merge (n thật → 1 đoán) | 163 | — |

**θ = 0,85** — 4873 sự kiện thật, 6796 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 2536 | 0,520 |
| Boundary | 862 | 0,177 |
| Substitution | 701 | 0,144 |
| Deletion | 774 | 0,159 |
| Insertion | 2697 | 0,553 |
| Fragmentation (1 thật → n đoán) | 659 | — |
| Merge (n thật → 1 đoán) | 124 | — |

**Boundary không phải một rổ độc lập với sed_eval.** Một dự báo đúng lớp nhưng lệch onset quá collar bị sed_eval tính **một Deletion + một Insertion**. Bảng trên tách nó ra vì hai lỗi cần hai cách sửa khác nhau: lệch biên sửa ở hậu xử lý, còn không-nghe-ra-gì phải sửa ở dữ liệu. Số S/D/I gốc của sed_eval:

| θ | Nref | Nsys | Substitution | Deletion | Insertion | error rate |
|---|---|---|---|---|---|---|
| θ = 0,50 | 4873 | 44340 | 1486 | 652 | 40119 | 8,672 |
| θ = 0,85 | 4873 | 6796 | 612 | 1725 | 3648 | 1,228 |

Đối chiếu số khớp đúng giữa hai phép ghép (greedy ở đây vs khớp tối ưu trong sed_eval) — lệch lớn nghĩa là phép ghép này không đáng tin và bảng trên phải bỏ:

| θ | ghép greedy | sed_eval Ntp | lệch |
|---|---|---|---|
| θ = 0,50 | 2735 | 2735 | +0 |
| θ = 0,85 | 2536 | 2536 | +0 |

## 2. Theo lát cắt

Chấm ở θ = 0,85. **Lát cắt chồng nhau** — một clip vừa `overlap` vừa `reverb` nằm ở cả hai hàng, nên tổng cột *clip* lớn hơn 1440 clip của tập. `sach` là clip không thuộc lát cắt khó nào; nó là mốc đối chiếu, không phải một lát cắt.

| lát cắt | clip | sự kiện thật | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|---|---|
| overlap | 437 | 1723 | 0.4135 | 0.7022 |
| low_snr | 390 | 1449 | 0.3694 | 0.6780 |
| reverb | 557 | 1888 | 0.4112 | 0.6737 |
| long_event | 144 | 559 | 0.3067 | 0.7066 |
| causal_chain | 203 | 845 | 0.4798 | 0.6944 |
| sach | 357 | 967 | 0.4846 | 0.7327 |

Phân loại lỗi **đếm lại trên từng tập con clip**, không chia tỉ lệ từ tổng — lát cắt chồng nhau nên cộng cột sẽ lớn hơn tổng toàn tập. Cột *phân mảnh/thật* là số sự kiện thật bị cắt thành nhiều mảnh, chia cho số sự kiện thật của chính lát cắt đó, nên so ngang giữa các hàng được.

| lát cắt | thật | Đúng | Boundary | Substitution | Deletion | Insertion | phân mảnh | phân mảnh/thật | gộp |
|---|---|---|---|---|---|---|---|---|---|
| overlap | 1723 | 828 | 316 | 267 | 312 | 871 | 242 | 0,140 | 58 |
| low_snr | 1449 | 627 | 307 | 224 | 291 | 788 | 213 | 0,147 | 40 |
| reverb | 1888 | 919 | 329 | 288 | 352 | 1046 | 252 | 0,133 | 49 |
| long_event | 559 | 236 | 143 | 97 | 83 | 504 | 147 | 0,263 | 17 |
| causal_chain | 845 | 463 | 131 | 94 | 157 | 397 | 83 | 0,098 | 14 |
| sach | 967 | 574 | 156 | 131 | 106 | 541 | 107 | 0,111 | 18 |

## 3. Ma trận nhầm lẫn

Hàng = lớp thật, cột = lớp đoán, `∅` = không có đối tác (Deletion ở cột, Insertion ở hàng). **Đường chéo gồm cả Đúng lẫn Boundary** — cả hai đều đúng lớp, chỉ khác thời điểm; đọc đường chéo như 'số đúng' là đọc sai.

```
         0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
  0    245    3    1    0    1    4    1    1    1    5    0    1    0    1    0   44   alarm_bell
  1      3  288    2    2    1    0    5    0    2    6    2    0    1    0    1   60   applause_cheering
  2      3    1  208    6    6    5   11    0    1   20    0    1    1    0    1   53   door_slam
  3      1    2   10  189    4    3   10    1    1    8    0    1    3    0    1   42   explosion
  4      3    6   10    7  231    1   11    0    0   39    0    0    4    1    0   59   fireworks
  5      2    0    9    2    0  265    9    1    3    9    0    0    0    1    1   26   glass_breaking
  6      1    2   12    5    4    1  175    0    0   16    0    1    0    1    0   50   gunshot
  7      0    7    3    0    1    1    1  221    2    3    2    6    1    1    1   63   laughter
  8      7    2   27    2    7   22    6    2  153   20    0    2    3    1    0   72   object_drop_dishes
  9      1    5    8    6   11    5    9    0    0  262    1    1    3    3    1   84   running_footsteps
 10      3    3    1    0    2    1    0    0    0    0  256    7    0    0    1   41   scream
 11      0    0    2    0    2    0    1    6    3    2    5  291    0    7    2   50   shout_yell
 12      8    1    0    1    1    0    1    1    0    1    2    0  243    1    1   13   siren
 13      0    5    2    1    2    2    1    3    1    7    3   22    0  223    0   41   speech_normal
 14      5    6   21    8    7    8    7    5    2    6    4    6    5    5  148   76   vehicle_crash
 15    215  173  285   46  197  190  297   68  163  523   66  169  122  150   33    0   ∅
```

Các cặp nhầm nhiều nhất, đối chiếu `confusable_with` khai trong `ontology_map.yaml`:

| thật | đoán | số | đã khai trong ontology |
|---|---|---|---|
| fireworks | running_footsteps | 39 | có |
| object_drop_dishes | door_slam | 27 | có |
| object_drop_dishes | glass_breaking | 22 | có |
| speech_normal | shout_yell | 22 | có |
| vehicle_crash | door_slam | 21 | **chưa** |
| door_slam | running_footsteps | 20 | có |
| object_drop_dishes | running_footsteps | 20 | có |
| gunshot | running_footsteps | 16 | có |
| gunshot | door_slam | 12 | có |
| door_slam | gunshot | 11 | có |
| fireworks | gunshot | 11 | có |
| running_footsteps | fireworks | 11 | có |
| explosion | door_slam | 10 | có |
| explosion | gunshot | 10 | có |
| fireworks | door_slam | 10 | có |

380/701 (54%) lượt nhầm rơi vào cặp đã khai. Phần còn lại là nhầm lẫn mà ontology **chưa dự đoán** — nếu tỉ lệ này cao thì thứ cần sửa là `confusable_with`, không phải model.

## 4. Ngưỡng theo từng lớp

θ tốt nhất **toàn cục** là 0,85. Bảng dưới cho thấy θ đó không tối ưu cho phần lớn lớp:

| lớp | sự kiện thật | θ* riêng | F1 @θ* riêng | F1 @θ=0,85 | chênh |
|---|---|---|---|---|---|
| alarm_bell | 308 | 0,85 | 0,4547 | 0,4547 | +0,0000 |
| applause_cheering | 373 | 0,60 | 0,4430 | 0,3056 | +0,1374 |
| door_slam | 317 | 0,90 | 0,3514 | 0,3290 | +0,0224 |
| explosion | 276 | 0,85 | 0,5771 | 0,5771 | +0,0000 |
| fireworks | 372 | 0,80 | 0,3387 | 0,3110 | +0,0277 |
| glass_breaking | 328 | 0,90 | 0,5444 | 0,5335 | +0,0109 |
| gunshot | 268 | 0,90 | 0,4422 | 0,3911 | +0,0510 |
| laughter | 313 | 0,85 | 0,5788 | 0,5788 | +0,0000 |
| object_drop_dishes | 326 | 0,85 | 0,4043 | 0,4043 | +0,0000 |
| running_footsteps | 400 | 0,90 | 0,3866 | 0,3557 | +0,0309 |
| scream | 315 | 0,85 | 0,6677 | 0,6677 | +0,0000 |
| shout_yell | 371 | 0,80 | 0,5047 | 0,4892 | +0,0155 |
| siren | 274 | 0,60 | 0,5820 | 0,5182 | +0,0638 |
| speech_normal | 313 | 0,85 | 0,4294 | 0,4294 | +0,0000 |
| vehicle_crash | 319 | 0,85 | 0,3490 | 0,3490 | +0,0000 |

| cấu hình ngưỡng | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|
| θ = 0,50 (mặc định lúc train) | 0,1111 | 0,2826 |
| θ = 0,85 toàn cục | 0,4347 | 0,7061 |
| θ riêng từng lớp | 0,4652 | 0,7393 |

Hàng cuối chọn 15 ngưỡng trên chính tập đang chấm, nên nó là **chặn trên**, không phải hiệu năng mong đợi trên dữ liệu mới. Muốn dùng số này làm kết quả thì ngưỡng phải khoá trên dev rồi đo một lần trên tập test độc lập — mà `gold_test` hiện **rỗng**.
