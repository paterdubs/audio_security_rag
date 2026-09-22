# Phân tích lỗi — panns_ft_tpb2 trên dev/all

Sinh bởi `ml/evaluation/error_analysis.py` từ `ml/runs/panns_ft_tpb2/predictions/dev_all.npz` (1440 clip, 4873 sự kiện tham chiếu). Không train lại, không dùng GPU.

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với tập train của `v3`, nên dev **thiên vị v3 theo thiết kế**. Mọi so sánh v1/v2/v3 đọc trên trang này đều mang thiên vị đó.
> ⚠️ θ\* và θ theo từng lớp đều chọn **trên chính tập đang chấm** → rò rỉ. Chúng là chặn trên lạc quan, không phải kết quả báo cáo được.

## 1. Sáu loại lỗi

Tầng 1 gán cho mỗi sự kiện tham chiếu **đúng một** nhãn (Đúng/Boundary/Substitution/Deletion); dự báo không được tiêu thụ là Insertion. Fragmentation và Merge là **tầng 2**, đếm riêng và có thể chồng lên tầng 1.

Collar onset 200 ms, chỉ khớp onset (offset của tiếng vang quá mơ hồ để chấm).

**θ = 0,50** — 4873 sự kiện thật, 3632 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 1380 | 0,283 |
| Boundary | 809 | 0,166 |
| Substitution | 360 | 0,074 |
| Deletion | 2324 | 0,477 |
| Insertion | 1083 | 0,222 |
| Fragmentation (1 thật → n đoán) | 631 | — |
| Merge (n thật → 1 đoán) | 103 | — |

**θ = 0,25** — 4873 sự kiện thật, 5650 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 2164 | 0,444 |
| Boundary | 837 | 0,172 |
| Substitution | 642 | 0,132 |
| Deletion | 1230 | 0,252 |
| Insertion | 2007 | 0,412 |
| Fragmentation (1 thật → n đoán) | 710 | — |
| Merge (n thật → 1 đoán) | 117 | — |

**Boundary không phải một rổ độc lập với sed_eval.** Một dự báo đúng lớp nhưng lệch onset quá collar bị sed_eval tính **một Deletion + một Insertion**. Bảng trên tách nó ra vì hai lỗi cần hai cách sửa khác nhau: lệch biên sửa ở hậu xử lý, còn không-nghe-ra-gì phải sửa ở dữ liệu. Số S/D/I gốc của sed_eval:

| θ | Nref | Nsys | Substitution | Deletion | Insertion | error rate |
|---|---|---|---|---|---|---|
| θ = 0,50 | 4873 | 3632 | 181 | 3312 | 2071 | 1,142 |
| θ = 0,25 | 4873 | 5650 | 491 | 2218 | 2995 | 1,171 |

Đối chiếu số khớp đúng giữa hai phép ghép (greedy ở đây vs khớp tối ưu trong sed_eval) — lệch lớn nghĩa là phép ghép này không đáng tin và bảng trên phải bỏ:

| θ | ghép greedy | sed_eval Ntp | lệch |
|---|---|---|---|
| θ = 0,50 | 1380 | 1380 | +0 |
| θ = 0,25 | 2164 | 2164 | +0 |

## 2. Theo lát cắt

Chấm ở θ = 0,25. **Lát cắt chồng nhau** — một clip vừa `overlap` vừa `reverb` nằm ở cả hai hàng, nên tổng cột *clip* lớn hơn 1440 clip của tập. `sach` là clip không thuộc lát cắt khó nào; nó là mốc đối chiếu, không phải một lát cắt.

| lát cắt | clip | sự kiện thật | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|---|---|
| overlap | 437 | 1723 | 0.3959 | 0.7004 |
| low_snr | 390 | 1449 | 0.3425 | 0.6665 |
| reverb | 557 | 1888 | 0.3767 | 0.6630 |
| long_event | 144 | 559 | 0.3066 | 0.7058 |
| causal_chain | 203 | 845 | 0.4479 | 0.6950 |
| sach | 357 | 967 | 0.4749 | 0.7436 |

Phân loại lỗi **đếm lại trên từng tập con clip**, không chia tỉ lệ từ tổng — lát cắt chồng nhau nên cộng cột sẽ lớn hơn tổng toàn tập. Cột *phân mảnh/thật* là số sự kiện thật bị cắt thành nhiều mảnh, chia cho số sự kiện thật của chính lát cắt đó, nên so ngang giữa các hàng được.

| lát cắt | thật | Đúng | Boundary | Substitution | Deletion | Insertion | phân mảnh | phân mảnh/thật | gộp |
|---|---|---|---|---|---|---|---|---|---|
| overlap | 1723 | 714 | 290 | 226 | 493 | 654 | 253 | 0,147 | 59 |
| low_snr | 1449 | 519 | 291 | 200 | 439 | 572 | 214 | 0,148 | 35 |
| reverb | 1888 | 743 | 317 | 250 | 578 | 747 | 268 | 0,142 | 49 |
| long_event | 559 | 220 | 122 | 103 | 114 | 431 | 142 | 0,254 | 18 |
| causal_chain | 845 | 389 | 155 | 82 | 219 | 266 | 95 | 0,112 | 11 |
| sach | 967 | 520 | 148 | 127 | 172 | 428 | 130 | 0,134 | 16 |

## 3. Ma trận nhầm lẫn

Hàng = lớp thật, cột = lớp đoán, `∅` = không có đối tác (Deletion ở cột, Insertion ở hàng). **Đường chéo gồm cả Đúng lẫn Boundary** — cả hai đều đúng lớp, chỉ khác thời điểm; đọc đường chéo như 'số đúng' là đọc sai.

```
         0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
  0    250    5    2    0    2    1    1    0    1    0    0    1    2    0    0   43   alarm_bell
  1      3  302    0    0    0    0    0    0    2    8    0    0    1    0    0   57   applause_cheering
  2      4    1  176    1    9    1   11    0    3   23    0    1    0    0    0   87   door_slam
  3      6    4    9  119    7    3    7    0    1   15    0    3    9    2    0   91   explosion
  4      2    4    3    3  239    1    7    0    1   32    0    2    2    0    1   75   fireworks
  5      4    0    2    0    3  216   10    0   12   14    0    1    1    0    0   65   glass_breaking
  6      2    3   10    2    5    0  149    0    0   21    0    0    1    0    0   75   gunshot
  7      5    6    1    0    4    1    0  172    2    8    0    0    5    1    0  108   laughter
  8     12    0   19    0    8    8    3    0  145   30    0    2    4    0    0   95   object_drop_dishes
  9      3    3    4    1    9    0   12    0    0  255    0    2    2    0    0  109   running_footsteps
 10      9    3    0    0    2    1    0    0    2    1  216    2    5    0    0   74   scream
 11      4    2    2    0    3    0    2    1    4    6    1  248    1    2    0   95   shout_yell
 12      5    1    0    0    0    0    0    0    0    1    0    1  247    0    0   19   siren
 13      4    5    1    0    5    1    0    1    3    7    0    9    2  161    0  114   speech_normal
 14     11    6   12    0    8    4    4    1    5   21    3    4    6    5  106  123   vehicle_crash
 15    311  185   98   15  246   83  117   26   78  483   57  112  129   64    3    0   ∅
```

Các cặp nhầm nhiều nhất, đối chiếu `confusable_with` khai trong `ontology_map.yaml`:

| thật | đoán | số | đã khai trong ontology |
|---|---|---|---|
| fireworks | running_footsteps | 32 | có |
| object_drop_dishes | running_footsteps | 30 | có |
| door_slam | running_footsteps | 23 | có |
| gunshot | running_footsteps | 21 | có |
| vehicle_crash | running_footsteps | 21 | **chưa** |
| object_drop_dishes | door_slam | 19 | có |
| explosion | running_footsteps | 15 | có |
| glass_breaking | running_footsteps | 14 | **chưa** |
| glass_breaking | object_drop_dishes | 12 | có |
| object_drop_dishes | alarm_bell | 12 | **chưa** |
| running_footsteps | gunshot | 12 | có |
| vehicle_crash | door_slam | 12 | **chưa** |
| door_slam | gunshot | 11 | có |
| vehicle_crash | alarm_bell | 11 | **chưa** |
| glass_breaking | gunshot | 10 | **chưa** |

311/642 (48%) lượt nhầm rơi vào cặp đã khai. Phần còn lại là nhầm lẫn mà ontology **chưa dự đoán** — nếu tỉ lệ này cao thì thứ cần sửa là `confusable_with`, không phải model.

## 4. Ngưỡng theo từng lớp

θ tốt nhất **toàn cục** là 0,25. Bảng dưới cho thấy θ đó không tối ưu cho phần lớn lớp:

| lớp | sự kiện thật | θ* riêng | F1 @θ* riêng | F1 @θ=0,25 | chênh |
|---|---|---|---|---|---|
| alarm_bell | 308 | 0,25 | 0,4496 | 0,4496 | +0,0000 |
| applause_cheering | 373 | 0,15 | 0,3992 | 0,3389 | +0,0603 |
| door_slam | 317 | 0,25 | 0,3628 | 0,3628 | +0,0000 |
| explosion | 276 | 0,15 | 0,5111 | 0,4700 | +0,0410 |
| fireworks | 372 | 0,15 | 0,2874 | 0,2842 | +0,0033 |
| glass_breaking | 328 | 0,15 | 0,5577 | 0,5185 | +0,0392 |
| gunshot | 268 | 0,35 | 0,4575 | 0,4501 | +0,0074 |
| laughter | 313 | 0,15 | 0,5061 | 0,4436 | +0,0625 |
| object_drop_dishes | 326 | 0,25 | 0,4513 | 0,4513 | +0,0000 |
| running_footsteps | 400 | 0,35 | 0,3790 | 0,3472 | +0,0318 |
| scream | 315 | 0,15 | 0,6256 | 0,5743 | +0,0513 |
| shout_yell | 371 | 0,15 | 0,4686 | 0,4295 | +0,0391 |
| siren | 274 | 0,15 | 0,5662 | 0,5297 | +0,0365 |
| speech_normal | 313 | 0,15 | 0,4281 | 0,3540 | +0,0741 |
| vehicle_crash | 319 | 0,15 | 0,3277 | 0,2844 | +0,0433 |

| cấu hình ngưỡng | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|
| θ = 0,50 (mặc định lúc train) | 0,3245 | 0,6229 |
| θ = 0,25 toàn cục | 0,4113 | 0,7064 |
| θ riêng từng lớp | 0,4408 | 0,7349 |

Hàng cuối chọn 15 ngưỡng trên chính tập đang chấm, nên nó là **chặn trên**, không phải hiệu năng mong đợi trên dữ liệu mới. Muốn dùng số này làm kết quả thì ngưỡng phải khoá trên dev rồi đo một lần trên tập test độc lập — mà `gold_test` hiện **rỗng**.
