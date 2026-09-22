# Phân tích lỗi — panns_ft_pw30 trên dev/all

Sinh bởi `ml/evaluation/error_analysis.py` từ `ml/runs/panns_ft_pw30/predictions/dev_all.npz` (1440 clip, 4873 sự kiện tham chiếu). Không train lại, không dùng GPU.

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với tập train của `v3`, nên dev **thiên vị v3 theo thiết kế**. Mọi so sánh v1/v2/v3 đọc trên trang này đều mang thiên vị đó.
> ⚠️ θ\* và θ theo từng lớp đều chọn **trên chính tập đang chấm** → rò rỉ. Chúng là chặn trên lạc quan, không phải kết quả báo cáo được.

## 1. Sáu loại lỗi

Tầng 1 gán cho mỗi sự kiện tham chiếu **đúng một** nhãn (Đúng/Boundary/Substitution/Deletion); dự báo không được tiêu thụ là Insertion. Fragmentation và Merge là **tầng 2**, đếm riêng và có thể chồng lên tầng 1.

Collar onset 200 ms, chỉ khớp onset (offset của tiếng vang quá mơ hồ để chấm).

**θ = 0,50** — 4873 sự kiện thật, 63064 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 2905 | 0,596 |
| Boundary | 1657 | 0,340 |
| Substitution | 299 | 0,061 |
| Deletion | 12 | 0,003 |
| Insertion | 58203 | 11,944 |
| Fragmentation (1 thật → n đoán) | 737 | — |
| Merge (n thật → 1 đoán) | 169 | — |

**θ = 0,90** — 4873 sự kiện thật, 6759 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 2337 | 0,480 |
| Boundary | 973 | 0,200 |
| Substitution | 691 | 0,142 |
| Deletion | 872 | 0,179 |
| Insertion | 2758 | 0,566 |
| Fragmentation (1 thật → n đoán) | 912 | — |
| Merge (n thật → 1 đoán) | 143 | — |

**Boundary không phải một rổ độc lập với sed_eval.** Một dự báo đúng lớp nhưng lệch onset quá collar bị sed_eval tính **một Deletion + một Insertion**. Bảng trên tách nó ra vì hai lỗi cần hai cách sửa khác nhau: lệch biên sửa ở hậu xử lý, còn không-nghe-ra-gì phải sửa ở dữ liệu. Số S/D/I gốc của sed_eval:

| θ | Nref | Nsys | Substitution | Deletion | Insertion | error rate |
|---|---|---|---|---|---|---|
| θ = 0,50 | 4873 | 63064 | 1598 | 370 | 58561 | 12,421 |
| θ = 0,90 | 4873 | 6759 | 600 | 1936 | 3822 | 1,305 |

Đối chiếu số khớp đúng giữa hai phép ghép (greedy ở đây vs khớp tối ưu trong sed_eval) — lệch lớn nghĩa là phép ghép này không đáng tin và bảng trên phải bỏ:

| θ | ghép greedy | sed_eval Ntp | lệch |
|---|---|---|---|
| θ = 0,50 | 2905 | 2905 | +0 |
| θ = 0,90 | 2337 | 2337 | +0 |

## 2. Theo lát cắt

Chấm ở θ = 0,90. **Lát cắt chồng nhau** — một clip vừa `overlap` vừa `reverb` nằm ở cả hai hàng, nên tổng cột *clip* lớn hơn 1440 clip của tập. `sach` là clip không thuộc lát cắt khó nào; nó là mốc đối chiếu, không phải một lát cắt.

| lát cắt | clip | sự kiện thật | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|---|---|
| overlap | 437 | 1723 | 0.3818 | 0.7083 |
| low_snr | 390 | 1449 | 0.3384 | 0.6832 |
| reverb | 557 | 1888 | 0.3752 | 0.6805 |
| long_event | 144 | 559 | 0.2715 | 0.7157 |
| causal_chain | 203 | 845 | 0.4486 | 0.7033 |
| sach | 357 | 967 | 0.4568 | 0.7529 |

Phân loại lỗi **đếm lại trên từng tập con clip**, không chia tỉ lệ từ tổng — lát cắt chồng nhau nên cộng cột sẽ lớn hơn tổng toàn tập. Cột *phân mảnh/thật* là số sự kiện thật bị cắt thành nhiều mảnh, chia cho số sự kiện thật của chính lát cắt đó, nên so ngang giữa các hàng được.

| lát cắt | thật | Đúng | Boundary | Substitution | Deletion | Insertion | phân mảnh | phân mảnh/thật | gộp |
|---|---|---|---|---|---|---|---|---|---|
| overlap | 1723 | 768 | 356 | 247 | 352 | 929 | 324 | 0,188 | 70 |
| low_snr | 1449 | 579 | 333 | 223 | 314 | 838 | 289 | 0,199 | 48 |
| reverb | 1888 | 825 | 370 | 279 | 414 | 1036 | 340 | 0,180 | 62 |
| long_event | 559 | 216 | 159 | 94 | 90 | 563 | 175 | 0,313 | 22 |
| causal_chain | 845 | 417 | 151 | 104 | 173 | 342 | 118 | 0,140 | 14 |
| sach | 967 | 542 | 176 | 129 | 120 | 559 | 171 | 0,177 | 19 |

## 3. Ma trận nhầm lẫn

Hàng = lớp thật, cột = lớp đoán, `∅` = không có đối tác (Deletion ở cột, Insertion ở hàng). **Đường chéo gồm cả Đúng lẫn Boundary** — cả hai đều đúng lớp, chỉ khác thời điểm; đọc đường chéo như 'số đúng' là đọc sai.

```
         0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
  0    237    3    1    0    1    4    1    1    1    0    0    1    2    2    0   54   alarm_bell
  1      3  293    1    2    1    0    3    0    1    6    1    0    3    0    0   59   applause_cheering
  2      3    3  178    6    7    5    8    0    4   25    0    0    1    1    2   74   door_slam
  3      1    7    7  195    4    0    5    1    0    3    0    1    4    0    2   46   explosion
  4      5    3    7    8  248    1    9    0    1   27    0    2    3    0    0   58   fireworks
  5      3    1    3    3    3  253    9    1    2    6    1    1    0    2    1   39   glass_breaking
  6      2    3   12    8    7    0  144    0    1   13    0    2    2    1    2   71   gunshot
  7      2    9    1    2    3    1    0  226    1    3    1    5    3    2    1   53   laughter
  8      9    6   29    3    7   26    5    1  120   17    0    2    2    0    0   99   object_drop_dishes
  9      3    7   13    5   17    3   13    0    2  228    0    0    3    3    1  102   running_footsteps
 10      2    4    0    0    1    1    0    0    0    1  264    6    1    0    1   34   scream
 11      1    2    1    0    2    1    1    4    2    4    7  294    2    3    1   46   shout_yell
 12      4    1    0    0    1    0    0    2    0    1    1    0  249    0    1   14   siren
 13      1    4    0    1    3    2    0    3    0    3    1   21    2  225    0   47   speech_normal
 14      6    4   11   10    9    2    6    4    2    8    4    6   10    5  156   76   vehicle_crash
 15    255  349  128  103  349  160  122  104   71  288  132  214  213  201   69    0   ∅
```

Các cặp nhầm nhiều nhất, đối chiếu `confusable_with` khai trong `ontology_map.yaml`:

| thật | đoán | số | đã khai trong ontology |
|---|---|---|---|
| object_drop_dishes | door_slam | 29 | có |
| fireworks | running_footsteps | 27 | có |
| object_drop_dishes | glass_breaking | 26 | có |
| door_slam | running_footsteps | 25 | có |
| speech_normal | shout_yell | 21 | có |
| object_drop_dishes | running_footsteps | 17 | có |
| running_footsteps | fireworks | 17 | có |
| gunshot | running_footsteps | 13 | có |
| running_footsteps | door_slam | 13 | có |
| running_footsteps | gunshot | 13 | có |
| gunshot | door_slam | 12 | có |
| vehicle_crash | door_slam | 11 | **chưa** |
| vehicle_crash | explosion | 10 | **chưa** |
| vehicle_crash | siren | 10 | có |
| fireworks | gunshot | 9 | có |

371/691 (54%) lượt nhầm rơi vào cặp đã khai. Phần còn lại là nhầm lẫn mà ontology **chưa dự đoán** — nếu tỉ lệ này cao thì thứ cần sửa là `confusable_with`, không phải model.

## 4. Ngưỡng theo từng lớp

θ tốt nhất **toàn cục** là 0,90. Bảng dưới cho thấy θ đó không tối ưu cho phần lớn lớp:

| lớp | sự kiện thật | θ* riêng | F1 @θ* riêng | F1 @θ=0,90 | chênh |
|---|---|---|---|---|---|
| alarm_bell | 308 | 0,85 | 0,4403 | 0,4308 | +0,0095 |
| applause_cheering | 373 | 0,60 | 0,3558 | 0,2276 | +0,1282 |
| door_slam | 317 | 0,90 | 0,3470 | 0,3470 | +0,0000 |
| explosion | 276 | 0,95 | 0,5320 | 0,5145 | +0,0176 |
| fireworks | 372 | 0,85 | 0,2818 | 0,2589 | +0,0228 |
| glass_breaking | 328 | 0,90 | 0,5184 | 0,5184 | +0,0000 |
| gunshot | 268 | 0,90 | 0,4343 | 0,4343 | +0,0000 |
| laughter | 313 | 0,90 | 0,5091 | 0,5091 | +0,0000 |
| object_drop_dishes | 326 | 0,85 | 0,4077 | 0,4045 | +0,0032 |
| running_footsteps | 400 | 0,90 | 0,3833 | 0,3833 | +0,0000 |
| scream | 315 | 0,90 | 0,6052 | 0,6052 | +0,0000 |
| shout_yell | 371 | 0,85 | 0,4392 | 0,4168 | +0,0224 |
| siren | 274 | 0,60 | 0,4934 | 0,4367 | +0,0568 |
| speech_normal | 313 | 0,90 | 0,3720 | 0,3720 | +0,0000 |
| vehicle_crash | 319 | 0,90 | 0,3094 | 0,3094 | +0,0000 |

| cấu hình ngưỡng | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|
| θ = 0,50 (mặc định lúc train) | 0,0855 | 0,2600 |
| θ = 0,90 toàn cục | 0,4018 | 0,7182 |
| θ riêng từng lớp | 0,4202 | 0,7324 |

Hàng cuối chọn 15 ngưỡng trên chính tập đang chấm, nên nó là **chặn trên**, không phải hiệu năng mong đợi trên dữ liệu mới. Muốn dùng số này làm kết quả thì ngưỡng phải khoá trên dev rồi đo một lần trên tập test độc lập — mà `gold_test` hiện **rỗng**.
