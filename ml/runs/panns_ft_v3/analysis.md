# Phân tích lỗi — panns_ft_v3 trên dev/all

Sinh bởi `ml/evaluation/error_analysis.py` từ `ml/runs/panns_ft_v3/predictions/dev_all.npz` (1440 clip, 4873 sự kiện tham chiếu). Không train lại, không dùng GPU.

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với tập train của `v3`, nên dev **thiên vị v3 theo thiết kế**. Mọi so sánh v1/v2/v3 đọc trên trang này đều mang thiên vị đó.
> ⚠️ θ\* và θ theo từng lớp đều chọn **trên chính tập đang chấm** → rò rỉ. Chúng là chặn trên lạc quan, không phải kết quả báo cáo được.

## 1. Sáu loại lỗi

Tầng 1 gán cho mỗi sự kiện tham chiếu **đúng một** nhãn (Đúng/Boundary/Substitution/Deletion); dự báo không được tiêu thụ là Insertion. Fragmentation và Merge là **tầng 2**, đếm riêng và có thể chồng lên tầng 1.

Collar onset 200 ms, chỉ khớp onset (offset của tiếng vang quá mơ hồ để chấm).

**θ = 0,50** — 4873 sự kiện thật, 65884 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 2901 | 0,595 |
| Boundary | 1667 | 0,342 |
| Substitution | 296 | 0,061 |
| Deletion | 9 | 0,002 |
| Insertion | 61020 | 12,522 |
| Fragmentation (1 thật → n đoán) | 721 | — |
| Merge (n thật → 1 đoán) | 165 | — |

**θ = 0,90** — 4873 sự kiện thật, 6998 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 2362 | 0,485 |
| Boundary | 971 | 0,199 |
| Substitution | 725 | 0,149 |
| Deletion | 815 | 0,167 |
| Insertion | 2940 | 0,603 |
| Fragmentation (1 thật → n đoán) | 927 | — |
| Merge (n thật → 1 đoán) | 146 | — |

**Boundary không phải một rổ độc lập với sed_eval.** Một dự báo đúng lớp nhưng lệch onset quá collar bị sed_eval tính **một Deletion + một Insertion**. Bảng trên tách nó ra vì hai lỗi cần hai cách sửa khác nhau: lệch biên sửa ở hậu xử lý, còn không-nghe-ra-gì phải sửa ở dữ liệu. Số S/D/I gốc của sed_eval:

| θ | Nref | Nsys | Substitution | Deletion | Insertion | error rate |
|---|---|---|---|---|---|---|
| θ = 0,50 | 4873 | 65884 | 1607 | 365 | 61376 | 13,000 |
| θ = 0,90 | 4873 | 6998 | 638 | 1873 | 3998 | 1,336 |

Đối chiếu số khớp đúng giữa hai phép ghép (greedy ở đây vs khớp tối ưu trong sed_eval) — lệch lớn nghĩa là phép ghép này không đáng tin và bảng trên phải bỏ:

| θ | ghép greedy | sed_eval Ntp | lệch |
|---|---|---|---|
| θ = 0,50 | 2901 | 2901 | +0 |
| θ = 0,90 | 2362 | 2362 | +0 |

## 2. Theo lát cắt

Chấm ở θ = 0,90. **Lát cắt chồng nhau** — một clip vừa `overlap` vừa `reverb` nằm ở cả hai hàng, nên tổng cột *clip* lớn hơn 1440 clip của tập. `sach` là clip không thuộc lát cắt khó nào; nó là mốc đối chiếu, không phải một lát cắt.

| lát cắt | clip | sự kiện thật | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|---|---|
| overlap | 437 | 1723 | 0.3799 | 0.7086 |
| low_snr | 390 | 1449 | 0.3369 | 0.6817 |
| reverb | 557 | 1888 | 0.3738 | 0.6811 |
| long_event | 144 | 559 | 0.2685 | 0.7133 |
| causal_chain | 203 | 845 | 0.4417 | 0.6968 |
| sach | 357 | 967 | 0.4487 | 0.7460 |

Phân loại lỗi **đếm lại trên từng tập con clip**, không chia tỉ lệ từ tổng — lát cắt chồng nhau nên cộng cột sẽ lớn hơn tổng toàn tập. Cột *phân mảnh/thật* là số sự kiện thật bị cắt thành nhiều mảnh, chia cho số sự kiện thật của chính lát cắt đó, nên so ngang giữa các hàng được.

| lát cắt | thật | Đúng | Boundary | Substitution | Deletion | Insertion | phân mảnh | phân mảnh/thật | gộp |
|---|---|---|---|---|---|---|---|---|---|
| overlap | 1723 | 779 | 352 | 260 | 332 | 987 | 330 | 0,192 | 77 |
| low_snr | 1449 | 592 | 331 | 234 | 292 | 908 | 294 | 0,203 | 46 |
| reverb | 1888 | 842 | 373 | 288 | 385 | 1114 | 359 | 0,190 | 68 |
| long_event | 559 | 218 | 156 | 91 | 94 | 600 | 178 | 0,318 | 20 |
| causal_chain | 845 | 422 | 149 | 109 | 165 | 386 | 127 | 0,150 | 17 |
| sach | 967 | 538 | 175 | 144 | 110 | 574 | 162 | 0,168 | 19 |

## 3. Ma trận nhầm lẫn

Hàng = lớp thật, cột = lớp đoán, `∅` = không có đối tác (Deletion ở cột, Insertion ở hàng). **Đường chéo gồm cả Đúng lẫn Boundary** — cả hai đều đúng lớp, chỉ khác thời điểm; đọc đường chéo như 'số đúng' là đọc sai.

```
         0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
  0    238    6    1    0    2    5    1    1    2    1    0    1    2    2    0   46   alarm_bell
  1      4  294    1    2    1    0    2    0    1    5    2    0    5    0    0   56   applause_cheering
  2      2    4  186    6    7    7   11    0    5   22    0    0    1    0    1   65   door_slam
  3      1    6    8  193    4    2    6    1    0    5    0    1    3    1    1   44   explosion
  4      4    6    6    7  237    4   11    0    1   26    0    1    3    2    1   63   fireworks
  5      1    0    5    2    3  259    7    1    3    5    0    1    1    1    3   36   glass_breaking
  6      2    3   12    9    4    0  153    0    1   19    0    1    1    1    0   62   gunshot
  7      1   10    2    0    3    1    0  224    1    2    2    8    2    2    1   54   laughter
  8      8    6   29    0    6   27   10    2  125   19    0    3    4    0    0   87   object_drop_dishes
  9      2    9   10    8   13    5   15    0    1  231    0    1    4    3    0   98   running_footsteps
 10      2    5    0    0    2    1    0    0    0    1  262    6    4    0    1   31   scream
 11      1    2    1    1    3    1    0    5    1    3    7  296    1    5    1   43   shout_yell
 12      5    2    0    0    1    0    0    1    0    1    2    0  249    1    2   10   siren
 13      0    4    0    1    2    3    1    4    0    5    2   17    2  231    0   41   speech_normal
 14      5    5   12    6    6    5    5    5    2    9    4    6    9    6  155   79   vehicle_crash
 15    259  375  152   90  354  173  156  114   94  328  128  225  209  228   55    0   ∅
```

Các cặp nhầm nhiều nhất, đối chiếu `confusable_with` khai trong `ontology_map.yaml`:

| thật | đoán | số | đã khai trong ontology |
|---|---|---|---|
| object_drop_dishes | door_slam | 29 | có |
| object_drop_dishes | glass_breaking | 27 | có |
| fireworks | running_footsteps | 26 | có |
| door_slam | running_footsteps | 22 | có |
| gunshot | running_footsteps | 19 | có |
| object_drop_dishes | running_footsteps | 19 | có |
| speech_normal | shout_yell | 17 | có |
| running_footsteps | gunshot | 15 | có |
| running_footsteps | fireworks | 13 | có |
| gunshot | door_slam | 12 | có |
| vehicle_crash | door_slam | 12 | **chưa** |
| door_slam | gunshot | 11 | có |
| fireworks | gunshot | 11 | có |
| laughter | applause_cheering | 10 | **chưa** |
| object_drop_dishes | gunshot | 10 | **chưa** |

392/725 (54%) lượt nhầm rơi vào cặp đã khai. Phần còn lại là nhầm lẫn mà ontology **chưa dự đoán** — nếu tỉ lệ này cao thì thứ cần sửa là `confusable_with`, không phải model.

## 4. Ngưỡng theo từng lớp

θ tốt nhất **toàn cục** là 0,90. Bảng dưới cho thấy θ đó không tối ưu cho phần lớn lớp:

| lớp | sự kiện thật | θ* riêng | F1 @θ* riêng | F1 @θ=0,90 | chênh |
|---|---|---|---|---|---|
| alarm_bell | 308 | 0,85 | 0,4356 | 0,4223 | +0,0133 |
| applause_cheering | 373 | 0,65 | 0,3588 | 0,2396 | +0,1191 |
| door_slam | 317 | 0,90 | 0,3396 | 0,3396 | +0,0000 |
| explosion | 276 | 0,95 | 0,5336 | 0,5225 | +0,0111 |
| fireworks | 372 | 0,80 | 0,2776 | 0,2373 | +0,0403 |
| glass_breaking | 328 | 0,90 | 0,5116 | 0,5116 | +0,0000 |
| gunshot | 268 | 0,90 | 0,4241 | 0,4241 | +0,0000 |
| laughter | 313 | 0,90 | 0,5127 | 0,5127 | +0,0000 |
| object_drop_dishes | 326 | 0,90 | 0,3943 | 0,3943 | +0,0000 |
| running_footsteps | 400 | 0,90 | 0,3697 | 0,3697 | +0,0000 |
| scream | 315 | 0,90 | 0,5939 | 0,5939 | +0,0000 |
| shout_yell | 371 | 0,85 | 0,4432 | 0,4286 | +0,0146 |
| siren | 274 | 0,60 | 0,4880 | 0,4289 | +0,0590 |
| speech_normal | 313 | 0,90 | 0,3744 | 0,3744 | +0,0000 |
| vehicle_crash | 319 | 0,90 | 0,3185 | 0,3185 | +0,0000 |

| cấu hình ngưỡng | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|
| θ = 0,50 (mặc định lúc train) | 0,0820 | 0,2531 |
| θ = 0,90 toàn cục | 0,3979 | 0,7155 |
| θ riêng từng lớp | 0,4157 | 0,7298 |

Hàng cuối chọn 15 ngưỡng trên chính tập đang chấm, nên nó là **chặn trên**, không phải hiệu năng mong đợi trên dữ liệu mới. Muốn dùng số này làm kết quả thì ngưỡng phải khoá trên dev rồi đo một lần trên tập test độc lập — mà `gold_test` hiện **rỗng**.
