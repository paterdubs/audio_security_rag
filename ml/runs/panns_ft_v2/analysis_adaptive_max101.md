# Phân tích lỗi — panns_ft_v2 trên dev/all

Sinh bởi `ml/evaluation/error_analysis.py` từ `ml/runs/panns_ft_v2/predictions/dev_all.npz` (1440 clip, 4873 sự kiện tham chiếu). Không train lại, không dùng GPU.

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với tập train của `v3`, nên dev **thiên vị v3 theo thiết kế**. Mọi so sánh v1/v2/v3 đọc trên trang này đều mang thiên vị đó.
> ⚠️ θ\* và θ theo từng lớp đều chọn **trên chính tập đang chấm** → rò rỉ. Chúng là chặn trên lạc quan, không phải kết quả báo cáo được.

## 1. Sáu loại lỗi

Tầng 1 gán cho mỗi sự kiện tham chiếu **đúng một** nhãn (Đúng/Boundary/Substitution/Deletion); dự báo không được tiêu thụ là Insertion. Fragmentation và Merge là **tầng 2**, đếm riêng và có thể chồng lên tầng 1.

Collar onset 200 ms, chỉ khớp onset (offset của tiếng vang quá mơ hồ để chấm).

**θ = 0,50** — 4873 sự kiện thật, 32933 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 2901 | 0,595 |
| Boundary | 1361 | 0,279 |
| Substitution | 496 | 0,102 |
| Deletion | 115 | 0,024 |
| Insertion | 28175 | 5,782 |
| Fragmentation (1 thật → n đoán) | 507 | — |
| Merge (n thật → 1 đoán) | 154 | — |

**θ = 0,85** — 4873 sự kiện thật, 6221 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 2177 | 0,447 |
| Boundary | 777 | 0,160 |
| Substitution | 782 | 0,161 |
| Deletion | 1137 | 0,233 |
| Insertion | 2485 | 0,510 |
| Fragmentation (1 thật → n đoán) | 534 | — |
| Merge (n thật → 1 đoán) | 111 | — |

**Boundary không phải một rổ độc lập với sed_eval.** Một dự báo đúng lớp nhưng lệch onset quá collar bị sed_eval tính **một Deletion + một Insertion**. Bảng trên tách nó ra vì hai lỗi cần hai cách sửa khác nhau: lệch biên sửa ở hậu xử lý, còn không-nghe-ra-gì phải sửa ở dữ liệu. Số S/D/I gốc của sed_eval:

| θ | Nref | Nsys | Substitution | Deletion | Insertion | error rate |
|---|---|---|---|---|---|---|
| θ = 0,50 | 4873 | 32933 | 1206 | 766 | 28826 | 6,320 |
| θ = 0,85 | 4873 | 6221 | 661 | 2035 | 3383 | 1,247 |

Đối chiếu số khớp đúng giữa hai phép ghép (greedy ở đây vs khớp tối ưu trong sed_eval) — lệch lớn nghĩa là phép ghép này không đáng tin và bảng trên phải bỏ:

| θ | ghép greedy | sed_eval Ntp | lệch |
|---|---|---|---|
| θ = 0,50 | 2901 | 2901 | +0 |
| θ = 0,85 | 2177 | 2177 | +0 |

## 2. Theo lát cắt

Chấm ở θ = 0,85. **Lát cắt chồng nhau** — một clip vừa `overlap` vừa `reverb` nằm ở cả hai hàng, nên tổng cột *clip* lớn hơn 1440 clip của tập. `sach` là clip không thuộc lát cắt khó nào; nó là mốc đối chiếu, không phải một lát cắt.

| lát cắt | clip | sự kiện thật | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|---|---|
| overlap | 437 | 1723 | 0.3788 | 0.6432 |
| low_snr | 390 | 1449 | 0.3448 | 0.6047 |
| reverb | 557 | 1888 | 0.3690 | 0.6072 |
| long_event | 144 | 559 | 0.2823 | 0.6305 |
| causal_chain | 203 | 845 | 0.4582 | 0.6568 |
| sach | 357 | 967 | 0.4286 | 0.6798 |

Phân loại lỗi **đếm lại trên từng tập con clip**, không chia tỉ lệ từ tổng — lát cắt chồng nhau nên cộng cột sẽ lớn hơn tổng toàn tập. Cột *phân mảnh/thật* là số sự kiện thật bị cắt thành nhiều mảnh, chia cho số sự kiện thật của chính lát cắt đó, nên so ngang giữa các hàng được.

| lát cắt | thật | Đúng | Boundary | Substitution | Deletion | Insertion | phân mảnh | phân mảnh/thật | gộp |
|---|---|---|---|---|---|---|---|---|---|
| overlap | 1723 | 722 | 285 | 268 | 448 | 814 | 191 | 0,111 | 54 |
| low_snr | 1449 | 548 | 258 | 242 | 401 | 682 | 161 | 0,111 | 29 |
| reverb | 1888 | 769 | 300 | 326 | 493 | 885 | 200 | 0,106 | 43 |
| long_event | 559 | 212 | 116 | 114 | 117 | 501 | 123 | 0,220 | 20 |
| causal_chain | 845 | 406 | 128 | 98 | 213 | 295 | 70 | 0,083 | 14 |
| sach | 967 | 495 | 142 | 143 | 187 | 563 | 95 | 0,098 | 18 |

## 3. Ma trận nhầm lẫn

Hàng = lớp thật, cột = lớp đoán, `∅` = không có đối tác (Deletion ở cột, Insertion ở hàng). **Đường chéo gồm cả Đúng lẫn Boundary** — cả hai đều đúng lớp, chỉ khác thời điểm; đọc đường chéo như 'số đúng' là đọc sai.

```
         0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
  0    230    1    1    0    2    4    1    2    1    2    1    1    9    0    0   53   alarm_bell
  1     10  264    2    0    2    1    1    0    0   10    3    1    5    1    1   72   applause_cheering
  2      3    1  163    1    7    4    7    1    2   30    1    1    1    1    0   94   door_slam
  3      6    3   13  132    9    2   11    0    1   13    0    2    5    1    0   78   explosion
  4      6    2    3    5  251    1    6    0    1   29    0    2    3    1    0   62   fireworks
  5      9    1    3    1    3  228    9    0    3   10    1    2    2    0    1   55   glass_breaking
  6      4    3   15    3   10    0  133    0    0   19    0    2    1    0    0   78   gunshot
  7      2    4    0    0    3    1    3  198    2    6    3   10    4    1    0   76   laughter
  8      7    1   24    0   12   21    6    1  121   19    0    2    3    0    0  109   object_drop_dishes
  9      3    5   21    4   19    1   11    0    3  198    1    3    1    1    0  129   running_footsteps
 10      3    3    0    0    1    2    0    0    0    3  228   17    3    1    0   54   scream
 11      3    1    3    0    4    1    0    3    1    4    0  276    0    9    0   66   shout_yell
 12      6    1    0    0    1    1    1    0    0    1    0    1  237    0    0   25   siren
 13      5    3    1    0    3    4    0    2    1   10    0   27    5  184    0   68   speech_normal
 14      7    2   10    3    8    7    5    8    5   15    3    7    6    4  111  118   vehicle_crash
 15    361   79  158   15  307  143  139   95  110  610   58  218   57  111   24    0   ∅
```

Các cặp nhầm nhiều nhất, đối chiếu `confusable_with` khai trong `ontology_map.yaml`:

| thật | đoán | số | đã khai trong ontology |
|---|---|---|---|
| door_slam | running_footsteps | 30 | có |
| fireworks | running_footsteps | 29 | có |
| speech_normal | shout_yell | 27 | có |
| object_drop_dishes | door_slam | 24 | có |
| object_drop_dishes | glass_breaking | 21 | có |
| running_footsteps | door_slam | 21 | có |
| gunshot | running_footsteps | 19 | có |
| object_drop_dishes | running_footsteps | 19 | có |
| running_footsteps | fireworks | 19 | có |
| scream | shout_yell | 17 | có |
| gunshot | door_slam | 15 | có |
| vehicle_crash | running_footsteps | 15 | **chưa** |
| explosion | door_slam | 13 | có |
| explosion | running_footsteps | 13 | có |
| object_drop_dishes | fireworks | 12 | **chưa** |

429/782 (55%) lượt nhầm rơi vào cặp đã khai. Phần còn lại là nhầm lẫn mà ontology **chưa dự đoán** — nếu tỉ lệ này cao thì thứ cần sửa là `confusable_with`, không phải model.

## 4. Ngưỡng theo từng lớp

θ tốt nhất **toàn cục** là 0,85. Bảng dưới cho thấy θ đó không tối ưu cho phần lớn lớp:

| lớp | sự kiện thật | θ* riêng | F1 @θ* riêng | F1 @θ=0,85 | chênh |
|---|---|---|---|---|---|
| alarm_bell | 308 | 0,90 | 0,4027 | 0,3905 | +0,0121 |
| applause_cheering | 373 | 0,65 | 0,4830 | 0,3989 | +0,0841 |
| door_slam | 317 | 0,85 | 0,3215 | 0,3215 | +0,0000 |
| explosion | 276 | 0,80 | 0,5349 | 0,5273 | +0,0076 |
| fireworks | 372 | 0,85 | 0,2781 | 0,2781 | +0,0000 |
| glass_breaking | 328 | 0,85 | 0,4887 | 0,4887 | +0,0000 |
| gunshot | 268 | 0,90 | 0,3879 | 0,3794 | +0,0086 |
| laughter | 313 | 0,85 | 0,5008 | 0,5008 | +0,0000 |
| object_drop_dishes | 326 | 0,85 | 0,3605 | 0,3605 | +0,0000 |
| running_footsteps | 400 | 0,90 | 0,2413 | 0,2350 | +0,0064 |
| scream | 315 | 0,80 | 0,5942 | 0,5831 | +0,0111 |
| shout_yell | 371 | 0,80 | 0,4111 | 0,4051 | +0,0060 |
| siren | 274 | 0,65 | 0,6198 | 0,5552 | +0,0646 |
| speech_normal | 313 | 0,85 | 0,4108 | 0,4108 | +0,0000 |
| vehicle_crash | 319 | 0,85 | 0,3246 | 0,3246 | +0,0000 |

| cấu hình ngưỡng | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|
| θ = 0,50 (mặc định lúc train) | 0,1535 | 0,3563 |
| θ = 0,85 toàn cục | 0,3925 | 0,6442 |
| θ riêng từng lớp | 0,4151 | 0,6657 |

Hàng cuối chọn 15 ngưỡng trên chính tập đang chấm, nên nó là **chặn trên**, không phải hiệu năng mong đợi trên dữ liệu mới. Muốn dùng số này làm kết quả thì ngưỡng phải khoá trên dev rồi đo một lần trên tập test độc lập — mà `gold_test` hiện **rỗng**.
