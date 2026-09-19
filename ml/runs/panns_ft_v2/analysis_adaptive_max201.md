# Phân tích lỗi — panns_ft_v2 trên dev/all

Sinh bởi `ml/evaluation/error_analysis.py` từ `ml/runs/panns_ft_v2/predictions/dev_all.npz` (1440 clip, 4873 sự kiện tham chiếu). Không train lại, không dùng GPU.

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với tập train của `v3`, nên dev **thiên vị v3 theo thiết kế**. Mọi so sánh v1/v2/v3 đọc trên trang này đều mang thiên vị đó.
> ⚠️ θ\* và θ theo từng lớp đều chọn **trên chính tập đang chấm** → rò rỉ. Chúng là chặn trên lạc quan, không phải kết quả báo cáo được.

## 1. Sáu loại lỗi

Tầng 1 gán cho mỗi sự kiện tham chiếu **đúng một** nhãn (Đúng/Boundary/Substitution/Deletion); dự báo không được tiêu thụ là Insertion. Fragmentation và Merge là **tầng 2**, đếm riêng và có thể chồng lên tầng 1.

Collar onset 200 ms, chỉ khớp onset (offset của tiếng vang quá mơ hồ để chấm).

**θ = 0,50** — 4873 sự kiện thật, 32636 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 2868 | 0,589 |
| Boundary | 1368 | 0,281 |
| Substitution | 518 | 0,106 |
| Deletion | 119 | 0,024 |
| Insertion | 27882 | 5,722 |
| Fragmentation (1 thật → n đoán) | 479 | — |
| Merge (n thật → 1 đoán) | 155 | — |

**θ = 0,85** — 4873 sự kiện thật, 6070 sự kiện dự báo

| loại | số | tỉ lệ / sự kiện thật |
|---|---|---|
| Đúng | 2144 | 0,440 |
| Boundary | 767 | 0,157 |
| Substitution | 778 | 0,160 |
| Deletion | 1184 | 0,243 |
| Insertion | 2381 | 0,489 |
| Fragmentation (1 thật → n đoán) | 480 | — |
| Merge (n thật → 1 đoán) | 107 | — |

**Boundary không phải một rổ độc lập với sed_eval.** Một dự báo đúng lớp nhưng lệch onset quá collar bị sed_eval tính **một Deletion + một Insertion**. Bảng trên tách nó ra vì hai lỗi cần hai cách sửa khác nhau: lệch biên sửa ở hậu xử lý, còn không-nghe-ra-gì phải sửa ở dữ liệu. Số S/D/I gốc của sed_eval:

| θ | Nref | Nsys | Substitution | Deletion | Insertion | error rate |
|---|---|---|---|---|---|---|
| θ = 0,50 | 4873 | 32636 | 1222 | 783 | 28546 | 6,269 |
| θ = 0,85 | 4873 | 6070 | 655 | 2074 | 3271 | 1,231 |

Đối chiếu số khớp đúng giữa hai phép ghép (greedy ở đây vs khớp tối ưu trong sed_eval) — lệch lớn nghĩa là phép ghép này không đáng tin và bảng trên phải bỏ:

| θ | ghép greedy | sed_eval Ntp | lệch |
|---|---|---|---|
| θ = 0,50 | 2868 | 2868 | +0 |
| θ = 0,85 | 2144 | 2144 | +0 |

## 2. Theo lát cắt

Chấm ở θ = 0,85. **Lát cắt chồng nhau** — một clip vừa `overlap` vừa `reverb` nằm ở cả hai hàng, nên tổng cột *clip* lớn hơn 1440 clip của tập. `sach` là clip không thuộc lát cắt khó nào; nó là mốc đối chiếu, không phải một lát cắt.

| lát cắt | clip | sự kiện thật | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|---|---|
| overlap | 437 | 1723 | 0.3760 | 0.6366 |
| low_snr | 390 | 1449 | 0.3459 | 0.5955 |
| reverb | 557 | 1888 | 0.3672 | 0.5999 |
| long_event | 144 | 559 | 0.2851 | 0.6228 |
| causal_chain | 203 | 845 | 0.4589 | 0.6540 |
| sach | 357 | 967 | 0.4247 | 0.6729 |

Phân loại lỗi **đếm lại trên từng tập con clip**, không chia tỉ lệ từ tổng — lát cắt chồng nhau nên cộng cột sẽ lớn hơn tổng toàn tập. Cột *phân mảnh/thật* là số sự kiện thật bị cắt thành nhiều mảnh, chia cho số sự kiện thật của chính lát cắt đó, nên so ngang giữa các hàng được.

| lát cắt | thật | Đúng | Boundary | Substitution | Deletion | Insertion | phân mảnh | phân mảnh/thật | gộp |
|---|---|---|---|---|---|---|---|---|---|
| overlap | 1723 | 708 | 284 | 269 | 462 | 782 | 170 | 0,099 | 54 |
| low_snr | 1449 | 540 | 250 | 243 | 416 | 640 | 146 | 0,101 | 27 |
| reverb | 1888 | 755 | 299 | 318 | 516 | 852 | 178 | 0,094 | 42 |
| long_event | 559 | 211 | 112 | 115 | 121 | 483 | 116 | 0,208 | 20 |
| causal_chain | 845 | 405 | 128 | 97 | 215 | 290 | 66 | 0,078 | 14 |
| sach | 967 | 484 | 140 | 146 | 197 | 542 | 85 | 0,088 | 16 |

## 3. Ma trận nhầm lẫn

Hàng = lớp thật, cột = lớp đoán, `∅` = không có đối tác (Deletion ở cột, Insertion ở hàng). **Đường chéo gồm cả Đúng lẫn Boundary** — cả hai đều đúng lớp, chỉ khác thời điểm; đọc đường chéo như 'số đúng' là đọc sai.

```
         0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
  0    230    1    2    0    2    4    1    2    1    2    1    1    8    0    0   53   alarm_bell
  1     10  246    2    0    3    1    1    1    1   12    4    1    2    1    1   87   applause_cheering
  2      3    0  163    1    7    4    7    1    2   30    1    1    0    1    0   96   door_slam
  3      6    2   13  132    9    2   10    0    1   13    0    3    1    1    0   83   explosion
  4      6    3    3    5  251    1    6    0    1   29    0    2    0    1    0   64   fireworks
  5      9    1    3    1    3  228    9    0    3   10    1    2    2    0    1   55   glass_breaking
  6      4    2   15    3   10    0  133    0    0   19    0    2    0    0    0   80   gunshot
  7      2    4    0    0    3    1    3  198    2    6    2   10    2    1    0   79   laughter
  8      6    2   22    0   12   21    6    1  121   20    0    2    3    0    0  110   object_drop_dishes
  9      3    3   21    4   19    1   11    0    3  198    1    3    0    1    0  132   running_footsteps
 10      4    2    0    0    1    2    0    0    0    3  228   18    0    1    0   56   scream
 11      3    1    3    0    4    1    0    3    1    4    0  276    0    9    0   66   shout_yell
 12     11    0    1    0    2    2    2    0    0    4    2    3  212    0    0   35   siren
 13      7    2    1    0    3    4    0    2    1   12    0   27    2  184    0   68   speech_normal
 14      7    1   11    3    8    7    6    8    5   15    3    7    3    4  111  120   vehicle_crash
 15    354   47  157   15  305  142  138   94  109  602   56  214   13  111   24    0   ∅
```

Các cặp nhầm nhiều nhất, đối chiếu `confusable_with` khai trong `ontology_map.yaml`:

| thật | đoán | số | đã khai trong ontology |
|---|---|---|---|
| door_slam | running_footsteps | 30 | có |
| fireworks | running_footsteps | 29 | có |
| speech_normal | shout_yell | 27 | có |
| object_drop_dishes | door_slam | 22 | có |
| object_drop_dishes | glass_breaking | 21 | có |
| running_footsteps | door_slam | 21 | có |
| object_drop_dishes | running_footsteps | 20 | có |
| gunshot | running_footsteps | 19 | có |
| running_footsteps | fireworks | 19 | có |
| scream | shout_yell | 18 | có |
| gunshot | door_slam | 15 | có |
| vehicle_crash | running_footsteps | 15 | **chưa** |
| explosion | door_slam | 13 | có |
| explosion | running_footsteps | 13 | có |
| applause_cheering | running_footsteps | 12 | có |

428/778 (55%) lượt nhầm rơi vào cặp đã khai. Phần còn lại là nhầm lẫn mà ontology **chưa dự đoán** — nếu tỉ lệ này cao thì thứ cần sửa là `confusable_with`, không phải model.

## 4. Ngưỡng theo từng lớp

θ tốt nhất **toàn cục** là 0,85. Bảng dưới cho thấy θ đó không tối ưu cho phần lớn lớp:

| lớp | sự kiện thật | θ* riêng | F1 @θ* riêng | F1 @θ=0,85 | chênh |
|---|---|---|---|---|---|
| alarm_bell | 308 | 0,90 | 0,4027 | 0,3905 | +0,0121 |
| applause_cheering | 373 | 0,65 | 0,4883 | 0,3942 | +0,0941 |
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
| siren | 274 | 0,65 | 0,6643 | 0,5785 | +0,0857 |
| speech_normal | 313 | 0,85 | 0,4108 | 0,4108 | +0,0000 |
| vehicle_crash | 319 | 0,85 | 0,3246 | 0,3246 | +0,0000 |

| cấu hình ngưỡng | F1(sự kiện) | F1(đoạn 1s) |
|---|---|---|
| θ = 0,50 (mặc định lúc train) | 0,1529 | 0,3570 |
| θ = 0,85 toàn cục | 0,3918 | 0,6376 |
| θ riêng từng lớp | 0,4153 | 0,6641 |

Hàng cuối chọn 15 ngưỡng trên chính tập đang chấm, nên nó là **chặn trên**, không phải hiệu năng mong đợi trên dữ liệu mới. Muốn dùng số này làm kết quả thì ngưỡng phải khoá trên dev rồi đo một lần trên tập test độc lập — mà `gold_test` hiện **rỗng**.
