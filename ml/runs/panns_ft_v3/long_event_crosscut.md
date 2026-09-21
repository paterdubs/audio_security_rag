# `long_event` cắt chéo lát cắt khác — panns_ft_v3 trên dev/all

Sinh bởi `ml/evaluation/long_event_crosscut.py` từ `ml/runs/panns_ft_v3/predictions/dev_all.npz` · theta*=0,90. Không train lại, không GPU.

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với tập train của `v3`, nên dev **thiên vị `v3` theo thiết kế**. theta* chọn trên chính tập đang chấm nên là **chặn trên lạc quan**.

> ⚠️ **Cỡ mẫu nhỏ**: mỗi ô giao chỉ 45–52 clip. Bảng này đủ để nói có/không có tương quan rõ, KHÔNG đủ để nói độ lớn hiệu ứng.

## Tỉ lệ sự kiện bị phân mảnh theo từng ô

| đối chiếu | ô | clip | sự kiện | vỡ | tỉ lệ vỡ |
|---|---|---:|---:|---:|---:|
| low_snr | `long_event+low_snr` | 45 | 163 | 57 | 0,35 |
| low_snr | `long_event-low_snr` | 99 | 396 | 121 | 0,31 |
| low_snr | `khac+low_snr` | 345 | 1286 | 237 | 0,18 |
| low_snr | `khac-low_snr` | 951 | 3028 | 512 | 0,17 |
| reverb | `long_event+reverb` | 52 | 203 | 59 | 0,29 |
| reverb | `long_event-reverb` | 92 | 356 | 119 | 0,33 |
| reverb | `khac+reverb` | 505 | 1685 | 300 | 0,18 |
| reverb | `khac-reverb` | 791 | 2629 | 449 | 0,17 |
| overlap | `long_event+overlap` | 47 | 182 | 53 | 0,29 |
| overlap | `long_event-overlap` | 97 | 377 | 125 | 0,33 |
| overlap | `khac+overlap` | 390 | 1541 | 277 | 0,18 |
| overlap | `khac-overlap` | 906 | 2773 | 472 | 0,17 |
| causal_chain | `long_event+causal_chain` | 15 | 68 | 23 | 0,34 |
| causal_chain | `long_event-causal_chain` | 129 | 491 | 155 | 0,32 |
| causal_chain | `khac+causal_chain` | 188 | 777 | 104 | 0,13 |
| causal_chain | `khac-causal_chain` | 1108 | 3537 | 645 | 0,18 |

## Chênh lệch tỉ lệ vỡ `long_event` trừ `khac`

Nếu chênh lệch còn nguyên ở **cả hai cột**, độ dài là biến giải thích thật. Nếu nó tan đi ở cột *không*, thủ phạm là lát cắt đối chiếu chứ không phải độ dài.

| đối chiếu | chênh lệch ở cột *có* | chênh lệch ở cột *không* |
|---|---:|---:|
| low_snr | 0,17 | 0,14 |
| reverb | 0,11 | 0,16 |
| overlap | 0,11 | 0,16 |
| causal_chain | 0,20 | 0,13 |

## F1 sự kiện và sáu loại lỗi theo từng ô

Tỉ lệ lấy trên số sự kiện THAM CHIẾU của chính ô đó, nên `thua` vượt 1,0 được — một sự kiện thật có thể hứng nhiều dự báo thừa. `bien` là phép CHIA NHỎ LẠI rổ Deletion+Insertion của sed_eval, không phải loại lỗi thứ bảy song song.

| ô | clip | sự kiện | event-F1 | đúng | biên | thay thế | thiếu | thừa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `long_event+low_snr` | 45 | 163 | 0,19 | 0,29 | 0,34 | 0,19 | 0,18 | 1,25 |
| `long_event-low_snr` | 99 | 396 | 0,30 | 0,43 | 0,26 | 0,15 | 0,16 | 1,00 |
| `khac+low_snr` | 345 | 1286 | 0,36 | 0,42 | 0,21 | 0,16 | 0,20 | 0,55 |
| `khac-low_snr` | 951 | 3028 | 0,44 | 0,53 | 0,18 | 0,14 | 0,15 | 0,54 |
| `long_event+reverb` | 52 | 203 | 0,26 | 0,36 | 0,24 | 0,20 | 0,20 | 0,99 |
| `long_event-reverb` | 92 | 356 | 0,27 | 0,40 | 0,30 | 0,14 | 0,15 | 1,12 |
| `khac+reverb` | 505 | 1685 | 0,39 | 0,46 | 0,19 | 0,15 | 0,20 | 0,54 |
| `khac-reverb` | 791 | 2629 | 0,44 | 0,52 | 0,19 | 0,15 | 0,14 | 0,54 |
| `long_event+overlap` | 47 | 182 | 0,28 | 0,41 | 0,26 | 0,16 | 0,17 | 1,04 |
| `long_event-overlap` | 97 | 377 | 0,26 | 0,38 | 0,29 | 0,16 | 0,17 | 1,09 |
| `khac+overlap` | 390 | 1541 | 0,39 | 0,46 | 0,20 | 0,15 | 0,20 | 0,52 |
| `khac-overlap` | 906 | 2773 | 0,43 | 0,52 | 0,18 | 0,15 | 0,15 | 0,56 |
| `long_event+causal_chain` | 15 | 68 | 0,32 | 0,43 | 0,26 | 0,12 | 0,19 | 0,87 |
| `long_event-causal_chain` | 129 | 491 | 0,26 | 0,38 | 0,28 | 0,17 | 0,16 | 1,10 |
| `khac+causal_chain` | 188 | 777 | 0,45 | 0,51 | 0,17 | 0,13 | 0,20 | 0,42 |
| `khac-causal_chain` | 1108 | 3537 | 0,41 | 0,50 | 0,19 | 0,15 | 0,16 | 0,57 |

## Cặp nhầm đậm nhất theo từng ô

Ngoài đường chéo (đường chéo gồm cả `dung` lẫn `bien`), tối đa 5 cặp mỗi ô.

| ô | cặp nhầm (thật → đoán) |
|---|---|
| `long_event+low_snr` | alarm_bell→glass_breaking (1) · applause_cheering→alarm_bell (1) · door_slam→alarm_bell (1) · door_slam→running_footsteps (1) · explosion→door_slam (1) |
| `long_event-low_snr` | door_slam→running_footsteps (3) · object_drop_dishes→fireworks (3) · vehicle_crash→shout_yell (3) · vehicle_crash→speech_normal (3) · alarm_bell→applause_cheering (2) |
| `khac+low_snr` | fireworks→running_footsteps (10) · object_drop_dishes→glass_breaking (10) · laughter→applause_cheering (6) · laughter→shout_yell (6) · object_drop_dishes→door_slam (6) |
| `khac-low_snr` | object_drop_dishes→door_slam (21) · door_slam→running_footsteps (16) · gunshot→running_footsteps (15) · object_drop_dishes→glass_breaking (15) · fireworks→running_footsteps (14) |
| `long_event+reverb` | vehicle_crash→speech_normal (3) · object_drop_dishes→fireworks (2) · running_footsteps→fireworks (2) · vehicle_crash→fireworks (2) · vehicle_crash→shout_yell (2) |
| `long_event-reverb` | door_slam→running_footsteps (4) · applause_cheering→alarm_bell (2) · fireworks→running_footsteps (2) · laughter→applause_cheering (2) · alarm_bell→applause_cheering (1) |
| `khac+reverb` | fireworks→running_footsteps (10) · object_drop_dishes→door_slam (10) · object_drop_dishes→glass_breaking (10) · gunshot→door_slam (9) · object_drop_dishes→gunshot (8) |
| `khac-reverb` | door_slam→running_footsteps (17) · object_drop_dishes→door_slam (17) · object_drop_dishes→glass_breaking (15) · fireworks→running_footsteps (14) · gunshot→running_footsteps (14) |
| `long_event+overlap` | applause_cheering→alarm_bell (2) · applause_cheering→gunshot (1) · door_slam→alarm_bell (1) · door_slam→gunshot (1) · explosion→siren (1) |
| `long_event-overlap` | door_slam→running_footsteps (4) · vehicle_crash→shout_yell (3) · vehicle_crash→speech_normal (3) · alarm_bell→applause_cheering (2) · fireworks→alarm_bell (2) |
| `khac+overlap` | object_drop_dishes→door_slam (10) · object_drop_dishes→glass_breaking (10) · fireworks→running_footsteps (7) · door_slam→running_footsteps (6) · vehicle_crash→door_slam (6) |
| `khac-overlap` | fireworks→running_footsteps (17) · object_drop_dishes→door_slam (17) · object_drop_dishes→glass_breaking (15) · gunshot→running_footsteps (13) · object_drop_dishes→running_footsteps (13) |
| `long_event+causal_chain` | alarm_bell→glass_breaking (1) · alarm_bell→siren (1) · door_slam→running_footsteps (1) · glass_breaking→door_slam (1) · gunshot→applause_cheering (1) |
| `long_event-causal_chain` | vehicle_crash→speech_normal (4) · door_slam→running_footsteps (3) · object_drop_dishes→fireworks (3) · vehicle_crash→shout_yell (3) · alarm_bell→applause_cheering (2) |
| `khac+causal_chain` | running_footsteps→gunshot (8) · door_slam→running_footsteps (6) · running_footsteps→explosion (6) · object_drop_dishes→glass_breaking (5) · object_drop_dishes→running_footsteps (5) |
| `khac-causal_chain` | object_drop_dishes→door_slam (25) · fireworks→running_footsteps (20) · object_drop_dishes→glass_breaking (20) · gunshot→running_footsteps (17) · speech_normal→shout_yell (16) |

