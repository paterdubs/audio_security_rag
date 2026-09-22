# `long_event` cắt chéo lát cắt khác — panns_ft_pw1 trên dev/all

Sinh bởi `ml/evaluation/long_event_crosscut.py` từ `ml/runs/panns_ft_pw1/predictions/dev_all.npz` · theta*=0,35. Không train lại, không GPU.

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với tập train của `v3`, nên dev **thiên vị `v3` theo thiết kế**. theta* chọn trên chính tập đang chấm nên là **chặn trên lạc quan**.

> ⚠️ **Cỡ mẫu nhỏ**: mỗi ô giao chỉ 45–52 clip. Bảng này đủ để nói có/không có tương quan rõ, KHÔNG đủ để nói độ lớn hiệu ứng.

## Tỉ lệ sự kiện bị phân mảnh theo từng ô

| đối chiếu | ô | clip | sự kiện | vỡ | tỉ lệ vỡ |
|---|---|---:|---:|---:|---:|
| low_snr | `long_event+low_snr` | 45 | 163 | 56 | 0,34 |
| low_snr | `long_event-low_snr` | 99 | 396 | 111 | 0,28 |
| low_snr | `khac+low_snr` | 345 | 1286 | 219 | 0,17 |
| low_snr | `khac-low_snr` | 951 | 3028 | 500 | 0,17 |
| reverb | `long_event+reverb` | 52 | 203 | 56 | 0,28 |
| reverb | `long_event-reverb` | 92 | 356 | 111 | 0,31 |
| reverb | `khac+reverb` | 505 | 1685 | 291 | 0,17 |
| reverb | `khac-reverb` | 791 | 2629 | 428 | 0,16 |
| overlap | `long_event+overlap` | 47 | 182 | 49 | 0,27 |
| overlap | `long_event-overlap` | 97 | 377 | 118 | 0,31 |
| overlap | `khac+overlap` | 390 | 1541 | 276 | 0,18 |
| overlap | `khac-overlap` | 906 | 2773 | 443 | 0,16 |
| causal_chain | `long_event+causal_chain` | 15 | 68 | 17 | 0,25 |
| causal_chain | `long_event-causal_chain` | 129 | 491 | 150 | 0,31 |
| causal_chain | `khac+causal_chain` | 188 | 777 | 101 | 0,13 |
| causal_chain | `khac-causal_chain` | 1108 | 3537 | 618 | 0,17 |

## Chênh lệch tỉ lệ vỡ `long_event` trừ `khac`

Nếu chênh lệch còn nguyên ở **cả hai cột**, độ dài là biến giải thích thật. Nếu nó tan đi ở cột *không*, thủ phạm là lát cắt đối chiếu chứ không phải độ dài.

| đối chiếu | chênh lệch ở cột *có* | chênh lệch ở cột *không* |
|---|---:|---:|
| low_snr | 0,17 | 0,12 |
| reverb | 0,10 | 0,15 |
| overlap | 0,09 | 0,15 |
| causal_chain | 0,12 | 0,13 |

## F1 sự kiện và sáu loại lỗi theo từng ô

Tỉ lệ lấy trên số sự kiện THAM CHIẾU của chính ô đó, nên `thua` vượt 1,0 được — một sự kiện thật có thể hứng nhiều dự báo thừa. `bien` là phép CHIA NHỎ LẠI rổ Deletion+Insertion của sed_eval, không phải loại lỗi thứ bảy song song.

| ô | clip | sự kiện | event-F1 | đúng | biên | thay thế | thiếu | thừa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `long_event+low_snr` | 45 | 163 | 0,26 | 0,37 | 0,28 | 0,17 | 0,18 | 1,04 |
| `long_event-low_snr` | 99 | 396 | 0,34 | 0,48 | 0,23 | 0,15 | 0,15 | 0,92 |
| `khac+low_snr` | 345 | 1286 | 0,39 | 0,43 | 0,21 | 0,15 | 0,21 | 0,47 |
| `khac-low_snr` | 951 | 3028 | 0,48 | 0,55 | 0,16 | 0,13 | 0,15 | 0,45 |
| `long_event+reverb` | 52 | 203 | 0,30 | 0,40 | 0,23 | 0,19 | 0,18 | 0,84 |
| `long_event-reverb` | 92 | 356 | 0,33 | 0,47 | 0,25 | 0,13 | 0,15 | 1,03 |
| `khac+reverb` | 505 | 1685 | 0,42 | 0,48 | 0,18 | 0,14 | 0,21 | 0,46 |
| `khac-reverb` | 791 | 2629 | 0,47 | 0,54 | 0,17 | 0,14 | 0,15 | 0,45 |
| `long_event+overlap` | 47 | 182 | 0,33 | 0,45 | 0,23 | 0,15 | 0,18 | 0,87 |
| `long_event-overlap` | 97 | 377 | 0,31 | 0,44 | 0,25 | 0,15 | 0,15 | 1,00 |
| `khac+overlap` | 390 | 1541 | 0,42 | 0,47 | 0,19 | 0,14 | 0,20 | 0,45 |
| `khac-overlap` | 906 | 2773 | 0,47 | 0,54 | 0,17 | 0,13 | 0,15 | 0,46 |
| `long_event+causal_chain` | 15 | 68 | 0,41 | 0,53 | 0,19 | 0,09 | 0,19 | 0,78 |
| `long_event-causal_chain` | 129 | 491 | 0,31 | 0,43 | 0,25 | 0,16 | 0,15 | 0,98 |
| `khac+causal_chain` | 188 | 777 | 0,49 | 0,52 | 0,17 | 0,12 | 0,19 | 0,33 |
| `khac-causal_chain` | 1108 | 3537 | 0,44 | 0,52 | 0,18 | 0,14 | 0,17 | 0,48 |

## Cặp nhầm đậm nhất theo từng ô

Ngoài đường chéo (đường chéo gồm cả `dung` lẫn `bien`), tối đa 5 cặp mỗi ô.

| ô | cặp nhầm (thật → đoán) |
|---|---|
| `long_event+low_snr` | applause_cheering→alarm_bell (1) · door_slam→alarm_bell (1) · door_slam→running_footsteps (1) · explosion→running_footsteps (1) · explosion→siren (1) |
| `long_event-low_snr` | object_drop_dishes→fireworks (4) · door_slam→running_footsteps (3) · vehicle_crash→fireworks (3) · fireworks→running_footsteps (2) · laughter→alarm_bell (2) |
| `khac+low_snr` | object_drop_dishes→glass_breaking (10) · door_slam→running_footsteps (8) · fireworks→running_footsteps (8) · object_drop_dishes→door_slam (8) · speech_normal→shout_yell (7) |
| `khac-low_snr` | door_slam→running_footsteps (19) · object_drop_dishes→running_footsteps (17) · speech_normal→shout_yell (16) · object_drop_dishes→door_slam (15) · running_footsteps→fireworks (15) |
| `long_event+reverb` | object_drop_dishes→fireworks (3) · running_footsteps→door_slam (2) · running_footsteps→fireworks (2) · shout_yell→object_drop_dishes (2) · vehicle_crash→applause_cheering (2) |
| `long_event-reverb` | door_slam→running_footsteps (3) · applause_cheering→alarm_bell (2) · explosion→running_footsteps (2) · laughter→alarm_bell (2) · laughter→shout_yell (2) |
| `khac+reverb` | running_footsteps→fireworks (13) · speech_normal→shout_yell (10) · fireworks→running_footsteps (9) · object_drop_dishes→running_footsteps (9) · gunshot→door_slam (8) |
| `khac-reverb` | door_slam→running_footsteps (23) · object_drop_dishes→glass_breaking (17) · object_drop_dishes→door_slam (15) · speech_normal→shout_yell (13) · gunshot→running_footsteps (12) |
| `long_event+overlap` | applause_cheering→alarm_bell (2) · object_drop_dishes→fireworks (2) · vehicle_crash→alarm_bell (2) · applause_cheering→running_footsteps (1) · door_slam→alarm_bell (1) |
| `long_event-overlap` | door_slam→running_footsteps (4) · object_drop_dishes→fireworks (3) · explosion→running_footsteps (2) · laughter→alarm_bell (2) · running_footsteps→door_slam (2) |
| `khac+overlap` | object_drop_dishes→glass_breaking (12) · door_slam→running_footsteps (9) · speech_normal→shout_yell (7) · object_drop_dishes→door_slam (6) · fireworks→running_footsteps (5) |
| `khac-overlap` | door_slam→running_footsteps (18) · object_drop_dishes→door_slam (17) · object_drop_dishes→running_footsteps (16) · running_footsteps→fireworks (16) · speech_normal→shout_yell (16) |
| `long_event+causal_chain` | door_slam→running_footsteps (1) · glass_breaking→door_slam (1) · gunshot→applause_cheering (1) · running_footsteps→door_slam (1) · shout_yell→object_drop_dishes (1) |
| `long_event-causal_chain` | object_drop_dishes→fireworks (5) · door_slam→running_footsteps (3) · vehicle_crash→fireworks (3) · vehicle_crash→speech_normal (3) · applause_cheering→alarm_bell (2) |
| `khac+causal_chain` | door_slam→running_footsteps (8) · object_drop_dishes→running_footsteps (6) · door_slam→fireworks (5) · glass_breaking→running_footsteps (4) · object_drop_dishes→glass_breaking (4) |
| `khac-causal_chain` | speech_normal→shout_yell (23) · object_drop_dishes→door_slam (21) · door_slam→running_footsteps (19) · object_drop_dishes→glass_breaking (19) · running_footsteps→fireworks (17) |

