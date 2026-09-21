# `long_event` cắt chéo lát cắt khác — panns_ft_v2 trên dev/all

Sinh bởi `ml/evaluation/long_event_crosscut.py` từ `ml/runs/panns_ft_v2/predictions/dev_all.npz` · theta*=0,90. Không train lại, không GPU.

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với tập train của `v3`, nên dev **thiên vị `v3` theo thiết kế**. theta* chọn trên chính tập đang chấm nên là **chặn trên lạc quan**.

> ⚠️ **Cỡ mẫu nhỏ**: mỗi ô giao chỉ 45–52 clip. Bảng này đủ để nói có/không có tương quan rõ, KHÔNG đủ để nói độ lớn hiệu ứng.

## Tỉ lệ sự kiện bị phân mảnh theo từng ô

| đối chiếu | ô | clip | sự kiện | vỡ | tỉ lệ vỡ |
|---|---|---:|---:|---:|---:|
| low_snr | `long_event+low_snr` | 45 | 163 | 51 | 0,31 |
| low_snr | `long_event-low_snr` | 99 | 396 | 115 | 0,29 |
| low_snr | `khac+low_snr` | 345 | 1286 | 239 | 0,19 |
| low_snr | `khac-low_snr` | 951 | 3028 | 514 | 0,17 |
| reverb | `long_event+reverb` | 52 | 203 | 54 | 0,27 |
| reverb | `long_event-reverb` | 92 | 356 | 112 | 0,31 |
| reverb | `khac+reverb` | 505 | 1685 | 291 | 0,17 |
| reverb | `khac-reverb` | 791 | 2629 | 462 | 0,18 |
| overlap | `long_event+overlap` | 47 | 182 | 55 | 0,30 |
| overlap | `long_event-overlap` | 97 | 377 | 111 | 0,29 |
| overlap | `khac+overlap` | 390 | 1541 | 268 | 0,17 |
| overlap | `khac-overlap` | 906 | 2773 | 485 | 0,17 |
| causal_chain | `long_event+causal_chain` | 15 | 68 | 16 | 0,24 |
| causal_chain | `long_event-causal_chain` | 129 | 491 | 150 | 0,31 |
| causal_chain | `khac+causal_chain` | 188 | 777 | 112 | 0,14 |
| causal_chain | `khac-causal_chain` | 1108 | 3537 | 641 | 0,18 |

## Chênh lệch tỉ lệ vỡ `long_event` trừ `khac`

Nếu chênh lệch còn nguyên ở **cả hai cột**, độ dài là biến giải thích thật. Nếu nó tan đi ở cột *không*, thủ phạm là lát cắt đối chiếu chứ không phải độ dài.

| đối chiếu | chênh lệch ở cột *có* | chênh lệch ở cột *không* |
|---|---:|---:|
| low_snr | 0,13 | 0,12 |
| reverb | 0,09 | 0,14 |
| overlap | 0,13 | 0,12 |
| causal_chain | 0,09 | 0,12 |

## F1 sự kiện và sáu loại lỗi theo từng ô

Tỉ lệ lấy trên số sự kiện THAM CHIẾU của chính ô đó, nên `thua` vượt 1,0 được — một sự kiện thật có thể hứng nhiều dự báo thừa. `bien` là phép CHIA NHỎ LẠI rổ Deletion+Insertion của sed_eval, không phải loại lỗi thứ bảy song song.

| ô | clip | sự kiện | event-F1 | đúng | biên | thay thế | thiếu | thừa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `long_event+low_snr` | 45 | 163 | 0,17 | 0,25 | 0,33 | 0,21 | 0,20 | 1,17 |
| `long_event-low_snr` | 99 | 396 | 0,27 | 0,40 | 0,23 | 0,18 | 0,19 | 1,15 |
| `khac+low_snr` | 345 | 1286 | 0,32 | 0,37 | 0,20 | 0,16 | 0,27 | 0,57 |
| `khac-low_snr` | 951 | 3028 | 0,39 | 0,47 | 0,16 | 0,16 | 0,21 | 0,63 |
| `long_event+reverb` | 52 | 203 | 0,25 | 0,34 | 0,21 | 0,24 | 0,21 | 0,96 |
| `long_event-reverb` | 92 | 356 | 0,23 | 0,36 | 0,29 | 0,17 | 0,19 | 1,27 |
| `khac+reverb` | 505 | 1685 | 0,34 | 0,40 | 0,18 | 0,16 | 0,27 | 0,57 |
| `khac-reverb` | 791 | 2629 | 0,39 | 0,47 | 0,17 | 0,16 | 0,20 | 0,63 |
| `long_event+overlap` | 47 | 182 | 0,24 | 0,36 | 0,28 | 0,17 | 0,19 | 1,16 |
| `long_event-overlap` | 97 | 377 | 0,24 | 0,35 | 0,25 | 0,20 | 0,20 | 1,16 |
| `khac+overlap` | 390 | 1541 | 0,35 | 0,40 | 0,18 | 0,17 | 0,25 | 0,56 |
| `khac-overlap` | 906 | 2773 | 0,38 | 0,46 | 0,17 | 0,15 | 0,21 | 0,64 |
| `long_event+causal_chain` | 15 | 68 | 0,27 | 0,37 | 0,28 | 0,10 | 0,25 | 0,96 |
| `long_event-causal_chain` | 129 | 491 | 0,24 | 0,35 | 0,25 | 0,20 | 0,19 | 1,19 |
| `khac+causal_chain` | 188 | 777 | 0,42 | 0,45 | 0,17 | 0,12 | 0,26 | 0,41 |
| `khac-causal_chain` | 1108 | 3537 | 0,36 | 0,44 | 0,17 | 0,17 | 0,22 | 0,65 |

## Cặp nhầm đậm nhất theo từng ô

Ngoài đường chéo (đường chéo gồm cả `dung` lẫn `bien`), tối đa 5 cặp mỗi ô.

| ô | cặp nhầm (thật → đoán) |
|---|---|
| `long_event+low_snr` | shout_yell→speech_normal (3) · speech_normal→alarm_bell (2) · vehicle_crash→alarm_bell (2) · vehicle_crash→siren (2) · applause_cheering→alarm_bell (1) |
| `long_event-low_snr` | alarm_bell→siren (5) · fireworks→running_footsteps (3) · laughter→running_footsteps (3) · object_drop_dishes→fireworks (3) · object_drop_dishes→running_footsteps (3) |
| `khac+low_snr` | fireworks→running_footsteps (10) · speech_normal→shout_yell (9) · running_footsteps→fireworks (8) · alarm_bell→siren (7) · object_drop_dishes→door_slam (7) |
| `khac-low_snr` | object_drop_dishes→glass_breaking (19) · object_drop_dishes→door_slam (16) · running_footsteps→fireworks (16) · gunshot→running_footsteps (15) · speech_normal→shout_yell (12) |
| `long_event+reverb` | running_footsteps→door_slam (3) · alarm_bell→siren (2) · explosion→siren (2) · gunshot→door_slam (2) · object_drop_dishes→fireworks (2) |
| `long_event-reverb` | alarm_bell→siren (3) · fireworks→running_footsteps (2) · gunshot→shout_yell (2) · laughter→running_footsteps (2) · running_footsteps→alarm_bell (2) |
| `khac+reverb` | running_footsteps→fireworks (15) · fireworks→running_footsteps (10) · object_drop_dishes→door_slam (10) · speech_normal→shout_yell (9) · gunshot→door_slam (7) |
| `khac-reverb` | object_drop_dishes→glass_breaking (18) · object_drop_dishes→door_slam (13) · gunshot→running_footsteps (12) · speech_normal→shout_yell (12) · door_slam→running_footsteps (11) |
| `long_event+overlap` | object_drop_dishes→fireworks (2) · running_footsteps→fireworks (2) · shout_yell→speech_normal (2) · alarm_bell→siren (1) · applause_cheering→alarm_bell (1) |
| `long_event-overlap` | alarm_bell→siren (4) · explosion→siren (3) · object_drop_dishes→running_footsteps (3) · running_footsteps→door_slam (3) · vehicle_crash→shout_yell (3) |
| `khac+overlap` | object_drop_dishes→glass_breaking (10) · gunshot→running_footsteps (8) · object_drop_dishes→door_slam (8) · running_footsteps→fireworks (8) · door_slam→running_footsteps (7) |
| `khac-overlap` | speech_normal→shout_yell (18) · running_footsteps→fireworks (16) · object_drop_dishes→door_slam (15) · object_drop_dishes→glass_breaking (15) · fireworks→running_footsteps (12) |
| `long_event+causal_chain` | alarm_bell→siren (2) · fireworks→running_footsteps (1) · gunshot→applause_cheering (1) · running_footsteps→alarm_bell (1) · running_footsteps→door_slam (1) |
| `long_event-causal_chain` | shout_yell→speech_normal (4) · vehicle_crash→siren (4) · alarm_bell→siren (3) · explosion→siren (3) · laughter→running_footsteps (3) |
| `khac+causal_chain` | fireworks→running_footsteps (7) · running_footsteps→fireworks (7) · door_slam→running_footsteps (5) · object_drop_dishes→door_slam (4) · object_drop_dishes→glass_breaking (4) |
| `khac-causal_chain` | object_drop_dishes→glass_breaking (21) · speech_normal→shout_yell (20) · object_drop_dishes→door_slam (19) · running_footsteps→fireworks (17) · gunshot→running_footsteps (15) |

