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

## F1 sự kiện theo từng ô

| ô | clip | sự kiện | event-F1 | segment-F1 |
|---|---:|---:|---:|---:|
| `long_event+low_snr` | 45 | 163 | 0,19 | 0,69 |
| `long_event-low_snr` | 99 | 396 | 0,30 | 0,72 |
| `khac+low_snr` | 345 | 1286 | 0,36 | 0,68 |
| `khac-low_snr` | 951 | 3028 | 0,44 | 0,73 |
| `long_event+reverb` | 52 | 203 | 0,26 | 0,65 |
| `long_event-reverb` | 92 | 356 | 0,27 | 0,75 |
| `khac+reverb` | 505 | 1685 | 0,39 | 0,69 |
| `khac-reverb` | 791 | 2629 | 0,44 | 0,73 |
| `long_event+overlap` | 47 | 182 | 0,28 | 0,71 |
| `long_event-overlap` | 97 | 377 | 0,26 | 0,71 |
| `khac+overlap` | 390 | 1541 | 0,39 | 0,71 |
| `khac-overlap` | 906 | 2773 | 0,43 | 0,72 |
| `long_event+causal_chain` | 15 | 68 | 0,32 | 0,74 |
| `long_event-causal_chain` | 129 | 491 | 0,26 | 0,71 |
| `khac+causal_chain` | 188 | 777 | 0,45 | 0,69 |
| `khac-causal_chain` | 1108 | 3537 | 0,41 | 0,72 |

