# Đối chiếu 3 lần chạy: panns_ft_v3 · panns_ft_pw1 · panns_ft_tpb2

Sinh bởi `ml/tracking/compare_runs.py` (Pha 5). Mỗi con số dưới đây đi kèm chỗ nó đến từ.

## 1. Metric

Nguồn: `analysis.json` của từng run — chấm trên cùng dev/all; θ* chọn trên chính tập đang chấm → chặn trên lạc quan.

| run | θ* | F1(sự kiện) @θ* | F1(đoạn 1s) @θ* | F1(sự kiện) @θ=0,50 |
|---|---|---|---|---|
| panns_ft_v3 | 0,90 | 0,3979 | 0,7155 | 0,0820 |
| panns_ft_pw1 | 0,35 | 0,4333 | 0,7486 | 0,4160 |
| panns_ft_tpb2 | 0,30 | 0,3466 | 0,7275 | 0,3252 |

## 2. Dữ liệu

Vân tay nhãn TRÙNG nhau giữa các run.

| run | vân tay nhãn (sha256) |
|---|---|
| panns_ft_v3 | `8ef43675247ecd87…` |
| panns_ft_pw1 | `8ef43675247ecd87…` |
| panns_ft_tpb2 | `8ef43675247ecd87…` |

## 3. Mã nguồn

> ⚠️ **KHONG_RO** — manifest lập hồi cứu (panns_ft_v3): băm mã là mã ở thời điểm lập manifest, không phải lúc train.

## 4. Cấu hình

`<không có>` nghĩa là khoá đó **không tồn tại** trong manifest của run — thường vì cờ ra đời sau run đó, tức run chạy bằng một phiên bản code khác. Đó là chuyện khác hẳn với chỉnh tham số.

| khoá | panns_ft_v3 | panns_ft_pw1 | panns_ft_tpb2 |
|---|---|---|---|
| force_du_lieu_chua_dat | `<không có>` | False | False |
| khong_bam_waveform | `<không có>` | False | False |
| pos_weight_max | `<không có>` | 1.0 | 1.0 |
| resume | `<không có>` | False | False |
| time_pool_blocks | 3 | 3 | 2 |

