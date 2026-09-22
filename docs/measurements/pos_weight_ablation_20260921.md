# Ablation `pos_weight` — số thô, chưa diễn giải — 2026-09-21

Sinh tự động bởi `scripts/chay_ablation_pos_weight.py`. **Trang này chỉ có số**; phần diễn giải và đồng bộ tài liệu làm tay sau, vì nó cần phán đoán.

Mọi lượt `pw*` dùng **cùng cấu hình v3**, chỉ đổi `--pos-weight-max`; cùng split train, cùng `time_pool_blocks=3`, chấm trên cùng `data/synthetic/dev`.

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với train → thiên vị theo thiết kế, **ngang nhau** cho mọi lượt ở đây. θ\* chọn trên chính tập đang chấm → **chặn trên lạc quan** cho mọi lượt.

> ⚠️ `panns_ft_v3` có manifest lập **hồi cứu**; các lượt `pw*` ghi manifest ngay lúc train. Xem cột cảnh báo của `compare_runs`.

| run | pos_weight | ECE tổng | θ\* | event-F1 @θ\* | event-F1 @0,5 |
|---|---:|---:|---:|---:|---:|
| `panns_ft_v3` | chưa đo | 0,3332 | 0,9000 | 0,3979 | 0,0820 |
| `panns_ft_pw1` | 1,0000 | 0,0231 | 0,3500 | 0,4333 | 0,4160 |
| `panns_ft_pw10` | 10,0000 | 0,2033 | 0,8000 | 0,4192 | 0,2340 |
| `panns_ft_pw30` | 30,0000 | 0,3308 | 0,9000 | 0,4018 | 0,0855 |

`panns_ft_v3` hiện `pos_weight` là *chưa đo* vì nó train TRƯỚC khi có cờ `--pos-weight-max`, nên manifest không ghi trường đó. Trần thực tế lúc đó là hằng `MAX_POS_WEIGHT = 30.0` trong mã — đó chính là lý do `pw30` được train lại tử tế thay vì dùng `v3` làm đối chứng.

## Câu hỏi trang này phải trả lời

1. Quan hệ pos_weight ↔ ECE có **đơn điệu** không (1 → 10 → 30)?
2. `pw30` có **tái hiện** được ECE ≈ 0,3332 của `v3` không? Nếu KHÔNG thì chênh lệch đến từ chỗ khác chứ không phải `pos_weight`, và kết luận hôm nay phải rút lại.
3. ECE giảm có kèm F1 giảm không? Nếu có thì đó là **đánh đổi**, không phải cải tiến.

So sánh đầy đủ: `docs/measurements/compare_runs_pos_weight_20260921.md`.

