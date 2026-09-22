# Ablation `time_pool_blocks` — số thô, chưa diễn giải — 2026-09-22

Sinh tự động bởi `scripts/chay_ablation_time_pool.py`. **Trang này chỉ có số**; phần diễn giải và đồng bộ tài liệu làm tay sau, vì nó cần phán đoán.

`panns_ft_tpb2` (time_pool_blocks=2) so với đối chứng `panns_ft_pw1` (time_pool_blocks=3); hai lượt khoá cùng `--pos-weight-max 1.0`, cùng mọi tham số khác, cùng split train, chấm trên cùng `data/synthetic/dev`.

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với train → thiên vị theo thiết kế, **ngang nhau** cho mọi lượt ở đây. θ\* chọn trên chính tập đang chấm → **chặn trên lạc quan** cho mọi lượt. 0,8917 số clip dev có ít nhất một nguồn foreground từng dùng trong train (`measurements/leakage_check_20260922.md`).

> ⚠️ **Không đọc cột thời gian của manifest để so giữa các run**: lượt này chạy ban ngày với GPU bị thảo nhiệt chặn xuống 210/2100 MHz, các lượt trước chạy ban đêm. Chỉ số chất lượng không bị ảnh hưởng.

| run | time_pool_blocks | ECE tổng | θ\* | event-F1 @θ\* | long_event: tỉ lệ vỡ | long_event: F1 |
|---|---:|---:|---:|---:|---:|---:|
| `panns_ft_v3` | 3 | 0,3332 | 0,9000 | 0,3979 | 0,3184 | 0,2685 |
| `panns_ft_pw1` | 3 | 0,0231 | 0,3500 | 0,4333 | 0,2987 | 0,3182 |
| `panns_ft_tpb2` | 2 | 0,0247 | 0,3000 | 0,3466 | 0,3918 | 0,2422 |

## Câu hỏi trang này phải trả lời

1. Tỉ lệ vỡ `long_event` của `tpb2` (time_pool_blocks=2) có THẤP hơn `pw1` (time_pool_blocks=3) không? Nếu có, trường tiếp nhận thời gian ngắn là một phần nguyên nhân thật của `long_event`. Nếu KHÔNG đổi hoặc đổi ngược, giả thuyết bị bác bỏ — và đó là ứng viên thứ NĂM bị loại.
2. F1 tổng của `tpb2` có đánh đổi lấy tỉ lệ vỡ thấp hơn không, hay thắng thuần? Ablation `pos_weight` đã cho một ca thắng thuần, đừng mặc định phải có đánh đổi.
3. Nếu tỉ lệ vỡ giảm mà F1 `long_event` KHÔNG tăng: phân mảnh không phải thứ đang giới hạn F1 trên sự kiện dài, và cả hướng điều tra này cần đặt lại đề.

