# Khe hở năng lượng bên trong `long_event` — panns_ft_v3 (dev/all)

Sinh bởi `ml/evaluation/long_event_gap.py`. 144 clip lát cắt `long_event`, 315 sự kiện đạt ngưỡng độ dài ≥1,00s (đã bỏ sự kiện ngắn hơn — không thể chứa khe hở có ý nghĩa). Ngưỡng "im": RMS < 20% trung vị RMS của chính sự kiện đó.

> ⚠️ `data/synthetic/dev` sinh cùng recipe B0–B9 với train của v3 → thiên vị v3 theo thiết kế. θ\* dùng ở đây lấy từ `analysis.json` của chính run — vẫn chọn trên tập đang chấm.

## So khe hở: nhóm bị phân mảnh vs nhóm không, trong CÙNG lát cắt

| nhóm | n sự kiện | khe hở trung vị | khe hở p75 |
|---|---|---|---|
| bị phân mảnh | 178 | 0,000s | 0,000s |
| không phân mảnh | 137 | 0,000s | 0,000s |

Chênh lệch trung vị (vỡ − nguyên) = **0,000s**.

Hai nhóm gần như không khác nhau — giả thuyết khe hở năng lượng thật bị **bác bỏ**. `long_event` đã loại BA ứng viên liên tiếp (cửa sổ lọc hẹp, trần cửa sổ thấp, khe hở năng lượng); nguyên nhân riêng của nó **vẫn chưa xác định**.
