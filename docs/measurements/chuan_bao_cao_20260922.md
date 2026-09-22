# Chuẩn báo cáo mới: hậu xử lý thích ứng-từ-train thay cho cửa sổ cố định — 2026-09-22

Tiếp theo phát hiện trong [time_pool_ablation_20260922.md](time_pool_ablation_20260922.md):
lúc so `panns_ft_pw1` với `panns_ft_tpb2`, việc điều tra một nghi vấn nhiễu (kết luận cuối
là **không có nhiễu**, xem §4) tình cờ lộ ra rằng cách chấm mặc định hiện tại không tối ưu.
Trang này chốt lại chuẩn chấm mới và chấm lại toàn bộ bảy run dưới chuẩn đó.

---

## 1. Ba cách chấm đang tồn tại trong mã, và tại sao chỉ một cái được dùng

`ml/evaluation/threshold_sweep.py::cua_so_loc()` có ba chế độ lọc trung vị hậu xử lý:

| Chế độ | Cờ | Nguồn cửa sổ | Rò rỉ? |
|---|---|---|---|
| **Cố định** | (mặc định, không cờ) | hằng số `DEFAULT_MEDIAN_FILTER_FRAMES = 7` khung | Không, nhưng là một con số chọn tuỳ tiện |
| **Thích ứng-từ-dev** | `--adaptive-postproc` (không `--adaptive-source`) | suy từ độ dài sự kiện của **chính tập đang chấm** | **Có** — dùng nhãn của tập đánh giá để chọn siêu tham số |
| **Thích ứng-từ-train** | `--adaptive-postproc --adaptive-source train` | suy từ độ dài sự kiện của **tập train** | Không |

Bảy run của dự án (`v1`–`v3`, `pw1`/`pw10`/`pw30`, `tpb2`) đến nay đều được báo cáo bằng
chế độ **cố định**. Không sai — nó không rò rỉ — nhưng `7` khung là một con số chọn trước,
không suy từ dữ liệu, và **không nhất quán về mặt thời gian thật** giữa các cấu hình khác
`time_pool_blocks` (xem §4). Chế độ thích ứng-từ-dev bị loại thẳng vì rò rỉ.

## 2. Quyết định: chuyển sang thích ứng-từ-train làm chuẩn báo cáo

Lý do, không phải vì nó cho số đẹp hơn:

1. **Không rò rỉ** — suy từ thống kê tập train, đúng thứ `train_sed.py::median_sizes_for`
   đã dùng lúc train, không đụng nhãn của tập đang chấm.
2. **Nhất quán theo thời gian thật.** Cửa sổ tính theo khung nhưng số khung/giây đổi theo
   `time_pool_blocks`; suy từ độ dài sự kiện thật thì cửa sổ luôn tương ứng đúng khoảng
   thời gian bất kể cấu hình khung.
3. **Cải thiện đo được trên chính `pw1`**, không phải suy luận: F1@θ\* +8% (0,4333→0,4682),
   F1 `long_event` +13% (0,3182→0,3602), tỉ lệ vỡ `long_event` −17% (0,2987→0,2469).

## 3. Bảng bảy run, chấm lại đồng nhất — THAY THẾ mọi bảng trước đó dùng cửa sổ cố định

Sinh bằng `error_analysis.py --adaptive-postproc --adaptive-source train` cho cả 7 run,
ghi vào `ml/runs/<run>/analysis_adaptive_train.json`. Không train lại, không GPU (~4–5
phút CPU mỗi run, dự đoán đã lưu sẵn).

| run | θ\* | F1@θ\* | F1@0,5 | `long_event` F1 | `sạch` F1 |
|---|---:|---:|---:|---:|---:|
| `panns_ft_v1` | 0,95 | 0,2953 | 0,1147 | 0,2592 | 0,3324 |
| `panns_ft_v2` | 0,85 | 0,3911 | 0,1496 | 0,2881 | 0,4203 |
| `panns_ft_v3` | 0,85 | 0,4347 | 0,1111 | 0,3067 | 0,4846 |
| `panns_ft_pw1` | **0,25** | **0,4682** | 0,4335 | **0,3602** | **0,5140** |
| `panns_ft_pw10` | 0,75 | 0,4579 | 0,3048 | 0,3415 | 0,5048 |
| `panns_ft_pw30` | 0,85 | 0,4397 | 0,1152 | 0,3130 | 0,4931 |
| `panns_ft_tpb2` | 0,25 | 0,4113 | 0,3245 | 0,3066 | 0,4749 |

`panns_ft_pw1` vẫn là mốc dẫn đầu ở mọi cột — kết luận của ablation `pos_weight`
([pos_weight_ket_luan_20260922.md](pos_weight_ket_luan_20260922.md)) **không đổi**, chỉ
con số chính xác hơn.

> ⚠️ Không so hàng này với các bảng ở `pos_weight_ket_luan_20260922.md` hay
> `time_pool_ablation_20260922.md` — hai bảng đó dùng cửa sổ cố định. Mọi bảng SAU trang
> này nên dùng chế độ thích ứng-từ-train, và ghi rõ nếu không dùng.

## 4. Rút lại một khẳng định đã đưa ra sáng nay — nói thẳng vì đã sai

Lúc so `pw1` với `tpb2`, tôi từng khẳng định phép so bị nhiễu vì "cửa sổ 7 khung ứng với
560 ms ở `pw1` nhưng chỉ 280 ms ở `tpb2`". **Sai.** Kiểm `predictions/dev_all.npz` cho thấy
lọc trung vị chạy SAU khi nội suy đoạn lên `n_frames=1001` khung cho **cả hai** run — nên
`7` khung là ~70 ms ở cả hai, và phép so ở cửa sổ cố định vốn đã công bằng. Đã rút lại
ngay khi phát hiện, không sửa lại kết luận cũ mà ghi ở đây để không lặp lại cách suy luận
đó (khẳng định cơ chế từ việc khớp số, chưa lần theo đường đi thật của dữ liệu).

Điều này **không** ảnh hưởng ba câu trả lời của `time_pool_ablation_20260922.md` — cả hai
chế độ chấm (cố định và thích ứng-từ-dev) đều cho cùng kết luận: `tpb2` không vỡ ít hơn,
thua thuần về F1. Chỉ khẳng định về NGUYÊN NHÂN của phép so là sai, không phải kết quả.

## 5. Việc KHÔNG làm

- Không đổi mặc định của `error_analysis.py`/`threshold_sweep.py` — chế độ cố định vẫn là
  mặc định khi không truyền cờ, đúng nguyên tắc "đổi mặc định làm run cũ không tái lập
  được". Muốn dùng chuẩn mới phải truyền tường minh `--adaptive-postproc --adaptive-source
  train`.
- Không xoá `analysis.json` (cửa sổ cố định) của bảy run — giữ làm chứng tích, chỉ thêm
  `analysis_adaptive_train.json` song song.
