# `time_pool_blocks` không giải thích được `long_event` — ứng viên thứ sáu bị loại — 2026-09-22

Diễn giải cho [time_pool_ablation_20260922.md](time_pool_ablation_20260922.md), trang đó
do `scripts/chay_ablation_time_pool.py` sinh tự động và **cố ý chỉ có số**, kèm ba câu hỏi
đặt ra TRƯỚC khi biết đáp án. Trang này trả lời đúng ba câu đó, rồi đo lại ở chế độ hậu xử
lý chuẩn mới ([chuan_bao_cao_20260922.md](chuan_bao_cao_20260922.md)) để loại một nghi vấn
nhiễu trước khi kết luận.

> Sửa một chỗ đếm trong file tự sinh: trang đó ghi "ứng viên thứ NĂM", đúng ra là **thứ
> sáu** — bốn giả thuyết đã liệt kê ở `long_event_crosscut_20260921.md` §4, cộng giả
> thuyết cơ chế Deletion/lệch biên đã tinh chỉnh ở `low_snr_co_che_20260921.md` là giả
> thuyết thứ năm. Không sửa file tự sinh (giữ nguyên làm chứng tích).

---

## 1. Số — cả hai chế độ hậu xử lý

| | cửa sổ cố định (7 khung) | | thích ứng-từ-train | |
|---|---:|---:|---:|---:|
| | `pw1` (tpb=3) | `tpb2` (tpb=2) | `pw1` (tpb=3) | `tpb2` (tpb=2) |
| θ\* | 0,35 | 0,30 | 0,25 | 0,25 |
| F1@θ\* | **0,4333** | 0,3466 | **0,4682** | 0,4113 |
| `long_event` tỉ lệ vỡ | 0,2987 | 0,3918 | 0,2469 | 0,2540 |
| `long_event` F1 | 0,3182 | 0,2422 | 0,3602 | 0,3066 |

`tpb2` train với `--pos-weight-max 1.0` khoá cứng, mọi tham số khác giống `pw1`, chỉ đổi
`--time-pool-blocks`. `compare_runs` khai cùng cây mã, hợp đồng dữ liệu `PASSED`, manifest
ghi lúc train — phép so sạch nhất dự án từng có cho một ablation.

## 2. Ba câu trả lời

**(1) Tỉ lệ vỡ có thấp hơn không? KHÔNG, ở cả hai chế độ.** Cửa sổ cố định: tệ hơn rõ
(0,2987→0,3918, +31% tương đối). Cửa sổ thích ứng: **ngang nhau** (0,2469 vs 0,2540, chênh
2,9%). Không có chế độ hậu xử lý nào mà `tpb2` vỡ ít hơn `pw1`. Giả thuyết *"trường tiếp
nhận thời gian ngắn là một phần nguyên nhân của `long_event`"* **bị bác bỏ**.

**(2) Đánh đổi hay thắng thuần? Thua thuần**, ở cả hai chế độ: F1@θ\* kém 0,087 (cố định)
và 0,057 (thích ứng); F1 `long_event` kém 0,076 và 0,054. Không có góc nào `tpb2` thắng.
**Quyết định cấu hình: giữ `time_pool_blocks=3`.**

**(3) Câu hỏi phụ đăng ký trước hoá ra là phát hiện chính.** Ở chế độ thích ứng, tỉ lệ vỡ
gần như bằng nhau (0,2469 vs 0,2540) nhưng F1 `long_event` vẫn chênh 0,054 (~15% tương
đối). Nếu phân mảnh là thứ giới hạn F1 trên sự kiện dài, hai F1 phải gần nhau khi hai tỉ
lệ vỡ gần nhau — chúng không gần nhau. **Tỉ lệ phân mảnh không nắm được phần chính của vấn
đề `long_event`.** Diễn giải đầy đủ và quyết định dừng điều tra ở
[long_event_dong_huong_20260922.md](long_event_dong_huong_20260922.md).

## 3. Một khẳng định sai đã rút lại

Lúc mới có số ở cửa sổ cố định, đã có một khẳng định rằng phép so bị nhiễu vì cửa sổ lọc
tương ứng 560 ms ở `pw1` nhưng chỉ 280 ms ở `tpb2` (suy từ `7 khung ÷ fps của đoạn`). Kiểm
lại `predictions/dev_all.npz` cho thấy lọc trung vị chạy SAU khi nội suy đoạn lên
`n_frames=1001` khung cho **cả hai** run, nên `7` khung là ~70 ms ở cả hai — phép so ở cửa
sổ cố định vốn đã công bằng, không có nhiễu. Đã rút lại ngay khi phát hiện. Điều này KHÔNG
đổi ba câu trả lời ở §2 — cả hai chế độ hậu xử lý đều đồng thuận về kết luận (không vỡ ít
hơn, thua thuần về F1); chỉ khẳng định về NGUYÊN NHÂN của một nghi vấn là sai.

## 4. Quyết định

1. **`--time-pool-blocks` giữ mặc định 3** cho mọi run tiếp theo.
2. **Đóng hướng điều tra `long_event`** — sáu giả thuyết đã kiểm và loại, chi phí GPU/công
   phân tích tiếp theo không cân xứng với giá trị so với việc chuyển sang W4/W5 (đóng góp
   chính, chưa bắt đầu). Chi tiết ở `long_event_dong_huong_20260922.md`.
3. **Chuyển mốc báo cáo sang hậu xử lý thích ứng-từ-train** — quyết định độc lập, có lợi
   ích riêng (không rò rỉ, nhất quán theo thời gian thật, cải thiện đo được), không phụ
   thuộc kết quả ablation này. Chi tiết ở `chuan_bao_cao_20260922.md`.
