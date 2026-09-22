# `pos_weight` là nguyên nhân của lệch hiệu chuẩn — xác nhận bằng ablation ba mức — 2026-09-22

Diễn giải cho [pos_weight_ablation_20260921.md](pos_weight_ablation_20260921.md), trang đó
do `scripts/chay_ablation_pos_weight.py` sinh tự động trong đêm 21→22/09 và **cố ý chỉ có
số**, kèm ba câu hỏi đặt ra TRƯỚC khi biết đáp án. Trang này trả lời đúng ba câu đó.

Nối tiếp [calibration_20260919.md](calibration_20260919.md), nơi ghi `pos_weight` là *"ứng
viên nguyên nhân, **chưa xác nhận**"*.

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với train → thiên vị theo thiết kế.
> Thiên vị này áp dụng **ngang nhau** cho cả bốn run nên không làm lệch phép so giữa chúng.
> θ\* của mọi run đều chọn trên chính tập đang chấm → **chặn trên lạc quan** cho mọi run.

---

## 1. Số

| run | pos_weight | ECE tổng | θ\* | event-F1 @θ\* | event-F1 @θ=0,5 |
|---|---:|---:|---:|---:|---:|
| `panns_ft_v3` | ~30 (hồi cứu) | 0,3332 | 0,90 | 0,3979 | 0,0820 |
| `panns_ft_pw30` | 30 | 0,3308 | 0,90 | 0,4018 | 0,0855 |
| `panns_ft_pw10` | 10 | 0,2033 | 0,80 | 0,4192 | 0,2340 |
| `panns_ft_pw1` | 1 | **0,0231** | **0,35** | **0,4333** | **0,4160** |

Mọi lượt `pw*` dùng **cùng cấu hình v3**, chỉ đổi `--pos-weight-max`; cùng split train, cùng
`time_pool_blocks=3`, hợp đồng dữ liệu `PASSED`, manifest ghi **lúc train** (không hồi cứu).

## 2. Ba câu trả lời

**(1) `pw30` tái hiện `v3` — gần như hoàn hảo.** ECE 0,3308 vs 0,3332 (lệch 0,7%), θ\* trùng
khít 0,90, event-F1 @θ\* lệch 0,0039, @0,5 lệch 0,0035. Giả thuyết **không** phải rút lại.

Kèm theo là một kết quả không nằm trong kế hoạch: `v3` train từ 18/09 với manifest lập **hồi
cứu**, nay được một run mới tái hiện trong vòng 1% trên cả bốn chỉ số. Đây là **bằng chứng
đầu tiên về khả năng tái lập của toàn pipeline** — thứ trước giờ chỉ được giả định.

**(2) Quan hệ đơn điệu tuyệt đối, không có ngưỡng lật.** pos_weight 30 → 10 → 1:

- ECE: 0,3308 → 0,2033 → 0,0231
- θ\*: 0,90 → 0,80 → 0,35
- event-F1 @θ\*: 0,4018 → 0,4192 → 0,4333
- event-F1 @0,5: 0,0855 → 0,2340 → 0,4160

Bốn cột, bốn lần đơn điệu cùng chiều. Quan hệ liều–đáp ứng trơn, không có chỗ gãy — nên
không cần quét thêm mức trung gian để tìm ngưỡng.

**(3) Không có đánh đổi ở bất kỳ mức nào.** Đây là điều đáng ngạc nhiên nhất. Giả thuyết ban
đầu coi `pos_weight` là một đánh đổi cổ điển: cân bằng lớp mua lại recall cho lớp hiếm bằng
giá hiệu chuẩn. Số liệu nói **không có gì được mua cả** — F1 tốt lên đơn điệu cùng chiều với
ECE. Trần 30 vừa làm hỏng hiệu chuẩn vừa làm hỏng F1.

## 3. Ý nghĩa thực tế nằm ở cột cuối

Cột `event-F1 @θ=0,5` là cột quan trọng nhất cho người dùng model, và nó chênh **gần 5 lần**:
0,0855 với `pos_weight=30` so với 0,4160 với `pos_weight=1`.

Ở 0,5 — ngưỡng mặc định mà bất kỳ ai cũng thử đầu tiên — model huấn luyện với trần 30 gần
như **vô dụng**. Nó chỉ dùng được nếu biết trước bí mật rằng phải cắt ở 0,90. Đó không phải
một model kém hơn một chút; đó là một model không dùng được nếu không đọc kỹ tài liệu.

Điều này cũng giải thích trọn vẹn chữ ký đã ghi ngày 19/09: *"vùng dự báo 0,4–0,6 mang hơn
nửa triệu khung mà tỉ lệ dương thật chỉ 1,7–3%"*. Trần `pos_weight` cao ép model đẩy điểm
dương lên bất kể có bằng chứng hay không, nên vùng giữa thang điểm đầy khung rác và ngưỡng
quyết định buộc phải chạy lên sát trần.

## 4. Điều kiện của phép so — nói cho đủ

`compare_runs` trên ba lượt `pw*` khai **`mã: khac`**, không phải `KHONG_RO`. Tức băm cây mã
**khác nhau** giữa ba run. Truy nguyên cụ thể:

| file | pw1 → pw30 |
|---|---|
| `ml/evaluation/error_analysis.py` | SỬA (thêm `nhom_cat_cheo` — hàm mới, không đụng hàm cũ) |
| `ml/evaluation/long_event_crosscut.py` | THÊM MỚI |
| `scripts/chay_ablation_pos_weight.py` | THÊM MỚI |
| **`ml/training/`, `ml/models/`** | **GIỐNG HỆT** |

Đường train byte-identical giữa ba lượt, và cả ba `analysis.json` do **cùng một phiên bản**
`error_analysis.py` sinh ra (mọi lượt đánh giá đều chạy sau khi các thay đổi trên đã xong).
Phép so hợp lệ — nhưng **chưa phải "sạch tuyệt đối"** theo nghĩa băm mã trùng nhau, và ghi
đúng như vậy thay vì khoe quá lời.

`KHONG_RO` chỉ còn xuất hiện khi kéo `v3` vào so, đúng như thiết kế — đó là run duy nhất còn
manifest hồi cứu. So sánh sạch nhất hiện có:
[compare_runs_pw_sach_20260922.md](compare_runs_pw_sach_20260922.md).

## 5. Quyết định kèm theo

**Không đổi hằng `MAX_POS_WEIGHT = 30.0` trong mã**, dù số liệu ủng hộ 1,0 ở mọi chiều.

Lý do: đổi mặc định làm mọi run cũ (v1/v2/v3) không còn tái lập được bằng lệnh mặc định —
ai chạy lại lệnh cũ sẽ ra kết quả khác mà không có gì cảnh báo. Đây đúng loại thay đổi im
lặng mà dự án đã phải trả giá vài lần. Thay vào đó:

- Cấu hình khuyến nghị cho v4 trở đi: **`--pos-weight-max 1.0`**, ghi tường minh trong lệnh.
- Hằng số giữ nguyên làm chứng tích của v1–v3.

## 6. Việc phép đo này KHÔNG làm

- **`--resume` vẫn chưa được kiểm bằng thực tế.** Runner qua đêm có watchdog khởi động lại
  bằng `--resume` nếu train chết, nhưng cả ba lượt chạy trót lọt nên nhánh đó **không lần
  nào được thực thi**. Món nợ kỹ thuật này còn nguyên.
- **Chưa đo trên dữ liệu ngoài `data/synthetic/dev`.** Mọi kết luận ở đây là trên dev tổng
  hợp thiên vị theo thiết kế. `gold_test` vẫn chưa dùng được (cổng §8.5 chưa đạt).
- **Chưa biết `pos_weight=1,0` có phải tối ưu không** — chỉ biết nó tốt hơn 10 và 30 ở mọi
  chiều. Quan hệ đơn điệu nên không có lý do nghi ngờ điểm gãy ở giữa, nhưng vùng dưới 1,0
  (tức phạt ngược lớp hiếm) chưa ai thử và cũng chưa có lý do để thử.
