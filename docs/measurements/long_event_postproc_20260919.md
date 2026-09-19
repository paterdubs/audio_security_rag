# `long_event` hỏng theo kiểu nào, và cửa sổ lọc giải thích được bao nhiêu — 2026-09-19

Tiếp theo [error_analysis_20260919.md](error_analysis_20260919.md) §3, trang đó kết thúc ở
câu *"chưa xác định nguyên nhân của `long_event`"*. Trang này trả lời bằng hai phép đo:
phân loại lỗi tách riêng **theo từng lát cắt**, và một ablation cửa sổ lọc trung vị.

Sinh bởi `ml/evaluation/error_analysis.py` — mặc định ghi `ml/runs/{run}/analysis.{json,md}`,
thêm `--adaptive-postproc` thì ghi `analysis_adaptive.{json,md}`. Không train lại, không GPU.

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với train của `v3` → dev **thiên vị v3
> theo thiết kế**. Mọi so sánh v1/v2/v3 dưới đây mang thiên vị đó.
>
> ⚠️ θ\* chọn trên chính tập đang chấm, và cửa sổ thích ứng suy từ độ dài sự kiện của
> **chính tập đang chấm** (xem cảnh báo ở `threshold_sweep.cua_so_loc`) → **rò rỉ hai
> tầng**. Đây là chặn trên lạc quan, không phải điểm báo cáo được. `gold_test` vẫn rỗng.

---

## 1. `long_event` hỏng theo kiểu nào

Phân loại lỗi **đếm lại trên từng tập con clip**, không chia tỉ lệ từ tổng — lát cắt chồng
nhau nên cộng cột sẽ lớn hơn tổng toàn tập. Chuẩn hoá theo số sự kiện thật của chính lát
cắt đó để so ngang được giữa các hàng. `v3` ở θ\* = 0,90, cửa sổ cố định 7 khung:

| lát cắt | clip | sự kiện thật | F1(sự kiện) | phân mảnh/thật | Insertion/thật | Deletion/thật |
|---|---|---|---|---|---|---|
| sach (mốc) | 357 | 967 | 0,4487 | 0,168 | 0,594 | 0,114 |
| causal_chain | 203 | 845 | 0,4417 | 0,150 | 0,457 | 0,195 |
| overlap | 437 | 1.723 | 0,3799 | 0,192 | 0,573 | 0,193 |
| reverb | 557 | 1.888 | 0,3738 | 0,190 | 0,590 | 0,204 |
| low_snr | 390 | 1.449 | 0,3369 | 0,203 | 0,627 | 0,202 |
| **long_event** | 144 | 559 | **0,2685** | **0,318** | **1,073** | 0,168 |

Chữ ký của `long_event` khác hẳn năm lát cắt còn lại, và nó **không phải** "không nghe ra":

- **Phân mảnh gấp 1,9 lần mốc** (0,318 so với 0,168). Gần một phần ba sự kiện dài bị cắt
  thành nhiều mảnh.
- **Insertion gấp 1,8 lần mốc** (1,073 so với 0,594) — và đây là **hệ quả cơ học của phân
  mảnh**: một sự kiện thật vỡ thành *k* mảnh thì một mảnh được ghép cặp, *k−1* mảnh còn
  lại thành Insertion. Không phải model nghe ra thêm sự kiện không có.
- **Deletion thì KHÔNG cao bất thường** (0,168 so với 0,114 của mốc, thấp hơn cả
  `low_snr` 0,202). Sự kiện dài ít khi bị bỏ sót hoàn toàn.

Cùng chữ ký đó ở cả ba run (v1: phân mảnh 0,154 so với mốc 0,072; v2: 0,297 so với 0,178),
nên nó là tính chất của **dữ liệu và bộ giải mã**, không phải của một lần chạy.

---

## 2. Ablation cửa sổ lọc — giả thuyết đúng một nửa

Giả thuyết đang ghi trong `error_analysis_20260919.md` §3: cửa sổ lọc trung vị cố định 7
khung (~70 ms) quá hẹp để lấp khe hở giữa chừng của một sự kiện dài. Ablation: `cua_so_loc`
với cửa sổ **riêng từng lớp**, suy từ độ dài sự kiện của lớp đó — thực tế rơi vào 17–51
khung, tức rộng gấp 2,4 đến 7,3 lần.

| | v1 @θ\* | v2 @θ\* | v3 @θ\* |
|---|---|---|---|
| cửa sổ 7 khung | 0,2953 (θ 0,95) | 0,3518 (θ 0,90) | 0,3979 (θ 0,90) |
| cửa sổ theo lớp | 0,2953 (θ 0,95) | **0,3911** (θ 0,85) | **0,4347** (θ 0,85) |

Trên `v3`, theo lát cắt:

| lát cắt | F1 (7 khung) | F1 (theo lớp) | phân mảnh/thật (7 → theo lớp) |
|---|---|---|---|
| sach (mốc) | 0,4487 | 0,4846 | 0,168 → 0,111 |
| causal_chain | 0,4417 | 0,4798 | 0,150 → 0,098 |
| overlap | 0,3799 | 0,4135 | 0,192 → 0,140 |
| reverb | 0,3738 | 0,4112 | 0,190 → 0,133 |
| low_snr | 0,3369 | 0,3694 | 0,203 → 0,147 |
| **long_event** | 0,2685 | **0,3067** | 0,318 → **0,263** |

**Phần đúng của giả thuyết:** cửa sổ rộng hơn có tác dụng thật. `long_event` lên +0,038 F1,
phân mảnh giảm từ 0,318 xuống 0,263.

**Phần sai — và đây mới là điều cần ghi:** nó **không giải thích được vì sao `long_event`
tệ hơn các lát cắt khác.** Khoảng cách tới mốc `sach` gần như không nhúc nhích:

| | 7 khung | theo lớp |
|---|---|---|
| `sach` − `long_event` | 0,4487 − 0,2685 = **0,180** | 0,4846 − 0,3067 = **0,178** |

Cửa sổ rộng hơn nâng **mọi** lát cắt lên xấp xỉ cùng một lượng (+0,033 đến +0,038). Đó là
một khiếm khuyết **toàn cục** của khâu hậu xử lý, không phải nguyên nhân riêng của
`long_event`. Câu "cửa sổ 7 khung là thủ phạm của `long_event`" bị phép đo này bác bỏ; câu
đúng là "cửa sổ 7 khung quá hẹp cho *mọi* lớp, còn `long_event` tệ vì một lý do khác chưa
tìm ra".

Ứng viên còn lại, chưa kiểm: cửa sổ tối đa bị chặn ở 51 khung (~0,5 s) trong khi sự kiện
của lát cắt này dài nhiều giây, nên ngay cả cửa sổ rộng nhất vẫn không đủ; và cách sinh
`long_event` của Scaper có thể tạo khe hở năng lượng thật bên trong sự kiện.

---

## 3. Hậu xử lý không chạm được vào v1

`v1` ra **đúng từng con số** ở cả hai cấu hình. Không phải lỗi cài đặt: xác suất mức đoạn
của v1 là **hằng số trên từng mảng ~32 khung** (31 đoạn giãn ra 1.001 khung), nên bộ lọc
trung vị với cửa sổ ≤ 51 khung không đổi được gì — đo trực tiếp trên mảng xác suất,
`max|lọc_7 − lọc_theo_lớp| = 0,0` tuyệt đối, so với 0,160 ở v3.

Đây là hệ quả thứ hai của lưới onset 319,7 ms đã nêu ở `error_analysis_20260919.md` §2:
v1 vừa có trần cứng 37,4% lỗi biên, vừa **miễn nhiễm với mọi tinh chỉnh hậu xử lý**. Mọi
cải tiến ở khâu giải mã đều vô nghĩa với v1 mà không vô nghĩa với v2/v3 — thêm một lý do
nữa để không xếp chung ba run vào một bảng như thể chúng so được với nhau.

---

## 4. Điều trang này KHÔNG chứng minh

- Cửa sổ theo lớp **không phải cải tiến báo cáo được**. Nó suy từ nhãn của chính tập đang
  chấm, và θ\* cũng chọn trên tập đó — rò rỉ hai tầng. Muốn dùng thật thì cửa sổ phải suy
  từ **tập train** (`median_sizes_for` lúc train đã làm đúng như vậy) rồi đo lại trên tập
  độc lập.
- **Nguyên nhân `long_event` vẫn chưa xác định.** Đã loại được một ứng viên, chưa tìm ra
  thủ phạm.
- `gold_test` vẫn rỗng. Toàn bộ trang này đo trên tập tổng hợp dùng chung foreground bank
  với train.
