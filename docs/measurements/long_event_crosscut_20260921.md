# `long_event`: ứng viên thứ tư cũng bị bác bỏ — độ dài là biến thật, không phải lát cắt trùng — 2026-09-21

Tiếp theo [long_event_gap_20260921.md](long_event_gap_20260921.md), trang đó kết thúc ở câu
*"`long_event` đã loại BA ứng viên liên tiếp; nguyên nhân riêng của nó vẫn chưa xác định"*.

Trang này kiểm một **giả thuyết khác về bản chất**: có thể `long_event` không hề xấu vì độ
dài — mà vì các clip dài **tình cờ trùng** với lát cắt khó khác. Giả thuyết này đáng kiểm
vì cả ba phép đo trước đều đứng trên cùng một cái hàng lẫn lộn của bảng lát cắt: trên
`data/synthetic/dev`, trong 144 clip `long_event` có **52** cũng là `reverb`, **47** cũng là
`overlap`, **45** cũng là `low_snr`. Hàng `long_event` một mình không tách được hai khả năng đó.

Đo bằng `ml/evaluation/long_event_crosscut.py` trên `panns_ft_v3` @θ\*=0,90, dev/all.
Không train lại, không GPU (16 s).

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với train của `v3` → thiên vị v3 theo
> thiết kế. θ\* lấy từ `analysis.json` của chính run, tức chọn trên tập đang chấm →
> **chặn trên lạc quan**.

> ⚠️ **Cỡ mẫu nhỏ**: mỗi ô giao chỉ 45–52 clip (`causal_chain` chỉ 15). Đủ để nói có/không
> có tương quan rõ, **không** đủ để nói độ lớn hiệu ứng.

---

## 1. Phương pháp — bảng 2×2, không phải một hàng

`nhom_cat_cheo()` chia toàn bộ 1.440 clip dev thành bốn ô cho mỗi lát cắt đối chiếu X:

|  | có X | không X |
|---|---|---|
| **là `long_event`** | `long_event+X` | `long_event-X` |
| **không phải** | `khac+X` | `khac-X` |

Chỉ tiêu đo là **tỉ lệ sự kiện tham chiếu bị phân mảnh** (bị ≥2 dự báo cùng lớp chồng lấn —
cùng định nghĩa với `danh_dau_phan_manh()` của phép đo trước), vì phân mảnh chính là kiểu
hỏng đặc trưng của `long_event`.

Logic quyết định đặt trước khi nhìn số:

- Nếu chênh lệch `long_event` − `khac` **tan đi ở cột "không X"** → độ dài không phải biến
  giải thích, thủ phạm là X. Giả thuyết độ dài bị bác bỏ.
- Nếu chênh lệch **còn nguyên ở cả hai cột** → độ dài là biến độc lập thật, và giả thuyết
  "trùng lát cắt" bị bác bỏ.

## 2. Số đo

| đối chiếu X | ô | clip | sự kiện | vỡ | tỉ lệ vỡ |
|---|---|---:|---:|---:|---:|
| low_snr | `long_event+low_snr` | 45 | 163 | 57 | **0,35** |
| low_snr | `long_event-low_snr` | 99 | 396 | 121 | **0,31** |
| low_snr | `khac+low_snr` | 345 | 1286 | 237 | 0,18 |
| low_snr | `khac-low_snr` | 951 | 3028 | 512 | 0,17 |
| reverb | `long_event+reverb` | 52 | 203 | 59 | **0,29** |
| reverb | `long_event-reverb` | 92 | 356 | 119 | **0,33** |
| reverb | `khac+reverb` | 505 | 1685 | 300 | 0,18 |
| reverb | `khac-reverb` | 791 | 2629 | 449 | 0,17 |
| overlap | `long_event+overlap` | 47 | 182 | 53 | **0,29** |
| overlap | `long_event-overlap` | 97 | 377 | 125 | **0,33** |
| overlap | `khac+overlap` | 390 | 1541 | 277 | 0,18 |
| overlap | `khac-overlap` | 906 | 2773 | 472 | 0,17 |
| causal_chain | `long_event+causal_chain` | 15 | 68 | 23 | **0,34** |
| causal_chain | `long_event-causal_chain` | 129 | 491 | 155 | **0,32** |
| causal_chain | `khac+causal_chain` | 188 | 777 | 104 | 0,13 |
| causal_chain | `khac-causal_chain` | 1108 | 3537 | 645 | 0,18 |

Chênh lệch `long_event` − `khac`, đo riêng trong từng cột:

| đối chiếu X | chênh lệch ở cột *có X* | chênh lệch ở cột *không X* |
|---|---:|---:|
| low_snr | 0,17 | 0,14 |
| reverb | 0,11 | 0,16 |
| overlap | 0,11 | 0,16 |
| causal_chain | 0,20 | 0,13 |

## 3. Kết luận — giả thuyết "trùng lát cắt" bị **bác bỏ**

Chênh lệch **không tan đi ở cột nào**: ở cả tám cột, `long_event` vỡ nhiều hơn `khac`
0,11–0,20 tuyệt đối. Tỉ lệ vỡ của `long_event` đứng ở **0,29–0,35** bất kể có `low_snr`,
`reverb`, `overlap` hay không; của `khac` đứng ở **0,13–0,18**, cũng gần như bất kể.

Đọc ngược lại còn mạnh hơn: `long_event` **sạch** các lát cắt khác vẫn vỡ nhiều gần y hệt
`long_event` có chúng (`-reverb` 0,33 vs `+reverb` 0,29; `-overlap` 0,33 vs `+overlap` 0,29
— hai trường hợp này thậm chí nhích **ngược chiều**, tức thêm vọng/chồng lấn không làm sự
kiện dài vỡ thêm chút nào).

Nghĩa là: **độ dài sự kiện là một biến khó độc lập**, không phải hệ quả của lát cắt trùng.
Điều này đóng một hướng điều tra và cũng là một kết quả dương tính yếu — nó xác nhận lát
cắt `long_event` đo đúng thứ nó định đo, chứ không phải một cái tên gắn nhầm lên tập con
"clip khó nói chung".

Phụ chú về F1 sự kiện (cùng chiều, không mâu thuẫn): `long_event+low_snr` 0,1916 ·
`long_event-low_snr` 0,3028 · `khac+low_snr` 0,3611 · `khac-low_snr` 0,4424. `low_snr` là
lát cắt duy nhất làm F1 của `long_event` sụt rõ thêm — nhưng nó không làm **tỉ lệ vỡ** tăng
tương ứng (0,35 vs 0,31), nên cái nó làm hỏng là chuyện khác (Deletion/Substitution), không
phải phân mảnh.

## 4. Ứng viên còn lại

Bốn ứng viên đã loại: cửa sổ lọc hẹp · trần cửa sổ thấp · khe hở năng lượng thật · trùng
lát cắt khó. Nguyên nhân **vẫn chưa xác định**, nhưng phạm vi đã hẹp lại đáng kể — thủ phạm
phải là thứ gắn với chính độ dài sự kiện, không phải hậu xử lý và không phải điều kiện âm
học đi kèm.

Hướng chưa kiểm, xếp theo mức khả thi:

1. **Nghe trực tiếp** các clip `long_event` vỡ nhiều nhất — quan sát định tính, cần tai
   người, chưa làm.
2. **Trường tiếp nhận thời gian của backbone**: CNN14 gộp thời gian theo khối; một sự kiện
   4+ giây dài hơn hẳn trường tiếp nhận hiệu dụng, model có thể không có cách nào "biết"
   rằng khung giây thứ 1 và khung giây thứ 4 thuộc cùng một sự kiện. Đây là giả thuyết về
   kiến trúc, kiểm được bằng ablation `--time-pool-blocks` nhưng tốn GPU.
3. **Hàm mất mát theo khung không phạt phân mảnh**: BCE theo khung cho điểm từng khung độc
   lập, một dự báo vỡ đôi có loss gần như y hệt một dự báo liền. Nếu đúng thì không phép đo
   nào trên dự đoán đã lưu phát hiện được — phải đổi loss rồi train lại.
