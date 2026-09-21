# `low_snr` hỏng bằng cơ chế nào — và tại sao câu trả lời phụ thuộc vào độ dài sự kiện — 2026-09-21

Tiếp theo [long_event_crosscut_20260921.md](long_event_crosscut_20260921.md) §3, trang đó để
lại một quan sát chưa ai theo đuổi: `low_snr` kéo event-F1 xuống 0,08–0,11 ở **cả hai** nhánh
của bảng 2×2, nhưng gần như **không** làm tỉ lệ phân mảnh nhúc nhích (0,31→0,35 và
0,17→0,18). Nghĩa là `low_snr` hỏng bằng một cơ chế khác hẳn `long_event`, và cơ chế đó chưa
được đo.

Đo bằng `ml/evaluation/long_event_crosscut.py` (đã có sẵn bảng 6 loại lỗi cho từng ô — phần
này trước đây bị bỏ qua ở khâu báo cáo, không cần đo lại). Chạy trên **cả `panns_ft_v3` lẫn
`panns_ft_v2`** @θ\*=0,90, dev/all. Không train lại, không GPU (23 s mỗi run).

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với train của `v3` → thiên vị v3 theo
> thiết kế. θ\* chọn trên chính tập đang chấm → **chặn trên lạc quan**.

> ⚠️ **Không so với `v1`**: lưới onset 319,7 ms của nó làm mọi số liệu biên không so được.

---

## 1. Ba giả thuyết, đặt TRƯỚC khi nhìn số

- **H1 — Deletion**: SNR thấp đẩy điểm số xuống dưới θ\*, model **không thấy** sự kiện.
  Kiểm bằng: `thieu` tăng mạnh ở `+low_snr`, `thua` không tăng.
- **H2 — Substitution**: model thấy nhưng **gọi sai lớp**. Kiểm bằng: `thay_the` tăng, và
  cấu trúc ma trận nhầm ở `+low_snr` lệch hẳn so với `-low_snr`.
- **H3 — lệch biên**: đúng lớp nhưng onset/offset trôi quá collar 200 ms. Kiểm bằng: `bien`
  tăng còn `dung` giảm tương ứng.

Một hiệu chỉnh về cách đọc, lấy từ quyết định đã ghi ngày 19/09: **`bien` là phép CHIA NHỎ
LẠI rổ Deletion+Insertion của `sed_eval`, không phải loại lỗi thứ bảy song song.** Nghĩa là
H1 và H3 **không** phân biệt được bằng D/I của `sed_eval` — chúng chỉ phân biệt được trong
bảng phân loại 6 loại của `error_taxonomy.phan_loai()`, nơi `bien`/`thieu`/`thua` loại trừ
lẫn nhau trên cùng mẫu số `n_ref`. Toàn bộ trang này đọc theo bảng đó.

## 2. Số đo — hai run độc lập

Tỉ lệ trên `n_ref` của chính ô. `thua` vượt 1,0 được: một sự kiện thật hứng nhiều dự báo thừa.

**`panns_ft_v3`** (θ\*=0,90)

| ô | n_ref | event-F1 | đúng | biên | thay thế | thiếu | thừa |
|---|---:|---:|---:|---:|---:|---:|---:|
| `long_event+low_snr` | 163 | 0,1916 | 0,294 | **0,337** | 0,190 | 0,178 | 1,252 |
| `long_event-low_snr` | 396 | 0,3028 | 0,429 | 0,255 | 0,152 | 0,164 | 1,000 |
| `khac+low_snr` | 1286 | 0,3611 | 0,423 | 0,215 | 0,158 | **0,205** | 0,547 |
| `khac-low_snr` | 3028 | 0,4424 | 0,528 | 0,178 | 0,142 | 0,151 | 0,540 |

**`panns_ft_v2`** (θ\*=0,90)

| ô | n_ref | event-F1 | đúng | biên | thay thế | thiếu | thừa |
|---|---:|---:|---:|---:|---:|---:|---:|
| `long_event+low_snr` | 163 | 0,1694 | 0,252 | **0,331** | 0,215 | 0,202 | 1,172 |
| `long_event-low_snr` | 396 | 0,2679 | 0,396 | 0,227 | 0,182 | 0,194 | 1,154 |
| `khac+low_snr` | 1286 | 0,3223 | 0,371 | 0,201 | 0,159 | **0,268** | 0,570 |
| `khac-low_snr` | 3028 | 0,3891 | 0,471 | 0,162 | 0,159 | 0,208 | 0,626 |

Phần `dung` mất đi khi thêm `low_snr`, tách theo nơi nó chảy vào:

| nhánh | run | Δ đúng | Δ biên | Δ thay thế | Δ thiếu | Δ thừa |
|---|---|---:|---:|---:|---:|---:|
| `khac` | v3 | −0,105 | +0,037 | +0,016 | **+0,054** | +0,007 |
| `khac` | v2 | −0,100 | +0,039 | +0,000 | **+0,060** | −0,056 |
| `long_event` | v3 | −0,135 | **+0,082** | +0,038 | +0,014 | +0,252 |
| `long_event` | v2 | −0,144 | **+0,104** | +0,033 | +0,008 | +0,018 |

Cặp nhầm đậm nhất (v3, ngoài đường chéo, tối đa 5 cặp/ô):

| ô | cặp nhầm (thật → đoán) |
|---|---|
| `khac+low_snr` | fireworks→running_footsteps (10) · object_drop_dishes→glass_breaking (10) · laughter→applause_cheering (6) · laughter→shout_yell (6) · object_drop_dishes→door_slam (6) |
| `khac-low_snr` | object_drop_dishes→door_slam (21) · door_slam→running_footsteps (16) · gunshot→running_footsteps (15) · object_drop_dishes→glass_breaking (15) · fireworks→running_footsteps (14) |

## 3. Kết luận — H1 và H3 đều đúng, nhưng ở hai chế độ rời nhau

Khung "ba giả thuyết loại trừ nhau, chọn một" **đặt sai đề**. Số liệu nói rõ: cơ chế hỏng của
`low_snr` **phụ thuộc vào độ dài sự kiện**.

- **Trên sự kiện thường (`khac`) → H1 đúng.** `thieu` là thành phần lớn nhất, hứng 51% (v3)
  và 60% (v2) phần `dung` mất đi; `thua` **không** tăng (+0,007 ở v3, **giảm** 0,056 ở v2).
  Đúng hai vế của dự đoán H1: model đơn giản là không nghe ra sự kiện.
- **Trên sự kiện dài (`long_event`) → H3 đúng, H1 gần như không đóng góp.** `bien` hứng 61%
  (v3) và 72% (v2) phần `dung` mất đi, còn `thieu` chỉ nhúc nhích +0,014/+0,008. Model vẫn
  nghe ra sự kiện, vẫn gọi đúng lớp, nhưng **không chốt được biên** trong tiếng ồn.
- **H2 bị bác bỏ như một cơ chế chính, ở cả hai nhánh.** `thay_the` tăng nhiều nhất +0,038,
  luôn là thành phần nhỏ nhất hoặc gần nhỏ nhất, và ở v2 nhánh `khac` nó đứng yên tuyệt đối
  (+0,000). Quan trọng hơn: **cấu trúc** ma trận nhầm không đổi — các cặp đậm nhất ở
  `+low_snr` vẫn là đúng những họ cặp của `-low_snr` (`object_drop_dishes`→`glass_breaking`/
  `door_slam`, `fireworks`→`running_footsteps`), chỉ thưa hơn theo mẫu số. `low_snr`
  **không sinh cặp nhầm mới**, nên không có gì phải khai thêm vào `confusable_with`.

Hai run khác nhau cho cùng một mẫu hình ở cả bốn ô, với chênh lệch đồng dấu và cùng cỡ —
phát hiện **không đặc thù một run**.

Hệ quả thực tế, vì hai cơ chế cần hai cách sửa khác nhau:

- Phần `khac`/Deletion là bài toán **ngưỡng và dữ liệu**: hạ θ hoặc tăng tỉ lệ clip SNR thấp
  trong train. Sửa ở hậu xử lý không cứu được thứ model không hề phát ra điểm số.
- Phần `long_event`/biên là bài toán **hậu xử lý và độ phân giải thời gian**: sự kiện đã được
  phát hiện đúng lớp rồi. Đây cũng là chỗ nối với ứng viên chưa kiểm của `long_event` (trường
  tiếp nhận thời gian của CNN14) — cả hai đều nói về khả năng chốt biên trên đoạn dài.

Một điều **không** kết luận được: `thua` ở nhánh `long_event` tăng +0,252 ở v3 nhưng chỉ
+0,018 ở v2. Hai run không đồng thuận, nên không nói gì về nó.

## 4. Hiệu chỉnh một câu đã ghi hôm nay

`long_event_crosscut_20260921.md` §3 viết rằng `low_snr` "hỏng theo kiểu
Deletion/Substitution". Nửa sau **sai**: Substitution là thành phần nhỏ nhất ở mọi ô, và cấu
trúc nhầm không đổi. Câu đúng là **Deletion trên sự kiện thường, lệch biên trên sự kiện dài**.

## 5. Ứng viên còn lại

- Lát cắt `reverb` và `overlap` chưa được mổ theo cùng cách. `reverb` đặc biệt đáng nghi ở
  chiều `bien` (vọng kéo dài đuôi sự kiện), và phép đo thì đã có sẵn — chỉ cần đọc bảng cho
  hai lát cắt đó thay vì `low_snr`.
- Ngưỡng θ riêng theo lát cắt: nếu `khac+low_snr` hỏng vì Deletion, một θ thấp hơn cho clip
  SNR thấp có thể lấy lại phần đó. Nhưng chọn θ theo lát cắt của **chính tập đang chấm** là
  rò rỉ, nên chỉ làm được nếu suy θ từ một đại lượng đo được lúc suy luận (ví dụ SNR ước
  lượng), không phải từ nhãn lát cắt.
