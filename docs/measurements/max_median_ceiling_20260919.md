# Nới trần MAX_MEDIAN_FRAMES — trần 51 khung có phải nguyên nhân của `long_event`? — 2026-09-19

Tiếp theo [long_event_postproc_20260919.md](long_event_postproc_20260919.md) §2, trang đó
để lại đúng một ứng viên chưa kiểm: *"cửa sổ tối đa bị chặn ở 51 khung (~0,5 s) trong khi
sự kiện của lát cắt này dài nhiều giây, nên ngay cả cửa sổ rộng nhất vẫn không đủ"*. Trang
này quét `MAX_MEDIAN_FRAMES ∈ {51, 101, 201, 401}` (qua `--max-median-frames`, xem
`ten_bao_cao()`) trên `v2` và `v3` (bỏ `v1` — miễn nhiễm với hậu xử lý) và đọc trực tiếp
cột **gộp** mà `sed_metrics.py:37` dùng làm lý do đặt trần.

> ⚠️ Cảnh báo rò rỉ và thiên vị dev giống hệt hai trang trước — không lặp lại ở đây, xem
> [error_analysis_20260919.md](error_analysis_20260919.md) đầu trang.

---

## 1. Nới trần KHÔNG đóng được khoảng cách `sach − long_event`

| trần | v2 F1(sự kiện) | v2 gap | v3 F1(sự kiện) | v3 gap |
|---|---|---|---|---|
| 7 khung (mốc) | 0,3518 | — | 0,3979 | 0,180 (mốc, xem trang trước) |
| 51 (mặc định) | 0,3911 | 0,1322 | 0,4347 | 0,1779 |
| 101 | 0,3925 | **0,1463** | 0,4371 | 0,1712 |
| 201 | 0,3918 | 0,1396 | 0,4368 | 0,1706 |
| 401 | 0,3858 | **0,1460** | 0,4330 | 0,1684 |

*gap = F1(`sach`) − F1(`long_event`) trong CHÍNH cấu hình đó — không so với mốc 7 khung.*

Hai run cho hai câu trả lời khác nhau:

- **`v2`: nới trần làm khoảng cách RỘNG RA**, từ 0,1322 lên 0,1396–0,1463. Giả thuyết
  "trần chặn `long_event`" bị **bác bỏ thẳng** ở run này.
- **`v3`: khoảng cách hẹp lại một chút**, 0,1779 → 0,1684 (giảm 0,0095, ~5%). Nhưng so
  với mốc 7-khung ban đầu (0,180 ở trang trước), tổng mức giảm sau CẢ HAI lượt nới (cửa sổ
  theo lớp rồi nới trần tới 401) chỉ là 0,180 → 0,1684 = **6,4%** — phần lớn khoảng cách
  vẫn còn nguyên.

**Không có run nào ủng hộ "trần 51 khung là nguyên nhân chính của `long_event`."**

## 2. Vì sao F1 không cải thiện dù Insertion giảm rõ — đánh đổi, không phải thắng thuần

Chi tiết lỗi của riêng lát cắt `long_event`, run `v3` (n_ref = 559 sự kiện thật, không đổi
qua các trần):

| trần | Đúng | Insertion | Deletion | gộp |
|---|---|---|---|---|
| 51 | 236 | 504 | 83 | 17 |
| 101 | 233 | 464 | 90 | 17 |
| 201 | 229 | 444 | 95 | 17 |
| 401 | 228 | 442 | 96 | 17 |

Insertion giảm thật (504 → 442, **−12,3%**) — cửa sổ rộng hơn đúng là lấp được một phần
khe hở giữa các mảnh. Nhưng **Đúng cũng giảm theo** (236 → 228) và **Deletion tăng** (83 →
96): cửa sổ đủ rộng để nối hai mảnh của MỘT sự kiện thật, nhưng cũng đủ rộng để nối nhầm
qua ranh giới, biến một cặp khớp đúng thành khớp lệch hoặc mất hẳn. Đây là đánh đổi, không
phải cải tiến một chiều — giải thích tại sao Insertion giảm 12% mà F1 gần như đứng yên.

`v2` cho đúng chữ ký này (Đúng 225→203, Insertion 546→479, Deletion 102→123) nhưng đánh
đổi lệch về phía bất lợi nhiều hơn `v3`, nên gap ở `v2` rộng ra thay vì hẹp lại.

## 3. Cột **gộp** — lý do đặt trần trong code chưa có số đỡ, đo xong vẫn vậy

`sed_metrics.py:37` ghi lý do đặt trần 51: *"rộng hơn nữa bắt đầu gộp hai sự kiện rời
thành một"*. Đo trực tiếp: **gộp đứng yên tuyệt đối** qua cả bốn trần — `v2` luôn = 20,
`v3` luôn = 17, kể cả ở trần 401 (~4 s). Lời cảnh báo đó vẫn **chưa có số đỡ** sau lượt đo
này; nếu đúng, hiện tượng gộp phải xảy ra ở một trần cao hơn 401, hoặc lý do đặt trần 51
không phải vì gộp.

Ghi chú kỹ thuật: ở cấu hình `--max-median-frames 401`, cửa sổ LỚN NHẤT thực sự được dùng
chỉ là 389 khung (lớp có độ dài sự kiện phân vị-25 dài nhất trong dev chưa tới 3,89 s) —
trần 401 không bị chính nó chặn lại, nên phép đo này đã quan sát được hành vi "không giới
hạn" trong phạm vi dữ liệu hiện có, không phải bị cắt ngang giữa chừng.

---

## 4. Kết luận — giả thuyết trần bị bác bỏ phần lớn; nguyên nhân `long_event` vẫn mở

- **Trần 51 khung không phải nguyên nhân chính** của khoảng cách `sach − long_event`. `v2`
  cho kết quả ngược hướng giả thuyết; `v3` chỉ đóng được 6,4% khoảng cách dù nới ceiling
  tới gần 4 giây — phần nới cửa sổ theo lớp (trang trước) và phần nới trần (trang này)
  cộng lại vẫn để lại phần lớn của con số 0,180 ban đầu chưa giải thích được.
- Ứng viên còn lại từ `long_event_postproc_20260919.md` §2: **Scaper có thể tạo khe hở
  năng lượng THẬT bên trong sự kiện dài** — chưa kiểm. Đây là hướng hợp lý nhất còn sót
  lại sau khi loại cả cửa sổ hẹp lẫn trần thấp.
- Cột **gộp** không đổi qua bốn trần — cảnh báo trong code về nguy cơ gộp nhầm chưa được
  xác nhận bằng số đo, ở phạm vi trần đã quét (≤ 401 khung).

## 5. Điều trang này KHÔNG chứng minh

- θ\* và cửa sổ theo lớp vẫn chọn trên chính tập đang chấm — hai tầng rò rỉ đã nêu ở
  `error_analysis_20260919.md` không đổi vì ablation này.
- Không quét trần > 401 khung (~4 s) — nếu có lớp nào có sự kiện dài hơn thế trong dữ liệu
  thật (không phải synthetic), trần cao hơn có thể cho kết quả khác.
- `gold_test` vẫn rỗng.
