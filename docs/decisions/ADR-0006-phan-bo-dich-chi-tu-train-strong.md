# ADR-0006 — Phân bố thời lượng đích chỉ lấy từ `train_strong`, giữ `eval_strong` cho gold_test

**Trạng thái:** Đã chốt · 2026-09-17
**Cổng:** W2 — thiết kế lại dữ liệu tổng hợp (STATUS §7, bước B3)

## Bối cảnh

Bước B3 bỏ `event_duration=("const", 1.0)` và thay bằng lấy mẫu lại thời lượng sự
kiện từ phân bố thực nghiệm của AudioSet Strong. Đây là thứ cho phép nói *"dữ liệu
huấn luyện khớp phân bố thời lượng của miền đích"* mà không phải bảo vệ thêm một giả
định phân phối nào.

Nhưng nguồn của phân bố đó có vấn đề. [`fetch_audioset_strong.py`](../../scripts/fetch_audioset_strong.py)
gộp cả hai split khi dựng `segments.jsonl`:

```python
rows = load_strong_rows(train_path) + load_strong_rows(eval_path)
```

Đối chiếu `ytid` với nhãn gốc: **937 segment = 825 từ `train_strong` + 112 từ
`eval_strong`** (3.482 + 530 sự kiện). `segments.jsonl` không mang trường nào phân
biệt hai phía.

`gold_test` trong `splits.csv` hiện vẫn rỗng — DATA_PLAN D8–D10 ghi đây là nút thắt,
phải do người gán mù, và §17 cấm máy đề xuất nhãn hay biên thời gian cho nó. Tức là
937 segment này chính là nguyên liệu sẽ được chia thành `dev` và `gold_test`.

## Vấn đề

Lấy phân bố thời lượng trên cả 937 clip để **thiết kế** dữ liệu huấn luyện nghĩa là
thống kê của những clip sẽ thành `gold_test` đã chảy ngược vào train.

Đây không phải rò rỉ nhãn theo nghĩa nặng — ta không chép nhãn hay audio nào sang
train, chỉ dùng thống kê gộp theo lớp. Nhưng nó đủ để hội đồng hỏi *"phân bố đích
anh lấy ở đâu ra"* mà ta không có câu trả lời sạch. Và vì `long_event` đã từng khai
đạt 791 clip trong khi thực tế sinh ra 0 sự kiện, dự án này không còn dư địa cho một
chỗ mờ nào nữa.

## Quyết định

`doc_phan_bo_dich` **chỉ đọc phía `train_strong`**. Phía `eval_strong` giữ nguyên,
không góp một con số nào vào thiết kế dữ liệu huấn luyện, làm hạt giống cho `gold_test`.

Áp dụng cho cả [`measure_distributions.py`](../../scripts/measure_distributions.py):
đối chiếu dữ liệu huấn luyện với tập sẽ dùng để chấm cũng là một dạng ngó trộm, chỉ
gián tiếp hơn.

Thiếu `audioset_eval_strong.tsv` thì **dừng hẳn**, không lặng lẽ gộp cả hai. Nếu
thiếu file mà vẫn chạy thì bản sửa này tự vô hiệu hoá đúng vào lúc không ai nhìn.

## Cái giá

Gần như bằng không. Đo trên dữ liệu thật:

| | toàn bộ 937 clip | chỉ 825 clip train |
|---|---:|---:|
| sự kiện làm phân bố đích | 4.012 | 3.482 |
| số lớp còn phủ | 15/15 | **15/15** |
| % sự kiện ≥ 2 s | 17.50 % | 17.63 % |
| % ≥ 4 s | 8.67 % | 8.93 % |
| % ≥ 8 s | 4.89 % | 5.14 % |
| **KS(trước vs sau)** | — | **0.0083** |

Không lớp nào mất hẳn, phân bố gần như không đổi. Ta được sạch về phương pháp mà
không mất gì.

## Hạn chế còn lại — phải ghi vào khoá luận

Phân bố ta khớp là **nhãn máy** của bản phát hành AudioSet Strong. Tập `gold_test`
cuối cùng sẽ là **nhãn người**, gán mù, theo `annotation_guideline.md`. Hai cái đó có
thể lệch nhau, và độ lệch đó **không** được đo bởi thiết kế hiện tại. Không giấu được
điều này: hội đồng hỏi một câu là lộ.

Ngoài ra ba lớp `siren`, `scream`, `explosion` vẫn không khớp nổi phân bố đích vì
giới hạn vật lý của bank foreground — xem STATUS §7 B3.

## Khoá bằng test

[`tests/test_scaper_duration.py`](../../tests/test_scaper_duration.py):

- `test_doc_phan_bo_dich_loai_clip_thuoc_eval_strong` — thời lượng phía eval không lọt vào
- `test_doc_phan_bo_dich_lop_chi_co_o_phia_eval_thi_bien_han` — không để lại lớp rỗng
- `test_doc_phan_bo_dich_thieu_nhan_eval_thi_CHET_chu_khong_gop_ca_hai` — thiếu nhãn thì dừng
- `test_ytid_eval_strong_boc_dung_ytid_khoi_segment_id` — ytid YouTube chứa cả `-` và `_`
