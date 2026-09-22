# Kiểm rò rỉ nguồn giữa các tập — 5 cổng đạt, và con số thật của "dev thiên vị theo thiết kế" — 2026-09-22

Đo bằng `scripts/check_leakage.py` (đã có từ trước với 3 kiểm tra; thêm kiểm tra 4–5 ngày
22/09). Không train lại, không GPU. Chia theo **nguồn**, không theo clip — hai đoạn cắt từ
cùng một video YouTube có `file_id` khác nhau nhưng là cùng một nguồn âm thanh.

---

## 1. Năm cổng — ĐẠT hết

| # | Kiểm tra | Soi cái gì | Kết quả |
|---|---|---|---|
| 1 | Không `source_group_id` nào nằm ở hai tập | `splits.csv` | ✅ 0/2.215 nhóm |
| 2 | Không audio nào (SHA-256) nằm ở hai tập | `raw_manifest.csv` | ✅ |
| 3 | Không nhóm trùng lặp nào bắc cầu giữa hai tập | `dedup_groups.csv` | ✅ |
| 4 | Nguồn foreground dùng THẬT trong synthetic train đều thuộc `foreground_bank_train` | **JAMS đã sinh** | ✅ 7.920 clip · 3.680 nguồn |
| 4 | Nguồn foreground dùng THẬT trong synthetic dev đều thuộc `foreground_bank_train` | **JAMS đã sinh** | ✅ 1.440 clip · 2.012 nguồn |

Kiểm tra 1–3 đã có từ trước và soi `splits.csv` — tức soi **ý định** chia tập. Kiểm tra 4
thêm hôm nay soi **đầu ra thật**: đọc cả 9.360 file JAMS rồi đối chiếu từng `source_file`
đã thực sự được Scaper dùng với bank được phép.

Vì sao cần cả hai: bộ lọc đầu vào của `scaper_generate.py` tự khai là đủ, nhưng dự án đã
từng bị `run_manifest.doc_hop_dong` khai `PASSED` trong khi thiếu file báo cáo. Một cổng
chặn tự khai báo mà hỏng thì tệ hơn không có cổng — nó phát giấy chứng nhận sạch. Kết quả:
**không có clip `gold_test` hay `dev` nào lọt vào nguyên liệu synthetic**.

## 2. Số phải khai báo — mức dùng chung nguồn giữa synthetic train và dev

Đây **không phải cổng và không phải lỗi**: `scaper_generate.py` cố ý sinh cả train lẫn dev
từ cùng một `foreground_bank_train` (bank chia ra chỉ để bảo vệ `gold_test`). Nhưng đây là
lần đầu mức trùng được đo thành số thay vì mô tả bằng câu "cùng recipe B0–B9".

| | train dùng | dev dùng | chung | tỉ lệ nguồn của dev cũng có trong train |
|---|---:|---:|---:|---:|
| foreground | 3.680 | 2.012 | 1.898 | **0,9433** |
| background | 1.105 | 434 | 364 | **0,8387** |

**Mức clip: 1.284/1.440 clip dev (0,8917) có ít nhất một nguồn foreground từng dùng trong
train.**

## 3. Nghĩa là gì — và KHÔNG có nghĩa là gì

Con số 0,8917 nói chính xác điều này: `data/synthetic/dev` đo được khả năng khái quát sang
**tổ hợp mới của những nguồn đã nghe** (SNR khác, vọng khác, chồng lấn khác, nền khác),
**không** đo được khả năng khái quát sang **nguồn chưa từng nghe**. Hai thứ đó khác nhau,
và mọi số hiện có của dự án — v1/v2/v3, ablation `pos_weight`, phân tích `long_event` và
`low_snr` — đều thuộc loại thứ nhất.

Điều này **không** làm các phép so giữa các run mất hiệu lực: thiên vị áp dụng ngang nhau
cho mọi run chấm trên cùng tập dev, nên so `pw1` với `pw30` vẫn hợp lệ. Nó chỉ chặn đúng
một loại phát biểu: **không được đọc event-F1 trên dev như ước lượng hiệu năng thực địa.**
Muốn con số đó phải có real dev / `gold_test`, mà `gold_test` thì vẫn chưa qua cổng §8.5.

Đây là số cần đưa thẳng vào phần **Hạn chế** của khoá luận, cạnh tỉ lệ loại file 34,8% của
pilot gold.

## 4. Một tập con sạch chưa được khai thác

2.012 − 1.898 = **114 nguồn foreground chỉ xuất hiện ở dev**, và
1.440 − 1.284 = **156 clip dev mà toàn bộ nguồn foreground đều chưa từng vào train**.

156 clip là ít và chắc chắn nhiễu, nhưng chúng là proxy *duy nhất hiện có* cho câu hỏi
"model làm gì với nguồn chưa từng nghe". Chấm lại trên đúng tập con đó là phép đo **rẻ**
(dự đoán đã lưu, chỉ là lọc clip rồi tính lại) và trả lời một câu hỏi chưa ai trả lời.
Chưa làm.

## 5. Việc phép đo này KHÔNG làm

- **Không kiểm `data/gold/`** — gold_test chưa có nhãn nên chưa có gì để rò rỉ vào đâu.
- **Không kiểm rò rỉ ở mức nội dung** (hai bản thu khác nhau của cùng một sự kiện thật).
  SHA-256 chỉ bắt được file giống hệt bit-for-bit.
- **Không chạy trong CI** — dự án vẫn chưa có CI. Cổng này hiện chỉ chạy khi có người gõ lệnh.
