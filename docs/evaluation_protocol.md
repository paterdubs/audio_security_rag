# evaluation_protocol.md — Giao thức đánh giá

> **Tài liệu này KHÔNG định nghĩa metric.** Định nghĩa nằm ở [SYSTEM.md §8](SYSTEM.md#8-phương-pháp-đánh-giá)
> và là nguồn chân lý duy nhất cho công thức. Ở đây trả lời câu khác:
> **chạy đánh giá thế nào để con số đọc được, và con số đó KHÔNG được phép nói gì.**
>
> Viết cho: người đọc chương đánh giá của khoá luận, và phiên làm việc tương lai trên repo
> này. Mọi quy tắc dưới đây đều đến từ một lỗi có thật đã xảy ra trong dự án, không phải từ
> sách giáo khoa.

---

## 1. Ba tập dữ liệu, ba câu hỏi khác nhau — đừng trộn

| Tập | Nguồn | Trả lời được câu gì | **KHÔNG** trả lời được câu gì |
|---|---|---|---|
| `data/synthetic/dev` (1.440 clip) | Scaper sinh từ `foreground_bank_train` (FSD50K · UrbanSound8K · DESED · ESC-50) | So sánh **giữa các run** với nhau | Hiệu năng thực địa |
| `gold_test` (367 segment) | AudioSet-strong (YouTube) | Khái quát hoá **xuyên bộ dữ liệu** | Hiệu năng trên audio giám sát thật |
| `dev` thật (570 segment) | AudioSet-strong, **giữ riêng, chưa gán nhãn** | — | — |

### 1.1 `synthetic/dev` thiên vị theo thiết kế — và đây là con số

Đo ngày 22/09 (`scripts/check_leakage.py`, số đầy đủ ở
[measurements/leakage_check_20260922.md](measurements/leakage_check_20260922.md)):

- **0,9433** số nguồn foreground của dev cũng xuất hiện trong train
- **1.284/1.440 clip dev (0,8917)** có ít nhất một nguồn foreground từng dùng trong train

Nghĩa chính xác: `synthetic/dev` đo khả năng khái quát sang **tổ hợp mới của những nguồn đã
nghe** (SNR khác, vọng khác, chồng lấn khác), **không** phải sang **nguồn chưa từng nghe**.

**Quy tắc:** mọi bảng số chấm trên `synthetic/dev` phải kèm câu cảnh báo này. Phép **so sánh
giữa các run** vẫn hợp lệ vì thiên vị áp dụng ngang nhau. Phép đọc **"đây là hiệu năng của hệ
thống"** thì không.

### 1.2 `gold_test` có một vết bẩn riêng, đừng tưởng nó sạch tuyệt đối

`gold_test` là AudioSet-strong, mà backbone của dự án là **PANNs CNN14 tiền huấn luyện trên
AudioSet**. Nghĩa là backbone gần như chắc chắn **đã nhìn thấy chính những clip đó** ở giai
đoạn tiền huấn luyện.

Không phải rò rỉ nhãn — người gán nghe mù và gán lại từ đầu (DATA_PLAN N1), nên nhãn mạnh là
của người, không phải của AudioSet. Nhưng **là rò rỉ biểu diễn**: backbone đã học đặc trưng
trên chính phân bố đó. Phải ghi vào phần Hạn chế.

Điểm mạnh bù lại, và nó thật: `foreground_bank_train` **không chứa một dòng AudioSet-strong
nào** (kiểm tra 4 của `check_leakage.py`, 5/5 cổng đạt). Nên `gold_test` vẫn là phép đo xuyên
bộ dữ liệu ở tầng dữ liệu huấn luyện, chỉ không sạch ở tầng tiền huấn luyện.

---

## 2. Chọn ngưỡng θ — quy tắc quan trọng nhất và dễ vi phạm nhất

`ml/evaluation/threshold_sweep.py` quét ngưỡng rồi ghi `theta_sao` (θ\*) vào `analysis.json`.

**θ\* được chọn trên CHÍNH tập đang chấm.** Vì vậy mọi `f1_sao` là **chặn trên lạc quan**,
không phải hiệu năng kỳ vọng trên dữ liệu chưa thấy. Không có ngoại lệ, không có run nào
thoát khỏi điều này.

Bắt buộc khi báo cáo:

1. Báo **cả** `f1_sao` (@θ\*) **và** `f1_05` (@θ=0,5). Chênh lệch giữa hai cột là thông tin,
   không phải nhiễu — ablation `pos_weight` cho thấy nó chênh **gần 5 lần** (0,0855 vs
   0,4160), và cột @0,5 mới là cột người dùng model thật sự gặp.
2. Ghi kèm giá trị θ\*. Một model cần cắt ở θ\*=0,90 là model **chỉ dùng được nếu biết trước
   bí mật đó**, khác hẳn model cắt ở 0,35.
3. Nếu sau này có tập validation riêng: chọn θ\* trên đó, chấm trên gold, và nói rõ đã làm
   vậy. Chừng nào chưa có, giữ nguyên cách ghi "chặn trên lạc quan".

**Cấm:** chọn θ riêng theo từng lát cắt dựa trên nhãn lát cắt của chính tập đang chấm. Đó là
rò rỉ. Chỉ được suy θ từ đại lượng đo được lúc suy luận (ví dụ SNR ước lượng).

---

## 3. Đọc bảng phân loại lỗi cho đúng

`ml/evaluation/error_taxonomy.py::phan_loai` chia mỗi sự kiện tham chiếu vào **sáu** rổ:
`dung` · `bien` · `thay_the` · `thieu` · `thua` (+ `gop`).

### 3.1 `bien` không phải loại lỗi thứ bảy song song với sed_eval

`bien` (dự báo đúng lớp nhưng lệch onset quá collar 200 ms) là phép **CHIA NHỎ LẠI rổ
Deletion + Insertion** của `sed_eval`, không phải một hạng mục độc lập. Một sự kiện lệch biên
bị `sed_eval` tính **một Deletion cộng một Insertion**.

Hệ quả bắt buộc nhớ:

- Hai cách đếm **không cộng ra cùng một số**. Báo cáo phải nói thẳng điều đó thay vì để người
  đọc cộng hai bảng rồi thấy vênh.
- **Deletion thật (`thieu`) và lệch biên (`bien`) không phân biệt được bằng D/I của
  `sed_eval`** — chỉ phân biệt được trong bảng 6 loại. Mọi phân tích cơ chế hỏng phải đọc
  bảng 6 loại, không đọc D/I.
- Tách ra vì hai lỗi cần hai cách sửa khác nhau: lệch biên sửa ở hậu xử lý / độ phân giải
  thời gian; không-nghe-ra-gì phải sửa ở dữ liệu hoặc ngưỡng.

### 3.2 Đường chéo ma trận nhầm gồm CẢ `dung` LẪN `bien`

`ma_tran_nham()` để `bien` nằm trên đường chéo (vì đúng lớp). Khi liệt kê "cặp nhầm đậm
nhất" phải **bỏ đường chéo**, nếu không cặp đậm nhất luôn là cặp lớp-với-chính-nó và nó che
mất cặp nhầm thật.

### 3.3 Mẫu số là `n_ref` của chính ô đang xét

Tỉ lệ `thua` **vượt 1,0 được** — một sự kiện thật có thể hứng nhiều dự báo thừa. Đó không
phải lỗi tính toán.

---

## 4. "Chưa đo" không phải là 0

Lỗi này đã xảy ra nhiều lần và luôn theo cùng một kiểu: một ô trống được in ra thành `0`,
rồi được đọc thành "đo rồi, kết quả bằng 0".

| Tình huống | Phải ghi | **Cấm ghi** |
|---|---|---|
| Epoch không chạy full-eval | "chưa đo" | F1 = 0 |
| Thiếu `calibration.json` | "chưa đo" | ECE = 0 |
| Ô cắt chéo có `n_ref = 0` | "chưa đo" / `—` | tỉ lệ = 0 |
| Run train trước khi có cờ `--pos-weight-max` | "chưa đo" | pos_weight = 0 |

Trong mã: trả `None`, đừng trả `0`. Trong `.md`: ghi `chưa đo` hoặc `—`. Có test riêng cho
quy tắc này ở `tests/test_chay_ablation_*.py` và `tests/test_check_leakage.py` — giữ chúng.

---

## 5. Khi nào hai run so được với nhau

`ml/tracking/compare_runs.py` khai ba phán quyết cho mỗi chiều (dữ liệu, mã):

| Phán quyết | Nghĩa | So được không |
|---|---|---|
| `giong` | vân tay trùng khớp | ✅ |
| `khac` | vân tay khác nhau, **biết khác chỗ nào** | ⚠️ so được nếu chỗ khác không nằm trên đường train |
| `KHONG_RO` | manifest lập **hồi cứu**, không biết gì | ❌ không được khẳng định |

`KHONG_RO` xuất hiện khi manifest được lập sau khi train xong (`hoi_cuu=true`) — khi đó
`code.tree_sha256` là băm mã ở thời điểm **lập manifest**, không phải lúc train. Đọc nó thành
"cùng mã nguồn" là chứng thực điều không ai biết. `v1`/`v2`/`v3` đều thuộc loại này.

**Điều kiện đủ để tuyên bố một phép so là hợp lệ**, theo mẫu đã dùng cho ablation
`pos_weight`: nêu rõ (a) `ml/training/` và `ml/models/` giống hệt nhau, (b) mọi `analysis.json`
do cùng một phiên bản `error_analysis.py` sinh ra, (c) hợp đồng dữ liệu `PASSED`, (d) manifest
ghi **lúc train**. Thiếu vế nào thì khai thiếu vế đó, đừng khoe quá lời.

### 5.1 Hai thứ TUYỆT ĐỐI không so

1. **`v1` với bất kỳ run nào khác** — lưới onset của nó là 319,7 ms, làm mọi số liệu biên
   không so được. Không có cách cứu.
2. **Cột thời gian (s/epoch) giữa các run chạy ở điều kiện nhiệt khác nhau.** Đo thật ngày
   22/09: cùng một máy, GPU bị thảo nhiệt chặn từ 2.100 MHz xuống **210 MHz** ở 89°C, epoch
   từ ~350 s lên ~600 s. Xung nhịp thấp làm **chậm**, không làm **sai** — nên chỉ số chất
   lượng vẫn dùng được bình thường, chỉ cột thời gian là vô nghĩa.

---

## 6. Điều kiện để một lần train được tính là hợp lệ

1. **Hợp đồng dữ liệu `PASSED`.** `train_sed.py::kiem_cong_hop_dong()` chặn cả `FAILED` lẫn
   `KHONG_RO`. Cờ `--force-du-lieu-chua-dat` bỏ qua được nhưng **ghi vào `manifest.json`** —
   run nào có cờ đó phải khai trong báo cáo.
2. **Manifest ghi lúc train**, theo từng epoch. Đếm epoch trong manifest là tín hiệu "đã xong"
   đáng tin duy nhất — dòng kết thúc của log có thể bị cắt khi tiến trình bị kill.
3. **Một biến một lượt.** Trộn hai biến trong cùng một lượt thì không quy kết được chênh lệch
   cho biến nào. `v2` đã vi phạm điều này (đổi đồng thời pooling, mixup, hậu xử lý) và hậu quả
   là không ai nói được cái nào gây ra thay đổi.
4. **Cấu hình khác mặc định phải truyền tường minh trong lệnh**, không sửa hằng số trong mã.
   Lý do: đổi mặc định làm mọi run cũ không còn tái lập được bằng lệnh cũ, mà không có cảnh
   báo nào. Ví dụ đang áp dụng: hằng `MAX_POS_WEIGHT = 30,0` giữ nguyên làm chứng tích của
   v1–v3; run mới truyền `--pos-weight-max 1.0`.

---

## 7. Báo cáo theo lát cắt

Bảng theo lát cắt là phần có giá trị nhất của chương đánh giá ([SYSTEM.md §8.3](SYSTEM.md#83-giao-thức-báo-cáo-theo-slice)) —
một F1 tổng che giấu hoàn toàn việc model hỏng ở đâu.

### 7.1 Lát cắt CHỒNG LẤN nhau, hàng đơn lẻ không tách được nguyên nhân

Trên `data/synthetic/dev` (1.440 clip): `reverb` 557 · `overlap` 437 · `low_snr` 390 ·
`sach` 357 · `causal_chain` 203 · `long_event` 144. Trong 144 clip `long_event` thì 52 cũng là
`reverb`, 47 cũng là `overlap`, 45 cũng là `low_snr`.

Vì vậy hàng `long_event` một mình **không tách được** "long_event tự nó khó" khỏi "clip
long_event tình cờ trùng lát cắt khó khác". Muốn tách phải dùng bảng 2×2 cắt chéo
(`ml/evaluation/long_event_crosscut.py`). Ba giả thuyết `long_event` đầu tiên của dự án đều
được đo trên cái hàng lẫn lộn đó.

### 7.2 Cỡ mẫu ô cắt chéo chỉ 45–52 clip

Đủ để kết luận **có / không có** tương quan rõ. **Không** đủ để kết luận **độ lớn** hiệu ứng.
Ô `long_event+causal_chain` chỉ có 15 clip — mọi con số ở đó phải kèm cảnh báo.

---

## 8. Cổng chất lượng của gold set

Thuộc [DATA_PLAN §8.5](DATA_PLAN.md#85-cổng-chặn), nhắc lại ở đây vì nó quyết định số nào
được phép đi vào khoá luận:

| Chỉ số | Ngưỡng | Áp cho |
|---|---|---|
| Tự-nhất-quán event-based (collar 200 ms) | ≥ 0,75 | batch pilot |
| Lệch onset trung vị giữa hai lần gán | ≤ 100 ms | batch pilot |
| Số event mỗi lớp | ≥ 20 | `gold_test` đầy đủ |
| Số clip mỗi lát cắt | ≥ 20 | `gold_test` đầy đủ |

**Đây là tự-nhất-quán (test–retest), KHÔNG phải kappa liên-người** — dự án chỉ có một người
gán. Gọi nhầm tên là sai về phương pháp, không phải sai về chữ nghĩa. Xem
[DATA_PLAN §8.1](DATA_PLAN.md#81-ràng-buộc-chỉ-có-một-người-gán-nhãn).

**Cảnh báo về sức mạnh thống kê:** pilot lần 1 chỉ có **52 sự kiện**, nên sai số chuẩn của
ước lượng ≈ 0,060, tức khoảng tin cậy 95% khoảng **±0,12**. Ở cỡ mẫu đó, kết quả 0,7339 và
ngưỡng 0,75 **không phân biệt được với nhau**. Khi thiết kế vòng pilot tiếp theo phải tính cỡ
mẫu trước, đừng áp một ngưỡng chính xác lên một phép đo không đủ sức phân giải.

---

## 9. Quy ước trình bày số

- **Dấu phẩy thập phân** trong mọi file `.md` (`0,4333` không phải `0.4333`).
- Ô chưa đo ghi `chưa đo` hoặc `—`, không ghi `0` (xem §4).
- Giữ 4 chữ số thập phân cho F1/ECE; làm tròn ít hơn sẽ che mất chênh lệch thật giữa các run
  (`pw30` vs `v3` lệch 0,0039 — làm tròn 2 chữ số là mất luôn thông tin).
- Mỗi bảng số đo phải có một file riêng trong `docs/measurements/` đặt tên kèm ngày, và
  **không sửa mục lịch sử**: nếu phép đo mới bác bỏ trang cũ, thêm mục đính chính trỏ sang
  trang mới, giữ nguyên trang cũ.

---

## 10. Chi phí đã đo của từng bước (để xếp lịch)

Đo trên máy hiện tại, `data/synthetic/dev` 1.440 clip:

| Bước | Thời gian |
|---|---|
| `predictions` | 29–38 s |
| `calibration` | 6 s |
| `threshold_sweep` | 96–101 s |
| `error_analysis` | 95–170 s |
| `long_event_crosscut` | 10–23 s |
| `check_leakage` (đủ 5 cổng, đọc 9.360 JAMS) | 20–60 s |
| Train 25 epoch | **~2,5 giờ** khi mát; **tới ~4 giờ** khi GPU bị thảo nhiệt chặn |

Bước nào trên 60 s thì chạy nền. Train qua đêm dùng `Start-Process` của PowerShell để tách
hẳn khỏi phiên làm việc.

---

## 11. Những gì bộ đánh giá hiện tại CHƯA làm được

Liệt kê ở đây để không ai đọc thừa vào các con số đang có:

- **Chưa có số nào trên audio giám sát thật.** `gold_test` là YouTube; thu thực địa tại IUH
  chưa bắt đầu.
- **Chưa có `psds_eval`** — L1 hiện chỉ có event-F1, segment-F1, onset MAE. PSDS trong
  SYSTEM.md §8.1 vẫn là kế hoạch.
- **L2 / L2b / L3 / L4 / L5 chưa có gì** — toàn bộ bộ metric hallucination (đóng góp C2) chưa
  cài, `EVENT_LEXICON` chưa kiểm định.
- **Chưa đo trên tập con "nguồn chưa từng nghe".** Có sẵn **156 clip dev** mà toàn bộ nguồn
  foreground chưa từng vào train — proxy duy nhất hiện có cho câu hỏi đó, chấm lại rẻ, **chưa
  làm**.
- **`--resume` chưa được kiểm bằng thực tế**, nên chưa ai biết một lần train bị ngắt rồi nối
  lại có cho cùng kết quả không.
