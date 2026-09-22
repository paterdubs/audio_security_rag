# CLAUDE.md

> **ĐỌC FILE NÀY ĐẦU TIÊN trong mỗi phiên làm việc mới.**
> **CẬP NHẬT FILE NÀY Ở CUỐI MỖI BLOCK CÔNG VIỆC** (xem [§9](#9-giao-thức-cập-nhật)).
>
> Đây là **living context + progress tracker**. Nó trả lời câu hỏi *"đang ở đâu, làm gì tiếp"*.
> Nó **không** chứa đặc tả (→ `docs/SYSTEM.md`) và **không** chứa kế hoạch dài hạn (→ `docs/PLAN.md`).

---

## 1. Dự án là gì

**Grounded Automated Audio Captioning for Security Surveillance and RAG-based Alert Retrieval**
Khoá luận tốt nghiệp — Khoa học dữ liệu — IUH K18.

Hệ thống nghe luồng âm thanh liên tục → phát hiện sự kiện có định vị thời gian → sinh mô tả ngôn ngữ tự nhiên **được ràng buộc vào bằng chứng âm học** → chuẩn hoá thành Security Event có mức rủi ro → cảnh báo tức thời + lưu trữ index vector → truy vấn lịch sử bằng tiếng Việt qua RAG có trích dẫn. Toàn bộ vận hành theo MLOps.

**Đóng góp nghiên cứu chính:** kiến trúc Grounded AAC dùng strong label làm tín hiệu grounding + bộ metric đo hallucination cho AAC. Xem `docs/SYSTEM.md` §1.5.

**Deadline:** 09/11/2026 (8 tuần từ 15/09/2026).

---

## 2. Bản đồ tài liệu

| File | Trả lời câu hỏi | Khi nào đọc |
|---|---|---|
| **CLAUDE.md** (file này) | *Đang ở đâu?* | **Mỗi phiên, đầu tiên** |
| `docs/PLAN.md` | *Làm gì, khi nào, nghiệm thu ra sao?* | Đầu mỗi tuần + khi chọn task |
| `docs/SYSTEM.md` | *Hệ thống là gì?* (đặc tả đầy đủ = khung báo cáo) | Khi cần chi tiết kỹ thuật |
| `docs/DATA_PLAN.md` | *Chuẩn bị dữ liệu thế nào?* (10 ngày, quy trình gán nhãn có máy hỗ trợ) | **Suốt W2–W3** |
| `docs/taxonomy.md` | 16 class, định nghĩa bao gồm/loại trừ | Khi làm việc với nhãn |
| `docs/annotation_guideline.md` | Quy ước gán nhãn | Khi gán nhãn |
| `docs/evaluation_protocol.md` (chưa có) | Metric và cách đo dự kiến | Tạm xem SYSTEM.md §8 |
| `docs/STATUS.md` | Snapshot đã đối soát, bằng chứng, giới hạn | Khi cần số liệu hiện hành |
| `docs/TRAINING_OPS_PLAN.md` | Tracking, kiểm tra dữ liệu, phân tích lỗi | Trước lần train mới |
| `docs/RELATED_WORK_2026.md` | Đối chiếu văn liệu và mức xác minh | Khi viết chương 2 |
| `docs/data_inventory.md` | Số giờ/class (sinh tự động) | Khi lo về dữ liệu |
| `docs/decisions/ADR-*.md` | Vì sao chọn thế này | Khi định thay đổi kiến trúc |

---

## 3. Trạng thái hiện tại

**Tuần:** W1 (15–21/09/2026). Walking skeleton và baseline SED được làm sớm song song.
**Đối soát workspace:** 19/09/2026, sau khi Pha 1–5 của Training Ops, lượt quét ngưỡng,
lượt phân tích lỗi và ablation hậu xử lý đã hoàn tất.
Số đo, phạm vi và bằng chứng ở [docs/STATUS.md](docs/STATUS.md); nhật ký §10 là lịch sử,
không phải trạng thái hiện hành.

### Đã có

- Git trên nhánh `master`; base trước lượt đồng bộ đầy đủ là `70ab498`. Snapshot dự án ngày
  18/09 được chuẩn bị để publish lên GitHub; xem `git log` thay vì ghi cứng HEAD hiện hành.
- Docker gồm 6 service. Upload offline → PANNs pretrained → caption template EN/VI → risk rules →
  PostgreSQL/pgvector → BGE-M3 → RAG template + citation; dashboard 3 màn, WebSocket và feedback.
  Docker Compose hiện đã được dừng chủ động để giải phóng tài nguyên; named volume được giữ nguyên.
- Taxonomy 16 lớp = 10 an ninh + 6 nhầm lẫn/nền. SED huấn luyện **15 đầu ra**,
  không có head riêng cho `ambient_noise`.
- Foreground **4.267 clip / 15 lớp sự kiện**, background **2.461**, RIR **505**.
  `shout_yell` 56/80: giữ nguyên theo ADR-0005. 1.831 lượt bulk-accept không phải nghe duyệt từng clip.
- Lô **mới** sau B0–B9 có **7.920 train + 1.440 dev**, 10 s/clip; `verify_synthetic.py`
  đã PASS ở hai split, không vi phạm hợp đồng. Lô **legacy** được giữ riêng chỉ với
  9.360 JAMS, `slice_index` và hai log stdout làm bằng chứng trước-sửa; audio legacy đã xoá.
- `scaper_generate.py` đã dùng lựa chọn source theo clip, nền 10 s, duration lấy từ
  `train_strong`, overlap ép thật, `long_event` 4 s giới hạn lớp cấp được, mô phỏng nhãn và
  seed riêng từng clip / chạy song song. Index vẫn chỉ được ghi ở cuối lượt sinh, nên đây vẫn là
  nợ kỹ thuật nếu job sau này bị ngắt.
- Precompute waveform PANNs 32 kHz hiện có đủ **train 7.920** (5,07 GB) và **dev 1.440**
  (0,92 GB). Đây là cache waveform cho baseline PANNs, **không phải** feature BEATs.
- **Cả ba run PANNs đã hoàn tất 25 epoch**, và 19/09 đã được **chấm lại trên cùng một
  tập dev 1.440 clip** với ngưỡng riêng từng run:

  | dev chung, θ tối ưu | v1 · pool 5 · legacy | v2 · pool 3 · legacy | v3 · pool 3 · lô mới |
  |---|---:|---:|---:|
  | clip mAP | 0,7464 | 0,7555 | **0,8132** |
  | event-F1 @ θ=0,50 | 0,1147 | 0,1123 | 0,0820 |
  | event-F1 @ θ* | 0,2981 (θ 0,96) | 0,3531 (θ 0,89) | **0,3986** (θ 0,91) |
  | segment-F1 @ θ* | 0,6208 | 0,6564 | **0,7159** |

  🔴 **Ngưỡng 0,5 sai nghiêm trọng ở cả ba run** — v3 tăng **4,86 lần** khi chọn ngưỡng
  đúng. Bảng cũ chấm ở 0,5 (mAP 0,8372/0,8176/0,8123 · event-F1 0,1282/0,1897/0,1077) đo
  trên ba tập val khác nhau và **không dùng để xếp hạng ba run được nữa**; nó vẫn còn ở
  STATUS §3 làm bản ghi lịch sử. Confound chưa gỡ: dev sinh cùng recipe với train của v3.
- **Pha 1–5 của [TRAINING_OPS_PLAN](docs/TRAINING_OPS_PLAN.md) đã xong (19/09):**
  `ml/tracking/{fingerprint,run_manifest,compare_runs}.py`,
  `ml/evaluation/{predictions,threshold_sweep,error_taxonomy,error_analysis}.py`.
  `train_sed.py` gieo toàn bộ RNG và ghi `manifest.json` trước epoch đầu.
- **Phân tích lỗi đã chạy trên cả ba run** (`ml/runs/{run}/analysis.{json,md}`). Ở θ = 0,5
  v3 chỉ bỏ sót **9/4.873** sự kiện nhưng chèn thêm **61.020** — hỏng ở ngưỡng, không phải
  ở năng lực phát hiện. **37,4%** lỗi biên của v1 là trần cứng do bước lưới onset 319,7 ms
  (v2/v3: 79,9 ms → trần 0%), nên **so event-F1 của v1 với v2/v3 là so một phần độ phân
  giải bộ giải mã**. Số đo: `docs/measurements/error_analysis_20260919.md`.
- **Pha 5 kết luận: KHÔNG có phép so sánh sạch nào giữa v1/v2/v3.** `compare_runs.py`
  khai `KHONG_RO` cho cả dữ liệu (vân tay v1/v2 là `null`) lẫn mã nguồn (cả ba manifest
  lập hồi cứu cùng lúc nên `tree_sha256` **trùng nhau** — đó là mã lúc lập manifest, không
  phải lúc train). Cũng đừng đọc `manifest.ket_qua_cuoi`: nó chấm trên val split **riêng**
  của từng run, xếp v2 > v1 > v3, **ngược hẳn** dev chung.
- **`long_event` hỏng vì phân mảnh, không phải vì bỏ sót** — phân mảnh gấp 1,9 lần mốc
  clip sạch, Insertion gấp 1,8 lần (hệ quả cơ học của phân mảnh), còn Deletion thì bình
  thường. Ablation **bác bỏ một phần** giả thuyết "cửa sổ lọc 7 khung quá hẹp": cửa sổ
  theo lớp nâng *mọi* lát cắt lên xấp xỉ cùng một lượng nên khoảng cách tới mốc gần như
  không đổi (0,180 → 0,178). Số đo: `docs/measurements/long_event_postproc_20260919.md`.
- **Trần `MAX_MEDIAN_FRAMES=51` cũng bị bác bỏ phần lớn** — quét tới 401 khung: `v2` gap
  rộng ra, `v3` chỉ hẹp 6,4%. **Cửa sổ suy từ train** (`cua_so_loc_tu_train()`, không rò
  rỉ) khớp cửa sổ suy từ dev tới 6 chữ số — vì 9/15 lớp đã kẹp trần ở cả hai nguồn, không
  phải vì train/dev giống nhau. **ECE đo lần đầu**: vùng dự báo 0,4–0,6 mang hơn nửa
  triệu khung mà tỉ lệ dương thật chỉ 1,7–3% — giải thích triệu chứng θ*≈0,9, chưa kết
  luận- **`long_event` đã loại BỐN ứng viên liên tiếp — nhưng độ dài đã được xác nhận là biến
  khó ĐỘC LẬP.** Ứng viên thứ tư "xấu vì trùng lát cắt khó khác" bị bác bỏ 21/09 bằng bảng
  2×2 (`ml/evaluation/long_event_crosscut.py`, v3 @θ*=0,90, dev/all): chênh lệch tỉ lệ phân
  mảnh `long_event` − `khac` là 0,11–0,20 và **không tan đi ở cột nào**; `long_event` vỡ
  0,29–0,35 bất kể `low_snr`/`reverb`/`overlap`, `khac` vỡ 0,13–0,18. `long_event` SẠCH
  `reverb` còn vỡ nhiều hơn `long_event` CÓ `reverb` (0,33 vs 0,29 — ngược chiều). Cỡ mẫu
  nhỏ (45–52 clip/ô). Ba ứng viên trước: cửa sổ hẹp, trần thấp, khe hở năng lượng thật
  (21/09, khe hở hai nhóm bằng nhau tuyệt đối 0,012s/0,046s; p90 nhóm KHÔNG vỡ còn cao hơn).
  Số đo: `docs/measurements/long_event_{gap,crosscut}_20260921.md`.
- **`pos_weight` ĐƯỢC XÁC NHẬN là nguyên nhân của lệch hiệu chuẩn** (22/09, ablation ba
  mức chạy qua đêm bằng `scripts/chay_ablation_pos_weight.py`). Cùng cấu hình v3, chỉ đổi
  `--pos-weight-max`; hợp đồng `PASSED`, manifest ghi lúc train:

  | run | pos_weight | ECE | θ* | F1 @θ* | **F1 @0,5** |
  |---|---:|---:|---:|---:|---:|
  | `pw30` | 30 | 0,3308 | 0,90 | 0,4018 | 0,0855 |
  | `pw10` | 10 | 0,2033 | 0,80 | 0,4192 | 0,2340 |
  | `pw1` | 1 | **0,0231** | **0,35** | **0,4333** | **0,4160** |

  Đơn điệu tuyệt đối ở cả bốn cột, **không có đánh đổi** — trần 30 làm hỏng CẢ hiệu chuẩn
  LẪN F1. `pw30` tái hiện `v3` trong vòng 0,7% (ECE 0,3308 vs 0,3332, θ* trùng khít), nên
  giả thuyết không phải rút lại — và đó cũng là **bằng chứng đầu tiên về khả năng tái lập
  của pipeline**. Ý nghĩa thực tế nằm ở cột cuối: ở ngưỡng mặc định 0,5, trần 30 cho F1
  0,0855 (gần như vô dụng) còn trần 1,0 cho 0,4160 — **gần 5 lần**.
  **Quyết định: KHÔNG đổi hằng `MAX_POS_WEIGHT = 30.0`** (đổi mặc định làm v1–v3 không tái
  lập được bằng lệnh mặc định); v4 trở đi truyền tường minh `--pos-weight-max 1.0`.
  Số đo: `docs/measurements/pos_weight_ket_luan_20260922.md`.
- **`low_snr` hỏng bằng hai cơ chế khác nhau tuỳ độ dài sự kiện** (21/09, tái hiện trên cả
  v2 lẫn v3): trên sự kiện thường là **Deletion** (`thieu` hứng 51–60% phần `dung` mất đi,
  `thua` không tăng), trên sự kiện dài là **lệch biên** (`bien` hứng 61–72%, `thieu` gần
  như đứng yên). Substitution nhỏ nhất ở mọi ô và **cấu trúc ma trận nhầm không đổi** → 
  `low_snr` không sinh cặp nhầm mới, không khai thêm `confusable_with`. Hai cơ chế cần hai
  cách sửa khác nhau: Deletion là bài toán ngưỡng/dữ liệu, lệch biên là bài toán hậu xử
  lý/độ phân giải thời gian. Số đo: `docs/measurements/low_snr_co_che_20260921.md`.
- **Cổng hợp đồng dữ liệu giờ CHẶN thật, và checkpoint resume có optimizer/scheduler/RNG
  đầy đủ.** `kiem_cong_hop_dong()` chặn cả `FAILED` lẫn `KHONG_RO` như nhau (trước 19/09
  cả hai chỉ in cảnh báo) — bỏ qua bằng `--force-du-lieu-chua-dat`, cờ ghi vào
  `manifest.json`. `checkpoint_resume.pt` ghi đè mỗi epoch (tách khỏi `best.pt` — không đổi
  schema mà `predictions.py`/`eval_sed.py` phụ thuộc); `--resume` khôi phục RNG nên dãy số
  ngẫu nhiên tiếp theo giống hệt như chưa hề dừng.
- **710 test đạt** (22/09, `pytest tests/` exit 0). Mốc trước thay đổi là 703.
  Từ 649: pilot gold lần 1 (30/30), bộ gán mù lần 2 (`make_blind_set.py`, đã chạy thật),
  `agreement.py` (**đã chạy thật 22/09** — vá lỗi đuôi `.wav`, thêm `chi_tiet_bat_dong()`,
  xem "Đang làm" bên dưới), `--pos-weight-max`.

### Đang làm / chưa nghiệm thu

- 🔴 **`long_event` là điều khoản BẤT KHẢ THI với bank hiện có** — phát hiện 17/09 15:00.
  Chỉ **90/4.267** clip bank còn ≥8 s sau khi cắt im lặng; 9/15 lớp có **0** clip;
  `siren` chỉ **2/311** (p90 4,01 s, do UrbanSound8K cắt sẵn ở 4 s). Vì thế lô mới dùng
  ngưỡng 4 s và chỉ các lớp cấp được; bảng đo và giới hạn ở STATUS §6–§7.
- AudioSet-strong: **937 WAV / 937 dòng segments** — khớp tuyệt đối. Process tải **đã dừng**,
  chưa hết hàng đợi. **21/09: đã vào manifest/split** — `build_manifest.py --source
  audioset_strong` (3.977 dòng sự kiện) rồi `make_splits.py` → **1.601 dòng sự kiện / 367
  file WAV riêng biệt vào `gold_test`** (2.376/570 vào `dev`), khớp đúng tỉ lệ 0,6/0,4 của
  `splits.yaml`. `scripts/select_gold_pilot.py` chọn 30 clip pilot rải đều 15 lớp (seed cố
  định, tái lập được) → `data/gold/pilot_v1_candidates.csv`. **Vẫn chưa có nhãn** — đây là
  danh sách ứng viên để bắt đầu DATA_PLAN §8.3 bước 1, chưa phải gold_test hoàn chỉnh.
- F1 bằng 0 tại epoch không full-eval là **chưa đo**, không phải F1 thực. Tất cả số trong
  bảng trên lấy từ epoch 25 có full-eval.
- CI, DVC pipeline, BEATs–Conformer–BART, grounded decoding, streaming Redis Streams,
  temporal aggregator, gold G1–G4 và tracking đầy đủ vẫn chưa xong.
- Inference đang dùng PANNs pretrained; **chưa deploy v1/v2/v3**. MLflow healthy không có nghĩa
  training đã log lên MLflow. `AUDIO_RETENTION_DAYS` chưa có tác vụ tự xoá audio hết hạn.
- Checkpoint/model weight `.pt/.pth/.ckpt/.onnx` đã được Git-ignore và giữ tại máy.
  `data/synthetic_legacy/` được đưa lên có chủ đích làm bằng chứng trước-sửa; JAMS legacy chứa
  đường dẫn tuyệt đối lịch sử của máy sinh dữ liệu nên dùng để audit, không replay portable nguyên trạng.

### Việc tiếp theo

0. ✅ **Kiểm lại `long_event`/`low_snr` trên θ\* hợp lý (`pw1`) — XONG 22/09.** Cả hai chẩn
   đoán đo trên `v3` @θ\*=0,90 đều được kiểm lại trên `pw1` @θ\*=0,35: tỉ lệ phân mảnh
   `long_event` **giữ nguyên** (không phải hệ quả của ngưỡng cực đoan); cơ chế Deletion
   trên sự kiện thường **giữ nguyên**; nhưng câu "lệch biên chiếm gần hết, Deletion gần
   như không đóng góp" trên sự kiện dài bị **phóng đại** — tỉ trọng Deletion tăng gấp ba
   (10,2%→32,1%) ở θ\* hợp lý hơn, dù lệch biên vẫn trội hơn. Đã đính chính trong
   [low_snr_co_che_20260921.md](docs/measurements/low_snr_co_che_20260921.md) §6, chi tiết
   đầy đủ ở [phan_tich_lai_tren_pw1_20260922.md](docs/measurements/phan_tich_lai_tren_pw1_20260922.md).
1. **Chốt cấu hình v4 và train nó.** Ablation đã xong: `--pos-weight-max 1.0` thắng ở mọi
   chiều. `panns_ft_pw1` **đã chính là** cấu hình v4 dự kiến — cần quyết định đề bạt thẳng
   nó làm mốc báo cáo (khuyến nghị, không tốn GPU) hay train lại dưới tên `v4` riêng.
2. **`long_event`: còn ba hướng, `--time-pool-blocks` giờ đã đủ điều kiện chạy.** Bốn giả
   thuyết đã bị loại (cửa sổ hẹp, trần thấp, khe hở năng lượng, trùng lát cắt khó), và mục
   0 vừa xác nhận cả hai chẩn đoán còn lại (phân mảnh, lệch biên) vững qua phép kiểm θ\*.
   Hướng còn lại: (a) nghe trực tiếp clip vỡ nhiều nhất — cần tai người; (b) trường tiếp
   nhận thời gian của CNN14 ngắn hơn sự kiện 4s — kiểm bằng ablation
   `--time-pool-blocks` ∈ {3,2} trên nền `--pos-weight-max 1.0`, tốn ~2 giờ GPU/lượt; (c)
   BCE theo khung không phạt phân mảnh — phải đổi loss rồi train lại, chưa làm.
3. **`--resume` vẫn CHƯA được kiểm bằng thực tế.** Runner qua đêm 21→22/09 có watchdog gọi
   `--resume` khi train chết, nhưng cả ba lượt chạy trót lọt nên nhánh đó không lần nào
   được thực thi. Món nợ còn nguyên.
4. ✅ **Gold pilot 30 clip (DATA_PLAN §8.3 bước 1) — XONG 22/09.** 30/30 gán được, sau
   **16 file loại / 46 lượt nghe = tỉ lệ thất bại 34,8%** (7 game/phim
   `synthetic_or_game_audio`, 6 rác/không liên quan `unusable`, 3 nhạc nền
   `music_no_event`). **Mẫu hình 100%: mọi file "unusable"/"synthetic" đều chạm ít nhất
   một trong 5 lớp `glass_breaking`/`scream`/`explosion`/`gunshot`/`siren`** — súng
   nổ/bom/kính vỡ/tiếng thét THẬT hiếm khi được quay đăng công khai lên YouTube, nên phần
   lớn nhãn AudioSet cho các lớp này thực ra là game/phim/dàn dựng. **Cần ghi nguyên văn
   con số 34,8% vào phần Hạn chế của khoá luận.**
   52 dòng sự kiện hợp lệ, nhưng **4 lớp vắng mặt hoàn toàn** (`fireworks`,
   `running_footsteps`, `shout_yell`, `vehicle_crash`) — hệ quả của việc chọn thay thế
   dồn về đúng các lớp hay hỏng nhất; không phải lỗi, pilot không yêu cầu phủ đều 15 lớp
   (đó là yêu cầu của `gold_test` đầy đủ ở cổng §8.5, không phải pilot).
   Case đồ chơi mô phỏng (`0yZGysisqY0`, không gán `gunshot`) cùng nguyên tắc ca chuông
   quầy phục vụ (§4.4 15/09). Quy tắc kiểm tra audio thật ở `annotation_guideline.md`
   §0.1. Tổng kết đầy đủ + khuyến nghị cho gán đại trà ở `data/gold/decision_log.md`.
   ✅ **Bước 2–3 XONG 22/09 — cổng §8.5 VỪA HỤT, chưa đủ điều kiện gán đại trà.**
   `pilot_v1_lan2.tsv` (60/60 clip gán mù) → `agreement.py` chạy thật lần đầu, phát hiện
   VÀ vá một lỗi code thật (đuôi `.wav` không khớp giữa `pilot_v1_lan1.tsv` và
   `mapping.csv`, làm mọi event-F1 ra NaN — xem "Đang làm" bên dưới). Kết quả sau khi vá:
   **event-F1 tự-nhất-quán 0,7339** (ngưỡng ≥0,75 — hụt 0,0161, KHÔNG ĐẠT), lệch onset
   trung vị **10,0 ms** (ngưỡng ≤100ms — ĐẠT). **13/30 clip pilot có ca bất đồng** (danh
   sách đầy đủ ở `agreement.py`'s output, mục "Danh sách ca bất đồng"). Ca đáng chú ý
   nhất: `as_strong_0yZGysisqY0_12000` (case "đồ chơi mô phỏng" của lần 1, không gán sự
   kiện) — lần 2 mù lại nghe ra `alarm_bell`+`explosion` (3 Insertion tuyệt đối).
   ✅ **Nghe lại lần ba 13/13 ca XONG 22/09 — 9 mục mới trong `decision_log.md`.** Ba loại
   nguyên nhân: lệch biên nhẹ (3 ca) · sự kiện thật bị bỏ sót ở lần 1 hoặc lần 2 (4 ca,
   không có bên nào luôn đúng hơn) · **một đổi lớp thật** (`0BEtPXdDQrs_250000`: `siren`
   không phải `alarm_bell` — cặp đã có sẵn trong `confusable_with` từ lỗi model, giờ xác
   nhận thêm ở người). Phát hiện mẫu hình mới: `speech_normal` nền bị bỏ sót dưới
   `applause_cheering` ở 2 clip — đã thêm cảnh báo vào §4 `annotation_guideline.md`.
   **Phát hiện quan trọng nhất:** `as_strong_0yZGysisqY0_12000` — phán quyết "đồ chơi" của
   lần 1 được CỦNG CỐ, nhưng lần 2 mù đã bị đồ chơi đánh lừa hoàn toàn. Kết luận: tự-nhất-
   quán đo ở §8 có thể **đánh giá thấp** độ tin cậy thật của quyết định "đồ chơi" khi mất
   ngữ cảnh — phải ghi vào Hạn chế khoá luận.
   **Vẫn CHƯA đủ điều kiện gán đại trà** — cổng §8.5 chưa đạt (0,7339 < 0,75); cần cân nhắc
   một vòng pilot mới sau khi áp dụng sửa đổi guideline, trước khi thử lại cổng và qua
   checklist §10.
   Đây là thứ DUY NHẤT gỡ được confound "dev cùng recipe với train của v3". Không dùng
   test để chọn threshold.
3. ✅ **Ablation `pos_weight` — XONG 22/09, giả thuyết được XÁC NHẬN.** Ba lượt
   `pw1`/`pw10`/`pw30` chạy qua đêm bằng `scripts/chay_ablation_pos_weight.py`, cùng cấu
   hình v3 chỉ đổi một biến. Quan hệ đơn điệu tuyệt đối, không đánh đổi; `pw30` tái hiện
   `v3` trong 0,7%. Chi tiết ở §0 và `docs/measurements/pos_weight_ket_luan_20260922.md`.
   Hằng `MAX_POS_WEIGHT` **giữ nguyên 30,0** có chủ đích; v4 truyền tường minh
   `--pos-weight-max 1.0`.

---

## 4. Quyết định đã khoá

Không thảo luận lại trừ khi có lý do mới. Mỗi thay đổi phải kèm một ADR.

Đây là quyết định **thiết kế đích**. Hiện serving dùng PANNs + template; RAG chưa có
provider LLM thực, embedding đã dùng rich document VI/EN + metadata. Xem STATUS để
phân biệt implementation với kiến trúc BEATs/Grounded AAC/Redis Streams dự kiến.

| Quyết định | Chọn | Ngày | Lý do ngắn |
|---|---|---|---|
| Số class | **16** (10 an ninh + 6 nhầm lẫn) | 14/09, sửa 15/09 | 6 lớp nhầm lẫn kiểm soát False Alarm Rate. **15/09:** tách `laughter_cheering` → `laughter` + `applause_cheering` — tiếng cười một giọng khác hẳn phổ/nhịp vỗ tay-đám đông |
| Loại nhãn | **Strong label** (onset/offset) | 14/09 | Điều kiện cần cho event-based F1 **và** cho grounding |
| Trường hợp đặc biệt | 6 lớp nhầm lẫn/nền **trong** taxonomy + **8 test slice** | 15/09 | 5 lớp có event head + ambient_noise nền; slice không phải class |
| Bối cảnh | **Tổng quát cho cả 4 khu vực** (trường học, bãi xe, dân cư, nhà máy) | 14/09 | `location` là metadata, không phải tham số model |
| Backend | **FastAPI toàn bộ** (không Django) | 14/09 | Async-native cho streaming; một framework cho 3 service |
| Vector DB | **pgvector** (không Qdrant) | 14/09 | Lọc metadata + vector search trong **một** câu SQL |
| Broker | **Redis Streams** (không Kafka) | 14/09 | Broker thật, zero ops thêm; Kafka là over-engineering ở 1 node |
| Audio encoder | **BEATs đóng băng**, precompute `.npy` | 14/09 | Feature mức frame + chi phí train khả thi trong 8 tuần |
| Captioning | **Grounded AAC tự xây**, không dùng API sẵn | 14/09 | Đây là đóng góp nghiên cứu, không phải tính năng |
| GPU | **Có** | 14/09 | Cho phép train Conformer + BART |
| LLM cho RAG | Interface pluggable, mặc định API, fallback Ollama | 14/09 | Không khoá cứng nhà cung cấp |
| Caption song ngữ | Sinh EN (benchmark) + dịch VI (RAG), embed bản **VI** | 14/09 | Truy vấn tiếng Việt, tránh sụt retrieval |
| Strong label cho train | **Scaper sinh tự động** từ foreground bank | 14/09 | Biên đặt nguồn, cần kiểm chứng phần âm thanh thực sự nghe được và hợp đồng slice |
| Gán nhãn có máy hỗ trợ | ✅ Cho foreground bank · ❌ **Cấm cho gold test set** | 14/09 | Máy đề xuất + người bấm duyệt ⇒ confirmation bias ⇒ gold nghiêng về model |
| Số người gán nhãn | **Một người** | 14/09 | Hệ quả: không tính được kappa liên-người → dùng test–retest, ngưỡng **0.75**, và **phải nêu ở phần Hạn chế** (DATA_PLAN §8) |

---

## 5. Thoả thuận làm việc với AI

Đọc kỹ — đây là phần quyết định hiệu quả của 8 tuần.

### Luôn làm
- **Đọc `CLAUDE.md` → `docs/PLAN.md` (tuần hiện tại) trước khi code.** Không tự chọn task ngoài kế hoạch.
- Task nào cũng bám vào một dòng trong `PLAN.md`. Nếu không có dòng nào khớp → hỏi trước, đừng tự thêm việc.
- Chạy script validate trước khi tuyên bố xong. **"Code chạy" ≠ "nghiệm thu"**.
- Với mỗi thay đổi kiến trúc: viết ADR vào `docs/decisions/`.
- Cập nhật `CLAUDE.md` ở cuối mỗi block (§9).
- Báo cáo trung thực: test trượt thì nói trượt kèm output; bỏ bước nào thì nói rõ.
- **Sau khi sửa `ontology_map.yaml` (đổi/tách/gộp lớp): grep toàn repo tìm tên lớp cũ TRƯỚC khi coi là xong.** File_id ở nhiều script (`build_manifest.py`, `auto_screen.py`, `normalize_audio.py`) chứa `class_id`, nên đổi ontology mà không quét lại toàn bộ nguồn liên quan để lại dòng "mồ côi" mang class đã mất — nó trùng checksum với dòng mới và không script nào tự xoá được, vì cách khớp theo `file_id`/`dict.update()` không bao giờ thấy dòng đó nữa. Đã xảy ra thật 15/09 ở `raw_manifest.csv`, `review_flags.csv`, và nặng hơn: `auto_screen.py --class-id X` từng **ghi đè toàn bộ** `screen_scores.csv` chỉ với điểm của X, xoá sạch điểm mọi lớp khác. Cả ba đã sửa (`drop_stale_rows`, `source_prefix`, `merge_scored`) nhưng bài học áp dụng cho MỌI file tích luỹ theo `file_id` trong tương lai, không riêng ba file này.

### Không bao giờ làm
- ❌ Bịa số liệu, bịa tên dataset, bịa trích dẫn paper. Không chắc → đánh dấu `⚠️ CẦN XÁC MINH` và ghi vào mục "Nợ kỹ thuật" của `PLAN.md`.
- ❌ Chạm vào test set để tuning bất cứ thứ gì. Ngưỡng chọn trên **dev**, khoá lại, test chạy **một lần**.
- ❌ Thêm cột vào TSV nhãn (vỡ tương thích `sed_eval`). Metadata đi file riêng.
- ❌ Tự viết lại metric đã có thư viện chuẩn (`sed_eval`, `psds_eval`, `aac-metrics`).
- ❌ Commit dữ liệu vào git — dữ liệu do DVC quản.
- ❌ Mở rộng scope. Có ý tưởng hay → ghi vào §8 "Ý tưởng để dành", không làm ngay.
- ❌ Chọn model/threshold dựa trên số đẹp hơn là phương pháp đúng.

### Ưu tiên khi phải đánh đổi
1. **Tính đúng đắn của phương pháp** (không rò rỉ, không tuning trên test) — không bao giờ hy sinh
2. **Nghiệm thu của tuần hiện tại**
3. **Đóng góp nghiên cứu C1/C2** (Grounded AAC + metric hallucination)
4. Độ hoàn thiện kỹ thuật
5. Độ đẹp của giao diện

### Ngôn ngữ
- Tài liệu, comment giải thích *vì sao*, giao diện, caption VI: **tiếng Việt**
- Code, tên biến, docstring API, `class_id`, caption EN: **tiếng Anh**
- Không dịch `class_id` — chúng là định danh, luôn là `glass_breaking`, không phải `kinh_vo`

---

## 6. Quy ước

| Hạng mục | Quy ước |
|---|---|
| Kích thước file | Code 200–400 dòng thường, 800 tối đa. Tách module khi vượt |
| Kích thước hàm | < 50 dòng |
| Nhãn TSV | `filename\tonset\toffset\tevent_label` — **không thêm cột** |
| Metadata | `{split}_metadata.csv`, join bằng `filename` |
| `event_id` | `EVT_YYYYMMDD_NNNNNN` |
| `class_id` | snake_case tiếng Anh, đúng 16 giá trị trong `docs/taxonomy.md` |
| Model version | `{component}-v{major}.{minor}`, ví dụ `aac-v2.0` |
| Commit | `<type>: <mô tả>` — feat, fix, refactor, docs, test, chore, exp |
| Nhánh | Hiện tại `master`; quy ước đích là nhánh tính năng, không tự đổi tên nhánh |
| Config | YAML trong `ml/configs/`, **không hardcode** hyperparameter trong code |
| Seed | Scaper train 20260914, dev 21260914; split baseline 20260917. Training chưa seed toàn bộ RNG; 3 seed kết quả cuối còn là kế hoạch |
| Secret | Chỉ qua `.env`, không bao giờ vào git. `.env.example` phải luôn cập nhật |
| API response | Envelope `{success, data, error, meta}` |

---

## 7. Lệnh hay dùng

```bash
# Hạ tầng
docker compose up -d                  # khởi động toàn bộ
docker compose logs -f inference      # xem log service
docker compose down                   # dừng, giữ named volumes; không dùng -v nếu cần giữ dữ liệu

# Dữ liệu
.venv/Scripts/python.exe scripts/data_inventory.py
.venv/Scripts/python.exe scripts/verify_ontology.py
.venv/Scripts/python.exe scripts/check_leakage.py
.venv/Scripts/python.exe scripts/scaper_generate.py --split train
.venv/Scripts/python.exe scripts/scaper_generate.py --split dev
.venv/Scripts/python.exe scripts/verify_synthetic.py --split train
.venv/Scripts/python.exe scripts/make_blind_set.py --help    # bộ gán lại mù pilot lần 2
.venv/Scripts/python.exe scripts/agreement.py --help          # tự-nhất-quán lần 1 vs lần 2
# validate_annotations.py, slice_coverage.py: chưa triển khai

# Huấn luyện & đánh giá
.venv/Scripts/python.exe -m ml.training.train_sed --help      # tự ghi manifest + gieo seed
.venv/Scripts/python.exe -m ml.evaluation.eval_sed --run panns_ft_v1
# eval_sed đọc time_pool_blocks từ checkpoint → history → mặc định kèm CẢNH BÁO.

# Training Ops (TRAINING_OPS_PLAN Pha 1–5)
.venv/Scripts/python.exe -m ml.tracking.run_manifest --run panns_ft_v3 --du-lieu hien-tai
.venv/Scripts/python.exe -m ml.evaluation.predictions --run panns_ft_v3 --split dev
.venv/Scripts/python.exe -m ml.evaluation.threshold_sweep --run panns_ft_v3 --split dev
.venv/Scripts/python.exe -m ml.evaluation.error_analysis --run panns_ft_v3 --split dev
.venv/Scripts/python.exe -m ml.evaluation.error_analysis --run panns_ft_v3 --adaptive-postproc
.venv/Scripts/python.exe -m ml.tracking.compare_runs --runs panns_ft_v1 panns_ft_v2 panns_ft_v3
# Grounded AAC: chưa triển khai.

# Chất lượng
.venv/Scripts/python.exe -m pytest tests/ -q
ruff check . && black --check . && mypy .

# DVC / MLflow
# Chưa có dvc.yaml/remote: dvc repro chưa phải lệnh tái lập hoạt động.
# MLflow UI: http://localhost:5000
```

> ⚠️ Windows: `psds_eval` / `aac-metrics` có thể khó build. Nếu vướng → dùng **WSL2**. Ghi kết quả kiểm tra vào "Nợ kỹ thuật" của `PLAN.md` (hạn W1).

---

## 8. Ý tưởng để dành

Ghi vào đây thay vì làm ngay. Xem lại ở W8 nếu còn thời gian.

- Adapter MQTT cho edge device (đang ở cut-list #3)
- Risk scoring học từ feedback thay vì rule-based
- Fusion với video
- Quantization + ONNX cho triển khai edge
- Định vị nguồn âm bằng mảng micro

---

## 9. Giao thức cập nhật

**Cuối mỗi block công việc, làm đúng 5 bước sau:**

1. Cập nhật **§3 Trạng thái hiện tại** — dời việc từ *Đang làm* → *Đã xong*, viết lại 3 việc tiếp theo
2. Tick `[x]` các task đã **nghiệm thu** trong `docs/PLAN.md` (không tick khi mới chỉ "code xong")
3. Ghi một dòng vào **§10 Nhật ký** (định dạng bên dưới)
4. Nếu có quyết định kiến trúc → thêm ADR + một dòng vào **§4 Quyết định đã khoá**
5. Nếu phát hiện việc cần xác minh → thêm vào mục "Nợ kỹ thuật" trong `docs/PLAN.md`

**Cuối tuần, thêm:** đối chiếu nghiệm thu tuần trong `PLAN.md` → cập nhật bảng trạng thái tổng → **nếu trượt thì kích hoạt cut-list ngay**, không dời sang tuần sau.

---

## 10. Nhật ký tiến độ

Định dạng: `YYYY-MM-DD | Wn | Làm được gì | Chặn ở đâu`
Ghi thêm ở **cuối**, không sửa mục cũ. Mỗi mục 1–3 dòng, không viết dài.

```
2026-09-14 | W0 | Chốt taxonomy 15 class + 8 slice + strong label. Chốt stack FastAPI/pgvector/
                  Redis Streams. Chốt kiến trúc Grounded AAC (trunk chia sẻ BEATs+Conformer,
                  3 head, grounded decoding) làm đóng góp nghiên cứu. Tạo project + 3 file nền.
                | Chưa có gì chặn. Việc gấp nhất: nộp đơn MIVIA.

2026-09-14 | W0 | Viết DATA_PLAN.md: ma trận thu thập 15 class, quy trình gán nhãn có máy hỗ trợ
                  4 giai đoạn, khử trùng lặp, chia tập theo source_group_id, Scaper + chuỗi nhân
                  quả (gồm chuỗi âm tính), cổng quyết định D7, ngân sách công người ~41-45h.
                | Rủi ro đã nhận diện: vehicle_crash (phụ thuộc MIVIA Road), explosion (khan hiếm).
```

### 2026-09-14 — D1: xác minh ontology + bắt đầu thu thập

- Tải metadata gốc về `data/reference/`: ontology AudioSet (632 lớp), danh sách 456 lớp có strong label, meta ESC-50.
- Viết `ontology_map.yaml` (15 lớp, **mọi ID đối chiếu file gốc**) + `verify_ontology.py` + 17 test → 0 lỗi.
- Viết `sources.yaml` (10 nguồn, dung lượng/license lấy từ API Zenodo) + `download_sources.py` (chặn dung lượng, checksum MD5, tải tiếp được).
- Tải xong ESC-50 (2000 clip). Sinh `raw_manifest.csv`: 320 clip / 6 lớp / 26.7 phút, có license **theo từng clip** và `source_group_id`.
- 4 phát hiện đổi kế hoạch: AudioSet không có lớp va chạm xe · ESC-50 không dùng được cho `door_slam` · `footsteps` là đi bộ không phải chạy · `fireworks` có 40 clip từ chỉ 16 bản ghi gốc (bằng chứng cho N2).
- Sửa: `Get-PSDrive` báo sai dung lượng ổ (37 GB thay vì 52 GB thật) — suýt bỏ FSD50K dev_audio oan.
- Đang chạy nền: tải `desed_soundbank` (2.42 GB).

### 2026-09-14 (tiếp) — chuẩn hoá + tài liệu gán nhãn

- Cập nhật theo 3 quyết định mới: MIVIA đã nộp đơn (chờ duyệt) · **một người gán nhãn** · ổ đĩa 52 → 95.7 GB.
- Hệ quả lớn nhất: **không tính được kappa liên-người** → chuyển sang test–retest, ngưỡng nâng lên **0.75**, kèm 3 biện pháp bù và **bắt buộc nêu ở phần Hạn chế**. Đã sửa đồng bộ ở DATA_PLAN §8, PLAN.md W3, SYSTEM.md §3/§11.
- `normalize_audio.py` + `preprocessing.yaml`: 16 kHz mono + EBU R128.
- Phát hiện khi hiệu chỉnh ngưỡng méo: **đo trên tín hiệu đã resample là sai cả hai chiều** — đếm mẫu chạm trần thì thổi phồng (138 clip), đếm đoạn cắt phẳng thì bỏ sót (19 clip); đo đoạn cắt phẳng trên **file gốc** mới đúng (119 clip). Ngưỡng đổi từ "tự loại 11%" sang "tự loại 4% + gắn cờ 6% cho người nghe".
- `taxonomy.md` (15 lớp, bảng tra nhanh 14 cặp nhầm lẫn) + `annotation_guideline.md` (onset/offset, quy tắc gộp 0.5 s, hai chế độ bank/gold, kỷ luật phiên làm việc).
- Thêm test khoá ràng buộc `taxonomy.md` ↔ `ontology_map.yaml` phải khớp — test này bắt được ngay 5 cặp nhầm lẫn thiếu trong bảng tra nhanh.

### 2026-09-14 (tiếp 2) — DESED + FSD50K, hai cái bẫy suýt làm hỏng dữ liệu

- **Bẫy 1 — trùng lặp bắc cầu.** DESED có `freesound_13613` và `freesound_34853` là hai id khác nhau nhưng **giống nhau từng byte** (cả 8 đoạn). Chia theo `source_group_id` sẽ xếp chúng hai bên train/test. `build_manifest.py` giờ tính SHA-256 mọi file và ghi `dedup_groups.csv` với cờ `risk=cross_group`. `make_splits.py` **phải gộp nhóm bắc cầu trước khi chia**.
- **Bẫy 2 — phân cấp ontology.** `Gunshot ⊂ Explosion`, `Fireworks ⊂ Explosion`, `Siren ⊂ Alarm`. FSD50K gắn nhãn theo phân cấp nên clip súng nào cũng kèm nhãn Explosion. Luật "hai lớp ⇒ nhiều sự kiện ⇒ loại" làm `gunshot`/`fireworks`/`siren` ra **0 clip**. Sau khi thu gọn về nhãn cụ thể nhất (`scripts/audioset_ontology.py`): gunshot **236**, fireworks **221**. Áp dụng lại khi xử lý AudioSet-strong.
- **PP/PNP của FSD50K là bộ lọc chất lượng cho không** — PP = âm thanh đích nổi trội, đúng thứ foreground bank cần. Biết trước sản lượng từng lớp **trước khi tải 18 GB audio**, chỉ nhờ 34 MB metadata.
- **Sửa hai đánh giá sai của chính mình:** (1) `explosion` không hề khan hiếm — FSD50K có 1388 clip, đánh giá 🔴 cũ dựa trên trực giác chứ không đếm; (2) ngưỡng `min_duration_sec` 0.3 s mâu thuẫn với guideline (100 ms) và cắt mất 43% lớp `object_drop_dishes`.
- Thêm `coverage_report.py` (bảng độ phủ + danh sách cho cổng D7) và tách `write_exclusions` ra `common.py`.

### 2026-09-14/15 — §4.2–4.7: sàng lọc, kiểm định, Scaper, background bank, RIR

- **`auto_screen.py` (PANNs CNN14)**: định tuyến 3 ngả (nhận/loại/hàng đợi) theo §4.2, đề xuất biên onset/offset bằng ngưỡng 50% đỉnh khung. Phát hiện & sửa: clip < 1s làm CNN14 vô nghĩa — **đệm bằng cách lặp** (tile), không đệm im lặng (đo trên 40 clip thật: zero-pad → 37/40 tự loại oan, tile-pad → 2/40). `guard_low_resolution_classes()` cứu các lớp mà tagger mù (ví dụ `shout_yell` 93% tự loại vì PANNs nghe ra Speech/Groan chứ không có nhãn riêng cho quát tháo).
- **`validate_screen.py` (§4.4)**: phiếu kiểm định **mù** — cố ý bỏ điểm số model khỏi phiếu để không mớm câu trả lời. Kiểm định 245 clip → **tỉ lệ lỗi 0.4% (Wilson CI 0.1%–2.3%)**, đạt xa dưới ngưỡng.
- **`scaper_generate.py` (§7)**: 5 lát cắt kiểm soát tỉ lệ (overlap/low_snr/reverb/long_event/causal_chain) + 2 chuỗi nhân quả **âm tính** (dạy model rằng chuỗi sự kiện liên tiếp không mặc nhiên nguy hiểm). Phát hiện quan trọng: tham số `reverb` của Scaper là hiệu ứng SoX tổng hợp, **không phải tích chập RIR** — phải tự tích chập (`apply_rir`) mới đúng yêu cầu DATA_PLAN §7. Sửa lỗi clipping cứng (25 dB SNR + −23 LUFS có thể vượt 0 dBFS) bằng `fix_clipping=True`.
- **`build_background_bank.py` (§5)**: ngưỡng loại 0.15, lấy đỉnh trên toàn bộ khung (không chỉ khung đầu) để không lọt sự kiện Nhóm A ẩn giữa clip nền.
- **`build_rir_bank.py`**: 505 RIR từ OpenSLR SLR28 (bỏ MIVIA, dùng nguồn công khai). Ba bẫy: (1) thư mục "real" trộn 92 file nhiễu — lọc theo `"_rir_" in name`, không theo `startswith`; (2) `simulated_rirs` không có `largeroom` dù README nói có — phải **đo RT60 thật** rồi xếp nhóm, không tin tên thư mục; (3) 73 RIR đo trong buồng tiêu âm (RT60≈0) bị loại, nếu không lát cắt "có vang" sẽ lẫn clip không vang.
- 287 → 300 test qua các đợt này.

### 2026-09-15 — Tách `laughter_cheering`, viết fetcher AudioSet-strong, hai lỗi mồ côi dòng cũ

- **`scripts/fetch_audioset_strong.py`**: tải 3 TSV nhãn strong-label, lọc `PRESENT` khớp 16 lớp, gộp sự kiện theo ổ 10 giây (`segment_id = ytid_windowstart_ms`), tải đúng đoạn bằng `yt-dlp --download-sections`. `build_manifest.adapt_audioset_strong` sinh **nhiều dòng cho một file** — khác mọi adapter khác — vì một ổ có thể chứa nhiều sự kiện chồng lấn, cần cho EOR. **Chưa chạy tải audio thật** (cần cài yt-dlp/ffmpeg, việc của người).
- **Tách `laughter_cheering` → `laughter` + `applause_cheering`** (phương án A do người dùng chọn): tiếng cười một giọng liên tục khác hẳn phổ/nhịp vỗ tay-đám đông. Sửa đồng bộ `ontology_map.yaml`, `sources.yaml`, `scaper_train.yaml` (2 chuỗi nhân quả), `taxonomy.md`/`DATA_PLAN.md`/`SYSTEM.md`.
- **Chạy lại thật** toàn bộ pipeline (không mô phỏng): `build_manifest` (esc50, fsd50k) → `normalize_audio` → `make_splits` → `check_leakage` → `auto_screen` (rescan toàn bộ `foreground_bank_train`, 4444 clip) → `promote_to_bank`. 25 dòng kiểm định cũ của `laughter_cheering` trong `screen_audit.csv` được patch theo checksum, **giữ nguyên** 244 verdict `ok` + 1 `wrong_class` đã có.
- **Hai lỗi im lặng lộ ra khi làm việc này** (xem quy tắc mới ở §5): `build_manifest.py` để lại dòng mồ côi khi đổi ontology (sửa: `drop_stale_rows`) · `auto_screen.py --class-id X` ghi đè toàn bộ `screen_scores.csv` chỉ với điểm của X — **đã xoá sạch điểm mọi lớp khác đã sàng lọc trước đó**, phải rescan lại toàn bộ 4444 clip để phục hồi (sửa: `merge_scored`). `normalize_audio.py` có lỗi tương tự ở `review_flags.csv` (sửa: `source_prefix`).
- Foreground bank sau khi rebuild: **2434/4444 clip** (13/16 lớp có mặt). 300 → 304 test.

### 2026-09-16 — `vehicle_crash_cc` thay MIVIA, duyệt hết hàng đợi, ADR-0005

- **`vehicle_crash_cc`** (huggingface.co/datasets/Titung/car-crash-audio-cc, CC-BY 3.0): người dùng tìm thấy, thay MIVIA Road làm nguồn chính cho `vehicle_crash`. 46 clip / 38 video → adapter mới (`adapt_vehicle_crash_cc`) dùng `video_id` làm `source_group_id`. Chạy thật: 33/46 qua chuẩn hoá (13 clip compilation >30s bị loại tự động) → sàng lọc → 3 tự nhận + 30 vào hàng đợi (PANNs không có nhãn cho va chạm xe, đúng dự đoán).
- **`validate_screen.py --queue`**: gộp `review_queue.csv` vào `screen_audit.csv`, thêm cột `origin` (sample/queue) để KHÔNG tính lẫn hàng đợi vào tỉ lệ lỗi τ đo trên mẫu auto-accept — hai population khác nhau về độ tin cậy, trộn là đo sai đối tượng. Theo yêu cầu người dùng, đổi cách sắp xếp phiếu sang `(class_id, file_id)`, không giữ 245 dòng mẫu cũ ở đầu. Thêm `--stage --all` để chép lại toàn bộ playlist bất kể đã duyệt hay chưa.
- **Quyết định quan trọng nhất của ngày**: người dùng chọn duyệt **hàng loạt** toàn bộ 1831 dòng còn trống trong hàng đợi thành `ok`, KHÔNG nghe từng clip — dựa trên tỉ lệ lỗi 0.4% đo được ở mẫu auto-accept. Đã cảnh báo rõ: tỉ lệ đó đo trên population khác (máy tự tin), không suy diễn được sang hàng đợi (máy không tự tin, có lớp bị guard ghi nhận tagger mù >50%). Người dùng xác nhận vẫn muốn làm sau khi nghe cảnh báo → đã thực hiện, ghi note `bulk_ok_2026-09-16` vào từng dòng để truy vết, coi là **hạn chế phải công bố trong khoá luận**, không phải một phần của con số kiểm định 0.4%.
- Bank sau khi promote (`--include-reviewed`): **4268 clip, 15/16 lớp đạt mức tối thiểu**. Chỉ `shout_yell` (56/80) dưới mức — **ADR-0005**: chốt giữ nguyên, không bổ sung thêm, người dùng xác nhận trực tiếp. `vehicle_crash` (33/30) vừa đủ.
- Viết ADR đầu tiên của dự án: `docs/decisions/ADR-0005-shout-yell-duoi-muc-toi-thieu.md` (số 0005 vì 0001–0004 đã dành cho 4 quyết định kiến trúc W1 chưa viết).

---

*Nếu bạn là một phiên AI mới: đọc xong file này, mở `docs/PLAN.md` tới tuần ghi ở §3, rồi bắt đầu từ "3 việc tiếp theo".*

### 2026-09-17 — Đối soát hiện trạng, tăng tốc Scaper, đồng bộ tài liệu

- Đọc lại mã nguồn, process/log/artifact; sửa các mô tả lỗi thời về Git, AudioSet, bank,
  walking skeleton và v1/v2. Số chốt và bằng chứng ở `docs/STATUS.md`.
- RIR trực tiếp → FFT; replay RNG khi resume, ghi WAV/JAMS qua file pending.
  Giữ nguyên recipe legacy, không tự đánh dấu dữ liệu đạt; giữ nguyên phiên train v2.
- Sao lưu hai cặp clip cuối trước khi resume; giữ các quyết định đã chốt và nhật ký cũ.

### 2026-09-17 (15:05–15:30) — Đối soát lại, phát hiện `long_event` bất khả thi

- **Đối soát lại snapshot 12:33 và sửa bốn chỗ lệch:** v2 đã xong (không phải đang chạy),
  AudioSet 937 (không phải 527), 394 test (không phải 76), và `long_event` sai vì lý do
  sâu hơn recipe. Mọi tiến trình Python/yt-dlp/ffmpeg đã dừng; Docker vẫn 6/6 healthy.
- **v2 hoàn tất:** event-F1 0.1282 → **0.1897** (+48 % tương đối), nhưng segment-F1
  0.4120 → 0.2980 và mAP 0.8372 → 0.8176. Ba yếu tố đổi cùng lúc nên chưa quy được nguyên
  nhân; không quét được ngưỡng vì predictions chưa lưu.
- 🔴 **Đo toàn bộ 4.267 clip bank:** chỉ **90** clip còn ≥8 s sau cắt im lặng, 9/15 lớp
  có **0** clip, `siren` 2/311. `long_event: min_event_duration_sec: 8.0` là điều khoản
  dữ liệu không bao giờ thoả được — **bác bỏ chẩn đoán trước đó** rằng chỉ cần bỏ
  `event_duration=("const", 1.0)`.
- Kiểm chứng `slice_index.jsonl` vẫn khớp code sau khi Codex sửa `scaper_generate.py`:
  **0/7.920 khác biệt**; tỉ lệ 5 lát cắt khớp config ±3 %.
- Người dùng duyệt kế hoạch **bốn bản sửa ①②③④** và quy trình sinh lại theo mốc
  **500 → 1.000 → 1.500**. Ghi ở `docs/STATUS.md` §6–§7.

### 2026-09-17 (tối) — B0–B9, lô dữ liệu mới và baseline v3

- Đo lại phân bố rồi thay kế hoạch bốn sửa bằng B0–B9: chọn source theo clip, nền 10 s,
  duration theo `train_strong`, overlap ép, `long_event` 4 s có giới hạn lớp, mật độ/đặt vào
  khoảng trống, mô phỏng nhãn và seed theo clip. Lô mới 7.920 train + 1.440 dev PASS hợp đồng.
- Precompute waveform PANNs cho cả train/dev; `panns_ft_v3` hoàn tất 25 epoch: mAP 0,8123,
  segment-F1 0,2748, event-F1 0,1077. Không kết luận nguyên nhân trước khi có threshold sweep.

### 2026-09-22 (tiếp 2) — Pilot gold lần 2 (mù) XONG, cổng §8.5 VỪA HỤT — vá lỗi thật trong agreement.py

- **60/60 clip gán mù xong** → `data/gold/pilot_v1_lan2.tsv` (96 dòng sự kiện, 51/60 file
  có sự kiện). 30/30 pilot có kết quả thật (không file pilot nào rơi vào diện "loại"); 6
  clip **mồi** bị đánh giá `nhạc`/`rác` — cross-check với `blind_v1_lan2_mapping.csv` xác
  nhận cả 6 đều là mồi (`la_pilot=False`), không phải pilot, nên không có khủng hoảng nhất
  quán với lần 1. Ghi 25 dòng mới vào `exclusions.csv` (`stage=gold_pilot`,
  `synthetic_or_game_audio`/`unusable`/`music_no_event`) cho 6 WAV đó — `gold_test` còn
  **345** file.
- **`scripts/agreement.py` chạy thật lần đầu → phát hiện một lỗi code thật, không phải
  phát hiện dữ liệu.** `pilot_v1_lan1.tsv` ghi filename CÓ đuôi `.wav` (đúng quy ước TSV —
  tên file WAV thật), còn cột `file_id` của `blind_v1_lan2_mapping.csv` KHÔNG có đuôi
  (đúng quy ước file_id xuyên suốt pipeline). Universe (suy từ mapping) không khớp được
  khoá nào của `lan1` → `lan1` rỗng tuyệt đối → event-F1 VÀ mọi F1 theo lớp ra **NaN**
  thay vì một cổng có ý nghĩa. Vá bằng `bo_duoi_wav()` + một test hồi quy TÁI HIỆN ĐÚNG
  lỗi thật (test cũ dùng id trần ở cả hai phía nên không bắt được mismatch này).
- **Kết quả sau khi vá:** event-F1 tự-nhất-quán **0,7339** (ngưỡng ≥0,75 — **hụt 0,0161,
  KHÔNG ĐẠT**); lệch onset trung vị **10,0 ms** (ngưỡng ≤100ms — ĐẠT thoải mái). 4 lớp NaN
  trong bảng theo-lớp (`fireworks`/`running_footsteps`/`shout_yell`/`vehicle_crash`) khớp
  ĐÚNG 4 lớp vắng mặt ở lần 1 — xác nhận khớp file_id đúng, không phải trùng hợp.
- **Bổ sung `chi_tiet_bat_dong()`** — danh sách ca bất đồng theo `clip_id` (DATA_PLAN
  §8.3 bước 3 / §8.2 B2, khoảng trống đã flag trước khi chạy thật). **13/30 clip pilot có
  ít nhất 1 ca bất đồng**, 21 cặp lệch/thiếu/thừa/thay thế. Ca đáng chú ý nhất:
  `as_strong_0yZGysisqY0_12000` — case "đồ chơi mô phỏng" của lần 1 (không gán sự kiện
  nào) — lần 2 mù lại nghe ra `alarm_bell`+`explosion` thật (3 Insertion tuyệt đối, tức
  toàn bộ nội dung lần 2 với clip này không khớp gì với lần 1).
- **Theo đúng DATA_PLAN §8.3: KHÔNG đạt cổng → KHÔNG đi bước 4 (gán đại trà).** Việc tiếp
  theo: nghe lại lần ba 13 clip bất đồng, chốt, ghi `decision_log.md`, sửa
  `annotation_guideline.md`, rồi cân nhắc một vòng pilot mới trước khi thử lại.
- 689 → 693 test đạt (4 test: hồi quy lỗi `.wav` + `chi_tiet_bat_dong()`).

### 2026-09-22 (tiếp) — Hạ tầng bước 2–3 (bộ gán mù + tự-nhất-quán) và ablation pos_weight

- **`scripts/make_blind_set.py` mới, ĐÃ CHẠY THẬT trên dữ liệu production.** Dựng bộ 60
  clip cho lần gán lại mù: 30 clip pilot (băm tên SHA-256, xáo thứ tự bằng
  `hash_fraction` salt riêng để khối 30 cũ không đứng liền nhau) cộng **30 clip "mồi"**
  lấy từ 351 file `gold_test` CHƯA gán, để không lộ thành một khối dễ nhận ra (DATA_PLAN
  §8.4 điều kiện 2+4 — điều kiện 1 "dự án Label Studio mới" và điều kiện 3 "nghỉ ≥3 ngày"
  là việc của người, script không lo được). Mồi lấy hash-sort đơn giản, KHÔNG round-robin
  theo lớp như `chon_pilot` — mục đích là che giấu, không phải phủ lớp; nhãn của mồi ở
  lần 2 vẫn tận dụng được thẳng cho gán đại trà sau này, không lãng phí công nghe.
  Kết quả thật: `data/gold/blind_v1_lan2/` (60 WAV, 104 MB) +
  `data/gold/blind_v1_lan2_mapping.csv` (ten_mu → file_id thật + cờ `la_pilot`; ⚠️
  **KHÔNG được mở file này trong lúc gán nhãn** — mở ra là lộ ngay đâu là 30 clip cũ).
- **`scripts/agreement.py` mới — chưa chạy thật, chờ `pilot_v1_lan2.tsv`.** Tính tự-nhất-
  quán event-F1 (tái dùng `sed_metrics.event_and_segment_f1`, không tự cài F1) và lệch
  onset trung vị (tái dùng `error_taxonomy.phan_loai`, lấy MỌI cặp có cả hai onset —
  dung/bien/thay_the, không chỉ `dung`, vì giới hạn ở `dung` sẽ tự động loại mọi cặp lệch
  quá collar và làm trung vị nhỏ giả tạo). Báo cáo THEO TỪNG LỚP (§8.2 B3) + hai cổng
  §8.5 (event-F1 ≥ 0,75 · lệch onset trung vị ≤ 100 ms) — hai cổng còn lại (≥20 event/lớp,
  ≥20 clip/slice) áp cho `gold_test` đầy đủ, KHÔNG áp cho batch pilot, script nói rõ điều
  này trong báo cáo thay vì để người đọc tự suy. Mọi thông điệp đều nói rõ đây KHÔNG phải
  kappa liên-người (DATA_PLAN §8.1) — dự án chỉ có một người gán.
- **`train_sed.py` thêm `--pos-weight-max`** (mặc định vẫn `MAX_POS_WEIGHT=30.0`, hành vi
  cũ giữ nguyên — không đổi mặc định ngầm). Trả lời câu hỏi còn treo của phép đo hiệu
  chuẩn 19/09 (ECE v3 0,3332, >500k khung dự báo 0,4–0,6 mà tỉ lệ dương thật chỉ 1,7–3%).
  `--pos-weight-max 1.0` tắt hẳn cân bằng lớp (mọi lớp có dương đều nhận weight=1.0, vì tỉ
  lệ âm/dương thật luôn ≥1 với sự kiện thưa). **Chưa chạy lượt train nào với cờ này** — cần
  quét ít nhất {1.0, 10, 30}, so cả ECE tổng lẫn event-F1@θ* để phân biệt cải thiện thật
  với đánh đổi (ECE giảm nhưng F1 cũng giảm thì không phải thắng).
- 660 → 675 (`make_blind_set.py`, 15 test) → 686 (`agreement.py`, 11 test) → **689 test
  đạt** (`--pos-weight-max`, 3 test).

### 2026-09-22 — Pilot gold lần 1 HOÀN TẤT: 30/30, tỉ lệ thất bại 34,8% đo được

- **30/30 clip pilot đã gán xong**, `data/gold/pilot_v1_lan1.tsv` — 52 dòng sự kiện hợp
  lệ, format sạch (đã kiểm: 0 trùng lặp, đủ tab thật, 3 chữ số thập phân, sắp đúng thứ
  tự, không nhãn lạ).
- **Phải loại và thay thế 16 lần liên tiếp** mới đủ 30 clip dùng được — tổng 46 lượt
  nghe, **tỉ lệ thất bại 34,8%**. Ba mã lý do tách riêng trong `exclusions.csv`
  (`stage=gold_pilot`): `synthetic_or_game_audio` (7 file, game/phim), `unusable` (6
  file, rác/không liên quan), `music_no_event` (3 file, nhạc nền — mã mới thêm khi phát
  hiện case này khác hẳn game/rác).
- **Xác nhận bằng số liệu: 100% file bị loại đều chạm ít nhất một trong 5 lớp
  `glass_breaking`/`scream`/`explosion`/`gunshot`/`siren`.** Súng nổ/bom/kính vỡ/tiếng
  thét THẬT hiếm khi được quay đăng công khai lên YouTube — phần lớn nhãn AudioSet cho
  các lớp này thực ra là game/phim/dàn dựng. **Phải ghi nguyên văn 34,8% vào phần Hạn
  chế của khoá luận**, và cảnh báo trước cho lần gán đại trà (bước 4 §8.3): các lớp này
  sẽ tốn nhiều lượt nghe hơn hẳn mức trung bình để đủ ≥20 sự kiện/lớp ở cổng §8.5.
- **Case mới (khác game/rác):** đồ chơi mô phỏng (`0yZGysisqY0_12000`, âm thanh tương tự
  `gunshot` nhưng là đồ chơi) — không gán vào lớp mục tiêu dù sóng âm giống, cùng nguyên
  tắc đã dùng cho ca chuông quầy phục vụ (`decision_log.md` 15/09). Không loại file, xử
  lý như "không có sự kiện" — file vẫn ở trong `gold_test`.
- **annotation_guideline.md §0.1 mới** — điều kiện tiên quyết "audio có phải thật
  không" phải kiểm TRƯỚC khi gán bất kỳ dòng nào, với dấu hiệu nhận biết cụ thể.
- 4 lớp vắng mặt trong 52 dòng sự kiện (`fireworks`, `running_footsteps`, `shout_yell`,
  `vehicle_crash`) — không phải lỗi, pilot không yêu cầu phủ đều 15 lớp.
- **Việc tiếp theo:** nghỉ ≥3 ngày (điều kiện bắt buộc, không nén được) rồi gán lại 30
  clip này trong điều kiện MÙ (bước 2-3 của §8.3) để tính tự-nhất-quán.

### 2026-09-21 (tiếp 3) — Pilot gold lần 1: 24/30 gán xong, 6 file rác đã loại

- **Người dùng gán tay xong 30/30 file pilot**, ghi vào `data/gold/pilot_v1_lan1.tsv`
  thô — có 6 file đánh nhãn tạm `bad` (rác/không liên quan) và 1 file đánh `nothing`
  (sạch nhưng không có sự kiện nào thuộc 16 lớp, chỉ nhiễu nền).
- **Xử lý dòng `bad`:** không ghi vào TSV — coi là loại cả clip theo N3. Tra
  `raw_manifest.csv` lấy đủ mọi `file_id` (mỗi WAV audioset_strong có nhiều dòng sự
  kiện) cho 6 segment này → **24 dòng exclusion**, ghi vào `exclusions.csv`
  (`stage=gold_pilot`, `reason_code=unusable`). Chạy lại `select_gold_pilot.py` → đúng
  **6 file thay thế mới**, 24 file cũ giữ nguyên (tính chất round-robin + seed ổn định).
- **Xử lý dòng `nothing`:** khác `bad` — clip vẫn hợp lệ, chỉ là không có sự kiện nào
  đáng gán. Theo §5 của guideline: **không ghi dòng nào** vào TSV (không dùng nhãn
  `none`/dòng rỗng). File này ĐƯỢC GIỮ trong pilot, không bị loại.
- **Dọn TSV:** khử 1 dòng trùng lặp y hệt (`object_drop_dishes` lặp ở
  `-99MQ06mPYE_30000.wav`), chuẩn hoá tab thật (file gốc lẫn khoảng trắng và tab),
  3 chữ số thập phân, sắp theo `filename` rồi `onset` — đúng §5. Kết quả: **42 dòng sự
  kiện hợp lệ** cho 23 file (24 file trừ 1 file `nothing`).
- **Còn lại:** gán 6 file thay thế trong `pilot_v1_candidates.csv` (đã đổi so với bản
  gán lần đầu) để đủ 30, rồi mới sang bước 2 của §8.3.

### 2026-09-21 (tiếp 2) — AudioSet-strong vào manifest/split, chọn pilot gold

- **`build_manifest.py --source audioset_strong`** — adapter đã có sẵn từ trước
  (`adapt_audioset_strong`, `source_group_id=youtube_{ytid}` để cả video luôn đi cùng một
  split), chỉ chưa ai chạy. 937 file → **3.977 dòng sự kiện** (một ổ 10s nhiều event), 0
  trùng lặp với nguồn khác đã quét.
- **`make_splits.py`** — **1.601 dòng sự kiện / 367 file WAV riêng biệt vào `gold_test`**
  (2.376/570 vào `dev`), khớp đúng tỉ lệ 0,6/0,4 của `splits.yaml`. Mọi lớp trong 15 lớp
  đều có ≥11 file trong `gold_test` (thấp nhất: `door_slam`).
- **`scripts/select_gold_pilot.py` mới** — round-robin theo lớp (không random thuần), có
  seed cố định để tái lập. Chọn ngẫu nhiên thuần trên 367 file sẽ để `speech_normal`
  (210/367) lấn át `door_slam` (11/367) trong pilot. Cột `classes_tham_khao` trong output
  CHỈ để chọn mẫu, không được nhìn lúc gán nhãn thật (annotation_guideline.md §7.2 cấm máy
  đề xuất ở chế độ GOLD).
- Kết quả: `data/gold/pilot_v1_candidates.csv` — 30 file, đủ 15 lớp (2–23 file/lớp,
  `speech_normal` nhiều nhất vì hay xuất hiện nền trong các event khác). **Chưa có nhãn**
  — đây là danh sách để bắt đầu DATA_PLAN §8.3 bước 1 (gán tay), không phải gold_test.
- 649 → **654 test đạt**.

### 2026-09-21 (tiếp) — cổng hợp đồng chặn thật, checkpoint resume đầy đủ

- **`kiem_cong_hop_dong()` mới trong `train_sed.py`** — chặn train khi hợp đồng dữ liệu
  khác `PASSED`, kể cả `KHONG_RO`. Cổng cũ chỉ in cảnh báo cho cả `FAILED` lẫn `KHONG_RO`,
  y hệt v1/v2 (lô legacy) đã train trót lọt. `--force-du-lieu-chua-dat` bỏ qua được, và cờ
  đó **ghi vào `manifest.json`** — một cổng có thể tắt im lặng thì không phải cổng.
- **`luu_checkpoint_resume()`/`nap_checkpoint_resume()` mới** — `checkpoint_resume.pt` ghi
  đè MỖI epoch (model/optimizer/scheduler/scaler + RNG bốn nguồn), tách khỏi `best.pt` để
  không đổi schema mà `predictions.py`/`eval_sed.py` đọc. `--resume` tiếp tục từ epoch vừa
  xong, khôi phục RNG nên dãy số ngẫu nhiên tiếp theo giống hệt như chưa hề dừng — test
  đơn vị xác nhận bằng cách so hai nhánh (tiếp tục tại chỗ vs reseed rồi nạp lại).
- Test đơn vị thuần, không chạy train thật (25 epoch tốn GPU nhiều giờ).
- 642 → **649 test đạt**. File mới: `tests/test_ml_train_sed.py`.

### 2026-09-21 — `long_event`: ứng viên cuối cùng cũng bị bác bỏ

- **`ml/evaluation/long_event_gap.py` mới** — kiểm ứng viên cuối "Scaper sinh khe hở năng
  lượng thật bên trong sự kiện dài". So nhóm sự kiện bị phân mảnh (n=178) với nhóm không
  (n=137), CÙNG lát cắt `long_event`, chỉ xét sự kiện ≥1s (315/559 sự kiện). Khe hở trung
  bình **bằng nhau tuyệt đối** ở cả hai nhóm (0,012s ở ngưỡng "im" 20%, 0,046s ở 40%); p90
  nhóm KHÔNG phân mảnh còn cao hơn — ngược hướng giả thuyết. Đã kiểm độ nhạy ngưỡng, kết
  luận không đổi ở cả hai mức.
- **`long_event` giờ đã loại BA ứng viên liên tiếp** (cửa sổ hẹp, trần thấp, khe hở năng
  lượng). Nguyên nhân riêng vẫn chưa xác định — không còn ứng viên thứ tư nào chờ sẵn.
- 631 → **642 test đạt**. Số đo: `docs/measurements/long_event_gap_20260921.md`.

### 2026-09-19 (tối, tiếp) — trần cửa sổ bị bác bỏ, rò rỉ đo bằng 0, hiệu chuẩn xác suất

- **Trần `MAX_MEDIAN_FRAMES=51` không phải nguyên nhân chính của `long_event`.** Quét
  `--max-median-frames` ∈ {51,101,201,401}: `v2` gap **rộng ra** (0,1322→0,1463), `v3`
  chỉ hẹp **6,4%** dù trần nới gần gấp 8 lần. Insertion giảm thật (v3: 504→442) nhưng
  Đúng cũng giảm và Deletion tăng — đánh đổi, không phải thắng thuần. Cột "gộp" đứng yên
  tuyệt đối qua cả 4 mức trần → cảnh báo "nới trần sẽ gộp nhầm" ở `sed_metrics.py:37`
  vẫn chưa có số đỡ.
- **`cua_so_loc_tu_train()` mới** — cửa sổ thích ứng suy từ `data/features/train_meta.json`
  trực tiếp (không nhận `du_doan` nào), thay cho `cua_so_loc()` vốn suy từ chính tập đang
  chấm. Độ lớn rò rỉ đo được **= 0,000000** ở cả v2/v3 — nhưng vì 9/15 lớp đã kẹp trần 51
  ở cả hai nguồn, không phải vì train/dev giống hệt nhau; chỉ đúng khi trần còn hiệu lực.
- **`ml/evaluation/calibration.py` mới** — ECE + reliability diagram, đo lần đầu vì sao
  θ* luôn rơi 0,85–0,95. Vùng dự báo 0,4–0,6 mang **hơn nửa triệu khung** (vehicle_crash,
  v3) mà tỉ lệ dương thật chỉ 1,7–3%. Không so ECE giữa các run như so chất lượng model:
  v1 (0,1041) thấp hơn v3 (0,3332) vì xác suất gần hằng số của nó hiếm khi đẩy ra thái
  cực, không phải vì hiệu chuẩn tốt hơn.
- 616 → **631 test đạt**. Số đo: `docs/measurements/max_median_ceiling_20260919.md`,
  `docs/measurements/adaptive_window_leak_20260919.md`,
  `docs/measurements/calibration_20260919.md`.

### 2026-09-19 (tối) — Pha 5, ablation hậu xử lý, và ontology khớp số đo

- **Pha 5 xong:** `ml/tracking/compare_runs.py` + 11 test. Chạy trên v1/v2/v3 thì cả dữ
  liệu lẫn mã nguồn đều ra `KHONG_RO` → **không có phép so sánh sạch nào giữa ba run hiện
  có**. Ba cái bẫy được cài test riêng: vân tay `null` không phải "giống nhau"; manifest
  hồi cứu thì băm mã trùng nhau là vô nghĩa; `ket_qua_cuoi` chấm trên val riêng từng run
  nên xếp hạng theo nó ra **ngược** dev chung (v2 > v1 > v3).
- **`bang_lat_cat()` xuất đủ 6 loại lỗi cho từng lát cắt**, đếm lại trên tập con chứ không
  chia tỉ lệ từ tổng. `--adaptive-postproc` giờ ghi ra `analysis_adaptive.{json,md}`, không
  đè báo cáo mặc định nữa.
- **`long_event` hỏng vì phân mảnh (1,9× mốc), không phải bỏ sót** — Deletion của nó còn
  thấp hơn `low_snr`. Ablation cửa sổ lọc **bác bỏ một phần** giả thuyết cũ: cửa sổ theo
  lớp nâng v3 0,3979 → 0,4347 nhưng nâng *mọi* lát cắt gần như cùng một lượng, khoảng cách
  tới mốc sạch không đổi (0,180 → 0,178). v1 **không đổi một con số nào** ở cả hai cấu
  hình vì xác suất của nó hằng số trên mảng ~32 khung — hệ quả thứ hai của lưới 319,7 ms.
- **8 cặp nhầm đo được vào `confusable_with`** (đối xứng) + 8 hàng câu hỏi quyết định vào
  bảng tra §3 của `taxonomy.md`. Tỉ lệ lượt nhầm đã khai của v3: 31% → **54,1%**. 7 cặp chỉ
  chạm ngưỡng ở đúng một run thì **không** khai — trường này điều khiển luật định tuyến
  sang người gán nhãn, nhầm-của-model không đồng nghĩa nhầm-của-người.
- 601 → **616 test đạt**. Số đo: `docs/measurements/long_event_postproc_20260919.md`.

### 2026-09-19 (chiều) — Training Ops Pha 4: phân tích lỗi

- **`ml/evaluation/error_taxonomy.py` + `error_analysis.py`.** Đọc dự đoán đã lưu, chấm
  trên CPU, ~3 phút/run. Xuất `ml/runs/{run}/analysis.{json,md}`.
- **Ranh giới cố ý với `sed_eval`:** S/D/I lấy từ `sed_eval` (chân lý); phép ghép cặp
  ref↔pred tự viết chỉ để dựng Boundary/Fragmentation/Merge và ma trận nhầm — vì
  `sed_eval` chỉ trả **tỉ lệ**, không phơi ra cặp nào khớp cặp nào. Đối chiếu tường minh
  trên v3: số khớp đúng **2.901 vs 2.901** (θ=0,50) và **2.362 vs 2.362** (θ=0,90), lệch
  +0. Không có phép đối chiếu này thì bảng phân loại lỗi không đứng được.
- **θ = 0,5 hỏng theo kiểu nào, đã có số.** v3 ở θ=0,5 bỏ sót **9 trên 4.873** sự kiện
  (0,2%) nhưng chèn thêm **61.020** — 12,52 chèn trên mỗi sự kiện thật. Lỗi nằm ở ngưỡng
  và hậu xử lý, KHÔNG ở năng lực phát hiện. Việc cần làm vì thế không phải thêm dữ liệu
  cho lớp hiếm.
- **Một phần lỗi biên của v1 là trần cứng, đo được chứ không suy luận.** Onset dự báo của
  v1 rơi **100%** trên lưới 319,7 ms (31 đoạn); onset lấy tại đầu đoạn hoạt động đầu tiên
  nên sai số lượng tử phân bố đều trong `[0, bước)`, cho trần **37,4%** số cặp không thể
  lọt collar 200 ms dù model đoán hoàn hảo. Quan sát 49,5%. v2/v3 lưới 79,9 ms → trần 0%,
  quan sát 29%. **Hệ quả: so event-F1 của v1 với v2/v3 là so một phần độ phân giải bộ
  giải mã, không thuần chất lượng model.** mAP mức clip không dính vì không xét vị trí.
- **θ riêng từng lớp KHÔNG phải cải tiến mặc định** — nó làm v1 tệ đi (0,2953 → 0,2897).
  F1 sự kiện tổng của `sed_eval` là micro-average, nên argmax F1 của từng lớp riêng không
  tối đa hoá con số gộp. Lợi ở v2/v3 (+0,006 / +0,018) cũng là chặn trên chọn trên chính
  tập chấm.
- **Giả định thiết kế bị số đo bác một phần:** chỉ **31%** lượt nhầm lớp rơi vào cặp đã
  khai trong `confusable_with`. `running_footsteps` là trung tâm nhầm lẫn, hút nhầm từ
  `fireworks`/`door_slam`/`gunshot`/`object_drop_dishes` và bắn nhầm ngược lại.
- **Lát cắt:** thứ tự khó giống nhau ở cả ba run nên là tính chất của **dữ liệu**.
  `long_event` tệ nhất (v3 0,2685 so với mốc clip sạch 0,4487); `causal_chain` **không**
  khó hơn clip sạch bao nhiêu, ngược giả định lúc thiết kế lát cắt.
- 580 → **601 test đạt**. Chi tiết: `docs/measurements/error_analysis_20260919.md`.

### 2026-09-19 — Training Ops Pha 1 + Pha 3, và một kết luận bị bác bỏ

- **Pha 1:** `ml/tracking/fingerprint.py` + `run_manifest.py` — vân tay ba tầng (nhãn /
  waveform SHA-256 / thống kê độ dài theo lớp), vân tay mã, git dirty diff, env, hợp đồng
  dữ liệu. `train_sed.py` gieo toàn bộ RNG (trước đó `--seed` chỉ chia train/val) và ghi
  manifest **trước** epoch đầu. Manifest hồi cứu cho v1/v2 (vân tay `null`, hợp đồng
  FAILED) và v3 (vân tay đầy đủ, PASSED).
- **Lỗi bắt được ngay lần chạy đầu của chính module mới:** hồi cứu v1/v2 đọc mặc định
  `synthetic_contract.csv` — file đang ghi lô B0–B9 PASS — nên khai **PASSED** cho hai run
  train trên lô legacy đã FAILED. Đã sửa thành `KHONG_RO` + `--hop-dong`, kèm test.
- **Pha 3:** `predictions.py` lưu xác suất mức **đoạn** (float16); mức khung là bản lặp
  nguyên xi nên lưu khung tốn gấp 32 lần mà không thêm bit nào — có test khẳng định dựng
  lại khớp **từng phần tử** ở cả pool 5 và pool 3. 1,3–4,6 MB/run cho 1.440 clip.
- **`threshold_sweep.py` bác bỏ một kết luận đang nằm trong tài liệu.** Chấm lại cả ba run
  trên cùng dev 1.440 clip: θ=0,5 sai ở cả ba (v3 **0,0820 → 0,3986**, 4,86×), và thứ tự
  v2/v3 **đảo lại**. "v3 kém hơn v2 ở cả ba chỉ số" là hiện vật của một ngưỡng cố định,
  không phải của model. Confound còn nguyên: dev cùng recipe với train của v3.
- **Một lỗi bậc hai đã sửa:** `event_and_segment_f1` gọi `MetaDataContainer.filter` cho
  từng clip — quét tuyến tính, O(clip × sự kiện). Đo: filter **101,5 s** so với evaluate
  **4,7 s**; gom một lượt đưa một lần chấm từ **102,3 s xuống 5,2 s**, điểm số giống hệt.
- **Lỗi thứ hai, tìm ra khi kiểm chứng chính phần mới viết:** `PrecomputedSedDataset` giữ
  `np.memmap` làm thuộc tính, nên `DataLoader(num_workers>0)` pickle **toàn bộ nội dung**
  sang từng worker — đo: pickle dev **921,8 MB → 0,159 MB** sau khi mở lười. v1/v2/v3 đều
  train ở `--workers 2` nên đều đã trả giá này; triệu chứng là crash KHÔNG ĐỀU
  (`pickle data was truncated`), đã đánh hỏng hai lượt chạy trong phiên.
- 550 → **580 test đạt**. Chi tiết số đo: `docs/measurements/threshold_sweep_20260919.md`.

### 2026-09-18 — Dọn artifact và đồng bộ tài liệu

- Xoá log/cache tạm, recovery trùng và stderr legacy; giữ JAMS legacy, stdout chứng minh lỗi,
  feature cache hiện hành và checkpoint cục bộ. Docker Compose được dừng không xoá volume.
- Đối chiếu lại artifact thật, sinh lại `docs/data_inventory.md`, rồi cập nhật trạng thái/docs
  theo lô B0–B9 và v3. Chưa chỉnh `.gitignore` hay push trong block tài liệu này.
