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
**Đối soát workspace:** 19/09/2026, sau khi Pha 1 + Pha 3 của Training Ops và lượt quét
ngưỡng đã hoàn tất.
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
- **Pha 1 + Pha 3 của [TRAINING_OPS_PLAN](docs/TRAINING_OPS_PLAN.md) đã xong (19/09):**
  `ml/tracking/{fingerprint,run_manifest}.py`, `ml/evaluation/{predictions,threshold_sweep}.py`.
  `train_sed.py` gieo toàn bộ RNG và ghi `manifest.json` trước epoch đầu. Pha 4/5 chưa có.
- **580 test đạt** (19/09, chạy đầy đủ `pytest tests/ -q`). Mốc trước thay đổi là 573.

### Đang làm / chưa nghiệm thu

- 🔴 **`long_event` là điều khoản BẤT KHẢ THI với bank hiện có** — phát hiện 17/09 15:00.
  Chỉ **90/4.267** clip bank còn ≥8 s sau khi cắt im lặng; 9/15 lớp có **0** clip;
  `siren` chỉ **2/311** (p90 4,01 s, do UrbanSound8K cắt sẵn ở 4 s). Vì thế lô mới dùng
  ngưỡng 4 s và chỉ các lớp cấp được; bảng đo và giới hạn ở STATUS §6–§7.
- AudioSet-strong: **937 WAV / 937 dòng segments** — khớp tuyệt đối. Process tải **đã dừng**,
  chưa hết hàng đợi. Raw mới chưa vào manifest/split → **real dev/gold chưa sẵn sàng**.
- F1 bằng 0 tại epoch không full-eval là **chưa đo**, không phải F1 thực. Tất cả số trong
  bảng trên lấy từ epoch 25 có full-eval.
- CI, DVC pipeline, BEATs–Conformer–BART, grounded decoding, streaming Redis Streams,
  temporal aggregator, gold G1–G4 và tracking đầy đủ vẫn chưa xong.
- Inference đang dùng PANNs pretrained; **chưa deploy v1/v2/v3**. MLflow healthy không có nghĩa
  training đã log lên MLflow. `AUDIO_RETENTION_DAYS` chưa có tác vụ tự xoá audio hết hạn.
- Checkpoint/model weight `.pt/.pth/.ckpt/.onnx` đã được Git-ignore và giữ tại máy.
  `data/synthetic_legacy/` được đưa lên có chủ đích làm bằng chứng trước-sửa; JAMS legacy chứa
  đường dẫn tuyệt đối lịch sử của máy sinh dữ liệu nên dùng để audit, không replay portable nguyên trạng.

### Ba việc tiếp theo

1. **Pha 4 của Training Ops** — chạy được ngay trên CPU vì prediction đã có sẵn cho cả ba
   run: phân loại lỗi (chèn/sót/nhầm lớp/lệch biên), ma trận nhầm lẫn 15×15 đối chiếu
   `confusable_with`, bảng F1 **theo lát cắt** (join `slice_index.jsonl` theo `clip_id`),
   và đường cong ngưỡng **theo từng lớp** — khoảng cách theo lớp hiện rất rộng.
2. **Đưa 937 AudioSet-strong WAV vào manifest/split không rò rỉ**, chuẩn bị real dev và
   gold; sau đó pilot gán mù theo DATA_PLAN §8. Đây là thứ DUY NHẤT gỡ được confound
   "dev cùng recipe với train của v3". Không dùng test để chọn threshold.
3. **Nâng cổng hợp đồng dữ liệu từ cảnh báo lên chặn** trong `train_sed.py`, và đo hiệu
   chuẩn xác suất (reliability diagram + ablation `pos_weight`) — cả ba run đạt đỉnh ở
   θ≈0,9 là triệu chứng chưa được giải thích bằng số đo.

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
# validate_annotations.py, agreement.py, slice_coverage.py: chưa triển khai

# Huấn luyện & đánh giá
.venv/Scripts/python.exe -m ml.training.train_sed --help      # tự ghi manifest + gieo seed
.venv/Scripts/python.exe -m ml.evaluation.eval_sed --run panns_ft_v1
# eval_sed đọc time_pool_blocks từ checkpoint → history → mặc định kèm CẢNH BÁO.

# Training Ops (TRAINING_OPS_PLAN Pha 1 + Pha 3)
.venv/Scripts/python.exe -m ml.tracking.run_manifest --run panns_ft_v3 --du-lieu hien-tai
.venv/Scripts/python.exe -m ml.evaluation.predictions --run panns_ft_v3 --split dev
.venv/Scripts/python.exe -m ml.evaluation.threshold_sweep --run panns_ft_v3 --split dev
# Grounded AAC, Pha 4 (phân tích lỗi) và Pha 5 (so run): chưa triển khai.

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
