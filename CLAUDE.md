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
| `docs/evaluation_protocol.md` | Metric và cách đo | Khi làm evaluation |
| `docs/data_inventory.md` | Số giờ/class (sinh tự động) | Khi lo về dữ liệu |
| `docs/decisions/ADR-*.md` | Vì sao chọn thế này | Khi định thay đổi kiến trúc |

---

## 3. Trạng thái hiện tại

**Tuần:** W1 (15–21/09/2026) · **Trạng thái:** 🟢 đang thu thập dữ liệu (D1)
**Cập nhật lần cuối:** 2026-09-15

### ✅ Đã xong
- Chốt taxonomy **16 class** (10 an ninh + **6** nhầm lẫn, xem thay đổi 15/09 dưới) và 8 test slice
- Chốt strong label + schema DCASE TSV
- Chốt kiến trúc Grounded AAC (BEATs❄️ + Conformer trunk chia sẻ, 3 head, grounded decoding)
- Chốt stack: FastAPI toàn bộ · PostgreSQL+pgvector · Redis Streams · MLflow · DVC · Docker Compose
- Viết 4 tài liệu nền: `SYSTEM.md`, `PLAN.md`, `DATA_PLAN.md`, `CLAUDE.md`
- Tạo cấu trúc thư mục
- Chốt quy trình dữ liệu: máy hỗ trợ gán nhãn cho foreground bank, **gán mù cho gold test set**
- **D1 — xác minh ontology xong**: `ml/configs/ontology_map.yaml` (16 lớp, mọi ID đối chiếu file gốc AudioSet) + `scripts/verify_ontology.py` + test → 0 lỗi
- **D1 — đăng ký nguồn xong**: `ml/configs/sources.yaml` (11 nguồn, dung lượng & license lấy từ API Zenodo) + `scripts/download_sources.py`
- Dựng môi trường: `.venv`, `requirements-data.txt`, `requirements-screen.txt` (torch + panns-inference), `.gitignore`
- **Chuẩn hoá kỹ thuật xong cho ESC-50 + FSD50K**: `normalize_audio.py` + `preprocessing.yaml`
- **Viết xong `taxonomy.md` + `annotation_guideline.md`** — hai tài liệu chặn đường găng pilot gán nhãn
- **Adapter DESED, FSD50K, ESC-50, UrbanSound8K xong** — phát hiện và ghi lại 8 nhóm trùng **bắc cầu** giữa hai id Freesound (DESED)
- **§4.2–4.4 DATA_PLAN xong**: `auto_screen.py` (PANNs CNN14) + `validate_screen.py` (phiếu kiểm định mù) → **kiểm định 245 clip, tỉ lệ lỗi auto-accept 0.4% (Wilson CI 0.1%–2.3%)**
- **§5 xong**: `build_background_bank.py` → background bank đã sàng lọc bằng ngưỡng 0.15
- **§6 xong**: `make_splits.py` + `check_leakage.py` (gộp nhóm bắc cầu trước khi chia, ràng buộc theo nguồn cho `gold_test`)
- **§7 xong**: `scaper_generate.py` — sinh soundscape, 5 lát cắt (overlap/low_snr/reverb/long_event/causal_chain), tích chập RIR thật (không dùng SoX reverb)
- **Bank RIR xong**: 505 RIR từ OpenSLR SLR28 (`build_rir_bank.py`), xếp nhóm theo RT60 đo được: small_room 151 · medium_room 198 · large_room 156
- **`scripts/fetch_audioset_strong.py` viết + kiểm thử xong** (yt-dlp) — nguồn duy nhất cho `dev`/`gold_test` không phụ thuộc MIVIA/thu thực địa. **Chưa chạy tải audio thật** — xem §Nợ kỹ thuật ở PLAN.md
- **15/09 — tách `laughter_cheering` → `laughter` + `applause_cheering`** (16 lớp), chạy lại toàn bộ pipeline thật, giữ nguyên 245 verdict kiểm định đã có (xem log §10)
- Foreground bank hiện tại: **2434 clip / ~92 phút** đã qua sàng lọc + duyệt tự động, 13/16 lớp có mặt. Còn thiếu hẳn: `vehicle_crash` (chờ MIVIA), yếu: `shout_yell`, `object_drop_dishes`, `door_slam`, `explosion`, `running_footsteps`
- 304 test, tất cả đạt (`pytest tests/ -q` — **chạy trong `tests/`, không chạy ở gốc repo vì ESC-50 tự mang theo `tests/test_dataset.py` gây lỗi collection**)

### 🔄 Đang làm
- Tải nền: `urbansound8k` · `fsd50k_dev_audio` (18.4 GB) · `tau2019_partial` — kiểm tra tiến trình cũ trước khi chạy lại (xem cảnh báo khoá `.part.lock` ở lịch sử §10 14/09)
- 1802-clip `review_queue.csv` — **trì hoãn có chủ đích**, chờ ổn định taxonomy (đợt tách lớp 15/09) rồi mới nghe hàng loạt, tránh nghe lại hai lần

> ⚠️ **Phải gọi `.venv/Scripts/python.exe`, không phải `python` trần.** Trong Git Bash,
> `python` trỏ vào Python hệ thống — script chết ngay ở `import yaml` và trông hệt như
> lệnh tải "tự dừng".

### ⏭️ 3 việc tiếp theo
1. 👤 **Cài `yt-dlp` + `ffmpeg`, chạy `fetch_audioset_strong.py` thật** — gỡ chặn `dev`/`gold_test` đang rỗng, cổng D7 chưa qua được. Dự kiến hụt 15–30% video, phải ghi tỉ lệ thật vào exclusions.csv và báo cáo
2. 👤 **Pilot gán nhãn 30 clip lần 1** (trên `gold_test` sau khi có audio) — nghỉ ≥3 ngày rồi gán lại mù mới tính tự-nhất-quán
3. Tải xong `urbansound8k`/`tau2019_partial`/`fsd50k_dev_audio` → chạy `build_manifest.py`/`normalize_audio.py` cho các nguồn còn lại → bù các lớp yếu (`shout_yell`, `door_slam`, `object_drop_dishes`, `explosion`, `running_footsteps`)

Song song: thu thực địa tại IUH (buổi 1: RIR hành lang/nhà xe/xưởng cho S3 + `gold_test`; buổi 2: `media_playback` cho S4 và kiểm tra phổ pháo hoa VN).

### ⛔ Đang bị chặn
- `vehicle_crash` — ⏳ đã nộp đơn MIVIA Road 14/09, chờ duyệt 1–2 tuần. Không có đường vòng.
- `dev`/`gold_test` — rỗng, chặn ở việc 👤 cài yt-dlp/ffmpeg (không phải việc AI làm được), xem việc tiếp theo #1.

---

## 4. Quyết định đã khoá

Không thảo luận lại trừ khi có lý do mới. Mỗi thay đổi phải kèm một ADR.

| Quyết định | Chọn | Ngày | Lý do ngắn |
|---|---|---|---|
| Số class | **16** (10 an ninh + 6 nhầm lẫn) | 14/09, sửa 15/09 | 6 lớp nhầm lẫn kiểm soát False Alarm Rate. **15/09:** tách `laughter_cheering` → `laughter` + `applause_cheering` — tiếng cười một giọng khác hẳn phổ/nhịp vỗ tay-đám đông |
| Loại nhãn | **Strong label** (onset/offset) | 14/09 | Điều kiện cần cho event-based F1 **và** cho grounding |
| Trường hợp đặc biệt | 5 lớp nhầm lẫn **trong** taxonomy + **8 test slice** | 14/09 | Slice là lát cắt đánh giá, không phải class |
| Bối cảnh | **Tổng quát cho cả 4 khu vực** (trường học, bãi xe, dân cư, nhà máy) | 14/09 | `location` là metadata, không phải tham số model |
| Backend | **FastAPI toàn bộ** (không Django) | 14/09 | Async-native cho streaming; một framework cho 3 service |
| Vector DB | **pgvector** (không Qdrant) | 14/09 | Lọc metadata + vector search trong **một** câu SQL |
| Broker | **Redis Streams** (không Kafka) | 14/09 | Broker thật, zero ops thêm; Kafka là over-engineering ở 1 node |
| Audio encoder | **BEATs đóng băng**, precompute `.npy` | 14/09 | Feature mức frame + chi phí train khả thi trong 8 tuần |
| Captioning | **Grounded AAC tự xây**, không dùng API sẵn | 14/09 | Đây là đóng góp nghiên cứu, không phải tính năng |
| GPU | **Có** | 14/09 | Cho phép train Conformer + BART |
| LLM cho RAG | Interface pluggable, mặc định API, fallback Ollama | 14/09 | Không khoá cứng nhà cung cấp |
| Caption song ngữ | Sinh EN (benchmark) + dịch VI (RAG), embed bản **VI** | 14/09 | Truy vấn tiếng Việt, tránh sụt retrieval |
| Strong label cho train | **Scaper sinh tự động** từ foreground bank | 14/09 | Strong label miễn phí, chính xác tuyệt đối, điều khiển được slice |
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
| Nhánh | `main` + nhánh tính năng; không commit thẳng lên `main` |
| Config | YAML trong `ml/configs/`, **không hardcode** hyperparameter trong code |
| Seed | Mặc định 42; kết quả cuối chạy 3 seed, báo cáo mean ± std |
| Secret | Chỉ qua `.env`, không bao giờ vào git. `.env.example` phải luôn cập nhật |
| API response | Envelope `{success, data, error, meta}` |

---

## 7. Lệnh hay dùng

```bash
# Hạ tầng
docker compose up -d                  # khởi động toàn bộ
docker compose logs -f inference      # xem log service
docker compose down -v                # dọn sạch (mất volume!)

# Dữ liệu
python scripts/data_inventory.py           # → docs/data_inventory.md
python scripts/validate_taxonomy.py
python scripts/validate_annotations.py --split test
python scripts/check_leakage.py
python scripts/agreement.py --split test   # tự-nhất-quán test-retest + onset MAE
python scripts/slice_coverage.py --split test
python scripts/scaper_generate.py --config ml/configs/scaper_train.yaml

# Huấn luyện & đánh giá
python -m ml.training.train --config ml/configs/sed_baseline.yaml
python -m ml.training.train --config ml/configs/aac_proposed.yaml
python -m ml.evaluation.run_all --model <run_id> --split test

# Chất lượng
pytest -q --cov=. --cov-report=term-missing
ruff check . && black --check . && mypy .

# DVC / MLflow
dvc repro && dvc push
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

---

*Nếu bạn là một phiên AI mới: đọc xong file này, mở `docs/PLAN.md` tới tuần ghi ở §3, rồi bắt đầu từ "3 việc tiếp theo".*
