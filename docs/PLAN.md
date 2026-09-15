# PLAN.md — Master Plan 8 tuần

> **Đề tài:** Grounded Automated Audio Captioning for Security Surveillance and RAG-based Alert Retrieval
>
> **Vai trò file này:** kế hoạch thực thi — *làm gì, khi nào, nghiệm thu thế nào*.
> Đặc tả hệ thống ở [SYSTEM.md](SYSTEM.md). Trạng thái phiên làm việc ở [../CLAUDE.md](../CLAUDE.md).
>
> **Quy ước:** tick `[x]` khi task **đã nghiệm thu** (không phải khi "code xong"). Mỗi tuần có tiêu chí nghiệm thu cứng — không đạt thì **không** sang tuần sau mà kích hoạt cut-list.

---

## Bảng trạng thái tổng

| Tuần | Khoảng ngày | Chủ đề | Trạng thái | Nghiệm thu |
|---|---|---|---|---|
| **W1** | 15–21/09/2026 | Nền tảng + Walking Skeleton | 🔜 chưa bắt đầu | ☐ |
| **W2** | 22–28/09/2026 | Dữ liệu + Scaper + Precompute | 🔜 | ☐ |
| **W3** | 29/09–05/10 | 4 Gold set + SED baseline | 🔜 | ☐ |
| **W4** | 06–12/10 | AAC baseline B1 + hạ tầng metric | 🔜 | ☐ |
| **W5** | 13–19/10 | **Grounded AAC (P)** — đóng góp chính | 🔜 | ☐ |
| **W6** | 20–26/10 | Ablation + báo cáo theo slice | 🔜 | ☐ |
| **W7** | 27/10–02/11 | Streaming + RAG + Dashboard | 🔜 | ☐ |
| **W8** | 03–09/11 | MLOps + E2E eval + bản nháp báo cáo | 🔜 | ☐ |

Ký hiệu: 🔜 chưa bắt đầu · 🔄 đang làm · ✅ xong · ⚠️ trễ · ⛔ bị chặn

**Phân vai:** 👤 = việc của người (AI không thay được) · 🤖 = AI làm chính · 👥 = cùng làm

---

## Quan hệ phụ thuộc

```
W1 docs+skeleton ──┬──► W2 data ──► W3 gold ──┬──► W4 B1 ──► W5 P ──► W6 ablation ─┐
                   │                          │                                     │
                   │                          └──► W3 SED baseline                  │
                   └──────────────────────────────────────► W7 streaming+RAG ───────┴──► W8
```

**Đường găng: W2 → W3 → W4 → W5 → W6.** Gán nhãn (W2–W3, 👤) là nút thắt duy nhất AI không gỡ được → **phải khởi động song song từ W2, không đợi W2 xong**.

W7 chạy được tương đối độc lập vì đã có walking skeleton từ W1 — nếu W6 trễ, W7 vẫn tiến hành được.

---

# W1 · 15–21/09 · Nền tảng + Walking Skeleton

**Mục tiêu:** có một hệ thống chạy được đầu-cuối *ngay tuần đầu*, dù mọi thành phần đều tạm. Đây là chống rủi ro quan trọng nhất — tránh thảm hoạ "tuần 8 mới ráp".

### Tasks

| ☐ | Task | Ai | Ghi chú |
|---|---|---|---|
| ☐ | **Nộp đơn xin MIVIA Audio Events + MIVIA Road** | 👤 | **NGÀY ĐẦU TIÊN.** Duyệt có thể mất 1–2 tuần |
| ☐ | `git init`, `.gitignore`, `README.md`, cấu trúc thư mục | 🤖 | |
| ☐ | Viết `docs/taxonomy.md` — 16 class, mỗi class có bao gồm/loại trừ + 3 ví dụ | 👥 | Trích từ SYSTEM.md §3.1 rồi mở rộng |
| ☐ | Viết `docs/annotation_guideline.md` | 👥 | SYSTEM.md §3.6 |
| ☐ | Viết `docs/evaluation_protocol.md` | 🤖 | SYSTEM.md §8 |
| ☐ | `docker-compose.yml`: db(pgvector) + redis + mlflow + api + inference + frontend | 🤖 | Mirror `example_project/docker-compose.yml` |
| ☐ | Khởi tạo DVC + remote local | 🤖 | |
| ☐ | Schema DB + migration (Alembic) | 🤖 | SYSTEM.md §4.3 |
| ☐ | `scripts/validate_taxonomy.py` | 🤖 | |
| ☐ | **Walking skeleton**: upload wav → PANNs tag (tạm) → caption template (tạm) → ghi event → embed → pgvector → `/rag/query` trả lời → frontend hiện | 🤖 | Chấp nhận thô, miễn là **chạy thông** |
| ☐ | CI khung: lint + test + smoke | 🤖 | |
| ☐ | ADR-0001 (FastAPI), ADR-0002 (pgvector), ADR-0003 (Redis Streams), ADR-0004 (kiến trúc Grounded AAC) | 🤖 | |

### Nghiệm thu W1
- [ ] `docker compose up` từ máy trắng → tất cả service healthy
- [ ] Upload 1 file wav → thấy event trên dashboard
- [ ] Hỏi 1 câu tiếng Việt → nhận câu trả lời **có citation**
- [ ] `validate_taxonomy.py` xanh, CI xanh
- [ ] Đơn MIVIA đã gửi

---

# W2 · 22–28/09 · Dữ liệu + Scaper + Precompute

**Mục tiêu:** có train/dev set strong-label đầy đủ và bắt đầu gán nhãn test set.

> 📘 **W2 và W3 thực thi theo [DATA_PLAN.md](DATA_PLAN.md)** — lịch 10 ngày D0–D10, quy trình gán nhãn có máy hỗ trợ, cổng quyết định D7, điều kiện đóng băng `data-v1.0`. Các task dưới đây là bản tóm tắt để tick tiến độ.

### Tasks

| ☐ | Task | Ai | Ghi chú |
|---|---|---|---|
| ☐ | Tải + lọc AudioSet-strong, DESED, FSD50K, UrbanSound8K, ESC-50 theo bảng ánh xạ | 🤖 | Ghi license **từng file** |
| ☐ | Xây **foreground bank** (event đơn, đã trim) và **background bank** | 👥 | Nghe kiểm tra chất lượng — 👤 |
| ☐ | Thu RIR hành lang / nhà xe / xưởng tại IUH | 👤 | Cho slice S3 |
| ☐ | Thu background thật + kịch bản `media_playback` (phát audio qua loa rồi thu lại) | 👤 | Cho slice S4 — **buổi thu số 1** |
| ☐ | `scripts/data_inventory.py` → `docs/data_inventory.md` | 🤖 | Số giờ/class, cảnh báo class < 10 phút |
| ☐ | **QUYẾT ĐỊNH:** class nào thiếu dữ liệu → gộp / loại khỏi macro-F1 / bù bằng tổng hợp | 👤 | Ghi thành ADR |
| ☐ | Pipeline Scaper: sinh soundscape 10s, tham số hoá SNR/overlap/RIR/thứ tự | 🤖 | Xuất wav + TSV + metadata |
| ☐ | Sinh train 15–25h + dev, đảm bảo tỉ lệ slice: overlap ~30%, low-SNR ~25%, reverb ~40%, chain ~15% | 🤖 | |
| ☐ | `scripts/check_leakage.py` + đưa vào CI | 🤖 | Chia theo `source_id`, không theo clip |
| ☐ | **Precompute đặc trưng BEATs → `.npy`** cho toàn bộ train/dev | 🤖 | Quyết định tiết kiệm then chốt |
| ☐ | Dựng Label Studio + import test set + **pilot 30 clip** | 👥 | |
| ☐ | Pilot 30 clip, **nghỉ ≥3 ngày rồi gán lại mù** → tính tự-nhất-quán → **sửa guideline** → mới gán đại trà | 👤 | Bỏ bước này = sai lầm tốn kém nhất |
| ☐ | 🔴 **Bắt đầu gán nhãn test set** (chạy song song suốt W2–W3) | 👤 | ~1 giờ audio; 20% sẽ gán lại lần hai sau ≥7 ngày |

### Nghiệm thu W2
- [ ] `docs/data_inventory.md` sinh tự động, mọi class có nguồn xác định
- [ ] Train ≥ 15h synthetic, đủ 8 slice có mặt
- [ ] `check_leakage.py` xanh
- [ ] Feature BEATs đã precompute xong
- [ ] Pilot 30 clip xong, guideline đã sửa, gán đại trà đang chạy

---

# W3 · 29/09–05/10 · 4 Gold set + SED baseline

**Mục tiêu:** hoàn tất nền đánh giá và có model SED đầu tiên.

### Tasks

| ☐ | Task | Ai | Ghi chú |
|---|---|---|---|
| ☐ | 🔴 Hoàn tất gán nhãn **G1 (SED gold)** ~1h audio | 👤 | Nút thắt của cả đề tài |
| ☐ | Gán lại mù 20%, nghe lại lần ba các ca bất đồng, tính tự-nhất-quán + onset MAE | 👤 | **Ngưỡng ≥ 0.75** (xem DATA_PLAN §8.5) |
| ☐ | Gắn `slice_flags` cho toàn bộ test set | 👥 | |
| ☐ | **G2 (Caption gold)**: sinh template từ G1 → người viết lại 3–5 reference EN + VI | 👥 | 200–300 clip |
| ☐ | Xây **kịch bản mô phỏng 7 ngày**: gán timestamp + location (4 khu vực) cho clip gold | 🤖 | Nền để đặt câu hỏi RAG |
| ☐ | **G3 (RAG gold)** 60–100 câu theo phân bố loại truy vấn ở SYSTEM.md §3.8, **gồm 5% câu không có đáp án** | 👥 | |
| ☐ | **G4 (Alert gold)** 40–60 kịch bản với `expected_severity` + `is_true_alarm` | 👥 | |
| ☐ | `scripts/validate_annotations.py`, `scripts/agreement.py`, `scripts/slice_coverage.py` | 🤖 | Vào CI |
| ☐ | Dataset/DataLoader đọc feature `.npy` + nhãn frame-level | 🤖 | |
| ☐ | **SED baseline**: BEATs(❄️) → Conv1D → Conformer → Head 1, huấn luyện + `sed_eval`/`psds_eval` | 🤖 | Đây đã là trunk dùng lại ở W4–W5 |
| ☐ | MLflow log đầy đủ: git SHA, DVC hash, seed, config, metric 4 tầng | 🤖 | |

### Nghiệm thu W3
- [ ] 4 gold set hoàn chỉnh, validate xanh
- [ ] **Kappa ≥ 0.70**; mỗi class ≥ 20 event trong test; mỗi slice ≥ 20 clip
- [ ] SED baseline có Event-F1 + PSDS + bảng per-class + bảng theo slice
- [ ] Toàn bộ số liệu log trên MLflow, tái lập được

> ⚠️ **Cổng chặn:** không đạt tự-nhất-quán ≥ 0.75 **và** onset lệch trung vị ≤ 100 ms thì **không** đi tiếp — quay lại làm rõ guideline và gán lại. Gold sai thì mọi con số phía sau đều vô nghĩa.
>
> Dự án chỉ có **một** người gán nhãn nên không tính được kappa liên-người; thay bằng test–retest. Ràng buộc này và ba biện pháp bù nằm ở [DATA_PLAN §8](DATA_PLAN.md#8-gold-test-set-loại-c--gán-mù-một-người-gán), và **phải** được nêu trong phần Hạn chế của khoá luận.

---

# W4 · 06–12/10 · AAC baseline B1 + hạ tầng metric

**Mục tiêu:** có baseline captioning và **bộ đo hallucination** hoạt động — phải có metric trước khi tối ưu theo nó.

### Tasks

| ☐ | Task | Ai | Ghi chú |
|---|---|---|---|
| ☐ | Tích hợp BART-base decoder + cross-attention lên trunk | 🤖 | |
| ☐ | Huấn luyện **B1** (chỉ NLL) trên Clotho/AudioCaps → fine-tune trên security data | 🤖 | |
| ☐ | Tích hợp `aac-metrics`: BLEU/METEOR/ROUGE-L/CIDEr/SPICE/SPIDEr/FENSE | 🤖 | |
| ☐ | Xây **`EVENT_LEXICON`** cho 16 class (EN), có xử lý phủ định + biến thể | 👥 | |
| ☐ | Cài **EHR / EOR / GS / TOA / CHR** | 🤖 | SYSTEM.md §8.2 |
| ☐ | 🔬 **Kiểm định bộ trích $P$ thủ công trên 100 caption**, báo cáo độ chính xác của chính bộ trích | 👤 | **Điều kiện cần để C2 được chấp nhận** |
| ☐ | Cài **B0** (structured captioner từ timeline SED + template + LLM viết lại) | 🤖 | Cận trên grounding |
| ☐ | Pipeline dịch EN→VI với glossary 16 class cố định | 🤖 | SYSTEM.md §7.4 |
| ☐ | Đo B0, B1 trên G2 với đủ metric + bảng theo slice | 🤖 | |

### Nghiệm thu W4
- [ ] B1 chạy được, có số SPIDEr trên G2
- [ ] Bộ metric hallucination chạy được, **độ chính xác bộ trích đã đo và ghi lại**
- [ ] B0 chạy được (EHR ≈ 0, SPIDEr thấp — xác nhận đúng kỳ vọng)
- [ ] Bảng 8.2 và 8.3 có 2 dòng đầu (B0, B1)

---

# W5 · 13–19/10 · Grounded AAC (P) — đóng góp chính

**Mục tiêu:** hiện thực hoá đóng góp C1.

### Tasks

| ☐ | Task | Ai | Ghi chú |
|---|---|---|---|
| ☐ | **Head 2**: global mean pool + projection + InfoNCE với text embedder ❄️ | 🤖 | |
| ☐ | Huấn luyện **B2** = B1 + InfoNCE | 🤖 | |
| ☐ | **Multi-task**: gắn Head 1 (SED) vào cùng trunk với Head 3 | 🤖 | |
| ☐ | Lịch huấn luyện 2 giai đoạn (Stage 1: SED+InfoNCE → Stage 2: cả ba, LR trunk ÷10) | 🤖 | SYSTEM.md §5.3 |
| ☐ | Huấn luyện **B3** = B1 + SED head | 🤖 | |
| ☐ | **Grounded decoding**: logit processor 3 vùng ($\theta_{low}$, $\theta_{high}$) + beam search | 🤖 | SYSTEM.md §5.4 |
| ☐ | Cơ chế fallback về structured caption khi bị chặn hết | 🤖 | Không bao giờ trả caption rỗng |
| ☐ | Chọn $\theta_{low}, \theta_{high}$ trên **dev**, khoá lại | 🤖 | Không được chạm test |
| ☐ | Huấn luyện **P** đầy đủ + **P−gd** | 🤖 | |
| ☐ | Chạy 3 seed cho P và B1, báo cáo mean ± std | 🤖 | |

### Nghiệm thu W5
- [ ] B2, B3, P, P−gd đều có số trên G2
- [ ] Bảng 8.3 đầy đủ 6 dòng
- [ ] Grounded decoding hoạt động, tỉ lệ fallback đã đo
- [ ] Mọi run trên MLflow, tái lập được

---

# W6 · 20–26/10 · Ablation + báo cáo theo slice

**Mục tiêu:** biến các con số thành **kết luận nghiên cứu**.

### Tasks

| ☐ | Task | Ai | Ghi chú |
|---|---|---|---|
| ☐ | **E5** — quét ngưỡng $\theta$, vẽ đường cong **CIDEr ↔ EHR** cho B1/B2/B3/P | 🤖 | **Hình trung tâm của khoá luận** |
| ☐ | **E3** — quét trọng số $\lambda$ | 🤖 | |
| ☐ | **E4** — ablation Nhóm B: train có/không 5 lớp nhầm lẫn → so FAR | 🤖 | Chứng minh **C3** |
| ☐ | **E1** — so backbone BEATs vs PANNs | 🤖 | Cắt được nếu trễ |
| ☐ | **E6** — bảng metric đầy đủ theo 8 slice cho SED + Captioning + Grounding | 🤖 | Bảng 8.4 |
| ☐ | **E9** — đo chênh lệch synthetic-dev vs real-test | 🤖 | Báo cáo trung thực |
| ☐ | Phân tích lỗi định tính: 20 caption tệ nhất, phân loại theo H1–H4 | 👤 | Rất giá trị cho chương thảo luận |
| ☐ | Chốt model đưa vào production, đăng ký MLflow Registry → Staging | 🤖 | |
| ☐ | Viết Ch.5 + Ch.6 báo cáo (từ SYSTEM.md §5, §8, §10) | 👥 | |

### Nghiệm thu W6
- [ ] Đường cong CIDEr↔EHR đã vẽ, có kết luận rõ ràng (kể cả kết luận âm tính)
- [ ] Bảng theo slice đầy đủ
- [ ] E4 có kết luận về đóng góp của Nhóm B
- [ ] Model đã promote lên Staging

---

# W7 · 27/10–02/11 · Streaming + RAG + Dashboard

**Mục tiêu:** thay toàn bộ stub của walking skeleton bằng đồ thật.

### Tasks

| ☐ | Task | Ai | Ghi chú |
|---|---|---|---|
| ☐ | Redis Streams producer/consumer + file replay simulator | 🤖 | |
| ☐ | Buffer 5s / overlap 2.5s + energy gate + health check tín hiệu (S8) | 🤖 | |
| ☐ | `services/inference`: load model thật, batch động, GPU | 🤖 | |
| ☐ | **Temporal Aggregator** (dedup IoU → gom cụm → đóng cụm → sinh caption) | 🤖 | SYSTEM.md §6.2 |
| ☐ | Cảnh báo sớm cho tier Critical (không đợi cụm đóng) | 🤖 | |
| ☐ | Risk scoring + **4 quy tắc chặn** (Nhóm B, media playback, confidence thấp, chống spam) | 🤖 | SYSTEM.md §6.4 |
| ☐ | WebSocket alert + toast/âm thanh trên frontend | 🤖 | |
| ☐ | RAG: query parser (trích thời gian/địa điểm/lớp) + hybrid retrieval một câu SQL | 🤖 | |
| ☐ | Prompt có ràng buộc + bắt buộc citation + cho phép trả lời "không tìm thấy" | 🤖 | |
| ☐ | Đo **L3**: Recall@5, P@5, MRR, nDCG@10 trên G3, tách theo loại truy vấn | 🤖 | |
| ☐ | **E7** — vector-only vs hybrid | 🤖 | |
| ☐ | Frontend 3 màn: live feed, event detail + phát lại audio + nút feedback, RAG chat | 🤖 | |

### Nghiệm thu W7
- [ ] Replay một file 10 phút → sự kiện xuất hiện đúng, aggregator không tách vụn
- [ ] Latency P95 đo được và ≤ 6s
- [ ] Chạy hết G3, có bảng 8.5
- [ ] Nhấn nút feedback trên dashboard → ghi được vào `event_feedback`

---

# W8 · 03–09/11 · MLOps + E2E eval + bản nháp báo cáo

**Mục tiêu:** đóng vòng MLOps và hoàn tất số liệu.

### Tasks

| ☐ | Task | Ai | Ghi chú |
|---|---|---|---|
| ☐ | MLflow Model Registry + `ml/configs/promotion.yaml` (**EHR không tăng** là điều kiện chặn) | 🤖 | SYSTEM.md §9.1 |
| ☐ | DVC pipeline hoàn chỉnh (`dvc repro` tái tạo từ raw → model) | 🤖 | |
| ☐ | Bảng `inference_metrics` + panel monitoring + tính PSI drift | 🤖 | |
| ☐ | Giám sát **tỉ lệ fallback grounded-decoding** như tín hiệu drift sớm | 🤖 | Chi tiết hay, đừng bỏ |
| ☐ | Vòng lặp feedback: export mẫu sai → dataset bump → CLI retrain | 🤖 | |
| ☐ | CI đầy đủ: lint + test(cov 80%) + data-validate + build + smoke | 🤖 | |
| ☐ | **E8** — đo latency đầy đủ theo chặng, P50/P95/P99 | 🤖 | |
| ☐ | **L4** — Alert P/R/F1 + FAR trên G4; Faithfulness + Answer Correctness trên G3 | 🤖 | |
| ☐ | Hiệu chuẩn LLM-judge bằng 30 mẫu người chấm | 👤 | Không được chỉ dùng judge |
| ☐ | Hoàn thiện `SYSTEM.md` → bản nháp báo cáo đầy đủ 8 chương | 👥 | |
| ☐ | README + hướng dẫn tái lập + video demo | 🤖 | |

### Nghiệm thu W8
- [ ] `docker compose up` từ máy trắng chạy được toàn bộ hệ thống
- [ ] `dvc repro` tái tạo được model
- [ ] Đủ 7 bảng kết quả (8.1–8.7)
- [ ] Bản nháp báo cáo đủ 8 chương

---

## Cut-list — thứ tự hy sinh khi trễ

Khi một tuần không đạt nghiệm thu, cắt theo **đúng thứ tự này**, không tự ý đảo:

| # | Cắt gì | Mất gì | Vẫn giữ được gì |
|---|---|---|---|
| 1 | Grafana / monitoring nâng cao | Biểu đồ đẹp | Bảng `inference_metrics` vẫn ghi |
| 2 | **E1** (so backbone) | Một bảng phụ | Đóng góp chính không đổi |
| 3 | Adapter MQTT | Chứng minh khả năng mở rộng | Redis Streams vẫn là streaming thật |
| 4 | Buổi thu thực địa số 2 | Đa dạng background | Slice S3, S4 vẫn có từ buổi 1 |
| 5 | G3 giảm 100 → 60 câu | Độ tin cậy thống kê của L3 | Vẫn đủ phân bố 6 loại truy vấn |
| 6 | Chạy 1 seed thay vì 3 | Không báo cáo được std | Vẫn có điểm số |
| 7 | **B2 và B3** (chỉ giữ B1 và P) | Không tách được đóng góp từng thành phần | Vẫn so được P vs B1 |
| 8 | Frontend rút còn 2 màn (bỏ event detail) | Trải nghiệm | RAG chat + live feed vẫn còn |
| 9 | **E4** (ablation Nhóm B) | Đóng góp C3 yếu đi | Taxonomy vẫn dùng được |

**Tuyệt đối không cắt:** G1 gold set · bộ metric hallucination (C2) · mô hình P (C1) · E5 (đường cong CIDEr↔EHR) · báo cáo theo slice. Cắt bất kỳ thứ nào trong năm thứ này là mất luôn lý do tồn tại của đề tài.

---

## Nợ kỹ thuật & việc phải xác minh

| ☐ | Việc | Hạn | Vì sao |
|---|---|---|---|
| ☐ | Đối chiếu lại mọi phát biểu về công trình liên quan (SYSTEM.md §2) với bản gốc | W6 | Chương 2 báo cáo không được viết từ trí nhớ |
| ☐ | Xác minh số giờ + license thực tế từng dataset | W2 | Đang là giả định |
| ☑ | **Xác minh ID ontology AudioSet** trong `ml/configs/ontology_map.yaml` từ file gốc | **D0** | Chép sai ID = tải sai lớp = hỏng từ gốc — xong 14/09, `verify_ontology.py` 0 lỗi |
| ☐ | Xác minh tần số frame BEATs và tỉ lệ downsample thực tế | W3 | Con số 50 Hz → 17 Hz cần kiểm chứng bằng code |
| ☐ | Kiểm định độ chính xác bộ trích `EVENT_LEXICON` | W4 | Điều kiện cần của C2 |
| ☐ | Xác nhận `aac-metrics` / `psds_eval` cài được trên Windows | W1 | Có gói khó build trên Windows → cân nhắc WSL2 |
| ☐ | Kiểm tra VRAM thực tế khi train Conformer + BART | W4 | Quyết định batch size |
| ☐ | Cài `yt-dlp` + `ffmpeg`, chạy `fetch_audioset_strong.py` thật | W2 | Script đã viết + kiểm thử 15/09, chưa tải audio thật — `dev`/`gold_test` vẫn rỗng, cổng D7 chưa qua |
| ☐ | Đo lại tỉ lệ hụt video AudioSet-strong (dự kiến 15–30%, ghi `unavailable` vào `exclusions.csv`) | W2 | Phải báo cáo tỉ lệ hụt thật trong khoá luận, không dùng số dự kiến |
| ☐ | Xác nhận ruling chuông quầy thức ăn nhanh trong `data/gold/decision_log.md` (đang ⚠️ CHỜ XÁC NHẬN) | W2 | Ảnh hưởng biên `alarm_bell` |
| ☐ | Xử lý 1802-clip `review_queue.csv` (đang trì hoãn có chủ đích) | W2 | Chờ ổn định taxonomy trước khi nghe hàng loạt, tránh nghe lại 2 lần |

---

## Nhịp làm việc hằng tuần

| Khi nào | Làm gì |
|---|---|
| **Đầu tuần** | Đọc `CLAUDE.md` → xem tuần hiện tại trong file này → chốt 3 việc ưu tiên |
| **Cuối mỗi block** | Cập nhật `CLAUDE.md` (progress log + next actions) |
| **Cuối tuần** | Đối chiếu nghiệm thu → tick `[x]` → cập nhật bảng trạng thái tổng → nếu trượt thì **kích hoạt cut-list ngay**, không dời sang tuần sau |
| **Mỗi quyết định kiến trúc** | Viết một ADR trong `docs/decisions/` |

---

*Cập nhật lần cuối: 2026-09-15 · Tuần hiện tại: **W1 (dữ liệu đang chạy sớm, xem CLAUDE.md §3)***
