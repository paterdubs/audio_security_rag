# SYSTEM.md — Đặc tả hệ thống

> **Đề tài:** Xây dựng hệ thống giám sát an ninh và truy xuất cảnh báo (RAG) thông qua sinh mô tả âm thanh tự động
>
> **English title:** *Grounded Automated Audio Captioning for Security Surveillance and RAG-based Alert Retrieval — an MLOps Approach*
>
> **Loại tài liệu:** Đặc tả kỹ thuật đầy đủ (single source of truth về *hệ thống LÀ GÌ*).
> **Không** chứa tiến độ (xem [PLAN.md](PLAN.md)) và **không** chứa trạng thái phiên làm việc (xem [../CLAUDE.md](../CLAUDE.md)).
>
> **Quy ước:** Mục lục của file này ánh xạ 1:1 với chương báo cáo khoá luận. Viết file này = viết báo cáo.
>
> | Phiên bản | Ngày | Ghi chú |
> |---|---|---|
> | 0.1 | 2026-09-14 | Bản khởi tạo, chốt taxonomy 15 class + 8 slice + kiến trúc Grounded AAC |

---

## Mục lục

| Chương | Nội dung | Mục |
|---|---|---|
| 1 | Tổng quan đề tài | [§1](#1-tổng-quan-đề-tài) |
| 2 | Cơ sở lý thuyết & công trình liên quan | [§2](#2-cơ-sở-lý-thuyết--công-trình-liên-quan) |
| 3 | Dữ liệu: taxonomy, nhãn, gold set | [§3](#3-dữ-liệu) |
| 4 | Kiến trúc hệ thống | [§4](#4-kiến-trúc-hệ-thống) |
| 5 | Mô hình: SED + Grounded Audio Captioning | [§5](#5-mô-hình-sed--grounded-audio-captioning) |
| 6 | Xử lý sự kiện, đánh giá rủi ro & cảnh báo | [§6](#6-xử-lý-sự-kiện-đánh-giá-rủi-ro--cảnh-báo) |
| 7 | Tầng RAG | [§7](#7-tầng-rag) |
| 8 | Phương pháp đánh giá | [§8](#8-phương-pháp-đánh-giá) |
| 9 | MLOps | [§9](#9-mlops) |
| 10 | Kế hoạch thực nghiệm | [§10](#10-kế-hoạch-thực-nghiệm) |
| 11 | Rủi ro, giả định & hạn chế | [§11](#11-rủi-ro-giả-định--hạn-chế) |
| 12 | Phụ lục | [§12](#12-phụ-lục) |

---

# 1. Tổng quan đề tài

## 1.1 Bối cảnh & động lực

Hệ thống giám sát an ninh hiện nay gần như hoàn toàn dựa vào **thị giác** (camera + computer vision). Cách tiếp cận này có ba điểm mù cố hữu:

1. **Điểm mù vật lý.** Camera chỉ thấy trong khung hình. Sự kiện xảy ra sau góc khuất, trong bóng tối, sau cánh cửa đóng, hoặc ngoài tầm nhìn thì không được ghi nhận. Âm thanh thì lan truyền qua vật cản và không phụ thuộc hướng nhìn.
2. **Điểm mù ngữ nghĩa.** Camera thấy một người đứng cạnh cửa kính, nhưng không phân biệt được người đó đang chờ hay đang đập vỡ kính. Tiếng kính vỡ là bằng chứng dứt khoát.
3. **Điểm mù chi phí.** Camera độ phân giải cao + GPU xử lý video 24/7 tốn kém hơn microphone + xử lý audio nhiều lần, ở cả chi phí thiết bị lẫn băng thông và điện năng.

Song song, nhân viên an ninh phải đối mặt với vấn đề **truy hồi thông tin**: khi cần biết *"đêm qua ở hành lang tầng 2 có gì bất thường không"*, họ phải tua lại hàng giờ ghi hình. Không có cơ chế **hỏi bằng ngôn ngữ tự nhiên** trên lịch sử sự kiện.

Về mặt kỹ thuật, ba dòng nghiên cứu đã đủ chín để ghép lại:

- **Sound Event Detection (SED)** — phát hiện *cái gì* kêu và *khi nào*.
- **Automated Audio Captioning (AAC)** — sinh mô tả ngôn ngữ tự nhiên cho âm thanh.
- **Retrieval-Augmented Generation (RAG)** — truy xuất bằng chứng rồi mới sinh câu trả lời.

Nhưng ghép ba thứ này lại làm lộ ra một vấn đề chưa được giải quyết thoả đáng: **AAC hallucinate**. Một model captioning sinh ra *"tiếng súng nổ"* trong khi thực tế chỉ có tiếng pháo, và mô tả sai đó sẽ được lưu vĩnh viễn vào cơ sở dữ liệu, được index vector, rồi được RAG trả về như bằng chứng cho một câu hỏi an ninh. **Sai lầm ở tầng captioning không dừng lại ở tầng captioning — nó bị khuếch đại và hợp thức hoá qua toàn bộ hệ thống.** Trong domain an ninh, một caption bịa nguy hiểm hơn không có caption.

Đây chính là chỗ đề tài định vị đóng góp của mình.

## 1.2 Phát biểu bài toán

Cho một luồng âm thanh liên tục $x(t)$ thu từ một hoặc nhiều điểm giám sát, hệ thống cần:

| Ký hiệu | Bài toán | Đầu vào → Đầu ra |
|---|---|---|
| **P1** | Sound Event Detection | $x(t)$ → tập $\{(c_i, t^{on}_i, t^{off}_i, p_i)\}$ với $c_i \in \mathcal{C}$, $|\mathcal{C}| = 15$ |
| **P2** | Grounded Audio Captioning | $x(t)$ + kết quả P1 → caption $y$ sao cho mọi thực thể sự kiện trong $y$ **đều có bằng chứng** trong P1 |
| **P3** | Security Event Extraction | P1 + P2 + ngữ cảnh (thời gian, vị trí) → Security Event có cấu trúc + mức rủi ro |
| **P4** | Alert | Security Event → quyết định phát cảnh báo, tối thiểu hoá đồng thời False Alarm Rate và Miss Rate |
| **P5** | RAG Retrieval & Answering | Câu hỏi tiếng Việt $q$ → tập sự kiện liên quan $E_q$ + câu trả lời $a$ **có trích dẫn** $E_q$ |
| **P6** | MLOps | Toàn bộ P1–P5 phải versioned, reproducible, monitored, retrainable |

## 1.3 Mục tiêu

**Mục tiêu tổng quát:** xây dựng và đánh giá một hệ thống giám sát an ninh bằng âm thanh hoạt động near-real-time, trong đó mô tả sinh tự động được **ràng buộc vào bằng chứng âm học**, và lịch sử cảnh báo truy vấn được bằng ngôn ngữ tự nhiên.

**Mục tiêu cụ thể:**

| # | Mục tiêu | Tiêu chí đo |
|---|---|---|
| M1 | Xây dựng taxonomy 15 lớp âm thanh an ninh có định nghĩa vận hành rõ ràng | Kappa liên người gán ≥ 0.70 |
| M2 | Xây dựng bộ dữ liệu strong-label + 4 gold set cho 4 tầng đánh giá | Đủ 8 test slice, mỗi slice ≥ 20 clip |
| M3 | Huấn luyện SED đa nhãn có định vị thời gian | Event-based F1, PSDS, báo cáo theo từng slice |
| M4 | **Đề xuất kiến trúc Grounded AAC chống hallucination** | Giảm Event Hallucination Rate so với baseline ở cùng mức CIDEr |
| M5 | **Đề xuất bộ metric đo hallucination cho AAC dựa trên strong label** | EHR / EOR / GS — xem [§8.2](#82-bộ-metric-hallucination-đề-xuất) |
| M6 | Xây dựng pipeline streaming + risk scoring + alert | Latency end-to-end P95 ≤ 6s |
| M7 | Xây dựng tầng RAG song ngữ có trích dẫn bằng chứng | Recall@5, MRR, Faithfulness |
| M8 | Triển khai vòng đời MLOps đầy đủ | `docker compose up` reproducible; MLflow + DVC + CI + monitoring |

## 1.4 Phạm vi & giới hạn

**Trong phạm vi:**

- 15 lớp sự kiện âm thanh (10 sự kiện an ninh + 5 lớp gây nhầm lẫn) — [§3.1](#31-taxonomy-15-lớp).
- Bối cảnh triển khai **tổng quát cho 4 khu vực**: trường học, bãi/nhà xe, khu dân cư, nhà máy. Hệ thống không tối ưu riêng cho một khu vực; `location` là metadata cấu hình được, không phải tham số của model.
- Tiếng Việt cho giao diện và truy vấn RAG; tiếng Anh cho caption benchmark.
- Near-real-time (mục tiêu P95 ≤ 6 giây end-to-end).
- Triển khai một node bằng Docker Compose.

**Ngoài phạm vi (nêu rõ để tránh bị hỏi vặn):**

| Không làm | Lý do |
|---|---|
| Nhận dạng người nói (speaker ID), nhận dạng nội dung lời nói (ASR) | Vấn đề pháp lý về quyền riêng tư; không cần thiết cho bài toán |
| Định vị nguồn âm (sound source localization) bằng mảng micro | Cần phần cứng chuyên dụng; là hướng phát triển |
| Fusion với video | Mở rộng scope sang multimodal; là hướng phát triển |
| Triển khai phân tán / Kubernetes / đa node | Docker Compose đủ chứng minh luận điểm |
| Hard real-time (< 1s) | Kéo đề tài sang tối ưu hệ thống thay vì nghiên cứu |
| Chống tấn công đối kháng (adversarial audio) | Là hướng phát triển |

**Cân nhắc đạo đức & pháp lý.** Hệ thống thu âm không gian công cộng/bán công cộng. Ba nguyên tắc bắt buộc trong thiết kế: (a) **không** lưu trữ nội dung lời nói dạng văn bản, chỉ lưu nhãn `speech_normal` mức sự kiện; (b) audio thô lưu có thời hạn (retention policy mặc định 30 ngày) và có thể tắt hoàn toàn, chỉ giữ metadata + caption; (c) mọi bản ghi thu tại IUH phục vụ khoá luận đều phải có sự đồng ý của người liên quan và được ghi nhận trong `data_inventory.md`.

## 1.5 Đóng góp

| # | Đóng góp | Loại |
|---|---|---|
| **C1** | **Kiến trúc Grounded AAC đa nhiệm**: chia sẻ trunk BEATs+Conformer giữa SED và captioning, dùng **strong label làm tín hiệu grounding**, kết hợp 3 mục tiêu (NLL + InfoNCE + frame-level SED) và **grounded decoding** chặn từ vựng sự kiện không có bằng chứng | Nghiên cứu |
| **C2** | **Bộ metric đo hallucination cho AAC** dựa trên strong label: Event Hallucination Rate, Event Omission Rate, Grounding Score, Temporal Order Accuracy | Nghiên cứu |
| **C3** | **Taxonomy 15 lớp an ninh có chủ đích đưa 5 lớp gây nhầm lẫn vào**, kèm 8 test slice mô phỏng điều kiện thực tế (đặc biệt `media_playback` — âm thanh nguy hiểm phát ra từ TV/loa) | Dữ liệu |
| **C4** | **Security Audio RAG Evaluation Set** — bộ gold 4 tầng cho SED / Captioning / Retrieval / End-to-end Alert | Dữ liệu |
| **C5** | Hệ thống end-to-end streaming → alert → RAG có trích dẫn, vận hành theo quy trình MLOps đầy đủ | Kỹ thuật |

## 1.6 Cấu trúc tài liệu

Xem [Mục lục](#mục-lục). Ba tài liệu vận hành song song:

```
CLAUDE.md      → "đang ở đâu"      (trạng thái, cập nhật cuối mỗi block)
docs/PLAN.md   → "làm gì, khi nào" (8 tuần, checkbox, nghiệm thu)
docs/SYSTEM.md → "hệ thống là gì"  (file này, = khung báo cáo)
```

---

# 2. Cơ sở lý thuyết & công trình liên quan

> ⚠️ **Ghi chú cho người viết:** các phát biểu về công trình liên quan trong chương này cần **đối chiếu lại với bản gốc** trước khi đưa vào báo cáo chính thức. Xem checklist trong [PLAN.md § Nợ kỹ thuật](PLAN.md).

## 2.1 Sound Event Detection

**Weak label vs Strong label.** Weak label chỉ cho biết *có* sự kiện trong clip (`clip → {glass_breaking}`). Strong label cho biết thêm *khi nào* (`clip → glass_breaking @ [1.25s, 3.64s]`). Sự khác biệt này quyết định:

| | Weak | Strong |
|---|---|---|
| Chi phí gán nhãn | Thấp (~1× realtime) | Cao (5–10× realtime) |
| Metric khả dụng | Clip-level F1, mAP | **Event-based F1, PSDS, onset/offset MAE** |
| Huấn luyện | Cần MIL / attention pooling | Giám sát trực tiếp mức frame |
| Grounding cho captioning | Không đủ (không biết thứ tự) | **Đủ (biết thứ tự và chồng lấn)** |

Đề tài này chọn **strong label**, và dòng cuối của bảng chính là lý do quyết định — nếu không biết thứ tự thời gian thì không thể ràng buộc caption vào bằng chứng, cũng không thể đo được caption có mô tả sai thứ tự hay không.

**Metric chuẩn.** Cộng đồng DCASE dùng:
- *Segment-based F1*: chia trục thời gian thành segment cố định (thường 1s), so khớp theo segment. Khoan dung với sai số biên.
- *Event-based F1*: so khớp từng event, chấp nhận sai lệch onset trong một **collar** (thường 200 ms) và offset trong tỉ lệ độ dài. Khắt khe hơn, phản ánh đúng chất lượng định vị.
- *PSDS (Polyphonic Sound Detection Score)*: tổng hợp trên nhiều ngưỡng vận hành, tránh phụ thuộc vào một threshold được tuning may mắn.

Đề tài báo cáo cả ba, lấy **event-based F1 (collar 200 ms)** làm số chính và **PSDS** làm số chống-tuning.

## 2.2 Automated Audio Captioning

AAC sinh một câu mô tả ngôn ngữ tự nhiên từ tín hiệu âm thanh. Kiến trúc chủ đạo hiện nay là **encoder–decoder**: một audio encoder đã pretrain (PANNs, AST, BEATs, HTS-AT) trích đặc trưng, một language decoder (Transformer, BART, GPT) sinh caption, nối với nhau qua cross-attention hoặc qua một adapter/mapping network.

**Bộ dữ liệu chuẩn:** AudioCaps (dựa trên AudioSet, caption do người viết) và Clotho (caption đa tham chiếu, mỗi audio 5 caption).

**Metric chuẩn:** BLEU, METEOR, ROUGE-L, CIDEr, SPICE, SPIDEr (trung bình CIDEr + SPICE), và FENSE (kết hợp sentence-BERT similarity với một bộ phát hiện lỗi trôi chảy).

**Điểm yếu chung của các metric trên:** chúng đo *độ giống caption tham chiếu về mặt n-gram/ngữ nghĩa*, **không** đo *tính đúng đắn về bằng chứng*. Một caption bịa thêm một sự kiện nhưng dùng đúng phong cách ngôn ngữ vẫn có thể đạt CIDEr cao. Đây là khoảng trống mà [§8.2](#82-bộ-metric-hallucination-đề-xuất) nhắm vào.

## 2.3 Hallucination trong AAC

Hallucination trong AAC biểu hiện ở bốn dạng, và đề tài phân loại chúng như sau:

| Dạng | Mô tả | Ví dụ |
|---|---|---|
| **H1 — Bịa thực thể** | Caption nhắc tới sự kiện không có trong audio | Audio chỉ có tiếng pháo → *"gunshots are fired"* |
| **H2 — Bỏ sót** | Caption bỏ qua sự kiện quan trọng có thật | Audio có kính vỡ + hét → *"a person is talking"* |
| **H3 — Sai thứ tự** | Đúng sự kiện, sai quan hệ thời gian | Thực tế: hét *rồi* kính vỡ → caption: *"glass breaks, then someone screams"* |
| **H4 — Bịa ngữ cảnh** | Thêm chi tiết môi trường/tác nhân không suy ra được từ audio | *"a burglar breaks into a house at night"* |

Nguyên nhân gốc: decoder ngôn ngữ được huấn luyện bằng NLL trên phân phối caption của tập train, nên nó có **prior ngôn ngữ mạnh**. Khi tín hiệu âm học yếu hoặc mơ hồ, decoder rơi về prior đó và sinh ra mô tả "nghe hợp lý" thay vì "đúng với audio". Với domain an ninh — nơi tín hiệu thường yếu (xa mic, ồn nền) và các lớp dễ nhầm nhau — đây không phải rủi ro lý thuyết mà là chế độ hỏng mặc định.

**Hướng khắc phục trong tài liệu:** (a) tăng cường liên kết audio–text bằng mục tiêu contrastive (dòng CLAP); (b) ràng buộc decoder bằng thông tin sự kiện tường minh; (c) can thiệp lúc giải mã (constrained decoding). Đề tài kết hợp cả ba — [§5](#5-mô-hình-sed--grounded-audio-captioning).

## 2.4 Biểu diễn âm thanh pretrained

| Model | Cơ chế | Lý do quan tâm |
|---|---|---|
| **PANNs (CNN14)** | CNN trên log-mel, pretrain AudioSet | Nhẹ, chạy được CPU, baseline tốt |
| **AST / HTS-AT** | Transformer trên patch spectrogram | Mạnh, nhưng nặng |
| **BEATs** | Self-supervised với acoustic tokenizer, iterative | **Đặc trưng mức frame chất lượng cao** — điều kiện cần cho grounding theo thời gian |
| **CLAP** | Contrastive audio–language pretraining | Nền tảng ý tưởng cho nhánh InfoNCE |

Đề tài chọn **BEATs làm encoder đóng băng**, vì hai lý do: (1) nó cho đặc trưng theo frame (~50 Hz) đủ mịn để giám sát bằng strong label; (2) đóng băng nó cho phép **precompute đặc trưng ra đĩa một lần**, giảm mạnh chi phí huấn luyện — quyết định then chốt cho một khoá luận 8 tuần.

## 2.5 Retrieval-Augmented Generation

RAG tách bài toán trả lời thành *truy xuất bằng chứng* rồi *sinh câu trả lời có điều kiện trên bằng chứng đó*. Với hệ thống này, đặc thù là:

- Bằng chứng không phải văn bản tự do mà là **Security Event có cấu trúc** — nên retrieval phải **lai** giữa tìm kiếm ngữ nghĩa (vector trên caption) và lọc theo metadata (thời gian, địa điểm, mức rủi ro, loại sự kiện).
- Truy vấn tiếng Việt, caption benchmark tiếng Anh → bắt buộc xử lý song ngữ ([§7.4](#74-xử-lý-song-ngữ)).
- Câu trả lời phải **trích dẫn `event_id`**, để người dùng nghe lại được audio gốc. Không trích dẫn được thì không dùng được trong an ninh.

## 2.6 MLOps

Bốn trụ cột áp dụng: **data versioning** (DVC), **experiment tracking + model registry** (MLflow), **CI/CD** (GitHub Actions + Docker), **monitoring & feedback** (inference metrics + xác nhận của người vận hành → retraining). Chi tiết ở [§9](#9-mlops).

## 2.7 Khoảng trống nghiên cứu

| Khoảng trống | Hệ quả | Đề tài xử lý |
|---|---|---|
| AAC được đánh giá bằng metric tương đồng văn bản, không có metric đo **tính có bằng chứng** | Không phân biệt được model trung thực với model nói hay | **C2** — bộ metric EHR/EOR/GS/TOA |
| AAC hiếm khi tận dụng **strong label** làm giám sát grounding | Bỏ phí tín hiệu mạnh nhất đang có | **C1** — head SED mức frame trên trunk chia sẻ |
| Dataset SED an ninh thường chỉ gồm lớp "nguy hiểm", thiếu lớp gây nhầm | Báo cáo F1 đẹp, deploy thì báo động giả liên tục | **C3** — 5 lớp nhầm lẫn trong taxonomy |
| Chưa có benchmark nào phủ chuỗi *audio → caption → security event → RAG* | Không đánh giá được hệ thống như một tổng thể | **C4** — gold set 4 tầng |
| Kịch bản **media playback** (TV/loa phát tiếng súng, tiếng hét) gần như không được xử lý | Nguồn báo động giả nghiêm trọng trong thực tế | Slice **S4** |

---

# 3. Dữ liệu

## 3.1 Taxonomy 15 lớp

Nguyên tắc thiết kế: **10 lớp sự kiện an ninh + 5 lớp gây nhầm lẫn/nền**. Năm lớp Nhóm B không phải "rác" — chúng là lớp có nhãn đầy đủ và chính chúng quyết định False Alarm Rate.

### Nhóm A — Sự kiện an ninh (10 lớp)

| # | `class_id` | Tên tiếng Việt | Tier | Bao gồm | Loại trừ | Đặc thù kỹ thuật |
|---|---|---|---|---|---|---|
| 1 | `gunshot` | Tiếng súng | Critical | Súng ngắn, súng trường, súng hơi; loạt bắn | Pháo (→11), nổ bô xe (→15), đóng sập cửa (→7) | Rất ngắn (<0.5s), transient mạnh → biên onset nhạy |
| 2 | `explosion` | Tiếng nổ | Critical | Nổ bình gas, nổ lớn, tiếng bùng cháy mạnh | Pháo hoa (→11), sấm (→15) | Năng lượng tần thấp, đuôi vang dài |
| 3 | `scream` | Tiếng hét hoảng loạn | Critical | Hét sợ hãi/đau đớn, cao độ cao, thất thanh | Reo hò, cười to, trẻ chơi đùa (→13), quát giận (→6) | Ranh giới với 6 và 13 là chỗ khó nhất của taxonomy |
| 4 | `glass_breaking` | Kính vỡ | Critical | Vỡ cửa kính, kính xe, chai lọ thuỷ tinh | Rơi chén đĩa sứ/nhựa (→12) | Phổ tần cao đặc trưng, thời lượng ngắn |
| 5 | `vehicle_crash` | Va chạm xe | Critical | Đâm va, tiếng rít lốp trước va chạm, biến dạng kim loại | Đóng cửa xe, còi xe (→15) | Lớp hiếm dữ liệu nhất |
| 6 | `shout_yell` | Quát tháo, cãi vã | High | Quát giận, đe doạ, tranh cãi lớn tiếng | Hét hoảng loạn (→3), nói to bình thường (→14) | Có nội dung lời nói, cao độ thấp hơn `scream` |
| 7 | `door_slam` | Đóng sập cửa / phá cửa | High | Sập cửa mạnh, đạp cửa, cạy cửa | Đóng cửa nhẹ (→15) | Cần hard negative "gió sập cửa" |
| 8 | `running_footsteps` | Bước chân chạy | Medium | Chạy, bước gấp, nhiều người di chuyển nhanh | Đi bộ thong thả (→15) | Lớp **bổ trợ ngữ cảnh**, hiếm khi tự nó là cảnh báo |
| 9 | `siren` | Còi hú | Medium | Còi cứu hoả, cứu thương, cảnh sát | Chuông báo động cố định (→10), còi xe (→15) | Thường là *hệ quả*, không phải *nguyên nhân* |
| 10 | `alarm_bell` | Chuông báo động | Medium | Báo cháy, báo trộm, chuông cảnh báo thiết bị | Chuông cửa, chuông điện thoại (→15) | Kéo dài rất lâu → kích hoạt slice S6 |

### Nhóm B — Lớp gây nhầm lẫn & nền (5 lớp)

| # | `class_id` | Tên tiếng Việt | Nhầm với | Lý do phải là lớp riêng |
|---|---|---|---|---|
| 11 | `fireworks` | Pháo, pháo hoa | 1, 2 | Bối cảnh Việt Nam: Tết, đám cưới, khai trương. **Nguồn báo động giả số 1** |
| 12 | `object_drop_dishes` | Rơi vỡ đồ vật, chén đĩa | 4 | Rất phổ biến ở căng-tin/khu dân cư; DESED có lớp `Dishes` sẵn strong label |
| 13a | `laughter` | Cười (một giọng) | 3 | Tách khỏi `applause_cheering` (2026-09-15): phổ và nhịp khác hẳn tiếng vỗ tay/đám đông |
| 13b | `applause_cheering` | Vỗ tay, reo hò, trẻ chơi đùa | 3, 6 | Nguồn báo động giả chủ đạo ở trường học và khu dân cư |
| 14 | `speech_normal` | Hội thoại bình thường | 6 | Lớp âm tính chủ đạo; đồng thời là tín hiệu "có người hiện diện" |
| 15 | `ambient_noise` | Nền: mưa, gió, sấm, giao thông, HVAC, im lặng | tất cả | Lớp hấp thụ; ngăn model gán nhãn nguy hiểm cho mọi tiếng động lạ |

**Luận điểm bảo vệ.** Phần lớn công trình về "audio surveillance" chỉ huấn luyện 3–8 lớp nguy hiểm rồi báo F1 cao. Khi triển khai, mọi âm thanh không thuộc tập huấn luyện đều bị ép vào một lớp nguy hiểm nào đó. Đưa 6 lớp nhầm lẫn vào taxonomy là **quyết định thiết kế có chủ đích**, và lợi ích của nó **đo được** bằng thí nghiệm ablation: huấn luyện có/không Nhóm B → so sánh False Alarm Rate trên test set ([§10](#10-kế-hoạch-thực-nghiệm), thí nghiệm E4).

### Phân cấp Tier và ý nghĩa

`Critical` > `High` > `Medium` > `Negative`(Nhóm B). Tier **không** là đầu ra của model — nó là tham số của bảng luật risk scoring ([§6.4](#64-risk-scoring)) và của trọng số trong macro-F1 có trọng số (báo cáo bổ sung, không thay thế macro-F1 thường).

## 3.2 Tám test slice — trường hợp đặc biệt

Slice **không phải class**. Mỗi clip trong test set được gắn thêm cờ `slice_flags`, và metric được báo cáo **riêng cho từng slice**. Đây là nơi khoá luận thể hiện chiều sâu: một con số F1 tổng thể che giấu hoàn toàn việc model hỏng ở đâu.

| # | `slice_id` | Định nghĩa vận hành | Cách tạo | Rủi ro nó bộc lộ |
|---|---|---|---|---|
| **S1** | `overlap` | ≥2 event chồng lấn ≥30% thời lượng của event ngắn hơn | Scaper: ép chồng lấn | Model chỉ đoán được 1 nhãn trội |
| **S2** | `low_snr` | SNR ≤ 5 dB giữa foreground và background | Scaper: điều khiển SNR; MIVIA có sẵn thang 5–30 dB | Sự kiện ở xa micro |
| **S3** | `reverb` | Convolve với RIR hành lang / nhà xe / xưởng | Thu RIR thật + tích chập | Domain gap: dữ liệu sạch → thực địa vang |
| **S4** | `media_playback` | Âm thanh sự kiện **phát ra từ loa/TV**, không phải sự kiện thật | Phát audio qua loa rồi thu lại | **Báo động giả nghiêm trọng nhất; gần như chưa được xử lý** |
| **S5** | `boundary_split` | Event bị cắt ngang giữa hai chunk buffer | Dịch offset khi chia chunk | Lỗi của tầng streaming, không phải của model |
| **S6** | `long_event` | Event dài hơn cửa sổ buffer (ví dụ chuông kêu 2 phút) | Scaper: kéo dài foreground | Aggregator đếm trùng 1 sự kiện thành N |
| **S7** | `causal_chain` | Chuỗi nhân quả ≥3 event trong 20s (glass → scream → footsteps) | Scaper: kịch bản có thứ tự | **Chính là lý do Temporal Aggregator và captioning tồn tại** |
| **S8** | `dead_channel` | Micro hỏng: im lặng tuyệt đối, clipping, nhiễu DC | Sinh tổng hợp | Health check hạ tầng, không phải AI |

**Ưu tiên đầu tư:** S4 (vấn đề thực tế chưa được giải quyết tốt) và S7 (chứng minh giá trị của captioning so với classification thuần). S8 không cần model — nó được xử lý bằng health check tín hiệu ở tầng ingest và phải được chứng minh là **không** đi vào model.

## 3.3 Nguồn dữ liệu

| Nguồn | Loại nhãn | Dùng cho lớp | Ghi chú license |
|---|---|---|---|
| **AudioSet (temporally-strong)** | Strong | 1,2,3,6,7,8,9,10,11,13,14,15 | Nhãn CC-BY của Google; audio là link YouTube → **chỉ phát hành script tái tạo, không phát hành audio** |
| **MIVIA Audio Events** | Strong + thang SNR | 1,3,4 | Cần đăng ký với đơn vị phát hành — **nộp đơn ngày đầu W1** |
| **MIVIA Road Audio Events** | Strong | 5 | Như trên |
| **DESED** | Strong (synthetic + validation) | 10,12,14,15 | Lớp `Dishes`, `Alarm_bell_ringing`, `Speech` dùng trực tiếp |
| **FSD50K** | Weak (clip-level) | 4,7,8,11,12 | CC; dùng làm **foreground bank** cho Scaper (clip đã trim = strong ngầm định) |
| **UrbanSound8K** | Weak + salience | 9,15 | Dùng cho `siren`, nền giao thông |
| **ESC-50** | Weak | bổ sung Nhóm B | Clip 5s sạch, tốt cho foreground bank |
| **AudioCaps / Clotho** | Caption | — | Pretrain/fine-tune nhánh captioning |
| **Thu tại chỗ (IUH)** | Strong (tự gán) | background, S3, S4 | Cần đồng thuận; ghi vào `data_inventory.md` |

> ⚠️ **Bắt buộc xác minh ở W1–W2:** số giờ thực tế và điều khoản license của **từng** nguồn. Không giả định. Kết quả ghi vào `docs/data_inventory.md` do `scripts/data_inventory.py` sinh.

## 3.4 Chiến lược dữ liệu: giải bài toán "strong label thì đắt"

Mâu thuẫn trung tâm: strong label là bắt buộc, nhưng gán tay tốn 5–10× realtime. Lời giải là **ba tập, ba cơ chế nhãn khác nhau**:

| Tập | Cơ chế nhãn | Quy mô mục tiêu | Vai trò |
|---|---|---|---|
| **Train** | **Scaper sinh tự động** — mix foreground lên background với SNR/overlap/thời điểm do ta điều khiển ⇒ strong label chính xác tuyệt đối, miễn phí | 15–25 giờ | Huấn luyện chính |
| **Dev** | Strong label công khai có sẵn (AudioSet-strong, DESED, MIVIA) | 3–5 giờ | Tuning, early stopping |
| **Test (Gold)** | **Người gán mù**, gán lại 20% sau ≥7 ngày để đo tự-nhất-quán | ~1 giờ | **Con số duy nhất được báo cáo** |

Scaper chính là công cụ mà DESED dùng để sinh soundscape có strong label. Nó giải quyết đồng thời hai việc: chi phí nhãn, **và** khả năng tạo có chủ đích các slice S1, S2, S3, S6, S7 (ta điều khiển được overlap, SNR, RIR, thời lượng, thứ tự sự kiện). Đó là lý do nó nằm ở trung tâm chiến lược chứ không phải một mẹo tiết kiệm.

**Hạn chế phải nói thẳng:** dữ liệu tổng hợp có domain gap so với thu thật. Cách xử lý: (a) test set **bắt buộc** là audio thật; (b) báo cáo tường minh chênh lệch `synthetic-dev` vs `real-test` như một **kết quả nghiên cứu**, không che giấu.

## 3.5 Label schema

Giữ **đúng chuẩn DCASE** cho file nhãn chính để dùng trực tiếp `sed_eval` / `psds_eval`, không phải viết lại metric.

**`data/annotations/{split}.tsv`** — tab-separated, một dòng một event:

```
filename	onset	offset	event_label
hallway_003.wav	1.250	2.780	glass_breaking
hallway_003.wav	2.410	5.630	scream
hallway_003.wav	3.900	8.150	running_footsteps
```

Mọi thứ khác đi vào file phụ, join bằng `filename`. **Tuyệt đối không thêm cột vào TSV nhãn** — sẽ vỡ tương thích công cụ.

**`data/annotations/{split}_metadata.csv`**

| Cột | Kiểu | Mô tả |
|---|---|---|
| `filename` | str | Khoá join |
| `duration` | float | Giây |
| `sample_rate` | int | Luôn 16000 sau chuẩn hoá |
| `source_dataset` | str | `audioset_strong` \| `mivia` \| `desed` \| `fsd50k` \| `scaper` \| `iuh_field` |
| `source_id` | str | ID gốc trong dataset nguồn (để tái tạo) |
| `license` | str | Điều khoản license của file |
| `is_synthetic` | bool | Do Scaper sinh hay audio thật |
| `snr_db` | float | Chỉ có với synthetic hoặc MIVIA |
| `rir_id` | str | RIR đã tích chập (nếu có) |
| `location_sim` | str | `school` \| `parking` \| `residential` \| `factory` \| `null` |
| `slice_flags` | str | Phân tách bằng `\|`, ví dụ `overlap\|low_snr\|media_playback` |
| `annotator_id` | str | Ai gán |
| `verified_by` | str | Lần gán nào được chốt, và ai chốt (với ca bất đồng giữa hai lần) |
| `split` | str | `train` \| `dev` \| `test` |

## 3.6 Quy ước gán nhãn

Viết thành `docs/annotation_guideline.md` **trước khi** gán bất kỳ file nào. Đây là thứ quyết định độ nhất quán và là một phụ lục đẹp của báo cáo.

| Quy tắc | Nội dung |
|---|---|
| **Onset** | Điểm năng lượng vượt nền **nghe được**, không phải điểm nhìn thấy trên spectrogram |
| **Offset** | Điểm âm thanh chìm hẳn vào nền; đuôi vang (reverb tail) **tính vào** event |
| **Collar đánh giá** | 200 ms cho onset; 20% độ dài event cho offset (chuẩn DCASE) |
| **Gộp** | Hai lần cùng loại cách nhau < 300 ms → gộp thành 1 event |
| **Loạt** | `gunshot` liên thanh: mỗi phát 1 event; > 5 phát trong 2s → gộp 1 event, gắn cờ `burst` |
| **Chồng lấn** | Gán đầy đủ mọi lớp đồng thời; không chọn "lớp trội" |
| **Không chắc** | Gán `ambient_noise` + ghi chú, **không** đoán bừa lớp nguy hiểm |
| **Media playback** | Nếu nghe rõ là phát từ loa (có méo, có nền phòng) → gán lớp sự kiện **và** cờ `media_playback` |

**Quy trình đảm bảo chất lượng:**
1. Pilot 30 clip → nghỉ ≥3 ngày → gán lại mù → tính tự-nhất-quán → sửa guideline → mới gán đại trà. (Bỏ bước pilot là sai lầm tốn kém nhất.)
2. Double-annotate 20% test set; bất đồng → trọng tài.
3. Báo cáo **tự-nhất-quán (test–retest) mức segment 1 giây, THEO TỪNG LỚP** + **độ lệch onset trung vị (ms)**.
   ⚠️ Dự án chỉ có một người gán → **không** được gọi con số này là kappa liên-người. Xem [DATA_PLAN §8](DATA_PLAN.md).
4. Ngưỡng chấp nhận: **tự-nhất-quán ≥ 0.75** và **onset lệch trung vị ≤ 100 ms**. Không đạt → làm rõ guideline, gán lại.

## 3.7 Chiến lược chia tập & chống rò rỉ

**Rò rỉ dữ liệu là rủi ro số một** với dataset âm thanh, vì các đoạn cắt từ **cùng một bản ghi gốc** rất giống nhau.

| Quy tắc | Lý do |
|---|---|
| Chia theo `source_id` gốc, **không** chia ngẫu nhiên theo clip | Hai đoạn từ cùng một video YouTube không được nằm ở hai tập khác nhau |
| Foreground bank dùng cho Scaper-train **không** được xuất hiện trong dev/test | Tránh model học thuộc mẫu foreground |
| Test set là **audio thật**, không có clip synthetic nào | Đảm bảo con số báo cáo phản ánh thực địa |
| Test set **không** được dùng để tuning bất kỳ tham số nào, kể cả ngưỡng quyết định | Ngưỡng chọn trên dev, khoá lại, rồi mới chạy test **một lần** |
| Script `scripts/check_leakage.py` chạy trong CI | Kiểm tra tự động, không dựa vào trí nhớ |

## 3.8 Bốn gold set

Sinh từ một **kịch bản mô phỏng 7 ngày**: gán `timestamp` + `location` (thuộc 4 khu vực) cho các clip gold → có ngay một cơ sở dữ liệu sự kiện có ngữ cảnh thật để đặt câu hỏi và **biết chắc đáp án đúng**. Đây là cách duy nhất để có gold answer không mơ hồ.

| Gold | File | Nội dung | Quy mô |
|---|---|---|---|
| **G1 — SED** | `data/gold/g1_sed.tsv` + `_metadata.csv` | Strong label, human-verified, có `slice_flags` | ~1 giờ audio |
| **G2 — Caption** | `data/gold/g2_captions.jsonl` | 3–5 reference/clip, song ngữ EN+VI | 200–300 clip |
| **G3 — RAG** | `data/gold/g3_rag_queries.jsonl` | `{question_vi, relevant_event_ids[], gold_answer_vi, query_type}` | 60–100 câu |
| **G4 — Alert** | `data/gold/g4_alerts.jsonl` | `{scenario_id, events[], expected_severity, is_true_alarm}` | 40–60 kịch bản |

**Phân bố loại truy vấn trong G3** (để không chỉ test một kiểu câu hỏi):

| Loại | Tỉ lệ | Ví dụ |
|---|---|---|
| Theo thời gian | 25% | *"Đêm qua từ 22h đến 24h có sự kiện bất thường nào?"* |
| Theo địa điểm | 20% | *"Bãi xe tuần này xảy ra chuyện gì?"* |
| Theo loại sự kiện | 20% | *"Lần gần nhất phát hiện tiếng kính vỡ là khi nào?"* |
| Theo mức rủi ro | 15% | *"Có cảnh báo Critical nào hôm nay không?"* |
| Kết hợp / tổng hợp | 15% | *"Tầng 2 trong tuần có bao nhiêu sự kiện liên quan đến cửa?"* |
| **Câu không có đáp án** | 5% | *"Có phát hiện tiếng nổ nào ở nhà xe không?"* (thực tế: không có) — **kiểm tra hệ thống có dám nói "không có" hay không** |

Nhóm cuối rất quan trọng: một hệ RAG tốt phải **từ chối bịa** khi không có bằng chứng.

## 3.9 Tiền xử lý audio

Pipeline chuẩn hoá, áp dụng đồng nhất cho train/dev/test và cho cả luồng streaming:

```
Audio thô
  → Resample 16 kHz          (chuẩn của BEATs và hầu hết pretrained encoder)
  → Mono (trung bình kênh)
  → Loudness normalization    (EBU R128, mục tiêu −23 LUFS) — KHÔNG peak-normalize
  → Cắt chunk 10s (train) / buffer 5s có overlap 2.5s (streaming)
  → BEATs feature @ ~50 Hz    (precompute ra .npy cho train/dev)
```

**Lưu ý quan trọng:** peak normalization phá thông tin về cường độ tương đối, mà cường độ lại là dấu hiệu phân biệt `gunshot` thật với `gunshot` phát từ TV (slice S4). Dùng loudness normalization và **lưu lại hệ số gain đã áp dụng** trong metadata để model có thể dùng nếu cần.

---

# 4. Kiến trúc hệ thống

## 4.1 Tổng quan

```
                          🎙️  NGUỒN ÂM THANH
              (micro thật / edge device / file replay simulator)
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │  services/stream         │
                    │  Redis Streams consumer  │
                    │  buffer 5s / overlap 2.5s│
                    │  health check tín hiệu   │  ← xử lý S8 tại đây
                    └────────────┬─────────────┘
                                 │  chunk PCM
                                 ▼
                    ┌──────────────────────────┐
                    │  services/inference      │
                    │  FastAPI + GPU           │
                    │  ┌────────────────────┐  │
                    │  │ BEATs (frozen)     │  │
                    │  │ Conv1D ↓ Conformer │  │  ← TRUNK CHIA SẺ
                    │  └───┬────────────┬───┘  │
                    │      │            │      │
                    │  SED head    BART decoder│
                    │  (frame)     (grounded)  │
                    └──────┬────────────┬──────┘
                           │            │
                    events @frame    caption
                           └─────┬──────┘
                                 ▼
                    ┌──────────────────────────┐
                    │  Temporal Aggregator     │  ← gộp chunk → 1 sự kiện
                    │  + Risk Scoring          │
                    └────────────┬─────────────┘
                                 │  SecurityEvent
                    ┌────────────┴─────────────┐
                    ▼                          ▼
          ┌──────────────────┐      ┌──────────────────────┐
          │  Alert Service   │      │  PostgreSQL          │
          │  WebSocket push  │      │  + pgvector          │
          │  (đường NÓNG)    │      │  (đường LỊCH SỬ)     │
          └────────┬─────────┘      └──────────┬───────────┘
                   │                           │
                   │                ┌──────────▼───────────┐
                   │                │  services/api        │
                   │                │  RAG: hybrid retrieval│
                   │                │  + LLM có trích dẫn  │
                   │                └──────────┬───────────┘
                   └────────────┬──────────────┘
                                ▼
                    ┌──────────────────────────┐
                    │  services/frontend       │
                    │  Live feed │ Event detail│
                    │  │ RAG chat              │
                    └──────────────────────────┘

        ═══════ MLOps chạy xuyên suốt ═══════
        DVC · MLflow · Model Registry · CI · Monitoring · Feedback
```

**Nguyên tắc kiến trúc số một: tách đường NÓNG và đường LỊCH SỬ.** Cảnh báo tức thời đi thẳng từ risk scoring ra WebSocket, **không đi qua LLM**. RAG chỉ phục vụ truy vấn lịch sử. Nhiều hệ thống mắc lỗi dùng RAG để phát cảnh báo — kết quả là vừa chậm (LLM latency) vừa không tin cậy (LLM có thể bịa).

## 4.2 Danh sách service

| Service | Công nghệ | Cổng | Trách nhiệm |
|---|---|---|---|
| `stream` | Python + FastAPI + redis-py | — | Ingest, buffer, health check, gọi inference, aggregate, risk scoring |
| `inference` | FastAPI + PyTorch (GPU) | 8001 | Load model 1 lần, phục vụ SED + captioning; batch động |
| `api` | FastAPI + SQLAlchemy + pgvector | 8000 | CRUD event, RAG, auth, feedback, WebSocket alert |
| `frontend` | React + Vite | 3000 | Live feed, event detail + phát lại audio, RAG chat |
| `db` | PostgreSQL 16 + pgvector | 5432 | Event store + vector index |
| `redis` | Redis 7 | 6379 | Redis Streams (broker) + cache |
| `mlflow` | MLflow | 5000 | Experiment tracking + model registry |

**Vì sao FastAPI toàn bộ:** async-native (bắt buộc cho streaming), một framework cho cả ba service Python, Pydantic validate ở mọi biên hệ thống (khớp nguyên tắc "validate at system boundaries"), và tránh ma sát của Django sync với workload audio.

**Vì sao Redis Streams thay vì Kafka:** Redis đã có trong stack cho cache; Redis Streams là broker thật, có consumer group và ack; zero chi phí vận hành thêm. Kafka là over-engineering ở quy mô một node. Nếu cần chứng minh khả năng mở rộng, viết một **adapter MQTT** (stretch goal) chứ không đổi broker.

**Vì sao pgvector thay vì Qdrant:** bớt một service; và quan trọng hơn — RAG này cần **lọc metadata (khoảng thời gian, địa điểm, mức rủi ro) kết hợp tìm kiếm ngữ nghĩa**. Với pgvector, cả hai nằm trong **một câu SQL**, không phải hai lượt gọi rồi tự merge. Đó là lợi thế kiến trúc, không phải sự tiện tay.

## 4.3 Lược đồ cơ sở dữ liệu

```sql
-- Sự kiện an ninh: đơn vị dữ liệu trung tâm của toàn hệ thống
CREATE TABLE security_events (
    event_id        TEXT PRIMARY KEY,            -- EVT_YYYYMMDD_NNNNNN
    window_start    TIMESTAMPTZ NOT NULL,
    window_end      TIMESTAMPTZ NOT NULL,
    location_id     TEXT NOT NULL REFERENCES locations(location_id),
    caption_en      TEXT NOT NULL,
    caption_vi      TEXT NOT NULL,
    severity        TEXT NOT NULL,               -- LOW|MEDIUM|HIGH|CRITICAL
    risk_score      REAL NOT NULL,
    audio_path      TEXT,                        -- NULL nếu retention đã xoá
    embedding       VECTOR(1024),                -- pgvector, embed caption_vi
    grounding_score REAL,                        -- xem §8.2
    model_versions  JSONB NOT NULL,              -- {"trunk":"v1.2","sed":"v1.2","aac":"v2.0"}
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Từng detection trong một sự kiện (quan hệ 1-n)
CREATE TABLE event_detections (
    id           BIGSERIAL PRIMARY KEY,
    event_id     TEXT REFERENCES security_events(event_id) ON DELETE CASCADE,
    class_id     TEXT NOT NULL,                  -- 1 trong 15 class
    onset_sec    REAL NOT NULL,                  -- tương đối so với window_start
    offset_sec   REAL NOT NULL,
    confidence   REAL NOT NULL
);

-- Phản hồi của người vận hành → nhiên liệu cho retraining
CREATE TABLE event_feedback (
    id             BIGSERIAL PRIMARY KEY,
    event_id       TEXT REFERENCES security_events(event_id),
    is_true_alarm  BOOLEAN,
    correct_classes TEXT[],                      -- nhãn đúng nếu model sai
    caption_ok     BOOLEAN,                      -- caption có bịa không
    note           TEXT,
    reviewed_by    TEXT,
    reviewed_at    TIMESTAMPTZ DEFAULT now()
);

-- Giám sát production
CREATE TABLE inference_metrics (
    id             BIGSERIAL PRIMARY KEY,
    ts             TIMESTAMPTZ DEFAULT now(),
    model_version  TEXT,
    latency_ms     INTEGER,
    stage          TEXT,                         -- sed|caption|aggregate|e2e
    mean_confidence REAL,
    class_histogram JSONB
);

CREATE TABLE locations (
    location_id  TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    area_type    TEXT NOT NULL,   -- school|parking|residential|factory
    description  TEXT
);

CREATE INDEX ON security_events USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON security_events (window_start DESC);
CREATE INDEX ON security_events (location_id, window_start DESC);
CREATE INDEX ON security_events (severity, window_start DESC);
```

## 4.4 Hợp đồng API

Tiền tố `/api/v1`. Mọi response dùng envelope thống nhất `{success, data, error, meta}`.

| Nhóm | Endpoint | Method | Mô tả |
|---|---|---|---|
| Ingest | `/audio/chunk` | POST | Nhận chunk audio (dùng khi không qua Redis Streams) |
| Ingest | `/audio/upload` | POST | Upload file để xử lý offline |
| Events | `/events` | GET | Lọc theo `from`, `to`, `location`, `severity`, `class`, phân trang |
| Events | `/events/{event_id}` | GET | Chi tiết + detections + đường dẫn audio |
| Events | `/events/{event_id}/audio` | GET | Stream audio để phát lại |
| Alerts | `/alerts/stream` | WS | WebSocket push cảnh báo real-time |
| RAG | `/rag/query` | POST | `{question, from?, to?, location?}` → `{answer, citations[], retrieved_events[]}` |
| Feedback | `/events/{event_id}/feedback` | POST | Xác nhận true/false alarm, sửa nhãn |
| Models | `/models/status` | GET | Model version đang active, thời điểm train, metric |
| Models | `/models/retrain` | POST | Trigger retrain (yêu cầu quyền admin) |
| Metrics | `/metrics/summary` | GET | Số liệu cho dashboard + biểu đồ drift |
| Health | `/health` | GET | Liveness/readiness cho Docker healthcheck |

**Ràng buộc bắt buộc:** `/rag/query` **luôn** trả `citations[]` là danh sách `event_id`. Câu trả lời không có citation bị coi là lỗi hệ thống, không phải câu trả lời hợp lệ.

## 4.5 Bảng công nghệ

| Tầng | Lựa chọn | Ghi chú |
|---|---|---|
| Audio encoder | **BEATs (đóng băng)** | Precompute feature ra `.npy` |
| Trunk | Conv1D downsample + **Conformer** | 50 Hz → ~17 Hz |
| Caption decoder | **BART-base** | Cross-attention từ trunk |
| Text embedder (contrastive) | Sentence embedder đóng băng (Instructor / E5) | Chỉ dùng khi huấn luyện |
| Embedding cho RAG | **BGE-M3** hoặc multilingual-E5 | Bắt buộc đa ngữ — xem §7.4 |
| SED metric | `sed_eval`, `psds_eval` | Không tự viết lại |
| Caption metric | `aac-metrics` (BLEU/METEOR/CIDEr/SPICE/SPIDEr/FENSE) | + metric hallucination tự định nghĩa |
| Sinh dữ liệu | **Scaper** | Soundscape + strong label tự động |
| Gán nhãn | **Label Studio** | Audio region labeling, đa annotator |
| Backend | **FastAPI** + Pydantic v2 + SQLAlchemy 2.0 | |
| DB | **PostgreSQL 16 + pgvector** | |
| Broker | **Redis Streams** | |
| Frontend | React 18 + Vite + TanStack Query | |
| LLM (RAG) | Interface pluggable: Claude API (mặc định) / Ollama local (fallback) | Không khoá cứng nhà cung cấp |
| Tracking | **MLflow** | |
| Data version | **DVC** | |
| Triển khai | **Docker Compose** | Mirror pattern của `example_project` |
| CI | GitHub Actions | |

---

# 5. Mô hình: SED + Grounded Audio Captioning

Đây là chương đóng góp nghiên cứu chính.

## 5.1 Ý tưởng cốt lõi

Kiến trúc AAC hiện đại (BEATs đóng băng → Conformer → BART decoder, có thêm mục tiêu contrastive InfoNCE với text embedder đóng băng) đạt chất lượng ngôn ngữ tốt, nhưng **không có cơ chế nào bắt buộc caption phải đúng với bằng chứng âm học**. Decoder được huấn luyện bằng NLL nên mang prior ngôn ngữ mạnh; khi tín hiệu yếu, nó rơi về prior và bịa.

Quan sát then chốt của đề tài: **chúng ta có strong label**. Nhãn `onset/offset` không chỉ để đánh giá SED — nó là **tín hiệu giám sát mức frame** có thể cắm thẳng vào trunk của captioner. Từ đó:

> **Đề xuất:** chia sẻ một trunk duy nhất giữa SED và captioning; dùng strong label để giám sát mức frame; dùng chính đầu ra SED đó để **ràng buộc từ vựng của decoder lúc giải mã**.

Ba lợi ích cùng lúc: (1) grounding cho caption; (2) một trunk thay vì hai model (tiết kiệm — then chốt cho 8 tuần); (3) SED head và caption head **nhất quán với nhau theo thiết kế**, không thể mâu thuẫn như khi chạy hai model rời.

## 5.2 Kiến trúc

```
                    audio 10s @16kHz
                          │
                 ┌────────▼────────┐
                 │  BEATs encoder  │  ❄️ ĐÓNG BĂNG (precompute → .npy)
                 └────────┬────────┘
                     ~50 Hz frames
                          │
                 ┌────────▼────────┐
                 │ Conv1D downsample│
                 └────────┬────────┘
                     ~17 Hz frames  (≈58.8 ms/frame)
                          │
                 ┌────────▼────────┐
                 │    Conformer    │  🔥 huấn luyện
                 └────────┬────────┘
                          │  H ∈ ℝ^{T×d},  T ≈ 170
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌────────────────┐
│  HEAD 1: SED  │ │ HEAD 2: Align │ │ HEAD 3: Caption│
│ Linear→sigmoid│ │ Global mean   │ │ cross-attn →   │
│ per frame     │ │ pool → proj   │ │ BART decoder   │
│  ŷ ∈ ℝ^{T×15} │ │  z_a ∈ ℝ^{d'} │ │  → tokens      │
└───────┬───────┘ └───────┬───────┘ └────────┬───────┘
        │                 │                  │
   L_SED (BCE)      L_InfoNCE ⟷ z_t      L_NLL
   ↑ strong label   ↑ Instructor ❄️      ↑ caption
        │
        └──────────► GROUNDED DECODING (lúc suy luận) ──────┐
                                                             ▼
                                              chặn/phạt token sự kiện
                                              không có bằng chứng
```

**Head 1 — SED mức frame.** Linear + sigmoid trên mỗi frame → $\hat{y} \in \mathbb{R}^{T \times 15}$, multi-label (cho phép chồng lấn). Giám sát bằng strong label chuyển thành ma trận nhị phân mức frame. Độ phân giải 17 Hz ⇒ 58.8 ms/frame, so với collar 200 ms là **~3.4 frame** — đủ mịn.

**Head 2 — Căn chỉnh audio–text.** Global mean pool trên $H$ → chiếu về không gian embedding văn bản → $z_a$. Caption tham chiếu đi qua text embedder **đóng băng** → $z_t$. InfoNCE trong batch: kéo cặp đúng lại gần, đẩy cặp sai ra xa. Mục đích: ép biểu diễn audio nằm trong không gian ngữ nghĩa của caption, giảm caption drift.

**Head 3 — Caption decoder.** BART-base, cross-attention lên $H$ (chuỗi 17 Hz, không phải vector gộp — giữ được thông tin thời gian). NLL trên caption tham chiếu.

## 5.3 Hàm mất mát

$$\mathcal{L} = \lambda_{1}\,\mathcal{L}_{\text{SED}} + \lambda_{2}\,\mathcal{L}_{\text{InfoNCE}} + \lambda_{3}\,\mathcal{L}_{\text{NLL}}$$

| Thành phần | Dạng | Ghi chú |
|---|---|---|
| $\mathcal{L}_{\text{SED}}$ | Binary cross-entropy mức frame, có trọng số lớp | Trọng số theo nghịch đảo tần suất, chặn trên để lớp hiếm không lấn át |
| $\mathcal{L}_{\text{InfoNCE}}$ | Contrastive hai chiều, nhiệt độ $\tau$ học được | Negative lấy trong batch |
| $\mathcal{L}_{\text{NLL}}$ | Cross-entropy token, label smoothing 0.1 | |

Khởi tạo $\lambda = (1.0,\ 0.5,\ 1.0)$; **tinh chỉnh $\lambda$ là một thí nghiệm ablation, không phải một hằng số tuỳ tiện** (thí nghiệm E3, [§10](#10-kế-hoạch-thực-nghiệm)).

**Lịch huấn luyện 2 giai đoạn** (quan trọng — huấn luyện cả ba cùng lúc từ đầu sẽ không hội tụ tốt):

| Giai đoạn | Bật | Mục đích |
|---|---|---|
| **Stage 1** | $\mathcal{L}_{\text{SED}}$ + $\mathcal{L}_{\text{InfoNCE}}$ | Trunk học biểu diễn có định vị thời gian và căn chỉnh ngữ nghĩa trước |
| **Stage 2** | cả ba, LR trunk giảm 10× | Decoder học sinh caption trên trunk đã ổn định |

## 5.4 Grounded decoding

Đây là can thiệp **lúc suy luận**, không cần huấn luyện thêm, và là phần dễ giải thích nhất trước hội đồng.

**Xây `EVENT_LEXICON`:** ánh xạ mỗi `class_id` → tập từ/cụm từ tiếng Anh chỉ sự kiện đó.

```python
EVENT_LEXICON = {
    "gunshot":        ["gunshot", "gunfire", "shot", "shooting", "firearm"],
    "glass_breaking": ["glass", "shatter", "shattering", "breaking glass"],
    "scream":         ["scream", "screaming", "shriek"],
    ...
}
```

**Thuật toán:**

1. Từ Head 1 lấy điểm số mức clip cho mỗi lớp: $s_c = \max_t \hat{y}_{t,c}$ (hoặc top-k mean để bớt nhạy nhiễu).
2. Chia ba vùng theo hai ngưỡng $\theta_{low} < \theta_{high}$ (chọn trên **dev**, khoá lại):
   - $s_c \ge \theta_{high}$ → **được phép** dùng từ vựng lớp $c$.
   - $\theta_{low} \le s_c < \theta_{high}$ → **phạt logit** $-\alpha$ (mềm, cho phép nếu ngữ cảnh rất mạnh).
   - $s_c < \theta_{low}$ → **chặn cứng** (logit $= -\infty$) toàn bộ token khởi đầu của các cụm từ trong `EVENT_LEXICON[c]`.
3. Beam search với logit processor trên.
4. **Cơ chế dự phòng:** nếu sau khi chặn, caption sinh ra rỗng hoặc không nhắc tới bất kỳ sự kiện nào trong khi $\exists c: s_c \ge \theta_{high}$ → rơi về **structured caption** sinh từ template trên timeline SED. Hệ thống **không bao giờ** trả caption rỗng.

**Chi phí phải thừa nhận:** grounded decoding có thể làm giảm CIDEr (vì chặn bớt lựa chọn từ ngữ) trong khi làm giảm mạnh hơn EHR. Chính **đường đánh đổi CIDEr ↔ EHR** khi quét $\theta$ là kết quả nghiên cứu quan trọng nhất của đề tài, chứ không phải một con số đơn lẻ.

## 5.5 Baseline & Ablation

| Ký hiệu | Model | Vai trò |
|---|---|---|
| **B0** | Structured captioner: timeline SED → template → (tuỳ chọn) LLM viết lại | Cận trên về grounding (EHR ≈ 0), cận dưới về tự nhiên |
| **B1** | BEATs → Conv1D → Conformer → BART, **chỉ NLL** | Baseline AAC thuần, tái hiện kiến trúc gốc |
| **B2** | B1 + InfoNCE | Kiểm chứng đóng góp của nhánh contrastive |
| **B3** | B1 + SED head (đa nhiệm) | Kiểm chứng đóng góp của grounding supervision |
| **P** | **B1 + InfoNCE + SED head + grounded decoding** | **Mô hình đề xuất** |
| **P−gd** | P nhưng tắt grounded decoding | Tách riêng đóng góp của can thiệp lúc suy luận |

Đây là ma trận ablation chuẩn: mỗi thành phần được bật/tắt độc lập, nên có thể quy công cho từng phần.

## 5.6 Chi tiết huấn luyện

| Tham số | Giá trị khởi điểm |
|---|---|
| Độ dài clip | 10 s (pad/crop) |
| Batch size | 32 (giới hạn bởi InfoNCE — cần đủ negative) |
| Optimizer | AdamW, weight decay 0.01 |
| LR | Trunk 1e-4, decoder 5e-5, warmup 5% + cosine |
| Precision | AMP (bf16 nếu GPU hỗ trợ) |
| Epoch | Stage 1: 30, Stage 2: 20 |
| Early stopping | Trên dev: `SPIDEr − EHR` (chọn model cân bằng, không chỉ chạy theo SPIDEr) |
| Augmentation | SpecAugment, mixup mức waveform, dịch thời gian, thay đổi SNR |
| Seed | Cố định 42; chạy 3 seed cho kết quả cuối, báo cáo mean ± std |

**Tối ưu chi phí quyết định:** precompute đặc trưng BEATs ra `.npy` một lần cho toàn bộ train/dev. Sau đó mỗi epoch chỉ chạy Conformer + BART — nhanh hơn nhiều lần và giảm mạnh nhu cầu VRAM. Đây là lý do một khoá luận 8 tuần có thể chạy được kiến trúc này.

**Điểm dừng thành thật:** nếu **P** không thắng **B1** về EHR ở cùng mức CIDEr, đó vẫn là một kết quả và **phải được báo cáo trung thực**, kèm phân tích nguyên nhân. Kết quả âm tính có phân tích tốt vẫn là khoá luận tốt; kết quả dương tính do chọn lọc số liệu thì không.

---

# 6. Xử lý sự kiện, đánh giá rủi ro & cảnh báo

## 6.1 Streaming & buffering

| Tham số | Giá trị | Lý do |
|---|---|---|
| Cửa sổ buffer | 5 s | Cân bằng ngữ cảnh và độ trễ |
| Overlap | 2.5 s (50%) | **Xử lý slice S5** — event bị cắt ở biên vẫn nằm trọn trong ít nhất một cửa sổ |
| Cửa sổ suy luận | 10 s (2 buffer liền kề) | Khớp độ dài huấn luyện |
| Gate năng lượng | RMS + VAD nhẹ | Bỏ qua đoạn im lặng → tiết kiệm GPU |
| Health check | Phát hiện im lặng tuyệt đối / clipping / DC offset | **Xử lý slice S8** trước khi vào model |

## 6.2 Temporal Aggregator

Thành phần quan trọng nhất của tầng streaming. Nếu thiếu nó, một vụ đột nhập kéo dài 20 giây sẽ trở thành 8 sự kiện rời rạc, không ai đọc nổi và RAG cũng không truy xuất đúng.

**Thuật toán:**

1. **Khử trùng lặp giữa các cửa sổ chồng lấn.** Cùng `class_id`, khoảng thời gian IoU > 0.5 → gộp, lấy hợp của khoảng và max của confidence.
2. **Gom cụm theo thời gian.** Các detection cách nhau < `GAP_THRESHOLD` (mặc định 3 s) thuộc về cùng một cụm.
3. **Đóng cụm** khi: không có detection mới trong `IDLE_TIMEOUT` (5 s), **hoặc** cụm đã dài quá `MAX_EVENT_DURATION` (60 s → cắt và mở cụm mới, **xử lý slice S6**).
4. **Sinh caption** cho cụm đã đóng, dùng cửa sổ audio phủ toàn cụm.
5. **Phát ra một `SecurityEvent`** duy nhất.

**Cảnh báo sớm:** với sự kiện tier `Critical`, alert được phát **ngay khi phát hiện**, không đợi cụm đóng. Cụm đóng sau đó chỉ bổ sung caption và cập nhật sự kiện. Không được để nạn nhân chờ aggregator.

## 6.3 Lược đồ Security Event

```json
{
  "event_id": "EVT_20260914_000123",
  "window": { "start": "2026-09-14T22:31:05+07:00", "end": "2026-09-14T22:31:22+07:00" },
  "location": { "id": "HALL_03", "name": "Hành lang tầng 2 nhà B", "area_type": "school" },
  "detections": [
    { "class_id": "glass_breaking", "onset": 1.20, "offset": 2.80, "confidence": 0.93 },
    { "class_id": "scream",         "onset": 3.10, "offset": 5.40, "confidence": 0.87 },
    { "class_id": "running_footsteps", "onset": 4.90, "offset": 11.30, "confidence": 0.78 }
  ],
  "caption_en": "A glass-breaking sound is followed by a person screaming and rapid footsteps.",
  "caption_vi": "Tiếng kính vỡ, sau đó là tiếng hét và tiếng bước chân chạy nhanh.",
  "caption_source": "grounded_aac",
  "grounding_score": 1.00,
  "risk_score": 0.91,
  "severity": "CRITICAL",
  "audio_path": "s3://events/2026/09/14/EVT_20260914_000123.wav",
  "model_versions": { "trunk": "v1.2", "sed": "v1.2", "aac": "v2.0", "embed": "bge-m3" },
  "slice_flags": [],
  "feedback": { "is_true_alarm": null, "reviewed_by": null }
}
```

Trường `feedback` để `null` ngay từ đầu — đó là chỗ vòng lặp MLOps cắm vào.

## 6.4 Risk scoring

Rule-based ở phiên bản này. Lý do: minh bạch, giải thích được trước hội đồng, và **không có dữ liệu nhãn rủi ro thật** để huấn luyện một model rủi ro. Nâng cấp ML-based là hướng phát triển, sau khi thu đủ feedback.

**Công thức:**

$$R = \min\left(1,\ \max_{i}\left(w_{c_i} \cdot p_i\right) + \beta \cdot \text{combo}(\mathcal{C}_{\text{event}}) + \gamma \cdot \text{ctx}\right)$$

| Thành phần | Nội dung |
|---|---|
| $w_c$ | Trọng số nghiêm trọng theo lớp: Critical 1.0 · High 0.7 · Medium 0.4 · Nhóm B 0.05 |
| $p_i$ | Confidence của detection |
| $\text{combo}$ | Thưởng khi các lớp xuất hiện cùng nhau: `glass_breaking + scream` +0.20; `+ running_footsteps` thêm +0.10; `gunshot + scream` +0.25 |
| $\text{ctx}$ | Ngữ cảnh: ban đêm (22h–05h) +0.10; khu vực hạn chế +0.10; tần suất bất thường trong 1h +0.05 |
| $\beta, \gamma$ | Hệ số điều chỉnh, chọn trên **dev**, khoá lại trước khi chạy test |

**Ánh xạ ngưỡng:** `LOW < 0.35 ≤ MEDIUM < 0.60 ≤ HIGH < 0.80 ≤ CRITICAL`.

**Quy tắc chặn (guard rails) — quan trọng hơn công thức:**

| Quy tắc | Nội dung |
|---|---|
| **Chặn Nhóm B** | Nếu lớp trội thuộc Nhóm B (`fireworks`, `applause_cheering`, …) và không có lớp Nhóm A nào vượt $\theta_{high}$ → **ép severity ≤ LOW** |
| **Chặn media playback** | Nếu bộ phân loại phụ đánh dấu `media_playback` → **giảm một bậc severity** và ghi rõ lý do vào sự kiện |
| **Chặn confidence thấp** | Mọi detection có $p < \theta_{low}$ không được tham gia tính risk |
| **Chống spam** | Cùng lớp, cùng địa điểm, lặp > N lần trong 10 phút → gộp thành một cảnh báo "lặp lại", không bắn N thông báo |

Bốn quy tắc này là nơi thực sự kiểm soát False Alarm Rate, nhiều hơn bản thân công thức.

## 6.5 Alert Service

| Kênh | Cơ chế | Dùng khi |
|---|---|---|
| Dashboard | WebSocket push | Mọi severity ≥ MEDIUM |
| Âm thanh + hiệu ứng UI | Frontend | CRITICAL |
| (Stretch) Telegram/Email | Webhook | CRITICAL |

Mọi cảnh báo đều mang `event_id`, cho phép người vận hành mở chi tiết, **nghe lại audio gốc**, và xác nhận đúng/sai ngay trên giao diện — thao tác xác nhận này chính là dữ liệu cho `event_feedback`.

---

# 7. Tầng RAG

## 7.1 Indexing

Mỗi `SecurityEvent` khi được ghi sẽ đồng thời được index:

```
caption_vi  →  BGE-M3  →  vector(1024)  →  cột embedding (pgvector, HNSW)
metadata    →  cột có index B-tree (window_start, location_id, severity)
detections  →  bảng riêng, join khi cần lọc theo class
```

**Chọn embed `caption_vi` chứ không phải `caption_en`** — xem [§7.4](#74-xử-lý-song-ngữ).

## 7.2 Hybrid retrieval

Truy vấn an ninh gần như luôn có **ràng buộc cứng** (khoảng thời gian, địa điểm) lẫn **ý định mềm** (ngữ nghĩa). Vector search thuần sẽ trả về sự kiện giống về mặt ngữ nghĩa nhưng sai ngày.

```
Câu hỏi tiếng Việt
   │
   ├─► Query Parser (LLM + regex)
   │      trích: khoảng thời gian, địa điểm, lớp sự kiện, mức rủi ro
   │      "đêm qua" → [2026-09-13T18:00, 2026-09-14T06:00]
   │
   ├─► Ràng buộc cứng → mệnh đề WHERE
   │
   └─► Embed câu hỏi → tìm kiếm vector
              │
              ▼
   MỘT câu SQL: WHERE + ORDER BY embedding <=> :q LIMIT k
              │
              ▼
   Rerank (tuỳ chọn) → Top-k bằng chứng
              │
              ▼
   LLM sinh câu trả lời CÓ TRÍCH DẪN
```

**Cấu hình:** $k = 10$ khi truy xuất, đưa tối đa 8 sự kiện vào prompt. Nếu bộ lọc cứng trả về 0 kết quả → **trả lời "không tìm thấy sự kiện nào phù hợp"**, tuyệt đối không nới lỏng bộ lọc để có cái gì đó mà trả lời.

## 7.3 Sinh câu trả lời có ràng buộc

Nguyên tắc prompt (chi tiết trong `services/api/rag/prompts.py`):

| Ràng buộc | Nội dung |
|---|---|
| Chỉ dùng bằng chứng | Chỉ được dùng thông tin trong danh sách sự kiện truy xuất được |
| Bắt buộc trích dẫn | Mỗi khẳng định phải kèm `event_id` |
| Cho phép nói không biết | Không có bằng chứng → phải trả lời "không tìm thấy", **không suy đoán** |
| Không suy diễn nguyên nhân | Được mô tả *"phát hiện tiếng kính vỡ"*, **không** được kết luận *"có kẻ đột nhập"* |
| Số liệu phải đếm được | Khi trả lời "có N sự kiện", N phải bằng đúng số citation |

Ràng buộc thứ tư quan trọng về mặt đạo đức và pháp lý: hệ thống báo cáo **bằng chứng âm học**, không kết tội.

## 7.4 Xử lý song ngữ

Vấn đề: caption benchmark là **tiếng Anh** (để so sánh với AudioCaps/Clotho), nhưng người dùng hỏi bằng **tiếng Việt**. Embed caption EN rồi query VI sẽ làm retrieval sụt nghiêm trọng. Đây là chi tiết nhỏ nhưng bỏ qua thì hỏng cả tầng RAG.

**Giải pháp:**

| Bước | Xử lý |
|---|---|
| Sinh caption | Model sinh **EN** (giữ so sánh được với benchmark) |
| Dịch | Sinh `caption_vi` bằng LLM, có **glossary cố định 15 class** để thuật ngữ nhất quán tuyệt đối |
| Index | Embed `caption_vi` bằng model đa ngữ (BGE-M3) |
| Đánh giá caption | Trên `caption_en` với metric chuẩn |
| Đánh giá RAG | Trên `caption_vi` với truy vấn tiếng Việt |
| Kiểm tra | So khớp thuật ngữ EN↔VI theo glossary; sai lệch là lỗi phải sửa |

Glossary 15 class là bắt buộc: nếu cùng một lớp lúc dịch là "tiếng kính vỡ", lúc là "tiếng vỡ kính", lúc là "kính bị đập", thì retrieval sẽ phân mảnh.

---

# 8. Phương pháp đánh giá

## 8.1 Bốn tầng metric

| Tầng | Gold | Metric chính | Metric phụ | Công cụ |
|---|---|---|---|---|
| **L1 — SED** | G1 | Event-based F1 (collar 200 ms) | PSDS, segment F1, onset MAE, per-class F1 | `sed_eval`, `psds_eval` |
| **L2 — Captioning** | G2 | SPIDEr | BLEU-4, METEOR, ROUGE-L, CIDEr, SPICE, FENSE | `aac-metrics` |
| **L2b — Grounding** | G1+G2 | **EHR** | EOR, GS, TOA | **Tự cài** — [§8.2](#82-bộ-metric-hallucination-đề-xuất) |
| **L3 — Retrieval** | G3 | Recall@5 | Precision@5, MRR, nDCG@10 | Tự cài |
| **L4 — End-to-end** | G3+G4 | Alert F1 | Alert Precision/Recall, **FAR**, Faithfulness, Answer Correctness | Tự cài + human + LLM-judge |
| **L5 — Hệ thống** | — | Latency P95 | P50/P99, throughput, VRAM, độ chính xác biên aggregator | Tự cài |

## 8.2 Bộ metric hallucination đề xuất

Đóng góp **C2**. Ý tưởng: vì có strong label, ta biết chính xác tập sự kiện có thật $G$ và thứ tự của chúng. So sánh với tập sự kiện **được nhắc tới trong caption** $P$ (trích bằng `EVENT_LEXICON` + đối sánh cụm từ).

| Metric | Công thức | Ý nghĩa | Chiều tốt |
|---|---|---|---|
| **EHR** (Event Hallucination Rate) | $\dfrac{\lvert P \setminus G\rvert}{\lvert P \rvert}$ | Tỉ lệ sự kiện được nhắc **mà không có thật** (dạng H1) | ↓ |
| **EOR** (Event Omission Rate) | $\dfrac{\lvert G \setminus P\rvert}{\lvert G \rvert}$ | Tỉ lệ sự kiện có thật **bị bỏ sót** (dạng H2) | ↓ |
| **GS** (Grounding Score) | $\dfrac{\lvert P \cap G\rvert}{\lvert P \cup G\rvert}$ | Jaccard giữa sự kiện nhắc tới và sự kiện có thật | ↑ |
| **TOA** (Temporal Order Accuracy) | tỉ lệ cặp $(a,b) \in (P \cap G)^2$ có thứ tự trong caption khớp thứ tự onset thật | Đo dạng H3 | ↑ |
| **CHR** (Critical Hallucination Rate) | EHR chỉ tính trên các lớp tier Critical | Bịa "tiếng súng" nguy hiểm hơn bịa "bước chân" | ↓ |

**Cách trích $P$ từ caption:** đối sánh cụm từ theo `EVENT_LEXICON`, có xử lý phủ định (*"no gunshots"* → không tính là nhắc tới) và biến thể hình thái. Bộ trích này **phải được kiểm định thủ công trên 100 caption**, báo cáo độ chính xác của chính bộ trích — nếu không, metric mới sẽ không đáng tin. Đây là điều kiện cần để C2 được chấp nhận.

**Kết quả trung tâm cần trình bày:** đường cong **CIDEr ↔ EHR** khi quét ngưỡng grounded decoding $\theta$, so sánh giữa B1, B2, B3, P. Nếu P nằm ở phía trên-trái của B1 (cùng CIDEr nhưng EHR thấp hơn, hoặc cùng EHR nhưng CIDEr cao hơn), đóng góp C1 được chứng minh.

## 8.3 Giao thức báo cáo theo slice

Mọi metric của L1, L2, L2b **phải được báo cáo cả tổng thể và theo từng slice**:

|  | Overall | S1 overlap | S2 low_snr | S3 reverb | S4 media | S5 boundary | S6 long | S7 chain |
|---|---|---|---|---|---|---|---|---|
| Event-F1 | | | | | | | | |
| SPIDEr | | | | | | | | |
| EHR | | | | | | | | |

Một F1 tổng thể che giấu hoàn toàn việc model hỏng ở đâu. **Bảng này là phần có giá trị nhất của chương đánh giá**, vì nó nói được điều mà con số tổng không nói được. Slice S8 báo cáo riêng ở dạng nhị phân (health check phát hiện đúng hay không).

## 8.4 Đánh giá đầu-cuối & hệ thống

**Alert metrics** (trên G4): Precision, Recall, F1, và **False Alarm Rate = FP / tổng số cửa sổ**. FAR quan trọng hơn Precision trong vận hành thực: một hệ báo động giả 5 lần/giờ sẽ bị người vận hành tắt, dù Precision trông có vẻ chấp nhận được.

**Faithfulness của câu trả lời RAG:** mỗi khẳng định trong câu trả lời có được hỗ trợ bởi `event_id` được trích dẫn không. Đánh giá bằng LLM-judge trên toàn bộ G3, **cộng với kiểm tra thủ công 30 mẫu** để hiệu chuẩn judge. Không được chỉ dùng LLM-judge.

**Latency:** đo tách từng chặng (`ingest → sed → caption → aggregate → alert`) và báo cáo P50/P95/P99. Mục tiêu **P95 end-to-end ≤ 6 s** cho đường nóng.

## 8.5 Mẫu bảng kết quả (điền dần)

```
Bảng 8.1 — SED trên G1 (test, human-verified)
  Model | Event-F1 | PSDS | Seg-F1 | Onset MAE (ms)

Bảng 8.2 — Captioning trên G2
  Model | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE | SPIDEr | FENSE

Bảng 8.3 — Grounding (ĐÓNG GÓP CHÍNH)
  Model | EHR↓ | EOR↓ | GS↑ | TOA↑ | CHR↓ | SPIDEr↑
  B0 (structured)   |  ~0  |      |      |      |  ~0  | (thấp)
  B1 (NLL only)     |      |      |      |      |      |
  B2 (+InfoNCE)     |      |      |      |      |      |
  B3 (+SED head)    |      |      |      |      |      |
  P  (đầy đủ)       |      |      |      |      |      |
  P−gd              |      |      |      |      |      |

Bảng 8.4 — Theo slice (xem §8.3)
Bảng 8.5 — RAG: Recall@5 | P@5 | MRR | nDCG@10 (theo từng loại truy vấn)
Bảng 8.6 — End-to-end: Alert P/R/F1 | FAR | Faithfulness | Answer Correctness
Bảng 8.7 — Hệ thống: latency P50/P95/P99 theo chặng | throughput | VRAM
```

---

# 9. MLOps

## 9.1 Data & model versioning

| Công cụ | Quản lý | Ghi chú |
|---|---|---|
| **DVC** | `data/` (raw, banks, synthetic, annotations, gold), feature `.npy` | Remote: ổ local hoặc Google Drive |
| **Git** | Code, config, docs, file `.dvc` | |
| **MLflow** | Params, metrics, artifacts, model registry | Mỗi run log đủ để tái lập |

**Bắt buộc log trong mỗi MLflow run:** git commit SHA, DVC data hash, seed, file config đầy đủ, **và toàn bộ metric của cả 4 tầng** (không chỉ loss). Một run không tái lập được là một run vô giá trị.

**Vòng đời model registry:** `Development → Staging → Production`. Điều kiện lên Production (ghi trong `ml/configs/promotion.yaml`):

- Event-F1 ≥ baseline hiện tại
- **EHR không tăng** so với model Production hiện tại
- Latency P95 không vượt ngân sách
- Không có slice nào sụt > 15% tương đối

Điều kiện thứ hai là điều đặc biệt: hệ thống này coi **hallucination là tiêu chí chặn phát hành**, ngang hàng với accuracy. Đó là hệ quả trực tiếp của luận điểm ở §1.1.

## 9.2 CI/CD

`.github/workflows/ci.yml`:

| Job | Nội dung |
|---|---|
| `lint` | ruff + black --check + mypy |
| `test` | pytest, ngưỡng coverage 80% |
| `data-validate` | `validate_taxonomy.py`, `validate_annotations.py`, `check_leakage.py` |
| `build` | Build image các service |
| `smoke` | `docker compose up` + gọi `/health` + một truy vấn RAG mẫu |

`data-validate` chạy trong CI là điểm khác biệt so với một dự án phần mềm thường: **dữ liệu cũng phải được kiểm thử**, không chỉ code.

## 9.3 Monitoring & drift

Ghi vào bảng `inference_metrics`, hiển thị trên dashboard:

| Nhóm | Chỉ số |
|---|---|
| Hiệu năng | Latency theo chặng, throughput, tỉ lệ lỗi, VRAM |
| Model | Phân phối confidence theo lớp, histogram lớp dự đoán, tỉ lệ kích hoạt grounded-decoding fallback |
| Chất lượng | FAR theo ngày, tỉ lệ feedback true/false alarm |
| Drift | So sánh phân phối confidence 7 ngày gần nhất với đường cơ sở lúc train (PSI hoặc KL) |

**Ví dụ kịch bản drift phải phát hiện được:** `glass_breaking` có confidence trung bình 0.89 lúc nghiệm thu; sau 3 tháng còn 0.62 → cảnh báo drift → điều tra (mùa mưa? micro bẩn? đổi vị trí lắp?) → gán nhãn bổ sung → retrain.

**Tỉ lệ kích hoạt fallback của grounded decoding** là một chỉ số giám sát đặc biệt hữu ích: nó tăng đột ngột nghĩa là SED head và caption decoder đang bất đồng ngày càng nhiều — dấu hiệu sớm của drift, xuất hiện *trước cả* khi accuracy tụt.

## 9.4 Vòng lặp phản hồi

```
Production event
   → Người vận hành xác nhận trên dashboard (true/false alarm, sửa nhãn, caption có bịa không)
   → Ghi vào event_feedback
   → Định kỳ: xuất các mẫu bị gán sai → gán strong label bổ sung
   → Thêm vào dataset (DVC bump version)
   → Retrain (thủ công ở phiên bản này) → đánh giá → so điều kiện promotion
   → Promote hoặc từ chối
```

Trường `caption_ok` trong `event_feedback` là dữ liệu quý nhất: nó cho **nhãn hallucination trong thực địa**, thứ mà không dataset công khai nào có.

## 9.5 Triển khai

`docker compose up` phải chạy được từ máy trắng và tự động: chờ healthcheck → migrate → seed `locations` + kịch bản demo → tải model từ registry → khởi động replay simulator → phục vụ. Mirror pattern của `example_project/docker-compose.yml` (healthcheck, `depends_on: condition: service_healthy`, `deploy.resources.limits`, named volumes).

---

# 10. Kế hoạch thực nghiệm

| ID | Thí nghiệm | Câu hỏi trả lời | Đầu ra |
|---|---|---|---|
| **E1** | SED: so sánh backbone (BEATs vs PANNs) | Backbone nào cho đặc trưng frame tốt hơn cho grounding? | Bảng 8.1 |
| **E2** | Ablation captioning: B0/B1/B2/B3/P/P−gd | Từng thành phần đóng góp bao nhiêu? | Bảng 8.3 |
| **E3** | Quét trọng số $\lambda$ | Cân bằng ba mục tiêu ở đâu là tốt nhất? | Đường cong $\lambda$ vs SPIDEr/EHR |
| **E4** | **Ablation Nhóm B** — huấn luyện có/không 5 lớp nhầm lẫn | 5 lớp nhầm lẫn có thực sự giảm FAR không? | Chứng minh **C3** |
| **E5** | Quét ngưỡng grounded decoding $\theta$ | Đánh đổi CIDEr ↔ EHR như thế nào? | **Hình trung tâm của khoá luận** |
| **E6** | Phân tích theo slice | Model hỏng ở điều kiện nào? | Bảng 8.4 |
| **E7** | RAG: vector-only vs hybrid | Lọc metadata đóng góp bao nhiêu vào Recall? | Bảng 8.5 |
| **E8** | Đo latency đầu-cuối | Có đạt ngân sách 6 s không? | Bảng 8.7 |
| **E9** | Chênh lệch synthetic → real | Domain gap lớn cỡ nào? | Phân tích trung thực |

E5 là hình quan trọng nhất của cả khoá luận. E4 và E9 là hai thí nghiệm mà hội đồng sẽ đánh giá cao vì chúng thể hiện sự trung thực về mặt phương pháp.

---

# 11. Rủi ro, giả định & hạn chế

## 11.1 Rủi ro

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| Gán nhãn strong tốn hơn dự kiến | **Cao** | Test chỉ ~1 giờ; train dùng Scaper; gán lại chỉ 20%; pilot 30 clip trước |
| Chỉ một người gán nhãn → không đo được đồng thuận liên-người | **Đã xảy ra** | Chuyển sang test–retest (ngưỡng cao hơn: 0.75); công bố guideline + gold để bên ngoài audit; **nêu rõ trong phần Hạn chế** |
| Domain gap synthetic → real | **Cao** | Test luôn là audio thật; báo cáo chênh lệch như một kết quả (E9) |
| **P không thắng B1** | **Trung bình–Cao** | Vẫn báo cáo trung thực + phân tích nguyên nhân; B0 đảm bảo hệ thống vẫn dùng được |
| MIVIA duyệt chậm | Trung bình | Nộp đơn ngày đầu W1; dự phòng FSD50K + AudioSet-strong |
| Lớp hiếm (`vehicle_crash`, `explosion`) thiếu dữ liệu | Trung bình | Chốt ở W2: gộp hoặc báo cáo riêng, loại khỏi macro-F1 (nêu rõ) |
| Bộ trích $P$ từ caption không chính xác → metric C2 không đáng tin | Trung bình | Kiểm định thủ công 100 caption, báo cáo độ chính xác của bộ trích |
| License AudioSet/FSD50K khi phát hành | Trung bình | Ghi license từng file; phát hành **script tái tạo**, không phát hành audio |
| 8 tuần không đủ | **Cao** | Cut-list định sẵn thứ tự hy sinh trong [PLAN.md](PLAN.md) |

## 11.2 Giả định

1. Micro đặt cố định, đã biết vị trí; không xử lý micro di động.
2. Một micro cho một khu vực; không fusion đa micro.
3. Ngôn ngữ nói trong `speech_normal` không cần nhận dạng nội dung.
4. Đồng hồ hệ thống được đồng bộ (NTP) — timestamp là cơ sở của toàn bộ RAG.
5. Có GPU cho huấn luyện; inference production có thể chạy GPU hoặc CPU (đo cả hai).

## 11.3 Hạn chế đã biết (viết vào báo cáo, không giấu)

| Hạn chế | Ảnh hưởng |
|---|---|
| Test set ~1 giờ là nhỏ | Khoảng tin cậy rộng; phải báo cáo CI, không chỉ điểm số |
| Phần lớn dữ liệu train là tổng hợp | Kết quả thực địa có thể thấp hơn |
| Risk scoring rule-based, hệ số do người đặt | Chưa tối ưu; nhưng minh bạch và giải thích được |
| Chưa có dữ liệu vận hành thật dài hạn | Không chứng minh được drift thực tế, chỉ mô phỏng được |
| `EVENT_LEXICON` do người xây, tiếng Anh | Có thể bỏ sót cách diễn đạt; đã kiểm định thủ công nhưng không hoàn hảo |
| LLM-judge có thiên lệch | Đã hiệu chuẩn bằng 30 mẫu người chấm, nhưng vẫn là hạn chế |

## 11.4 Hướng phát triển

Fusion với video · định vị nguồn âm bằng mảng micro · risk scoring học từ feedback · triển khai edge (quantization, ONNX) · mở rộng taxonomy · continual learning không quên lớp cũ · chống tấn công đối kháng.

---

# 12. Phụ lục

## 12.1 Từ điển thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| **AAC** | Automated Audio Captioning — sinh mô tả ngôn ngữ tự nhiên cho âm thanh |
| **SED** | Sound Event Detection — phát hiện sự kiện âm thanh có định vị thời gian |
| **Strong label** | Nhãn có `onset`/`offset` |
| **Weak label** | Nhãn chỉ ở mức clip |
| **Collar** | Dung sai thời gian khi so khớp event (200 ms) |
| **PSDS** | Polyphonic Sound Detection Score |
| **Slice** | Lát cắt điều kiện của test set (S1–S8) |
| **Grounding** | Ràng buộc đầu ra ngôn ngữ vào bằng chứng có thật |
| **EHR / EOR / GS / TOA / CHR** | Bộ metric hallucination đề xuất — [§8.2](#82-bộ-metric-hallucination-đề-xuất) |
| **Trunk** | Phần mạng chia sẻ giữa các head |
| **Hot path / History path** | Đường cảnh báo tức thời / đường truy vấn lịch sử |

## 12.2 Ánh xạ ontology

Bảng đầy đủ `class_id → AudioSet ontology ID → dataset nguồn → số giờ` được sinh tự động vào `docs/data_inventory.md` bởi `scripts/data_inventory.py`. Không duy trì thủ công ở file này.

## 12.3 Cấu trúc thư mục

```
audio-security-rag/
├── CLAUDE.md                    ★ trạng thái phiên làm việc
├── README.md
├── docker-compose.yml
├── .env.example
├── dvc.yaml
├── docs/
│   ├── SYSTEM.md                ★ file này
│   ├── PLAN.md                  ★ kế hoạch 8 tuần + trạng thái
│   ├── taxonomy.md
│   ├── annotation_guideline.md
│   ├── evaluation_protocol.md
│   ├── data_inventory.md        (sinh tự động)
│   └── decisions/ADR-XXXX.md
├── data/                        (DVC quản; git ignore)
│   ├── raw/ banks/{foreground,background,rir}/ synthetic/
│   ├── annotations/ gold/ features/
├── ml/
│   ├── datasets/ features/ models/ training/ evaluation/ captioning/ configs/
├── services/
│   ├── inference/ stream/ api/ frontend/
├── scripts/
├── tests/
├── notebooks/
└── .github/workflows/ci.yml
```

## 12.4 Ánh xạ sang chương báo cáo

| Chương báo cáo | Mục trong file này |
|---|---|
| Ch.1 Mở đầu | §1 |
| Ch.2 Cơ sở lý thuyết & công trình liên quan | §2 |
| Ch.3 Dữ liệu & phương pháp xây dựng bộ dữ liệu | §3 |
| Ch.4 Thiết kế hệ thống | §4, §6, §7 |
| Ch.5 Mô hình đề xuất | §5 |
| Ch.6 Thực nghiệm & đánh giá | §8, §10 |
| Ch.7 Triển khai & MLOps | §9 |
| Ch.8 Kết luận & hướng phát triển | §11 |

---

*Cập nhật lần cuối: 2026-09-14 · Phiên bản 0.1*
