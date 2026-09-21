# Trạng thái dự án đã đối soát

> **Snapshot hiện hành: 19/09/2026, sau Pha 1–5 của Training Ops, lượt quét ngưỡng,
> lượt phân tích lỗi và ablation hậu xử lý.**
> Trước đó: 18/09 sau B0–B9, train `panns_ft_v3` và dọn artifact.
> Các phép đo dữ liệu/training được thực hiện ngày 17/09; kiểm kê file và cache trong workspace
> được đối chiếu lại ngày 18/09. Phân biệt rõ **lô legacy** (bằng chứng trước-sửa) với **lô mới**
> đang dùng cho v3. Các mục lịch sử trong CLAUDE.md/ADR giữ nguyên ngữ cảnh tại thời điểm ghi.
>
> **Đối soát lại lúc 17/09 15:05–15:10.** Snapshot 12:33 trước đó có bốn chỗ đã lạc hậu hoặc
> sai; bảng này vẫn là bằng chứng lịch sử của lần đối soát đó:
>
> | Mục | Snapshot 12:33 | Đo lại 15:05 |
> |---|---|---|
> | panns_ft_v2 | "đang chạy, epoch 10/25" | **hoàn tất 25/25** |
> | AudioSet | 527 WAV, process còn sống | **937 WAV / 937 dòng**, process đã dừng |
> | Số test | "76 test đạt" (tập con) | **394 test đạt** (toàn bộ `tests/`) |
> | long_event | "recipe không thực thi được" | **điều khoản bất khả thi với bank** — xem §6 |
>
> Mọi tiến trình Python/yt-dlp/ffmpeg đã dừng ở cuối phiên. Docker từng đạt 6/6 healthy lúc
> 12:16, sau đó được `docker compose down` có chủ đích để nhường tài nguyên; named volume được giữ.

## 0. Tóm tắt hiện hành

- Lô mới `data/synthetic/`: **7.920 train + 1.440 dev**, WAV/JAMS/index đầy đủ và
  `verify_synthetic.py` PASS ở cả hai split (file báo cáo hiện tại không có dòng vi phạm).
  Đây là lô huấn luyện `panns_ft_v3`; không có real dev/gold độc lập.
- Cache waveform PANNs 32 kHz hiện có cả train (**5.068.800.128 byte**) lẫn dev
  (**921.600.128 byte**). Nó là cache tái tạo được, không phải feature BEATs.
- `panns_ft_v3` hoàn tất 25 epoch: clip mAP **0,8123**, segment-F1 **0,2748**,
  event-F1 **0,1077** — đo trên val tách từ train, ở ngưỡng cố định 0,5.
- 🔴 **19/09 — quét ngưỡng bác bỏ kết luận rút ra từ bảng đó.** Sau khi Pha 3 lưu dự đoán
  mức đoạn và chấm lại **cả ba run trên cùng tập dev 1.440 clip**: ngưỡng 0,5 sai nghiêm
  trọng ở cả ba, và thứ tự v2/v3 **đảo lại**. Bảng đầy đủ ở §3; phép đo ở
  [measurements/threshold_sweep_20260919.md](measurements/threshold_sweep_20260919.md).
- 🔴 **19/09 — phân tích lỗi (Pha 4) chỉ ra θ=0,5 hỏng theo kiểu nào, và một trần cứng
  của v1.** Ở θ=0,5, v3 bỏ sót **9 trên 4.873** sự kiện (0,2%) nhưng chèn thêm **61.020**
  — lỗi nằm ở ngưỡng/hậu xử lý, không ở năng lực phát hiện. Riêng v1: onset dự báo rơi
  **100%** trên lưới 319,7 ms (31 đoạn), cho **trần 37,4%** số cặp không thể lọt collar
  200 ms dù model đoán hoàn hảo (v2/v3 lưới 79,9 ms → trần 0%). **So event-F1 của v1 với
  v2/v3 là so một phần độ phân giải bộ giải mã.** Phép đo ở
  [measurements/error_analysis_20260919.md](measurements/error_analysis_20260919.md).
- 🔴 **19/09 — Pha 5 kết luận: KHÔNG có phép so sánh sạch nào giữa v1/v2/v3.**
  `compare_runs.py` khai `KHONG_RO` cho cả dữ liệu lẫn mã nguồn: vân tay của v1/v2 là
  `null` vĩnh viễn, còn `code.tree_sha256` của cả ba **trùng nhau** vì manifest lập hồi
  cứu trong cùng một phút — đó là mã lúc lập manifest, không phải lúc train. Thêm một bẫy
  đã bị chặn bằng test: `manifest.ket_qua_cuoi` chấm trên val split **riêng** của từng run
  (0,1282 / 0,1897 / 0,1077), xếp **v2 > v1 > v3**, ngược hẳn dev chung.
- 🔴 **19/09 — `long_event` hỏng vì phân mảnh, không phải bỏ sót.** Phân mảnh gấp **1,9
  lần** mốc clip sạch, Insertion gấp 1,8 lần (hệ quả cơ học của phân mảnh), còn Deletion
  thì thấp hơn cả `low_snr`. Ablation **bác bỏ một phần** giả thuyết "cửa sổ lọc 7 khung
  quá hẹp": cửa sổ theo lớp nâng mọi lát cắt gần như cùng một lượng nên khoảng cách tới
  mốc không đổi (0,180 → 0,178). Nguyên nhân riêng của lát cắt này **vẫn chưa tìm ra**.
  Phép đo ở [measurements/long_event_postproc_20260919.md](measurements/long_event_postproc_20260919.md).
- 🔴 **19/09 (tối) — trần cửa sổ lọc cũng bị bác bỏ phần lớn, rò rỉ cửa sổ đo được bằng 0,
  hiệu chuẩn xác suất lệch nặng ở vùng giữa.** (1) Nới `MAX_MEDIAN_FRAMES` 51→401 khung
  không đóng được khoảng cách `sach`−`long_event`: `v2` còn **rộng ra** (0,1322→0,1463),
  `v3` chỉ hẹp **6,4%**; cột 'gộp' đứng yên tuyệt đối qua cả 4 mức trần, cảnh báo "nới
  trần sẽ gộp nhầm" trong code vẫn chưa có số đỡ. (2) Cửa sổ suy từ TRAIN (không rò rỉ,
  `cua_so_loc_tu_train`) cho F1 khớp cửa sổ suy từ dev đến 6 chữ số — nhưng vì **9/15 lớp
  đã kẹp ở trần 51 tại cả hai nguồn**, không phải vì train/dev giống nhau; kết luận "rò rỉ
  = 0" chỉ đúng khi trần còn hiệu lực. (3) ECE đo lần đầu: vùng dự báo 0,4–0,6 mang **hơn
  nửa triệu khung** mà tỉ lệ dương thật chỉ 1,7–3% — giải thích hợp lý cho θ* luôn rơi
  0,85–0,95, **chưa kết luận nguyên nhân** (`pos_weight` cần train lại để xác nhận). Phép
  đo ở [measurements/max_median_ceiling_20260919.md](measurements/max_median_ceiling_20260919.md),
  [measurements/adaptive_window_leak_20260919.md](measurements/adaptive_window_leak_20260919.md),
  [measurements/calibration_20260919.md](measurements/calibration_20260919.md).
- 🔴 **21/09 — ứng viên cuối của `long_event` cũng bị bác bỏ, hết giả thuyết đang chờ.**
  `ml/evaluation/long_event_gap.py` so khe hở năng lượng trong nhãn giữa nhóm sự kiện bị
  phân mảnh (n=178) và nhóm không (n=137), cùng lát cắt `long_event`, chỉ xét sự kiện ≥1s.
  Khe hở trung bình **bằng nhau tuyệt đối** ở ngưỡng "im" 20% (0,012s) lẫn 40% (0,046s);
  p90 nhóm KHÔNG phân mảnh còn cao hơn — ngược hướng giả thuyết. `long_event` giờ đã loại
  **ba** ứng viên liên tiếp (cửa sổ hẹp, trần thấp, khe hở năng lượng); nguyên nhân riêng
  **vẫn chưa xác định**. Phép đo ở
  [measurements/long_event_gap_20260921.md](measurements/long_event_gap_20260921.md).
- Pha 1–5 của [TRAINING_OPS_PLAN](TRAINING_OPS_PLAN.md) đã có module và đã chạy:
  `ml/runs/{v1,v2,v3}/manifest.json` + `predictions/dev_all.npz` +
  `analysis{,_adaptive}.{json,md}`. `train_sed.py` nay gieo toàn bộ RNG và ghi manifest
  trước epoch đầu.
- Lô legacy đã được chuyển sang `data/synthetic_legacy/`: giữ **9.360 JAMS**, hai
  `slice_index.jsonl` và hai stdout log; WAV legacy, cache tạm, recovery trùng và stderr log đã xoá.
  Đây là bằng chứng trước-sửa, không phải dữ liệu để train lại.
- Checkpoint cục bộ (5 `best.pt`, khoảng 1,56 GiB) được giữ và đã Git-ignore cùng các weight
  `.pth/.ckpt/.onnx`. JAMS legacy được theo dõi có chủ đích; chúng giữ đường dẫn tuyệt đối lịch sử,
  phù hợp audit nhưng không portable để replay trực tiếp trên máy clone.

## 1. Lô Scaper legacy: sinh xong, chất lượng chưa đạt

| Kiểm tra | Train | Dev |
|---|---:|---:|
| Mục tiêu / WAV / JAMS / dòng index | 7.920 / 7.920 / 7.920 / 7.920 | 1.440 / 1.440 / 1.440 / 1.440 |
| Tổng thời lượng giải mã | 22 giờ | 4 giờ |
| Tập ID khớp chính xác 0..N−1, index không trùng ID | Đạt | Đạt |
| Giải mã toàn bộ; 16 kHz, mono, 160.000 mẫu; JAMS 10 s; giá trị hữu hạn | Đạt | Đạt |
| File pending còn lại | 0 | 0 |
| Clip không có foreground theo plan | 788 | 153 |
| Hợp đồng dữ liệu | **FAILED** | **FAILED** |

Toàn bộ **9.360 clip** đã được đọc, không có file lỗi giải mã hoặc sai kích thước trong
phép kiểm trên. Peak lớn nhất 1,0: phép kiểm này **không chứng minh không clipping**, không
đánh giá chất lượng cảm nhận hay xác nhận sự kiện nghe được đúng nhãn. Thư mục TSV vẫn rỗng.

### Lỗi chất lượng đã đo trên toàn bộ hai split

| Mã lỗi | Train | Dev |
|---|---:|---:|
| slice_overlap | 1.870 | 331 |
| slice_long_event | 791 | 144 |
| duration_truncated (cảnh báo toàn split) | 1 | 1 |
| **Tổng dòng vi phạm** | **2.662** | **476** |
| Số clip riêng biệt có vi phạm cấp clip | 2.443 | 435 |

Một clip có thể vi phạm nhiều điều khoản; **không gọi 2.662/476 là số clip hỏng**.
Recipe hiện vẫn dùng `source_time=0`, `event_duration=1.0` và đặt event ngẫu nhiên nên
nhãn kế hoạch overlap/long_event không đảm bảo được thực thi. Thời lượng event tối đa
đo từ JAMS khoảng 1,1 s, không đáp ứng long_event ≥8 s.

Báo cáo đầy đủ:
[train](../data/manifests/synthetic_contract_train_20260917.csv),
[dev](../data/manifests/synthetic_contract_dev_20260917.csv).
Hai lượt verifier trả exit code **1 vì phát hiện lỗi dữ liệu**, không phải job Scaper bị crash.
Giữ nguyên báo cáo thử cũ `synthetic_contract.csv`; không ghi đè bằng lượt kiểm này.

### Thao tác hoàn tất và tăng tốc

- Scaper cũ **còn chạy**, không phải đã chết: PID 24204 và CPU time tăng; lúc chuyển giao
  có 7.400 cặp file, trong đó clip cuối có thể chưa xong RIR.
- Benchmark một RIR 31.916 mẫu: direct convolution **30,468 s**, FFT **0,018 s**,
  sai khác tuyệt đối tối đa **8,33×10⁻¹⁷**. Đây là tốc độ của phép RIR riêng, không phải
  hệ số tăng tốc toàn pipeline; FFT không cam kết bit-identical với direct convolution.
- Dừng đúng cây shell Scaper cũ, không dừng AudioSet hay v2. Sao lưu cặp WAV/JAMS
  `train_007398` và `train_007399` vào
  `data/synthetic/recovery_20260917_1215/`, rồi sinh lại hai clip này. Backup sau đó đã được
  đối chiếu là trùng với legacy và xoá trong lượt dọn 18/09; không mất bằng chứng độc nhất.
- Giữ nguyên seed/config/bank; replay RNG khi skip clip, bỏ khởi tạo Scaper cho clip đã có.
  Clip mới ghi qua `.pending` và chỉ công bố tên cuối sau RIR; sửa đường dẫn trong JAMS về tên cuối.
- Chạy tiếp train và dev song song từ **12:15:13**, mỗi process giới hạn BLAS/OMP/MKL 1 luồng.
  Train kết thúc **12:17:19**; dev kết thúc **12:20:22**. Khoảng **5 phút 9 giây** để đóng lô còn lại.
- Log tại thời điểm đó là `scaper_{train,dev}_fast_20260917.{stdout,stderr}.log`.
  Lượt dọn 18/09 đã bỏ log sinh tạm và stderr; hai stdout legacy được giữ vì ghi bằng chứng
  cho tỷ lệ slice legacy hỏng.

Lô này được giữ như **legacy baseline**, không âm thầm đổi recipe giữa chừng hoặc tuyên bố
đạt `data-v1.0`. Synthetic dev dùng **cùng foreground bank** với train; khác seed không
tạo độc lập nguồn. Index tái dựng từ seed chỉ ghi recipe dự kiến, không tự chứng minh
mọi clip lịch sử đã qua RIR đúng ở các lần chạy bị ngắt trước đây.

## 2. Dữ liệu còn lại

- Foreground **4.267 WAV / 15 lớp sự kiện**, background **2.461 WAV**, RIR **505 WAV**;
  khớp báo cáo [data_inventory.md](data_inventory.md). Taxonomy có 16 lớp vì ambient_noise
  không cần foreground/head SED riêng.
- `shout_yell` **56/80**, vẫn tính macro-F1 theo ADR-0005. `vehicle_crash` **33/30**,
  nguồn vehicle_crash_cc; MIVIA Road không còn là nút chặn lớp này.
- `screen_audit.csv`: 244 sample-ok, 1 sample-wrong_class, 1.831 queue-ok và 1 queue-wrong_class.
  1.831 queue-ok là bulk-accept theo quyết định 16/09, không phải nghe từng clip.
- AudioSet lúc **15:05**: **937 WAV / 937 dòng segments.jsonl** — hai số **khớp tuyệt đối**,
  xác nhận bản vá ghi tăng dần (`append_one_segment` + `flush`) đã chấm dứt tình trạng
  metadata mồ côi (từng là 375 WAV / 3 dòng). Process tải **đã dừng** cùng phiên trước,
  chưa hết hàng đợi 120 clip/lớp. Đây là số raw, chưa khẳng định mỗi clip đủ QA/gold.
- `raw_manifest.csv` vẫn **5.430 dòng**; batch AudioSet mới chưa nhập lại manifest/splits.
  Real dev và gold_test trong bảng split vẫn 0, không mâu thuẫn với raw đã tải.
- `data/gold/` chỉ có decision_log.md, chưa có G1–G4/pilot annotation artifacts.
  Ruling chuông quầy phục vụ vẫn chờ xác nhận của người gán nhãn.

## 3. Training

### Cache và ba run hiện có

Cache hiện hành là **7.920 train + 1.440 dev** waveform 32 kHz: `train_wave32k.npy`
**5.068.800.128 byte**, `dev_wave32k.npy` **921.600.128 byte**, cùng hai file metadata.
Cache này được tạo sau khi lô mới PASS hợp đồng và đã dùng cho v3. Nó tái tạo được từ WAV;
không phải feature BEATs và không phải split real dev/gold.

| Chỉ số (epoch cuối, có full-eval) | v1 · legacy/pooling 5 | v2 · legacy/pooling 3 + mixup + adaptive | v3 · lô mới, tham số v2 |
|---|---:|---:|---:|
| clip mAP | 0,8372 | 0,8176 | 0,8123 |
| segment-F1 (1 s) | 0,4120 | 0,2980 | 0,2748 |
| event-F1 | 0,1282 | **0,1897** | 0,1077 |

Best clip mAP: v1 **0,8374** ở epoch 23; v2 **0,8183** ở epoch 23; v3 **0,8123** ở
epoch 25. Không gán F1 epoch cuối cho checkpoint best của v1/v2. Các F1 bằng 0 ở epoch
không full-eval là placeholder **chưa đo**, không phải F1 = 0.

**Cách đọc đúng giới hạn.** V1/v2 cùng dùng snapshot legacy 6.568 clip (5.911/657 split
tạm) và đồng thời đổi pooling, mixup, hậu xử lý. V3 giữ tham số v2 nhưng đổi sang lô mới
có phân bố event, phủ sóng và overlap khác. Vì vậy không có phép so sánh nào trong bảng
tự chứng minh được một thay đổi cụ thể gây ra chênh lệch. V3 thấp hơn v2 ở cả ba chỉ số
không đồng nghĩa sửa dữ liệu thất bại: nó học/chấm trên bài khác và threshold 0,5 chưa được
tối ưu lại.

Nguồn số của bảng trên: `ml/runs/panns_ft_{v1,v2,v3}/history.json`.

### Chấm lại trên giao thức chung — 19/09

Bảng trên có ba khuyết tật đã được gỡ: mỗi run chấm trên một tập khác nhau, ngưỡng cố định
0,5, và không có dự đoán để kiểm lại. Sau Pha 3, cả ba được chấm trên **cùng
`data/synthetic/dev` — 1.440 clip / 4.873 sự kiện tham chiếu**, không run nào từng huấn
luyện trên đó, cùng bộ lọc 7 khung, cùng mã chấm điểm:

| chỉ số (dev chung) | v1 · pool 5 · legacy | v2 · pool 3 · legacy | v3 · pool 3 · lô B0–B9 |
|---|---:|---:|---:|
| mAP mức clip | 0,7464 | 0,7555 | **0,8132** |
| event-F1 @ θ=0,50 | 0,1147 | 0,1123 | 0,0820 |
| θ* tối ưu event-F1 | 0,96 | 0,89 | 0,91 |
| **event-F1 @ θ\*** | 0,2981 | 0,3531 | **0,3986** |
| **segment-F1 @ θ\*** | 0,6208 | 0,6564 | **0,7159** |

**Ba điều bảng này chứng minh.** (1) θ=0,5 sai ở cả ba run — v3 tăng **4,86 lần**
(0,0820 → 0,3986); ở 0,5 nó dự báo 65.884 sự kiện cho 4.873 sự kiện thật. (2) Xếp hạng ở
θ=0,5 **không có giá trị**: ở đó v3 trông tệ nhất, ở ngưỡng đúng nó tốt nhất trên cả ba
chỉ số. (3) Mức tăng "+48 % tương đối" của v2 so với v1 không tái hiện trên tập chung ở
θ=0,5 (0,1123 so với 0,1147); khoảng cách thật chỉ hiện ra khi mỗi run dùng ngưỡng riêng.

**Điều bảng này KHÔNG chứng minh.** `data/synthetic/dev` sinh bằng **cùng recipe B0–B9**
với train của v3, chỉ khác seed; v1/v2 học trên lô legacy có phân bố khác hẳn. Dev vì thế
thiên vị v3 **theo thiết kế**. Bảng chứng minh v3 khớp phân bố đích hiện hành tốt hơn, không
chứng minh kiến trúc hay dữ liệu mới tốt hơn nói chung. Chỉ real dev / gold_test tách được.

**Một giả thuyết chưa đo.** Cả ba đạt đỉnh ở θ ≈ 0,90–0,96, tức xác suất bị thổi lên có hệ
thống. `pos_weight` trần 30 trong `BCEWithLogitsLoss` là cơ chế khả dĩ — **giả thuyết**,
cần reliability diagram + ablation trước khi viết vào báo cáo.

`eval_sed.py` **đã** đọc `time_pool_blocks` theo thứ tự checkpoint → `history.json` →
mặc định kèm cảnh báo (`doc_time_pool_blocks`, 3 test). Riêng `panns_ft_v1` không ghi ở
đâu cả nên vẫn rơi về mặc định 5; đo gián tiếp trên dev ủng hộ (pool 5 → mAP 0,7464, pool 3
→ 0,7244) nhưng **không chứng minh**. Đây là khoảng trống R1 không xoá hồi cứu được.

## 4. Hệ thống và MLOps

Lúc **17/09 12:16**, Docker Compose có **6/6 service healthy**: API, inference, frontend,
PostgreSQL/pgvector, Redis, MLflow. Đây là bằng chứng lịch sử của walking skeleton, không
phải trạng thái tiến trình hiện tại: sau đó đã chạy `docker compose down` có chủ đích để nhường
CPU/RAM cho sinh dữ liệu/train. Named volume được giữ; chỉ `docker compose down -v` mới xoá dữ liệu.
Healthy không thay thế nghiệm thu E2E trên máy trắng.

Thực tế: upload offline → PANNs pretrained CPU → template EN/VI → risk rules → DB/vector →
RAG template/citation → dashboard. Embedding là BGE-M3 1024 chiều trên rich document VI/EN
kèm metadata; ngưỡng similarity 0,52 mới hiệu chỉnh mẫu nhỏ, không bảo đảm không truy hồi sai.
Inference **chưa dùng v1/v2**. WebSocket hiện trong process API, chưa phải Redis pub/sub đa worker.

BEATs–Conformer–BART, grounded decoding, temporal aggregator, streaming Redis Streams,
CI workflows, DVC pipeline/remote và MLflow tracking của training chưa triển khai đầy đủ.
MLflow server đang chạy không đồng nghĩa đã có run được log. Chưa có job xoá audio hết hạn retention.

Git: `master`; base trước lần đồng bộ đầy đủ là `70ab498`. Weight, cache, raw và `.env` đã được
lọc khỏi snapshot; code, tài liệu, manifest, metric và JAMS legacy được chuẩn bị để publish.
DVC pipeline/remote vẫn chưa có, nên GitHub không thay thế quản lý dữ liệu lớn.

## 5. Kiểm chứng và phạm vi cập nhật

- **642 test đạt** (21/09), chạy đầy đủ `pytest tests/ -q` sau khi thêm 30 test cho
  Pha 1, Pha 3, phép gom sự kiện và lỗi pickle memmap, 21 test nữa cho Pha 4 (phép ghép
  cặp, lát cắt, ghi bền, đối chiếu `sed_eval`), 15 test cho Pha 5 và phân loại lỗi theo
  lát cắt, 15 test cho ablation trần cửa sổ + cửa sổ suy từ train + hiệu chuẩn xác suất,
  rồi 11 test nữa cho khe hở năng lượng của `long_event`. Mốc 550 là của B9 (17/09); mốc
  394 là của snapshot 15:05.
- Lượt 19–21/09 **có** chạy lại suite trước và sau thay đổi mã: 573 đạt trước khi thêm
  test mới, 580 sau Pha 1 + Pha 3, 601 sau Pha 4, 616 sau Pha 5, 631 sau ablation
  trần/hiệu chuẩn, **642** sau đo khe hở năng lượng. `train_sed --zero-shot --workers 2`
  chạy thật, xác nhận crash pickle đã hết. Không tuyên bố đã chạy lại test API/frontend.
- **`slice_index.jsonl` vẫn khớp code hiện tại.** `scaper_generate.py` được sửa lúc
  12:35:54, sau khi clip cuối sinh lúc 12:20:22 — tức dữ liệu sinh bằng code cũ. Đã chạy
  lại `plan_clips` với code mới và so từng clip: **0/7.920 khác biệt**. Tỉ lệ 5 lát cắt
  trên đĩa khớp config trong ±3 % ở cả hai split. Phân tích theo lát cắt vẫn nhất quán.
- Ontology validate: 0 lỗi/0 cảnh báo; vẫn có 2 thông tin chưa xác minh vocab nguồn
  UrbanSound/DESED. Leakage check đạt **trên manifest hiện có chỉ một split**,
  chưa chứng minh train/dev/gold mới độc lập sau khi nhập AudioSet.
- Đã giải mã/đối chiếu toàn bộ 9.360 WAV/JAMS/index; chạy verifier đầy đủ cả hai split.
- Clip hoàn tất cũ train_007398 khi tái sinh có waveform khớp từng mẫu với bản backup.
  Đây là kiểm tra một ca resume thật, không chứng minh bit-identical cho mọi clip dùng FFT.
- Kiểm tra toàn bộ 13 file Markdown dự án: không có liên kết file nội bộ bị hỏng.
- Ruff trên 3 file Python vừa sửa và git diff --check trong phạm vi sửa đều đạt;
  Git chỉ nhắc chuyển LF/CRLF theo cấu hình Windows, không có lỗi whitespace.
- Đã sinh lại inventory từ manifest/index và rà soát toàn bộ Markdown do dự án quản lý.
  Measurement và ADR được giữ nguyên như bản ghi lịch sử; trạng thái hiện hành được đồng bộ ở
  STATUS, CLAUDE, PLAN, DATA_PLAN, TRAINING_OPS_PLAN, SYSTEM, README và RELATED_WORK.
- Không tuyên bố test API/frontend/ML đã được chạy lại trong lượt cập nhật tài liệu này.

Ưu tiên tiếp theo: xem §6 — kế hoạch sửa dữ liệu đã được duyệt lúc 15:15.
Không cần chạy lại lô Scaper legacy chỉ vì còn lỗi hợp đồng đã biết.

---

## 6. Điều khoản `long_event` bất khả thi — đo lúc 15:00

Trước 15:00, cả tài liệu và tôi đều cho rằng `long_event` sai **chỉ vì**
`event_duration=("const", 1.0)`. Phép đo dưới đây bác bỏ điều đó: bỏ hằng số 1.0 đi thì
`long_event` **vẫn sai**, chỉ là sai vì lý do khác.

Số clip trong bank foreground còn đủ **≥ 8 giây** sau khi cắt im lặng đầu/cuối (ngưỡng
−30 dB so với đỉnh), đo trên **toàn bộ 4.267 clip**:

| lớp | ≥8 s (gốc) | ≥8 s (sau cắt) | p90 sau cắt |
|---|---:|---:|---:|
| applause_cheering | 32 | 32 | 11,72 s |
| speech_normal | 19 | 19 | 3,03 s |
| shout_yell | 11 | 10 | 15,41 s |
| vehicle_crash | 10 | 9 | 17,57 s |
| fireworks | 7 | 7 | 4,06 s |
| alarm_bell | 6 | 5 | 4,19 s |
| object_drop_dishes | 9 | 5 | 1,74 s |
| **siren** | 2 | **2** / 311 | **4,01 s** |
| running_footsteps | 1 | 1 | 4,68 s |
| door_slam · explosion · glass_breaking · gunshot · laughter · scream | 0 | **0** | 1,10–2,35 s |
| **TỔNG** | **97** | **90** / 4.267 | |

**Chín trong mười lăm lớp có 0 clip đủ dài.** Đáng chú ý nhất là `siren` — lớp "sự kiện
kéo dài" mẫu mực — chỉ có **2/311** clip ≥8 s, p90 chỉ 4,01 s, vì nguồn UrbanSound8K vốn
cắt sẵn ở 4 giây. `min_event_duration_sec: 8.0` là một điều khoản mà **dữ liệu không bao
giờ thoả được**, độc lập với mọi lỗi trong recipe.

Script đo: `scratchpad/kha_thi_long_event.py` (phiên 17/09, chưa đưa vào repo).

## 7. Thiết kế lại dữ liệu tổng hợp — B0–B9 hoàn tất 17/09

> Kế hoạch bốn bản sửa ①–④ duyệt lúc 15:15 đã được **thay** bằng kế hoạch B0–B9 sau khi
> đo lại. Phép đo bác bỏ chính thứ tự ưu tiên cũ: bản sửa ① (cắt im lặng bank) chỉ chạm
> **51/4.267 clip (1,2%)** — `lead` p50 = 0,000 s ở cả 15/15 lớp — trong khi nguyên nhân
> gốc thì không nằm rõ ràng trong bản nào.

### Nguyên nhân gốc: lệch phân bố thời lượng train/test

Toàn bộ **16.545 sự kiện** của lô 17/09 nằm trong **[0,19 s – 1,10 s]**, không một ngoại
lệ. Trần 1,10 s chính là `event_duration=1.0` nhân hệ số kéo 1,1.

| | TRAIN lô cũ | AudioSet `train_strong`, mix đều |
|---|---:|---:|
| p95 | 1,09 s | **10,00 s** |
| % sự kiện ≥ 2 s | **0,0 %** | 25,1 % |
| % ≥ 4 s | **0,0 %** | 15,5 % |
| % ≥ 8 s | **0,0 %** | 9,5 % |
| độ phủ sóng | 18,0 % | 53,0 % |

⚠️ **So sánh phải dùng cột "mix đều".** AudioSet lệch nặng về `speech_normal` (42,5 % số
sự kiện) và `running_footsteps` (13,8 %) — đều là lớp ngắn; dữ liệu của ta cân bằng đều
15 lớp. Ở mốc 4 s, đích đọc ra 8,9 % theo mix tự nhiên nhưng **15,5 %** theo mix đều.
Nhìn nhầm cột là kết luận ngược. `measure_distributions.py` in cả hai.

### Các bước đã xong

| # | Việc | Kết quả đo được |
|---|---|---|
| **B0** | `measure_distributions.py` — đọc ngược file nhãn đã sinh | mốc đối chiếu trong `docs/measurements/` |
| **B1** | `probe_bank.py` → `bank_trim.csv` | 4.267 clip, **0 lỗi**; bank giữ nguyên bit-for-bit |
| **B2** | tự chọn file nguồn (`source_file=("const", …)`) | gỡ cùng lúc cả ba lỗi ①②④ mà **không ghi đè bank** |
| **B2.5** | `build_background_10s.py` — nền liên tục | 1.411 file; lặp vòng 4 s giảm mạnh ở 3/4 khu vực |
| **B3** | rút thời lượng theo phân bố `train_strong` | KS **0,051**; ≥2 s: 24,3 % so với đích 24,6 % |
| **B3.5** | chỉ lấy phía `train_strong` — [ADR-0006](decisions/ADR-0006-phan-bo-dich-chi-tu-train-strong.md) | bỏ 112 clip eval, KS(trước/sau) = **0,0083** |
| **B4** | ép chồng lấn thật | **2378/2378 = 100 %** đạt ngưỡng; **0** sự kiện tràn mép (trước: 1.225) |
| **B5** | `long_event` 8 s → 4 s — [ADR-0007](decisions/ADR-0007-ha-nguong-long-event.md) | **772/772 = 100 %** đạt (trước: **0/791**) |
| **B6** | mật độ sự kiện + đặt vào khoảng trống — [ADR-0008](decisions/ADR-0008-mat-do-su-kien-va-cach-dat.md) | độ phủ sóng **18,0 % → 50,9 %** (đích 53,0 %) |
| **B7** | `--simulate-labels` — nghiệm thu nhãn trước khi sinh audio | **2,8 giây** cho 7.920 clip |
| **B8** | seed theo clip (`seed_clip`) + `--workers` | jams **giống hệt từng bit** giữa tuần tự và song song, đo trên 200 clip |

531 test đạt.

### B8 — tốc độ sinh: ước tính 18,6 giờ đã lỗi thời

Đo lại trên 200 clip sinh thật:

| | clip/phút | 9.360 clip |
|---|---:|---:|
| lô cũ (đo 17/09) | 8,4 | **18,6 giờ** |
| tuần tự, sau B0–B7 | 169 | **56 phút** |
| **6 worker** | **491** | **21 phút** |
| 12 worker | 442 | 21 phút |

Phần lớn mức tăng **không** đến từ song song hoá mà từ B2: `source_file=("choose", [])`
bắt Scaper quét thư mục bank cho TỪNG sự kiện; truyền đường dẫn `const` bỏ hẳn việc đó.
B8 cho thêm ~2,9×. Từ 12 worker trở lên không còn lợi.

Kèm theo, seed theo clip làm **resume trở nên chính xác**: clip đã xong được bỏ qua hẳn,
không còn phải gọi lại `populate()` chỉ để đẩy dòng RNG chung tới đúng chỗ.

### B7 — vì sao đáng làm

`quyet_dinh_clip` là nơi DUY NHẤT quyết định một clip chứa gì. Bộ mô phỏng đọc thẳng kết
quả của nó, còn khâu sinh audio chỉ dịch kết quả đó sang lời gọi Scaper — hai đường không
thể lệch nhau vì chúng là một. Đóng khoảng cách "log báo kế hoạch, không ai đọc kết quả"
bằng **cấu trúc**, không bằng kỷ luật.

Ngay lần chạy đầu nó bắt được một vi phạm còn sót: **2/791 clip `long_event` trượt ngưỡng**
vì `eff × time_stretch` tụt xuống dưới 4 s khi hệ số kéo < 1. Hai clip trên bảy trăm chín
mốt là thứ không ai phát hiện bằng cách nghe. Đã sửa bằng `nguong_co_bien()`; biên này
không làm mất lớp nào (vẫn 8/15).

Lệnh, trả mã lỗi khi hợp đồng không đạt nên cắm được vào tiền kiểm:

```
python scripts/scaper_generate.py --split train --simulate-labels
```

| | train | dev |
|---|---:|---:|
| sự kiện/clip | 3,35 | 3,38 |
| `overlap ≥ 0.30` | **100,0 %** | **100,0 %** |
| `long_event ≥ 4s` | **100,0 %** | **100,0 %** |
| sự kiện tràn khỏi clip | 0 | 0 |
| độ phủ sóng | 50,8 % | 50,8 % |

### Mô phỏng 7.920 clip sau B0–B6

| chỉ tiêu | lô cũ | sau B6 | đích (mix đều) |
|---|---:|---:|---:|
| sự kiện/clip | 2,09 | 3,40 | 4,22 |
| độ phủ sóng | 18,0 % | **50,9 %** | 53,0 % |
| % sự kiện ≥ 2 s | 0,0 % | 26,2 % | 25,1 % |
| % ≥ 4 s | 0,0 % | **15,1 %** | 15,5 % |
| % ≥ 8 s | 0,0 % | 6,6 % | 9,5 % |
| `overlap` đạt ngưỡng | không ép | **100 %** | 100 % |
| `long_event` đạt ngưỡng | **0 %** | **100 %** | 100 % |
| sự kiện tràn mốc 10 s | 1.225 | **0** | 0 |

### Hai điều khoản từng hỏng 100 % mà không có triệu chứng

| Hợp đồng | Log sinh khai | Thực tế trong nhãn |
|---|---:|---:|
| `long_event ≥ 8 s` · 10 % | 791 clip | **0 sự kiện** |
| `overlap` · `min_overlap_ratio 0.30` | 2.387 clip | 1.752 clip có chồng lấn, ngưỡng **chưa từng được ép** |

Log báo **kế hoạch**, không báo **kết quả**. Đó là lý do B0 tồn tại.

### Giới hạn còn lại — phải ghi vào khoá luận

- Ba lớp không khớp nổi phân bố đích vì bank: `siren` 35,9 % · `scream` 76,5 % · `explosion` 78,4 %
- `long_event` chỉ khả thi ở **8/15 lớp** ở ngưỡng 4 s (đủ cả bank ≥8 clip lẫn có sự kiện thật trong miền đích)
- Nền `factory` còn 68/88 đơn vị phải lặp — giới hạn của nguồn
- **Không khớp được đồng thời** độ phủ sóng và mật độ chồng lấn: đã ưu tiên phủ sóng, nên chồng lấn dày hơn thực tế (nén 0,783 so với 0,829; clip sạch 41 % so với 62,7 %) — [ADR-0008](decisions/ADR-0008-mat-do-su-kien-va-cach-dat.md)
- Phân bố ta khớp là **nhãn máy** của AudioSet; `gold_test` sẽ là **nhãn người**, độ lệch giữa hai cái **không** được đo bởi thiết kế hiện tại

### B9 — dữ liệu xong, v3 đã train

| bước | trạng thái |
|---|---|
| lô cũ → `data/synthetic_legacy/` | ✅ giữ 9.360 JAMS + `slice_index` + 2 stdout log; **đã xoá audio** (4,6 GB), recovery trùng và stderr log |
| mốc đối chiếu tái dựng từ legacy | ✅ `docs/measurements/dist_20260917_truoc_sua.md` |
| sinh lô mới | ✅ 7.920 train + 1.440 dev, 0 lỗi |
| `verify_synthetic` | ✅ **ĐẠT cả hai split**, 0 vi phạm |
| precompute train + dev | ✅ 5,07 GB + 0,92 GB |
| train `panns_ft_v3` | ✅ 25 epoch, đúng tham số v2, chỉ đổi dữ liệu: mAP 0,8123 · segment-F1 0,2748 · event-F1 0,1077 |

Kế hoạch sinh **theo mốc 500 → 1.000 → 1.500 không còn cần thiết**: sau B2 và B8, cả lô
mất ~47 phút thay vì 18,6 giờ.

### Phân bố cuối — điều toàn bộ B0–B9 nhắm tới

| chỉ tiêu | lô cũ | **lô mới** | đích (mix đều) | lệch |
|---|---:|---:|---:|---:|
| p05 (s) | 0,48 | **0,14** | 0,14 | **−0,00** |
| p50 (s) | 0,97 | **0,82** | 0,80 | **+0,01** |
| p95 (s) | 1,09 | **8,93** | 10,00 | −1,07 |
| max (s) | 1,10 | **10,00** | 10,00 | **+0,00** |
| % ≥ 2 s | 0,0 % | **26,1 %** | 25,1 % | +1,0 |
| % ≥ 4 s | 0,0 % | **15,1 %** | 15,5 % | **−0,4** |
| % ≥ 8 s | 0,0 % | 5,9 % | 9,5 % | −3,6 |
| độ phủ sóng | 18,0 % | **50,0 %** | 53,0 % | −3,0 |

Mốc ≥ 8 s vẫn hụt — giới hạn bank, xem [ADR-0007](decisions/ADR-0007-ha-nguong-long-event.md).

### Ba lỗi B9 phơi ra, và ai bắt được

| Lỗi | Ai bắt | Sau bao lâu |
|---|---|---|
| sự kiện 1 ms làm Scaper vỡ (nhãn thoái hoá của AudioSet) | lượt sinh crash | **45 phút** |
| 91 clip `causal_chain` mắt xích sai thứ tự | `verify_synthetic` | sau 47 phút sinh |
| 2.663 clip `reverb` mất `rir_used` khi resume | `verify_synthetic` | sau 10 phút sinh |

Hai lỗi đầu đã được kéo về bộ mô phỏng B7 — nay bắt trong **3 giây**. Nhưng B7 **không
thay thế** `verify_synthetic`: nó chỉ kiểm những gì ta biết để kiểm, còn cổng kiểm đọc
lại sản phẩm thật và bắt được cả thứ chưa ai nghĩ tới, như `rir_used`.

550 test đạt.

### Còn lại

| # | Việc |
|---|---|
| Ops Pha 1 + 3 | ✅ **xong 19/09** — manifest/vân tay/seed đầy đủ, prediction mức đoạn + quét ngưỡng cho cả ba run |
| A/B | ✅ **đã chạy cùng giao thức** (§3). Kết luận có điều kiện: v3 tốt nhất trên dev, nhưng dev cùng recipe với train của v3 — confound còn nguyên |
| Ops Pha 4 | ✅ **xong 19/09** — 6 loại lỗi (cả mức toàn tập lẫn **theo từng lát cắt**), ma trận nhầm 16×16, θ theo từng lớp; CPU ~100–170 s/run. Số đo: [measurements/error_analysis_20260919.md](measurements/error_analysis_20260919.md) |
| Ops Pha 5 | ✅ **xong 19/09** — `ml/tracking/compare_runs.py`; chạy trên ba run thì dữ liệu và mã nguồn đều `KHONG_RO`, tức chưa so sạch được run nào với run nào |
| `long_event` | **chưa xong** — biết hỏng vì phân mảnh; đã loại BA ứng viên liên tiếp ("cửa sổ lọc quá hẹp", "trần 51 khung", "khe hở năng lượng thật" — 21/09: hai nhóm vỡ/nguyên khe hở bằng nhau tuyệt đối); nguyên nhân riêng **vẫn chưa xác định**, hết ứng viên đang chờ sẵn |
| Cửa sổ suy từ train | ✅ **19/09** — `cua_so_loc_tu_train()`; độ lớn rò rỉ đo được = 0 nhưng vì 9/15 lớp đã kẹp trần 51 ở cả hai nguồn, không phải vì train/dev giống nhau |
| `confusable_with` | ✅ **19/09** — 8 cặp đo được đã khai (đối xứng) + bảng tra §3 `taxonomy.md`; tỉ lệ lượt nhầm đã khai của v3 lên 54,1% |
| Cổng hợp đồng | `train_sed.py` mới **cảnh báo** khi hợp đồng khác PASSED, chưa **chặn**. Một lô FAILED vẫn train được |
| Hiệu chuẩn | ✅ **19/09** — ECE + reliability diagram đo xong (`ml/evaluation/calibration.py`); vùng dự báo 0,4–0,6 chiếm hơn nửa triệu khung mà tỉ lệ dương thật chỉ 1,7–3%, giải thích triệu chứng θ*≈0,9. Chưa ablation `pos_weight` — cần train lại để xác nhận nguyên nhân |
| gold_test | vẫn RỖNG; DATA_PLAN D8–D10 là nút thắt, cần người gán mù |
| Publish prep | ✅ Weight/cache/secret đã ignore; JAMS legacy theo dõi có chủ đích; snapshot được audit trước khi push GitHub |
