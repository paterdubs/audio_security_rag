# Kế hoạch hạ tầng theo dõi & phân tích lỗi huấn luyện

> Lập ngày **17/09/2026**. Duyệt cùng ngày.
>
> **Mục tiêu:** khi một lần train ra kết quả xấu, phải trả lời được *sai ở đâu* và *cần
> sửa chỗ nào* — bằng số đo đã lưu sẵn, không phải bằng phỏng đoán hay train lại.
>
> Tài liệu liên quan: [PLAN.md](PLAN.md) (lịch W1–W8) · [DATA_PLAN.md](DATA_PLAN.md)
> (quy trình dữ liệu) · [SYSTEM.md](SYSTEM.md) (kiến trúc) ·
> [RELATED_WORK_2026.md](RELATED_WORK_2026.md) (đối chiếu văn liệu)

> **Đối soát 19/09/2026: Pha 1, Pha 2 và Pha 3 đã có module và đã chạy thật.**
> Pha 4 và Pha 5 vẫn là kế hoạch. Bằng chứng ở [STATUS.md](STATUS.md) và
> [measurements/threshold_sweep_20260919.md](measurements/threshold_sweep_20260919.md).
>
> **R5 đã gỡ, và ngay lượt đầu nó bác bỏ một kết luận đang nằm trong tài liệu.** Sau khi
> lưu dự đoán mức đoạn cho cả ba run rồi quét ngưỡng trên cùng một tập dev 1.440 clip:
> ngưỡng 0,5 sai nghiêm trọng ở cả ba (v3: event-F1 **0,0820 → 0,3986** ở θ=0,91, tăng
> 4,86 lần), và thứ tự v2/v3 **đảo lại** so với bảng chấm ở 0,5. Đây đúng là thứ mà
> "không lưu dự đoán" đã che suốt hai ngày.
>
> **R1 gỡ được cho v3, KHÔNG gỡ được hồi cứu cho v1.** `panns_ft_v1` không ghi
> `time_pool_blocks` ở checkpoint lẫn history; đo gián tiếp ủng hộ pool 5 (mAP 0,7464 so
> với 0,7244) nhưng không chứng minh. Từ 19/09 `train_sed.py` gieo toàn bộ RNG và ghi
> `manifest.json` ngay trước epoch đầu, nên khoảng trống này không lặp lại ở run sau.

---

## 0. Lý do tài liệu này tồn tại

Ngày 17/09/2026, ba lỗi **im lặng** được tìm ra trong cùng một buổi sáng. Không lỗi nào
làm chương trình dừng; cả ba chỉ làm số liệu sai:

| Lỗi | Triệu chứng nhìn thấy | Triệu chứng thật |
|---|---|---|
| `event_duration=("const", 1.0)` | không có | Mọi sự kiện bị cắt còn 1 s bất kể nguồn dài bao nhiêu |
| `source_time=("const", 0)` | không có | `vehicle_crash`: giây đầu yếu hơn đỉnh **−25.7 dB**, tức nhãn dán lên đoạn im lặng |
| `segments.jsonl` ghi ở cuối `run()` | không có | 375 file .wav trên đĩa / 3 dòng metadata |

Và một lỗi thứ tư chưa ai phát hiện cho tới khi lập kế hoạch này:

> `forced_slices.long_event = {ratio: 0.10, min_event_duration_sec: 8.0}` — 10% clip được
> gắn nhãn "sự kiện dài ≥ 8 giây", trong khi độ dài **tối đa thực tế là 1.1 giây**.
> Một lát cắt đánh giá nguyên vẹn đang là nhãn rỗng.

Ở thời điểm phát hiện, chưa có cổng kiểm tra các điều khoản này. Hiện đã có verifier
chạy riêng, nhưng chưa được nối thành cổng chặn training tự động.

---

## 1. Yêu cầu

| # | Yêu cầu | Diễn giải kỹ thuật |
|---|---|---|
| R1 | Lưu **mọi** thông số | Config + siêu tham số + môi trường + seed, đủ để tái lập |
| R2 | Lưu **dữ liệu train** đã dùng | Vân tay kiểm chứng được, không phải chỉ đếm số clip |
| R3 | Kết quả xấu → biết **sai ở đâu** | Chia theo lớp / lát cắt / loại lỗi, không chỉ một con số F1 |
| R4 | Biết **cần cải thiện chỗ nào** | Phân loại lỗi có thể hành động được |
| R5 | Giữ **toàn bộ log** để phân tích sau | Lưu dự đoán thô → phân tích lại không cần train lại |

---

## 2. Hiện trạng

| Đã có | Còn thiếu |
|---|---|
| `manifest.json` cho v1/v2/v3 (hồi cứu) và tự động cho run mới; vân tay dữ liệu/mã/git/env | Vân tay dữ liệu của v1/v2 là `null` vĩnh viễn — audio legacy đã xoá |
| Gieo toàn bộ seed Python/NumPy/Torch/CUDA trong `train_sed.py` | Checkpoint vẫn chỉ có weights; chưa lưu optimizer/scheduler/RNG để resume |
| Dự đoán mức đoạn `predictions/{split}_{subset}.npz` cho cả ba run (1,3–4,6 MB/run) | Chưa lưu cho real dev/gold vì hai tập đó chưa tồn tại |
| Quét ngưỡng chạy được trên CPU, 5 s/ngưỡng trên 1.440 clip | Chưa quét ngưỡng **theo từng lớp**; chưa có F1 theo **lát cắt** |
| mAP theo lớp, event-F1 theo lớp ở mọi ngưỡng | Không phân loại lỗi (chèn / sót / nhầm lớp / lệch biên) — Pha 4 |
| `verify_synthetic.py --output`: legacy FAIL, lô B0–B9 PASS train/dev | Cổng chặn tự động trong train mới chỉ CẢNH BÁO, chưa chặn; index vẫn ghi cuối lượt sinh |

**Tài sản đang bị bỏ phí.** Dự án đã kỳ công dựng 5 lát cắt ép buộc và 5 chuỗi nhân quả:

```yaml
forced_slices:
  overlap:      {ratio: 0.30, min_events: 2, min_overlap_ratio: 0.30}
  low_snr:      {ratio: 0.25, max_snr_db: 5}
  reverb:       {ratio: 0.40}
  long_event:   {ratio: 0.10, min_event_duration_sec: 4.0}
  causal_chain: {ratio: 0.15}

causal_chains: [break_in, forced_entry, emergency,
                false_alarm_fireworks, false_alarm_dishes]
```

Training hiện chưa báo cáo metric theo slice. Plan/index đã có để join, nhưng phải QA
điều khoản trước: nhãn slice sai không tạo ra một bảng đánh giá đáng tin cậy.

---

## 3. Quy ước của dự án phải bám theo

Không phát minh lại; mọi thứ dưới đây đã có tiền lệ trong repo.

| Loại | Nguồn tham chiếu | Quy ước |
|---|---|---|
| Lineage | `data/manifests/exclusions.csv` | CSV có `file_id, stage, reason_code, detail, decided_by, decided_at` |
| Ghi bền | `scripts/fetch_audioset_strong.py::append_one_segment` | Ghi **ngay từng dòng** + `flush()`, không dồn tới cuối |
| Nguồn chân lý | `services/api/app/ontology.py` | Đọc từ YAML, không chép lại hằng số vào code |
| Test | `tests/` | Tiếng Việt; docstring nêu **lỗi thật đã xảy ra**, không nêu lý thuyết |
| Đơn vị đo | `ml/evaluation/sed_metrics.py` | Dùng `sed_eval` tham chiếu, không tự cài công thức |
| Encoding | `scripts/common.py::enable_utf8_output` | Gọi ở mọi entrypoint — console Windows mặc định cp1252 |

---

## 4. Các pha

### ✅ Pha 1 — Vân tay dữ liệu & sổ ghi lần chạy *(xong 19/09/2026)*

**File:** `ml/tracking/fingerprint.py` + `ml/tracking/run_manifest.py`
→ ghi `ml/runs/{name}/manifest.json`

| Trường | Vì sao cần |
|---|---|
| `data.fingerprint` | SHA256 của clip_id/nhãn **và checksum audio hoặc snapshot waveform**, config sinh, bank/source lineage. Đếm clip hoặc hash nhãn không đủ phát hiện audio đổi |
| `data.n_clips`, `n_events_per_class` | Phát hiện lệch lớp |
| `data.duration_stats_per_class` | Sẽ bắt ngay lỗi kiểu `event_duration=const 1.0` |
| `data.split_indices_hash` | Khẳng định tập val không đổi giữa các run |
| `config` | Toàn bộ `args` + **nội dung** 3 file YAML đã dùng (không phải đường dẫn) |
| `env` | Phiên bản torch / numpy / CUDA, tên GPU, `pip freeze` |
| `code.tree_hash` | Hash nội dung `ml/` + `scripts/` |
| `timing` | Thời điểm bắt đầu/kết thúc từng epoch |
| `data_contract` | `PASSED` / `FAILED` — kết quả Pha 2 tại thời điểm train |

> Các run v1/v2/v3 được tạo khi base Git còn ở `70ab498` và worktree có thay đổi chưa commit:
> lưu cả SHA và dirty diff/tree hash khi lập manifest hồi cứu; SHA đơn lẻ không đủ tái lập mã
> đã chạy. Snapshot sau đó được publish, nhưng không được gán ngược commit mới cho run cũ.

Pha 1 còn phải lưu toàn bộ seed Python/NumPy/Torch/CUDA, kiến trúc (đặc biệt
`time_pool_blocks`), trạng thái optimizer/scheduler/scaler/RNG và checkpoint cuối có thể
resume. Hiện `--seed` chỉ điều khiển chia train/val; `best.pt` chỉ có weights, class_ids,
epoch, clip_map. `eval_sed.py` mặc định pool=5 nên **chưa dùng an toàn cho v2 pool=3**.

### ◐ Pha 2 — Kiểm tra dữ liệu đúng hợp đồng ⭐ *đã triển khai một phần*

**File:** `scripts/verify_synthetic.py` — chạy **trước** mỗi lần train.

Verifier đã chạy trên cả hai thế hệ dữ liệu. Báo cáo legacy riêng theo split ở
`data/manifests/synthetic_contract_{train,dev}_20260917.csv` ghi các lỗi đã biết;
`data/manifests/synthetic_contract.csv` hiện chỉ có header, tức lô B0–B9 PASS với **0**
vi phạm. **19/09: thêm `--output`** nên mỗi lượt kiểm giữ được báo cáo riêng, và
`manifest.json` trỏ thẳng vào file báo cáo kèm SHA-256 của nó.

⚠️ **Không được mượn báo cáo của lô khác.** Lỗi này xảy ra thật ngay lần chạy đầu của
`run_manifest.py`: manifest hồi cứu cho v1/v2 đọc mặc định `synthetic_contract.csv` —
file đang ghi lô B0–B9 PASS — nên khai **PASSED** cho hai run train trên lô legacy đã
**FAILED**. Đã sửa: hồi cứu không có `--hop-dong` thì ghi `KHONG_RO` kèm cảnh báo, chứ
không mượn.

Thiếu index thì verifier fallback dựng plan, không xác minh được reverb từ dữ liệu lịch sử.
Đếm file bằng nhau chưa chứng minh ID khớp, audio giải mã được hay đạt số lượng đích; verifier
và các phép đối soát riêng kiểm các điều đó ở thời điểm sinh.

| Kiểm tra | Bắt được lỗi gì |
|---|---|
| Clip gắn `long_event` **phải có** sự kiện ≥ ngưỡng | Bắt legacy rỗng 100%. Lô mới dùng ngưỡng 4 s + giới hạn lớp và PASS; ngưỡng 8 s vẫn bất khả thi (STATUS §6) |
| Clip gắn `low_snr` phải có SNR ≤ 5 dB (đọc từ `.jams`) | Slice ép buộc không được thực thi |
| Clip gắn `overlap` phải có ≥2 sự kiện chồng lấn ≥30% | |
| Clip gắn `reverb` phải có `rir_used` khác rỗng | Bank RIR thiếu → slice rỗng âm thầm |
| Clip gắn `causal_chain` phải đúng **thứ tự lớp** của chuỗi | Chuỗi cụt mang nhãn chuỗi đủ |
| Phân bố độ dài mỗi lớp so với bank nguồn | 🔴 Bắt được `event_duration=const 1.0` |
| Số lượng `audio/` = `jams/` = dòng `slice_index.jsonl` | Sinh dở, lệch đôi |
| Mọi nhãn đều thuộc `ontology_map.yaml` | Taxonomy đã đổi sau khi sinh |

Ghi kết quả ra `data/manifests/synthetic_contract.csv` theo đúng schema của
`exclusions.csv` (`stage=verify_synthetic`), để lỗi dữ liệu nằm cùng một chỗ với mọi
quyết định loại bỏ khác.

**Còn cần sửa:** `scripts/scaper_generate.py::write_plan_index` vẫn chỉ chạy ở cuối `main()`,
và `--plan-only` `return` trước khi tới đó → `slice_index.jsonl` **có thể biến mất** nếu job bị
ngắt. Đây là nợ kỹ thuật còn lại cùng loại lỗi đã làm mất 372 dòng `segments.jsonl`. Phải:
- ghi `slice_index.jsonl` cả ở chế độ `--plan-only`
- ghi tăng dần trong lúc sinh, không dồn tới cuối

Đã sửa riêng việc công bố WAV/JAMS: ghi `.pending` rồi chuyển sang tên cuối sau RIR.
Index của cả hai split mới hiện có sau khi chạy xong, nhưng đây chưa phải incremental index.

### ✅ Pha 3 — Lưu dự đoán thô *(xong 19/09/2026)*

**File:** `ml/evaluation/predictions.py` → `ml/runs/{name}/predictions/{split}_{subset}.npz`
· quét ngưỡng: `ml/evaluation/threshold_sweep.py`

Đo thật: v1 **1,3 MB**, v2 **4,6 MB**, v3 **4,5 MB** cho 1.440 clip dev (v1 chỉ 31 đoạn/clip
nên nhỏ hơn ba lần). Inference 11–30 s/run trên RTX 3070. Ước tính ~8 MB/run của kế hoạch
là đúng bậc.

Lưu xác suất **mức đoạn** (31 hoặc 125 đoạn × 15 lớp, float16), kèm `clip_id`, nhãn tham
chiếu, ngưỡng và cửa sổ lọc đã dùng.

> Lưu mức đoạn chứ **không** phải mức khung: 1001 khung chỉ là bản lặp lại của 31 đoạn
> (đã đo — mỗi giá trị lặp đúng 32 lần), nên lưu khung tốn gấp 32 lần mà không thêm một
> bit thông tin nào. Ước tính: ~1 KB/clip → ~8 MB cho 8000 clip.

Cho phép làm **không cần GPU**: quét ngưỡng · đổi cửa sổ lọc · ma trận nhầm lẫn · phân
tích theo lát cắt · so hai run.

### ☐ Pha 4 — Phân tích lỗi có thể hành động

**File:** `ml/evaluation/error_analysis.py` → `ml/runs/{name}/analysis.json` + báo cáo `.md`

**(a) Phân loại lỗi.** F1 nói *bao nhiêu sai*; bảng này nói *sai kiểu gì* và do đó *phải sửa gì*:

| Loại lỗi | Nghĩa | Hành động tương ứng |
|---|---|---|
| Insertion | Đoán ra sự kiện không có | Ngưỡng quá thấp / nhiễu nền |
| Deletion | Bỏ sót sự kiện thật | Lớp hiếm / SNR thấp → cần dữ liệu hoặc mixup |
| Substitution | Đúng thời điểm, **sai lớp** | Lớp gây nhầm → đối chiếu `confusable_with` |
| Boundary | Đúng lớp, lệch onset > collar | Kiểm tra độ phân giải, nhãn và hậu xử lý; bước 323 ms không tự chứng minh bất khả thi với collar 200 ms |
| Fragmentation | 1 sự kiện thật → n sự kiện đoán | Hậu xử lý |
| Merge | n sự kiện thật → 1 sự kiện đoán | Độ phân giải |

**(b) Bảng theo lát cắt** — F1 riêng cho `overlap` / `low_snr` / `reverb` / `long_event` /
`causal_chain` / clip sạch.

> F1 tụt ở `low_snr` là manh mối, chưa xác định nguyên nhân: cần kiểm tra phân bố lớp,
> chất lượng nhãn, ngưỡng và nhiễu trước khi chọn biện pháp. Không kết luận một thay đổi
> chắc chắn có ích hoặc vô ích khi chưa làm ablation.

**(c) Ma trận nhầm lẫn 15×15**, đối chiếu với `confusable_with` đã khai trong
`ontology_map.yaml` — kiểm chứng giả định thiết kế bằng số đo thật.

**(d) Đường cong ngưỡng** — F1 theo ngưỡng cho **từng lớp**. Ngưỡng 0.5 gần như chắc chắn
không tối ưu cho lớp hiếm.

### ☐ Pha 5 — So sánh lần chạy *(tuỳ chọn)*

**File:** `ml/tracking/compare_runs.py`

Bảng đối chiếu n run: metric + **diff của config** + diff của vân tay dữ liệu.

> Điểm mấu chốt: khi v2 hơn v1, công cụ chỉ ra **đúng những gì đã đổi**. Nếu dữ liệu cũng
> đổi thì nó cảnh báo rằng so sánh không sạch — thay vì để ta quy công cho thay đổi kiến
> trúc trong khi thực ra chỉ là dữ liệu nhiều hơn.

**MLflow.** Container đã chạy sẵn ở cổng 5000. Nhưng **JSON/CSV là nguồn chân lý**, MLflow
chỉ là màn hình xem. Cài `mlflow` vào `.venv` có rủi ro kéo numpy 2.x → dùng constraints
file như đã làm với `dcase_util`; nếu không an toàn thì đẩy qua REST API bằng `urllib`
(không thêm dependency nào).

---

## 5. Rủi ro

| Rủi ro | Khả năng | Xử lý |
|---|---|---|
| Cài `mlflow` phá `numpy==1.26.4` → hỏng Scaper | **Cao** | Constraints file + kiểm tra `import scaper` ngay sau khi cài; không đạt thì dùng REST |
| Git có nhưng worktree dirty, commit không mô tả đủ run | Cao | Lưu SHA + dirty diff/tree hash; đưa code vào version control trước run mới |
| Lưu dự đoán phình đĩa | Trung bình | Lưu mức **đoạn**, float16 → ~8 MB/run; đo lại dung lượng trống trước mỗi run vì cache waveform hiện hành đã ~5,6 GiB |
| Index mất khi job mới bị ngắt trước cuối main | Cao | Hai index legacy hiện đã có; vẫn cần ghi tăng dần. Dựng lại plan không chứng minh audio/RIR đã hoàn tất |
| Làm quá tay, thành hạ tầng thay vì khoá luận | Trung bình | Pha 1–4 là tối thiểu; Pha 5 và MLflow tuỳ chọn |

---

## 6. Kiểm chứng

```bash
# Lô B0–B9 hiện tại phải PASS; lỗi khác 0 là cổng chặn train.
.venv/Scripts/python.exe scripts/verify_synthetic.py --split train

# Pha 1 — manifest hồi cứu (chỉ cho run đã xong; run mới tự ghi khi train)
.venv/Scripts/python.exe -m ml.tracking.run_manifest --run panns_ft_v3 --du-lieu hien-tai
.venv/Scripts/python.exe -m ml.tracking.run_manifest --run panns_ft_v1 --du-lieu khong-con \
    --hop-dong data/manifests/synthetic_contract_train_20260917.csv

# Pha 3 — lưu dự đoán rồi quét ngưỡng (khâu quét chạy trên CPU)
.venv/Scripts/python.exe -m ml.evaluation.predictions --run panns_ft_v3 --split dev
.venv/Scripts/python.exe -m ml.evaluation.threshold_sweep --run panns_ft_v3 --split dev

# LỆNH DỰ KIẾN — Pha 4/5 CHƯA CÓ module, không phải runbook đang chạy được
.venv/Scripts/python.exe -m ml.evaluation.error_analysis --run panns_ft_v1
.venv/Scripts/python.exe -m ml.tracking.compare_runs panns_ft_v1 panns_ft_v2

.venv/Scripts/python.exe -m pytest tests/ -q          # test cũ vẫn xanh
```

**Cổng nghiệm thu chính:** `verify_synthetic.py` phải **PASS** trên lô mới trước mỗi train.
Hồ sơ regression legacy giữ báo cáo FAIL đã biết (`long_event` rỗng, độ dài bị kẹp 1,1 s),
chứng minh verifier từng bắt được lỗi thật.

> Một bộ kiểm tra báo xanh trên dữ liệu đã biết là sai thì vô giá trị. Đây là cách duy nhất
> để biết chính bộ kiểm tra có hoạt động hay không.

---

## 7. Ước lượng

**Độ phức tạp: TRUNG BÌNH — 6–9 giờ.** Chạy được song song với sinh dữ liệu và train
(không chiếm GPU).

| Pha | Thời gian | Bắt buộc | Trạng thái |
|---|---|---|---|
| 1 · Vân tay + manifest | 1.5–2 h | ✅ | **xong 19/09** |
| 2 · Kiểm tra hợp đồng dữ liệu | 2–2.5 h | ✅ ⭐ | **xong** (cổng mới CẢNH BÁO, chưa chặn) |
| 3 · Lưu dự đoán + quét ngưỡng | 1 h | ✅ | **xong 19/09** |
| 4 · Phân tích lỗi | 2–3 h | ✅ | chưa bắt đầu |
| 5 · So run + MLflow | 1 h | tuỳ chọn | chưa bắt đầu |

---

## 8. Thứ tự thực thi

1. ✅ **Pha 2** — PASS cho B0–B9; còn phải nâng từ cảnh báo lên cổng chặn thật trong train.
2. ✅ **Pha 1 + 3** — xong 19/09. v1/v2 chỉ hồi cứu được một phần (dữ liệu legacy đã xoá).
3. **Pha 4 — việc tiếp theo.** Prediction đã có sẵn cho cả ba run nên chạy được ngay trên
   CPU: phân loại lỗi, ma trận nhầm lẫn, bảng theo lát cắt (join `slice_index.jsonl` theo
   `clip_id`), và đường cong ngưỡng **theo từng lớp**. Bảng theo lớp ở §2 của
   [measurements/threshold_sweep_20260919.md](measurements/threshold_sweep_20260919.md)
   cho thấy khoảng cách theo lớp rất rộng — đó là nơi Pha 4 có giá trị nhất.
4. **Pha 5** — chỉ thực hiện sau Pha 4; manifest đã đủ để chỉ rõ confound dữ liệu.

---

## 9. Ghi chú về v1/v2/v3 đã hoàn tất

Lô legacy **không đạt hợp đồng**; v1/v2 chỉ là baseline thử nghiệm trên snapshot chung
6.568 clip, chia tạm 5.911/657 từ cùng nguồn. V3 dùng lô B0–B9 đã PASS hợp đồng với cache
7.920 train + 1.440 dev. Cả ba không phải kết quả gold độc lập.

V1 hoàn tất 25 epoch: best mAP 0.8374 ở epoch 23; F1 cuối epoch 25 là segment 0.41196,
event 0.12819, không phải số F1 của best.pt. V2 best mAP 0.8183 ở epoch 23; v3 best/final
mAP 0.8123 ở epoch 25, segment-F1 0.2748, event-F1 0.1077. V2 đồng thời đổi pooling,
mixup và hậu xử lý; v3 đổi dữ liệu. Không quy toàn bộ chênh lệch cho kiến trúc hay dữ liệu.
F1=0 trên epoch không full-eval nghĩa là chưa đo.

**19/09: đã làm.** `ml/runs/{v1,v2,v3}/manifest.json` tồn tại; v1/v2 gắn `FAILED` (trỏ
báo cáo legacy `synthetic_contract_train_20260917.csv`, vân tay dữ liệu `null` vì audio đã
xoá), v3 gắn `PASSED` với vân tay đầy đủ. Dự đoán mức đoạn có cho cả ba trên dev.

Và số của chúng đã đổi nghĩa: chấm lại trên **cùng dev 1.440 clip** ở ngưỡng tối ưu riêng
từng run, event-F1 là v1 **0,2981** · v2 **0,3531** · v3 **0,3986**, mAP clip là
0,7464 · 0,7555 · 0,8132. Bảng cũ ở θ=0,5 cố định không còn là cơ sở để xếp hạng ba run.
Confound còn lại: dev sinh cùng recipe với train của v3 — xem measurements 19/09.
