# Kế hoạch hạ tầng theo dõi & phân tích lỗi huấn luyện

> Lập ngày **17/09/2026**. Duyệt cùng ngày.
>
> **Mục tiêu:** khi một lần train ra kết quả xấu, phải trả lời được *sai ở đâu* và *cần
> sửa chỗ nào* — bằng số đo đã lưu sẵn, không phải bằng phỏng đoán hay train lại.
>
> Tài liệu liên quan: [PLAN.md](PLAN.md) (lịch W1–W8) · [DATA_PLAN.md](DATA_PLAN.md)
> (quy trình dữ liệu) · [SYSTEM.md](SYSTEM.md) (kiến trúc) ·
> [RELATED_WORK_2026.md](RELATED_WORK_2026.md) (đối chiếu văn liệu)

> **Đối soát 19/09/2026: Pha 1–5 đã có module và đã chạy thật.**
> Bằng chứng ở [STATUS.md](STATUS.md),
> [measurements/threshold_sweep_20260919.md](measurements/threshold_sweep_20260919.md) và
> [measurements/error_analysis_20260919.md](measurements/error_analysis_20260919.md).
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
| Gieo toàn bộ seed Python/NumPy/Torch/CUDA trong `train_sed.py`; `checkpoint_resume.pt` lưu đủ optimizer/scheduler/RNG mỗi epoch (21/09) | `--resume` chưa được kiểm bằng một lượt train thật đứt giữa chừng — chỉ có test đơn vị |
| Dự đoán mức đoạn `predictions/{split}_{subset}.npz` cho cả ba run (1,3–4,6 MB/run) | Chưa lưu cho real dev/gold vì hai tập đó chưa tồn tại |
| Quét ngưỡng chạy được trên CPU, 5 s/ngưỡng trên 1.440 clip | θ riêng từng lớp đã đo và **làm v1 tệ đi** — F1 tổng là micro-average, xem measurements §5 |
| Phân loại 6 loại lỗi, F1 theo lát cắt, ma trận nhầm 16×16, θ theo lớp | Chưa có tập độc lập để xác nhận: `gold_test` rỗng, chưa có real dev |
| `verify_synthetic.py --output`: legacy FAIL, lô B0–B9 PASS train/dev | index vẫn ghi cuối lượt sinh — chưa ghi tăng dần |

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

### ✅ Pha 4 — Phân tích lỗi có thể hành động *(xong 19/09)*

> **Đã chạy thật trên cả ba run.** Số đo:
> [measurements/error_analysis_20260919.md](measurements/error_analysis_20260919.md).
> Ba kết quả đáng chú ý: (a) ở θ = 0,5, v3 chỉ bỏ sót **9/4.873** sự kiện nhưng chèn thêm
> **61.020** — hỏng ở ngưỡng, không phải ở năng lực phát hiện; (b) **37,4%** lỗi biên của
> v1 là **trần cứng** do bước lưới onset 319,7 ms, đo được chứ không suy luận; (c) chỉ
> **31%** lượt nhầm lớp rơi vào cặp đã khai trong `confusable_with`.

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

### ✅ Pha 5 — So sánh lần chạy *(xong 19/09)*

**File:** `ml/tracking/compare_runs.py` · test: `tests/test_ml_compare_runs.py` (11)

Bảng đối chiếu n run: metric + **diff của config** + diff của vân tay dữ liệu.

> Điểm mấu chốt: khi v2 hơn v1, công cụ chỉ ra **đúng những gì đã đổi**. Nếu dữ liệu cũng
> đổi thì nó cảnh báo rằng so sánh không sạch — thay vì để ta quy công cho thay đổi kiến
> trúc trong khi thực ra chỉ là dữ liệu nhiều hơn.

Chạy trên v1/v2/v3, công cụ từ chối khai ba thứ mà bảng so sánh thủ công sẽ khai bừa:

> - **dữ liệu: `KHONG_RO`** — vân tay của v1/v2 là `null` (audio legacy xoá 18/09).
>   `None == None` không phải bằng chứng cùng dữ liệu.
> - **mã nguồn: `KHONG_RO`** — cả ba manifest lập hồi cứu trong cùng một phút nên
>   `code.tree_sha256` của chúng **trùng nhau**; băm đó là mã lúc lập manifest, không
>   phải lúc train.
> - **metric**: ưu tiên `analysis.json` (cùng `data/synthetic/dev`) và in kèm θ. Đọc
>   `manifest.ket_qua_cuoi` là sai: nó chấm trên val split **riêng** của từng run
>   (0,1282 / 0,1897 / 0,1077 → xếp v2 > v1 > v3, **ngược hẳn** dev chung).
>
> Diff cấu hình phân biệt `<không có>` với "giá trị khác": v1 không có `mixup_alpha`,
> `time_pool_blocks`, `adaptive_postproc` vì các cờ đó ra đời sau v1 — nghĩa là v1 chạy
> bằng một phiên bản code khác, không phải ai đó chỉnh tham số. v2 và v3 khác nhau
> **đúng mỗi `name`**, nên thứ phân biệt chúng nằm ở dữ liệu — mà dữ liệu thì `KHONG_RO`.

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
| Làm quá tay, thành hạ tầng thay vì khoá luận | Trung bình | Pha 1–5 xong, tổng ~1.500 dòng; MLflow vẫn tuỳ chọn và vẫn chưa cài |

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

# Pha 4 — phân tích lỗi (CPU, ~100 s/run)
.venv/Scripts/python.exe -m ml.evaluation.error_analysis --run panns_ft_v3 --split dev
# ablation hậu xử lý — ghi ra analysis_adaptive.{json,md}, KHÔNG đè báo cáo mặc định
.venv/Scripts/python.exe -m ml.evaluation.error_analysis --run panns_ft_v3 --adaptive-postproc
# quét trần cửa sổ — mỗi mức ra file riêng (analysis_adaptive_max<N>.{json,md}), ~300-420s/lượt
.venv/Scripts/python.exe -m ml.evaluation.error_analysis --run panns_ft_v3 \
    --adaptive-postproc --max-median-frames 401
# cửa sổ suy từ TRAIN thay vì từ chính tập đang chấm — không rò rỉ (analysis_adaptive_train.{json,md})
.venv/Scripts/python.exe -m ml.evaluation.error_analysis --run panns_ft_v3 \
    --adaptive-postproc --adaptive-source train

# Hiệu chuẩn xác suất — ECE + reliability diagram, không train lại, ~5-6s/run
.venv/Scripts/python.exe -m ml.evaluation.calibration --run panns_ft_v3 --split dev

# long_event — khe hở năng lượng thật bên trong nhãn (CPU, ~1-3s/run)
.venv/Scripts/python.exe -m ml.evaluation.long_event_gap --run panns_ft_v3 --split dev

# Cổng hợp đồng CHẶN train nếu khác PASSED — dùng cờ dưới đây chỉ khi cố ý bỏ qua
.venv/Scripts/python.exe -m ml.training.train_sed --force-du-lieu-chua-dat
# Resume — tiếp tục ml/runs/<name>/checkpoint_resume.pt nếu có, khôi phục RNG đầy đủ
.venv/Scripts/python.exe -m ml.training.train_sed --name panns_ft_v4 --resume

# Pha 5 — đối chiếu nhiều run (tức thì; --out để ghi .md thay vì in ra màn hình)
.venv/Scripts/python.exe -m ml.tracking.compare_runs --runs panns_ft_v1 panns_ft_v2 panns_ft_v3

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
| 2 · Kiểm tra hợp đồng dữ liệu | 2–2.5 h | ✅ ⭐ | **xong** (cổng CHẶN thật từ 21/09, `--force-du-lieu-chua-dat` bỏ qua có ghi manifest) |
| 3 · Lưu dự đoán + quét ngưỡng | 1 h | ✅ | **xong 19/09** |
| 4 · Phân tích lỗi | 2–3 h | ✅ | **xong 19/09** |
| 5 · So run + MLflow | 1 h | tuỳ chọn | **xong 19/09** (MLflow vẫn chưa cài) |

---

## 8. Thứ tự thực thi

1. ✅ **Pha 2** — PASS cho B0–B9; cổng nâng từ cảnh báo lên chặn thật trong train xong 21/09.
2. ✅ **Pha 1 + 3** — xong 19/09. v1/v2 chỉ hồi cứu được một phần (dữ liệu legacy đã xoá).
3. ✅ **Pha 4** — xong 19/09 trên cả ba run, CPU, ~3 phút/run. Kết quả ở
   [measurements/error_analysis_20260919.md](measurements/error_analysis_20260919.md).
   Việc theo sau mà chính Pha 4 sinh ra: bổ sung các cặp nhầm thật vào `confusable_with`
   (69% lượt nhầm chưa được khai), và tìm nguyên nhân `long_event` — lát cắt tệ nhất ở
   cả v2 lẫn v3.
4. ✅ **Pha 5** — xong 19/09. Chạy trên v1/v2/v3 thì cả dữ liệu lẫn mã nguồn đều ra
   `KHONG_RO`, tức **không có phép so sánh sạch nào giữa ba run hiện có**. Đó là kết
   luận, không phải lỗi công cụ.
5. ✅ **Ba việc theo sau Pha 4/5** — xong 19/09 (tối). (a) Quét `--max-median-frames`
   ∈ {51,101,201,401}: trần 51 khung **không phải** nguyên nhân chính của `long_event`
   — `v2` gap rộng ra, `v3` chỉ hẹp 6,4%; cột "gộp" đứng yên qua cả 4 mức, cảnh báo
   "nới trần sẽ gộp nhầm" ở `sed_metrics.py:37` vẫn chưa có số đỡ. (b)
   `cua_so_loc_tu_train()` — cửa sổ thích ứng suy từ `data/features/train_meta.json`
   thay vì từ chính tập đang chấm: độ lớn rò rỉ đo được **= 0**, nhưng vì 9/15 lớp đã
   kẹp trần 51 ở cả hai nguồn, không phải vì train/dev giống nhau. (c)
   `ml/evaluation/calibration.py` — ECE + reliability diagram lần đầu: vùng dự báo
   0,4–0,6 chiếm hơn nửa triệu khung mà tỉ lệ dương thật chỉ 1,7–3%, giải thích hợp lý
   cho θ* luôn rơi 0,85–0,95 (chưa kết luận nguyên nhân, `pos_weight` cần train lại để
   xác nhận). Phép đo:
   [measurements/max_median_ceiling_20260919.md](measurements/max_median_ceiling_20260919.md) ·
   [measurements/adaptive_window_leak_20260919.md](measurements/adaptive_window_leak_20260919.md) ·
   [measurements/calibration_20260919.md](measurements/calibration_20260919.md).
6. ✅ **`long_event` — kiểm ứng viên cuối** — xong 21/09. `ml/evaluation/long_event_gap.py`
   so khe hở năng lượng trong nhãn giữa nhóm sự kiện bị phân mảnh (n=178) và nhóm không
   (n=137), cùng lát cắt, chỉ xét sự kiện ≥1s. Khe hở trung bình **bằng nhau tuyệt đối**
   ở cả hai ngưỡng "im" đã thử (20%: 0,012s; 40%: 0,046s); p90 nhóm KHÔNG phân mảnh còn
   cao hơn — ngược hướng giả thuyết. **Ba ứng viên liên tiếp đều bị loại** (cửa sổ hẹp,
   trần thấp, khe hở năng lượng); nguyên nhân riêng của `long_event` **vẫn chưa xác
   định**, không còn ứng viên thứ tư đang chờ sẵn. Phép đo:
   [measurements/long_event_gap_20260921.md](measurements/long_event_gap_20260921.md).
7. ✅ **Cổng hợp đồng CHẶN thật + checkpoint resume** — xong 21/09.
   `kiem_cong_hop_dong()` chặn cả `FAILED` lẫn `KHONG_RO` (cổng cũ chỉ cảnh báo cho cả
   hai); `--force-du-lieu-chua-dat` bỏ qua được, cờ ghi vào `manifest.json`.
   `checkpoint_resume.pt` ghi đè mỗi epoch (model/optimizer/scheduler/scaler + RNG bốn
   nguồn), tách khỏi `best.pt`; `--resume` khôi phục RNG cho dãy số tiếp theo giống hệt
   như chưa hề dừng. Test đơn vị, **chưa** kiểm bằng một lượt train thật đứt giữa chừng.
8. **Việc tiếp theo (chốt 21/09):** `gold_test` vẫn rỗng — vẫn là nút thắt duy nhất chưa
   gỡ được của confound "dev cùng recipe với train của v3". `long_event` cần hướng điều
   tra mới — nghe trực tiếp clip bị phân mảnh, hoặc so theo trục SNR/reverb thay vì trục
   độ dài sự kiện. Ablation `pos_weight` giờ không còn bị hai việc trên chặn nữa, có thể
   bắt đầu.
10. ✅ **Ablation `pos_weight` CHẠY XONG — 22/09, giả thuyết được xác nhận.** Ba lượt
    `panns_ft_pw1`/`pw10`/`pw30` chạy tuần tự qua đêm không người trông bằng
    `scripts/chay_ablation_pos_weight.py` (chờ train → đánh giá → train lượt sau → ... →
    `compare_runs` → sinh báo cáo; mọi bước bỏ qua được nếu đã có kết quả; watchdog
    `--resume` nếu train chết). Cùng cấu hình `v3`, chỉ đổi `--pos-weight-max`; hợp đồng
    dữ liệu `PASSED`, manifest ghi **lúc train**.

    | run | pos_weight | ECE | θ* | F1@θ* | F1@0,5 |
    |---|---:|---:|---:|---:|---:|
    | `pw30` | 30 | 0,3308 | 0,90 | 0,4018 | 0,0855 |
    | `pw10` | 10 | 0,2033 | 0,80 | 0,4192 | 0,2340 |
    | `pw1` | 1 | **0,0231** | **0,35** | **0,4333** | **0,4160** |

    Ba kết luận: (a) `pw30` tái hiện `v3` trong 0,7% → giả thuyết không phải rút lại, và
    đây là **bằng chứng đầu tiên về khả năng tái lập của pipeline**; (b) quan hệ **đơn điệu
    tuyệt đối** ở cả bốn cột, không có ngưỡng lật nên không cần quét thêm mức trung gian;
    (c) **không có đánh đổi** — trần 30 làm hỏng cả hiệu chuẩn lẫn F1, trái giả định ban đầu.

    `compare_runs` nay khai `mã: khac` thay vì `KHONG_RO` (đã hết, chỉ còn khi kéo `v3` vào
    so). Băm cây mã khác nhau vì code ĐO thay đổi giữa các lượt; `ml/training/` và
    `ml/models/` **giống hệt**, mọi `analysis.json` do cùng một phiên bản `error_analysis.py`
    sinh ra. Phép so hợp lệ nhưng **chưa "sạch tuyệt đối"** theo nghĩa băm mã trùng nhau.

    **Quyết định:** giữ nguyên hằng `MAX_POS_WEIGHT = 30,0`; v4 truyền tường minh
    `--pos-weight-max 1.0`. Số đo: `docs/measurements/pos_weight_ket_luan_20260922.md`.

    ⚠️ **`--resume` vẫn chưa được kiểm bằng thực tế** — cả ba lượt chạy trót lọt nên nhánh
    watchdog không lần nào được thực thi.

11. ✅ **Kiểm lại hai chẩn đoán `long_event`/`low_snr` trên θ\* của `pw1` — 22/09.** Cả hai
    chẩn đoán đo trên `v3` @θ\*=0,90 — ngưỡng cắt sát trần, nên đáng nghi bị phóng đại.
    Chạy lại `ml/evaluation/long_event_crosscut.py --run panns_ft_pw1` (10 s, không train
    lại): tỉ lệ phân mảnh `long_event` **giữ nguyên** ở mọi ô đối chiếu (chênh lệch với
    `khac` vẫn 0,09–0,17, so với 0,11–0,20 ở `v3`) — không phải hệ quả của θ\* cực đoan.
    Cơ chế Deletion trên sự kiện thường **giữ nguyên** (tỉ trọng 50,6%→52,2%). Nhưng câu đã
    ghi "lệch biên chiếm 61–72%, Deletion gần như không đóng góp" trên `long_event` **bị
    phóng đại**: tỉ trọng Deletion trong phần `dung` mất đi tăng gấp ba (10,2%→32,1%) khi
    θ\* hợp lý hơn — lệch biên vẫn trội hơn (50,3%) nhưng không còn áp đảo. Đã đính chính
    trong `docs/measurements/low_snr_co_che_20260921.md` §6, chi tiết đầy đủ ở
    `docs/measurements/phan_tich_lai_tren_pw1_20260922.md`. Không có code/test mới.

9. ✅ **Hạ tầng ablation `pos_weight` + bộ gán mù lần 2 — xong 21–22/09.**
   `train_sed.py` thêm `--pos-weight-max` (mặc định vẫn 30,0, hành vi cũ không đổi) —
   **chưa chạy lượt train nào với cờ này**, vẫn còn nguyên là việc phải làm để xác nhận
   giả thuyết ECE. Song song: `scripts/make_blind_set.py` (đã chạy thật) và
   `scripts/agreement.py` dựng xong cho pilot gold lần 2 — không thuộc phạm vi
   TRAINING_OPS_PLAN (đây là quy trình dữ liệu DATA_PLAN §8), ghi ở đây chỉ để nói rõ
   **`gold_test` KHÔNG còn rỗng nữa** nhưng cũng **CHƯA sẵn sàng dùng**: 30 clip pilot đã
   gán và nghe lại lần ba đủ 3 lượt (`data/gold/pilot_v1_lan1.tsv`,
   `pilot_v1_lan2.tsv`, `decision_log.md`), nhưng cổng tự-nhất-quán §8.5 **chưa đạt**
   (event-F1 0,7339 < 0,75) — cần một vòng pilot mới trước khi 30 clip này (hoặc phần mở
   rộng của `gold_test`) dùng được cho đánh giá thật. Không có thay đổi nào ở đây ảnh
   hưởng tới các con số v1/v2/v3 đã có — chúng vẫn đo trên `data/synthetic/dev`.

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
