# DATA_PLAN.md — Kế hoạch chuẩn bị dữ liệu

> **Vai trò:** kế hoạch chi tiết để đi từ *không có gì* → *một bộ dữ liệu chuẩn, đủ 16 class, có strong label, đóng băng được và tái lập được*.
>
> Liên quan: [SYSTEM.md §3](SYSTEM.md#3-dữ-liệu) (đặc tả dữ liệu) · [PLAN.md W2–W3](PLAN.md) (lịch) · [taxonomy.md](taxonomy.md) (16 class)
>
> **Nguyên tắc bao trùm:** *dữ liệu nào không truy vết được nguồn gốc và lý do tồn tại thì không được vào dataset.*

---

## 0. Ba nguyên tắc bất khả xâm phạm

Đọc kỹ ba điều này trước mọi thứ khác. Vi phạm bất kỳ điều nào là hỏng toàn bộ phần thực nghiệm.

### N1 — Gold test set phải gán nhãn MÙ (blind)

Máy **được phép** hỗ trợ gán nhãn cho foreground bank và train/dev. Máy **tuyệt đối không** được đề xuất nhãn hay biên thời gian cho gold test set.

**Lý do:** khi model đề xuất `glass_breaking @ [1.2, 2.8]` và người chỉ việc bấm ✓, người sẽ chấp nhận cả những đề xuất sai một cách hệ thống. Gold set khi đó nghiêng về phía model — model được chấm bằng chính định kiến của nó. Mọi con số Event-F1, EHR, Alert Precision đều mất giá trị, và đây là lỗi hội đồng hoàn toàn có thể hỏi.

**Thoả hiệp duy nhất được phép:** máy được đánh dấu *"vùng này có âm thanh vượt nền"* (chỉ năng lượng, không phân lớp, không đề xuất biên chính xác) để người biết chỗ nào cần nghe kỹ. Nếu dùng thoả hiệp này thì **phải ghi vào báo cáo**.

### N2 — Chia tập theo nguồn gốc, không theo file

Hai đoạn cắt từ cùng một video YouTube, cùng một bản ghi Freesound, hoặc cùng một người upload thì **phải nằm cùng một tập**. Chia ngẫu nhiên theo file là dạng rò rỉ dữ liệu phổ biến nhất và khó phát hiện nhất với audio.

Hệ quả: mọi file phải mang `source_group_id` từ lúc tải về, không phải gán sau.

### N3 — Mọi file bị loại bỏ đều phải ghi lý do

`data/manifests/exclusions.csv` ghi từng file bị loại + mã lý do. Không có file này thì dataset không tái lập được, và không trả lời được câu hỏi *"tại sao class X chỉ còn 80 clip?"*.

---

## 1. Ba loại dữ liệu, ba mức chi phí khác nhau

Sai lầm hay gặp là coi "gán nhãn" là một việc duy nhất. Thực tế có ba việc rất khác nhau:

| Loại | Cần gì | Chi phí | Dùng cho | Máy hỗ trợ được? |
|---|---|---|---|---|
| **A — Foreground bank** | Clip đơn sự kiện, đã cắt gọn, đúng lớp | ~1–2× realtime | Nguyên liệu cho Scaper | ✅ **Rất nhiều** |
| **B — Background bank** | Đoạn nền **không chứa** sự kiện Nhóm A | ~0.5× realtime | Nền cho Scaper + lớp `ambient_noise` | ✅ Sàng lọc tự động |
| **C — Gold test set** | Strong label onset/offset trên audio thật | **5–10× realtime** | Con số duy nhất được báo cáo | ❌ **Cấm** (xem N1) |

**Chìa khoá của cả kế hoạch:** strong label cho *train* không cần gán tay chút nào — Scaper biết chính xác nó đặt foreground ở đâu, nên nó **sinh ra strong label miễn phí và chính xác tuyệt đối**. Việc của người chỉ là làm sạch nguyên liệu (loại A, rẻ) và gán gold (loại C, đắt nhưng ít).

```
Loại A (rẻ, máy hỗ trợ)  ─┐
                          ├─► Scaper ─► 20-30h train/dev CÓ STRONG LABEL MIỄN PHÍ
Loại B (rẻ, máy sàng)    ─┘

Loại C (đắt, người mù)   ────────────► ~1h gold test set
```

---

## 2. Ma trận thu thập theo lớp

Cột **Nhãn có sẵn** cho biết ta được cho không cái gì; cột **Việc phải làm** cho biết ta phải tự làm gì.

### Nhóm A — Sự kiện an ninh

| # | `class_id` | Nguồn chính | Nhãn có sẵn | Mục tiêu FG | Việc phải làm | Rủi ro |
|---|---|---|---|---|---|---|
| 1 | `gunshot` | MIVIA, AudioSet-strong, FSD50K | **Strong** (MIVIA, AS-strong) · Weak (FSD50K) | 200 | Cắt gọn FSD50K; tách loạt bắn | 🟡 Dễ lẫn `fireworks` |
| 2 | `explosion` | AudioSet-strong, FSD50K | Strong (AS-strong) · Weak | 120 | Cắt gọn; loại nhầm pháo hoa | 🔴 **Thiếu dữ liệu** |
| 3 | `scream` | MIVIA, AudioSet-strong, FSD50K | **Strong** (MIVIA) · Weak | 200 | Tách khỏi `shout_yell`, `laughter`, `applause_cheering` | 🟡 Ranh giới lớp khó |
| 4 | `glass_breaking` | MIVIA, FSD50K, ESC-50 | **Strong** (MIVIA) · Weak | 250 | Cắt gọn; loại nhầm chén đĩa | 🟢 |
| 5 | `vehicle_crash` | MIVIA Road ~~, AudioSet-strong~~ | **Strong** (MIVIA Road) | 80 | Phụ thuộc **tuyệt đối** vào MIVIA Road | 🔴 **Rủi ro cao nhất** |
| 6 | `shout_yell` | AudioSet-strong, FSD50K | Strong (AS-strong) · Weak | 200 | Tách khỏi `scream` theo guideline | 🟡 |
| 7 | `door_slam` | FSD50K, AudioSet-strong ~~, ESC-50~~ | Strong (AS-strong) · Weak | 250 | Cắt gọn; gom `Door`+`Slam` | 🟢 |
| 8 | `running_footsteps` | FSD50K, AudioSet-strong, ESC-50 ⚠️ | Strong (AS-strong) · Weak | 250 | Phân biệt đi bộ / chạy | 🟡 |
| 9 | `siren` | UrbanSound8K, ESC-50, AudioSet | Weak + salience | 300 | Cắt đoạn; tách `siren` vs `alarm_bell` | 🟢 |
| 10 | `alarm_bell` | DESED, FSD50K, ESC-50 | **Strong** (DESED) · Weak | 250 | DESED cho trực tiếp | 🟢 |

### Nhóm B — Lớp gây nhầm lẫn & nền

| # | `class_id` | Nguồn chính | Nhãn có sẵn | Mục tiêu FG | Việc phải làm | Rủi ro |
|---|---|---|---|---|---|---|
| 11 | `fireworks` | FSD50K, AudioSet-strong | Strong (AS-strong) · Weak | 200 | **Thu bổ sung tại VN** nếu thiếu | 🟡 Quan trọng, dễ thiếu |
| 12 | `object_drop_dishes` | DESED (`Dishes`), FSD50K | **Strong** (DESED) | 250 | DESED cho trực tiếp | 🟢 |
| 13a | `laughter` | FSD50K, AudioSet-strong, ESC-50 | Strong (AS-strong) · Weak | 150 | Tách khỏi `applause_cheering` (2026-09-15) — một giọng liên tục | 🟢 |
| 13b | `applause_cheering` | FSD50K, AudioSet-strong, ESC-50 | Strong (AS-strong) · Weak | 250 | Gom `Cheering`+`Applause`+`Children shouting` | 🟢 |
| 14 | `speech_normal` | DESED (`Speech`), AudioSet, LibriSpeech | **Strong** (DESED) | 300 | Rất dồi dào; cần đa dạng ngôn ngữ | 🟢 |
| 15 | `ambient_noise` | TAU/DCASE scenes, UrbanSound8K, **thu tại chỗ** | — | *(dùng background bank)* | Xác minh **không chứa** sự kiện Nhóm A | 🟢 |

**Ba lớp đỏ cần xử lý sớm:** `vehicle_crash` (phụ thuộc MIVIA Road → **nộp đơn ngày đầu**), `explosion` (khan hiếm tự nhiên), `fireworks` (dữ liệu quốc tế có thể khác pháo VN).

### Bảng ánh xạ ontology — file cấu hình, không phải tài liệu

Tạo `ml/configs/ontology_map.yaml` làm **nguồn chân lý duy nhất** để script dùng:

```yaml
gunshot:
  audioset_ids: ["/m/032s66", "/m/04zjc", "/m/073cg4"]   # ⚠️ CẦN XÁC MINH
  audioset_names: ["Gunshot, gunfire", "Machine gun", "Cap gun"]
  fsd50k_labels: ["Gunshot_and_gunfire"]
  esc50_labels: []
  urbansound_labels: ["gun_shot"]
  desed_labels: []
  mivia_labels: ["gunshot"]
  confusable_with: ["fireworks", "explosion", "door_slam"]
  target_foreground: 200
  min_acceptable: 80
```

`confusable_with` không phải trang trí — nó điều khiển luật định tuyến sang người ở [§4.2](#42-giai-đoạn-2--sàng-lọc-tự-động).

⚠️ **Toàn bộ ID ontology phải được xác minh từ file ontology gốc của AudioSet**, không chép từ trí nhớ. Đây là task đầu tiên của D1.

### ✅ Kết quả xác minh D1 (14/09/2026)

Đã tải file gốc về `data/reference/` và đối chiếu bằng `scripts/verify_ontology.py`:

| File tham chiếu | Nội dung | Dùng để |
|---|---|---|
| `audioset_ontology.json` | 632 lớp AudioSet | Xác minh mọi `audioset_ids` **và tên đi kèm** |
| `audioset_strong_mid_to_display_name.tsv` | **456** lớp có strong label | Biết lớp nào thực sự có onset/offset |
| `esc50_meta.csv` | 2000 clip / 50 lớp | Xác minh `esc50_labels` |

**Bốn phát hiện làm thay đổi kế hoạch:**

1. 🔴 **AudioSet không có lớp nào nghĩa là "va chạm xe".** Gần nhất là `/m/07pjjrj Smash, crash` — tiếng vỡ/đập nói chung, không đặc thù xe cộ. `vehicle_crash` vì thế phụ thuộc MIVIA Road **tuyệt đối**, chứ không phải "chủ yếu" như đánh giá ban đầu. Nộp đơn ngay.

2. ❌ **ESC-50 không dùng được cho `door_slam`.** Hai lớp cửa của ESC-50 là `door_wood_knock` (gõ cửa) và `door_wood_creaks` (cọt kẹt) — không lớp nào là đóng sầm. Bảng trên đã sửa.

3. ⚠️ **`footsteps` (ESC-50) và `Walk, footsteps` (AudioSet) phần lớn là đi bộ, không phải chạy.** Lớp của ta là `running_footsteps` → tỉ lệ loại ở lớp này sẽ cao bất thường. Đây là hiện tượng bình thường, không phải lỗi quy trình.

4. ✅ 47/49 ID ta cần **có** strong label. Chỉ `Skidding` và `Chatter` là weak-only, và cả hai đều đã khai báo rõ trong `audioset_weak_only`.

Kiểm tra lại bất cứ lúc nào: `python scripts/verify_ontology.py` (0 lỗi = đạt). Có 17 test ở `tests/test_verify_ontology.py` chứng minh script này thực sự bắt lỗi chứ không chỉ báo đạt.

---

### ⚠️ Bẫy phân cấp ontology — đọc trước khi chọn clip từ BẤT KỲ nguồn nào

AudioSet gắn nhãn theo phân cấp, và FSD50K kế thừa điều đó: clip nào mang nhãn con thì **luôn** mang cả nhãn cha.

```
Gunshot, gunfire  ⊂  Explosion
Fireworks         ⊂  Explosion
Siren             ⊂  Alarm
Shatter           ⊂  Glass
Slam              ⊂  Door
```

Ba dòng đầu bắc cầu giữa **hai lớp khác nhau của ta**. Nên nếu áp luật *"clip có nhãn thuộc hai lớp của ta ⇒ nhiều sự kiện ⇒ loại"*, kết quả là:

| Lớp | Không xử lý phân cấp | Có xử lý phân cấp |
|---|---:|---:|
| `gunshot` | **0** clip | **236** clip chất lượng PP |
| `fireworks` | **0** | **221** |
| `siren` | **0** | 27 |

Ta sẽ kết luận "FSD50K không có ba lớp này" và bỏ mất nguồn tốt nhất cho chúng — **một kết luận sai mà không có thông báo lỗi nào**.

**Cách xử lý:** `scripts/audioset_ontology.py` bỏ mọi mã là tổ tiên của một mã khác trong cùng clip, giữ mã cụ thể nhất. `{Gunshot, Explosion}` → `{Gunshot}`; nhưng `{Gunshot, Fireworks}` là hai anh em nên giữ cả hai — đó mới thật sự là clip nhiều sự kiện.

**Áp dụng lại khi xử lý AudioSet-strong**, vì nguồn đó cũng dùng cùng ontology.

### 🎁 PP/PNP của FSD50K — bộ lọc chất lượng cho không

FSD50K công bố `pp_pnp_ratings_FSD50K.json`: mỗi nhãn được người đánh giá chấm **PP** (âm thanh có mặt và **nổi trội**), PNP (có mặt nhưng không nổi trội), U (không chắc), NP (không có).

Foreground bank cần đúng clip **PP**. Lấy điều kiện *một lớp duy nhất sau khi thu gọn phân cấp* **và** *mọi phiếu chấm đều PP*, ta có sẵn số liệu dưới đây **trước khi tải một byte audio nào**:

| Lớp | Đơn lớp | PP | Tối thiểu | |
|---|---:|---:|---:|---|
| `alarm_bell` | 2560 | 441 | 100 | ✅ |
| `laughter_cheering` *(số cũ, trước tách lớp 15/09)* | 1643 | 779 | 100 | ⚠️ cần đo lại `laughter`/`applause_cheering` riêng |
| `speech_normal` | 1906 | 912 | 120 | ✅ |
| `glass_breaking` | 1073 | 516 | 100 | ✅ |
| `door_slam` | 1244 | 397 | 100 | ✅ |
| `running_footsteps` | 758 | 341 | 100 | ✅ |
| `object_drop_dishes` | 765 | 338 | 100 | ✅ |
| `scream` | 308 | 237 | 80 | ✅ |
| `gunshot` | 463 | 236 | 80 | ✅ |
| `fireworks` | 442 | 221 | 80 | ✅ |
| `explosion` | 425 | 111 | 50 | ✅ |
| `shout_yell` | 197 | 77 | 80 | 🟡 sát ngưỡng |
| `siren` | 120 | 27 | 120 | 🔴 trông vào UrbanSound8K |
| `vehicle_crash` | 0 | 0 | 30 | ⛔ MIVIA Road |

**Ý nghĩa:** FSD50K mở khoá 11/13 lớp còn thiếu, và ta biết điều đó **trước khi tải**. Đây là lý do phải tải `FSD50K.metadata` (~34 MB) ngay từ đầu, tách khỏi audio (18.4 GB) — metadata rẻ nhưng quyết định toàn bộ chiến lược.

---

## 3. Hợp đồng thư mục & manifest

### 3.1 Cấu trúc

```
data/
├── reference/                    # ★ metadata GỐC của các dataset nguồn
│   ├── audioset_ontology.json          # 632 lớp — nguồn chân lý cho mọi audioset_id
│   ├── audioset_strong_mid_to_display_name.tsv   # 456 lớp CÓ strong label
│   └── esc50_meta.csv                  # 2000 clip / 50 lớp
├── raw/                          # NGUYÊN BẢN, không bao giờ sửa
│   ├── audioset_strong/<class_id>/
│   ├── fsd50k/<class_id>/
│   ├── esc50/<class_id>/
│   ├── urbansound8k/<class_id>/
│   ├── desed/<class_id>/
│   ├── mivia/<class_id>/
│   └── iuh_field/<session_id>/
├── interim/
│   └── normalized/<class_id>/    # 16kHz mono, EBU R128
├── banks/
│   ├── foreground/<class_id>/    # ✅ đã duyệt, đã cắt gọn
│   ├── background/<area_type>/   # ✅ đã xác minh không có sự kiện
│   └── rir/<space_type>/
├── synthetic/
│   ├── train/{audio,jams,tsv}/
│   └── dev/{audio,jams,tsv}/
├── gold/
│   ├── audio/
│   ├── g1_sed.tsv
│   ├── g1_metadata.csv
│   ├── g2_captions.jsonl
│   ├── g3_rag_queries.jsonl
│   └── g4_alerts.jsonl
└── manifests/
    ├── raw_manifest.csv          # MỌI file từng tải về
    ├── foreground_manifest.csv   # file đã vào bank
    ├── background_manifest.csv
    ├── exclusions.csv            # ★ mọi file bị loại + lý do
    └── dedup_groups.csv          # nhóm trùng lặp phát hiện được
```

**Quy tắc:** `raw/` chỉ ghi một lần, không bao giờ sửa. Mọi biến đổi tạo file mới ở tầng sau. Muốn làm lại từ đầu thì xoá `interim/` trở đi, `raw/` giữ nguyên.

### 3.2 Schema manifest

`raw_manifest.csv` — nguồn chân lý cho mọi file audio trong dự án:

| Cột | Mô tả |
|---|---|
| `file_id` | ID nội bộ, bất biến: `{source}_{class}_{hash8}` |
| `path_raw` | Đường dẫn trong `data/raw/` |
| `source_dataset` | `audioset_strong` \| `fsd50k` \| `esc50` \| `urbansound8k` \| `desed` \| `mivia` \| `iuh_field` |
| `source_id` | ID gốc trong dataset nguồn (YTID, fsd_id, …) |
| **`source_group_id`** | **Nhóm chống rò rỉ**: video YouTube, uploader Freesound, buổi thu. ★ N2 |
| `claimed_class` | Lớp mà nguồn nói |
| `label_type_orig` | `strong` \| `weak` \| `isolated_event` \| `none` — `isolated_event` nghĩa là cả file LÀ một sự kiện đã tách sẵn (soundbank DESED), tức nguyên liệu Scaper dùng được ngay |
| `orig_onset`, `orig_offset` | Nếu nguồn có strong label |
| `orig_split` | Split/fold do CHÍNH nguồn quy định (`train`/`eval` của DESED, `fold1..5` của ESC-50). Split train/eval của DESED đã kiểm chứng là **tách nguồn hoàn toàn** nên giữ được làm ranh giới |
| `duration`, `sample_rate_orig`, `channels` | Thuộc tính kỹ thuật |
| `license` | Điều khoản license của **chính file này** |
| `attribution` | Tác giả, để ghi công |
| `redistributable` | `yes` \| `no` \| `labels_only` |
| `download_date`, `checksum_sha256` | Truy vết |

`exclusions.csv`: `file_id`, `stage`, `reason_code`, `detail`, `decided_by`, `decided_at`

**Mã lý do loại bỏ:** `wrong_class` · `multiple_events` · `too_noisy` · `music_overlay` · `speech_overlay` · `clipped` · `too_short` · `too_long` · `upsampled` · `duplicate` · `license_blocked` · `silent`

---

## 4. Quy trình gán nhãn có máy hỗ trợ (loại A — foreground bank)

Đây là phần trả lời trực tiếp yêu cầu *"tự gán nhãn cho từng folder, class kết hợp với máy"*.

### 4.1 Giai đoạn 1 — Chuẩn hoá kỹ thuật (hoàn toàn tự động)

```
raw/ → kiểm tra chất lượng → chuẩn hoá → interim/normalized/
```

| Kiểm tra | Ngưỡng | Hành động nếu trượt |
|---|---|---|
| Sample rate gốc | ≥ 16 kHz | Loại (`upsampled`) — audio 8 kHz upsample đã mất tần cao, mà tần cao chính là dấu hiệu của `glass_breaking` |
| Clipping | `\|x\| ≥ 0.999` ở > 0.1% mẫu | Loại (`clipped`) |
| Im lặng | RMS < −50 dBFS toàn file | Loại (`silent`) |
| Thời lượng (foreground) | 0.2 s ≤ d ≤ 10 s | Loại (`too_short` / `too_long`) |

Chuẩn hoá: resample 16 kHz → mono → **EBU R128 −23 LUFS** (không peak-normalize, xem `SYSTEM.md` §3.9) → lưu gain đã áp vào manifest.

### 4.2 Giai đoạn 2 — Sàng lọc tự động

Chạy một bộ tagger AudioSet đã pretrain (PANNs CNN14) lên **mọi** clip, lấy xác suất mức clip cho các ontology ID đã ánh xạ.

```
p_target      = max xác suất của các ID thuộc lớp mục tiêu
p_confusable  = max xác suất của các ID thuộc confusable_with
```

| Điều kiện | Định tuyến | Ghi chú |
|---|---|---|
| `p_target ≥ 0.30` **và** `p_target > p_confusable` | → **Tự động nhận** | Vẫn chịu kiểm tra ngẫu nhiên ở §4.4 |
| `p_target < 0.05` | → **Tự động loại** (`wrong_class`) | |
| `p_confusable > p_target` | → 🔴 **Hàng đợi người, ưu tiên cao** | Chính xác là những ca quyết định FAR |
| còn lại | → Hàng đợi người, ưu tiên thường | |

Luật thứ ba là điểm quan trọng nhất: nó **tự động dồn đúng những ca khó nhất cho người**, thay vì bắt người duyệt tuần tự từ đầu đến cuối.

### 4.3 Giai đoạn 3 — Đề xuất biên + người duyệt theo folder

**Đề xuất biên tự động** cho mỗi clip vào hàng đợi:

```
Xác suất mức frame của lớp mục tiêu
  → lọc trung vị (median filter, cửa sổ ~5 frame)
  → ngưỡng tại 0.5 × max
  → lấy vùng liên tục dài nhất
  → nới ±50 ms
  → đề xuất [onset, offset]
```

**Giao diện duyệt** — Label Studio, gom **theo folder lớp**, mỗi mẻ 50 clip, có sẵn waveform + spectrogram + vùng đề xuất. Người chỉ làm một trong bốn thao tác:

| Phím | Hành động |
|---|---|
| `1` | ✓ Nhận, giữ biên đề xuất |
| `2` | ✂ Nhận, kéo sửa biên |
| `3` | ✗ Loại (chọn mã lý do) |
| `4` | ⇄ Chuyển sang lớp khác (chọn lớp) |

**Năng suất thực tế:** ~150 clip/giờ khi có đề xuất sẵn, so với 30–40 clip/giờ nếu gán mù. Đây là toàn bộ giá trị của "kết hợp với máy" — nó **không** làm thay việc phán xét, nó chỉ loại bỏ thao tác cơ học.

### 4.4 Giai đoạn 4 — Kiểm định chất lượng của chính bộ sàng lọc

Không có bước này thì tự động hoá là vô căn cứ.

1. Lấy ngẫu nhiên **10%** số clip đã *tự động nhận*, cho người duyệt lại như bình thường.
2. Tính **tỉ lệ lỗi của auto-accept**.
3. Nếu > 10% → **nâng ngưỡng `p_target` và chạy lại toàn bộ**. Nếu ≤ 10% → chấp nhận, **ghi con số này vào báo cáo**.

Câu phải viết được trong khoá luận: *"Foreground bank được sàng lọc tự động với ngưỡng τ = 0.30; kiểm định trên 10% mẫu ngẫu nhiên cho tỉ lệ lỗi X%."* Không có X thì không ai tin bank sạch.

---

## 5. Background bank (loại B)

Yêu cầu duy nhất nhưng nghiêm ngặt: **không được chứa sự kiện Nhóm A**. Một tiếng chuông lọt vào background sẽ được Scaper gán nhãn là "không có sự kiện" → model học sai một cách hệ thống.

**Quy trình:**
1. Thu / tải đoạn nền dài (30–120 s) theo 4 loại khu vực: `school`, `parking`, `residential`, `factory`.
2. Chạy tagger lên toàn bộ; nếu **bất kỳ** lớp Nhóm A nào vượt 0.15 ở bất kỳ frame nào → 🚩 cờ cần người nghe.
3. Người nghe các đoạn bị cờ: cắt bỏ phần nhiễm, hoặc loại cả đoạn.
4. Nhận → `banks/background/<area_type>/`.

Ngưỡng 0.15 cố ý đặt thấp — ở đây thà báo nhầm nhiều còn hơn lọt.

**Mục tiêu:** 2–4 giờ, ≥ 100 đoạn, phủ cả 4 khu vực, có cả ngày và đêm.

**Thu tại chỗ (IUH) — 2 buổi:**

| Buổi | Nội dung | Thời lượng |
|---|---|---|
| **1** | Background 4 khu vực (ngày + tối) · RIR hành lang/nhà xe/xưởng (vỗ tay hoặc sweep) | ~3 h |
| **2** | 🎯 **Kịch bản `media_playback` (slice S4)**: phát tiếng súng/hét/kính vỡ qua loa rồi thu lại ở nhiều khoảng cách · bổ sung `fireworks` nếu thiếu | ~3 h |

Buổi 2 tạo ra dữ liệu **không dataset công khai nào có** — nó chính là nền của slice S4, và S4 là một trong hai điểm nhấn của đề tài. Đừng cắt buổi này trước buổi 1.

⚠️ Thu ở nơi có người: phải có sự đồng ý, ghi vào `data_inventory.md`. Không thu lén hội thoại.

---

## 6. Khử trùng lặp & chống rò rỉ

FSD50K, ESC-50 đều lấy từ Freesound; AudioSet từ YouTube. **Trùng lặp giữa các dataset là có thật**, và một bản trùng nằm hai bên train/test là rò rỉ.

### Bằng chứng đo được trên ESC-50 (D1, 14/09/2026)

ESC-50 có sẵn cột `src_file` = bản ghi Freesound gốc, tức là `source_group_id` cho không. Đếm thử trên 8 lớp ta dùng:

| Lớp ESC-50 | → lớp của ta | Clip | Bản ghi gốc | Clip/bản ghi |
|---|---|---:|---:|---:|
| `fireworks` | `fireworks` | 40 | **16** | **2.5** |
| `siren` | `siren` | 40 | 28 | 1.4 |
| `clapping` | `applause_cheering` | 40 | 33 | 1.2 |
| `footsteps` | `running_footsteps` | 40 | 32 | 1.2 |
| 4 lớp còn lại | | 160 | 145 | 1.1 |
| **Tổng** | | **320** | **254** | **1.3** |

**`fireworks` là trường hợp tệ nhất, và đó chính xác là lớp nguy hiểm nhất để rò rỉ.** 40 clip chỉ đến từ 16 bản ghi — chia ngẫu nhiên theo file thì gần như chắc chắn có hai lát cắt của cùng một chuỗi pháo nằm ở hai bên train/test. Mà `fireworks` ↔ `gunshot` chính là cặp nhầm lẫn ta dựng cả taxonomy để đo. Rò rỉ ở đây sẽ thổi phồng đúng con số quan trọng nhất của khoá luận.

Kết luận: N2 không phải đề phòng lý thuyết. Dùng `src_file` làm `source_group_id` cho ESC-50, `fsID` cho UrbanSound8K, uploader cho FSD50K, YTID cho AudioSet.

⚠️ Cũng lưu ý: ESC-50 chỉ cho **40 clip/lớp**, quá ít so với mục tiêu (vd. `fireworks` cần 200). ESC-50 là nguồn bổ sung, không phải nguồn chính.

### Bằng chứng đo được trên DESED (D1, 14/09/2026)

Trùng lặp **bắc cầu** không phải chuyện lý thuyết — nó có sẵn trong DESED:

```
freesound_13613/{_0.._7}.wav   ≡   freesound_34853/{_0.._7}.wav
```

Tám cặp file **giống nhau từng byte**, nhưng mang **hai id Freesound khác nhau** (hai lần upload cùng một bản ghi). Chia tập theo `source_group_id` sẽ xếp chúng vào hai nhóm khác nhau và cùng một âm thanh nằm ở cả hai bên.

**Hệ quả bắt buộc:** `make_splits.py` phải **gộp các nhóm nguồn bắc cầu** trước khi chia, chứ không chỉ chia theo `source_group_id` nguyên bản.

Phần này đã tự động: `build_manifest.py` tính SHA-256 mọi file và ghi `dedup_groups.csv`, đánh dấu `risk=cross_group` cho nhóm bắc cầu. Đây là **khử trùng chính xác tuyệt đối và miễn phí**; phần dưới đây xử lý trùng *gần đúng*, tốn kém hơn nhiều.

**Quy trình (`scripts/dedup.py`) — trùng GẦN ĐÚNG:**
1. Trích embedding (PANNs/BEATs) cho mọi clip đã chuẩn hoá.
2. Tìm cặp có cosine ≥ 0.98 → nghi trùng.
3. Người xác nhận ngẫu nhiên 30 cặp để hiệu chuẩn ngưỡng.
4. Mỗi nhóm trùng giữ **một** đại diện (ưu tiên sample rate cao nhất, license thoáng nhất); các bản còn lại vào `exclusions.csv` với mã `duplicate`.
5. Ghi `dedup_groups.csv` để biết ai trùng ai.

**Chia tập (`scripts/make_splits.py`)** — theo `source_group_id`, **không** theo file:

```
Mọi source_group_id → gán vào đúng MỘT trong: {foreground_bank_train, dev, gold_test}
  ↳ foreground_bank_train : nguyên liệu Scaper sinh train
  ↳ dev                   : audio thật có strong label sẵn (AS-strong/DESED/MIVIA)
  ↳ gold_test             : audio thật, người gán mù
```

`scripts/check_leakage.py` chạy trong CI, kiểm tra ba điều:
- Không `source_group_id` nào xuất hiện ở hai tập
- Không clip foreground nào của train xuất hiện trong dev/test
- Không cặp trùng (theo `dedup_groups.csv`) nào bắc cầu giữa hai tập

---

## 7. Sinh dữ liệu huấn luyện bằng Scaper

Sau khi có bank sạch, đây là bước **sinh strong label miễn phí**.

**Mục tiêu:** train 20–25 h · dev-synthetic 3–5 h · clip 10 s.

**Tham số sinh (`ml/configs/scaper_train.yaml`):**

| Tham số | Phân phối |
|---|---|
| Số sự kiện/clip | 0–4 (0 sự kiện ~10% → dạy model biết im lặng) |
| SNR foreground/background | Uniform(−5, 25) dB |
| Pitch shift | Uniform(−1, 1) semitone |
| Time stretch | Uniform(0.9, 1.1) |
| Nền | Chọn ngẫu nhiên theo `area_type` |
| Tích chập RIR | 40% clip |

**Tỉ lệ slice ép buộc** (không để phân phối tự nhiên quyết định):

| Slice | Tỉ lệ | Cách ép |
|---|---|---|
| S1 `overlap` | 30% | Ép ≥2 sự kiện chồng ≥30% |
| S2 `low_snr` | 25% | Ép SNR ≤ 5 dB |
| S3 `reverb` | 40% | Tích chập RIR |
| S6 `long_event` | 10% | Foreground kéo dài > 8 s |
| S7 `causal_chain` | 15% | Kịch bản có thứ tự (xem dưới) |

**Kịch bản chuỗi nhân quả (S7)** — định nghĩa tường minh trong config, vì đây chính là thứ Temporal Aggregator và captioning sinh ra để xử lý:

```yaml
causal_chains:
  - name: break_in
    sequence: [glass_breaking, scream, running_footsteps]
    gaps_sec: [[0.5, 2.0], [0.3, 1.5]]
  - name: forced_entry
    sequence: [door_slam, shout_yell, running_footsteps]
  - name: emergency
    sequence: [alarm_bell, shout_yell, running_footsteps]
  - name: false_alarm_fireworks      # ★ chuỗi ÂM TÍNH, rất quan trọng
    sequence: [fireworks, applause_cheering]
  - name: false_alarm_dishes
    sequence: [object_drop_dishes, laughter]
```

Hai chuỗi âm tính cuối dạy model rằng *một chuỗi sự kiện liên tiếp không mặc nhiên là sự cố an ninh*. Thiếu chúng, model sẽ học "cứ có chuỗi là nguy hiểm".

**Đầu ra:** mỗi clip có `.wav` + `.jams` (Scaper ghi đầy đủ công thức) + dòng trong `.tsv`. File `.jams` cho phép **tái tạo chính xác** toàn bộ train set từ bank — đó là lý do có thể công bố "dataset" mà không cần công bố audio.

---

## 8. Gold test set (loại C) — gán mù, MỘT người gán

**Quy mô:** ~1 giờ audio thật · 100% gán mù · 20% gán lại lần hai để đo tự-nhất-quán.

**Nguồn:** ưu tiên audio thật chưa từng dùng ở bất kỳ đâu — thu tại IUH + phần AudioSet-strong/MIVIA được giữ riêng cho test ngay từ khi chia nhóm.

### 8.1 Ràng buộc: chỉ có một người gán nhãn

Quyết định ngày 14/09: dự án chỉ có **một** người gán nhãn.

**Hệ quả bắt buộc phải thừa nhận:** kappa liên-người (Cohen's kappa) là phép đo mức đồng thuận giữa **hai người độc lập** — không có người thứ hai thì không tính được, và không được phép gọi bất cứ con số nào khác là "kappa liên-người". Cái mất đi ở đây là thật: đồng thuận liên-người trả lời câu hỏi *"guideline có rõ ràng với người khác không?"*, và không có cách nào một người tự trả lời câu đó cho chính mình.

**Thay thế:** đo **tự-nhất-quán (intra-annotator / test–retest)** — cùng một người gán lại cùng một tập sau một khoảng nghỉ, trong điều kiện mù với lần gán đầu. Đây là phép đo chuẩn mực, có tên gọi riêng, và báo cáo được — nhưng nó chỉ trả lời *"guideline có ổn định không?"*, yếu hơn một bậc.

**Bắt buộc ghi vào phần Hạn chế của khoá luận** (`SYSTEM.md §11`), nguyên văn tinh thần: *gold test set do một người gán; độ tin cậy được đo bằng tự-nhất-quán test–retest chứ không phải đồng thuận liên-người; thiên lệch hệ thống của người gán không bị phát hiện bởi thiết kế này.* Giấu điều này đi thì hội đồng hỏi một câu là lộ.

### 8.2 Ba biện pháp bù

Không thay được người thứ hai, nhưng giảm được thiệt hại:

| # | Biện pháp | Bù cho điều gì |
|---|---|---|
| B1 | **Công bố `annotation_guideline.md` + gold TSV + `.jams`** để bên ngoài kiểm chứng được | Người khác vẫn audit được, chỉ là sau khi nộp |
| B2 | **Nhật ký bất đồng với chính mình**: mọi ca lần 1 ≠ lần 2 phải nghe lại lần ba, chốt, và **ghi lý do** | Biến điểm yếu thành dữ liệu: đây chính là danh sách ca mơ hồ, là vật liệu tốt nhất cho chương phân tích lỗi |
| B3 | **Báo cáo tự-nhất-quán THEO TỪNG LỚP**, không chỉ một con số gộp | Người đọc biết lớp nào lung lay. Dự đoán: `scream` ↔ `shout_yell` sẽ tệ nhất |

### 8.3 Quy trình

| Bước | Nội dung | Ai |
|---|---|---|
| 1 | **Pilot 30 clip** — gán lần 1 | 👤 |
| 2 | **Nghỉ ≥ 3 ngày**, gán lại 30 clip đó trong điều kiện mù (xem 8.4) | 👤 |
| 3 | Tính tự-nhất-quán + độ lệch onset → **sửa `annotation_guideline.md`** → ghi lại đã sửa gì | 👤 |
| 4 | Gán đại trà, **không có đề xuất của máy** (N1) | 👤 |
| 5 | **Nghỉ ≥ 7 ngày**, gán lại 20% ngẫu nhiên trong điều kiện mù | 👤 |
| 6 | Nghe lại lần ba mọi ca bất đồng; chốt; ghi lý do vào nhật ký B2 | 👤 |
| 7 | Gắn `slice_flags` cho từng clip | 👤 |
| 8 | `validate_annotations.py` + `agreement.py` + `slice_coverage.py` | 🤖 |

### 8.4 Điều kiện "mù" ở lần gán lại — không có cái này thì số đo vô nghĩa

Gán lại mà vẫn nhớ lần trước thì chỉ đang đo trí nhớ, không đo guideline. Bốn điều kiện bắt buộc:

1. **Dự án Label Studio mới hoàn toàn**, không import nhãn lần 1 dưới bất kỳ hình thức nào.
2. **Xáo thứ tự** và **đổi tên file thành mã băm** — thứ tự và tên file đều là manh mối gợi nhớ.
3. **Khoảng nghỉ tối thiểu** 3 ngày (pilot) / 7 ngày (đại trà).
4. Trộn lẫn clip lần hai vào giữa các clip khác để không tạo thành một khối dễ nhận ra.

### 8.5 Cổng chặn

| Chỉ số | Ngưỡng | Vì sao mức này |
|---|---|---|
| Tự-nhất-quán (event-based, collar 200 ms) | **≥ 0.75** | **Cao hơn** mức 0.70 của kappa liên-người: tự đồng ý với chính mình vốn dễ hơn đồng ý với người khác, nên giữ nguyên 0.70 là tự hạ chuẩn |
| Độ lệch onset trung vị giữa hai lần | **≤ 100 ms** | Bằng nửa collar 200 ms — lệch hơn thế thì chính collar đang che lỗi gán nhãn |
| Số event mỗi lớp | ≥ 20 | |
| Số clip mỗi slice | ≥ 20 | |

Không đạt → quay lại bước 3, **không** đi tiếp.

Bỏ pilot ở bước 1–3 là sai lầm tốn kém nhất trong toàn bộ kế hoạch: gán 1 giờ audio bằng guideline mơ hồ rồi phát hiện tự-nhất-quán 0.5 nghĩa là mất trắng 6 giờ công.

---

## 9. Giấy phép & khả năng công bố

| Nguồn | License | Có phát hành lại audio được? |
|---|---|---|
| AudioSet | Nhãn CC-BY; audio là YouTube | ❌ **Không** — chỉ script tái tạo |
| FSD50K | CC theo từng clip (có CC-BY-NC) | ⚠️ Tuỳ file → cột `license` quyết định |
| ESC-50 | CC BY-NC | ⚠️ Phi thương mại |
| UrbanSound8K | CC BY-NC + điều kiện riêng | ⚠️ Phải đọc kỹ điều khoản |
| DESED | Hỗn hợp | ⚠️ Theo từng phần |
| MIVIA | Theo thoả thuận đăng ký | ❌ Thường không |
| **Thu tại IUH** | **Của bạn** | ✅ Có, nếu có đồng thuận |

**Kết luận thực tế:** không thể công bố một bộ audio gộp. **Có thể** công bố — và nên công bố như một phần đóng góp:

- `taxonomy.md` + `annotation_guideline.md`
- Nhãn TSV của gold set + `slice_flags`
- **File `.jams` của Scaper** (công thức tái tạo toàn bộ train set)
- Script tải + tái tạo
- Audio thu tại IUH (đã có đồng thuận)
- G2/G3/G4

Đây đúng là mô hình DESED dùng, hoàn toàn phòng thủ được trước hội đồng.

---

## 10. Lịch thực hiện & ngân sách công người

### Lịch 10 ngày (khớp W2–W3 trong [PLAN.md](PLAN.md))

| Ngày | Việc | Ai | Có chặn gì không |
|---|---|---|---|
| **D0** | Nộp đơn MIVIA · viết `ontology_map.yaml` · **xác minh ID ontology từ file gốc** · khung script | 👥 | MIVIA chặn lớp 5 |
| **D1–D2** | Tải AudioSet-strong / FSD50K / ESC-50 / UrbanSound8K / DESED · ghi `raw_manifest.csv` đủ cột (đặc biệt `source_group_id`, `license`) | 🤖 | |
| **D2** | Chuẩn hoá + kiểm tra chất lượng (§4.1) · khử trùng lặp (§6) | 🤖 | |
| **D3** | Sàng lọc tự động (§4.2) · sinh đề xuất biên (§4.3) · dựng Label Studio | 🤖 | |
| **D3–D6** | 🔴 **Duyệt foreground bank theo folder** | 👤 | **Nút thắt** |
| **D4** | 🎙️ Buổi thu 1: background + RIR | 👤 | |
| **D5** | Sàng lọc background (§5) | 👥 | |
| **D6** | 🎙️ Buổi thu 2: `media_playback` (S4) + `fireworks` | 👤 | |
| **D6** | Kiểm định 10% auto-accept (§4.4) | 👤 | |
| **D7** | 🚦 **CỔNG QUYẾT ĐỊNH: phân tích thiếu hụt** (§11) | 👥 | |
| **D7** | Chia tập theo `source_group_id` · `check_leakage.py` | 🤖 | |
| **D7–D8** | Scaper sinh train 20–25h + dev | 🤖 | |
| **D8** | Pilot gold 30 clip (lần 1) → sửa guideline sau khi gán lại ở D10+ | 👤 | Khoảng nghỉ ≥3 ngày là bắt buộc |
| **D8–D10** | 🔴 **Gán gold test set (mù)** + double 20% + trọng tài | 👤 | **Nút thắt** |
| **D10** | Validate toàn bộ · DVC tag `data-v1.0` · viết `data_inventory.md` | 🤖 | |

### Ngân sách công người — con số quan trọng nhất của kế hoạch này

| Việc | Giờ |
|---|---|
| Duyệt foreground bank (hàng đợi ~1.200–1.500 clip @150/h) | **8–10** |
| Kiểm định 10% auto-accept | **1.5** |
| Sàng lọc background | **2** |
| Hai buổi thu tại chỗ | **6** |
| Pilot gold + sửa guideline | **3** |
| Gán gold mù (~1h audio @6×) | **6** |
| Double-annotate 20% + trọng tài | **2.5** |
| **Tiểu tổng — dữ liệu thô** | **≈ 29–31 h** |
| G2 caption reference (200–300 clip × 3–5 ref) | **6–8** |
| G3 câu hỏi RAG (60–100) | **4** |
| G4 kịch bản alert (40–60) | **2** |
| **TỔNG** | **≈ 41–45 h** |

Trong 10 ngày là **~4 giờ/ngày công người**, song song với việc AI viết code. Khả thi nhưng **không có dư địa**.

**Ba đòn bẩy nén nếu trễ** (dùng theo thứ tự):
1. ~~Tìm người gán thứ hai~~ — **đã loại bỏ** (quyết định 14/09: một người gán). Mất luôn đòn bẩy nén lớn nhất, nên hai đòn bẩy còn lại thành thiết yếu chứ không phải tuỳ chọn.
2. Giảm G2 từ 300 → 150 clip (vẫn đủ để đo captioning).
3. Nâng ngưỡng auto-accept cho các lớp **không** nằm trong `confusable_with` của nhau (ví dụ `siren`, `speech_normal`) → giảm hàng đợi người.

**Không được nén:** gán gold mù, pilot, gán lại mù 20% (test–retest). Khoảng nghỉ ≥3/≥7 ngày cũng không nén được — nó là điều kiện để số đo có nghĩa, nên phải xếp lịch pilot **sớm** để kịp quay lại gán lần hai.

---

## 11. Cổng quyết định D7 — phân tích thiếu hụt

Chạy `scripts/coverage_report.py`, sinh bảng:

| `class_id` | Đã có | Mục tiêu | Tối thiểu | Trạng thái |
|---|---|---|---|---|
| … | 187 | 200 | 80 | 🟢 |
| `vehicle_crash` | 41 | 80 | 80 | 🔴 |

Với mỗi lớp 🔴, chọn **một** trong bốn phương án và **ghi thành ADR**:

| Phương án | Khi nào dùng | Cái giá |
|---|---|---|
| **A. Bổ sung có chủ đích** | Còn nguồn chưa khai thác (Freesound trực tiếp, thu thêm) | Tốn thời gian |
| **B. Gộp lớp** | Hai lớp hiếm gần nhau về âm học (vd `explosion` → `gunshot` thành `impulsive_blast`) | Taxonomy còn 14 lớp — **phải sửa toàn bộ tài liệu** |
| **C. Giữ lớp, loại khỏi macro-F1** | Muốn hệ thống vẫn phát hiện được, nhưng thừa nhận không đủ dữ liệu đánh giá | Phải nêu rõ trong mọi bảng kết quả |
| **D. Bỏ lớp** | Không cứu được | Taxonomy còn 14 lớp |

Nguyên tắc: **quyết ở D7, không kéo dài**. Một lớp còn 🔴 vào D8 mà chưa quyết sẽ làm hỏng cả lịch W3.

Khuyến nghị mặc định cho `vehicle_crash` nếu MIVIA Road không về kịp: chọn **C** — giữ lớp, loại khỏi macro-F1, nêu rõ. Đó là cách trung thực nhất và không phá taxonomy.

---

## 12. Định nghĩa "xong" — điều kiện đóng băng `data-v1.0`

Chỉ tag DVC khi **toàn bộ** mục dưới đây đạt. Đây là ý nghĩa cụ thể của *"bắt đầu dự án từ data chuẩn nhất"*.

### Tính đầy đủ
- [ ] 16/16 lớp có trạng thái 🟢 hoặc có ADR quyết định rõ ràng
- [ ] Foreground bank ≥ `min_acceptable` cho mọi lớp còn giữ
- [ ] Background ≥ 2 h, phủ đủ 4 `area_type`, có cả ngày và đêm
- [ ] RIR ≥ 10 bản, ≥ 3 loại không gian
- [ ] Train synthetic ≥ 20 h · dev ≥ 3 h
- [ ] Gold test ≈ 1 h; mỗi class ≥ 20 event; mỗi slice ≥ 20 clip

### Tính đúng đắn
- [ ] `check_leakage.py` xanh (3 kiểm tra ở §6)
- [ ] `validate_annotations.py` xanh (offset > onset, không vượt duration, nhãn ∈ taxonomy)
- [ ] `validate_taxonomy.py` xanh
- [ ] Kappa ≥ 0.70, **đã ghi số**
- [ ] Tỉ lệ lỗi auto-accept ≤ 10%, **đã ghi số**
- [ ] Gold set **không** có clip synthetic nào
- [ ] Gold set gán mù, **không** dùng đề xuất của máy

### Tính truy vết
- [ ] Mọi file trong `raw_manifest.csv` có đủ `source_group_id`, `license`, `checksum`
- [ ] Mọi file bị loại có mặt trong `exclusions.csv` kèm mã lý do
- [ ] `dedup_groups.csv` đầy đủ
- [ ] `.jams` của Scaper được giữ (tái tạo được train set)
- [ ] `docs/data_inventory.md` sinh tự động, khớp manifest

### Đóng băng
- [ ] `dvc add` toàn bộ `data/` + `dvc push`
- [ ] `git tag data-v1.0`
- [ ] ADR ghi lại mọi quyết định ở cổng D7
- [ ] **Gold test set niêm phong** — không được xem, không được tuning trên nó cho tới khi đánh giá cuối

---

## 13. Danh sách script cần viết

| Script | Nhiệm vụ | Khi nào |
|---|---|---|
| `scripts/verify_ontology.py` | Đối chiếu `ontology_map.yaml` với file ontology gốc AudioSet | D0 |
| `scripts/download_sources.py` | Tải từng nguồn, ghi `raw_manifest.csv` | D1 |
| `scripts/normalize_audio.py` | Kiểm tra chất lượng + chuẩn hoá 16k/mono/R128 | D2 |
| `scripts/dedup.py` | Trùng lặp bằng embedding | D2 |
| `scripts/auto_screen.py` | Sàng lọc + định tuyến hàng đợi (§4.2) | D3 |
| `scripts/propose_boundaries.py` | Đề xuất onset/offset (§4.3) | D3 |
| `scripts/export_labelstudio.py` / `import_labelstudio.py` | Cầu nối với Label Studio | D3 |
| `scripts/screen_background.py` | Phát hiện sự kiện lọt vào nền (§5) | D5 |
| `scripts/qa_sample.py` | Lấy mẫu 10% kiểm định auto-accept | D6 |
| `scripts/coverage_report.py` | Bảng thiếu hụt cho cổng D7 | D7 |
| `scripts/make_splits.py` | Chia tập theo `source_group_id` | D7 |
| `scripts/check_leakage.py` | 3 kiểm tra rò rỉ (CI) | D7 + CI |
| `scripts/scaper_generate.py` | Sinh soundscape + slice + chuỗi nhân quả | D7 |
| `scripts/validate_annotations.py` | Kiểm tra tính hợp lệ nhãn (CI) | D10 + CI |
| `scripts/agreement.py` | Kappa + onset MAE | D10 |
| `scripts/slice_coverage.py` | Đếm clip theo slice | D10 |
| `scripts/data_inventory.py` | Sinh `docs/data_inventory.md` | D10 |

---

*Cập nhật lần cuối: 2026-09-14 · Phiên bản 0.1*
