# Quét ngưỡng trên dev tổng hợp — 2026-09-19

Sinh bởi `ml/evaluation/threshold_sweep.py` từ dự đoán mức đoạn đã lưu
(`ml/runs/{run}/predictions/dev_all.npz`, Pha 3 của
[TRAINING_OPS_PLAN](../TRAINING_OPS_PLAN.md)). Không train lại, không cần GPU cho khâu chấm.

**Giao thức chung:** cả ba run được chấm trên **cùng một tập** — `data/synthetic/dev`,
**1.440 clip / 4.873 sự kiện tham chiếu** — với cùng bộ lọc trung vị cố định 7 khung và
cùng mã chấm điểm. Không run nào trong ba được huấn luyện trên tập này.

---

## 1. Phát hiện chính: ngưỡng 0,5 sai nghiêm trọng ở cả ba run

| run | event-F1 @ θ=0,50 | θ* (tối ưu event-F1) | event-F1 @ θ* | lần tăng |
|---|---:|---:|---:|---:|
| panns_ft_v1 | 0,1147 | 0,96 | **0,2981** | 2,60× |
| panns_ft_v2 | 0,1123 | 0,89 | **0,3531** | 3,14× |
| panns_ft_v3 | 0,0820 | 0,91 | **0,3986** | **4,86×** |

Ở θ=0,50, v3 dự báo **65.884** sự kiện cho 4.873 sự kiện thật — thừa **13,5 lần**. Ở θ*
con số này còn **6.638**. Toàn bộ chênh lệch F1 trong bảng là chênh lệch của *một tham số
hậu xử lý*, không phải của trọng số model: cùng một checkpoint, cùng một tập.

Cả ba đỉnh đều **nằm trong** lưới quét (v1 đạt 0,2981 ở 0,96 rồi tụt xuống 0,2948 ở 0,97;
v3 đạt 0,3986 ở 0,91 rồi tụt), nên đây là cực đại thật chứ không phải mép lưới.

---

## 2. So ba run trên cùng giao thức

| chỉ số (dev, 1.440 clip) | v1 · pool 5 · legacy | v2 · pool 3 · legacy | v3 · pool 3 · lô B0–B9 |
|---|---:|---:|---:|
| mAP mức clip *(không phụ thuộc ngưỡng)* | 0,7464 | 0,7555 | **0,8132** |
| event-F1 @ θ=0,50 | 0,1147 | 0,1123 | 0,0820 |
| **event-F1 @ θ\*** | 0,2981 | 0,3531 | **0,3986** |
| **segment-F1 @ θ\*** | 0,6208 | 0,6564 | **0,7159** |
| segment-F1 tốt nhất (θ riêng) | 0,6465 (θ=0,91) | 0,6564 (θ=0,89) | **0,7159** (θ=0,91) |
| số đoạn / clip | 31 | 125 | 125 |

### Hai kết luận cũ bị bác bỏ

**(a) "v3 kém hơn v2 ở cả ba chỉ số".** Ghi ở STATUS §3 và CLAUDE §3 dựa trên bảng chấm
ở θ=0,50 cố định. Trên giao thức chung với ngưỡng chọn đúng, **thứ tự đảo lại**: v3 hơn
v2 ở cả mAP, segment-F1 lẫn event-F1. Ở θ=0,50 v3 trông *tệ nhất* chỉ vì nó lệch hiệu
chuẩn xa nhất khỏi 0,50, không phải vì nó học kém nhất.

**(b) "v2 hơn v1 +48 % tương đối ở event-F1" (0,1282 → 0,1897).** Con số đó đo trên tập
val tách từ chính lô legacy, mỗi run một tập. Trên dev chung ở θ=0,50, hai run **ngang
nhau** (0,1123 so với 0,1147); khoảng cách thật chỉ hiện ra khi mỗi run được chấm ở ngưỡng
riêng của nó (0,3531 so với 0,2981). Kết luận cũ không sai hoàn toàn nhưng được rút ra từ
một phép so không chung tập.

### Confound CHƯA gỡ được — không được bỏ qua khi viết báo cáo

`data/synthetic/dev` sinh bằng **cùng recipe B0–B9** với tập train của v3, chỉ khác seed.
v1/v2 train trên lô legacy có phân bố thời lượng khác hẳn (mọi sự kiện ≤ 1,10 s). Vì vậy
dev **thiên vị v3 theo thiết kế**: nó là bài kiểm tra cùng phân bố với dữ liệu học của v3
và khác phân bố với dữ liệu học của v1/v2.

Kết quả bảng trên **chứng minh** rằng v3 khớp phân bố đích hiện hành tốt hơn; nó **không
chứng minh** rằng kiến trúc hay dữ liệu mới tốt hơn nói chung. Chỉ real dev / gold_test
(937 WAV AudioSet-strong, chưa nhập manifest) mới tách được hai điều đó.

---

## 3. Giả thuyết về hiệu chuẩn — chưa đo, không được ghi là kết luận

Cả ba run đạt đỉnh ở θ ≈ 0,90–0,96, không phải quanh 0,5. Nghĩa là xác suất đầu ra bị
**thổi lên có hệ thống**. Một cơ chế khả dĩ nằm sẵn trong mã: `train_sed.positive_weights`
dùng `pos_weight` tới trần **30** trong `BCEWithLogitsLoss`, và pos_weight lớn đẩy toàn bộ
xác suất lớp dương lên cao.

Đây là **giả thuyết**, không phải chẩn đoán. Kiểm chứng cần: đường hiệu chuẩn (reliability
diagram) theo lớp, và một ablation pos_weight. Chưa làm cái nào thì chưa được viết vào
báo cáo như nguyên nhân.

---

## 4. Ghi chú phương pháp

- Ngưỡng chọn trên **dev**, theo CLAUDE.md §5. Dev tổng hợp dùng **chung foreground bank**
  với train nên không phải tập độc lập — nó đủ tư cách chọn siêu tham số, **không** đủ tư
  cách làm kết quả báo cáo.
- `panns_ft_v1` **không ghi `time_pool_blocks` ở bất kỳ đâu** (checkpoint không có trường
  này, `history.json` ghi `null`). Eval rơi về mặc định 5 kèm cảnh báo. Đo gián tiếp trên
  dev: pool 5 cho mAP **0,7464**, pool 3 cho **0,7244** — ủng hộ giả định "v1 = pool 5"
  nhưng **không chứng minh** nó; khoảng cách 0,022 không đủ lớn để loại trừ. Đây đúng là
  khoảng trống R1 mà Pha 1 sinh ra để xoá, và với v1 thì nó **không xoá hồi cứu được**.
- Đường cong đầy đủ (θ = 0,05 → 0,99, bước 0,01) ở
  `ml/runs/{run}/threshold_sweep_dev_all.json` và log `threshold_sweep_dev.log`.

---

## 5. Chi phí chấm điểm — một lỗi bậc hai đã sửa

`event_and_segment_f1` gọi `MetaDataContainer.filter(filename=…)` cho từng clip; hàm đó
quét **tuyến tính** toàn bộ danh sách sự kiện, nên phép chấm là O(số clip × số sự kiện).

Đo trên dev v3 ở θ=0,50 (1.440 clip / 65.884 sự kiện dự báo):

| khâu | trước | sau |
|---|---:|---:|
| `filter` (tra cứu) | **101,5 s** | — (gom một lượt) |
| `evaluate` (tính toán) | 4,7 s | 4,7 s |
| **tổng một lần chấm** | **102,3 s** | **5,2 s** |

98 % thời gian nằm ở khâu tra cứu, không ở khâu tính toán. Sau khi gom sự kiện theo clip
một lượt (`_nhom_theo_clip`), một lượt quét 19 ngưỡng giảm từ **32 phút xuống 2 phút** —
đó là khác biệt giữa "quét ngưỡng được" và "quét ngưỡng trên giấy". Điểm số **giống hệt**
trước và sau (segment-F1 0,2531 · event-F1 0,0820).

---

## 6. Lỗi thứ hai tìm ra cùng ngày: dataset bị pickle nguyên memmap

Phát hiện khi chạy kiểm chứng cho chính phần Pha 1 mới viết. Triệu chứng là một crash
**không đều**: `_pickle.UnpicklingError: pickle data was truncated`, xảy ra hai lần
(lượt lưu dự đoán của v1, và lượt smoke train), còn các lượt khác thì trót lọt.

`PrecomputedSedDataset` giữ `np.memmap` làm thuộc tính. `DataLoader(num_workers>0)` trên
Windows dùng spawn nên **pickle cả dataset** rồi đẩy qua pipe cho từng worker — và
`np.memmap` không pickle theo kiểu "một đường dẫn" mà pickle **toàn bộ nội dung mảng**.

| | pickle của dataset | file .npy tương ứng |
|---|---:|---:|
| dev, trước khi sửa | **921,8 MB** | 921,6 MB |
| dev, sau khi sửa | **0,159 MB** | 921,6 MB |
| train (suy ra) | ~5,07 GB × số worker | 5,07 GB |

Đúng thứ mà `mmap_mode="r"` được đưa vào để tránh, bị một dòng `self.waves = …` vô hiệu hoá.
**v1, v2 và v3 đều train ở `--workers 2`**, nên cả ba đã trả cái giá này ở mọi epoch: khi
may thì tốn thêm RAM và thời gian dựng worker, khi không may thì tiến trình chết.

Sửa: mở memmap **lười**, mỗi tiến trình một lần, và `__getstate__` loại nó khỏi bản pickle.
Dữ liệu đọc ra giống hệt trước và sau (đã kiểm từng mẫu). Có test hồi quy so kích thước
bản pickle với kích thước dữ liệu.
