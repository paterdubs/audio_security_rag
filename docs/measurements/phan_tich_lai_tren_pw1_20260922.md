# Phân tích lại `long_event` và cơ chế `low_snr` trên `pw1` — hai kết luận cũ có sống sót qua θ\* mới không? — 2026-09-22

Tiếp theo [pos_weight_ket_luan_20260922.md](pos_weight_ket_luan_20260922.md): `pw1` (pos_weight=1)
đưa θ\* từ 0,90 (`v3`) xuống 0,35. Hai chẩn đoán `long_event` và `low_snr` đều đo **trên `v3`
@θ\*=0,90** — một ngưỡng cắt ở sát rìa thang điểm. Câu hỏi trung tâm: bao nhiêu phần của hai
chẩn đoán đó là hệ quả thật của độ dài sự kiện, và bao nhiêu chỉ là hệ quả của việc buộc phải
cắt ở ngưỡng cực đoan?

Đo lại bằng `ml/evaluation/long_event_crosscut.py --run panns_ft_pw1` (10 s, không train lại,
không GPU) — script đã có sẵn, không cần code mới. Không có test mới; 710 test không đổi.

> ⚠️ `data/synthetic/dev` sinh cùng recipe B0–B9 với train → thiên vị theo thiết kế, ngang nhau
> giữa `pw1` và `v3`. θ\* của mỗi run chọn trên chính tập đang chấm → chặn trên lạc quan.
> Cỡ mẫu mỗi ô giao chỉ 45–52 clip (`causal_chain` chỉ 15) — đủ nói hướng, không đủ nói độ lớn.

---

## Phần A — `long_event`: tỉ lệ phân mảnh có đổi khi θ\* đổi không?

### A.1 Số

| đối chiếu | ô | tỉ lệ vỡ `v3` (θ\*=0,90) | tỉ lệ vỡ `pw1` (θ\*=0,35) | Δ |
|---|---|---:|---:|---:|
| low_snr | `long_event+low_snr` | 0,3497 | 0,3436 | −0,006 |
| low_snr | `long_event-low_snr` | 0,3056 | 0,2803 | −0,025 |
| low_snr | `khac+low_snr` | 0,1843 | 0,1703 | −0,014 |
| low_snr | `khac-low_snr` | 0,1691 | 0,1651 | −0,004 |
| reverb | `long_event+reverb` | 0,2906 | 0,2759 | −0,015 |
| reverb | `long_event-reverb` | 0,3343 | 0,3118 | −0,023 |
| reverb | `khac+reverb` | 0,1780 | 0,1727 | −0,005 |
| reverb | `khac-reverb` | 0,1708 | 0,1628 | −0,008 |
| overlap | `long_event+overlap` | 0,2912 | 0,2692 | −0,022 |
| overlap | `long_event-overlap` | 0,3316 | 0,3130 | −0,019 |
| overlap | `khac+overlap` | 0,1798 | 0,1791 | −0,001 |
| overlap | `khac-overlap` | 0,1702 | 0,1598 | −0,010 |
| causal_chain | `long_event+causal_chain` | 0,3382 | 0,2500 | −0,088 |
| causal_chain | `long_event-causal_chain` | 0,3157 | 0,3055 | −0,010 |
| causal_chain | `khac+causal_chain` | 0,1338 | 0,1300 | −0,004 |
| causal_chain | `khac-causal_chain` | 0,1824 | 0,1747 | −0,008 |

Chênh lệch `long_event` − `khac`, so hai run:

| đối chiếu | cột *có X*, `v3` | cột *có X*, `pw1` | cột *không X*, `v3` | cột *không X*, `pw1` |
|---|---:|---:|---:|---:|
| low_snr | 0,17 | 0,17 | 0,14 | 0,12 |
| reverb | 0,11 | 0,10 | 0,16 | 0,15 |
| overlap | 0,11 | 0,09 | 0,16 | 0,15 |
| causal_chain | 0,20 | 0,12 | 0,13 | 0,13 |

### A.2 Kết luận — kết luận cũ **vững**, không phải hiện tượng của θ\* cực đoan

Mọi tỉ lệ vỡ **giảm nhẹ và gần đều** khi chuyển sang `pw1` (đa số −0,004 đến −0,025, tức θ\*
hợp lý hơn giúp phân loại sạch hơn một chút ở **mọi** ô — cả `long_event` lẫn `khac`). Không có
ô nào của `long_event` sụp đổ về gần mức `khac`. Chênh lệch `long_event − khac` gần như giữ
nguyên biên độ (0,09–0,17 ở `pw1` so với 0,11–0,20 ở `v3`).

Ngoại lệ duy nhất đáng chú ý: `long_event+causal_chain` cột *có* tụt từ 0,20 xuống 0,12 — nhưng
ô này chỉ có **15 clip**, nằm trong biên cảnh báo cỡ mẫu nhỏ ngay từ đầu, không đủ để nói gì
chắc chắn.

**Kết luận của [long_event_crosscut_20260921.md](long_event_crosscut_20260921.md) đứng vững**:
độ dài sự kiện là biến khó độc lập, không phải hệ quả của việc `v3` phải cắt ở θ\*=0,90 cực
đoan. Không cần đính chính trang đó.

## Phần B — cơ chế `low_snr`: hướng đúng, nhưng biên độ bị thổi phồng bởi θ\* cực đoan

### B.1 Số — phân rã Δ`dung` theo nơi chảy vào, cả hai run

| nhánh | run | θ\* | Δ đúng | Δ biên | Δ thay thế | Δ thiếu | Δ thừa |
|---|---|---:|---:|---:|---:|---:|---:|
| `khac` | v3 | 0,90 | −0,1054 | +0,0366 | +0,0155 | **+0,0533** | +0,0071 |
| `khac` | pw1 | 0,35 | −0,1162 | +0,0413 | +0,0142 | **+0,0607** | +0,0140 |
| `long_event` | v3 | 0,90 | −0,1348 | **+0,0824** | +0,0387 | +0,0138 | +0,2515 |
| `long_event` | pw1 | 0,35 | −0,1092 | **+0,0549** | +0,0192 | +0,0351 | +0,1187 |

Tỉ trọng biên/thiếu trong phần `dung` mất đi (chia cho |Δ đúng|):

| nhánh | run | % vào biên | % vào thiếu |
|---|---|---:|---:|
| `khac` | v3 | 34,7% | 50,6% |
| `khac` | pw1 | 35,5% | 52,2% |
| `long_event` | v3 | 61,1% | 10,2% |
| `long_event` | pw1 | 50,3% | 32,1% |

### B.2 Kết luận — H1/H3 vẫn phân theo độ dài, nhưng khoảng cách ở nhánh `long_event` hẹp lại đáng kể

**Nhánh `khac` (sự kiện thường): tái hiện gần như nguyên vẹn.** Tỉ trọng `thieu` vẫn áp đảo
(50,6% → 52,2%), `biên` vẫn phụ (34,7% → 35,5%). H1-Deletion là cơ chế chính, không đổi khi
θ\* đổi. Không cần sửa gì ở đây.

**Nhánh `long_event`: hướng giữ nguyên, biên độ co lại rõ rệt.** `biên` vẫn là thành phần lớn
hơn `thiếu` ở cả hai run (50,3% vs 32,1% tại `pw1`) — H3-lệch biên vẫn là cơ chế chính. Nhưng
tỷ lệ đóng góp của `thiếu` **tăng gấp ba** (10,2% → 32,1%) khi θ\* hạ từ 0,90 xuống 0,35.

Đây là điểm cần đính chính: câu *"model vẫn nghe ra sự kiện... `thieu` gần như không đóng góp"*
ở [low_snr_co_che_20260921.md](low_snr_co_che_20260921.md) §3 đã **phóng đại** mức độ Deletion
biến mất trên `long_event`. Một phần của "thiếu gần như không đóng góp" là hệ quả của việc
`v3` phải cắt ở θ\*=0,90 sát trần — ở ngưỡng đó model gần như luôn *đã* vượt ngưỡng nếu nghe ra
gì đó, nên phần còn sót lại cho `thieu` bị ép nhỏ giả tạo. Ở `pw1` với θ\* hợp lý hơn, cả hai
cơ chế cùng có mặt trên sự kiện dài — lệch biên vẫn trội hơn, nhưng không áp đảo như từng ghi.

**Kết luận thực chất không đổi** (biên/thời gian là bài toán riêng của sự kiện dài, tách khỏi
Deletion của sự kiện thường) — nhưng câu diễn đạt "gần như không đóng góp" phải sửa thành
"vẫn là thành phần nhỏ hơn, không còn là không đáng kể".

### B.3 Cấu trúc ma trận nhầm — vẫn trong cùng họ lớp, không có cặp mới lạ

So `pw1` với `v3` cho bốn ô `low_snr`: các cặp đậm nhất vẫn nằm trong cùng cụm lớp đã biết
(`object_drop_dishes`/`glass_breaking`/`door_slam`, `fireworks`/`running_footsteps`,
`speech_normal`/`shout_yell`). Thứ hạng trong mỗi cụm xáo trộn (ví dụ `door_slam→
running_footsteps` và `speech_normal→shout_yell` nổi lên rõ hơn ở `pw1`), nhưng không xuất
hiện cặp nào ngoài các họ lớp cũ. Không có gì phải khai thêm vào `confusable_with` của
`ml/configs/ontology_map.yaml` — kết luận của §2 trang cũ đứng vững.

## Tổng kết

| chẩn đoán cũ | sống sót qua θ\* mới? |
|---|---|
| `long_event` là biến khó độc lập (không phải trùng lát cắt) | **Có, gần như nguyên vẹn.** |
| `low_snr` hỏng theo Deletion trên sự kiện thường | **Có, gần như nguyên vẹn.** |
| `low_snr` hỏng theo lệch biên trên sự kiện dài, Deletion gần như không đóng góp | **Hướng đúng, biên độ phóng đại** — `thiếu` đóng góp gấp ba khi θ\* hợp lý hơn. |
| `low_snr` không sinh cặp nhầm mới | **Có, gần như nguyên vẹn.** |

Việc phân tích lại này **không** làm đổi quyết định ở
[long_event_gap_20260921.md](long_event_gap_20260921.md) §4 — hướng ưu tiên tiếp theo cho
`long_event` vẫn là ablation `--time-pool-blocks` (trường tiếp nhận thời gian của CNN14), vì cả
hai chẩn đoán (phân mảnh và lệch biên) đều đứng vững qua phép kiểm tra hiệu chuẩn này.
