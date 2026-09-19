# Cửa sổ lọc suy từ train — đo độ lớn của phần rò rỉ — 2026-09-19

Tiếp theo [long_event_postproc_20260919.md](long_event_postproc_20260919.md) §4, trang đó
kết thúc ở câu *"cửa sổ theo lớp không phải cải tiến báo cáo được — nó suy từ nhãn của
chính tập đang chấm"*. Trang này đo thẳng độ lớn của phần rò rỉ đó, bằng cách thêm một
đường tính cửa sổ thứ hai — `ml/evaluation/threshold_sweep.py::cua_so_loc_tu_train()` —
đọc `data/features/train_meta.json` trực tiếp, không nhận tham số `du_doan` nào của tập
đang chấm.

> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với train của `v3` → dev **thiên vị
> v3 theo thiết kế**. θ\* vẫn chọn trên chính tập đang chấm ở cả hai cấu hình dưới đây —
> ablation này chỉ cô lập MỘT trong hai tầng rò rỉ đã nêu ở
> [error_analysis_20260919.md](error_analysis_20260919.md), không phải cả hai.

---

## 1. Ba cấu hình, đo trên `v2` và `v3` (bỏ `v1` — miễn nhiễm, xem
   [long_event_postproc_20260919.md](long_event_postproc_20260919.md) §3)

| | v2 @θ\* | v3 @θ\* |
|---|---|---|
| cửa sổ cố định 7 khung | 0,3518 (θ 0,90) | 0,3979 (θ 0,90) |
| cửa sổ theo lớp, suy từ **dev** (rò rỉ) | 0,3911 (θ 0,85) | 0,4347 (θ 0,85) |
| cửa sổ theo lớp, suy từ **train** (không rò rỉ) | **0,3911 (θ 0,85)** | **0,4347 (θ 0,85)** |

Sáu chữ số thập phân, không làm tròn: `v2` 0,391090 so với 0,391090; `v3` 0,434656 so với
0,434656. **Chênh lệch đo được là 0,000000 ở cả hai run.**

## 2. Vì sao chênh lệch gần như triệt tiêu — không phải trùng hợp, mà do trần

So trực tiếp mảng cửa sổ theo từng lớp (15 lớp, `v2` và `v3` cho cùng một cặp mảng vì
cùng suy từ cùng hai tập nhãn train/dev):

| lớp | suy từ dev | suy từ train |
|---|---|---|
| alarm_bell | 29 | 29 |
| applause_cheering | 51 | 51 |
| **door_slam** | **23** | **25** |
| explosion | 51 | 51 |
| fireworks | 51 | 51 |
| glass_breaking | 47 | 47 |
| **gunshot** | **23** | **21** |
| laughter | 51 | 51 |
| object_drop_dishes | 21 | 21 |
| running_footsteps | 17 | 17 |
| scream | 51 | 51 |
| shout_yell | 51 | 51 |
| siren | 51 | 51 |
| speech_normal | 51 | 51 |
| vehicle_crash | 51 | 51 |

**9/15 lớp đã chạm trần `MAX_MEDIAN_FRAMES = 51` ở CẢ HAI nguồn** — với các lớp đó, hai
đường tính luôn ra cùng một số bất kể độ dài thật của lớp trên train và dev khác nhau bao
nhiêu, vì cả hai đều bị kẹp ở trần trước khi tới lúc so sánh. Chỉ `door_slam` và `gunshot`
là còn khác nhau, và khác đúng **2 khung (~20 ms)** — không đủ để đổi quyết định biên ở
θ=0,85 trên bất kỳ sự kiện nào trong dev.

**Kết luận cho ablation này:** độ lớn rò rỉ đo được là **không đáng kể**, nhưng lý do là
cấu trúc bị trần chặn trước, không phải vì train và dev giống hệt nhau về phân bố độ dài
sự kiện. Nếu trần được nới (ablation `MAX_MEDIAN_FRAMES` đang chạy song song, xem
`max_median_ceiling_20260919.md` khi xong), nhiều lớp hơn sẽ thoát trần và phần rò rỉ có
thể không còn bằng không nữa — trang này **không chứng minh rò rỉ luôn bằng không**, chỉ
đo đúng một điểm trong không gian cấu hình.

---

## 3. Điều trang này KHÔNG chứng minh

- **θ\* vẫn chọn trên chính tập đang chấm** ở cả hai cấu hình theo lớp — đây là tầng rò rỉ
  còn lại, chưa đo. Số 0,3911/0,4347 vẫn là chặn trên lạc quan vì lý do đó, dù cửa sổ
  không còn rò rỉ.
- Kết luận "chênh lệch bằng không" **chỉ đúng khi trần 51 khung còn hiệu lực**. Đây là một
  điểm dữ liệu, không phải một định luật.
- `gold_test` vẫn rỗng — không có tập độc lập nào để đo cửa sổ suy từ train mà không dùng
  dev tổng hợp (thiên vị v3) làm nơi chấm điểm.
