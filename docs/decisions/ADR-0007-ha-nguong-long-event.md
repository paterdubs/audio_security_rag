# ADR-0007 — Hạ ngưỡng `long_event` từ 8 s xuống 4 s, giới hạn vào lớp cấp được

**Trạng thái:** Đã chốt · 2026-09-17
**Cổng:** W2 — thiết kế lại dữ liệu tổng hợp (STATUS §7, bước B5)

## Bối cảnh

`ml/configs/scaper_train.yaml` khai:

```yaml
long_event:   {ratio: 0.10, min_event_duration_sec: 8.0}
```

Log sinh lô 17/09 báo `long_event 10% · 791 clip`. Đếm lại trên chính 7.920 file
`.jams` đã sinh: **số sự kiện dài ≥ 8 giây là 0**.

Không phải lệch nhẹ. Lát cắt hỏng **100 %**, suốt hai vòng train, không một triệu
chứng nào. Nhãn lát cắt có, nội dung không có — mọi kết luận từng rút ra trên lát
này đều vô hiệu.

Nguyên nhân: log báo **kế hoạch**, không báo **kết quả**. `plan_clips` đánh dấu đủ
791 clip, nhưng khâu sinh đặt `event_duration=("const", 1.0)` cho mọi sự kiện, nên
không có đường nào để một sự kiện dài 8 giây ra đời.

## Vì sao 8 giây là bất khả thi, không chỉ là chưa làm

Ngay cả sau khi bỏ hằng số 1.0, ngưỡng 8 giây vẫn không cấp nổi. Một lớp chỉ dùng
được cho lát cắt này khi đủ **cả hai** điều kiện:

1. bank có ít nhất 8 clip với phần có tiếng ≥ ngưỡng (ít hơn thì mọi sự kiện dài
   của lớp dồn vào vài file, và model học thuộc file thay vì học lớp);
2. miền đích có ít nhất một sự kiện thật đạt ngưỡng.

Số lớp đủ cả hai, đo trên bank 4.267 clip và `train_strong`:

| ngưỡng | 2 s | 3 s | **4 s** | 5 s | 6 s | 8 s |
|---|---:|---:|---:|---:|---:|---:|
| số lớp đủ | 14 | 9 | **8** | 8 | 6 | **4** |

Ở 8 giây chỉ còn 4/15 lớp. Và ngưỡng đó cũng **sai so với thực tế**: trong miền
đích chỉ 9,5 % sự kiện vượt 8 giây (tính theo mix lớp đều), còn 15,5 % vượt 4 giây.

## Quyết định

**Hạ ngưỡng xuống 4,0 giây**, giữ nguyên `ratio: 0.10`.

**Danh sách lớp không ghi cứng trong config** mà tính từ bank lúc chạy bằng
`lop_cap_duoc_su_kien_dai()`. Ghi cứng thì nó sẽ âm thầm lệch khi bank đổi — đúng
kiểu hỏng hóc ADR này sinh ra để chặn. Ở ngưỡng 4 s, 8 lớp đủ điều kiện:

`alarm_bell` · `applause_cheering` · `fireworks` · `running_footsteps` ·
`shout_yell` · `siren` · `speech_normal` · `vehicle_crash`

Điều kiện 2 **không thừa**: `object_drop_dishes` có 22 clip bank ≥ 4 s nhưng AudioSet
không ghi nhận một tiếng rơi bát đĩa nào kéo dài 4 giây. Ép nó vào là dạy model một
thứ sẽ không bao giờ gặp, và tệ hơn là làm nó bỏ qua tiếng rơi bát đĩa thật.

## Không lặp, không nối clip ngắn thành dài

Đã cân nhắc và bỏ. Tạm chấp nhận được với `siren` (âm tuần hoàn) nhưng là **bịa dữ
liệu** với `gunshot`, và nhãn của một chuỗi hai tiếng súng nối nhau không còn là
"một sự kiện dài" nữa. Thà nói thẳng là bank không cấp nổi.

## Bảo đảm bằng thiết kế, không bằng thiện chí

`chon_nguon` có nhánh hạ thời lượng khi bank thiếu nguồn. Nếu nhánh đó với tới được
clip ngắn hơn ngưỡng thì hợp đồng lại hỏng đúng như cũ, chỉ khác là nhãn ghi 4 s thay
vì 8 s. Nên ứng viên được **lọc trước** bằng `ung_vien_su_kien_dai()`: mọi clip còn
lại đều có `eff ≥ ngưỡng`, nên `min(d_muốn, eff)` không thể tụt xuống dưới.

`rut_thoi_luong_dai` lấy mẫu từ **phần đuôi** của phân bố đích chứ không kẹp cứng
xuống đúng ngưỡng — kẹp cứng thì mọi sự kiện dài đều dài y hệt nhau, và đó lại là
một quy luật hình học của bộ sinh chứ không phải của âm thanh. Không có giá trị nào
đạt ngưỡng thì **nổ**, không lặng lẽ trả về giá trị ngắn nhất.

## Kết quả

Mô phỏng 7.920 clip:

| | lô cũ | sau B5 |
|---|---:|---:|
| clip mang nhãn `long_event` | 791 | 772 |
| **trong đó đạt ngưỡng** | **0 (0,0 %)** | **772 (100,0 %)** |

Ngoài ra **24,9 %** clip *không* mang nhãn `long_event` vẫn tự nhiên có sự kiện ≥ 4 s,
nhờ B3. Lát cắt ép giờ chủ yếu để **bảo đảm một tập con báo cáo được**, chứ không
còn là nguồn duy nhất của sự kiện dài.

## Khoá bằng test

`tests/test_scaper_duration.py`:

- `test_lop_cap_duoc_su_kien_dai_doi_HAI_dieu_kien`
- `test_lop_cap_duoc_su_kien_dai_loai_lop_dai_tren_giay_nhung_khong_co_that` — ca `object_drop_dishes`
- `test_rut_thoi_luong_dai_khong_co_gia_tri_nao_thi_NO_chu_khong_im_lang`
- `test_hop_dong_long_event_duoc_BAO_DAM_khong_the_tut_duoi_nguong`

`tests/test_scaper_placement.py`:

- `test_plan_clip_co_long_event_luon_con_it_nhat_mot_su_kien_tu_do` — chuỗi nhân quả không được ăn hết ngân sách sự kiện
