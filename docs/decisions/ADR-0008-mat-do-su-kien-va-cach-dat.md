# ADR-0008 — Mật độ sự kiện (bộ E) và đặt sự kiện vào khoảng trống

**Trạng thái:** Đã chốt · 2026-09-17
**Cổng:** W2 — thiết kế lại dữ liệu tổng hợp (STATUS §7, bước B6)

## Bối cảnh

Sau B3–B5, phân bố thời lượng sự kiện đã bám sát miền đích, nhưng **độ phủ sóng**
— tỉ lệ thời lượng clip nằm trong hợp các khoảng sự kiện — vẫn chỉ 36,3 % so với
53,0 % của AudioSet `train_strong`.

Độ phủ sóng không phải một chỉ số trang trí: nó **chính là tỉ lệ frame dương**.
Lệch nó làm ngưỡng quyết định của model chệch có hệ thống, và đó là một trong ba
giả thuyết về việc segment-F1 thấp ở hai vòng train trước.

## Ba phép đo đã đổi cách hiểu vấn đề

**① Độ phủ sóng bão hoà theo số sự kiện.** Quét `count_weights` từ trung bình 1,90
lên 3,59 sự kiện/clip:

| trung bình | 1,90 | 2,26 | 2,62 | 2,90 | 3,24 | 3,59 |
|---|---:|---:|---:|---:|---:|---:|
| độ phủ sóng | 35,7 % | 38,3 % | 41,2 % | 43,8 % | 46,7 % | 47,7 % |
| clip có chồng lấn | 55,2 % | 60,6 % | 64,3 % | 68,2 % | 72,1 % | 74,0 % |

Gần gấp đôi số sự kiện chỉ nâng phủ sóng 12 điểm, trong khi chồng lấn phình 19
điểm. Thêm sự kiện là cách rất tệ để tăng phủ sóng.

**② Miền đích xếp sự kiện gần như NỐI TIẾP.** Chỉ 17,1 % thời lượng sự kiện của
AudioSet bị chồng lấn — hệ số nén (hợp / tổng) = **0,829**. Đặt ngẫu nhiên đều như
lô cũ cho 0,709–0,812 tuỳ mật độ, tức chồng lấn dày hơn thực tế đáng kể. Đây chính
là nguyên nhân của ①.

**③ Phân bố chồng lấn của miền đích là LƯỠNG CỰC.** Hệ số nén *từng clip* của
AudioSet có p50 = **1,000** và p75 = **1,000**: **62,7 % clip không chồng lấn chút
nào**, còn thiểu số có thì chồng rất nặng. Không phải "chồng lấn đều đều khắp nơi".
Một phần lớn là vì 26,2 % clip AudioSet chỉ có đúng một sự kiện.

## Quyết định 1 — Đặt sự kiện vào khoảng trống

Thay "rút vị trí ngẫu nhiên đều" bằng: tính các khoảng còn trống, đặt hẳn vào một
khoảng trống (rút đều **theo độ dài khe**, không đều theo số khe); hết chỗ mới chấp
nhận vị trí ít va chạm nhất trong 16 vị trí thử.

Không phải "tránh chồng lấn bằng mọi giá" — chồng lấn có thật ngoài đời, chỉ ít hơn
nhiều so với đặt ngẫu nhiên. Cặp bị ép của lát cắt `overlap` vẫn chồng đúng hợp đồng.

Kết quả đo: tỉ lệ clip không chồng lấn nhảy từ 10–16 % lên **33–51 %**, và **độ phủ
sóng cũng tăng**. Hiếm khi một thay đổi tốt cùng lúc cho hai chỉ số đối nghịch, nên
phần này không có đánh đổi để cân nhắc.

## Quyết định 2 — `count_weights` bộ E

```yaml
count_weights: {0: 0.10, 1: 0.10, 2: 0.14, 3: 0.20, 4: 0.20, 5: 0.14, 6: 0.08, 7: 0.04}
```

Không thể khớp đồng thời độ phủ sóng và mật độ chồng lấn. Biên hiệu quả đo được
(đã dùng cách đặt vào khoảng trống):

| bộ | sk/clip | phủ sóng | clip chồng | nén gộp | clip sạch |
|---|---:|---:|---:|---:|---:|
| A (cũ) | 2,16 | 37,7 % | 43,7 % | 0,858 | 51,2 % |
| C | 2,79 | 44,0 % | 49,0 % | 0,812 | 45,3 % |
| D | 3,04 | 47,8 % | 51,7 % | 0,807 | 42,3 % |
| **E (chọn)** | **3,35** | **50,0 %** | 52,8 % | 0,783 | 41,0 % |
| F | 3,69 | 54,1 % | 56,7 % | 0,754 | 36,7 % |
| **ĐÍCH** | 4,22 | **53,0 %** | **37,3 %** | **0,829** | **62,7 %** |

Đã thử cả việc chép nguyên phân bố số đếm của AudioSet (đuôi dài tới 35 sự kiện/clip,
cắt trần ở 6/8/10/12). Bản cắt trần 6 cho hệ số nén khớp đúng 0,831 nhưng độ phủ sóng
chỉ 39,2 %. Không thoát được đánh đổi.

**Lý do ưu tiên độ phủ sóng:**

1. Nó chính là tỉ lệ frame dương, và lệch 18 % → 53 % là hỏng hóc nặng nhất đang sửa.
2. Chồng lấn dư làm **train khó hơn test** — hướng lệch an toàn hơn chiều ngược lại.
3. "Clip có chồng lấn 52,8 % so với 37,3 %" nghe to nhưng là chỉ số **nhị phân** cực
   nhạy: 1 ms giao nhau cũng tính. Hệ số nén 0,783 so với 0,829 mới là thước đo lượng
   chồng lấn, và lệch đó nhỏ.

## Vì sao không khớp được cả hai

Sự kiện của ta dài hơn AudioSet (trung bình ~1,94 s so với ~1,51 s theo mix tự nhiên),
nên cùng số sự kiện thì chiếm nhiều thời gian hơn và va vào nhau nhiều hơn. Thêm nữa
AudioSet sạch phần lớn nhờ 26,2 % clip chỉ có một sự kiện, trong khi ta cần mật độ cao
hơn để đạt phủ sóng.

Đây là **giới hạn của thiết kế hiện tại**, phải ghi vào phần Hạn chế của khoá luận
chứ không trình bày như đã khớp.

## Kết quả cuối, mô phỏng 7.920 clip

| chỉ tiêu | lô cũ | bộ E | đích (mix đều) |
|---|---:|---:|---:|
| sự kiện/clip | 2,09 | 3,40 | 4,22 |
| độ phủ sóng | 18,0 % | **50,9 %** | 53,0 % |
| % sự kiện ≥ 2 s | 0,0 % | 26,2 % | 25,1 % |
| % ≥ 4 s | 0,0 % | **15,1 %** | 15,5 % |
| % ≥ 8 s | 0,0 % | 6,6 % | 9,5 % |
| `overlap` đạt ngưỡng | không ép | **100 %** | 100 % |
| `long_event` đạt ngưỡng | **0 %** | **100 %** | 100 % |
| sự kiện tràn mốc 10 s | 1.225 | **0** | 0 |

Mốc ≥ 8 s vẫn hụt (6,6 % so với 9,5 %) vì giới hạn bank ở `siren`/`scream`/`explosion`
— xem ADR-0007.

## Khoá bằng test

`tests/test_scaper_placement.py`:

- `test_khoang_trong_*` — bốn ca: khe còn lại, khe quá hẹp, gộp sự kiện chồng nhau, clip rỗng/kín
- `test_moc_trong_khoang_trong_rut_deu_theo_DO_DAI_khe` — rút đều theo số khe sẽ dồn sự kiện vào khe hẹp
- `test_dat_cum_dung_cho_trong_khi_con_cho` — ba sự kiện ngắn thì không được chồng lấn chút nào
- `test_dat_cum_giam_chong_lan_so_voi_dat_thuan_ngau_nhien` — hệ số nén phải nhích hẳn về phía 0,829
- `test_dat_cum_van_giu_bao_dam_chong_lan_khi_bi_ep` — B6 không được phá bảo đảm của B4
