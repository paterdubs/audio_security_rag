# data_inventory.md — Tổng kết dữ liệu (sinh tự động)

> **KHÔNG sửa tay.** Chạy `python scripts/data_inventory.py` để sinh lại.

## Foreground bank

| Lớp | Nhóm | Clip | Phút | Mục tiêu | Tối thiểu | Trạng thái | Nguồn |
|---|---|---:|---:|---:|---:|---|---|
| `alarm_bell` | A | 487 | 14.0 | 250 | 100 | ✅ đủ | desed_soundbank, esc50, fsd50k |
| `door_slam` | A | 151 | 2.5 | 250 | 100 | 🟡 trên tối thiểu | fsd50k |
| `explosion` | A | 64 | 1.5 | 120 | 50 | 🟡 trên tối thiểu | fsd50k |
| `glass_breaking` | A | 322 | 6.4 | 250 | 100 | ✅ đủ | esc50, fsd50k |
| `gunshot` | A | 398 | 6.4 | 200 | 80 | ✅ đủ | fsd50k, urbansound8k |
| `running_footsteps` | A | 163 | 6.7 | 250 | 100 | 🟡 trên tối thiểu | esc50, fsd50k |
| `scream` | A | 131 | 2.5 | 200 | 80 | 🟡 trên tối thiểu | fsd50k |
| `shout_yell` | A | 56 | 4.6 | 200 | 80 | 🔴 dưới tối thiểu | fsd50k |
| `siren` | A | 311 | 18.0 | 300 | 120 | ✅ đủ | esc50, fsd50k, urbansound8k |
| `vehicle_crash` | A | 33 | 4.2 | 80 | 30 | 🟡 trên tối thiểu | vehicle_crash_cc |
| `ambient_noise` | B | 0 | 0.0 | 0 | 0 | — |  |
| `applause_cheering` | B | 162 | 15.1 | 250 | 100 | 🟡 trên tối thiểu | esc50, fsd50k |
| `fireworks` | B | 199 | 7.3 | 200 | 80 | 🟡 trên tối thiểu | esc50, fsd50k |
| `laughter` | B | 390 | 9.1 | 150 | 60 | ✅ đủ | esc50, fsd50k |
| `object_drop_dishes` | B | 418 | 10.4 | 250 | 100 | ✅ đủ | desed_soundbank, fsd50k |
| `speech_normal` | B | 982 | 29.7 | 300 | 120 | ✅ đủ | desed_soundbank, fsd50k |

**Tổng: 4267 clip / 138.4 phút.**

## Background bank

| Khu vực | Clip | Phút |
|---|---:|---:|
| factory | 364 | 22.7 |
| parking | 84 | 5.6 |
| residential | 426 | 28.4 |
| school | 1587 | 229.4 |

## RIR bank

| Loại phòng | Số RIR | RT60 trung vị (s) |
|---|---:|---:|
| large_room | 156 | 2.16 |
| medium_room | 198 | 0.73 |
| small_room | 151 | 0.26 |

## Train/dev tổng hợp (Scaper)

| Tập | Clip | Giờ | Clip im lặng | Lát cắt |
|---|---:|---:|---:|---|

## dev / gold_test — audio thật

| Tập | Clip | Phút | Số lớp có mặt |
|---|---:|---:|---:|
| dev | 0 | 0.0 | 0 |
| gold_test | 0 | 0.0 | 0 |
