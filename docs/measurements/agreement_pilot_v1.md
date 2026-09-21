# Tự-nhất-quán pilot gold lần 1 ↔ lần 2 — DATA_PLAN §8.3 bước 3

KHÔNG phải kappa liên-người — một người gán lại chính mình sau khoảng nghỉ, trong điều kiện mù (DATA_PLAN §8.1). Collar onset 200 ms.

⚠️ 22 tên trong `--lan2` không khớp file_id nào trong mapping (mồi, hoặc gõ nhầm project): blind_138883ef3033460e.wav, blind_1ee54ee9ad8b1505.wav, blind_1fe3e9dbdedbaff7.wav, blind_32b71b12b32d5b36.wav, blind_3713dd8c054dd6f3.wav, blind_4b705dafbd20bfcd.wav, blind_4f58429397e45658.wav, blind_528813416c4bb356.wav, blind_568ac90631fcb137.wav, blind_57e7871021d700e0.wav, blind_625aca7051e42ac0.wav, blind_7a1269e46f8bf0e0.wav, blind_82d95fd2df84dacb.wav, blind_8be5452f42b23b89.wav, blind_9c6e4344dfb94518.wav, blind_9e8322c6cb27f76b.wav, blind_a37336b2454db55a.wav, blind_d030f8ec7bcb618a.wav, blind_d1435be4e56fb74b.wav, blind_d5e3a8b5e0e542ea.wav, blind_dede95efc7eca4c4.wav, blind_e0088e40b4ede801.wav

## Cổng §8.5 (áp cho batch pilot, KHÔNG áp cho gold_test đầy đủ)

- Tự-nhất-quán event-F1: **0,7339** (ngưỡng ≥ 0,75) → ❌ KHÔNG ĐẠT
- Lệch onset trung vị: **10,0 ms** (ngưỡng ≤ 100 ms) → ✅ ĐẠT

→ Không đạt cả hai cổng: quay lại bước 3 (DATA_PLAN §8.3), sửa `annotation_guideline.md`, KHÔNG đi tiếp bước 4.

Hai cổng còn lại của §8.5 (≥20 event/lớp, ≥20 clip/slice) áp cho `gold_test` đầy đủ sau khi gán đại trà, KHÔNG áp cho batch pilot này.

## Tự-nhất-quán event-F1 theo từng lớp (DATA_PLAN §8.2 B3)

| Lớp | Event-F1 |
|---|---|
| alarm_bell | 0,7273 |
| applause_cheering | 0,8333 |
| door_slam | 0,4000 |
| explosion | 0,6667 |
| fireworks | nan |
| glass_breaking | 0,6667 |
| gunshot | 0,7500 |
| laughter | 0,0000 |
| object_drop_dishes | 1,0000 |
| running_footsteps | nan |
| scream | 1,0000 |
| shout_yell | nan |
| siren | 0,8000 |
| speech_normal | 0,7111 |
| vehicle_crash | nan |

## Danh sách ca bất đồng — nghe lại lần ba (DATA_PLAN §8.3 bước 3 / §8.2 B2)

| clip_id | loại | lớp lần 1 | lớp lần 2 | onset lần 1 | onset lần 2 |
|---|---|---|---|---|---|
| as_strong_-0TTFAArJ9k_30000 | bien | speech_normal | speech_normal | 0,280 | 0,000 |
| as_strong_-3MV6T3u9Ig_30000 | thieu | speech_normal | ∅ | 7,950 | ∅ |
| as_strong_-99MQ06mPYE_30000 | bien | speech_normal | speech_normal | 3,500 | 3,970 |
| as_strong_-AIHL5EIKUg_30000 | thua | ∅ | speech_normal | ∅ | 0,000 |
| as_strong_-AIHL5EIKUg_30000 | thua | ∅ | door_slam | ∅ | 6,630 |
| as_strong_-AIHL5EIKUg_30000 | bien | door_slam | door_slam | 9,060 | 9,700 |
| as_strong_-FiHp5DFJmY_30000 | bien | laughter | laughter | 3,120 | 2,690 |
| as_strong_-FiHp5DFJmY_30000 | thieu | speech_normal | ∅ | 4,880 | ∅ |
| as_strong_-HIozxABvUQ_30000 | thua | ∅ | applause_cheering | ∅ | 1,170 |
| as_strong_-HIozxABvUQ_30000 | bien | speech_normal | speech_normal | 3,210 | 7,230 |
| as_strong_-glN59TmfME_30000 | bien | speech_normal | speech_normal | 2,470 | 0,000 |
| as_strong_-glN59TmfME_30000 | thieu | speech_normal | ∅ | 6,600 | ∅ |
| as_strong_-gqN3xq05ns_10000 | thua | ∅ | gunshot | ∅ | 2,340 |
| as_strong_-gqN3xq05ns_10000 | thua | ∅ | gunshot | ∅ | 4,240 |
| as_strong_03bZUQDFJU8_30000 | thieu | applause_cheering | ∅ | 8,720 | ∅ |
| as_strong_0BCXcbaLeoA_30000 | thua | ∅ | glass_breaking | ∅ | 9,700 |
| as_strong_0BEtPXdDQrs_250000 | thay_the | alarm_bell | siren | 0,000 | 0,000 |
| as_strong_0yZGysisqY0_12000 | thua | ∅ | alarm_bell | ∅ | 0,000 |
| as_strong_0yZGysisqY0_12000 | thua | ∅ | explosion | ∅ | 7,070 |
| as_strong_0yZGysisqY0_12000 | thua | ∅ | alarm_bell | ∅ | 8,130 |
| as_strong_13ApmSOcZpU_30000 | thay_the | speech_normal | siren | 3,400 | 3,460 |
