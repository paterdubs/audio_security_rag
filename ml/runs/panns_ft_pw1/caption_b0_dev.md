# B0 × hallucination — panns_ft_pw1 trên dev (tự nhất quán, không phải G2) — 2026-09-22

Caption B0 sinh từ timeline DỰ BÁO của chính model, so với nhãn tham chiếu strong-label. EHR > 0 ở đây nghĩa là model dự báo sai, KHÔNG phải lỗi lexicon (lexicon đã kiểm bất biến với mọi cụm B0 sinh ra — xem test_ml_hallucination.py).

`n_clip` = 1440.

| metric | trung bình | đo được | chưa đo |
|---|---:|---:|---:|
| EHR↓ | 0,2498 | 1309 | 131 |
| EOR↓ | 0,2352 | 1287 | 153 |
| GS↑ | 0,6134 | 1316 | 124 |
| CHR↓ | 0,2297 | 845 | 595 |
| TOA↑ | 0,9311 | 1034 | 406 |

