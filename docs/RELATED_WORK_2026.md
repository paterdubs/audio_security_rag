# Đối chiếu thiết kế với nghiên cứu liên quan

> Cập nhật **18/09/2026**. Phân biệt kết quả của bài báo, suy luận thiết kế và
> implementation của dự án. Snapshot vận hành: [STATUS.md](STATUS.md).
> Không dùng tài liệu này để tuyên bố “SOTA hiện tại” hoặc tự động đổi hướng đề tài.

## 1. Các điểm đã kiểm tra lại từ nguồn gốc

| Công trình | Nội dung được nguồn hỗ trợ | Ý nghĩa cho đề tài |
|---|---|---|
| Fine-tune the pretrained ATST model for SED (ICASSP 2024) | Nghiên cứu fine-tune ATST-Frame; báo cáo PSDS1/PSDS2 0.587/0.812 trên benchmark DCASE Task 4 của bài | Có cơ sở thử fine-tune/encoder frame-level; không chứng minh BEATs đóng băng luôn kém trên dữ liệu an ninh |
| BRACE (arXiv 2025) | **Benchmark** đánh giá metric chất lượng caption và khả năng căn chỉnh audio–text, gồm Main và Hallucination | Không gọi BRACE là một “reference-free metric”; cần đối chiếu khi định vị C2 |
| Audio-Aware Decoding (2025) | Contrastive decoding so logits **có và không có audio context** để giảm object hallucination của LALM | Tiền lệ can thiệp lúc giải mã; không đồng nhất cơ chế này với hard gating bằng SED head |
| CAF-Score (arXiv 2026) | **Metric reference-free**, kết hợp CLAP và suy luận LALM; được đánh giá trên BRACE | Đối chứng cho đánh giá audio–caption, không thay thế phép đo event/onset/offset theo gold |

Nguồn trực tiếp:
[ATST SED](https://arxiv.org/abs/2309.08153),
[BRACE](https://arxiv.org/abs/2512.10403),
[Audio-Aware Decoding](https://arxiv.org/abs/2506.07233),
[CAF-Score](https://arxiv.org/abs/2603.19615).

Các số ATST là **kết quả công bố trong điều kiện của bài**, không phải xác nhận thứ hạng
tháng 09/2026. PSDS của DCASE không so trực tiếp với mAP/F1 trên synthetic security.
Tên hội nghị/version chưa được kiểm tra ngoài trang nguồn thì không tự suy thành đã nghiệm thu xuất bản.

## 2. Định vị đóng góp — còn là giả thuyết nghiên cứu

C1 của dự án dự kiến dùng SED head có giám sát làm bằng chứng tường minh cho decoder nhỏ,
kết hợp multi-task và grounding. AAD đã chứng minh có tiền lệ giảm hallucination bằng
can thiệp lúc decoding; muốn khẳng định khác biệt cần đọc toàn văn, so cơ chế và ablation,
không viết “chưa ai làm” từ việc dùng tên khác.

C2 dự kiến đối chiếu sự kiện, biên và thứ tự với **strong label gold độc lập**.
BRACE là benchmark; CAF-Score là metric. Không gom BRACE, FENSE và CLAPScore thành một
loại “reference-free metric”. Chưa có đối chiếu toàn diện để tuyên bố EHR/EOR/GS/TOA là mới.
Chất lượng bộ trích event từ caption cũng phải được kiểm định riêng.

EHR của template **không mặc định bằng 0**: caption có thể phản ánh một dự đoán SED sai.
Cần tách lỗi SED, lỗi sinh câu và lỗi bộ trích metric; không đánh giá bằng chính nhãn model dự đoán.

## 3. Hiện trạng thực nghiệm của dự án

- Serving: PANNs pretrained + caption template, chưa có BEATs–Conformer–BART hay LALM.
- Baseline SED có 15 đầu ra; taxonomy đầy đủ 16 lớp vì ambient_noise không có event head.
- V1/v2 hoàn tất trên snapshot legacy 6.568 clip; v2 đổi pooling + mixup hard-label +
  adaptive postprocessing. Các kỹ thuật này **đã có trong code v2**, không còn là việc
  “chưa làm”.
- V3 hoàn tất trên lô B0–B9 PASS hợp đồng: mAP 0,8123, segment-F1 0,2748, event-F1 0,1077.
  Lô mới có duration/phủ sóng/overlap khác legacy; cả ba run chưa có đánh giá gold.
- Không quy chênh lệch v1/v2 cho riêng độ phân giải, hoặc v2/v3 cho riêng chất lượng dữ liệu:
  chưa seed đầy đủ, chưa có lineage và chưa lưu predictions để quét threshold/hậu xử lý.
- Bước thời gian ~323 ms không tự chứng minh bất khả thi với collar 200 ms.
  Chất lượng định vị còn phụ thuộc nhãn, nội suy, threshold và hậu xử lý.
- Audio lưu 16 kHz, baseline PANNs resample 32 kHz. Upsample không khôi phục dải tần
  đã mất; chưa có ablation để gọi kết quả hiện tại là “mức sàn” định lượng của CNN14.

## 4. Ưu tiên có căn cứ trong phạm vi hiện tại

1. Sửa và kiểm định dữ liệu trước khi dùng kết quả để chọn kiến trúc.
2. Lưu seed/config/checksum/split/predictions và checkpoint có cấu hình pooling;
   đánh giá cùng split/metric khi so các model.
3. Sau đó mới thử từng yếu tố riêng: pooling, mixup, postprocessing, rồi encoder.
   Không mặc định mixup hoặc adaptive filtering chắc chắn cải thiện.
4. Giữ BEATs đóng băng như phương án tiết kiệm tính toán **dự kiến**, không gọi đó là tối ưu.
   Mở băng cần bỏ hoặc thay đường precompute feature cố định, không chỉ đổi requires_grad.
5. CIDEr/SPIDEr/FENSE và PSDS là kế hoạch đánh giá; chỉ thêm metric sau khi thống nhất
   giao thức và tham số. Cùng tên metric chưa đủ để so với bảng DCASE.
6. Một LALM zero-shot có thể là đối chứng sau này; cần kiểm tra VRAM, định dạng input,
   license và giao thức. Không khẳng định mọi model 7B đều không chạy trên 8 GB, hoặc
   LALM chắc chắn hallucinate nhiều hơn trên tiếng Việt khi chưa đo.

AudioSet-strong có annotation công khai; việc raw mới chưa nhập pipeline không biến nguồn
này thành “unlabelled”. Không sử dụng gold held-out cho semi-supervised training hoặc tuning.

## 5. Nguồn giữ lại để đọc tiếp — chưa đối soát toàn văn trong lượt này

Các liên kết dưới đây là danh sách khảo sát, **không phải bằng chứng đã đủ cho số liệu/ranking**.
Đặc biệt chưa xác nhận lại bảng X-ARES, ATST-SEDv2, chi phí VRAM hay tính mới của temporal metrics.

- [ATST-Frame](https://arxiv.org/abs/2306.04186) · [ATST-SED repo](https://github.com/Audio-WestlakeU/ATST-SED)
- [X-ARES](https://arxiv.org/abs/2505.16369): điểm tổng hợp nhiều downstream không phải điểm SED riêng của dự án.
- [DCASE 2024 Task 4](https://arxiv.org/abs/2406.08056)
- [SSL fusion và adaptive post-processing](https://arxiv.org/abs/2505.11889)
- [Frequency Dynamic Convolutions](https://arxiv.org/abs/2506.12785)
- [SpotSound](https://arxiv.org/abs/2604.13023)
- [Audio Flamingo 2](https://arxiv.org/abs/2503.03983)
- [Qwen2-Audio](https://arxiv.org/abs/2407.10759)
- [Audio Captioning Using Sound Event Detection](https://arxiv.org/abs/2110.01210)
- [Performance–Complexity Trade-Offs in SED](https://arxiv.org/abs/2503.11373)

Không suy khoảng tin cậy, thứ hạng model hay tính đủ của 1.500/400 clip nếu chưa có
thiết kế thí nghiệm và số đo. Cắt nhỏ dữ liệu để debug là quyết định vận hành,
không tự tạo ra một benchmark nghiên cứu hợp lệ.
