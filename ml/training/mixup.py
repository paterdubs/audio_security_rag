"""Mixup cho SED — trộn hai clip trong batch và trộn luôn nhãn của chúng.

Vì sao mixup hợp bài toán này hơn hẳn một dataset cân bằng: bank foreground lệch rất
nặng (`vehicle_crash` 33 clip, `shout_yell` 56, `explosion` 64 — so với `speech_normal`
982). Với 33 clip, model chỉ cần nhớ thuộc 33 mẫu đó là hạ được loss, và nó sẽ nhớ thật.
Mixup sinh ra tổ hợp mới ở mỗi bước nên không có mẫu nào để nhớ thuộc.

**Trộn ở miền SÓNG ÂM, không phải miền log-mel.** Cộng hai sóng âm là đúng thứ xảy ra khi
hai âm thanh phát cùng lúc; cộng hai log-mel thì không tương ứng với bất kỳ tín hiệu nào
(log là hàm phi tuyến, log(a)+log(b) ≠ log(a+b)). Đây là lý do mixup được đặt trước khi
model trích đặc trưng, chứ không dùng `do_mixup` sẵn có của PANNs vốn chạy trên phổ.
"""

from __future__ import annotations

import numpy as np
import torch

# Hai cách gán nhãn cho clip đã trộn. Khác biệt KHÔNG nhỏ, xem giải thích ở `mix_batch`.
LABEL_SOFT = "soft"
LABEL_HARD = "hard"


def mix_batch(waveform: torch.Tensor, frame_target: torch.Tensor, clip_target: torch.Tensor,
              alpha: float, label_mode: str = LABEL_SOFT,
              generator: torch.Generator | None = None):
    """Trộn batch với chính nó đã xáo thứ tự. Trả về (sóng âm, nhãn khung, nhãn clip).

    Hai chế độ nhãn:

    `soft` — nhãn = λ·y_a + (1−λ)·y_b. Đây là mixup nguyên bản; nhãn mềm chính là nguồn
        gốc tác dụng chính quy hoá.
    `hard` — nhãn = hợp của hai tập nhãn (max). Lập luận theo vật lý: trộn hai tín hiệu
        thì CẢ HAI sự kiện đều thật sự nghe được, nên người gán nhãn sẽ gán cả hai ở mức
        đầy đủ. Với lớp hiếm, cách này giữ nguyên cường độ tín hiệu dương thay vì làm
        nhạt nó đi theo λ — đó là lý do nó đáng thử trước ở đúng bài toán lệch lớp này.

    Chưa có bằng chứng nào trong đề tài nói cách nào hơn, nên để thành tham số chọn được
    và phải ĐO. Đừng chọn theo cảm tính.
    """
    if alpha <= 0:
        return waveform, frame_target, clip_target

    # Một λ cho cả batch (mixup nguyên bản), không phải mỗi mẫu một λ.
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(waveform.size(0), device=waveform.device, generator=generator)

    mixed_waveform = lam * waveform + (1.0 - lam) * waveform[permutation]
    if label_mode == LABEL_HARD:
        mixed_frame = torch.clamp(frame_target + frame_target[permutation], max=1.0)
        mixed_clip = torch.clamp(clip_target + clip_target[permutation], max=1.0)
    else:
        mixed_frame = lam * frame_target + (1.0 - lam) * frame_target[permutation]
        mixed_clip = lam * clip_target + (1.0 - lam) * clip_target[permutation]
    return mixed_waveform, mixed_frame, mixed_clip
