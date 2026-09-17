"""PANNs CNN14 mức khung, đổi đầu ra sang 15 lớp sự kiện của đề tài.

Đây là baseline B0 của nhánh SED (PLAN.md W3 đặt kiến trúc chính là BEATs→Conformer).
Giữ nó KHÔNG phải để vứt đi: một baseline đã pretrain trên AudioSet là mốc bắt buộc để
biết BEATs+Conformer có thật sự hơn hay không — thiếu mốc này thì mọi con số của mô hình
chính đều không có gì để so.

⚠️ LỆCH TẦN SỐ LẤY MẪU, phải nói trước khi đọc kết quả: pipeline dữ liệu của đề tài chạy
16 kHz (chọn theo BEATs), còn CNN14 pretrain ở 32 kHz với fmax 14 kHz. Nâng mẫu 16→32 kHz
không tạo ra thông tin trong dải 8–14 kHz, nên dải đó TRỐNG so với lúc pretrain. Vì vậy
con số của baseline này là một mức SÀN bị thiệt, không phải năng lực thật của CNN14.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# Tham số mel PHẢI khớp checkpoint pretrain. Đổi bất kỳ số nào ở đây là làm hỏng ý nghĩa
# của trọng số đã tải: các conv block đầu học trên đúng cách chia mel này.
PANNS_MEL = {
    "sample_rate": 32000, "window_size": 1024, "hop_size": 320,
    "mel_bins": 64, "fmin": 50, "fmax": 14000,
}
PANNS_CLASSES = 527          # số lớp AudioSet của checkpoint gốc
EMBED_DIM = 2048             # đầu ra fc1 của CNN14


# Số block conv (trong 5 block đầu) còn gộp theo TRỤC THỜI GIAN.
#
# CNN14 gốc gộp thời gian ở cả 5 block → chia 32: clip 10 s ra đúng 31 đoạn, tức độ phân
# giải 323 ms. Đo thật 17/09/2026 và đây là con số quyết định:
#
#     độ phân giải model   323 ms
#     collar của DCASE     200 ms
#
# Model KHÔNG THỂ trúng collar khi bước lượng tử của nó còn thô hơn chính collar. Đó là
# trần cứng của F1 mức sự kiện, và không bộ hậu xử lý nào phá được — bộ lọc trung vị cửa
# sổ ≤ 63 khung còn là phép toán vô nghĩa trên tín hiệu hằng-từng-đoạn 32 khung (đã đo:
# lọc cửa sổ 7 và 51 cho ra mảng GIỐNG HỆT từng byte).
#
# Gộp tần số vẫn giữ nguyên ở mọi block; chỉ bỏ gộp thời gian ở các block cuối. Pooling
# không có tham số nên trọng số pretrain vẫn dùng được nguyên vẹn.
#
#     5 block → chia 32 → 323 ms   (CNN14 gốc)
#     3 block → chia  8 →  81 ms   ← lọt dưới collar 200 ms
#     2 block → chia  4 →  40 ms   (ngang ATST-Frame, nhưng tốn bộ nhớ)
DEFAULT_TIME_POOL_BLOCKS = 5


class PannsSed(nn.Module):
    """CNN14_DecisionLevelMax + head mới, trả về LOGIT thay vì xác suất.

    Vì sao phải viết lại forward thay vì gọi thẳng model gốc: forward của
    `panns_inference` bọc sigmoid ngay bên trong rồi mới trả kết quả. Huấn luyện bằng
    BCELoss trên xác suất dưới AMP fp16 là công thức gây NaN (log(0) khi sigmoid bão hoà);
    BCEWithLogitsLoss cần logit. Bản sao forward dưới đây được test đối chiếu với bản gốc
    (`tests/test_panns_sed.py`) nên nếu thư viện đổi, test sẽ bắt được chứ không trôi âm thầm.
    """

    def __init__(self, n_classes: int, checkpoint_path: Path | None = None,
                 freeze_backbone: bool = False,
                 time_pool_blocks: int = DEFAULT_TIME_POOL_BLOCKS):
        super().__init__()
        self.time_pool_blocks = time_pool_blocks
        # `backbone.interpolator` ghim cứng tỉ lệ 32 theo kiến trúc gốc. Bớt gộp thời gian
        # mà vẫn dùng nó thì số khung nội suy ra VƯỢT số khung mel và pad_framewise_output
        # nổ với kích thước âm. Dựng bộ nội suy riêng theo đúng mức gộp đang dùng.
        from panns_inference.models import Interpolator

        self.interpolator = Interpolator(2 ** time_pool_blocks)
        from panns_inference.models import Cnn14_DecisionLevelMax

        self.backbone = Cnn14_DecisionLevelMax(classes_num=PANNS_CLASSES, **PANNS_MEL)
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.backbone.load_state_dict(state["model"])

        # Head mới: 527 lớp AudioSet → 15 lớp của ta. KHÔNG tái dùng fc_audioset cũ.
        self.head = nn.Linear(EMBED_DIM, n_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def embed_frames(self, waveform: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Sóng âm (B, L) → đặc trưng mức đoạn (B, T, 2048) và số khung mel gốc.

        Sao chép nguyên logic của Cnn14_DecisionLevelMax.forward tới TRƯỚC fc_audioset.
        """
        backbone = self.backbone
        # Trích đặc trưng PHẢI chạy fp32, kể cả khi vòng huấn luyện bật AMP.
        #
        # Đo thật (17/09/2026): dưới autocast fp16, spectrogram vẫn hữu hạn nhưng log-mel
        # ra inf — biên độ nhỏ underflow về 0 trong fp16 rồi log(0) = -inf, và inf lan ra
        # toàn bộ mạng thành NaN. Triệu chứng lộ ra rất xa nguồn: sklearn báo
        # "Input contains NaN" lúc tính mAP, không ai nghĩ tới tầng mel.
        with torch.autocast("cuda", enabled=False):
            x = backbone.spectrogram_extractor(waveform.float())
            x = backbone.logmel_extractor(x)
        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = backbone.bn0(x)
        x = x.transpose(1, 3)

        # SpecAugment chỉ bật khi chính backbone đang train. Đóng băng backbone mà vẫn
        # augment thì đặc trưng nhiễu đi trong khi không có gradient nào học được từ nhiễu đó.
        if self.training and not self.freeze_backbone:
            x = backbone.spec_augmenter(x)

        blocks = (backbone.conv_block1, backbone.conv_block2, backbone.conv_block3,
                  backbone.conv_block4, backbone.conv_block5)
        for index, block in enumerate(blocks):
            # (thời gian, tần số). Bỏ gộp thời gian ở các block CUỐI chứ không phải block
            # đầu: block đầu chạy trên feature map to nhất nên giữ nguyên độ phân giải ở
            # đó tốn bộ nhớ nhất mà lợi ích thì như nhau.
            time_pool = 2 if index < self.time_pool_blocks else 1
            x = block(x, pool_size=(time_pool, 2), pool_type="avg")
            x = F.dropout(x, p=0.2, training=self.training)
        x = backbone.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)                                   # gộp trục mel
        x = F.max_pool1d(x, 3, stride=1, padding=1) + F.avg_pool1d(x, 3, stride=1, padding=1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(backbone.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return x, frames_num

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        from panns_inference.models import pad_framewise_output

        segment_embed, frames_num = self.embed_frames(waveform)
        segment_logits = self.head(segment_embed)                  # (B, T, C)

        # Mức clip lấy MAX theo thời gian — đúng tinh thần "DecisionLevelMax": một sự kiện
        # dài 0,3 giây trong clip 10 giây vẫn phải làm nhãn clip bật. Lấy trung bình sẽ
        # dìm chết mọi sự kiện ngắn, mà sự kiện an ninh phần lớn là ngắn.
        clip_logits = segment_logits.max(dim=1).values

        frame_logits = self.interpolator(segment_logits)
        frame_logits = pad_framewise_output(frame_logits, frames_num)
        return {"frame_logits": frame_logits, "clip_logits": clip_logits}


def trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
