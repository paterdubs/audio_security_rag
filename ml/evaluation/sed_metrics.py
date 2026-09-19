"""Đo chất lượng SED — mAP mức clip, F1 mức đoạn, F1 mức sự kiện.

Ba mức đo ba thứ KHÁC NHAU và một bản báo cáo chỉ có một trong ba là báo cáo thiếu:

  mAP mức clip      — "có nghe ra lớp này trong clip không", bỏ qua hoàn toàn vị trí.
  F1 mức đoạn (1 s) — "có đúng khoảng thời gian không", rộng lượng với sai lệch biên.
  F1 mức sự kiện    — "có tách đúng TỪNG sự kiện không", phạt cả gộp lẫn tách nhầm.

F1 mức sự kiện gần như luôn thấp hơn nhiều so với hai cái kia. Đó là bình thường, không
phải dấu hiệu hỏng: nó đòi khớp onset trong 200 ms, và với chuỗi tiếng súng liên tiếp thì
gộp hai phát thành một là đủ để mất điểm cả hai.

Dùng `sed_eval` (bản tham chiếu của DCASE) cho hai F1 thay vì tự cài: định nghĩa collar và
cách khớp sự kiện có nhiều biến thể, tự cài rất dễ ra một con số đẹp hơn thực tế mà không
ai phát hiện được khi đọc báo cáo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score

# Tham số giải mã khung → sự kiện. Đây là GIÁ TRỊ MẶC ĐỊNH, cần chỉnh theo dev set;
# ngưỡng 0.5 hiếm khi tối ưu cho lớp hiếm.
DEFAULT_THRESHOLD = 0.5
DEFAULT_MEDIAN_FILTER_FRAMES = 7     # ~70 ms ở 100 Hz — lấp khe hở nhỏ, bỏ gai đơn lẻ

# Collar chuẩn DCASE cho F1 mức sự kiện.
ONSET_COLLAR_SEC = 0.200
OFFSET_COLLAR_RATIO = 0.200


# Chặn trên/dưới cho cửa sổ lọc thích ứng.
MIN_MEDIAN_FRAMES = 3      # dưới 3 khung thì không lọc được gai đơn lẻ nào
MAX_MEDIAN_FRAMES = 51     # ~0,5 s; rộng hơn nữa bắt đầu gộp hai sự kiện rời thành một

# Phân vị độ dài dùng để chọn cửa sổ. CỐ Ý thấp, không phải trung vị: lấy trung vị nghĩa
# là một nửa số sự kiện của lớp đó ngắn hơn cửa sổ và sẽ bị chính bộ lọc xoá mất.
DURATION_PERCENTILE = 25


def median_filter(probabilities: np.ndarray, size) -> np.ndarray:
    """Lọc trung vị dọc trục thời gian. `size` là một số, hoặc một cửa sổ cho MỖI lớp.

    Không lọc thì một khung nhiễu lẻ cũng thành một "sự kiện" dài 10 ms, và F1 mức sự
    kiện tụt thảm hại vì hàng trăm dương tính giả tí hon.
    """
    from scipy.ndimage import median_filter as _median_filter

    if np.isscalar(size):
        if size <= 1:
            return probabilities
        return _median_filter(probabilities, size=(int(size), 1), mode="nearest")

    # Mỗi lớp một cửa sổ riêng — phải lọc từng cột, scipy không nhận kernel khác nhau
    # theo trục thứ hai trong một lần gọi.
    smoothed = probabilities.copy()
    for class_idx, window in enumerate(size):
        window = int(window)
        if window > 1:
            smoothed[:, class_idx] = _median_filter(
                probabilities[:, class_idx], size=window, mode="nearest"
            )
    return smoothed


def adaptive_median_sizes(durations_sec, frames_per_second: float,
                          min_frames: int = MIN_MEDIAN_FRAMES,
                          max_frames: int = MAX_MEDIAN_FRAMES) -> np.ndarray:
    """Cửa sổ lọc theo TỪNG LỚP, suy ra từ độ dài sự kiện thật của lớp đó.

    Vì sao một cửa sổ chung cho mọi lớp là sai: các lớp của đề tài dài ngắn khác nhau cả
    bậc. Tiếng súng ~80 ms (≈8 khung ở 100 Hz), tiếng còi hú kéo dài nhiều giây (hàng
    trăm khung). Một cửa sổ 7 khung thì quá RỘNG với tiếng súng (bộ lọc trung vị xoá sạch
    mọi đoạn ngắn hơn nửa cửa sổ, tức xoá luôn sự kiện) và quá HẸP với tiếng còi (không
    lấp nổi các khe hở giữa chừng, một sự kiện bị vỡ thành chục mảnh).

    Công thức: cửa sổ ≈ độ dài sự kiện ở phân vị thấp của lớp. Bộ lọc trung vị cửa sổ L
    giữ được đoạn dài hơn L/2, nên đặt L ≈ D cho biên an toàn gấp đôi.
    """
    frames = np.asarray(durations_sec, dtype=float) * frames_per_second
    # Lớp không có dữ liệu độ dài (không xuất hiện trong tập tính toán) → rơi về mặc định.
    frames = np.where(np.isfinite(frames) & (frames > 0), frames, DEFAULT_MEDIAN_FILTER_FRAMES)
    sizes = np.clip(np.round(frames), min_frames, max_frames).astype(int)
    # Ép lẻ: cửa sổ chẵn làm kết quả lệch nửa khung về một phía, và lệch pha giữa các lớp
    # là thứ không ai nhìn ra khi đọc điểm số.
    return np.where(sizes % 2 == 0, sizes + 1, sizes)


def class_event_durations(events_per_clip, n_classes: int,
                          percentile: float = DURATION_PERCENTILE) -> np.ndarray:
    """Độ dài sự kiện (giây) ở phân vị `percentile` cho từng lớp.

    `events_per_clip`: chuỗi các danh sách (class_idx, onset, offset).
    Lớp không xuất hiện lần nào trả về NaN — bên gọi quyết định giá trị thay thế.
    """
    collected: list[list[float]] = [[] for _ in range(n_classes)]
    for events in events_per_clip:
        for class_idx, onset, offset in events:
            collected[int(class_idx)].append(float(offset) - float(onset))
    return np.array([
        np.percentile(values, percentile) if values else np.nan for values in collected
    ])


def frames_to_events(probabilities: np.ndarray, class_ids: list[str], duration: float,
                     threshold: float = DEFAULT_THRESHOLD,
                     median_size=DEFAULT_MEDIAN_FILTER_FRAMES) -> list[dict]:
    """(n_frames, n_classes) xác suất → danh sách sự kiện {event_label, onset, offset}.

    `median_size` nhận một số (cùng cửa sổ cho mọi lớp) hoặc một mảng dài `n_classes`
    (cửa sổ thích ứng theo lớp — xem `adaptive_median_sizes`).
    """
    smoothed = median_filter(probabilities, median_size)
    active = smoothed >= threshold
    n_frames = active.shape[0]
    seconds_per_frame = duration / n_frames

    events: list[dict] = []
    for class_idx, class_id in enumerate(class_ids):
        column = active[:, class_idx]
        if not column.any():
            continue
        # Tìm các đoạn liên tục bằng sai phân trên mảng đã đệm hai đầu — tránh vòng lặp
        # Python trên hàng nghìn khung × 15 lớp.
        padded = np.concatenate(([False], column, [False]))
        edges = np.diff(padded.astype(np.int8))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            events.append({
                "event_label": class_id,
                "onset": float(start * seconds_per_frame),
                "offset": float(end * seconds_per_frame),
            })
    return events


@dataclass
class SedScores:
    clip_map: float
    clip_ap_per_class: dict[str, float]
    segment_f1: float
    event_f1: float
    event_f1_per_class: dict[str, float]

    def summary(self) -> str:
        return (f"mAP(clip)={self.clip_map:.4f}  "
                f"F1(đoạn 1s)={self.segment_f1:.4f}  F1(sự kiện)={self.event_f1:.4f}")


def clip_level_map(scores: np.ndarray, targets: np.ndarray,
                   class_ids: list[str]) -> tuple[float, dict[str, float]]:
    """mAP macro + AP từng lớp.

    Lớp không xuất hiện lần nào trong tập đánh giá bị BỎ QUA thay vì tính là 0: AP của một
    lớp vắng mặt không định nghĩa được, tính 0 sẽ kéo mAP xuống theo cách phản ánh độ hiếm
    của lớp chứ không phản ánh năng lực model.
    """
    per_class: dict[str, float] = {}
    for class_idx, class_id in enumerate(class_ids):
        column = targets[:, class_idx]
        if column.sum() == 0:
            continue
        per_class[class_id] = float(average_precision_score(column, scores[:, class_idx]))
    macro = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return macro, per_class


def _to_sed_eval_list(events_by_clip: dict[str, list[dict]]) -> list[dict]:
    return [
        {"filename": clip_id, "event_label": e["event_label"],
         "onset": e["onset"], "offset": e["offset"]}
        for clip_id, events in events_by_clip.items()
        for e in events
    ]


def _nhom_theo_clip(events_by_clip: dict[str, list[dict]], all_clips: list[str]) -> dict:
    """{clip_id: MetaDataContainer} — dựng MỘT lượt thay vì lọc lại cho từng clip.

    `MetaDataContainer.filter(filename=...)` quét TUYẾN TÍNH toàn bộ danh sách sự kiện.
    Gọi nó trong vòng lặp qua clip là một phép O(số clip × số sự kiện), và đo thật
    19/09/2026 trên dev 1.440 clip: filter **101,5 s** so với evaluate **4,7 s** — tức
    98% thời gian chấm điểm nằm ở khâu tra cứu, không ở khâu tính toán. Với một lượt quét
    19 ngưỡng, đó là 32 phút thành 2 phút.
    """
    from dcase_util.containers import MetaDataContainer

    gom: dict[str, list[dict]] = {clip_id: [] for clip_id in all_clips}
    for clip_id, events in events_by_clip.items():
        gom[clip_id].extend(
            {"filename": clip_id, "event_label": e["event_label"],
             "onset": e["onset"], "offset": e["offset"]}
            for e in events
        )
    return {clip_id: MetaDataContainer(rows) for clip_id, rows in gom.items()}


def event_and_segment_f1(reference: dict[str, list[dict]], predicted: dict[str, list[dict]],
                         class_ids: list[str], duration: float) -> tuple[float, float, dict[str, float]]:
    import sed_eval

    # sed_eval cần MỌI file xuất hiện ở cả hai phía, kể cả file không có sự kiện nào —
    # thiếu file ở phía dự đoán sẽ bị hiểu là "không đánh giá" chứ không phải "đoán rỗng",
    # và điểm sẽ cao lên một cách sai.
    all_clips = sorted(set(reference) | set(predicted))
    reference_list = _nhom_theo_clip(reference, all_clips)
    predicted_list = _nhom_theo_clip(predicted, all_clips)

    segment_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=class_ids, time_resolution=1.0
    )
    event_metrics = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=class_ids,
        t_collar=ONSET_COLLAR_SEC,
        percentage_of_length=OFFSET_COLLAR_RATIO,
        evaluate_onset=True,
        evaluate_offset=False,   # chỉ khớp onset: offset của tiếng vang rất mơ hồ
    )
    for clip_id in all_clips:
        ref = reference_list[clip_id]
        pred = predicted_list[clip_id]
        # Clip không có sự kiện nào đi vào bằng danh sách RỖNG, không phải một sự kiện
        # nhãn None (sed_eval sẽ ném `None is not in list`). Phải truyền
        # `evaluated_length_seconds`: không có nó, sed_eval suy độ dài từ chính các sự
        # kiện, nên một clip im lặng bị tính là dài 0 giây và mọi dương tính giả trong đó
        # biến mất khỏi mẫu số — điểm cao lên một cách sai.
        segment_metrics.evaluate(reference_event_list=ref, estimated_event_list=pred,
                                 evaluated_length_seconds=duration)
        event_metrics.evaluate(reference_event_list=ref, estimated_event_list=pred)

    segment_f1 = float(segment_metrics.results_overall_metrics()["f_measure"]["f_measure"])
    event_overall = float(event_metrics.results_overall_metrics()["f_measure"]["f_measure"])
    per_class = {
        label: float(values["f_measure"]["f_measure"] or 0.0)
        for label, values in event_metrics.results_class_wise_metrics().items()
    }
    return segment_f1, event_overall, per_class
