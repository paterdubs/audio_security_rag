"""Test cho quét ngưỡng và cho phép gom sự kiện theo clip.

Phép gom là một tối ưu thuần tốc độ — và tối ưu thuần tốc độ là nơi dễ đổi kết quả nhất
mà không ai để ý, vì không có triệu chứng nào ngoài một con số khác đi.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.predictions import DuDoan  # noqa: E402
from ml.evaluation.sed_metrics import (  # noqa: E402
    DEFAULT_MEDIAN_FILTER_FRAMES,
    MAX_MEDIAN_FRAMES,
    _nhom_theo_clip,
    event_and_segment_f1,
)
from ml.evaluation.threshold_sweep import (  # noqa: E402
    cua_so_loc,
    cua_so_loc_tu_train,
    luoi_nguong,
)


def _du_doan_gia(events, class_ids, n_frames=1001, duration=10.0) -> DuDoan:
    """`DuDoan` tối giản — chỉ đủ trường mà `cua_so_loc` đọc tới."""
    ref_clip, ref_class, ref_onset, ref_offset = [], [], [], []
    for clip_idx, class_idx, on, off in events:
        ref_clip.append(clip_idx)
        ref_class.append(class_idx)
        ref_onset.append(on)
        ref_offset.append(off)
    n = len(class_ids)
    return DuDoan(
        doan_prob=np.zeros((1, 1, n), dtype=np.float16),
        clip_prob=np.zeros((1, n), dtype=np.float16),
        clip_ids=["a"], class_ids=list(class_ids),
        ref_clip=np.array(ref_clip), ref_class=np.array(ref_class),
        ref_onset=np.array(ref_onset, dtype=float), ref_offset=np.array(ref_offset, dtype=float),
        meta={"n_frames": n_frames, "duration": duration, "ratio": 1},
    )

# ── Lưới ngưỡng ──────────────────────────────────────────────────────────────


def test_luoi_nguong_co_ca_hai_dau_mut():
    luoi = luoi_nguong("0.05:0.95:0.05")
    assert luoi[0] == 0.05
    assert luoi[-1] == 0.95
    assert len(luoi) == 19


def test_luoi_nguong_khong_troi_vi_sai_so_dau_phay_dong():
    """0.05 + 0.05 + … bằng float sẽ ra 0.30000000000000004 và in ra bảng thành rác.
    Lưới phải tính từ chỉ số rồi làm tròn, không cộng dồn."""
    assert 0.3 in luoi_nguong("0.05:0.95:0.05")
    assert all(round(x, 4) == x for x in luoi_nguong("0.05:0.99:0.01"))


# ── Gom sự kiện theo clip ────────────────────────────────────────────────────


def test_gom_giu_lai_ca_clip_khong_su_kien():
    """Clip im lặng phải có mặt với danh sách RỖNG. Thiếu nó, sed_eval hiểu là 'không
    đánh giá' chứ không phải 'đoán rỗng', và mọi dương tính giả trong clip đó biến mất
    khỏi mẫu số — điểm cao lên một cách sai."""
    gom = _nhom_theo_clip({"a": [{"event_label": "gunshot", "onset": 1.0, "offset": 2.0}]},
                          ["a", "b"])
    assert set(gom) == {"a", "b"}
    assert len(gom["a"]) == 1
    assert len(gom["b"]) == 0


def test_gom_khop_voi_filter_cua_dcase_util():
    """Phép gom thay `MetaDataContainer.filter(filename=…)` — một phép quét TUYẾN TÍNH
    lặp lại cho từng clip, tức O(số clip × số sự kiện).

    Đo thật 19/09/2026 trên dev 1.440 clip / 65.884 sự kiện dự báo: filter **101,5 s** so
    với evaluate **4,7 s**; 98% thời gian chấm điểm nằm ở khâu tra cứu. Test này khẳng
    định bản thay thế cho ra ĐÚNG cùng nội dung, không chỉ nhanh hơn.
    """
    from dcase_util.containers import MetaDataContainer

    su_kien = {
        "a": [{"event_label": "gunshot", "onset": 1.0, "offset": 2.0},
              {"event_label": "siren", "onset": 3.0, "offset": 8.0}],
        "b": [{"event_label": "siren", "onset": 0.5, "offset": 1.5}],
        "c": [],
    }
    tat_ca = sorted(su_kien)
    phang = MetaDataContainer([
        {"filename": clip_id, "event_label": e["event_label"],
         "onset": e["onset"], "offset": e["offset"]}
        for clip_id, events in su_kien.items() for e in events
    ])
    gom = _nhom_theo_clip(su_kien, tat_ca)

    for clip_id in tat_ca:
        mong_doi = sorted((i["event_label"], i["onset"], i["offset"])
                          for i in phang.filter(filename=clip_id))
        thuc_te = sorted((i["event_label"], i["onset"], i["offset"]) for i in gom[clip_id])
        assert thuc_te == mong_doi


def test_clip_im_lang_co_du_bao_sai_van_bi_phat():
    """Hệ quả trực tiếp của test trên, đo bằng chính điểm số: một clip không có sự kiện
    nào mà model đoán ra một sự kiện thì F1 phải < 1,0."""
    tham_chieu = {"a": [{"event_label": "gunshot", "onset": 1.0, "offset": 2.0}], "b": []}
    du_bao = {"a": [{"event_label": "gunshot", "onset": 1.0, "offset": 2.0}],
              "b": [{"event_label": "siren", "onset": 4.0, "offset": 5.0}]}

    _, event_f1, _ = event_and_segment_f1(tham_chieu, du_bao, ["gunshot", "siren"], 10.0)

    assert event_f1 < 1.0
    assert event_f1 == pytest.approx(2 / 3, abs=1e-6)   # 1 đúng, 1 chèn, 1 sót


# ── Trần cửa sổ lọc thích ứng ────────────────────────────────────────────────


def test_tran_mac_dinh_chan_cua_so_su_kien_dai():
    """`long_event` có sự kiện dài 3 s — trần MAX_MEDIAN_FRAMES=51 (~0,5s) cắt cụt cửa sổ
    lẽ ra phải rộng gần 300 khung. Đây là nghi phạm còn lại sau khi ablation cửa sổ theo
    lớp không giải thích được khoảng cách sach−long_event (xem
    docs/measurements/long_event_postproc_20260919.md §2)."""
    du_doan = _du_doan_gia([(0, 0, 1.0, 4.0)], class_ids=["long_event"])
    sizes = cua_so_loc(du_doan, thich_ung=True)
    assert sizes[0] == MAX_MEDIAN_FRAMES


def test_max_frames_tuy_chinh_thao_tran_cho_su_kien_dai():
    du_doan = _du_doan_gia([(0, 0, 1.0, 4.0)], class_ids=["long_event"])
    sizes = cua_so_loc(du_doan, thich_ung=True, max_frames=401)
    assert MAX_MEDIAN_FRAMES < sizes[0] <= 401


def test_max_frames_khong_dung_toi_khi_cua_so_co_dinh():
    """`max_frames` chỉ có nghĩa khi `thich_ung=True` — truyền vào lúc cửa sổ cố định
    không được lặng lẽ đổi hành vi 7 khung mặc định."""
    du_doan = _du_doan_gia([(0, 0, 1.0, 4.0)], class_ids=["long_event"])
    assert cua_so_loc(du_doan, thich_ung=False, max_frames=401) == DEFAULT_MEDIAN_FILTER_FRAMES


# ── Cửa sổ suy từ TRAIN (không rò rỉ) ────────────────────────────────────────


def _ghi_train_meta_gia(tmp_path: Path, class_ids, events_per_clip) -> Path:
    """`{features_dir}/train_meta.json` tối giản — đúng định dạng mà
    `PrecomputedSedDataset.events_of()` đọc (`sed_data.py:68`): mỗi clip có `events` là
    danh sách (class_idx, onset, offset)."""
    meta = {
        "class_ids": list(class_ids), "duration": 10.0,
        "clips": [{"clip_id": f"c{i}", "events": events}
                 for i, events in enumerate(events_per_clip)],
    }
    (tmp_path / "train_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return tmp_path


def test_cua_so_tu_train_doc_thang_tu_meta_khong_qua_du_doan(tmp_path):
    """`cua_so_loc_tu_train` không nhận tham số `du_doan` nào cả — về mặt chữ ký hàm,
    nhãn của tập đang chấm KHÔNG THỂ lọt vào phép tính này. Đó là điều `cua_so_loc()`
    (suy từ chính tập đang chấm) không đảm bảo được, và là lý do việc này tồn tại: biến
    +0,037 F1 đo được ở long_event_postproc_20260919.md từ chặn trên lạc quan thành số
    báo cáo được."""
    features_dir = _ghi_train_meta_gia(
        tmp_path, class_ids=["long_event", "gunshot"],
        events_per_clip=[[(0, 1.0, 4.0)], [(1, 0.0, 0.08)]],
    )
    sizes = cua_so_loc_tu_train(["long_event", "gunshot"], frames_per_second=100.0,
                                features_dir=features_dir)
    assert sizes[0] == MAX_MEDIAN_FRAMES        # sự kiện 3s bị trần 51 chặn, như cua_so_loc
    assert sizes[1] < sizes[0]                  # gunshot 80ms → cửa sổ hẹp hơn hẳn


def test_cua_so_tu_train_tran_tuy_chinh_giong_het_cua_so_loc(tmp_path):
    features_dir = _ghi_train_meta_gia(
        tmp_path, class_ids=["long_event"], events_per_clip=[[(0, 1.0, 4.0)]],
    )
    du_doan = _du_doan_gia([(0, 0, 1.0, 4.0)], class_ids=["long_event"])
    tu_dev = cua_so_loc(du_doan, thich_ung=True, max_frames=401)
    tu_train = cua_so_loc_tu_train(["long_event"], frames_per_second=100.1,
                                   features_dir=features_dir, max_frames=401)
    assert list(tu_train) == list(tu_dev)        # cùng dữ liệu, hai đường đọc phải khớp


def test_cua_so_tu_train_le_thu_tu_lop_khac_meta_thi_no(tmp_path):
    """`train_meta.json` và `class_ids` của run đang chấm là hai file NẠP RIÊNG — nếu thứ
    tự lớp lệch nhau thì `class_event_durations` gán nhầm độ dài của lớp này cho lớp khác
    một cách im lặng. Đây là ranh giới hệ thống, phải chặn cứng chứ không được đoán."""
    features_dir = _ghi_train_meta_gia(
        tmp_path, class_ids=["gunshot", "long_event"], events_per_clip=[[(0, 0.0, 0.08)]],
    )
    with pytest.raises(ValueError, match="thứ tự lớp"):
        cua_so_loc_tu_train(["long_event", "gunshot"], frames_per_second=100.0,
                            features_dir=features_dir)
