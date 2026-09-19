"""Test cho quét ngưỡng và cho phép gom sự kiện theo clip.

Phép gom là một tối ưu thuần tốc độ — và tối ưu thuần tốc độ là nơi dễ đổi kết quả nhất
mà không ai để ý, vì không có triệu chứng nào ngoài một con số khác đi.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.sed_metrics import _nhom_theo_clip, event_and_segment_f1  # noqa: E402
from ml.evaluation.threshold_sweep import luoi_nguong  # noqa: E402

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
