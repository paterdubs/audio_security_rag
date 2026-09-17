"""Test cho nhánh SED trong ml/ — nhãn, giải mã sự kiện, và tương đương với PANNs gốc.

Các lỗi ở đây đều IM LẶNG: nhãn lệch một khung, sự kiện ngắn biến mất, hay bản sao
forward trôi khỏi thư viện gốc — tất cả vẫn chạy ra số, chỉ là số sai.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.datasets.synthetic_sed import (  # noqa: E402
    BACKGROUND_CLASS, clip_targets, event_class_ids, frame_targets, parse_jams,
)
from ml.evaluation.sed_metrics import clip_level_map, frames_to_events, median_filter  # noqa: E402

ONTOLOGY_PATH = REPO_ROOT / "ml" / "configs" / "ontology_map.yaml"


# ── Danh sách lớp ────────────────────────────────────────────────────────────


def test_khong_co_ambient_noise_trong_head():
    """ambient_noise là nền luôn bật suốt 10 s ở MỌI clip Scaper. Để nó trong head thì
    AP của nó luôn ≈ 1.0 và mAP tổng bị thổi phồng mà không phản ánh năng lực nào."""
    assert BACKGROUND_CLASS not in event_class_ids(ONTOLOGY_PATH)


def test_dung_15_lop_su_kien():
    assert len(event_class_ids(ONTOLOGY_PATH)) == 15


def test_thu_tu_lop_on_dinh_giua_cac_lan_goi():
    """Chỉ số lớp được nướng vào checkpoint — thứ tự đổi là mọi cột trong head đổi nghĩa."""
    assert event_class_ids(ONTOLOGY_PATH) == event_class_ids(ONTOLOGY_PATH)
    assert event_class_ids(ONTOLOGY_PATH) == sorted(event_class_ids(ONTOLOGY_PATH))


# ── Đọc nhãn từ .jams ────────────────────────────────────────────────────────


def _viet_jams(path: Path, observations: list[dict]) -> None:
    path.write_text(json.dumps({"annotations": [{"namespace": "scaper", "duration": 10.0,
                                                 "data": observations}]}), encoding="utf-8")


def test_bo_qua_observation_nen(tmp_path):
    """Observation nền mang nhãn KHU VỰC (`parking`), không phải nhãn lớp âm thanh.
    Nhận nhầm nó sẽ tạo ra 4 lớp ma không có trong taxonomy."""
    path = tmp_path / "a.jams"
    _viet_jams(path, [
        {"time": 0.0, "duration": 10.0, "value": {"label": "parking", "role": "background"}},
        {"time": 2.0, "duration": 1.0, "value": {"label": "gunshot", "role": "foreground"}},
    ])
    events = parse_jams(path, {"gunshot": 0})
    assert events == ((0, 2.0, 3.0),)


def test_nhan_la_thi_bao_loi_chu_khong_nuot_lang(tmp_path):
    """Taxonomy đổi sau khi sinh dữ liệu là chuyện có thật (đã tách laughter_cheering).
    Bỏ qua lặng lẽ sẽ cho ra dataset thiếu nhãn mà mọi thứ vẫn chạy bình thường."""
    path = tmp_path / "a.jams"
    _viet_jams(path, [{"time": 1.0, "duration": 1.0,
                       "value": {"label": "lop_khong_ton_tai", "role": "foreground"}}])
    with pytest.raises(KeyError, match="lop_khong_ton_tai"):
        parse_jams(path, {"gunshot": 0})


# ── Nhãn mức khung ───────────────────────────────────────────────────────────


def test_su_kien_ngan_hon_mot_khung_van_chiem_it_nhat_mot_khung():
    """Tiếng chén vỡ ~0,2 s; nếu làm tròn cả hai đầu cùng chiều thì sự kiện biến mất khỏi
    nhãn và model học được rằng nó không tồn tại."""
    target = frame_targets([(0, 5.0, 5.001)], n_frames=100, n_classes=1, duration=10.0)
    assert target.sum() >= 1


def test_nhan_khung_dung_vi_tri():
    target = frame_targets([(1, 2.0, 4.0)], n_frames=100, n_classes=3, duration=10.0)
    assert target[:20, 1].sum() == 0          # trước onset
    assert target[20:40, 1].all()             # trong sự kiện
    assert target[40:, 1].sum() == 0          # sau offset
    assert target[:, 0].sum() == 0            # lớp khác không bị bật lây


def test_su_kien_vuot_bien_bi_cat_chu_khong_tran():
    target = frame_targets([(0, 9.5, 12.0)], n_frames=100, n_classes=1, duration=10.0)
    assert target.shape == (100, 1)
    assert target[-1, 0] == 1.0


def test_nhan_muc_clip_gop_moi_lan_xuat_hien():
    assert clip_targets([(0, 1.0, 2.0), (0, 5.0, 6.0), (2, 3.0, 4.0)], 3).tolist() == [1.0, 0.0, 1.0]


# ── Giải mã khung → sự kiện ──────────────────────────────────────────────────


def test_giai_ma_mot_doan_lien_tuc_thanh_mot_su_kien():
    probabilities = np.zeros((100, 1), dtype=np.float32)
    probabilities[20:40, 0] = 0.9
    events = frames_to_events(probabilities, ["gunshot"], duration=10.0, median_size=1)
    assert len(events) == 1
    assert events[0]["event_label"] == "gunshot"
    assert events[0]["onset"] == pytest.approx(2.0, abs=0.1)
    assert events[0]["offset"] == pytest.approx(4.0, abs=0.1)


def test_hai_doan_roi_nhau_ra_hai_su_kien():
    probabilities = np.zeros((100, 1), dtype=np.float32)
    probabilities[10:20, 0] = 0.9
    probabilities[60:70, 0] = 0.9
    assert len(frames_to_events(probabilities, ["gunshot"], 10.0, median_size=1)) == 2


def test_loc_trung_vi_xoa_gai_don_le():
    """Một khung nhiễu lẻ thành một 'sự kiện' dài 10 ms; hàng trăm cái như thế làm F1
    mức sự kiện sụp đổ vì dương tính giả tí hon."""
    probabilities = np.zeros((100, 1), dtype=np.float32)
    probabilities[50, 0] = 0.99          # đúng MỘT khung
    assert len(frames_to_events(probabilities, ["x"], 10.0, median_size=7)) == 0
    assert len(frames_to_events(probabilities, ["x"], 10.0, median_size=1)) == 1


def test_loc_trung_vi_khong_doi_hinh_dang_mang():
    probabilities = np.random.rand(200, 15).astype(np.float32)
    assert median_filter(probabilities, 7).shape == probabilities.shape


# ── mAP ──────────────────────────────────────────────────────────────────────


def test_lop_vang_mat_bi_bo_qua_chu_khong_tinh_bang_khong():
    """AP của lớp không xuất hiện lần nào thì không định nghĩa được. Tính 0 sẽ kéo mAP
    xuống theo độ HIẾM của lớp chứ không theo năng lực model."""
    scores = np.array([[0.9, 0.1], [0.8, 0.2]])
    targets = np.array([[1.0, 0.0], [1.0, 0.0]])      # lớp thứ hai không bao giờ dương
    macro, per_class = clip_level_map(scores, targets, ["co", "vang"])
    assert "vang" not in per_class
    assert macro == pytest.approx(per_class["co"])


def test_du_doan_hoan_hao_cho_map_bang_mot():
    scores = np.array([[0.9, 0.1], [0.1, 0.9]])
    targets = np.array([[1.0, 0.0], [0.0, 1.0]])
    macro, _ = clip_level_map(scores, targets, ["a", "b"])
    assert macro == pytest.approx(1.0)


# ── Bản sao forward phải khớp thư viện gốc ──────────────────────────────────


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "reference" / "Cnn14_DecisionLevelMax.pth").exists(),
    reason="chưa có checkpoint PANNs",
)
def test_forward_tu_viet_khop_ban_goc():
    """`PannsSed.embed_frames` CHÉP logic nội bộ của panns_inference để lấy được LOGIT
    (BCEWithLogitsLoss cần logit; BCELoss trên xác suất dưới AMP fp16 sinh NaN).

    Bản chép sẽ trôi âm thầm nếu thư viện đổi — test này là thứ duy nhất bắt được.
    """
    import torch
    from panns_inference.models import Cnn14_DecisionLevelMax

    from ml.models.panns_sed import PANNS_CLASSES, PANNS_MEL, PannsSed

    checkpoint = REPO_ROOT / "data" / "reference" / "Cnn14_DecisionLevelMax.pth"
    goc = Cnn14_DecisionLevelMax(classes_num=PANNS_CLASSES, **PANNS_MEL)
    goc.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    goc.eval()

    cua_ta = PannsSed(PANNS_CLASSES, checkpoint)
    cua_ta.head = goc.fc_audioset          # dùng chung head để hai đường phải trùng khít
    cua_ta.eval()

    torch.manual_seed(0)
    x = torch.randn(1, 32000 * 2) * 0.1
    with torch.no_grad():
        chuan = goc(x)
        cua_ta_out = cua_ta(x)

    assert torch.allclose(chuan["framewise_output"],
                          torch.sigmoid(cua_ta_out["frame_logits"]), atol=1e-5)
    assert torch.allclose(chuan["clipwise_output"],
                          torch.sigmoid(cua_ta_out["clip_logits"]), atol=1e-5)


# ── Độ phân giải thời gian ───────────────────────────────────────────────────


@pytest.mark.parametrize("time_pool_blocks,doan_toi_thieu", [(5, 31), (4, 62), (3, 125), (2, 250)])
def test_bot_gop_thoi_gian_thi_do_phan_giai_min_hon(time_pool_blocks, doan_toi_thieu):
    """CNN14 gốc gộp thời gian ở cả 5 block → 31 đoạn cho 10 s = 323 ms mỗi đoạn, THÔ HƠN
    chính collar 200 ms của DCASE. Model không thể trúng collar khi bước lượng tử của nó
    còn lớn hơn collar — đó là trần cứng của F1 mức sự kiện, hậu xử lý không phá được."""
    import torch

    from ml.models.panns_sed import PannsSed

    model = PannsSed(15, checkpoint_path=None, time_pool_blocks=time_pool_blocks).eval()
    with torch.no_grad():
        embed, _ = model.embed_frames(torch.zeros(1, 320000))
    assert embed.shape[1] == doan_toi_thieu


@pytest.mark.parametrize("time_pool_blocks", [5, 4, 3, 2])
def test_so_khung_dau_ra_luon_khop_so_khung_mel(time_pool_blocks):
    """Interpolator của PANNs ghim cứng tỉ lệ 32. Dùng lại nó khi đã bớt gộp thời gian sẽ
    nội suy vượt số khung mel và `pad_framewise_output` nổ với kích thước ÂM."""
    import torch

    from ml.models.panns_sed import PannsSed

    model = PannsSed(15, checkpoint_path=None, time_pool_blocks=time_pool_blocks).eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 320000))
        _, frames_num = model.embed_frames(torch.zeros(1, 320000))
    assert output["frame_logits"].shape[1] == frames_num


def test_loc_trung_vi_vo_nghia_khi_tin_hieu_hang_tung_doan():
    """Đo thật 17/09/2026: với cấu hình gốc, lọc cửa sổ 7 và 51 cho ra mảng GIỐNG HỆT.

    Không phải bộ lọc hỏng — đầu ra của model là hằng trên từng khối 32 khung (nội suy
    lặp từ 31 đoạn), nên mọi cửa sổ ≤ 63 khung đều có đa số nằm trong cùng một khối và
    trung vị bằng đúng giá trị khối đó. Đây là lý do hậu xử lý thích ứng KHÔNG thể giúp
    gì ở độ phân giải 323 ms, và chỉ có ý nghĩa khi khối nhỏ lại."""
    khoi = 32
    tin_hieu = np.repeat(np.random.default_rng(0).random(31), khoi)[:1001, None].astype(np.float32)

    assert np.allclose(median_filter(tin_hieu, 7), median_filter(tin_hieu, 51))

    # Cùng tín hiệu nhưng khối nhỏ (8 khung, tương ứng 80 ms) thì bộ lọc có tác dụng thật.
    khoi_nho = np.repeat(np.random.default_rng(0).random(125), 8)[:1000, None].astype(np.float32)
    assert not np.allclose(median_filter(khoi_nho, 7), median_filter(khoi_nho, 51))


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "reference" / "Cnn14_DecisionLevelMax.pth").exists(),
    reason="chưa có checkpoint PANNs",
)
def test_log_mel_khong_ra_inf_duoi_amp():
    """Đo thật 17/09/2026: dưới autocast fp16, spectrogram còn hữu hạn nhưng log-mel ra
    inf (biên độ nhỏ underflow về 0 rồi log(0)). inf lan thành NaN khắp mạng, và triệu
    chứng lộ ra rất xa nguồn — sklearn báo 'Input contains NaN' lúc tính mAP."""
    import torch

    from ml.models.panns_sed import PannsSed

    if not torch.cuda.is_available():
        pytest.skip("cần CUDA để kiểm tra autocast fp16")

    model = PannsSed(15, REPO_ROOT / "data" / "reference" / "Cnn14_DecisionLevelMax.pth").cuda().eval()
    x = torch.randn(2, 32000 * 2, device="cuda") * 0.1
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        output = model(x)
    assert torch.isfinite(output["frame_logits"]).all()
    assert torch.isfinite(output["clip_logits"]).all()


# ── Độ phân giải thời gian phải đi cùng trọng số ─────────────────────────────


def test_doc_time_pool_blocks_uu_tien_checkpoint(tmp_path):
    """Checkpoint là nguồn chắc chắn nhất — thắng cả history.json."""
    from ml.evaluation.eval_sed import doc_time_pool_blocks

    (tmp_path / "history.json").write_text('{"args": {"time_pool_blocks": 5}}', encoding="utf-8")

    assert doc_time_pool_blocks(tmp_path, {"time_pool_blocks": 3}) == 3


def test_doc_time_pool_blocks_doc_duoc_run_cu_tu_history(tmp_path):
    """`panns_ft_v2` train ở pool=3 nhưng checkpoint cũ không ghi — phải lấy từ history."""
    from ml.evaluation.eval_sed import doc_time_pool_blocks

    (tmp_path / "history.json").write_text('{"args": {"time_pool_blocks": 3}}', encoding="utf-8")

    assert doc_time_pool_blocks(tmp_path, {}) == 3


def test_doc_time_pool_blocks_khong_biet_thi_CANH_BAO(tmp_path, capsys):
    """Không có nguồn nào thì phải kêu lên.

    Dựng sai độ phân giải cho ra một bộ số vô nghĩa mà chương trình vẫn chạy trót lọt
    — im lặng ở đây là không chấp nhận được.
    """
    from ml.evaluation.eval_sed import DEFAULT_TIME_POOL_BLOCKS, doc_time_pool_blocks

    ra = doc_time_pool_blocks(tmp_path, {})

    assert ra == DEFAULT_TIME_POOL_BLOCKS
    assert "⚠️" in capsys.readouterr().out
