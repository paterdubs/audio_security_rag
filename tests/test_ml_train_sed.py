"""Test cho `train_sed.py` — cổng hợp đồng dữ liệu và checkpoint resume.

Trước 19/09 cổng hợp đồng chỉ IN CẢNH BÁO khi `hop_dong != "PASSED"`, không chặn gì cả —
v1/v2 đã train trên lô legacy không đạt hợp đồng mà không có gì ngăn lại. `KHONG_RO`
("chưa ai kiểm") và `FAILED` ("đã kiểm, hỏng") bị đối xử NHƯ NHAU ở cổng cũ: cả hai chỉ in
cùng một dòng cảnh báo rồi train tiếp — cùng lớp lỗi với `run_manifest.doc_hop_dong` từng
khai `PASSED` khi thiếu file báo cáo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.training.train_sed import (  # noqa: E402
    kiem_cong_hop_dong,
    luu_checkpoint_resume,
    nap_checkpoint_resume,
)

# ── Cổng hợp đồng dữ liệu ────────────────────────────────────────────────────


def test_cong_mo_khi_PASSED():
    assert kiem_cong_hop_dong("PASSED", bo_qua=False) is None


def test_cong_chan_khi_FAILED_va_khong_bo_qua():
    thong_diep = kiem_cong_hop_dong("FAILED", bo_qua=False)
    assert thong_diep is not None
    assert "FAILED" in thong_diep
    assert "--force-du-lieu-chua-dat" in thong_diep


def test_cong_chan_khi_KHONG_RO_va_khong_bo_qua():
    """Đây là hành vi MỚI so với cổng cũ: trước 19/09, KHONG_RO ('chưa ai kiểm') chỉ in
    cảnh báo rồi train tiếp giống hệt PASSED. 'Chưa ai kiểm' không phải lý do để train —
    nó là lý do để KHÔNG train cho tới khi kiểm."""
    thong_diep = kiem_cong_hop_dong("KHONG_RO", bo_qua=False)
    assert thong_diep is not None
    assert "KHONG_RO" in thong_diep


def test_cong_mo_khi_bo_qua_du_FAILED():
    assert kiem_cong_hop_dong("FAILED", bo_qua=True) is None


def test_cong_mo_khi_bo_qua_du_KHONG_RO():
    assert kiem_cong_hop_dong("KHONG_RO", bo_qua=True) is None


# ── Checkpoint resume: optimizer + scheduler + RNG ──────────────────────────


def _model_optim_sched():
    """Model/optimizer/scheduler bé — đủ để kiểm state_dict khớp, không cần GPU."""
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=5, steps_per_epoch=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, optimizer, scheduler, scaler


def test_luu_va_nap_checkpoint_resume_khop_state_dict(tmp_path):
    """Lưu → nạp vào một bộ model/optimizer/scheduler MỚI (mô phỏng resume ở tiến trình
    khác) → state_dict phải khớp. Đây là bất biến tối thiểu: thiếu nó thì --resume tưởng
    thành công nhưng thực ra tiếp tục train từ một trạng thái optimizer trống rỗng."""
    model, optimizer, scheduler, scaler = _model_optim_sched()
    for _ in range(3):
        loss = model(torch.randn(1, 4)).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    path = tmp_path / "checkpoint_resume.pt"
    luu_checkpoint_resume(path, model, optimizer, scheduler, scaler, epoch=3, best_map=0.42)

    model2, optimizer2, scheduler2, scaler2 = _model_optim_sched()
    epoch, best_map = nap_checkpoint_resume(path, model2, optimizer2, scheduler2, scaler2)

    assert epoch == 3
    assert best_map == pytest.approx(0.42)
    for k, v in model.state_dict().items():
        assert torch.equal(v, model2.state_dict()[k])
    assert optimizer.state_dict()["state"].keys() == optimizer2.state_dict()["state"].keys()
    assert scheduler.state_dict()["last_epoch"] == scheduler2.state_dict()["last_epoch"]


def test_nap_checkpoint_resume_khoi_phuc_dung_day_so_ngau_nhien_tiep_theo(tmp_path):
    """Bất biến quan trọng nhất của resume: sau khi nạp lại, dãy số ngẫu nhiên tiếp theo
    phải GIỐNG HỆT như thể chưa hề dừng — không phải một dãy mới bắt đầu lại từ seed gốc.
    Thiếu nó thì lượt --resume và lượt chạy liền mạch cho hai kết quả khác nhau dù cùng
    seed, và không ai biết vì sao vì cả hai đều 'chạy đúng', chỉ là RNG khác nhau.
    """
    import random

    random.seed(20260921)
    np.random.seed(20260921)
    torch.manual_seed(20260921)

    model, optimizer, scheduler, scaler = _model_optim_sched()
    # "Dùng" RNG một chút trước khi lưu, như vài epoch train thật đã làm.
    random.random()
    np.random.rand()
    torch.randn(3)

    path = tmp_path / "checkpoint_resume.pt"
    luu_checkpoint_resume(path, model, optimizer, scheduler, scaler, epoch=1, best_map=0.1)

    # Nhánh A: KHÔNG dừng, tiếp tục dùng RNG ngay tại chỗ — đây là "kết quả đúng".
    ky_vong_python = random.random()
    ky_vong_numpy = np.random.rand()
    ky_vong_torch = torch.randn(3).clone()

    # Nhánh B: mô phỏng tiến trình MỚI khởi động lại RNG bằng seed gốc (như --resume thật
    # sự làm: gieo_mam() chạy trước, rồi nap_checkpoint_resume() phải GHI ĐÈ lại đúng chỗ).
    random.seed(20260921)
    np.random.seed(20260921)
    torch.manual_seed(20260921)
    model2, optimizer2, scheduler2, scaler2 = _model_optim_sched()
    nap_checkpoint_resume(path, model2, optimizer2, scheduler2, scaler2)

    thuc_te_python = random.random()
    thuc_te_numpy = np.random.rand()
    thuc_te_torch = torch.randn(3)

    assert thuc_te_python == ky_vong_python
    assert thuc_te_numpy == ky_vong_numpy
    assert torch.equal(thuc_te_torch, ky_vong_torch)
