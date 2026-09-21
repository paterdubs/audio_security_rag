"""Chạy nốt ablation pos_weight KHÔNG CẦN NGƯỜI TRÔNG — dành cho một đêm.

    .venv/Scripts/python.exe scripts/chay_ablation_pos_weight.py

Lộ trình script tự đi, tuần tự, không chạy song song (chạy song song là thứ đã làm nhiễu
cột thời gian của `panns_ft_pw1` sáng nay):

    1. chờ `panns_ft_pw10` train xong (tiến trình đã chạy sẵn từ trước)
    2. đánh giá pw10: predictions → threshold_sweep → calibration → error_analysis
    3. train `panns_ft_pw30` (25 epoch)
    4. đánh giá pw30
    5. compare_runs trên v3 + pw1 + pw10 + pw30
    6. sinh docs/measurements/pos_weight_ablation_20260921.md từ json, không gõ tay số nào

MỌI BƯỚC ĐỀU BỎ QUA ĐƯỢC NẾU ĐÃ CÓ KẾT QUẢ: chạy lại script sau khi mất điện thì nó đi
tiếp từ chỗ dở, không train lại từ đầu.

CANH CHỪNG TIẾN TRÌNH CHẾT: nếu log của một lượt train đứng yên quá `STALE_PHUT` mà chưa
đủ epoch, script coi tiến trình đã chết và khởi động lại bằng `--resume` (checkpoint có
đủ optimizer/scheduler/RNG từ 21/09). Tối đa `TOI_DA_KHOI_DONG_LAI` lần cho mỗi lượt, để
một lỗi lặp vô hạn không ngốn cả đêm GPU.

KHÔNG tự sửa CLAUDE.md/STATUS.md/PLAN.md/TRAINING_OPS_PLAN.md và KHÔNG tự commit: diễn
giải kết quả và đồng bộ tài liệu cần phán đoán, để sáng làm. Script chỉ để lại số.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from common import enable_utf8_output  # noqa: E402

enable_utf8_output()

PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
RUNS_DIR = REPO_ROOT / "ml" / "runs"
MEASUREMENTS = REPO_ROOT / "docs" / "measurements"
NHAT_KY = RUNS_DIR / "ablation_pos_weight.log"

SO_EPOCH = 25
STALE_PHUT = 30           # epoch chậm nhất đã đo là 7,5 phút — 30 phút là 4x biên an toàn
TOI_DA_KHOI_DONG_LAI = 2
NHIP_KIEM_GIAY = 60

# Giữ NGUYÊN cấu hình v3, chỉ đổi --pos-weight-max. Lấy từ ml/runs/panns_ft_v3/manifest.json.
CAU_HINH_V3 = [
    "--split", "train", "--epochs", str(SO_EPOCH), "--batch-size", "16",
    "--lr", "1e-3", "--backbone-lr-scale", "0.05", "--workers", "0",
    "--val-ratio", "0.1", "--seed", "20260917", "--sample-rate", "32000",
    "--duration", "10.0", "--threshold", "0.5", "--clip-loss-weight", "0.5",
    "--full-eval-every", "5", "--mixup-alpha", "0.2", "--mixup-prob", "0.5",
    "--mixup-label", "hard", "--adaptive-postproc", "--time-pool-blocks", "3",
]

LUOT = [("panns_ft_pw10", 10.0), ("panns_ft_pw30", 30.0)]
MOI_RUN = ["panns_ft_v3", "panns_ft_pw1", "panns_ft_pw10", "panns_ft_pw30"]


# ── Ghi nhật ký ──────────────────────────────────────────────────────────────


def ghi(dong: str) -> None:
    """In VÀ ghi đĩa ngay — script chạy lúc không có ai nhìn màn hình."""
    dau = time.strftime("%H:%M:%S")
    print(f"[{dau}] {dong}", flush=True)
    NHAT_KY.parent.mkdir(parents=True, exist_ok=True)
    with NHAT_KY.open("a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {dong}\n")
        f.flush()


def so(x: float | None) -> str:
    """Dấu phẩy thập phân như mọi báo cáo khác của dự án; None là CHƯA ĐO."""
    return "chưa đo" if x is None else f"{x:.4f}".replace(".", ",")


# ── Đọc trạng thái ───────────────────────────────────────────────────────────


def _json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def train_xong(run_dir: Path, so_epoch: int = SO_EPOCH) -> bool:
    """Đủ số epoch trong manifest mới tính là xong.

    Không dựa vào dòng "✓ ... mAP tốt nhất" của log: log có thể bị cắt giữa chừng khi
    tiến trình bị kill, còn manifest được ghi lại sau MỖI epoch (quyết định 21/09).
    """
    m = _json(run_dir / "manifest.json")
    if not m:
        return False
    return len(m.get("timing", {}).get("epochs", [])) >= so_epoch


def doc_ket_qua(run_dir: Path) -> dict:
    """Gom số của một run từ ba file json. Thiếu file → None, KHÔNG phải 0."""
    cal = _json(run_dir / "calibration.json") or {}
    ana = _json(run_dir / "analysis.json") or {}
    man = _json(run_dir / "manifest.json") or {}
    args = man.get("config", {}).get("args", {})
    return {
        "pos_weight_max": args.get("pos_weight_max"),
        "ece_tong": cal.get("ece_tong"),
        "theta_sao": ana.get("theta_sao"),
        "f1_sao": ana.get("f1_sao"),
        "f1_05": ana.get("f1_05"),
        "hoi_cuu": bool(man.get("hoi_cuu")) or bool(man.get("canh_bao")),
    }


# ── Chạy lệnh con ────────────────────────────────────────────────────────────


def chay(mo_ta: str, lenh: list[str], log: Path) -> bool:
    ghi(f"▶ {mo_ta}")
    bat_dau = time.time()
    with log.open("a", encoding="utf-8") as f:
        kq = subprocess.run([str(PYTHON), *lenh], stdout=f, stderr=subprocess.STDOUT,
                            cwd=REPO_ROOT, check=False)
    giay = time.time() - bat_dau
    dat = kq.returncode == 0
    ghi(f"{'✓' if dat else '❌'} {mo_ta} · {giay:.0f}s · mã thoát {kq.returncode}")
    return dat


def khoi_dong_train(ten: str, pos_weight: float, tiep_tuc: bool) -> subprocess.Popen:
    log = RUNS_DIR / f"train_{ten.replace('panns_ft_', '')}.log"
    lenh = [str(PYTHON), "-m", "ml.training.train_sed", *CAU_HINH_V3,
            "--pos-weight-max", str(pos_weight), "--name", ten]
    if tiep_tuc:
        lenh.append("--resume")
    ghi(f"▶ train {ten} (pos_weight={pos_weight}{', --resume' if tiep_tuc else ''})")
    f = log.open("a", encoding="utf-8")
    return subprocess.Popen(lenh, stdout=f, stderr=subprocess.STDOUT, cwd=REPO_ROOT)


def cho_train(ten: str, pos_weight: float) -> bool:
    """Chờ một lượt train xong; khởi động lại bằng --resume nếu log đứng yên quá lâu."""
    run_dir = RUNS_DIR / ten
    log = RUNS_DIR / f"train_{ten.replace('panns_ft_', '')}.log"
    da_khoi_dong_lai = 0
    if not run_dir.exists() and not log.exists():
        khoi_dong_train(ten, pos_weight, tiep_tuc=False)
    while not train_xong(run_dir):
        time.sleep(NHIP_KIEM_GIAY)
        moc = run_dir / "manifest.json"
        tre = (time.time() - moc.stat().st_mtime) / 60 if moc.exists() else STALE_PHUT + 1
        if tre <= STALE_PHUT:
            continue
        if da_khoi_dong_lai >= TOI_DA_KHOI_DONG_LAI:
            ghi(f"❌ {ten}: đứng yên {tre:.0f} phút và đã khởi động lại "
                f"{da_khoi_dong_lai} lần — BỎ CUỘC, không ngốn thêm GPU.")
            return False
        da_khoi_dong_lai += 1
        ghi(f"⚠️ {ten}: log đứng yên {tre:.0f} phút — coi như tiến trình đã chết, "
            f"khởi động lại lần {da_khoi_dong_lai} bằng --resume.")
        khoi_dong_train(ten, pos_weight, tiep_tuc=True).wait()
    ghi(f"✓ train {ten} đủ {SO_EPOCH} epoch")
    return True


def danh_gia(ten: str) -> bool:
    """predictions → threshold_sweep → calibration → error_analysis, bỏ qua bước đã có."""
    run_dir = RUNS_DIR / ten
    log = RUNS_DIR / f"eval_{ten.replace('panns_ft_', '')}.log"
    buoc = [
        ("predictions", run_dir / "predictions" / "dev_all.npz",
         ["-m", "ml.evaluation.predictions", "--run", ten, "--split", "dev"]),
        ("threshold_sweep", run_dir / "threshold_sweep_dev_all.json",
         ["-m", "ml.evaluation.threshold_sweep", "--run", ten, "--split", "dev"]),
        ("calibration", run_dir / "calibration.json",
         ["-m", "ml.evaluation.calibration", "--run", ten, "--split", "dev"]),
        ("error_analysis", run_dir / "analysis.json",
         ["-m", "ml.evaluation.error_analysis", "--run", ten, "--split", "dev"]),
    ]
    for mo_ta, dich, lenh in buoc:
        if dich.exists():
            ghi(f"↷ {ten}/{mo_ta} đã có, bỏ qua")
            continue
        if not chay(f"{ten}/{mo_ta}", lenh, log):
            return False
    return True


# ── Báo cáo ──────────────────────────────────────────────────────────────────


def bang_tom_tat(ket_qua: dict[str, dict]) -> Iterator[str]:
    yield "| run | pos_weight | ECE tổng | θ\\* | event-F1 @θ\\* | event-F1 @0,5 |"
    yield "|---|---:|---:|---:|---:|---:|"
    for ten, k in ket_qua.items():
        pw = "chưa đo" if k["pos_weight_max"] is None else so(k["pos_weight_max"])
        yield (f"| `{ten}` | {pw} | {so(k['ece_tong'])} | {so(k['theta_sao'])} "
               f"| {so(k['f1_sao'])} | {so(k['f1_05'])} |")


def viet_bao_cao(ket_qua: dict[str, dict], duong_dan: Path) -> Path:
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    with duong_dan.open("w", encoding="utf-8") as f:
        for dong in dong_bao_cao(ket_qua):
            f.write(dong + "\n")
            f.flush()
    return duong_dan


def dong_bao_cao(ket_qua: dict[str, dict]) -> Iterator[str]:
    yield "# Ablation `pos_weight` — số thô, chưa diễn giải — 2026-09-21"
    yield ""
    yield ("Sinh tự động bởi `scripts/chay_ablation_pos_weight.py`. **Trang này chỉ có số**; "
           "phần diễn giải và đồng bộ tài liệu làm tay sau, vì nó cần phán đoán.")
    yield ""
    yield ("Mọi lượt `pw*` dùng **cùng cấu hình v3**, chỉ đổi `--pos-weight-max`; cùng split "
           "train, cùng `time_pool_blocks=3`, chấm trên cùng `data/synthetic/dev`.")
    yield ""
    yield ("> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với train → thiên vị theo "
           "thiết kế, **ngang nhau** cho mọi lượt ở đây. θ\\* chọn trên chính tập đang chấm "
           "→ **chặn trên lạc quan** cho mọi lượt.")
    yield ""
    yield ("> ⚠️ `panns_ft_v3` có manifest lập **hồi cứu**; các lượt `pw*` ghi manifest ngay "
           "lúc train. Xem cột cảnh báo của `compare_runs`.")
    yield ""
    yield from bang_tom_tat(ket_qua)
    yield ""
    yield ("`panns_ft_v3` hiện `pos_weight` là *chưa đo* vì nó train TRƯỚC khi có cờ "
           "`--pos-weight-max`, nên manifest không ghi trường đó. Trần thực tế lúc đó là "
           "hằng `MAX_POS_WEIGHT = 30.0` trong mã — đó chính là lý do `pw30` được train "
           "lại tử tế thay vì dùng `v3` làm đối chứng.")
    yield ""
    yield "## Câu hỏi trang này phải trả lời"
    yield ""
    yield ("1. Quan hệ pos_weight ↔ ECE có **đơn điệu** không (1 → 10 → 30)?")
    yield ("2. `pw30` có **tái hiện** được ECE ≈ 0,3332 của `v3` không? Nếu KHÔNG thì chênh "
           "lệch đến từ chỗ khác chứ không phải `pos_weight`, và kết luận hôm nay phải rút lại.")
    yield ("3. ECE giảm có kèm F1 giảm không? Nếu có thì đó là **đánh đổi**, không phải cải tiến.")
    yield ""
    yield "So sánh đầy đủ: `docs/measurements/compare_runs_pos_weight_20260921.md`."
    yield ""


# ── Chạy ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ghi("═══ bắt đầu chạy nốt ablation pos_weight ═══")
    for ten, pos_weight in LUOT:
        if train_xong(RUNS_DIR / ten):
            ghi(f"↷ {ten} đã train đủ {SO_EPOCH} epoch, bỏ qua")
        elif not cho_train(ten, pos_weight):
            ghi(f"❌ dừng lộ trình: {ten} không train xong")
            break
        if not danh_gia(ten):
            ghi(f"❌ dừng lộ trình: {ten} đánh giá thất bại")
            break

    co_mat = [r for r in MOI_RUN if (RUNS_DIR / r / "analysis.json").exists()]
    if len(co_mat) >= 2:
        chay("compare_runs", ["-m", "ml.tracking.compare_runs", "--runs", *co_mat,
                              "--out", str(MEASUREMENTS / "compare_runs_pos_weight_20260921.md")],
             RUNS_DIR / "ablation_compare.log")

    ket_qua = {r: doc_ket_qua(RUNS_DIR / r) for r in MOI_RUN}
    ra = viet_bao_cao(ket_qua, MEASUREMENTS / "pos_weight_ablation_20260921.md")
    ghi(f"✓ {ra.relative_to(REPO_ROOT)}")
    for dong in bang_tom_tat(ket_qua):
        ghi(dong)
    ghi("═══ xong. Chưa commit, chưa sửa tài liệu — để sáng làm tay. ═══")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
