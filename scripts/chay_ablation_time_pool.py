"""Chạy ablation `--time-pool-blocks` KHÔNG CẦN NGƯỜI TRÔNG — dành cho một đêm.

    .venv/Scripts/python.exe scripts/chay_ablation_time_pool.py

Tiếp theo ablation `pos_weight` (`scripts/chay_ablation_pos_weight.py`, xong 22/09):
`panns_ft_pw1` ĐÃ CHÍNH LÀ cấu hình v4 dự kiến, nhưng người dùng chọn train lại dưới tên
`panns_ft_v4` riêng để có mốc báo cáo sạch. Nhân tiện train v4, kiểm luôn giả thuyết còn
lại của `long_event` (docs/measurements/long_event_gap_20260921.md §4): trường tiếp nhận
thời gian của CNN14 ngắn hơn sự kiện 4+ giây, nên hạ `--time-pool-blocks` từ 3 xuống 2
(tăng độ phân giải thời gian) có thể giảm tỉ lệ phân mảnh `long_event`.

Hai lượt, MỘT biến mỗi lượt — `--pos-weight-max` khoá cứng ở 1.0 cho cả hai (bài học từ
ablation trước: trộn biến sẽ không quy kết được chênh lệch cho biến nào):

    1. `panns_ft_v4`   time_pool_blocks=3 (mặc định — giống hệt `pw1`, chỉ khác tên)
    2. `panns_ft_tpb2` time_pool_blocks=2

Sau mỗi lượt: predictions → threshold_sweep → calibration → error_analysis (bước cuối
TỰ sinh tỉ lệ phân mảnh `long_event` trong `analysis.json['lat_cat']`, không cần đo lại
bằng script khác). Cuối cùng `compare_runs` trên bốn run rồi sinh báo cáo SỐ THÔ — diễn
giải làm tay sau, đúng khuôn đã dùng cho ablation `pos_weight`.

MỌI BƯỚC ĐỀU BỎ QUA ĐƯỢC NẾU ĐÃ CÓ KẾT QUẢ: chạy lại script sau khi mất điện thì nó đi
tiếp từ chỗ dở, không train lại từ đầu.

CANH CHỪNG TIẾN TRÌNH CHẾT: nếu log của một lượt train đứng yên quá `STALE_PHUT` mà chưa
đủ epoch, script coi tiến trình đã chết và khởi động lại bằng `--resume`. Tối đa
`TOI_DA_KHOI_DONG_LAI` lần cho mỗi lượt.

KHÔNG tự sửa CLAUDE.md/STATUS.md/PLAN.md/TRAINING_OPS_PLAN.md và KHÔNG tự commit: diễn
giải kết quả và đồng bộ tài liệu cần phán đoán, để làm tay sau. Script chỉ để lại số.
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
NHAT_KY = RUNS_DIR / "ablation_time_pool.log"

SO_EPOCH = 25
STALE_PHUT = 30           # xem chay_ablation_pos_weight.py — epoch chậm nhất đo được 7,5 phút
TOI_DA_KHOI_DONG_LAI = 2
NHIP_KIEM_GIAY = 60

POS_WEIGHT_MAX = "1.0"    # khoá cứng — quyết định 22/09, xem pos_weight_ket_luan_20260922.md

# Giữ NGUYÊN cấu hình v3 NGOẠI TRỪ --time-pool-blocks (biến ablation, thêm mỗi lượt) và
# --pos-weight-max (khoá 1.0, không phải trần 30 mặc định của v3).
CAU_HINH_BASE = [
    "--split", "train", "--epochs", str(SO_EPOCH), "--batch-size", "16",
    "--lr", "1e-3", "--backbone-lr-scale", "0.05", "--workers", "0",
    "--val-ratio", "0.1", "--seed", "20260917", "--sample-rate", "32000",
    "--duration", "10.0", "--threshold", "0.5", "--clip-loss-weight", "0.5",
    "--full-eval-every", "5", "--mixup-alpha", "0.2", "--mixup-prob", "0.5",
    "--mixup-label", "hard", "--adaptive-postproc",
]

LUOT = [("panns_ft_v4", 3), ("panns_ft_tpb2", 2)]
MOI_RUN = ["panns_ft_v3", "panns_ft_pw1", "panns_ft_v4", "panns_ft_tpb2"]


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
    """Đủ số epoch trong manifest mới tính là xong — không dựa vào dòng cuối của log."""
    m = _json(run_dir / "manifest.json")
    if not m:
        return False
    return len(m.get("timing", {}).get("epochs", [])) >= so_epoch


def doc_long_event(ana: dict) -> dict:
    """Tỉ lệ phân mảnh + F1 của lát cắt `long_event`, từ khối `lat_cat` của error_analysis.

    `n_ref=0` hoặc thiếu hẳn khối `lat_cat` (run chưa chạy error_analysis) phải trả
    `ti_le=None` — CHƯA ĐO, không phải chia cho 0 và không phải phân mảnh bằng 0.
    """
    for h in ana.get("lat_cat", []):
        if h.get("lat_cat") == "long_event":
            n_ref = h.get("n_ref")
            phan_manh = h.get("phan_manh")
            ti_le = (phan_manh / n_ref) if n_ref else None
            return {"phan_manh": phan_manh, "n_ref": n_ref, "ti_le": ti_le,
                    "event_f1": h.get("event_f1")}
    return {"phan_manh": None, "n_ref": None, "ti_le": None, "event_f1": None}


def doc_ket_qua(run_dir: Path) -> dict:
    """Gom số của một run từ ba file json. Thiếu file → None, KHÔNG phải 0."""
    cal = _json(run_dir / "calibration.json") or {}
    ana = _json(run_dir / "analysis.json") or {}
    man = _json(run_dir / "manifest.json") or {}
    args = man.get("config", {}).get("args", {})
    return {
        "time_pool_blocks": args.get("time_pool_blocks"),
        "pos_weight_max": args.get("pos_weight_max"),
        "ece_tong": cal.get("ece_tong"),
        "theta_sao": ana.get("theta_sao"),
        "f1_sao": ana.get("f1_sao"),
        "f1_05": ana.get("f1_05"),
        "long_event": doc_long_event(ana),
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


def xay_lenh_train(ten: str, time_pool_blocks: int, *, tiep_tuc: bool) -> list[str]:
    """`--pos-weight-max` khoá cứng 1.0; chỉ `--time-pool-blocks` đổi theo lượt."""
    lenh = ["-m", "ml.training.train_sed", *CAU_HINH_BASE,
            "--pos-weight-max", POS_WEIGHT_MAX,
            "--time-pool-blocks", str(time_pool_blocks), "--name", ten]
    if tiep_tuc:
        lenh.append("--resume")
    return lenh


def khoi_dong_train(ten: str, time_pool_blocks: int, tiep_tuc: bool) -> subprocess.Popen:
    log = RUNS_DIR / f"train_{ten.replace('panns_ft_', '')}.log"
    lenh = xay_lenh_train(ten, time_pool_blocks, tiep_tuc=tiep_tuc)
    ghi(f"▶ train {ten} (time_pool_blocks={time_pool_blocks}"
        f"{', --resume' if tiep_tuc else ''})")
    f = log.open("a", encoding="utf-8")
    return subprocess.Popen([str(PYTHON), *lenh], stdout=f, stderr=subprocess.STDOUT,
                            cwd=REPO_ROOT)


def cho_train(ten: str, time_pool_blocks: int) -> bool:
    """Chờ một lượt train xong; khởi động lại bằng --resume nếu log đứng yên quá lâu."""
    run_dir = RUNS_DIR / ten
    log = RUNS_DIR / f"train_{ten.replace('panns_ft_', '')}.log"
    da_khoi_dong_lai = 0
    if not run_dir.exists() and not log.exists():
        khoi_dong_train(ten, time_pool_blocks, tiep_tuc=False)
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
        khoi_dong_train(ten, time_pool_blocks, tiep_tuc=True).wait()
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
    yield ("| run | time_pool_blocks | ECE tổng | θ\\* | event-F1 @θ\\* | "
           "long_event: tỉ lệ vỡ | long_event: F1 |")
    yield "|---|---:|---:|---:|---:|---:|---:|"
    for ten, k in ket_qua.items():
        tpb = "chưa đo" if k["time_pool_blocks"] is None else str(k["time_pool_blocks"])
        le = k["long_event"]
        yield (f"| `{ten}` | {tpb} | {so(k['ece_tong'])} | {so(k['theta_sao'])} "
               f"| {so(k['f1_sao'])} | {so(le['ti_le'])} | {so(le['event_f1'])} |")


def viet_bao_cao(ket_qua: dict[str, dict], duong_dan: Path) -> Path:
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    with duong_dan.open("w", encoding="utf-8") as f:
        for dong in dong_bao_cao(ket_qua):
            f.write(dong + "\n")
            f.flush()
    return duong_dan


def dong_bao_cao(ket_qua: dict[str, dict]) -> Iterator[str]:
    yield "# Ablation `time_pool_blocks` — số thô, chưa diễn giải — 2026-09-22"
    yield ""
    yield ("Sinh tự động bởi `scripts/chay_ablation_time_pool.py`. **Trang này chỉ có số**; "
           "phần diễn giải và đồng bộ tài liệu làm tay sau, vì nó cần phán đoán.")
    yield ""
    yield ("Cả hai lượt khoá `--pos-weight-max 1.0`, chỉ đổi `--time-pool-blocks`; cùng split "
           "train, chấm trên cùng `data/synthetic/dev`. `panns_ft_v4` (time_pool_blocks=3) có "
           "cấu hình GIỐNG HỆT `panns_ft_pw1`, chỉ khác tên — dùng để kiểm tái lập, tương tự "
           "cách `pw30` từng kiểm tái lập `v3`.")
    yield ""
    yield ("> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với train → thiên vị theo "
           "thiết kế, **ngang nhau** cho mọi lượt ở đây. θ\\* chọn trên chính tập đang chấm "
           "→ **chặn trên lạc quan** cho mọi lượt.")
    yield ""
    yield from bang_tom_tat(ket_qua)
    yield ""
    yield "## Câu hỏi trang này phải trả lời"
    yield ""
    yield "1. `panns_ft_v4` có tái hiện `panns_ft_pw1` trong biên độ tương tự `pw30`↔`v3` không?"
    yield ("2. Tỉ lệ vỡ `long_event` của `tpb2` (time_pool_blocks=2) có THẤP hơn `v4` "
           "(time_pool_blocks=3) không? Nếu có, trường tiếp nhận thời gian ngắn là một phần "
           "nguyên nhân thật. Nếu KHÔNG đổi hoặc đổi ngược, giả thuyết bị bác bỏ.")
    yield "3. F1 tổng của `tpb2` có đánh đổi lấy tỉ lệ vỡ thấp hơn không, hay thắng thuần?"
    yield ""


# ── Chạy ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ghi("═══ bắt đầu ablation time_pool_blocks (v4 + tpb2) ═══")
    for ten, time_pool_blocks in LUOT:
        if train_xong(RUNS_DIR / ten):
            ghi(f"↷ {ten} đã train đủ {SO_EPOCH} epoch, bỏ qua")
        elif not cho_train(ten, time_pool_blocks):
            ghi(f"❌ dừng lộ trình: {ten} không train xong")
            break
        if not danh_gia(ten):
            ghi(f"❌ dừng lộ trình: {ten} đánh giá thất bại")
            break

    co_mat = [r for r in MOI_RUN if (RUNS_DIR / r / "analysis.json").exists()]
    if len(co_mat) >= 2:
        chay("compare_runs", ["-m", "ml.tracking.compare_runs", "--runs", *co_mat,
                              "--out", str(MEASUREMENTS / "compare_runs_time_pool_20260922.md")],
             RUNS_DIR / "ablation_compare_time_pool.log")

    ket_qua = {r: doc_ket_qua(RUNS_DIR / r) for r in MOI_RUN}
    ra = viet_bao_cao(ket_qua, MEASUREMENTS / "time_pool_ablation_20260922.md")
    ghi(f"✓ {ra.relative_to(REPO_ROOT)}")
    for dong in bang_tom_tat(ket_qua):
        ghi(dong)
    ghi("═══ xong. Chưa commit, chưa sửa tài liệu — để làm tay sau. ═══")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
