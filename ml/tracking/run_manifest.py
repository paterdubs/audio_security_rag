"""Sổ ghi lần chạy — Pha 1 của docs/TRAINING_OPS_PLAN.md.

    # lập manifest hồi cứu cho một run đã xong
    .venv/Scripts/python.exe -m ml.tracking.run_manifest --run panns_ft_v3 --du-lieu hien-tai
    .venv/Scripts/python.exe -m ml.tracking.run_manifest --run panns_ft_v1 --du-lieu khong-con

Manifest trả lời đúng một câu: *lần chạy này đã dùng mã nào, dữ liệu nào, cấu hình nào?*
Không có nó thì mọi so sánh giữa hai run đều là suy đoán — và dự án đã trả giá: v1 ghi
`time_pool_blocks: null, mixup_alpha: null` chỉ vì hai cờ đó ra đời sau v1, nên bảng
v1-vs-v2 không tự chứng minh được là so sánh sạch.

**Hồi cứu không bịa.** Trường nào không khôi phục được thì để `null` kèm `ghi_chu`, không
điền giá trị đoán. Cụ thể: audio legacy của v1/v2 đã bị xoá ngày 18/09, nên vân tay dữ
liệu của hai run đó là `null` vĩnh viễn — ghi `khong-con` là lời khai đúng, còn băm dữ
liệu hiện tại rồi gán cho v1 là lời khai sai.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from common import enable_utf8_output  # noqa: E402

enable_utf8_output()

from ml.tracking.fingerprint import (  # noqa: E402
    bam_file,
    moi_truong,
    noi_dung_config,
    trang_thai_git,
    van_tay_du_lieu,
    van_tay_ma,
)

RUNS_DIR = REPO_ROOT / "ml" / "runs"
FEATURES_DIR = REPO_ROOT / "data" / "features"
CONFIGS_DIR = REPO_ROOT / "ml" / "configs"
CONTRACT_PATH = REPO_ROOT / "data" / "manifests" / "synthetic_contract.csv"

# Config ảnh hưởng tới Ý NGHĨA của nhãn và của chỉ số lớp. Đổi bất kỳ file nào trong đây
# là đổi bài toán, không phải đổi siêu tham số.
CONFIG_FILES = ("ontology_map.yaml", "scaper_train.yaml", "preprocessing.yaml")

MANIFEST_VERSION = 1


def duong_dan_ngan(path: Path) -> str:
    """Duong dan tuong doi so voi repo khi co the; tuyet doi khi nam ngoai repo.

    `Path.relative_to` nem ValueError voi duong dan ngoai repo — da lam vo test hai lan
    trong cung mot phien, vi tmp_path cua pytest nam o o C con repo o o D.
    """
    # `resolve()` truoc khi so: duong dan tuong doi tu CLI khong bao gio
    # `is_relative_to(REPO_ROOT)`, nen thieu buoc nay thi manifest ghi duong dan Windows
    # co dau nguoc, khong doc duoc tren may khac.
    tuyet_doi = path.resolve()
    return (tuyet_doi.relative_to(REPO_ROOT).as_posix()
            if tuyet_doi.is_relative_to(REPO_ROOT) else str(tuyet_doi))


def doc_hop_dong(report_path: Path = CONTRACT_PATH) -> dict:
    """Kết quả Pha 2 tại thời điểm train, đọc từ chính file báo cáo của verifier.

    Trả `KHONG_RO` chứ không trả `PASSED` khi thiếu file: "chưa ai kiểm" và "đã kiểm, sạch"
    là hai trạng thái khác nhau, và gộp chúng lại chính là cách một lô dữ liệu hỏng đi
    lọt vào train mà manifest vẫn xanh.
    """
    if not report_path.exists():
        return {"ket_qua": "KHONG_RO", "ghi_chu": f"không thấy {report_path.name}"}
    with report_path.open(encoding="utf-8", newline="") as handle:
        n_vi_pham = sum(1 for _ in csv.DictReader(handle))
    return {
        "ket_qua": "PASSED" if n_vi_pham == 0 else "FAILED",
        "n_vi_pham": n_vi_pham,
        "bao_cao": duong_dan_ngan(report_path),
        "sha256": bam_file(report_path),
        "mtime": datetime.fromtimestamp(report_path.stat().st_mtime, UTC).isoformat(),
    }


def tao_manifest(run_name: str, args: dict, *, split: str,
                 van_tay: dict | None, ghi_chu_du_lieu: str | None = None,
                 chia_tap: dict | None = None, seeds: dict | None = None,
                 hoi_cuu: bool = False, pip_freeze: bool = True,
                 contract_path: Path = CONTRACT_PATH) -> dict:
    """Gom mọi thứ cần để tái lập một lần chạy thành một dict ghi được ra JSON."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "run": run_name,
        "tao_luc": datetime.now(UTC).isoformat(timespec="seconds"),
        "hoi_cuu": hoi_cuu,
        "config": {
            "args": args,
            "seeds": seeds,
            "files": noi_dung_config([CONFIGS_DIR / f for f in CONFIG_FILES], REPO_ROOT),
        },
        "data": {
            "split": split,
            "van_tay": van_tay,
            "ghi_chu": ghi_chu_du_lieu,
            "chia_tap": chia_tap,
            "hop_dong": doc_hop_dong(contract_path),
        },
        "code": van_tay_ma(REPO_ROOT),
        "git": trang_thai_git(REPO_ROOT),
        "env": moi_truong(pip_freeze=pip_freeze),
        "timing": {"bat_dau": None, "ket_thuc": None, "epochs": []},
    }


def ghi(run_dir: Path, manifest: dict) -> Path:
    """Ghi manifest NGAY, không dồn tới cuối run.

    Cùng bài học với `fetch_audioset_strong.append_one_segment`: lần trước dồn ghi tới
    cuối `run()` đã để lại 375 file .wav trên đĩa với đúng 3 dòng metadata. Một run train
    25 epoch bị ngắt ở epoch 18 mà không có manifest thì checkpoint còn đó cũng vô dụng.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, default=str)
        handle.flush()
    return path


def ghi_epoch(run_dir: Path, manifest: dict, ban_ghi: dict) -> None:
    """Thêm một dòng timing rồi ghi lại toàn bộ manifest."""
    manifest["timing"]["epochs"].append(ban_ghi)
    manifest["timing"]["ket_thuc"] = ban_ghi.get("ket_thuc")
    ghi(run_dir, manifest)


# ── Hồi cứu cho v1/v2/v3 ─────────────────────────────────────────────────────

# v1/v2 train trên lô legacy; audio của lô đó đã bị xoá trong lượt dọn 18/09 (giữ lại
# JAMS làm bằng chứng). Không có cách nào băm lại waveform đã không còn tồn tại.
GHI_CHU_KHONG_CON = (
    "Audio của lô legacy đã bị xoá ngày 18/09/2026 (giữ 9.360 JAMS + slice_index làm bằng "
    "chứng trước-sửa). Vân tay dữ liệu KHÔNG khôi phục được và không được thay bằng vân "
    "tay của lô B0–B9 hiện tại — đó là một lô dữ liệu khác."
)
GHI_CHU_HIEN_TAI = (
    "Vân tay băm từ data/features hiện tại. Hợp lệ khi và chỉ khi cache chưa bị sinh lại "
    "kể từ lần train; manifest hồi cứu không tự chứng minh được điều đó."
)


# Sentinel cho "không có báo cáo hợp đồng nào thuộc về run này". Dùng một đường dẫn
# chắc chắn không tồn tại thay vì None để `doc_hop_dong` chỉ có MỘT đường xử lý.
KHONG_CO_HOP_DONG = Path("<khong-co-bao-cao-hop-dong-cho-run-nay>")


def hoi_cuu(run_name: str, nguon_du_lieu: str, pip_freeze: bool = True,
            hop_dong_path: Path | None = None) -> int:
    run_dir = RUNS_DIR / run_name
    history_path = run_dir / "history.json"
    if not history_path.exists():
        print(f"❌ không thấy {history_path}")
        return 1

    history = json.loads(history_path.read_text(encoding="utf-8"))
    args = history.get("args", {})
    split = args.get("split", "train")

    # Hợp đồng dữ liệu phải thuộc về CHÍNH lô đã train. Mặc định đọc file chung
    # `synthetic_contract.csv` là một cái bẫy: nó hiện ghi lô B0–B9 PASSED, nên một
    # manifest hồi cứu cho v1/v2 (train trên lô legacy đã FAILED) sẽ khai PASSED — đúng
    # loại lời khai sai mà Pha 1 sinh ra để loại bỏ. Lô legacy có báo cáo riêng:
    # data/manifests/synthetic_contract_train_20260917.csv.
    if hop_dong_path is None:
        hop_dong_path = CONTRACT_PATH if nguon_du_lieu == "hien-tai" else KHONG_CO_HOP_DONG

    van_tay = ghi_chu = None
    if nguon_du_lieu == "hien-tai":
        print("▶ băm waveform hiện tại (5 GB — mất vài chục giây)…", flush=True)
        van_tay = van_tay_du_lieu(FEATURES_DIR, split,
                                  sample_rate=int(args.get("sample_rate", 32000)))
        ghi_chu = GHI_CHU_HIEN_TAI
    else:
        ghi_chu = GHI_CHU_KHONG_CON

    manifest = tao_manifest(
        run_name, args, split=split, van_tay=van_tay, ghi_chu_du_lieu=ghi_chu,
        # `--seed` của train_sed chỉ điều khiển phép chia train/val; các seed RNG khác
        # chưa từng được đặt ở v1/v2/v3, nên khai `null` thay vì chép lại con số đó.
        seeds={"split": args.get("seed"), "python": None, "numpy": None, "torch": None},
        hoi_cuu=True, pip_freeze=pip_freeze, contract_path=hop_dong_path,
    )
    manifest["timing"]["epochs"] = [
        {"epoch": e["epoch"], "seconds": e.get("seconds")} for e in history.get("history", [])
    ]
    manifest["ket_qua_cuoi"] = history.get("final")
    manifest["canh_bao"] = [
        "Manifest lập HỒI CỨU: code/git/env băm ở thời điểm lập, KHÔNG phải thời điểm train.",
    ]
    if nguon_du_lieu != "hien-tai":
        manifest["canh_bao"].append("Vân tay dữ liệu null — xem data.ghi_chu.")
    if manifest["data"]["hop_dong"]["ket_qua"] == "KHONG_RO":
        manifest["canh_bao"].append(
            "Hợp đồng dữ liệu KHONG_RO: chưa trỏ tới báo cáo verify_synthetic của chính lô "
            "đã train. Dùng --hop-dong <đường dẫn> để gắn bằng chứng đúng."
        )

    path = ghi(run_dir, manifest)
    print(f"✓ {duong_dan_ngan(path)}")
    print(f"  hợp đồng dữ liệu: {manifest['data']['hop_dong']['ket_qua']}")
    print(f"  vân tay dữ liệu:  {'có' if van_tay else 'null — ' + (ghi_chu or '')[:60]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="tên thư mục trong ml/runs/")
    parser.add_argument("--du-lieu", choices=["hien-tai", "khong-con"], required=True,
                        help="'hien-tai' băm data/features hiện có; 'khong-con' ghi null "
                             "cho run mà dữ liệu đã bị xoá (v1/v2 legacy)")
    parser.add_argument("--hop-dong", type=Path,
                        help="báo cáo verify_synthetic CỦA CHÍNH lô đã train; thiếu thì "
                             "manifest ghi KHONG_RO chứ không mượn báo cáo của lô khác")
    parser.add_argument("--khong-pip-freeze", action="store_true",
                        help="bỏ pip freeze cho nhanh khi chạy thử")
    args = parser.parse_args()
    return hoi_cuu(args.run, args.du_lieu, pip_freeze=not args.khong_pip_freeze,
                   hop_dong_path=args.hop_dong)


if __name__ == "__main__":
    raise SystemExit(main())
