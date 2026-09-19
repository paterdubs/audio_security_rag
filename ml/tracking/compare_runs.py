"""Bảng đối chiếu nhiều lần chạy — Pha 5 của docs/TRAINING_OPS_PLAN.md.

    .venv/Scripts/python.exe -m ml.tracking.compare_runs --runs panns_ft_v1 panns_ft_v2 panns_ft_v3
    .venv/Scripts/python.exe -m ml.tracking.compare_runs --runs panns_ft_v1 panns_ft_v3 --out docs/measurements/so_sanh.md

Bảng này đứng ở cuối chuỗi đo: người đọc tin thẳng vào cột "khác nhau ở đâu" mà không
mở lại manifest. Nên nó phải từ chối khai những gì nó không biết. Ba chỗ dễ khai sai,
mỗi chỗ có một test riêng trong `tests/test_ml_compare_runs.py`:

- **Vân tay dữ liệu `null`** (v1/v2 — audio legacy bị xoá 18/09) → `KHONG_RO`. `None ==
  None` không phải bằng chứng cùng dữ liệu. Cùng bài học với `run_manifest.doc_hop_dong`
  trả `KHONG_RO` thay vì `PASSED` khi thiếu báo cáo.
- **Manifest hồi cứu** → băm mã là mã ở thời điểm LẬP manifest, không phải lúc train.
  Ba manifest hiện có lập trong cùng một phút nên `tree_sha256` của chúng trùng nhau;
  đọc thành "cùng mã nguồn" là chứng thực một điều không ai biết.
- **`ket_qua_cuoi`** chấm trên val split riêng của từng run (`val_ratio=0.1` trên chính
  tập train của nó), nên KHÔNG so ngang được: số thật xếp v2 > v1 > v3, ngược hẳn dev
  chung. Có `analysis.json` (Pha 4, cùng `data/synthetic/dev`) thì dùng nó và in kèm θ.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from common import enable_utf8_output  # noqa: E402

enable_utf8_output()

RUNS_DIR = REPO_ROOT / "ml" / "runs"

KHONG_RO = "KHONG_RO"
THIEU = "<không có>"

# Khác nhau ở những khoá này là hiển nhiên, không phải phát hiện.
BO_QUA_KHOA = frozenset({"name"})


def doc_run(run_dir: Path) -> dict:
    """Đọc manifest + analysis.json (nếu có) của một run.

    Thiếu manifest thì NÉM, không bỏ qua: bảng in ra vẫn đẹp với ít cột hơn và không ai
    để ý một run đã biến mất khỏi phép so sánh.
    """
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{run_dir.name}: không có manifest.json — chạy "
                                f"-m ml.tracking.run_manifest --run {run_dir.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    analysis_path = run_dir / "analysis.json"
    return {
        "ten": manifest.get("run", run_dir.name),
        "manifest": manifest,
        "analysis": (json.loads(analysis_path.read_text(encoding="utf-8"))
                     if analysis_path.exists() else None),
    }


def _van_tay_cua(run: dict) -> str:
    van_tay = run["manifest"].get("data", {}).get("van_tay")
    if not van_tay:
        return KHONG_RO
    return van_tay.get("nhan_sha256", KHONG_RO)


def diff_van_tay(runs: list[dict]) -> dict:
    """Cùng dữ liệu hay không. Chỉ có KHÔNG khai `KHONG_RO` khi MỌI run đều khai vân tay.

    Biết một nửa không đủ để nói "khác nhau", càng không đủ để nói "giống nhau".
    """
    theo_run = {r["ten"]: _van_tay_cua(r) for r in runs}
    gia_tri = set(theo_run.values())
    if KHONG_RO in gia_tri:
        thieu = sorted(k for k, v in theo_run.items() if v == KHONG_RO)
        return {"ket_luan": KHONG_RO, "theo_run": theo_run,
                "ly_do": f"không có vân tay dữ liệu: {', '.join(thieu)} — "
                         "xem data.ghi_chu trong manifest"}
    return {"ket_luan": "giong" if len(gia_tri) == 1 else "khac", "theo_run": theo_run,
            "ly_do": ""}


def diff_ma(runs: list[dict]) -> dict:
    """Cùng mã nguồn hay không. Manifest hồi cứu → `KHONG_RO` dù băm có trùng."""
    theo_run = {r["ten"]: r["manifest"].get("code", {}).get("tree_sha256") or KHONG_RO
                for r in runs}
    hoi_cuu = sorted(r["ten"] for r in runs if r["manifest"].get("hoi_cuu"))
    if hoi_cuu:
        return {"ket_luan": KHONG_RO, "theo_run": theo_run,
                "ly_do": f"manifest lập hồi cứu ({', '.join(hoi_cuu)}): băm mã là mã ở "
                         "thời điểm lập manifest, không phải lúc train"}
    if KHONG_RO in set(theo_run.values()):
        return {"ket_luan": KHONG_RO, "theo_run": theo_run, "ly_do": "thiếu băm cây mã"}
    return {"ket_luan": "giong" if len(set(theo_run.values())) == 1 else "khac",
            "theo_run": theo_run, "ly_do": ""}


def diff_config(runs: list[dict]) -> list[dict]:
    """Các khoá cấu hình khác nhau giữa các run.

    `<không có>` KHÁC HẲN một giá trị khác: v1 không có khoá `mixup_alpha` vì cờ đó ra
    đời sau v1 — nghĩa là v1 chạy bằng một phiên bản code khác, chứ không phải ai đó
    chỉnh tham số. Gộp hai trường hợp là mất đúng thông tin quan trọng nhất.
    """
    args = {r["ten"]: r["manifest"].get("config", {}).get("args", {}) for r in runs}
    khoa = sorted({k for a in args.values() for k in a} - BO_QUA_KHOA)
    hang = []
    for k in khoa:
        gia_tri = {ten: (THIEU if k not in a else str(a[k])) for ten, a in args.items()}
        if len(set(gia_tri.values())) > 1:
            hang.append({"khoa": k, "gia_tri": gia_tri,
                         "co_thieu": THIEU in gia_tri.values()})
    return hang


def bang_metric(runs: list[dict]) -> dict:
    """Bảng metric. Ưu tiên `analysis.json` (dev chung) — chỉ khi MỌI run đều có.

    Trộn dev chung cho run này với val riêng cho run kia là so hai thứ khác nhau, nên
    thiếu một run là hạ cả bảng xuống `val_rieng` và tắt cờ `so_ngang_duoc`.
    """
    if all(r["analysis"] for r in runs):
        a0 = runs[0]["analysis"]
        return {
            "nguon": "dev_chung", "so_ngang_duoc": True,
            "canh_bao": (f"chấm trên cùng {a0.get('split', 'dev')}/{a0.get('subset', 'all')}; "
                         "θ* chọn trên chính tập đang chấm → chặn trên lạc quan"),
            "hang": [{
                "run": r["ten"],
                "theta_sao": r["analysis"]["theta_sao"],
                "event_f1_sao": r["analysis"]["f1_sao"],
                "segment_f1_sao": r["analysis"]["seg_sao"],
                "event_f1_05": r["analysis"]["f1_05"],
            } for r in runs],
        }
    thieu = sorted(r["ten"] for r in runs if not r["analysis"])
    return {
        "nguon": "val_rieng", "so_ngang_duoc": False,
        "canh_bao": ("lấy từ manifest.ket_qua_cuoi — chấm trên val split RIÊNG của từng "
                     "run (val_ratio trên chính tập train của nó), KHÔNG so ngang được. "
                     f"Chưa có analysis.json: {', '.join(thieu)}"),
        "hang": [{
            "run": r["ten"],
            **{k: r["manifest"].get("ket_qua_cuoi", {}).get(k)
               for k in ("clip_map", "segment_f1", "event_f1")},
        } for r in runs],
    }


# ── Báo cáo ──────────────────────────────────────────────────────────────────


def _bang(tieu_de: list[str], hang: list[list[str]]) -> Iterator[str]:
    yield "| " + " | ".join(tieu_de) + " |"
    yield "|" + "|".join("---" for _ in tieu_de) + "|"
    for r in hang:
        yield "| " + " | ".join(r) + " |"


def _so(x, n: int = 4) -> str:
    return "—" if x is None else f"{x:.{n}f}".replace(".", ",")


def dong_bao_cao(runs: list[dict], metric: dict, config: list[dict],
                 van_tay: dict, ma: dict) -> Iterator[str]:
    ten = [r["ten"] for r in runs]
    yield f"# Đối chiếu {len(ten)} lần chạy: {' · '.join(ten)}"
    yield ""
    yield "Sinh bởi `ml/tracking/compare_runs.py` (Pha 5). Mỗi con số dưới đây đi kèm chỗ nó đến từ."
    yield ""

    yield "## 1. Metric"
    yield ""
    if metric["so_ngang_duoc"]:
        yield f"Nguồn: `analysis.json` của từng run — {metric['canh_bao']}."
        yield ""
        yield from _bang(
            ["run", "θ*", "F1(sự kiện) @θ*", "F1(đoạn 1s) @θ*", "F1(sự kiện) @θ=0,50"],
            [[h["run"], _so(h["theta_sao"], 2), _so(h["event_f1_sao"]),
              _so(h["segment_f1_sao"]), _so(h["event_f1_05"])] for h in metric["hang"]],
        )
    else:
        yield f"> ⚠️ **Không so ngang được.** {metric['canh_bao']}"
        yield ""
        yield from _bang(
            ["run", "mAP(clip)", "F1(đoạn 1s)", "F1(sự kiện)"],
            [[h["run"], _so(h["clip_map"]), _so(h["segment_f1"]), _so(h["event_f1"])]
             for h in metric["hang"]],
        )
    yield ""

    yield "## 2. Dữ liệu"
    yield ""
    if van_tay["ket_luan"] == KHONG_RO:
        yield (f"> ⚠️ **{KHONG_RO}** — {van_tay['ly_do']}. Không kết luận được hai run "
               "này cùng hay khác dữ liệu, nên mọi chênh lệch metric ở §1 **chưa quy được "
               "cho mô hình**.")
    else:
        yield (f"Vân tay nhãn {'TRÙNG' if van_tay['ket_luan'] == 'giong' else 'KHÁC'} "
               "nhau giữa các run.")
    yield ""
    yield from _bang(["run", "vân tay nhãn (sha256)"],
                     [[k, v if v == KHONG_RO else f"`{v[:16]}…`"]
                      for k, v in van_tay["theo_run"].items()])
    yield ""

    yield "## 3. Mã nguồn"
    yield ""
    if ma["ket_luan"] == KHONG_RO:
        yield f"> ⚠️ **{KHONG_RO}** — {ma['ly_do']}."
    else:
        yield f"Băm cây mã {'trùng' if ma['ket_luan'] == 'giong' else 'khác'} nhau."
    yield ""

    yield "## 4. Cấu hình"
    yield ""
    if not config:
        yield ("**Không có khác biệt nào ĐƯỢC GHI LẠI** trong `config.args` (bỏ qua "
               "`name`). Đó không phải bằng chứng hai run giống nhau — thứ phân biệt "
               "chúng có thể nằm ở dữ liệu, xem §2.")
    else:
        yield (f"`{THIEU}` nghĩa là khoá đó **không tồn tại** trong manifest của run — "
               "thường vì cờ ra đời sau run đó, tức run chạy bằng một phiên bản code "
               "khác. Đó là chuyện khác hẳn với chỉnh tham số.")
        yield ""
        # Bọc backtick khi in: `<không có>` trần bị trình render markdown nuốt như thẻ HTML.
        yield from _bang(["khoá", *ten],
                         [[h["khoa"], *(f"`{h['gia_tri'][t]}`" if h["gia_tri"][t] == THIEU
                                        else h["gia_tri"][t] for t in ten)]
                          for h in config])
    yield ""


def viet_dan(path: Path, dong: Iterator[str]) -> Path:
    """Ghi từng dòng + flush — không dồn tới cuối hàm (bài học 372 dòng segments.jsonl)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in dong:
            f.write(d + "\n")
            f.flush()
    return path


def run(args: argparse.Namespace) -> int:
    if len(args.runs) < 2:
        print("❌ cần ít nhất hai run để đối chiếu")
        return 1
    runs = [doc_run(RUNS_DIR / ten) for ten in args.runs]

    metric = bang_metric(runs)
    config = diff_config(runs)
    van_tay = diff_van_tay(runs)
    ma = diff_ma(runs)

    dong = list(dong_bao_cao(runs, metric, config, van_tay, ma))
    if args.out:
        out = viet_dan(Path(args.out) if Path(args.out).is_absolute()
                       else REPO_ROOT / args.out, iter(dong))
        print(f"✓ {out.relative_to(REPO_ROOT)}")
    else:
        print("\n".join(dong))

    print(f"\ndữ liệu: {van_tay['ket_luan']} · mã: {ma['ket_luan']} · "
          f"metric: {metric['nguon']} · {len(config)} khoá cấu hình khác nhau", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="tên thư mục trong ml/runs/")
    parser.add_argument("--out", default=None, help="ghi .md thay vì in ra stdout")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
