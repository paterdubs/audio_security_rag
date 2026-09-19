"""Phân tích lỗi trên dự đoán đã lưu — Pha 4 của docs/TRAINING_OPS_PLAN.md.

    .venv/Scripts/python.exe -m ml.evaluation.error_analysis --run panns_ft_v3 --split dev

Không train lại, không cần GPU: đọc `ml/runs/{run}/predictions/{split}_{subset}.npz`.

Pha 3 trả lời "ngưỡng nào tốt nhất". Pha 4 trả lời câu còn lại: **hỏng ở đâu**. Bốn phần:

  1. Sáu loại lỗi — Insertion, Deletion, Substitution, Boundary, Fragmentation, Merge.
  2. Bảng theo lát cắt — overlap / low_snr / reverb / long_event / causal_chain / sạch.
  3. Ma trận nhầm lẫn, đối chiếu `confusable_with` đã khai trong ontology_map.yaml.
  4. Đường cong ngưỡng theo TỪNG lớp.

⚠️ Hai cảnh báo phải đi kèm mọi con số ra từ đây, không được rút gọn:

  · `data/synthetic/dev` sinh CÙNG RECIPE B0–B9 với tập train của v3, nên dev thiên vị
    v3 theo thiết kế. So sánh v1/v2/v3 trên nó không phải bằng chứng v3 tốt hơn.
  · θ* và θ theo từng lớp đều chọn TRÊN CHÍNH TẬP ĐANG CHẤM. Đó là rò rỉ; số đó là chặn
    trên lạc quan, không phải kết quả báo cáo được.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from common import enable_utf8_output  # noqa: E402

enable_utf8_output()

from ml.datasets.synthetic_sed import BACKGROUND_CLASS  # noqa: E402
from ml.evaluation.error_taxonomy import (  # noqa: E402
    LOAI_TANG_MOT,
    MOI_LOAI,
    ma_tran_nham,
    nham_theo_cap,
    phan_loai,
)
from ml.evaluation.predictions import doc  # noqa: E402
from ml.evaluation.sed_metrics import (  # noqa: E402
    ONSET_COLLAR_SEC,
    event_and_segment_f1,
    frames_to_events,
    thong_ke_su_kien,
)
from ml.evaluation.threshold_sweep import cua_so_loc, luoi_nguong, quet  # noqa: E402

RUNS_DIR = REPO_ROOT / "ml" / "runs"
DATA_DIR = REPO_ROOT / "data" / "synthetic"
ONTOLOGY_PATH = REPO_ROOT / "ml" / "configs" / "ontology_map.yaml"

# Clip không thuộc lát cắt khó nào. 357/1.440 clip dev rơi vào đây và chúng là MỐC ĐỐI
# CHIẾU: không có cột này thì năm lát cắt khó không so được với cái gì.
LAT_CAT_SACH = "sach"
LAT_CAT = ("overlap", "low_snr", "reverb", "long_event", "causal_chain", LAT_CAT_SACH)

NHAN_LOAI = {
    "dung": "Đúng", "bien": "Boundary", "thay_the": "Substitution",
    "thieu": "Deletion", "thua": "Insertion",
}


def so(x: float, n: int = 2) -> str:
    """Số thập phân kiểu Việt. Báo cáo viết cho người đọc tiếng Việt; trộn `0,50` với
    `0.90` trong cùng một bảng là lỗi trình bày đã lọt ra ở lượt chạy đầu."""
    return f"{x:.{n}f}".replace(".", ",")


def nhan_theta(x: float) -> str:
    return f"θ = {so(x)}"


# ── Đọc dữ liệu phụ trợ ──────────────────────────────────────────────────────


def doc_lat_cat(path: Path) -> dict[str, list[str]]:
    """slice_index.jsonl → {clip_id: [lát cắt]}. Clip không lát cắt nào nhận nhãn `sach`."""
    lat_cat: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ban_ghi = json.loads(line)
            slices = list(ban_ghi.get("slices") or [])
            lat_cat[ban_ghi["clip_id"]] = slices or [LAT_CAT_SACH]
    return lat_cat


def nhom_theo_lat_cat(lat_cat: dict[str, list[str]],
                      clip_ids: Iterable[str]) -> dict[str, list[str]]:
    """{lát cắt: [clip_id]} — CHỈ gồm clip có mặt trong file dự đoán.

    slice_index.jsonl là của TOÀN split; file dự đoán có thể chỉ chứa một subset. Lấy
    nhầm clip ngoài subset vào bảng làm mẫu số sai mà không có lỗi nào bắn ra.

    ⚠️ Lát cắt CHỒNG NHAU: một clip vừa overlap vừa reverb nằm ở cả hai hàng. Trên dev
    thật tổng các hàng là 2.088 trên 1.440 clip.
    """
    trong_du_doan = list(clip_ids)
    them = sorted({s for slices in lat_cat.values() for s in slices} - set(LAT_CAT))
    nhom: dict[str, list[str]] = {ten: [] for ten in (*LAT_CAT, *them)}
    for clip_id in trong_du_doan:
        for ten in lat_cat.get(clip_id, [LAT_CAT_SACH]):
            nhom[ten].append(clip_id)
    return nhom


def doc_confusable(path: Path) -> dict[str, list[str]]:
    """{lớp: [lớp dễ nhầm]} từ ontology_map.yaml — nguồn chân lý duy nhất, không hardcode."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        name: list(spec.get("confusable_with") or [])
        for name, spec in config["classes"].items()
        if name != BACKGROUND_CLASS
    }


def viet_dan(path: Path, dong: Iterable[str]) -> Path:
    """Ghi từng dòng + flush. KHÔNG dồn tới cuối hàm.

    Dồn output rồi mới ghi một lượt đã làm mất 372 dòng `segments.jsonl` khi tiến trình
    chết giữa chừng: cái đã tính xong nằm trong RAM và biến mất cùng tiến trình.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for line in dong:
            f.write(line + "\n")
            f.flush()
    return path


# ── Ngưỡng ───────────────────────────────────────────────────────────────────


def nguong_toi_uu_theo_lop(ket_qua_quet: list[dict],
                           class_ids: list[str]) -> dict[str, dict]:
    """θ* riêng cho TỪNG lớp = argmax F1 sự kiện của chính lớp đó.

    Hợp lệ vì class-wise metrics của sed_eval tính độc lập theo lớp: đổi ngưỡng của lớp
    A không đụng đến TP/FP/FN của lớp B. Lớp không xuất hiện ở bảng nào (không có sự
    kiện tham chiếu) rơi về F1 = 0 — đó là "chưa đo được", không phải "model dở".
    """
    tot: dict[str, dict] = {}
    for class_id in class_ids:
        diem = [(r["event_f1_theo_lop"].get(class_id, 0.0), r["threshold"])
                for r in ket_qua_quet]
        f1, nguong = max(diem) if diem else (0.0, float("nan"))
        tot[class_id] = {"threshold": nguong, "event_f1": f1}
    return tot


def vector_nguong(class_ids: list[str], tot: dict[str, dict], mac_dinh: float) -> np.ndarray:
    """(C,) ngưỡng riêng từng lớp. `frames_to_events` broadcast thẳng mảng này."""
    return np.array([
        tot[c]["threshold"] if np.isfinite(tot[c]["threshold"]) else mac_dinh
        for c in class_ids
    ], dtype=np.float32)


def giai_ma(du_doan, khung: list[np.ndarray], nguong, median_size) -> dict[str, list[dict]]:
    return {
        clip_id: frames_to_events(khung[i], du_doan.class_ids, du_doan.duration,
                                  nguong, median_size)
        for i, clip_id in enumerate(du_doan.clip_ids)
    }


# ── Các bảng ─────────────────────────────────────────────────────────────────


def bang_loi(tham_chieu, du_bao, class_ids) -> dict:
    """Sáu loại lỗi + đối chiếu với S/D/I của sed_eval trên cùng bộ dự báo."""
    kq = phan_loai(tham_chieu, du_bao, ONSET_COLLAR_SEC)
    sed = thong_ke_su_kien(tham_chieu, du_bao, class_ids)
    n_tp_sed = sum(v["n_tp"] for v in sed["theo_lop"].values())
    return {
        "n_ref": kq.n_ref, "n_pred": kq.n_pred,
        "dem": {k: int(kq.dem[k]) for k in ("dung", *LOAI_TANG_MOT[1:], "thua")},
        "ty_le": {k: round(v, 4) for k, v in kq.ty_le().items()},
        "phan_manh": kq.n_phan_manh, "gop": kq.n_gop,
        "sed_eval": {k: sed[k] for k in
                     ("n_ref", "n_sys", "error_rate", "substitution", "deletion", "insertion")},
        "doi_chieu_dung_vs_ntp": {
            "ghep_greedy": int(kq.dem["dung"]), "sed_eval_ntp": n_tp_sed,
            "lech": int(kq.dem["dung"]) - n_tp_sed,
        },
        "_kq": kq,
    }


DEM_RONG = dict.fromkeys(MOI_LOAI, 0)


def ten_bao_cao(thich_ung: bool) -> str:
    """Tên file báo cáo. Hai cấu hình hậu xử lý phải ra hai file — ghi đè lên nhau thì
    lượt ablation sau xoá mất lượt trước mà không báo gì."""
    return "analysis_adaptive" if thich_ung else "analysis"


def bang_lat_cat(tham_chieu, du_bao, class_ids, duration, nhom) -> list[dict]:
    """F1 + phân loại 6 loại lỗi riêng cho từng lát cắt.

    Chấm lại sed_eval VÀ phân loại lại trên tập con clip, không nội suy từ tổng: lát cắt
    chồng nhau nên một clip đóng góp lỗi cho nhiều hàng, nhân tỉ lệ với tổng toàn tập sẽ
    ra số nhỏ hơn thật mà không có lỗi nào bắn ra.
    """
    hang = []
    for ten, clip_ids in nhom.items():
        if not clip_ids:
            hang.append({"lat_cat": ten, "n_clip": 0, "n_ref": 0, "n_pred": 0,
                         "segment_f1": float("nan"), "event_f1": float("nan"),
                         "dem": dict(DEM_RONG), "phan_manh": 0, "gop": 0})
            continue
        ref_con = {c: tham_chieu.get(c, []) for c in clip_ids}
        pred_con = {c: du_bao.get(c, []) for c in clip_ids}
        segment_f1, event_f1, _ = event_and_segment_f1(ref_con, pred_con, class_ids, duration)
        kq = phan_loai(ref_con, pred_con, ONSET_COLLAR_SEC)
        hang.append({
            "lat_cat": ten, "n_clip": len(clip_ids),
            "n_ref": sum(len(v) for v in ref_con.values()),
            "n_pred": sum(len(v) for v in pred_con.values()),
            "segment_f1": segment_f1, "event_f1": event_f1,
            "dem": {k: int(kq.dem[k]) for k in MOI_LOAI},
            "phan_manh": kq.n_phan_manh, "gop": kq.n_gop,
        })
        print(f"  {ten:<14}{len(clip_ids):>6} clip  F1(sự kiện)={event_f1:.4f}", flush=True)
    return hang


# ── Báo cáo ──────────────────────────────────────────────────────────────────


def _bang(tieu_de: list[str], hang: list[list[str]]) -> Iterator[str]:
    yield "| " + " | ".join(tieu_de) + " |"
    yield "|" + "|".join("---" for _ in tieu_de) + "|"
    for r in hang:
        yield "| " + " | ".join(r) + " |"


def dong_bao_cao(p: dict) -> Iterator[str]:
    """Sinh báo cáo .md từng dòng một — `viet_dan` ghi tới đâu flush tới đó."""
    yield f"# Phân tích lỗi — {p['run']} trên {p['split']}/{p['subset']}"
    yield ""
    yield (f"Sinh bởi `ml/evaluation/error_analysis.py` từ `{p['nguon']}` "
           f"({p['n_clip']} clip, {p['n_ref']} sự kiện tham chiếu). "
           "Không train lại, không dùng GPU.")
    yield ""
    yield ("> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với tập train của `v3`, "
           "nên dev **thiên vị v3 theo thiết kế**. Mọi so sánh v1/v2/v3 đọc trên trang này "
           "đều mang thiên vị đó.")
    yield ("> ⚠️ θ\\* và θ theo từng lớp đều chọn **trên chính tập đang chấm** → rò rỉ. "
           "Chúng là chặn trên lạc quan, không phải kết quả báo cáo được.")
    yield ""

    yield "## 1. Sáu loại lỗi"
    yield ""
    yield ("Tầng 1 gán cho mỗi sự kiện tham chiếu **đúng một** nhãn "
           "(Đúng/Boundary/Substitution/Deletion); dự báo không được tiêu thụ là Insertion. "
           "Fragmentation và Merge là **tầng 2**, đếm riêng và có thể chồng lên tầng 1.")
    yield ""
    yield (f"Collar onset {ONSET_COLLAR_SEC * 1000:.0f} ms, chỉ khớp onset "
           "(offset của tiếng vang quá mơ hồ để chấm).")
    yield ""
    for ten, b in p["loi"].items():
        yield f"**{ten}** — {b['n_ref']} sự kiện thật, {b['n_pred']} sự kiện dự báo"
        yield ""
        yield from _bang(
            ["loại", "số", "tỉ lệ / sự kiện thật"],
            [[NHAN_LOAI[k], f"{b['dem'][k]}", so(b["ty_le"][k], 3)] for k in NHAN_LOAI]
            + [["Fragmentation (1 thật → n đoán)", f"{b['phan_manh']}", "—"],
               ["Merge (n thật → 1 đoán)", f"{b['gop']}", "—"]],
        )
        yield ""
    yield ("**Boundary không phải một rổ độc lập với sed_eval.** Một dự báo đúng lớp nhưng "
           "lệch onset quá collar bị sed_eval tính **một Deletion + một Insertion**. Bảng "
           "trên tách nó ra vì hai lỗi cần hai cách sửa khác nhau: lệch biên sửa ở hậu xử "
           "lý, còn không-nghe-ra-gì phải sửa ở dữ liệu. Số S/D/I gốc của sed_eval:")
    yield ""
    yield from _bang(
        ["θ", "Nref", "Nsys", "Substitution", "Deletion", "Insertion", "error rate"],
        [[ten, f"{b['sed_eval']['n_ref']}", f"{b['sed_eval']['n_sys']}",
          f"{b['sed_eval']['substitution']}", f"{b['sed_eval']['deletion']}",
          f"{b['sed_eval']['insertion']}", so(b["sed_eval"]["error_rate"], 3)]
         for ten, b in p["loi"].items()],
    )
    yield ""
    yield ("Đối chiếu số khớp đúng giữa hai phép ghép (greedy ở đây vs khớp tối ưu trong "
           "sed_eval) — lệch lớn nghĩa là phép ghép này không đáng tin và bảng trên phải bỏ:")
    yield ""
    yield from _bang(
        ["θ", "ghép greedy", "sed_eval Ntp", "lệch"],
        [[ten, f"{b['doi_chieu_dung_vs_ntp']['ghep_greedy']}",
          f"{b['doi_chieu_dung_vs_ntp']['sed_eval_ntp']}",
          f"{b['doi_chieu_dung_vs_ntp']['lech']:+d}"] for ten, b in p["loi"].items()],
    )
    yield ""

    yield "## 2. Theo lát cắt"
    yield ""
    yield (f"Chấm ở {nhan_theta(p['theta_sao'])}. **Lát cắt chồng nhau** — một clip vừa "
           "`overlap` vừa `reverb` nằm ở cả hai hàng, nên tổng cột *clip* lớn hơn "
           f"{p['n_clip']} clip của tập. `{LAT_CAT_SACH}` là clip không thuộc lát cắt khó "
           "nào; nó là mốc đối chiếu, không phải một lát cắt.")
    yield ""
    yield from _bang(
        ["lát cắt", "clip", "sự kiện thật", "F1(sự kiện)", "F1(đoạn 1s)"],
        [[h["lat_cat"], f"{h['n_clip']}", f"{h['n_ref']}",
          "—" if h["n_clip"] == 0 else f"{h['event_f1']:.4f}",
          "—" if h["n_clip"] == 0 else f"{h['segment_f1']:.4f}"] for h in p["lat_cat"]],
    )
    yield ""
    yield ("Phân loại lỗi **đếm lại trên từng tập con clip**, không chia tỉ lệ từ tổng — "
           "lát cắt chồng nhau nên cộng cột sẽ lớn hơn tổng toàn tập. Cột *phân mảnh/thật* "
           "là số sự kiện thật bị cắt thành nhiều mảnh, chia cho số sự kiện thật của chính "
           "lát cắt đó, nên so ngang giữa các hàng được.")
    yield ""
    yield from _bang(
        ["lát cắt", "thật", *NHAN_LOAI.values(), "phân mảnh", "phân mảnh/thật", "gộp"],
        [[h["lat_cat"], f"{h['n_ref']}",
          *(f"{h['dem'][k]}" for k in NHAN_LOAI),
          f"{h['phan_manh']}",
          "—" if not h["n_ref"] else so(h["phan_manh"] / h["n_ref"], 3),
          f"{h['gop']}"] for h in p["lat_cat"]],
    )
    yield ""

    yield "## 3. Ma trận nhầm lẫn"
    yield ""
    yield ("Hàng = lớp thật, cột = lớp đoán, `∅` = không có đối tác (Deletion ở cột, "
           "Insertion ở hàng). **Đường chéo gồm cả Đúng lẫn Boundary** — cả hai đều đúng "
           "lớp, chỉ khác thời điểm; đọc đường chéo như 'số đúng' là đọc sai.")
    yield ""
    yield "```"
    lop = p["class_ids"]
    yield "     " + "".join(f"{i:>5}" for i in range(len(lop) + 1))
    for i, hang_ten in enumerate([*lop, "∅"]):
        yield f"{i:>3}  " + "".join(f"{v:>5}" for v in p["ma_tran"][i]) + f"   {hang_ten}"
    yield "```"
    yield ""
    yield "Các cặp nhầm nhiều nhất, đối chiếu `confusable_with` khai trong `ontology_map.yaml`:"
    yield ""
    yield from _bang(
        ["thật", "đoán", "số", "đã khai trong ontology"],
        [[r["ref"], r["pred"], f"{r['n']}", "có" if r["da_khai"] else "**chưa**"]
         for r in p["nham"][:15]],
    )
    yield ""
    tong = sum(r["n"] for r in p["nham"]) or 1
    khai = sum(r["n"] for r in p["nham"] if r["da_khai"])
    yield (f"{khai}/{tong} ({khai/tong:.0%}) lượt nhầm rơi vào cặp đã khai. Phần còn lại là "
           "nhầm lẫn mà ontology **chưa dự đoán** — nếu tỉ lệ này cao thì thứ cần sửa là "
           "`confusable_with`, không phải model.")
    yield ""

    yield "## 4. Ngưỡng theo từng lớp"
    yield ""
    yield (f"θ tốt nhất **toàn cục** là {so(p['theta_sao'])}. Bảng dưới cho thấy θ đó "
           "không tối ưu cho phần lớn lớp:")
    yield ""
    yield from _bang(
        ["lớp", "sự kiện thật", "θ* riêng", "F1 @θ* riêng", f"F1 @θ={so(p['theta_sao'])}", "chênh"],
        [[r["class_id"], f"{r['n_ref']}", so(r["threshold"]), so(r["f1_rieng"], 4),
          so(r["f1_toan_cuc"], 4), f"{r['f1_rieng'] - r['f1_toan_cuc']:+.4f}".replace(".", ",")]
         for r in p["theo_lop"]],
    )
    yield ""
    yield from _bang(
        ["cấu hình ngưỡng", "F1(sự kiện)", "F1(đoạn 1s)"],
        [["θ = 0,50 (mặc định lúc train)", so(p["f1_05"], 4), so(p["seg_05"], 4)],
         [f"{nhan_theta(p['theta_sao'])} toàn cục", so(p["f1_sao"], 4), so(p["seg_sao"], 4)],
         ["θ riêng từng lớp", so(p["f1_vector"], 4), so(p["seg_vector"], 4)]],
    )
    yield ""
    yield ("Hàng cuối chọn 15 ngưỡng trên chính tập đang chấm, nên nó là **chặn trên**, "
           "không phải hiệu năng mong đợi trên dữ liệu mới. Muốn dùng số này làm kết quả "
           "thì ngưỡng phải khoá trên dev rồi đo một lần trên tập test độc lập — mà "
           "`gold_test` hiện **rỗng**.")


# ── Chạy ─────────────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> int:
    nguon = RUNS_DIR / args.run / "predictions" / f"{args.split}_{args.subset}.npz"
    if not nguon.exists():
        print(f"❌ chưa có {nguon}\n   Chạy: -m ml.evaluation.predictions --run {args.run} "
              f"--split {args.split}")
        return 1
    slice_path = DATA_DIR / args.split / "slice_index.jsonl"
    if not slice_path.exists():
        print(f"❌ chưa có {slice_path} — không dựng được bảng lát cắt")
        return 1

    bat_dau = time.time()
    du_doan = doc(nguon)
    class_ids = du_doan.class_ids
    median_size = cua_so_loc(du_doan, args.adaptive_postproc)
    tham_chieu = du_doan.tham_chieu()
    khung = [du_doan.khung(i) for i in range(len(du_doan.clip_ids))]

    print(f"▶ {args.run} · {args.split}/{args.subset} · {len(du_doan.clip_ids)} clip · "
          f"{sum(len(v) for v in tham_chieu.values())} sự kiện thật", flush=True)

    print("\n── quét ngưỡng ──", flush=True)
    ket_qua_quet = quet(du_doan, luoi_nguong(args.thresholds), median_size, khung=khung)
    tot = max(ket_qua_quet, key=lambda r: r["event_f1"])
    theta_sao = float(args.threshold) if args.threshold != "auto" else tot["threshold"]
    tai_05 = min(ket_qua_quet, key=lambda r: abs(r["threshold"] - 0.5))

    du_bao_05 = giai_ma(du_doan, khung, 0.5, median_size)
    du_bao_sao = giai_ma(du_doan, khung, theta_sao, median_size)

    print("\n── phân loại lỗi ──", flush=True)
    loi = {nhan_theta(0.5): bang_loi(tham_chieu, du_bao_05, class_ids),
           nhan_theta(theta_sao): bang_loi(tham_chieu, du_bao_sao, class_ids)}
    for ten, b in loi.items():
        print(f"  {ten}: " + "  ".join(f"{NHAN_LOAI[k]}={b['dem'][k]}" for k in NHAN_LOAI)
              + f"  phân mảnh={b['phan_manh']} gộp={b['gop']}", flush=True)

    print("\n── theo lát cắt ──", flush=True)
    nhom = nhom_theo_lat_cat(doc_lat_cat(slice_path), du_doan.clip_ids)
    lat_cat_hang = bang_lat_cat(tham_chieu, du_bao_sao, class_ids, du_doan.duration, nhom)

    kq_sao = loi[nhan_theta(theta_sao)]["_kq"]
    confusable = doc_confusable(ONTOLOGY_PATH)
    mt = ma_tran_nham(kq_sao, class_ids)
    nham = nham_theo_cap(mt, class_ids, confusable)

    print("\n── ngưỡng theo lớp ──", flush=True)
    tot_theo_lop = nguong_toi_uu_theo_lop(ket_qua_quet, class_ids)
    nguong_vec = vector_nguong(class_ids, tot_theo_lop, theta_sao)
    du_bao_vec = giai_ma(du_doan, khung, nguong_vec, median_size)
    seg_vector, f1_vector, _ = event_and_segment_f1(tham_chieu, du_bao_vec, class_ids,
                                                    du_doan.duration)
    n_ref_lop = {c: 0 for c in class_ids}
    for events in tham_chieu.values():
        for e in events:
            n_ref_lop[e["event_label"]] += 1
    theo_lop = [{
        "class_id": c, "n_ref": n_ref_lop[c],
        "threshold": tot_theo_lop[c]["threshold"], "f1_rieng": tot_theo_lop[c]["event_f1"],
        "f1_toan_cuc": tot["event_f1_theo_lop"].get(c, 0.0) if theta_sao == tot["threshold"]
        else next((r["event_f1_theo_lop"].get(c, 0.0) for r in ket_qua_quet
                   if r["threshold"] == theta_sao), 0.0),
    } for c in class_ids]
    print(f"  F1(sự kiện): θ=0,50 {tai_05['event_f1']:.4f} · θ={so(theta_sao)} "
          f"{tot['event_f1']:.4f} · θ riêng từng lớp {f1_vector:.4f}", flush=True)

    payload = {
        "run": args.run, "split": args.split, "subset": args.subset,
        "nguon": str(nguon.relative_to(REPO_ROOT)).replace("\\", "/"),
        "n_clip": len(du_doan.clip_ids), "n_ref": kq_sao.n_ref,
        "class_ids": class_ids, "collar_sec": ONSET_COLLAR_SEC,
        "adaptive_postproc": bool(args.adaptive_postproc),
        "median_size": (int(median_size) if np.isscalar(median_size)
                        else [int(x) for x in median_size]),
        "theta_sao": theta_sao,
        "f1_05": tai_05["event_f1"], "seg_05": tai_05["segment_f1"],
        "f1_sao": tot["event_f1"], "seg_sao": tot["segment_f1"],
        "f1_vector": f1_vector, "seg_vector": seg_vector,
        "loi": {k: {kk: vv for kk, vv in v.items() if kk != "_kq"} for k, v in loi.items()},
        "lat_cat": lat_cat_hang,
        "ma_tran": mt.tolist(), "nham": nham,
        "theo_lop": theo_lop,
        "nguong_theo_lop": {c: tot_theo_lop[c]["threshold"] for c in class_ids},
        "giay": round(time.time() - bat_dau, 1),
        "ghi_chu": ("dev tổng hợp sinh cùng recipe B0–B9 với train của v3 → thiên vị v3 "
                    "theo thiết kế; θ* và θ theo lớp chọn trên chính tập đang chấm → rò rỉ"),
    }

    ten = ten_bao_cao(args.adaptive_postproc)
    out_json = RUNS_DIR / args.run / f"{ten}.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md = viet_dan(RUNS_DIR / args.run / f"{ten}.md", dong_bao_cao(payload))
    print(f"\n✓ {out_json.relative_to(REPO_ROOT)}\n✓ {out_md.relative_to(REPO_ROOT)} "
          f"· {payload['giay']:.0f}s", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="tên thư mục trong ml/runs/")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--subset", default="all")
    parser.add_argument("--threshold", default="auto",
                        help="'auto' = θ tốt nhất của lượt quét, hoặc một số cụ thể")
    parser.add_argument("--thresholds", default="0.05:0.95:0.05", help="dau:cuoi:buoc")
    parser.add_argument("--adaptive-postproc", action="store_true",
                        help="cửa sổ lọc riêng từng lớp; xem cảnh báo rò rỉ ở cua_so_loc()")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
