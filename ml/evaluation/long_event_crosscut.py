"""Ứng viên thứ tư của `long_event` — nó xấu vì độ dài, hay vì trùng lát cắt khó khác?

    .venv/Scripts/python.exe -m ml.evaluation.long_event_crosscut --run panns_ft_v3 --split dev

Ba ứng viên trước đều bị đo bác bỏ: cửa sổ lọc trung vị 7 khung quá hẹp
(docs/measurements/long_event_postproc_20260919.md), trần `MAX_MEDIAN_FRAMES=51` quá thấp
(max_median_ceiling_20260919.md), khe hở năng lượng thật bên trong nhãn (long_event_gap.py).
Cả ba đều đo trên HÀNG `long_event` của bảng lát cắt — mà các lát cắt CHỒNG NHAU: trên
`data/synthetic/dev`, 144 clip `long_event` thì 52 cũng là `reverb`, 47 cũng là `overlap`,
45 cũng là `low_snr`. Hàng đó vì thế không tách được "dài thì khó" khỏi "clip dài tình cờ
cũng nhiễu/vọng".

Bảng 2x2 tách ra được: nếu chênh lệch `long_event` vs `khac` còn nguyên ở CẢ cột "có" lẫn
cột "không" của lát cắt đối chiếu, thì độ dài thật sự là biến giải thích; nếu chênh lệch
tan đi ở cột "không", giả thuyết độ dài bị bác bỏ và thủ phạm là lát cắt kia.

CỠ MẪU NHỎ: mỗi ô `long_event` giao lát cắt khác chỉ 45-52 clip trên dev. Đủ để nói
có/không có tương quan rõ, KHÔNG đủ để nói độ lớn hiệu ứng.

Không train lại, không GPU: đọc dự đoán đã lưu (Pha 3).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from common import enable_utf8_output  # noqa: E402

enable_utf8_output()

from ml.evaluation.error_analysis import (  # noqa: E402
    bang_lat_cat,
    doc_lat_cat,
    giai_ma,
    nhom_cat_cheo,
    so,
    viet_dan,
)
from ml.evaluation.error_taxonomy import ma_tran_nham, phan_loai  # noqa: E402
from ml.evaluation.long_event_gap import danh_dau_phan_manh  # noqa: E402
from ml.evaluation.predictions import doc  # noqa: E402
from ml.evaluation.sed_metrics import (  # noqa: E402
    DEFAULT_MEDIAN_FILTER_FRAMES,
    ONSET_COLLAR_SEC,
)

RUNS_DIR = REPO_ROOT / "ml" / "runs"
DATA_DIR = REPO_ROOT / "data" / "synthetic"

TRUC_MAC_DINH = "long_event"
DOI_CHIEU_MAC_DINH = ("low_snr", "reverb", "overlap", "causal_chain")
NHANH_KHAC = "khac"
TOP_CAP_NHAM = 5          # số cặp nhầm đậm nhất giữ lại cho mỗi ô


# ── Đo ───────────────────────────────────────────────────────────────────────


def ti_le_phan_manh(tham_chieu: dict[str, list[dict]],
                    du_bao: dict[str, list[dict]],
                    clip_ids: Sequence[str]) -> dict:
    """Tỉ lệ sự kiện tham chiếu bị vỡ thành >=2 dự báo cùng lớp, trên một ô của bảng 2x2.

    Ô RỖNG trả `ti_le=None`, không phải 0: ô rỗng là CHƯA ĐO. Trả 0 ở đó sẽ đẻ ra đúng
    kiểu kết luận sai mà dự án đã dặn (F1=0 vì chưa đo bị đọc thành F1 bằng 0).
    """
    clip_ids = list(clip_ids)
    n_ref = 0
    n_vo = 0
    for clip_id in clip_ids:
        ref = tham_chieu.get(clip_id) or []
        co = danh_dau_phan_manh(ref, du_bao.get(clip_id) or [])
        n_ref += len(ref)
        n_vo += sum(1 for x in co if x)
    return {"n_clip": len(clip_ids), "n_ref": n_ref, "n_vo": n_vo,
            "ti_le": (n_vo / n_ref) if n_ref else None}


def ti_le_cap_nham(tham_chieu: dict[str, list[dict]],
                   du_bao: dict[str, list[dict]],
                   clip_ids: Sequence[str],
                   class_ids: list[str]) -> list[dict]:
    """Cặp (lớp thật -> lớp đoán) ngoài đường chéo, chỉ trong một ô, sắp giảm dần.

    `ma_tran_nham()` vốn nhận `KetQuaPhanLoai` của TOÀN tập; ở đây phân loại lại trên đúng
    tập con clip của ô, không lọc sau — lọc sau là đúng cái bẫy mẫu số của
    `nhom_theo_lat_cat`, cặp nhầm của ô khác sẽ hiện lên như cặp nhầm của ô này.

    Bỏ đường chéo: nó gồm CẢ `dung` LẪN `bien` (xem cảnh báo ở `ma_tran_nham`), giữ lại thì
    cặp "nhầm" đậm nhất luôn là cặp lớp-với-chính-nó, che mất cặp nhầm thật. Bỏ luôn hàng/
    cột ∅ vì thiếu/thừa đã được đếm riêng ở `dem["thieu"]`/`dem["thua"]`.
    """
    clip_ids = list(clip_ids)
    if not clip_ids:
        return []
    con = {c: tham_chieu.get(c) or [] for c in clip_ids}
    con_du_bao = {c: du_bao.get(c) or [] for c in clip_ids}
    kq = phan_loai(con, con_du_bao, ONSET_COLLAR_SEC)
    mt = ma_tran_nham(kq, class_ids)
    rong = len(class_ids)
    cap = [{"ref_lop": class_ids[i], "pred_lop": class_ids[j], "n": int(mt[i, j])}
           for i in range(rong) for j in range(rong)
           if i != j and mt[i, j] > 0]
    return sorted(cap, key=lambda c: (-c["n"], c["ref_lop"], c["pred_lop"]))


def bang_2x2(tham_chieu, du_bao, nhom: dict[str, list[str]]) -> dict[str, dict]:
    return {ten: ti_le_phan_manh(tham_chieu, du_bao, clip_ids)
            for ten, clip_ids in nhom.items()}


def chenh_lech(phan_manh: dict[str, dict], truc: str, ten: str) -> dict:
    """Chênh lệch tỉ lệ vỡ truc trừ khac, đo RIÊNG trong cột 'có' và cột 'không'."""
    ra = {}
    for dau, cot in (("+", "co"), ("-", "khong")):
        a = phan_manh[f"{truc}{dau}{ten}"]["ti_le"]
        b = phan_manh[f"{NHANH_KHAC}{dau}{ten}"]["ti_le"]
        ra[cot] = None if (a is None or b is None) else a - b
    return ra


# ── Báo cáo ──────────────────────────────────────────────────────────────────


def dong_bao_cao(payload: dict) -> Iterator[str]:
    truc = payload["truc"]
    yield (f"# `{truc}` cắt chéo lát cắt khác — {payload['run']} trên "
           f"{payload['split']}/{payload['subset']}")
    yield ""
    yield (f"Sinh bởi `ml/evaluation/long_event_crosscut.py` từ `{payload['nguon']}` "
           f"· theta*={so(payload['theta_sao'])}. Không train lại, không GPU.")
    yield ""
    yield ("> ⚠️ `data/synthetic/dev` sinh **cùng recipe B0–B9** với tập train của `v3`, nên "
           "dev **thiên vị `v3` theo thiết kế**. theta* chọn trên chính tập đang chấm nên là "
           "**chặn trên lạc quan**.")
    yield ""
    yield ("> ⚠️ **Cỡ mẫu nhỏ**: mỗi ô giao chỉ 45–52 clip. Bảng này đủ để nói có/không có "
           "tương quan rõ, KHÔNG đủ để nói độ lớn hiệu ứng.")
    yield ""
    yield "## Tỉ lệ sự kiện bị phân mảnh theo từng ô"
    yield ""
    yield "| đối chiếu | ô | clip | sự kiện | vỡ | tỉ lệ vỡ |"
    yield "|---|---|---:|---:|---:|---:|"
    for ten in payload["doi_chieu"]:
        for ben in (truc, NHANH_KHAC):
            for dau in ("+", "-"):
                o = f"{ben}{dau}{ten}"
                h = payload["phan_manh"][o]
                ti = "—" if h["ti_le"] is None else so(h["ti_le"])
                yield f"| {ten} | `{o}` | {h['n_clip']} | {h['n_ref']} | {h['n_vo']} | {ti} |"
    yield ""
    yield f"## Chênh lệch tỉ lệ vỡ `{truc}` trừ `{NHANH_KHAC}`"
    yield ""
    yield ("Nếu chênh lệch còn nguyên ở **cả hai cột**, độ dài là biến giải thích thật. Nếu "
           "nó tan đi ở cột *không*, thủ phạm là lát cắt đối chiếu chứ không phải độ dài.")
    yield ""
    yield "| đối chiếu | chênh lệch ở cột *có* | chênh lệch ở cột *không* |"
    yield "|---|---:|---:|"
    for ten in payload["doi_chieu"]:
        c = payload["chenh_lech"][ten]
        yield (f"| {ten} | {'—' if c['co'] is None else so(c['co'])} "
               f"| {'—' if c['khong'] is None else so(c['khong'])} |")
    yield ""
    yield "## F1 sự kiện và sáu loại lỗi theo từng ô"
    yield ""
    yield ("Tỉ lệ lấy trên số sự kiện THAM CHIẾU của chính ô đó, nên `thua` vượt 1,0 được — "
           "một sự kiện thật có thể hứng nhiều dự báo thừa. `bien` là phép CHIA NHỎ LẠI rổ "
           "Deletion+Insertion của sed_eval, không phải loại lỗi thứ bảy song song.")
    yield ""
    yield "| ô | clip | sự kiện | event-F1 | đúng | biên | thay thế | thiếu | thừa |"
    yield "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    for h in payload["lat_cat"]:
        if h["n_clip"] == 0 or not h["n_ref"]:
            yield f"| `{h['lat_cat']}` | {h['n_clip']} | {h['n_ref']} | — | — | — | — | — | — |"
            continue
        d = h["dem"]
        ti = [so(d[k] / h["n_ref"]) for k in ("dung", "bien", "thay_the", "thieu", "thua")]
        yield (f"| `{h['lat_cat']}` | {h['n_clip']} | {h['n_ref']} | {so(h['event_f1'])} "
               f"| {' | '.join(ti)} |")
    yield ""
    yield "## Cặp nhầm đậm nhất theo từng ô"
    yield ""
    yield ("Ngoài đường chéo (đường chéo gồm cả `dung` lẫn `bien`), tối đa "
           f"{TOP_CAP_NHAM} cặp mỗi ô.")
    yield ""
    yield "| ô | cặp nhầm (thật → đoán) |"
    yield "|---|---|"
    for ten, cap in payload["cap_nham"].items():
        mo_ta = " · ".join(f"{c['ref_lop']}→{c['pred_lop']} ({c['n']})" for c in cap) or "—"
        yield f"| `{ten}` | {mo_ta} |"
    yield ""


# ── Chạy ─────────────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> int:
    nguon = RUNS_DIR / args.run / "predictions" / f"{args.split}_{args.subset}.npz"
    if not nguon.exists():
        print(f"❌ chưa có {nguon}\n   Chạy: -m ml.evaluation.predictions --run {args.run} "
              f"--split {args.split}")
        return 1
    slice_path = DATA_DIR / args.split / "slice_index.jsonl"
    if not slice_path.exists():
        print(f"❌ chưa có {slice_path} — không dựng được lát cắt")
        return 1
    analysis_path = RUNS_DIR / args.run / "analysis.json"
    if not analysis_path.exists():
        print(f"❌ chưa có {analysis_path}\n   Chạy: -m ml.evaluation.error_analysis "
              f"--run {args.run} --split {args.split}  (cần theta* của run này)")
        return 1

    bat_dau = time.time()
    theta_sao = json.loads(analysis_path.read_text(encoding="utf-8"))["theta_sao"]
    du_doan = doc(nguon)
    khung = [du_doan.khung(i) for i in range(len(du_doan.clip_ids))]
    du_bao_sao = giai_ma(du_doan, khung, theta_sao, DEFAULT_MEDIAN_FILTER_FRAMES)
    tham_chieu = du_doan.tham_chieu()

    doi_chieu = list(args.doi_chieu)
    nhom = nhom_cat_cheo(doc_lat_cat(slice_path), du_doan.clip_ids, args.truc, doi_chieu)
    print(f"▶ {args.run} · {args.split}/{args.subset} · theta*={so(theta_sao)} · "
          f"trục={args.truc} · {len(doi_chieu)} lát cắt đối chiếu", flush=True)

    phan_manh = bang_2x2(tham_chieu, du_bao_sao, nhom)
    for ten in doi_chieu:
        c = chenh_lech(phan_manh, args.truc, ten)
        print(f"  {ten}: chênh lệch vỡ  có={c['co']}  không={c['khong']}", flush=True)

    lat_cat_hang = bang_lat_cat(tham_chieu, du_bao_sao, du_doan.class_ids,
                                du_doan.duration, nhom)
    cap_nham = {ten: ti_le_cap_nham(tham_chieu, du_bao_sao, clip_ids,
                                    du_doan.class_ids)[:TOP_CAP_NHAM]
                for ten, clip_ids in nhom.items()}

    payload = {
        "run": args.run, "split": args.split, "subset": args.subset,
        "nguon": str(nguon.relative_to(REPO_ROOT)).replace("\\", "/"),
        "theta_sao": theta_sao, "truc": args.truc, "doi_chieu": doi_chieu,
        "phan_manh": phan_manh,
        "chenh_lech": {ten: chenh_lech(phan_manh, args.truc, ten) for ten in doi_chieu},
        "lat_cat": lat_cat_hang,
        "cap_nham": cap_nham,
        "giay": round(time.time() - bat_dau, 1),
        "ghi_chu": "dev tổng hợp thiên vị v3; theta* chọn trên chính tập đang chấm; cỡ mẫu nhỏ.",
    }
    out_json = RUNS_DIR / args.run / "long_event_crosscut.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md = viet_dan(RUNS_DIR / args.run / "long_event_crosscut.md", dong_bao_cao(payload))
    print(f"\n✓ {out_json.relative_to(REPO_ROOT)}\n✓ {out_md.relative_to(REPO_ROOT)} "
          f"· {payload['giay']:.0f}s", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--subset", default="all")
    parser.add_argument("--truc", default=TRUC_MAC_DINH)
    parser.add_argument("--doi-chieu", nargs="+", default=list(DOI_CHIEU_MAC_DINH))
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
