"""Ghép cặp sự kiện tham chiếu ↔ dự báo và phân loại lỗi — Pha 4 của TRAINING_OPS_PLAN.

**Vì sao phải tự viết phép ghép, trong khi repo có quy ước "không tự cài công thức F1".**
`sed_eval` chỉ trả về TỈ LỆ insertion/deletion/substitution (chuẩn hoá theo Nref) chứ
không phơi ra cặp nào khớp cặp nào. Ma trận nhầm lẫn 15×15, lỗi lệch biên, phân mảnh và
gộp đều cần biết cặp cụ thể. Nên phân công như sau, và nó là ranh giới không được mờ:

  · F1, precision, recall, S/D/I  →  `sed_eval` qua `ml/evaluation/sed_metrics.py`. Chân lý.
  · Cặp ref↔pred, 6 loại lỗi     →  module này. Là phép CHIA NHỎ LẠI các rổ D/I của
                                     sed_eval, không phải số đối chọi với nó.

Cụ thể: một dự báo đúng lớp nhưng lệch onset 0,5 s bị sed_eval tính MỘT Deletion + MỘT
Insertion. Module này gọi nó là Boundary. Hai cách đếm đều đúng theo định nghĩa của mình;
chúng KHÔNG cộng ra cùng một con số và báo cáo phải nói thẳng điều đó.

Hai tầng, cố ý tách:

  Tầng 1 — mỗi sự kiện tham chiếu nhận ĐÚNG MỘT nhãn: dung / bien / thay_the / thieu.
           Mỗi dự báo không bị tiêu thụ → thua. Bất biến kiểm được:
           dung + bien + thay_the + thieu == số sự kiện tham chiếu.
  Tầng 2 — cấu trúc, ĐẾM RIÊNG và được phép chồng lên tầng 1: phan_manh, gop.

Trừ chéo hai tầng vào nhau là cách nhanh nhất để mất khả năng kiểm tổng — một sự kiện
đếm hai lần không có triệu chứng nào ngoài "tỉ lệ Deletion hơi thấp".
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

# Nhãn tầng 1. Thứ tự này là thứ tự ưu tiên khi ghép, không phải thứ tự tuỳ ý:
# khớp chặt trước, nới dần, vét cuối cùng.
LOAI_TANG_MOT = ("dung", "bien", "thay_the", "thieu")
LOAI_THUA = "thua"
MOI_LOAI = (*LOAI_TANG_MOT, LOAI_THUA)

# Nhãn hiển thị cho hàng/cột "không có đối tác" trong ma trận nhầm lẫn.
NHAN_RONG = "∅"


@dataclass(frozen=True)
class CapGhep:
    """Một cặp đã ghép, hoặc một sự kiện lẻ (thiếu/thừa) — khi đó một phía là None."""

    clip_id: str
    loai: str
    ref_lop: str | None = None
    pred_lop: str | None = None
    ref_onset: float | None = None
    pred_onset: float | None = None


@dataclass
class KetQuaPhanLoai:
    cap: list[CapGhep] = field(default_factory=list)
    dem: Counter = field(default_factory=Counter)
    n_ref: int = 0
    n_pred: int = 0
    n_phan_manh: int = 0
    n_gop: int = 0

    def ty_le(self) -> dict[str, float]:
        """Tỉ lệ theo số sự kiện tham chiếu — cùng mẫu số với error rate của sed_eval."""
        if self.n_ref == 0:
            return {loai: 0.0 for loai in MOI_LOAI}
        return {loai: self.dem[loai] / self.n_ref for loai in MOI_LOAI}


def chong_lan(a: dict, b: dict) -> float:
    """Độ dài phần giao (giây). 0 nghĩa là rời nhau hoàn toàn."""
    return max(0.0, min(a["offset"], b["offset"]) - max(a["onset"], b["onset"]))


def ghep_mot_clip(clip_id: str, ref: list[dict], pred: list[dict],
                  collar: float) -> tuple[list[CapGhep], int, int]:
    """Ghép trong phạm vi MỘT clip → (cặp, số ref bị phân mảnh, số pred gộp nhiều ref).

    Ghép greedy theo ba vòng ưu tiên. Không dùng khớp tối ưu (Hungarian) như sed_eval:
    mục tiêu ở đây là GIẢI THÍCH lỗi chứ không phải chấm điểm, và greedy cho phép nêu
    được lý do từng cặp. Chênh lệch giữa hai cách được đối chiếu tường minh ở
    `error_analysis.py` thay vì giấu đi.
    """
    con_lai = set(range(len(pred)))
    nhan: dict[int, tuple[str, int | None]] = {}

    # Vòng 1 — đúng lớp và onset trong collar. Chọn dự báo GẦN onset nhất, không phải dự
    # báo đứng trước: lấy nhầm thì dự báo còn lại thành Insertion và ma trận lệch một ô.
    for i, r in enumerate(ref):
        ung_vien = [j for j in con_lai
                    if pred[j]["event_label"] == r["event_label"]
                    and abs(pred[j]["onset"] - r["onset"]) <= collar]
        if ung_vien:
            j = min(ung_vien, key=lambda j: abs(pred[j]["onset"] - ref[i]["onset"]))
            nhan[i] = ("dung", j)
            con_lai.discard(j)

    # Vòng 2 — đúng lớp, có chồng lấn, nhưng onset ngoài collar → lệch biên.
    for i, r in enumerate(ref):
        if i in nhan:
            continue
        ung_vien = [j for j in con_lai
                    if pred[j]["event_label"] == r["event_label"] and chong_lan(r, pred[j]) > 0]
        if ung_vien:
            j = max(ung_vien, key=lambda j: chong_lan(ref[i], pred[j]))
            nhan[i] = ("bien", j)
            con_lai.discard(j)

    # Vòng 3 — sai lớp nhưng đúng thời điểm → thay thế. Đây là dữ liệu của ma trận nhầm.
    for i, r in enumerate(ref):
        if i in nhan:
            continue
        ung_vien = [j for j in con_lai if chong_lan(r, pred[j]) > 0]
        if ung_vien:
            j = max(ung_vien, key=lambda j: chong_lan(ref[i], pred[j]))
            nhan[i] = ("thay_the", j)
            con_lai.discard(j)

    cap: list[CapGhep] = []
    for i, r in enumerate(ref):
        loai, j = nhan.get(i, ("thieu", None))
        cap.append(CapGhep(
            clip_id=clip_id, loai=loai,
            ref_lop=r["event_label"], ref_onset=float(r["onset"]),
            pred_lop=None if j is None else pred[j]["event_label"],
            pred_onset=None if j is None else float(pred[j]["onset"]),
        ))
    for j in sorted(con_lai):
        cap.append(CapGhep(clip_id=clip_id, loai=LOAI_THUA,
                           pred_lop=pred[j]["event_label"], pred_onset=float(pred[j]["onset"])))

    return cap, *_dem_cau_truc(ref, pred)


def _dem_cau_truc(ref: list[dict], pred: list[dict]) -> tuple[int, int]:
    """(số ref bị vỡ thành ≥2 mảnh, số pred nuốt ≥2 ref) — cùng lớp, có chồng lấn.

    Đếm độc lập với tầng 1: một sự kiện vừa được ghép đúng vừa bị phân mảnh là chuyện
    bình thường, và đó chính là thông tin cần giữ.
    """
    phan_manh = sum(
        1 for r in ref
        if sum(1 for p in pred
               if p["event_label"] == r["event_label"] and chong_lan(r, p) > 0) >= 2
    )
    gop = sum(
        1 for p in pred
        if sum(1 for r in ref
               if r["event_label"] == p["event_label"] and chong_lan(r, p) > 0) >= 2
    )
    return phan_manh, gop


def phan_loai(tham_chieu: dict[str, list[dict]], du_bao: dict[str, list[dict]],
              collar: float) -> KetQuaPhanLoai:
    """Cộng dồn qua MỌI clip có ở một trong hai phía.

    Lấy hợp của hai tập khoá, không lấy khoá của riêng tham chiếu: clip im lặng mà model
    đoán bừa vẫn phải đếm Insertion. Bỏ nó đi là đúng cái lỗi đã làm điểm cao lên một
    cách sai ở khâu chấm sed_eval.
    """
    kq = KetQuaPhanLoai()
    for clip_id in sorted(set(tham_chieu) | set(du_bao)):
        ref = tham_chieu.get(clip_id, [])
        pred = du_bao.get(clip_id, [])
        cap, n_manh, n_gop = ghep_mot_clip(clip_id, ref, pred, collar)
        kq.cap.extend(cap)
        kq.dem.update(c.loai for c in cap)
        kq.n_ref += len(ref)
        kq.n_pred += len(pred)
        kq.n_phan_manh += n_manh
        kq.n_gop += n_gop

    for loai in MOI_LOAI:
        kq.dem.setdefault(loai, 0)
    thieu_hut = sum(kq.dem[loai] for loai in LOAI_TANG_MOT) - kq.n_ref
    if thieu_hut != 0:
        raise AssertionError(
            f"tầng 1 đếm {sum(kq.dem[x] for x in LOAI_TANG_MOT)} nhãn cho {kq.n_ref} "
            f"sự kiện tham chiếu (lệch {thieu_hut:+d}) — phép ghép đã đếm trùng hoặc bỏ sót"
        )
    return kq


def ma_tran_nham(kq: KetQuaPhanLoai, class_ids: list[str]) -> np.ndarray:
    """(C+1, C+1) đếm cặp: hàng = lớp tham chiếu, cột = lớp dự báo, chỉ số C = ∅.

    ⚠️ Đường chéo gồm CẢ `dung` LẪN `bien` — cả hai đều đúng lớp, chỉ khác ở thời điểm.
    Đọc đường chéo như "số đúng" là đọc sai; số đúng nằm ở `dem["dung"]`.
    """
    chi_so = {c: i for i, c in enumerate(class_ids)}
    rong = len(class_ids)
    mt = np.zeros((rong + 1, rong + 1), dtype=np.int64)
    for c in kq.cap:
        hang = chi_so.get(c.ref_lop, rong) if c.ref_lop is not None else rong
        cot = chi_so.get(c.pred_lop, rong) if c.pred_lop is not None else rong
        mt[hang, cot] += 1
    return mt


def nham_theo_cap(mt: np.ndarray, class_ids: list[str],
                  confusable: dict[str, list[str]]) -> list[dict]:
    """Các ô nhầm lẫn ngoài đường chéo, kèm cờ đã-khai-trong-ontology hay chưa.

    Mục đích không phải để khoe ma trận mà để trả lời một câu cụ thể: nhầm lẫn thật có
    rơi vào các cặp `confusable_with` đã dự đoán trong `ontology_map.yaml` không. Nếu
    phần lớn nhầm lẫn nằm ngoài danh sách đó thì chính danh sách đó cần sửa.
    """
    ket_qua = []
    for i, lop_ref in enumerate(class_ids):
        for j, lop_pred in enumerate(class_ids):
            if i == j or mt[i, j] == 0:
                continue
            ket_qua.append({
                "ref": lop_ref, "pred": lop_pred, "n": int(mt[i, j]),
                "da_khai": lop_pred in confusable.get(lop_ref, []),
            })
    return sorted(ket_qua, key=lambda r: -r["n"])
