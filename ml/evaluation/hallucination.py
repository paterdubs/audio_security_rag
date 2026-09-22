"""Bộ metric hallucination cho AAC — đóng góp C2, SYSTEM.md §8.2.

Ý tưởng: vì có strong label, ta biết chính xác tập sự kiện CÓ THẬT `G`. So nó với tập sự
kiện **được nhắc tới trong caption** `P` (trích bằng `ml/configs/event_lexicon.yaml`).

    EHR = |P \\ G| / |P|      nhắc mà không có thật   (ảo giác H1)   ↓
    EOR = |G \\ P| / |G|      có thật mà không nhắc   (bỏ sót H2)    ↓
    GS  = |P ∩ G| / |P ∪ G|  Jaccard                                ↑
    TOA = tỉ lệ cặp trong P∩G có thứ tự nhắc khớp thứ tự onset thật ↑
    CHR = EHR chỉ tính trên lớp tier Critical                        ↓

⚠️ **Đây là metric tự cài, không có thư viện tham chiếu để đối chứng.** Độ tin cậy của nó
bằng đúng độ tin cậy của bộ trích `trich_su_kien()`. SYSTEM.md §8.2 đặt điều kiện: bộ
trích phải được kiểm định thủ công trên 100 caption và báo cáo độ chính xác của CHÍNH NÓ,
nếu không thì C2 không được chấp nhận. **Việc đó chưa làm** — xem cờ
`meta.da_kiem_dinh_thu_cong` trong YAML.

⚠️ Mọi metric trả `None` khi mẫu số bằng 0, KHÔNG trả 0. Lý do cụ thể ở EHR: một caption
không nhắc sự kiện nào thì `|P| = 0`; trả 0 sẽ biến "im lặng" thành điểm hoàn hảo và
khiến "nói càng ít càng tốt" thành chiến lược thắng giả tạo. Khi tổng hợp trên nhiều clip
phải BỎ QUA các `None`, không được thay bằng 0.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

LEXICON_PATH = REPO_ROOT / "ml" / "configs" / "event_lexicon.yaml"

# Tier Critical theo ml/configs/ontology_map.yaml — bịa "tiếng súng" nguy hiểm hơn hẳn
# bịa "bước chân", nên chúng được đo riêng bằng CHR.
LOP_CRITICAL = ("gunshot", "explosion", "scream", "glass_breaking", "vehicle_crash")


def _nap_lexicon(path: Path = LEXICON_PATH) -> tuple[dict[str, list[str]], list[str], int]:
    """Nạp từ vựng. Ép mọi cụm về `str` để chống bẫy boolean của YAML 1.1.

    YAML 1.1 đọc `no`/`yes`/`on`/`off` không nháy thành BOOLEAN. `no` là từ phủ định phổ
    biến nhất tiếng Anh, nên bẫy này làm hỏng đúng chỗ quan trọng nhất. File YAML đã để
    trong nháy, nhưng ép kiểu ở đây để một lần sửa file sau này không âm thầm phá lại.
    """
    d = yaml.safe_load(path.read_text(encoding="utf-8"))

    def _chuoi(xs) -> list[str]:
        return [x if isinstance(x, str) else str(x).lower() for x in xs]

    tu_vung = {lop: _chuoi([*spec.get("en", []), *spec.get("vi", [])])
               for lop, spec in d["classes"].items()}
    phu_dinh = _chuoi([*d["phu_dinh"].get("en", []), *d["phu_dinh"].get("vi", [])])
    return tu_vung, phu_dinh, int(d["cua_so_phu_dinh"])


EVENT_LEXICON, CUM_PHU_DINH, CUA_SO_PHU_DINH = _nap_lexicon()


def _mau(cum: str) -> re.Pattern:
    """Khớp theo BIÊN TỪ, không phải chuỗi con.

    Không có `\\b` thì 'snapshot' khớp 'shot' và 'gunshot' bị đếm oan. `re.escape` vì cụm
    có thể chứa ký tự đặc biệt (ví dụ "n't").
    """
    return re.compile(rf"\b{re.escape(cum.lower())}\b")


_MAU_LOP: dict[str, list[tuple[str, re.Pattern]]] = {
    lop: [(cum, _mau(cum)) for cum in sorted(cums, key=len, reverse=True)]
    for lop, cums in EVENT_LEXICON.items()
}


def _bi_phu_dinh(caption_thuong: str, vi_tri: int, cua_so: int) -> bool:
    """Có cụm phủ định trong `cua_so` token ngay TRƯỚC vị trí khớp không.

    NegEx rút gọn: đủ cho caption một hai câu, KHÔNG đủ cho văn bản dài. Đây chính là chỗ
    mong manh nhất của bộ trích và là lý do §8.2 bắt kiểm định thủ công 100 caption.
    """
    truoc = caption_thuong[:vi_tri]
    # Giữ dấu nháy trong token: cắt theo \w thuần sẽ tách "isn't" thành ["isn", "t"].
    tokens = re.findall(r"[\w']+", truoc)[-cua_so:]
    if not tokens:
        return False
    noi = " ".join(tokens)
    for p in CUM_PHU_DINH:
        # "n't" nằm GIỮA từ ("isn't") nên không có biên từ trước nó — phải xét theo đuôi
        # token, không dùng \b được.
        if p.startswith("n'"):
            if any(t.endswith(p) for t in tokens):
                return True
        elif re.search(rf"\b{re.escape(p)}\b", noi):
            return True
    return False


def trich_su_kien(caption: str, cua_so_phu_dinh: int = CUA_SO_PHU_DINH) -> set[str]:
    """Tập lớp sự kiện ĐƯỢC NHẮC TỚI trong caption, đã loại phần bị phủ định.

    Một lớp được tính nếu có ÍT NHẤT một cụm của nó xuất hiện mà không bị phủ định — nên
    "no gunshot but a gunshot later" vẫn tính là có nhắc, đúng ý nghĩa.
    """
    thuong = caption.lower()
    ra: set[str] = set()
    for lop, mau_cua_lop in _MAU_LOP.items():
        for _cum, mau in mau_cua_lop:
            if any(not _bi_phu_dinh(thuong, m.start(), cua_so_phu_dinh)
                   for m in mau.finditer(thuong)):
                ra.add(lop)
                break
    return ra


def vi_tri_nhac(caption: str, lop: str) -> int | None:
    """Vị trí ký tự của lần nhắc ĐẦU TIÊN (không bị phủ định) của một lớp; None nếu không."""
    thuong = caption.lower()
    vi_tri = [m.start() for _cum, mau in _MAU_LOP.get(lop, [])
              for m in mau.finditer(thuong)
              if not _bi_phu_dinh(thuong, m.start(), CUA_SO_PHU_DINH)]
    return min(vi_tri) if vi_tri else None


# ── Năm metric ───────────────────────────────────────────────────────────────


def ehr(nhac: Iterable[str], that: Iterable[str]) -> float | None:
    """|P \\ G| / |P| — tỉ lệ sự kiện được nhắc mà không có thật. `|P|=0` → None."""
    P, G = set(nhac), set(that)
    return len(P - G) / len(P) if P else None


def eor(nhac: Iterable[str], that: Iterable[str]) -> float | None:
    """|G \\ P| / |G| — tỉ lệ sự kiện có thật mà bị bỏ sót. `|G|=0` → None."""
    P, G = set(nhac), set(that)
    return len(G - P) / len(G) if G else None


def grounding_score(nhac: Iterable[str], that: Iterable[str]) -> float | None:
    """Jaccard |P ∩ G| / |P ∪ G|. Hai tập cùng rỗng → None (không xác định, không phải 1)."""
    P, G = set(nhac), set(that)
    hop = P | G
    return len(P & G) / len(hop) if hop else None


def chr_critical(nhac: Iterable[str], that: Iterable[str],
                 critical: Iterable[str] = LOP_CRITICAL) -> float | None:
    """EHR chỉ tính trên lớp tier Critical. Không nhắc lớp Critical nào → None."""
    c = set(critical)
    return ehr(set(nhac) & c, set(that) & c)


def toa(caption: str, onset_that: dict[str, float]) -> float | None:
    """Tỉ lệ cặp lớp trong `P ∩ G` có thứ tự NHẮC khớp thứ tự ONSET thật.

    Cặp có onset bằng nhau bị loại khỏi mẫu số: thứ tự của chúng không xác định, tính là
    sai sẽ phạt oan caption nói "cùng lúc" — mà nói "cùng lúc" mới là caption đúng.
    Dưới hai cặp hợp lệ → None (CHƯA ĐO, không phải 0).
    """
    chung = [lop for lop in trich_su_kien(caption) & set(onset_that)
             if vi_tri_nhac(caption, lop) is not None]
    dung = tong = 0
    for i, a in enumerate(sorted(chung)):
        for b in sorted(chung)[i + 1:]:
            if onset_that[a] == onset_that[b]:
                continue
            tong += 1
            thu_tu_nhac = vi_tri_nhac(caption, a) < vi_tri_nhac(caption, b)
            thu_tu_that = onset_that[a] < onset_that[b]
            dung += thu_tu_nhac == thu_tu_that
    return dung / tong if tong else None
