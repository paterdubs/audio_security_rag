"""Sàng lọc foreground bank bằng PANNs CNN14 — DATA_PLAN §4.2–4.3.

    python scripts/auto_screen.py --split foreground_bank_train
    python scripts/auto_screen.py --class-id siren --limit 50
    python scripts/auto_screen.py --report-only

Việc của script này KHÔNG phải thay người phán xét, mà là bỏ bớt thao tác cơ học và
DỒN ĐÚNG CA KHÓ cho người. Năng suất thực tế: ~150 clip/giờ khi có đề xuất biên sẵn,
so với 30–40 clip/giờ khi gán mù.

Đầu ra:
  data/manifests/screen_scores.csv   điểm thô của mọi clip (để hiệu chuẩn lại ngưỡng)
  data/manifests/review_queue.csv    hàng đợi người, đã xếp theo độ khó
  exclusions.csv (stage=auto_screen) clip bị loại tự động

Cần torch + panns-inference — xem requirements-screen.txt. Phần định tuyến và đề xuất
biên KHÔNG cần torch, nên kiểm thử được độc lập.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from common import REPO_ROOT, enable_utf8_output, write_exclusions

NORMALIZED_DIR = REPO_ROOT / "data" / "interim" / "normalized"
SPLITS_PATH = REPO_ROOT / "data" / "manifests" / "splits.csv"
ONTOLOGY_PATH = REPO_ROOT / "ml" / "configs" / "ontology_map.yaml"
SCORES_PATH = REPO_ROOT / "data" / "manifests" / "screen_scores.csv"
QUEUE_PATH = REPO_ROOT / "data" / "manifests" / "review_queue.csv"
STAGE = "auto_screen"

ACCEPT_THRESHOLD = 0.30      # p_target ≥ ngưỡng này và vượt confusable → tự động nhận
REJECT_THRESHOLD = 0.05      # p_target dưới ngưỡng này → tự động loại
# Nếu tagger loại quá nửa một lớp, nó không phải bộ lọc hợp lệ cho lớp đó — xem
# guard_low_resolution_classes(). Nửa là ranh giới tự nhiên, không phải số tinh chỉnh.
GUARD_REJECT_RATE = 0.50
SAMPLE_RATE = 32000          # PANNs CNN14 được huấn luyện ở 32 kHz, không phải 16 kHz
MIN_SAMPLES = SAMPLE_RATE    # 1 giây — dưới mức này CNN14 sập ở tầng pooling

SCORE_FIELDS = ["file_id", "class_id", "p_target", "p_confusable", "top_confusable",
                "decision", "priority", "onset", "offset", "path_norm"]
QUEUE_FIELDS = ["priority", "file_id", "class_id", "p_target", "p_confusable",
                "top_confusable", "onset", "offset", "path_norm"]

# Đề xuất biên (§4.3)
MEDIAN_WINDOW = 5            # frame
BOUNDARY_FRACTION = 0.5      # ngưỡng = phần này × đỉnh
BOUNDARY_PAD_SEC = 0.05      # nới ±50 ms


@dataclass(frozen=True)
class Routing:
    decision: str            # auto_accept | auto_reject | review
    priority: str            # high | normal | ""
    reason: str


# ── Định tuyến ───────────────────────────────────────────────────────────────


def route(p_target: float, p_confusable: float) -> Routing:
    """Quyết định số phận một clip từ hai con số. DATA_PLAN §4.2.

    THỨ TỰ LUẬT QUAN TRỌNG, vì luật 2 và luật 3 chồng nhau: p_target = 0.02 với
    p_confusable = 0.50 thoả cả hai. Theo đúng thứ tự bảng thì luật 2 thắng, và đó là
    lựa chọn đúng — một clip khai là `gunshot` mà máy chấm 0.02 cho gunshot và 0.50
    cho fireworks thì gần như chắc chắn là fireworks, loại thẳng chứ không tốn thời
    gian của người. Đảo thứ tự sẽ dồn hàng nghìn clip rõ ràng sai lớp vào hàng đợi.
    """
    if p_target >= ACCEPT_THRESHOLD and p_target > p_confusable:
        return Routing("auto_accept", "", f"p_target={p_target:.2f} ≥ {ACCEPT_THRESHOLD}")
    if p_target < REJECT_THRESHOLD:
        return Routing("auto_reject", "", f"p_target={p_target:.2f} < {REJECT_THRESHOLD}")
    if p_confusable > p_target:
        # Luật quan trọng nhất: đây chính là những ca quyết định tỉ lệ báo động giả.
        return Routing("review", "high", f"lớp dễ nhầm thắng ({p_confusable:.2f} > {p_target:.2f})")
    return Routing("review", "normal", f"p_target={p_target:.2f} chưa đủ chắc")


def priority_rank(routing: Routing) -> int:
    """Khoá sắp xếp hàng đợi. Người duyệt phải gặp ca khó khi còn tỉnh táo."""
    return {"high": 0, "normal": 1}.get(routing.priority, 2)


# ── Đề xuất biên sự kiện ─────────────────────────────────────────────────────


def median_filter(values: np.ndarray, window: int) -> np.ndarray:
    """Lọc trung vị 1 chiều, giữ nguyên độ dài.

    Xác suất mức frame của PANNs rất răng cưa; không lọc thì "vùng liên tục dài nhất"
    sẽ bị một frame tụt xuống dưới ngưỡng cắt làm đôi, và biên đề xuất ngắn hơn sự
    kiện thật. Dùng trung vị chứ không phải trung bình: trung bình kéo cả vùng xuống
    khi gặp một frame 0, làm lệch chính cái biên ta muốn tìm.
    """
    if window <= 1 or values.size == 0:
        return values
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    return np.array([np.median(padded[i:i + window]) for i in range(values.size)])


def longest_run(mask: np.ndarray) -> tuple[int, int]:
    """(đầu, cuối-mở) của dải True dài nhất. (0, 0) nếu không có dải nào."""
    best = (0, 0)
    start = None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start > best[1] - best[0]:
                best = (start, index)
            start = None
    if start is not None and mask.size - start > best[1] - best[0]:
        best = (start, mask.size)
    return best


def propose_boundary(frame_probs: np.ndarray, duration: float) -> tuple[float, float]:
    """[onset, offset] đề xuất từ xác suất mức frame. DATA_PLAN §4.3.

    Ngưỡng đặt ở 0.5 × ĐỈNH chứ không phải một hằng số tuyệt đối: một clip mà máy chỉ
    chắc 0.2 vẫn có biên đúng, chỉ là toàn bộ đường cong thấp. Ngưỡng tuyệt đối sẽ trả
    về vùng rỗng cho đúng những clip cần người xem nhất.
    """
    if frame_probs.size == 0 or duration <= 0:
        return 0.0, duration
    smoothed = median_filter(frame_probs, MEDIAN_WINDOW)
    peak = float(smoothed.max())
    if peak <= 0:
        return 0.0, duration

    start, end = longest_run(smoothed >= peak * BOUNDARY_FRACTION)
    if end <= start:
        return 0.0, duration

    seconds_per_frame = duration / frame_probs.size
    onset = max(0.0, start * seconds_per_frame - BOUNDARY_PAD_SEC)
    offset = min(duration, end * seconds_per_frame + BOUNDARY_PAD_SEC)
    return round(onset, 3), round(offset, 3)


# ── Ánh xạ ontology → chỉ số lớp của PANNs ───────────────────────────────────


def load_ontology() -> dict:
    return yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8"))


def mid_to_index(labels: list[str], ontology_json: Path) -> dict[str, int]:
    """{mã AudioSet: chỉ số cột đầu ra của PANNs}.

    PANNs xuất 527 cột theo ĐÚNG thứ tự nhãn AudioSet. Ghép theo TÊN HIỂN THỊ là chỗ
    dễ sai nhất — tên có dấu phẩy, có ngoặc, và trùng nhau giữa các nhánh. Ghép theo
    mã `/m/xxxxx` thì không có chỗ cho sự mơ hồ.
    """
    import json

    ontology = json.loads(ontology_json.read_text(encoding="utf-8"))
    by_name = {entry["name"]: entry["id"] for entry in ontology}
    mapping = {}
    for index, name in enumerate(labels):
        mid = by_name.get(name)
        if mid:
            mapping[mid] = index
    return mapping


def class_indices(ontology: dict, class_id: str, index_of: dict[str, int]) -> tuple[list[int], dict[str, list[int]]]:
    """(chỉ số của lớp đích, {lớp dễ nhầm: chỉ số của nó})."""
    spec = ontology["classes"][class_id]
    target = [index_of[mid] for mid in spec.get("audioset_ids", []) if mid in index_of]
    confusable = {}
    for other in spec.get("confusable_with", []):
        other_spec = ontology["classes"].get(other, {})
        found = [index_of[mid] for mid in other_spec.get("audioset_ids", []) if mid in index_of]
        if found:
            confusable[other] = found
    return target, confusable


def score_clip(clip_probs: np.ndarray, target: list[int],
               confusable: dict[str, list[int]]) -> tuple[float, float, str]:
    """(p_target, p_confusable, tên lớp dễ nhầm mạnh nhất)."""
    p_target = float(clip_probs[target].max()) if target else 0.0
    best_name, p_confusable = "", 0.0
    for name, indices in confusable.items():
        value = float(clip_probs[indices].max())
        if value > p_confusable:
            best_name, p_confusable = name, value
    return p_target, p_confusable, best_name


# ── Chạy mô hình ─────────────────────────────────────────────────────────────


CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
CHECKPOINT_PATH = Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth"
CHECKPOINT_BYTES = 327428481


def ensure_checkpoint() -> Path:
    """Tải checkpoint CNN14 nếu chưa có, và ghi lại SHA-256 của nó.

    panns-inference tự tải bằng `os.system("wget ...")`. Trên Windows không có wget
    nên nó tạo sẵn thư mục rồi chết ở FileNotFoundError — thông báo không hề nhắc tới
    wget, và người dùng đi tìm nhầm chỗ. Tự tải vừa chạy được trên mọi nền, vừa tận
    dụng cơ chế thử-lại-khi-504 đã viết ở download_sources.py.

    Tác giả PANNs KHÔNG công bố checksum. Ta tự ghi lại SHA-256 của bản đã tải để lần
    sau còn đối chiếu — không có nó thì không có cách nào biết mình đang chạy đúng bộ
    trọng số của lần thí nghiệm trước.
    """
    import hashlib

    from download_sources import fetch_with_retry

    if CHECKPOINT_PATH.exists() and CHECKPOINT_PATH.stat().st_size == CHECKPOINT_BYTES:
        return CHECKPOINT_PATH

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    part = CHECKPOINT_PATH.with_suffix(".pth.part")
    print(f"▶ tải checkpoint CNN14 (~312 MB) → {CHECKPOINT_PATH}")
    done = part.stat().st_size if part.exists() else 0
    done = fetch_with_retry(CHECKPOINT_URL, part, done, CHECKPOINT_PATH.name)
    if done != CHECKPOINT_BYTES:
        part.unlink(missing_ok=True)
        raise SystemExit(f"❌ checkpoint tải về {done} byte, mong đợi {CHECKPOINT_BYTES}")
    part.replace(CHECKPOINT_PATH)

    digest = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    record = CHECKPOINT_PATH.with_suffix(".pth.sha256")
    record.write_text(digest, encoding="utf-8")
    print(f"  ✓ SHA-256 {digest[:16]}… đã ghi vào {record.name}")
    return CHECKPOINT_PATH


def load_tagger():
    """Nạp PANNs CNN14. Import ở đây chứ không ở đầu file: phần định tuyến và đề xuất
    biên phải kiểm thử được trên máy không cài torch (requirements-data.txt §ghi chú)."""
    import torch
    from panns_inference import SoundEventDetection, AudioTagging

    checkpoint = ensure_checkpoint()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"▶ nạp PANNs CNN14 trên {device}")
    return AudioTagging(checkpoint_path=str(checkpoint), device=device), \
        SoundEventDetection(checkpoint_path=str(checkpoint), device=device), device


def pad_to_minimum(audio: np.ndarray) -> np.ndarray:
    """Kéo dài clip ngắn bằng cách LẶP LẠI chính nó, không phải đệm im lặng.

    Hai lý do, lý do thứ hai mới là lý do thật:

    1. CNN14 sập với clip quá ngắn: chồng pooling đưa kích thước tensor về 0 và ném
       `RuntimeError: Output size is too small`. 1076/5260 clip của bank dưới 1 giây.

    2. Đệm im lặng LÀM SAI KẾT QUẢ, im lặng. CNN14 gộp theo cả trung bình lẫn cực đại
       trên trục thời gian, nên chèn 0.9 giây im lặng vào một tiếng súng 0.1 giây sẽ
       dìm xác suất xuống. Đo trên 40 clip ngắn thật (<0.6 s):

           đệm im lặng : p_target trung bình 0.012 → 37/40 clip bị TỰ ĐỘNG LOẠI
           lặp lại clip: p_target trung bình 0.400 →  2/40 clip bị loại

       Tức đệm im lặng sẽ xoá khoảng 1000 clip khỏi bank với lý do "sai lớp", mà
       chúng không hề sai lớp. Và mất mát ấy tập trung đúng vào các lớp xung kích —
       object_drop_dishes, gunshot, glass_breaking, door_slam — tức đúng những lớp
       quyết định của bài toán an ninh.
    """
    if len(audio) >= MIN_SAMPLES or len(audio) == 0:
        return audio
    repeats = int(np.ceil(MIN_SAMPLES / len(audio)))
    return np.tile(audio, repeats)[:MIN_SAMPLES]


def real_frame_count(frames: int, original_samples: int) -> int:
    """Số frame đầu tiên thuộc về audio THẬT, phần còn lại là bản lặp.

    Đề xuất biên chỉ được đọc phần thật: đọc cả phần lặp sẽ cho ra một biên trỏ vào
    bản sao thứ ba của sự kiện, tức một khoảng thời gian không tồn tại trong file gốc.
    """
    if original_samples >= MIN_SAMPLES:
        return frames
    return max(1, round(frames * original_samples / MIN_SAMPLES))


def load_audio_32k(path: Path) -> np.ndarray:
    """Đọc clip đã chuẩn hoá và đưa về 32 kHz — tần số PANNs được huấn luyện.

    Bank lưu ở 16 kHz theo preprocessing.yaml. Đưa thẳng 16 kHz vào PANNs sẽ chạy
    được và cho ra số, nhưng là số của một tín hiệu bị dịch cao độ một quãng tám —
    sai hoàn toàn mà không có thông báo nào.
    """
    import librosa

    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return audio


# ── Vào/ra ───────────────────────────────────────────────────────────────────


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def clips_to_screen(split: str | None, class_id: str | None, limit: int | None) -> list[dict]:
    rows = [r for r in read_csv(SPLITS_PATH) if not split or r["split"] == split]
    if class_id:
        rows = [r for r in rows if r["class_id"] == class_id]
    existing = {p.stem: p for p in NORMALIZED_DIR.rglob("*.wav")}
    rows = [{**r, "path_norm": str(existing[r["file_id"]])} for r in rows if r["file_id"] in existing]
    return rows[:limit] if limit else rows


def write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def merge_scored(existing: list[dict], fresh: list[dict], rescanned_classes: set[str]) -> list[dict]:
    """Gộp điểm mới vào điểm cũ theo file_id — KHÔNG ghi đè toàn bộ file.

    `--class-id X` chỉ sàng lọc lớp X, nhưng write_rows ghi đè cả file: mọi lớp KHÁC
    đã sàng lọc ở lần chạy trước biến mất khỏi screen_scores.csv (và theo đó khỏi
    bank), dù chưa hề bị chạy lại. Bỏ dòng cũ CỦA CHÍNH các lớp vừa quét (để dòng
    file_id không còn tồn tại — ví dụ vì đổi ontology — không nằm lại vĩnh viễn),
    rồi giữ nguyên mọi lớp khác.
    """
    kept = [r for r in existing if r["class_id"] not in rescanned_classes]
    return kept + fresh


def guard_low_resolution_classes(scored: list[dict]) -> dict[str, int]:
    """Huỷ quyết định tự động loại ở những lớp mà tagger không đủ phân giải.

    Luật `p_target < 0.05 → loại` ngầm giả định tagger đáng tin cho lớp đó. Giả định
    ấy SAI với một số lớp, và cái sai không hề lộ ra: clip bị loại lặng lẽ, bank teo
    lại, không ai biết.

    Đo được ngày 14/09 trên bank 5260 clip:

        shout_yell         loại 93%  → PANNs nghe ra "Speech" ở 22/25 clip mẫu,
                                       kèm Groan, Gasp, Sigh, Wail — đều là phát âm
                                       của người. Clip ĐÚNG là tiếng hét; PANNs chỉ
                                       gộp chúng vào lớp tổng quát hơn.
        object_drop_dishes loại 87%  → nghe ra Chink/clink, Coin dropping, Glass,
                                       Percussion — đúng nội dung âm học, sai nhãn.

    Mà những clip này đã qua cổng PP của FSD50K, tức CON NGƯỜI đã chấm nhãn là có mặt
    và nổi trội. Giữa một tagger mAP 0.431 và một phiếu chấm của người, tin người.

    Nên: lớp nào bị loại quá nửa thì mọi quyết định loại của nó chuyển thành "để người
    duyệt". Đắt hơn về công, nhưng cái giá còn lại là xoá nhầm clip đúng — mà với
    `shout_yell` (56 clip, đang thiếu) thì xoá 93% là xoá luôn cả lớp.
    """
    from collections import Counter

    per_class: dict[str, Counter] = {}
    for row in scored:
        per_class.setdefault(row["class_id"], Counter())[row["decision"]] += 1

    cuu = {}
    for class_id, counts in per_class.items():
        total = sum(counts.values())
        if total and counts["auto_reject"] / total > GUARD_REJECT_RATE:
            cuu[class_id] = counts["auto_reject"]

    for row in scored:
        if row["class_id"] in cuu and row["decision"] == "auto_reject":
            row["decision"] = "review"
            row["priority"] = "normal"
    return cuu


def report(scored: list[dict]) -> None:
    from collections import Counter

    counts = Counter(row["decision"] for row in scored)
    total = len(scored)
    print(f"\n{'định tuyến':<26}{'clip':>7}{'tỉ lệ':>9}")
    print("─" * 44)
    for name, label in [("auto_accept", "tự động nhận"), ("auto_reject", "tự động loại"),
                        ("review", "hàng đợi người")]:
        print(f"{label:<26}{counts[name]:>7}{counts[name] / total:>8.1%}")

    high = sum(1 for row in scored if row["priority"] == "high")
    print(f"\n  trong đó ưu tiên CAO (lớp dễ nhầm thắng): {high}")
    print(f"  ước tính công người: {counts['review'] / 150:.1f} giờ ở 150 clip/giờ")
    print(f"\n⚠️ Chưa có tỉ lệ lỗi của auto-accept. Phải duyệt lại 10% mẫu ngẫu nhiên")
    print(f"   ({max(1, counts['auto_accept'] // 10)} clip) rồi ghi con số đó vào báo cáo — DATA_PLAN §4.4.")


def main() -> int:
    enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="foreground_bank_train")
    parser.add_argument("--class-id", help="chỉ sàng lọc một lớp")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--report-only", action="store_true", help="đọc lại screen_scores.csv, không chạy mô hình")
    args = parser.parse_args()

    if args.report_only:
        scored = read_csv(SCORES_PATH)
        if not scored:
            print("chưa có screen_scores.csv — chạy không kèm --report-only trước")
            return 1
        report(scored)
        return 0

    clips = clips_to_screen(args.split, args.class_id, args.limit)
    if not clips:
        print("không có clip nào khớp. Đã chạy make_splits.py và normalize_audio.py chưa?")
        return 1
    print(f"▶ sàng lọc {len(clips)} clip")

    ontology = load_ontology()
    tagger, detector, device = load_tagger()
    from panns_inference.config import labels
    index_of = mid_to_index(list(labels), REPO_ROOT / "data" / "reference" / "audioset_ontology.json")
    print(f"  ghép được {len(index_of)}/{len(labels)} nhãn PANNs theo mã AudioSet")

    scored, reasons = [], {}
    padded_count = 0
    for number, clip in enumerate(clips, 1):
        if number % 100 == 0:
            print(f"  {number}/{len(clips)}…", flush=True)
        audio = load_audio_32k(Path(clip["path_norm"]))
        duration = len(audio) / SAMPLE_RATE
        model_input = pad_to_minimum(audio)
        if len(model_input) > len(audio):
            padded_count += 1
        target, confusable = class_indices(ontology, clip["class_id"], index_of)

        clip_probs = tagger.inference(model_input[None, :])[0][0]
        p_target, p_confusable, top_name = score_clip(clip_probs, target, confusable)
        routing = route(p_target, p_confusable)

        onset = offset = ""
        if routing.decision != "auto_reject" and target:
            frame_probs = detector.inference(model_input[None, :])[0][:, target].max(axis=1)
            # Chỉ đọc phần audio THẬT; phần còn lại là bản lặp do đệm.
            frame_probs = frame_probs[:real_frame_count(len(frame_probs), len(audio))]
            onset, offset = propose_boundary(frame_probs, duration)

        scored.append({**clip, "p_target": round(p_target, 4), "p_confusable": round(p_confusable, 4),
                       "top_confusable": top_name, "decision": routing.decision,
                       "priority": routing.priority, "onset": onset, "offset": offset})
        reasons[clip["file_id"]] = routing.reason

    # Guard chạy TRƯỚC khi ghi exclusions. Ghi trong vòng lặp thì clip được guard cứu
    # vẫn nằm trong exclusions.csv — vừa ở hàng đợi duyệt vừa bị đánh dấu đã loại,
    # đúng loại mâu thuẫn làm dataset không tái lập được.
    cuu = guard_low_resolution_classes(scored)
    for class_id, count in sorted(cuu.items(), key=lambda kv: -kv[1]):
        print(f"  ⚠️ `{class_id}`: tagger loại >{GUARD_REJECT_RATE:.0%} số clip → "
              f"không đủ phân giải cho lớp này, {count} quyết định loại chuyển thành duyệt tay")
    rejected = {row["file_id"]: ("wrong_class", reasons[row["file_id"]])
                for row in scored if row["decision"] == "auto_reject"}

    if padded_count:
        print(f"  {padded_count} clip ngắn hơn 1s — đã kéo dài bằng cách lặp lại "
              f"(đệm im lặng sẽ làm 92% trong số đó bị loại oan, xem pad_to_minimum)")
    write_exclusions(STAGE, rejected, {c["file_id"] for c in clips})
    rescanned_classes = {c["class_id"] for c in clips}
    all_scored = merge_scored(read_csv(SCORES_PATH), scored, rescanned_classes)
    write_rows(SCORES_PATH, SCORE_FIELDS, all_scored)
    queue = sorted((r for r in all_scored if r["decision"] == "review"),
                   key=lambda r: (priority_rank(Routing("review", r["priority"], "")),
                                  r["class_id"], -float(r["p_target"])))
    write_rows(QUEUE_PATH, QUEUE_FIELDS, queue)

    report(scored)
    print(f"\n✓ {SCORES_PATH.relative_to(REPO_ROOT)} · {QUEUE_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
