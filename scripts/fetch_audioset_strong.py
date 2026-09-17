"""Tải nhãn strong-label của AudioSet và audio tương ứng qua yt-dlp — DATA_PLAN §6.

    python scripts/fetch_audioset_strong.py --labels-only     # chỉ tải 3 file TSV, vài trăm KB
    python scripts/fetch_audioset_strong.py                   # + tải audio (cần yt-dlp, ffmpeg)
    python scripts/fetch_audioset_strong.py --limit 20         # chạy thử

Đây là nguồn DUY NHẤT cho `dev`/`gold_test` không phụ thuộc MIVIA (đang chờ duyệt) hay
buổi thu thực địa IUH (chưa diễn ra). Không có nguồn này, cổng D7 (gold_test rỗng) không
qua được và không đo được EHR/EOR/GS/TOA/CHR — tức mất luôn đóng góp M5 của khoá luận.

Ba điều khác biệt so với các nguồn khác trong sources.yaml:
  1. Nhãn là STRONG THẬT (onset/offset theo giây), nhưng đơn vị audio ta tải là một
     Ổ SỔ 10 GIÂY (segment_id = f"{ytid}_{window_start_ms}"), và MỘT ổ có thể chứa
     NHIỀU sự kiện — khác các adapter khác vốn một file = một sự kiện. Vì vậy kết quả
     ở đây không phải RawEntry trực tiếp: script ghi ra segments.jsonl, để
     build_manifest.adapt_audioset_strong sinh NHIỀU dòng manifest trỏ cùng một file .wav.
  2. Audio thuộc về chủ video YouTube, KHÔNG được phát hành lại — chỉ công bố YTID +
     mốc thời gian (đã ghi trong sources.yaml). Vì vậy segments.jsonl chỉ chứa audio
     tải cho MÁY CỦA TA dùng, không đưa vào data công bố.
  3. 15–30% video dự kiến không tải được (đã gỡ / chặn vùng / private). Đây là hụt dữ
     liệu BÌNH THƯỜNG của nguồn YouTube, không phải lỗi — nhưng PHẢI ghi vào
     exclusions.csv với mã `unavailable` và báo cáo trong khoá luận (N3).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import soundfile
import yaml

from common import REPO_ROOT, enable_utf8_output, write_exclusions

CONFIG_PATH = REPO_ROOT / "ml" / "configs" / "sources.yaml"
ONTOLOGY_PATH = REPO_ROOT / "ml" / "configs" / "ontology_map.yaml"
RAW_ROOT = REPO_ROOT / "data" / "raw" / "audioset_strong"
LABELS_DIR = RAW_ROOT / "labels"
AUDIO_DIR = RAW_ROOT / "audio"
SEGMENTS_PATH = RAW_ROOT / "segments.jsonl"

WINDOW_SECONDS = 10.0          # AudioSet strong nhãn theo cửa sổ 10 giây gốc của AudioSet
MAX_DURATION_DRIFT_SEC = 1.0    # lệch quá mức này thì coi như cắt sai, loại
STAGE = "fetch_audioset_strong"

# Trần thời gian cho MỘT clip. Không có trần, một video treo là khoá cả job vĩnh viễn —
# đã xảy ra thật 17/09/2026: yt-dlp + ffmpeg đứng im 56 phút, tốn 1,3 giây CPU, job
# không tải thêm được clip nào và cũng không báo lỗi gì. 10 giây audio qua mạng bình
# thường mất vài giây; 180s là rộng rãi cho cả video dài phải seek lâu.
DOWNLOAD_TIMEOUT_SEC = 180


# ── Tải 3 file TSV nhãn ───────────────────────────────────────────────────────


def download_labels(urls: list[str]) -> None:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    for url in urls:
        dest = LABELS_DIR / url.rsplit("/", 1)[-1]
        if dest.exists():
            continue
        print(f"  ↓ {dest.name}")
        with urllib.request.urlopen(url, timeout=60) as response:
            dest.write_bytes(response.read())


# ── Phân tích nhãn ───────────────────────────────────────────────────────────


def load_mid_to_display_name(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) >= 2 and row[0].startswith("/"):
                mapping[row[0]] = row[1]
    return mapping


def parse_segment_id(segment_id: str) -> tuple[str, int]:
    """"<ytid>_<window_start_ms>" → (ytid, window_start_ms).

    YTID của YouTube có thể chứa dấu gạch dưới, nên phải tách từ BÊN PHẢI —
    tách từ bên trái cắt nhầm giữa ytid, làm sai luôn cả video lẫn mốc thời gian.
    """
    ytid, _, start_ms = segment_id.rpartition("_")
    return ytid, int(start_ms)


def load_strong_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def mid_to_class_map(ontology: dict, known_mids: set[str]) -> dict[str, str]:
    """{mid: class_id}, giới hạn ở các mid THỰC SỰ có trong bộ vocab strong-label.

    audioset_ids trong ontology_map.yaml gồm cả mid chỉ có ở tập weak (527 lớp) —
    đối chiếu với known_mids (456 lớp strong) trước khi dùng, kẻo lọc ra rỗng mà
    không ai biết vì sao.
    """
    mapping: dict[str, str] = {}
    for class_id, spec in ontology["classes"].items():
        for mid in spec.get("audioset_ids", []):
            if mid in known_mids:
                mapping[mid] = class_id
    return mapping


def select_segments(rows: list[dict], mid_to_class: dict[str, str]) -> dict[tuple[str, int], list[dict]]:
    """{(ytid, window_start_ms): [{class_id, onset, offset}]} — chỉ nhãn khớp 16 lớp.

    File TSV thật của AudioSet strong CHỈ có 4 cột (segment_id, start_time_seconds,
    end_time_seconds, label) — KHÔNG có cột present/uncertain như bản đặc tả cũ từng
    giả định. Mọi dòng trong file đã LÀ nhãn strong xác nhận, không cần lọc thêm theo
    độ chắc chắn. (Sửa 16/09: giả định sai này làm select_segments() luôn trả rỗng.)
    """
    segments: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        class_id = mid_to_class.get(row["label"])
        if class_id is None:
            continue
        ytid, window_start_ms = parse_segment_id(row["segment_id"])
        key = (ytid, window_start_ms)
        segments.setdefault(key, []).append({
            "class_id": class_id,
            "onset": round(float(row["start_time_seconds"]), 3),
            "offset": round(float(row["end_time_seconds"]), 3),
        })
    return segments


# ── Tải audio bằng yt-dlp ────────────────────────────────────────────────────


def kill_process_tree(pid: int) -> None:
    """Giết cả cây tiến trình, không chỉ tiến trình cha."""
    if sys.platform == "win32":
        # /T = cả cây con, /F = cưỡng bức. Không có /T thì ffmpeg sống sót thành mồ côi
        # và tiếp tục giữ file đích lẫn pipe.
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        import os
        import signal
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def run_with_hard_timeout(command: list[str], timeout: float) -> int | None:
    """Chạy lệnh với trần thời gian THẬT SỰ cưỡng chế được. None = quá hạn.

    Vì sao không dùng thẳng `subprocess.run(..., timeout=, capture_output=True)`:
    đã thử và KHÔNG ĐỦ (đo thật 17/09/2026, job đứng 34,7 phút với ffmpeg chỉ tốn 0,64 s
    CPU). yt-dlp sinh ffmpeg làm tiến trình CHÁU, và tiến trình cháu thừa hưởng hai đầu
    pipe stdout/stderr. Khi hết giờ, Python giết yt-dlp rồi gọi communicate() để dọn, mà
    communicate() chờ pipe ĐÓNG — pipe chỉ đóng khi ffmpeg chết, nên chính đường thoát
    của timeout bị khoá lại. Trần thời gian có chạy nhưng không thoát ra được.

    Hai thay đổi chữa tận gốc: (1) DEVNULL thay cho PIPE nên không còn pipe để chờ;
    (2) giết cả CÂY tiến trình chứ không chỉ tiến trình cha.
    """
    popen_kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if sys.platform != "win32":
        popen_kwargs["start_new_session"] = True     # để killpg gom đúng nhóm

    process = subprocess.Popen(command, **popen_kwargs)
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_process_tree(process.pid)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        return None


def download_window(ytid: str, window_start_ms: int, dest: Path) -> bool:
    """Tải đúng 10 giây [start, start+10) của video bằng yt-dlp. True nếu thành công.

    --download-sections cắt TRƯỚC KHI tải hết video — video 30 phút mà chỉ cần
    10 giây, tải cả video là phí băng thông gấp trăm lần không cần thiết.

    --force-keyframes-at-cuts BẮT BUỘC: không có nó, ffmpeg copy-stream cắt tại
    keyframe GẦN NHẤT (có thể lệch hàng giây), cho ra file dài gấp đôi mà không báo
    lỗi gì — đã đo thật: một clip ra 19.994s thay vì 10s, làm onset/offset trong
    segments.jsonl (tính theo cửa sổ [0,10] giả định) không còn khớp audio thật.

    Kiểm lại ĐỘ DÀI THẬT sau khi tải, không chỉ tin exit code — exit code 0 không
    đảm bảo cắt đúng vị trí, chỉ đảm bảo ffmpeg không crash.
    """
    start = window_start_ms / 1000
    end = start + WINDOW_SECONDS
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "yt-dlp", "-f", "bestaudio",
        "--download-sections", f"*{start}-{end}",
        "--force-keyframes-at-cuts",
        "--extract-audio", "--audio-format", "wav",
        "-o", str(dest.with_suffix("")) + ".%(ext)s",
        "--no-playlist", "--quiet", "--no-warnings",
        # socket-timeout chặn ở tầng mạng, trần dưới đây chặn ở tầng tiến trình.
        "--socket-timeout", "30",
        f"https://www.youtube.com/watch?v={ytid}",
    ]
    returncode = run_with_hard_timeout(command, DOWNLOAD_TIMEOUT_SEC)
    if returncode is None:
        # Quá hạn. Bỏ file dở: một wav cụt vẫn "tồn tại", nên lần chạy sau sẽ tưởng đã
        # tải xong và nhận một clip hỏng vào manifest.
        dest.unlink(missing_ok=True)
        return False
    if returncode != 0 or not dest.exists():
        return False
    try:
        duration = soundfile.info(str(dest)).duration
    except Exception:
        dest.unlink(missing_ok=True)
        return False
    if abs(duration - WINDOW_SECONDS) > MAX_DURATION_DRIFT_SEC:
        dest.unlink(missing_ok=True)
        return False
    return True


def stratified_keys(segments: dict[tuple[str, int], list[dict]], per_class_limit: int) -> list[tuple[str, int]]:
    """Chọn tối đa `per_class_limit` ổ cho MỖI lớp, không lấy phẳng N ổ đầu.

    66229 ổ tổng, nhưng lệch rất mạnh (speech_normal 54068 ổ, fireworks chỉ 728) —
    lấy N ổ đầu theo thứ tự bất kỳ sẽ toàn `speech_normal`/`laughter`, bỏ đói mọi lớp
    hiếm. Một ổ có thể thoả nhiều lớp cùng lúc, nên tổng số ổ chọn được có thể ÍT HƠN
    n_lớp × per_class_limit — đó là bình thường, không phải lỗi.
    """
    by_class: dict[str, list[tuple[str, int]]] = {}
    for key, events in segments.items():
        for event in events:
            by_class.setdefault(event["class_id"], []).append(key)

    selected: set[tuple[str, int]] = set()
    for class_id in sorted(by_class):
        keys = sorted(set(by_class[class_id]))
        selected.update(keys[:per_class_limit])
    return sorted(selected)


# ── Điều phối ────────────────────────────────────────────────────────────────


def load_existing_segment_ids() -> set[str]:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    if not SEGMENTS_PATH.exists():
        return set()
    with SEGMENTS_PATH.open(encoding="utf-8") as handle:
        return {json.loads(line)["file_id"] for line in handle if line.strip()}


def append_one_segment(row: dict, seen: set[str]) -> None:
    """Ghi NGAY một dòng sau khi tải xong, không dồn tới cuối `run()`.

    Bug đã xảy ra thật (17/09/2026, tắt máy giữa lúc chạy): bản cũ gom `new_rows` trong
    RAM rồi chỉ ghi MỘT LẦN sau khi vòng lặp xử lý hết TOÀN BỘ `keys`. Tiến trình bị giết
    giữa đường (crash, mất điện, TaskStop) làm mọi file .wav đã tải nằm trên đĩa nhưng
    KHÔNG có dòng nào trong segments.jsonl trỏ tới — 375 file, 3 dòng. Ghi từng dòng ngay
    khi có, append-only + `flush()`, biến mỗi lần tải thành một đơn vị không thể mất.
    """
    if row["file_id"] in seen:
        return
    with SEGMENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
    seen.add(row["file_id"])


def run(limit: int | None, labels_only: bool, per_class_limit: int | None) -> int:
    sources = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["sources"]["audioset_strong"]
    ontology = yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8"))

    download_labels(sources["label_urls"])
    train_path = LABELS_DIR / "audioset_train_strong.tsv"
    eval_path = LABELS_DIR / "audioset_eval_strong.tsv"
    vocab_path = LABELS_DIR / "mid_to_display_name.tsv"

    mid_to_name = load_mid_to_display_name(vocab_path)
    mid_to_class = mid_to_class_map(ontology, set(mid_to_name))
    print(f"  {len(mid_to_class)} mã AudioSet strong khớp với 16 lớp của ta")

    rows = load_strong_rows(train_path) + load_strong_rows(eval_path)
    segments = select_segments(rows, mid_to_class)
    print(f"  {len(segments)} ổ 10 giây có ít nhất một sự kiện thuộc 16 lớp")

    if labels_only:
        return 0

    keys = stratified_keys(segments, per_class_limit) if per_class_limit is not None else sorted(segments)
    if limit is not None:
        keys = keys[:limit]

    # Trần cho exclusions.csv (không dùng cho segments.jsonl — cái đó luôn ghi ngay,
    # xem append_one_segment). write_exclusions viết lại TOÀN BỘ file mỗi lần gọi nên
    # gọi mỗi vòng là lãng phí; nhưng chỉ gọi ở cuối thì crash giữa đường mất hết bản ghi
    # lỗi của phiên này — cân bằng bằng cách flush định kỳ.
    EXCLUSION_FLUSH_EVERY = 20

    seen_ids = load_existing_segment_ids()
    ok, failed, processed_ids = 0, {}, set()
    for index, (ytid, window_start_ms) in enumerate(keys):
        file_id = f"as_strong_{ytid}_{window_start_ms}"
        dest = AUDIO_DIR / f"{file_id}.wav"
        processed_ids.add(file_id)
        if not dest.exists():
            if not download_window(ytid, window_start_ms, dest):
                failed[file_id] = ("unavailable", f"yt-dlp thất bại: {ytid}")
                continue
        ok += 1
        append_one_segment(
            {
                "file_id": file_id,
                "path": str(dest.relative_to(REPO_ROOT)).replace("\\", "/"),
                "ytid": ytid,
                "window_start_ms": window_start_ms,
                "events": segments[(ytid, window_start_ms)],
            },
            seen_ids,
        )
        if (index + 1) % EXCLUSION_FLUSH_EVERY == 0:
            write_exclusions(STAGE, failed, processed_ids)

    write_exclusions(STAGE, failed, processed_ids)
    print(f"  tải được {ok}/{len(keys)} — {len(failed)} lỗi ghi vào exclusions.csv")
    return 0


def main() -> int:
    enable_utf8_output()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="chặn TỔNG số ổ tải, sau khi phân tầng")
    parser.add_argument("--per-class-limit", type=int, default=None,
                        help="tối đa bấy nhiêu ổ CHO MỖI LỚP (phân tầng) — 66229 ổ tổng lệch rất mạnh")
    parser.add_argument("--labels-only", action="store_true")
    args = parser.parse_args()
    return run(args.limit, args.labels_only, args.per_class_limit)


if __name__ == "__main__":
    sys.exit(main())
