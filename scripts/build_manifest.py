"""Quét data/raw/ và sinh data/manifests/raw_manifest.csv.

    python scripts/build_manifest.py --source esc50
    python scripts/build_manifest.py --source esc50 --limit 50   # chạy thử

Manifest là nguồn chân lý cho MỌI file audio trong dự án (DATA_PLAN §3.2).
Ba cột không được để trống, vì thiếu chúng là hỏng cả phần thực nghiệm:

  source_group_id  — nhóm chống rò rỉ (N2). Với ESC-50 là id bản ghi Freesound gốc:
                     40 clip `fireworks` chỉ đến từ 16 bản ghi, chia theo file là rò rỉ.
  license          — ESC-50 cấp license THEO TỪNG CLIP (CC0 / CC-BY / CC-BY-NC lẫn lộn),
                     không phải một license chung. Ghi gộp một dòng là ghi sai.
  checksum_sha256  — để phát hiện file đổi/hỏng về sau.

Mỗi nguồn cần một adapter riêng vì cách đặt nhãn và cách nhóm nguồn gốc khác nhau.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import soundfile
import yaml

from audioset_ontology import collapse_to_most_specific, load_parents
from common import REPO_ROOT, enable_utf8_output, write_exclusions

RAW_DIR = REPO_ROOT / "data" / "raw"
MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "raw_manifest.csv"
DEDUP_PATH = REPO_ROOT / "data" / "manifests" / "dedup_groups.csv"
DEDUP_FIELDS = ["checksum_sha256", "n_copies", "representative_file_id", "source_group_ids", "members", "risk"]
ONTOLOGY_PATH = REPO_ROOT / "ml" / "configs" / "ontology_map.yaml"

# License nào cho phép phát hành lại audio, license nào chỉ cho phát hành nhãn.
REDISTRIBUTABLE = {"CC0": "yes", "CC-BY": "yes"}
DEFAULT_REDISTRIBUTABLE = "labels_only"

# Quy URL license về cùng bộ tên ngắn với ESC-50, để cột `license` của manifest
# có MỘT từ vựng duy nhất. Khớp theo thứ tự, lấy mục đầu tiên trúng.
LICENSE_URL_PATTERNS = [
    ("publicdomain/zero", "CC0"),
    ("licenses/by-nc-sa", "CC-BY-NC-SA"),
    ("licenses/by-nc", "CC-BY-NC"),
    ("licenses/by-sa", "CC-BY-SA"),
    ("licenses/sampling+", "CC-Sampling+"),
    ("licenses/by", "CC-BY"),
]

ESC50_LICENSE_RE = re.compile(
    r"^- \[(?P<filename>[^\]]+)\]:.*?\((?P<url>https?://[^)]+)\)\s+by\s+(?P<author>.+?)\s+\[(?P<license>[^\]]+)\]"
)


@dataclass
class RawEntry:
    file_id: str
    path_raw: str
    source_dataset: str
    source_id: str
    source_group_id: str
    claimed_class: str
    label_type_orig: str
    orig_split: str = ""          # split/fold do CHÍNH nguồn quy định, nếu có
    orig_onset: str = ""
    orig_offset: str = ""
    duration: float = 0.0
    sample_rate_orig: int = 0
    channels: int = 0
    license: str = ""
    attribution: str = ""
    redistributable: str = DEFAULT_REDISTRIBUTABLE
    download_date: str = field(default_factory=lambda: date.today().isoformat())
    checksum_sha256: str = ""


# ── Tiện ích ─────────────────────────────────────────────────────────────────


def audio_files_in(directory: Path) -> list[Path]:
    """Mọi file .wav thật trong thư mục, đã bỏ file rác.

    DESED được đóng gói trên macOS nên tarball có kèm file AppleDouble `._*.wav` —
    chúng là metadata vài trăm byte, không phải audio, và sẽ làm soundfile ném lỗi.
    """
    return sorted(p for p in directory.glob("*.wav") if not p.name.startswith("._"))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audio_properties(path: Path) -> tuple[float, int, int]:
    """(thời lượng giây, sample rate, số kênh). Đo thật, không giả định."""
    info = soundfile.info(str(path))
    return round(info.duration, 3), info.samplerate, info.channels


def reverse_label_map(ontology: dict, source_key: str) -> dict[str, str]:
    """{nhãn của nguồn: class_id của ta}, ví dụ {'glass_breaking': 'glass_breaking'}."""
    mapping: dict[str, str] = {}
    for class_id, spec in ontology["classes"].items():
        for label in spec.get(source_key, []):
            mapping[label] = class_id
    return mapping


# ── Adapter: ESC-50 ──────────────────────────────────────────────────────────


def parse_esc50_licenses(license_path: Path) -> dict[str, dict[str, str]]:
    """{tên file không đuôi: {url, author, license}} từ file LICENSE của ESC-50.

    File LICENSE liệt kê license và tác giả cho TỪNG clip — bỏ qua nó là mất
    khả năng ghi công và mất luôn cơ sở pháp lý để công bố.
    """
    entries: dict[str, dict[str, str]] = {}
    for line in license_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ESC50_LICENSE_RE.match(line.strip())
        if match:
            entries[Path(match["filename"]).stem] = {
                "url": match["url"],
                "author": match["author"],
                "license": match["license"],
            }
    return entries


def adapt_esc50(ontology: dict, limit: int | None) -> Iterator[RawEntry]:
    root = RAW_DIR / "esc50" / "ESC-50-master"
    label_map = reverse_label_map(ontology, "esc50_labels")
    licenses = parse_esc50_licenses(root / "LICENSE")
    print(f"  đọc được license của {len(licenses)} clip từ file LICENSE")

    with (root / "meta" / "esc50.csv").open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["category"] in label_map]

    print(f"  {len(rows)} clip thuộc {len(set(label_map.values()))} lớp của ta")
    for row in rows[:limit]:
        audio_path = root / "audio" / row["filename"]
        if not audio_path.exists():
            print(f"  ⚠️ thiếu file {row['filename']}")
            continue
        yield _esc50_entry(row, audio_path, label_map, licenses)


def esc50_license_key(audio_stem: str) -> str:
    """`1-100032-A-0` (tên file audio) → `1-100032-A` (khoá trong file LICENSE).

    Tên file audio là {fold}-{clipID}-{take}-{target}, còn LICENSE bỏ phần {target}.
    Không cắt hậu tố này thì tra cứu trượt 100% và mọi clip mất license lẫn ghi công.
    """
    return audio_stem.rsplit("-", 1)[0]


def _esc50_entry(row: dict, audio_path: Path, label_map: dict, licenses: dict) -> RawEntry:
    checksum = sha256_of(audio_path)
    duration, sample_rate, channels = audio_properties(audio_path)
    info = licenses.get(esc50_license_key(audio_path.stem), {})
    license_name = info.get("license", "UNKNOWN")
    class_id = label_map[row["category"]]

    return RawEntry(
        file_id=f"esc50_{class_id}_{checksum[:8]}",
        path_raw=str(audio_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        source_dataset="esc50",
        source_id=audio_path.stem,
        # src_file = id bản ghi Freesound gốc; nhiều clip cắt ra từ cùng một bản ghi.
        source_group_id=f"freesound_{row['src_file']}",
        claimed_class=class_id,
        label_type_orig="weak",
        orig_split=f"fold{row['fold']}",     # fold kiểm chứng chéo của ESC-50
        duration=duration,
        sample_rate_orig=sample_rate,
        channels=channels,
        license=license_name,
        attribution=f"{info.get('author', '?')} ({info.get('url', '?')})",
        redistributable=REDISTRIBUTABLE.get(license_name, DEFAULT_REDISTRIBUTABLE),
        checksum_sha256=checksum,
    )


# ── Adapter: DESED soundbank ─────────────────────────────────────────────────

# Chỉ lấy `foreground/`. Trong phần eval, ba thư mục foreground/ foreground_on_off/
# foreground_short/ chứa CÙNG MỘT bộ tên file (đã kiểm: trùng 63/63 và 62/62) — đó là
# ba biến thể xử lý của cùng nguồn, lấy cả ba là nhân ba cùng một âm thanh.
DESED_FOREGROUND_DIRS = {
    "train": "audio/train/soundbank/foreground",
    "eval": "audio/eval/soundbank/foreground",
}
DESED_LICENSE_FILES = {"train": "license_training.tsv", "eval": "license_eval.tsv"}


def normalize_license_url(url: str) -> str:
    """URL license → tên ngắn. Trả về 'UNKNOWN' nếu không nhận ra.

    Một số dòng của DESED là link chuyển hướng YouTube có nhúng URL creativecommons
    bên trong dạng mã hoá phần trăm, nên phải giải mã trước khi so khớp.
    """
    from urllib.parse import unquote

    decoded = unquote(url)
    return next((name for pattern, name in LICENSE_URL_PATTERNS if pattern in decoded), "UNKNOWN")


def parse_desed_licenses(root: Path) -> dict[tuple[str, str], dict[str, str]]:
    """{(split, tên file): {id, dataset, license, username}}.

    Cột `filename` là đường dẫn tuyệt đối trên máy của tác giả gốc nên chỉ dùng được
    phần tên file; khoá phải kèm split vì hai split có thể trùng tên file.
    """
    entries: dict[tuple[str, str], dict[str, str]] = {}
    for split, license_file in DESED_LICENSE_FILES.items():
        path = root / license_file
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="	"):
                key = (split, Path(row["filename"].replace("\\", "/")).name)
                entries[key] = {
                    "id": row["id"],
                    "dataset": row["dataset"],
                    "license": normalize_license_url(row["license"]),
                    "username": row.get("username", ""),
                }
    return entries


def adapt_desed_soundbank(ontology: dict, limit: int | None) -> Iterator[RawEntry]:
    root = RAW_DIR / "desed_soundbank"
    label_map = reverse_label_map(ontology, "desed_labels")
    licenses = parse_desed_licenses(root)
    print(f"  đọc được license của {len(licenses)} clip")

    produced = 0
    for split, relative in DESED_FOREGROUND_DIRS.items():
        for desed_label, class_id in sorted(label_map.items()):
            for audio_path in audio_files_in(root / relative / desed_label):
                if limit is not None and produced >= limit:
                    return
                yield _desed_entry(audio_path, split, class_id, licenses)
                produced += 1


def _desed_entry(audio_path: Path, split: str, class_id: str, licenses: dict) -> RawEntry:
    checksum = sha256_of(audio_path)
    duration, sample_rate, channels = audio_properties(audio_path)
    info = licenses.get((split, audio_path.name), {})
    license_name = info.get("license", "UNKNOWN")

    # Tên file dạng `<id nguồn>_<số thứ tự đoạn>.wav` — nhiều đoạn cắt từ cùng một
    # bản ghi gốc, nên phần trước dấu gạch dưới chính là nhóm chống rò rỉ (N2).
    origin = info.get("dataset", "freesound")
    origin_id = info.get("id") or audio_path.name.split("_")[0]

    return RawEntry(
        file_id=f"desed_{class_id}_{checksum[:8]}",
        path_raw=str(audio_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        source_dataset="desed_soundbank",
        source_id=f"{split}/{audio_path.stem}",
        source_group_id=f"{origin}_{origin_id}",
        claimed_class=class_id,
        # Cả file LÀ một sự kiện đã tách sẵn — đó chính là thứ Scaper cần làm nguyên liệu.
        label_type_orig="isolated_event",
        orig_split=split,
        orig_onset="0.0",
        orig_offset=f"{duration:.3f}",
        duration=duration,
        sample_rate_orig=sample_rate,
        channels=channels,
        license=license_name,
        attribution=f"{info.get('username', '?')} ({origin}:{origin_id})",
        redistributable=REDISTRIBUTABLE.get(license_name, DEFAULT_REDISTRIBUTABLE),
        checksum_sha256=checksum,
    )


# ── Adapter: FSD50K ──────────────────────────────────────────────────────────

FSD50K_META = RAW_DIR / "fsd50k_metadata"
FSD50K_AUDIO = {
    "dev": RAW_DIR / "fsd50k_dev_audio" / "FSD50K.dev_audio",
    "eval": RAW_DIR / "fsd50k_eval_audio" / "FSD50K.eval_audio",
}
ONTOLOGY_JSON = REPO_ROOT / "data" / "reference" / "audioset_ontology.json"
FSD50K_STAGE = "select_fsd50k"


def load_fsd50k_rows() -> list[dict]:
    rows = []
    for filename, portion in [("dev.csv", "dev"), ("eval.csv", "eval")]:
        path = FSD50K_META / "FSD50K.ground_truth" / filename
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["portion"] = portion
                rows.append(row)
    return rows


def load_fsd50k_clip_info() -> dict[str, dict]:
    """{fname: {uploader, license, ...}} — `uploader` là nhóm chống rò rỉ của FSD50K."""
    info: dict[str, dict] = {}
    for filename in ["dev_clips_info_FSD50K.json", "eval_clips_info_FSD50K.json"]:
        path = FSD50K_META / "FSD50K.metadata" / filename
        if path.exists():
            info.update(json.loads(path.read_text(encoding="utf-8")))
    return info


def is_predominant(clip_ratings: dict, target_mids: set[str]) -> bool:
    """True khi MỌI phiếu chấm cho nhãn đích đều là PP (âm thanh có mặt và nổi trội).

    Thang điểm của FSD50K: PP=1.0 · PNP=0.5 · U=0 · NP=-1. Foreground bank cần PP:
    âm thanh đích phải nổi trội thì clip mới dùng làm nguyên liệu Scaper được.
    """
    votes = [v for mid, v in clip_ratings.items() if mid in target_mids]
    return bool(votes) and all(all(score == 1.0 for score in v) for v in votes)


def select_fsd50k_clips(ontology: dict) -> tuple[list[dict], dict[str, tuple[str, str]]]:
    """(clip được chọn, clip bị loại kèm lý do).

    Hai cổng, theo thứ tự:
      1. Sau khi THU GỌN PHÂN CẤP, clip chỉ được thuộc đúng một lớp của ta.
         Không thu gọn thì `gunshot`/`fireworks`/`siren` ra 0 clip — xem
         scripts/audioset_ontology.py và DATA_PLAN §2.
      2. Nhãn đích phải được chấm PP.
    """
    mid_to_class = {mid: cls for cls, spec in ontology["classes"].items() for mid in spec["audioset_ids"]}
    parents = load_parents(ONTOLOGY_JSON)
    ratings = json.loads((FSD50K_META / "FSD50K.metadata" / "pp_pnp_ratings_FSD50K.json").read_text(encoding="utf-8"))

    selected, rejected = [], {}
    for row in load_fsd50k_rows():
        ours = {mid for mid in row["mids"].split(",") if mid in mid_to_class}
        if not ours:
            continue
        specific = collapse_to_most_specific(ours, parents)
        classes = {mid_to_class[mid] for mid in specific}
        key = f"fsd50k_{row['fname']}"

        if len(classes) > 1:
            rejected[key] = ("multiple_events", f"nhiều lớp: {sorted(classes)}")
        elif not is_predominant(ratings.get(row["fname"], {}), specific):
            rejected[key] = ("not_predominant", "nhãn không được chấm PP")
        else:
            selected.append({**row, "class_id": classes.pop(), "target_mids": specific})
    return selected, rejected


def adapt_fsd50k(ontology: dict, limit: int | None) -> Iterator[RawEntry]:
    selected, rejected = select_fsd50k_clips(ontology)
    print(f"  {len(selected)} clip đạt hai cổng (đơn lớp sau thu gọn phân cấp + chấm PP)")
    print(f"  {len(rejected)} clip bị loại — đã ghi vào exclusions.csv")
    write_exclusions(FSD50K_STAGE, rejected, set(rejected))

    clip_info = load_fsd50k_clip_info()
    missing = 0
    for row in selected[:limit]:
        audio_path = FSD50K_AUDIO[row["portion"]] / f"{row['fname']}.wav"
        if not audio_path.exists():
            missing += 1
            continue
        yield _fsd50k_entry(row, audio_path, clip_info.get(row["fname"], {}))
    if missing:
        print(f"  ⚠️ {missing} clip chưa có audio — đã tải FSD50K.{row['portion']}_audio chưa?")


def _fsd50k_entry(row: dict, audio_path: Path, info: dict) -> RawEntry:
    checksum = sha256_of(audio_path)
    duration, sample_rate, channels = audio_properties(audio_path)
    license_name = normalize_license_url(info.get("license", ""))
    uploader = info.get("uploader", "unknown")

    return RawEntry(
        file_id=f"fsd50k_{row['class_id']}_{checksum[:8]}",
        path_raw=str(audio_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        source_dataset="fsd50k",
        source_id=row["fname"],
        # FSD50K lấy từ Freesound; DATA_PLAN N2 quy định gom theo NGƯỜI UPLOAD,
        # vì một người thường upload nhiều bản ghi của cùng một nguồn âm.
        source_group_id=f"freesound_uploader_{uploader}",
        claimed_class=row["class_id"],
        label_type_orig="weak",
        orig_split=row.get("split") or row["portion"],
        duration=duration,
        sample_rate_orig=sample_rate,
        channels=channels,
        license=license_name,
        attribution=f"{uploader} (freesound:{row['fname']})",
        redistributable=REDISTRIBUTABLE.get(license_name, DEFAULT_REDISTRIBUTABLE),
        checksum_sha256=checksum,
    )


# ── Adapter: UrbanSound8K ────────────────────────────────────────────────────

US8K_ROOT = RAW_DIR / "urbansound8k" / "UrbanSound8K"
US8K_META = US8K_ROOT / "metadata" / "UrbanSound8K.csv"
US8K_AUDIO = US8K_ROOT / "audio"
US8K_STAGE = "select_urbansound8k"
# License đồng nhất toàn bộ dataset (không per-clip như FSD50K) — xem sources.yaml.
US8K_LICENSE = "CC-BY-NC"
# Cột salience: 1 = foreground, 2 = background. Foreground bank chỉ nhận 1;
# clip salience=2 KHÔNG bị vứt đi mà để dành cho background bank (DATA_PLAN §3).
US8K_FOREGROUND_SALIENCE = "1"


def load_us8k_rows() -> list[dict]:
    with US8K_META.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_us8k_labels(rows: list[dict], label_map: dict[str, str]) -> None:
    """Nhãn ta ánh xạ phải CÓ THẬT trong cột `class` của CSV.

    ontology_map.yaml đánh dấu ⚠️ cho urbansound_labels vì chúng được viết trước
    khi có file gốc. Gõ sai một chữ thì adapter ra 0 clip mà không báo gì — đúng
    kiểu lỗi đã làm `gunshot`/`siren` biến mất ở lần xử lý FSD50K. Nên sai là dừng.
    """
    actual = {row["class"] for row in rows}
    unknown = sorted(set(label_map) - actual)
    if unknown:
        raise SystemExit(
            f"❌ nhãn không có trong {US8K_META.name}: {unknown}\n"
            f"   nhãn thật của UrbanSound8K: {sorted(actual)}\n"
            f"   sửa `urbansound_labels` trong ml/configs/ontology_map.yaml"
        )


def adapt_urbansound8k(ontology: dict, limit: int | None) -> Iterator[RawEntry]:
    label_map = reverse_label_map(ontology, "urbansound_labels")
    rows = load_us8k_rows()
    verify_us8k_labels(rows, label_map)

    rejected: dict[str, tuple[str, str]] = {}
    selected: list[dict] = []
    processed: set[str] = set()
    for row in rows:
        class_id = label_map.get(row["class"])
        if class_id is None:
            continue
        key = f"urbansound8k_{row['slice_file_name']}"
        processed.add(key)
        if row["salience"] != US8K_FOREGROUND_SALIENCE:
            rejected[key] = ("background_salience", "salience=2, để dành background bank")
        else:
            selected.append({**row, "class_id": class_id})

    print(f"  {len(selected)} clip salience=1 thuộc lớp của ta")
    print(f"  {len(rejected)} clip salience=2 — đã ghi vào exclusions.csv")
    write_exclusions(US8K_STAGE, rejected, processed)

    missing = 0
    for row in selected[:limit]:
        audio_path = US8K_AUDIO / f"fold{row['fold']}" / row["slice_file_name"]
        if not audio_path.exists():
            missing += 1
            continue
        yield _us8k_entry(row, audio_path)
    if missing:
        print(f"  ⚠️ {missing} clip chưa có audio — đã giải nén UrbanSound8K.tar.gz chưa?")


def _us8k_entry(row: dict, audio_path: Path) -> RawEntry:
    checksum = sha256_of(audio_path)
    duration, sample_rate, channels = audio_properties(audio_path)

    return RawEntry(
        file_id=f"urbansound8k_{row['class_id']}_{checksum[:8]}",
        path_raw=str(audio_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        source_dataset="urbansound8k",
        source_id=row["slice_file_name"],
        # `fsID` là id bản ghi Freesound gốc; nhiều lát cắt dùng chung một fsID nên
        # đây chính là nhóm chống rò rỉ (DATA_PLAN N2).
        source_group_id=f"freesound_{row['fsID']}",
        claimed_class=row["class_id"],
        # Lát cắt 4 giây CÓ CHỨA sự kiện, không phải đã cắt sát biên sự kiện.
        # Vì vậy để trống orig_onset/orig_offset thay vì khai cả file là sự kiện.
        label_type_orig="weak_with_salience",
        orig_split=f"fold{row['fold']}",
        duration=duration,
        sample_rate_orig=sample_rate,
        channels=channels,
        license=US8K_LICENSE,
        attribution=f"freesound:{row['fsID']}",
        redistributable=REDISTRIBUTABLE.get(US8K_LICENSE, DEFAULT_REDISTRIBUTABLE),
        checksum_sha256=checksum,
    )


# ── Adapter: vehicle_crash_cc (qua HF hub, xem fetch_vehicle_crash_cc.py) ────
# Thay thế MIVIA Road làm nguồn chính cho `vehicle_crash` (15/09/2026, xem sources.yaml).

VCC_ROOT = RAW_DIR / "vehicle_crash_cc"
VCC_METADATA = VCC_ROOT / "metadata.csv"
VCC_LICENSE = "CC-BY"
VCC_CLASS = "vehicle_crash"


def adapt_vehicle_crash_cc(ontology: dict, limit: int | None) -> Iterator[RawEntry]:
    """Toàn bộ nguồn chỉ có một lớp (`label` trong metadata.csv luôn là "Car Crash"),
    nên không cần tra `ontology` qua reverse_label_map như các nguồn nhiều lớp khác.

    `video_id` PHẢI dùng làm source_group_id: vài video góp nhiều clip (hậu tố `_00`,
    `_01`... trong `file_name`) — chia theo file là rò rỉ N2, đúng kiểu lỗi đã bắt được
    ở DESED (hai id Freesound trỏ cùng audio) và FSD50K (chung uploader).
    """
    if not VCC_METADATA.exists():
        return
    with VCC_METADATA.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    produced = 0
    for row in rows:
        if limit is not None and produced >= limit:
            return
        audio_path = VCC_ROOT / row["file_name"]
        if not audio_path.exists():
            continue
        yield _vcc_entry(row, audio_path)
        produced += 1


def _vcc_entry(row: dict, audio_path: Path) -> RawEntry:
    checksum = sha256_of(audio_path)
    duration, sample_rate, channels = audio_properties(audio_path)

    return RawEntry(
        file_id=f"vehicle_crash_cc_{checksum[:8]}",
        path_raw=str(audio_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        source_dataset="vehicle_crash_cc",
        source_id=row["file_name"],
        source_group_id=f"youtube_{row['video_id']}",
        claimed_class=VCC_CLASS,
        # File đã tải VỐN LÀ đúng đoạn [start_sec, end_sec] của video gốc — coi cả file
        # là một sự kiện đã cắt sẵn, giống DESED soundbank, KHÔNG dùng lại start_sec làm
        # onset (đó là mốc trong video gốc, không phải trong file đã tải).
        label_type_orig="isolated_event",
        orig_onset="0.0",
        orig_offset=f"{duration:.3f}",
        duration=duration,
        sample_rate_orig=sample_rate,
        channels=channels,
        license=VCC_LICENSE,
        attribution=f"{row.get('channel', '?')} (youtube:{row['video_id']})",
        redistributable=REDISTRIBUTABLE.get(VCC_LICENSE, DEFAULT_REDISTRIBUTABLE),
        checksum_sha256=checksum,
    )


# ── Adapter: AudioSet-strong (qua yt-dlp, xem fetch_audioset_strong.py) ──────

AS_STRONG_ROOT = RAW_DIR / "audioset_strong"
AS_STRONG_SEGMENTS = AS_STRONG_ROOT / "segments.jsonl"
AS_STRONG_LICENSE = "YouTube"


def load_audioset_strong_segments() -> list[dict]:
    if not AS_STRONG_SEGMENTS.exists():
        return []
    with AS_STRONG_SEGMENTS.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def adapt_audioset_strong(ontology: dict, limit: int | None) -> Iterator[RawEntry]:
    """Một ổ 10 giây có thể chứa NHIỀU sự kiện — không giống các adapter khác,
    một dòng manifest ở đây là MỘT SỰ KIỆN, và nhiều dòng có thể trỏ cùng một file
    .wav. `source_group_id` = ytid nên các dòng cùng video luôn đi cùng một split
    (make_splits.py gộp theo group, không theo file) — cắt rời chúng ra là rò rỉ.
    """
    segments = load_audioset_strong_segments()
    produced = 0
    missing = 0
    for segment in segments:
        audio_path = REPO_ROOT / segment["path"]
        if not audio_path.exists():
            missing += 1
            continue
        checksum = None
        for event in segment["events"]:
            if limit is not None and produced >= limit:
                if missing:
                    print(f"  ⚠️ {missing} ổ chưa có audio — đã chạy fetch_audioset_strong.py chưa?")
                return
            if checksum is None:
                checksum = sha256_of(audio_path)
                duration, sample_rate, channels = audio_properties(audio_path)
            yield RawEntry(
                file_id=f"as_strong_{event['class_id']}_{segment['file_id']}_{event['onset']}",
                path_raw=str(audio_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                source_dataset="audioset_strong",
                source_id=segment["file_id"],
                source_group_id=f"youtube_{segment['ytid']}",
                claimed_class=event["class_id"],
                label_type_orig="strong",
                orig_onset=f"{event['onset']:.3f}",
                orig_offset=f"{event['offset']:.3f}",
                duration=duration,
                sample_rate_orig=sample_rate,
                channels=channels,
                license=AS_STRONG_LICENSE,
                attribution=f"youtube:{segment['ytid']}",
                redistributable="labels_only",
                checksum_sha256=checksum,
            )
            produced += 1
    if missing:
        print(f"  ⚠️ {missing} ổ chưa có audio — đã chạy fetch_audioset_strong.py chưa?")


ADAPTERS = {
    "esc50": adapt_esc50,
    "desed_soundbank": adapt_desed_soundbank,
    "fsd50k": adapt_fsd50k,
    "urbansound8k": adapt_urbansound8k,
    "audioset_strong": adapt_audioset_strong,
    "vehicle_crash_cc": adapt_vehicle_crash_cc,
}


# ── Ghi manifest ─────────────────────────────────────────────────────────────


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["file_id"]: row for row in csv.DictReader(handle)}


def drop_stale_rows(rows: dict[str, dict], source: str, limit: int | None) -> dict[str, dict]:
    """Bỏ dòng cũ của `source` trước khi nạp lại — chỉ khi quét TOÀN BỘ nguồn.

    file_id chứa claimed_class, nên một lần đổi ánh xạ ontology (tách/đổi tên lớp)
    làm file_id đổi theo; `dict.update()` theo file_id không bao giờ xoá được dòng
    cũ mang class_id đã mất — nó nằm lại vĩnh viễn, trùng checksum với dòng mới.
    Với --limit (chạy thử) thì KHÔNG xoá: xoá lúc đó sẽ mất luôn phần chưa quét tới.
    """
    if limit is not None:
        return rows
    return {fid: row for fid, row in rows.items() if row["source_dataset"] != source}


def write_manifest(path: Path, rows: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(RawEntry.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows.values(), key=lambda r: r["file_id"]):
            writer.writerow(row)


def find_exact_duplicates(all_rows: list[dict], new_entries: list[RawEntry]) -> list[dict]:
    """Nhóm các file GIỐNG NHAU TỪNG BYTE, kể cả khác dataset hay khác nhóm nguồn.

    Đây không phải chuyện lý thuyết: DESED chứa hai id Freesound khác nhau (13613 và
    34853) trỏ tới cùng một audio. Chia tập theo `source_group_id` sẽ xếp chúng vào
    hai nhóm khác nhau, và cùng một âm thanh nằm ở cả train lẫn test — đúng kiểu rò rỉ
    bắc cầu mà DATA_PLAN §6 cảnh báo. Phải gộp nhóm nguồn TRƯỚC khi chia tập.
    """
    by_checksum: dict[str, list[dict]] = {}
    for row in all_rows:
        by_checksum.setdefault(row["checksum_sha256"], []).append(row)

    # Manifest chỉ giữ MỘT dòng cho mỗi checksum, nên mọi bản sao khác đã bị dict gộp
    # mất — kể cả `source_group_id` của chúng. Phải lấy lại từ danh sách vừa quét,
    # nếu không sẽ báo nhầm là "trùng trong cùng nhóm" và bỏ lọt đúng ca nguy hiểm.
    extra_paths: dict[str, set[str]] = {}
    extra_groups: dict[str, set[str]] = {}
    for entry in new_entries:
        extra_paths.setdefault(entry.checksum_sha256, set()).add(entry.path_raw)
        extra_groups.setdefault(entry.checksum_sha256, set()).add(entry.source_group_id)

    groups = []
    for checksum, rows in sorted(by_checksum.items()):
        paths = {r["path_raw"] for r in rows} | extra_paths.get(checksum, set())
        if len(paths) < 2:
            continue
        group_ids = sorted({r["source_group_id"] for r in rows} | extra_groups.get(checksum, set()))
        groups.append({
            "checksum_sha256": checksum,
            "n_copies": len(paths),
            "representative_file_id": rows[0]["file_id"],
            "source_group_ids": "|".join(group_ids),
            "members": "|".join(sorted(paths)),
            # Trùng trong cùng một nhóm nguồn thì vô hại; trùng BẮC CẦU giữa hai
            # nhóm nguồn mới là thứ làm hỏng phép chia tập.
            "risk": "cross_group" if len(group_ids) > 1 else "same_group",
        })
    return groups


def merge_dedup_groups(existing: list[dict], found: list[dict]) -> list[dict]:
    """Gộp nhóm trùng mới tìm được vào nhóm đã biết, khoá theo checksum.

    KHÔNG được ghi đè. Manifest chỉ giữ MỘT dòng cho mỗi checksum, nên bản trùng chỉ
    còn tồn tại trong chính file này; quét một nguồn khác sẽ không tìm lại được nhóm
    của nguồn cũ (mọi bản sao của chúng đã bị manifest gộp mất) và ghi đè sẽ xoá sạch
    bằng chứng. Đã xảy ra thật: chạy adapter urbansound8k làm biến mất 8 nhóm bắc cầu
    của DESED, và check_leakage vẫn báo xanh vì nó đọc chính file rỗng đó.

    Trùng byte là quan hệ bất biến — nhóm tìm được hôm qua vẫn đúng hôm nay — nên giữ
    lại là an toàn; chỉ lần quét mới mới được cập nhật nhóm cùng checksum.
    """
    by_checksum = {row["checksum_sha256"]: row for row in existing}
    by_checksum.update({row["checksum_sha256"]: row for row in found})
    return [by_checksum[key] for key in sorted(by_checksum)]


def write_dedup_groups(groups: list[dict]) -> None:
    DEDUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = load_dedup_groups()
    rows = merge_dedup_groups(existing, groups)
    kept = len(rows) - len(groups)
    if kept > 0:
        print(f"  giữ lại {kept} nhóm trùng của các nguồn khác đã quét trước đó")
    with DEDUP_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEDUP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_dedup_groups() -> list[dict]:
    if not DEDUP_PATH.exists():
        return []
    with DEDUP_PATH.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def report_duplicates(groups: list[dict]) -> None:
    if not groups:
        return
    cross = [g for g in groups if g["risk"] == "cross_group"]
    print(f"✓ {DEDUP_PATH.relative_to(REPO_ROOT)}: {len(groups)} nhóm trùng khớp từng byte"
          f" ({len(cross)} nhóm BẮC CẦU giữa các nhóm nguồn)")
    if not cross:
        return
    print("  ⚠️ Nhóm bắc cầu PHẢI được gộp trước khi chia tập, nếu không là rò rỉ:")
    for group in cross[:5]:
        print(f"     {group['source_group_ids']}  ({group['n_copies']} bản sao)")


def summarise(entries: list[RawEntry]) -> None:
    by_class: dict[str, list[RawEntry]] = {}
    for entry in entries:
        by_class.setdefault(entry.claimed_class, []).append(entry)

    print(f"\n{'lớp':<22}{'clip':>6}{'nhóm nguồn':>12}{'phút':>8}  license")
    print("─" * 78)
    for class_id, items in sorted(by_class.items()):
        groups = len({i.source_group_id for i in items})
        minutes = sum(i.duration for i in items) / 60
        licenses = ", ".join(sorted({i.license for i in items}))
        print(f"{class_id:<22}{len(items):>6}{groups:>12}{minutes:>8.1f}  {licenses[:32]}")

    total_groups = len({e.source_group_id for e in entries})
    print("─" * 78)
    print(f"{'TỔNG':<22}{len(entries):>6}{total_groups:>12}{sum(e.duration for e in entries) / 60:>8.1f}")

    unknown = [e for e in entries if e.license == "UNKNOWN"]
    if unknown:
        print(f"\n⚠️ {len(unknown)} clip không đọc được license — phải xử lý trước khi công bố")


def main() -> int:
    enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, choices=sorted(ADAPTERS), help="nguồn cần quét")
    parser.add_argument("--limit", type=int, help="chỉ xử lý N file đầu (để chạy thử)")
    args = parser.parse_args()

    ontology = yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    print(f"▶ quét nguồn {args.source}")

    entries = list(ADAPTERS[args.source](ontology, args.limit))
    if not entries:
        print("Không tìm thấy file nào. Đã tải nguồn này chưa?")
        return 1

    rows = load_existing(MANIFEST_PATH)
    before = len(rows)
    rows = drop_stale_rows(rows, args.source, args.limit)
    rows.update({e.file_id: asdict(e) for e in entries})
    write_manifest(MANIFEST_PATH, rows)

    groups = find_exact_duplicates(list(rows.values()), entries)
    write_dedup_groups(groups)

    summarise(entries)
    print(f"\n✓ {MANIFEST_PATH.relative_to(REPO_ROOT)}: {before} → {len(rows)} dòng")
    report_duplicates(groups)
    return 0


if __name__ == "__main__":
    sys.exit(main())
