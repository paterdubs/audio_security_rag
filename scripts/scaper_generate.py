"""Sinh train/dev synthetic kèm strong label bằng Scaper — DATA_PLAN §7.

    python scripts/scaper_generate.py --split train
    python scripts/scaper_generate.py --split dev --limit 20
    python scripts/scaper_generate.py --split train --plan-only

Đây là bước sinh strong label MIỄN PHÍ: Scaper biết chính xác nó đặt sự kiện nào ở
đâu, nên onset/offset là sự thật tuyệt đối chứ không phải ước lượng của người gán.

Ba điều script này làm mà gọi thẳng Scaper không làm được:
  1. CHẶN RÒ RỈ ở đầu vào. Chỉ clip được chia vào foreground_bank_train mới được
     dùng làm nguyên liệu. Một clip của gold_test lọt vào đây là hỏng cả bài test,
     và không có triệu chứng nào ngoài việc điểm số đẹp lên.
  2. ÉP TỈ LỆ SLICE. overlap/low_snr/reverb/long_event/causal_chain là các lát cắt
     dùng để báo cáo; để phân phối tự nhiên quyết định thì chúng quá thưa.
  3. CHUỖI NHÂN QUẢ, gồm hai chuỗi ÂM TÍNH (pháo hoa + reo hò, rơi bát đĩa + cười).
     Thiếu chúng model học "cứ có chuỗi sự kiện là nguy hiểm".

Cần SoX binary — xem requirements-data.txt.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from common import REPO_ROOT, enable_utf8_output

CONFIG_PATH = REPO_ROOT / "ml" / "configs" / "scaper_train.yaml"
SPLITS_PATH = REPO_ROOT / "data" / "manifests" / "splits.csv"
FOREGROUND_DIR = REPO_ROOT / "data" / "banks" / "foreground"
BACKGROUND_DIR = REPO_ROOT / "data" / "banks" / "background"
RIR_DIR = REPO_ROOT / "data" / "banks" / "rir"
OUTPUT_DIR = REPO_ROOT / "data" / "synthetic"

BANK_SPLIT = "foreground_bank_train"
SLICE_NAMES = ["overlap", "low_snr", "reverb", "long_event", "causal_chain"]
# `reverb` là phép tích chập áp lên cả clip nên clip im lặng vẫn mang được;
# bốn lát còn lại định nghĩa trên sự kiện nên không.
EVENT_BASED_SLICES = {"overlap", "low_snr", "long_event", "causal_chain"}


@dataclass
class ClipPlan:
    """Công thức của một clip, quyết định TRƯỚC khi đụng tới audio.

    Tách hẳn khỏi phần sinh audio để kiểm thử được: mọi lỗi tỉ lệ slice hay chuỗi
    nhân quả đều nằm ở đây, mà phần này không cần SoX, không cần bank, chạy trong
    mili giây.
    """

    index: int
    slices: set[str] = field(default_factory=set)
    n_events: int = 0
    chain: str | None = None
    area_type: str = ""
    rir_used: str = ""


# ── Cổng chống rò rỉ ─────────────────────────────────────────────────────────


def bank_eligible_ids() -> set[str]:
    """file_id được phép làm nguyên liệu. Đọc từ splits.csv, không đoán theo thư mục."""
    if not SPLITS_PATH.exists():
        raise SystemExit("❌ chưa có data/manifests/splits.csv — chạy scripts/make_splits.py trước")
    with SPLITS_PATH.open(encoding="utf-8", newline="") as handle:
        return {row["file_id"] for row in csv.DictReader(handle) if row["split"] == BANK_SPLIT}


def check_bank_clean(bank_files: list[Path], eligible: set[str]) -> list[str]:
    """file trong bank foreground mà KHÔNG thuộc foreground_bank_train.

    Trả danh sách vi phạm thay vì ném lỗi, để gọi được từ test và để báo cáo liệt kê
    hết một lượt chứ không dừng ở cái đầu tiên.
    """
    return sorted(p.stem for p in bank_files if p.stem not in eligible)


# ── Lập kế hoạch ─────────────────────────────────────────────────────────────


def sample_event_count(rng: random.Random, weights: dict) -> int:
    """Số sự kiện trong clip. Nhánh 0 sự kiện (~10%) là nhánh dạy model biết im lặng."""
    counts = sorted(int(k) for k in weights)
    return rng.choices(counts, weights=[float(weights[k]) for k in counts])[0]


def silent_probability(config: dict) -> float:
    weights = config["events"]["count_weights"]
    total = sum(float(v) for v in weights.values())
    return float(weights.get(0, weights.get("0", 0.0))) / total


def slice_probabilities(config: dict) -> dict[str, float]:
    """Xác suất rút cho từng lát cắt, đã bù phần clip im lặng.

    Tỉ lệ trong config là tỉ lệ trên TOÀN BỘ split — đó là con số sẽ viết vào báo
    cáo. Nhưng 4 trong 5 lát cắt định nghĩa trên sự kiện nên không gán được cho
    clip im lặng (~10%). Rút thẳng theo tỉ lệ đích sẽ cho kết quả thấp hơn đích
    đúng bằng 10%: đo được 26.9% trong khi tài liệu ghi 30%. Chênh lệch nhỏ và
    không có triệu chứng, nên chia cho phần clip có sự kiện ngay tại đây.
    """
    forced = config["forced_slices"]
    with_events = 1.0 - silent_probability(config)
    probability = {}
    for name in SLICE_NAMES:
        ratio = float(forced[name]["ratio"])
        if name in EVENT_BASED_SLICES:
            if ratio > with_events:
                raise SystemExit(
                    f"❌ lát cắt `{name}` đòi {ratio:.0%} nhưng chỉ {with_events:.0%} clip có sự kiện.\n"
                    f"   Giảm ratio, hoặc giảm trọng số clip 0 sự kiện trong scaper_train.yaml."
                )
            ratio /= with_events
        probability[name] = ratio
    return probability


def plan_clips(config: dict, n_clips: int, seed: int, chains: list[dict] | None = None,
               areas: list[str] | None = None) -> list[ClipPlan]:
    """Kế hoạch cho n_clips, tất định theo seed.

    Mỗi slice rút Bernoulli ĐỘC LẬP theo tỉ lệ của nó, nên một clip có thể vừa
    overlap vừa reverb — đúng ý đồ: các lát cắt trong DATA_PLAN chồng lấn nhau,
    tổng tỉ lệ của chúng vượt 100%. Chia rời thành 5 nhóm loại trừ nhau sẽ làm sai
    tỉ lệ từng lát và không sinh được ca khó nhất (low_snr + overlap cùng lúc).
    """
    forced = config["forced_slices"]
    chains = config["causal_chains"] if chains is None else chains
    rng = random.Random(seed)
    probability = slice_probabilities(config)

    plans = []
    for index in range(n_clips):
        plan = ClipPlan(index=index)
        # Khu vực nền chọn TRƯỚC, vì loại phòng dùng để tích chập phải khớp với nó.
        plan.area_type = rng.choice(areas) if areas else ""
        plan.n_events = sample_event_count(rng, config["events"]["count_weights"])
        silent = plan.n_events == 0
        for name in SLICE_NAMES:
            # Clip im lặng không mang được lát cắt nào định nghĩa trên sự kiện.
            if silent and name in EVENT_BASED_SLICES:
                continue
            if rng.random() < probability[name]:
                plan.slices.add(name)

        if "overlap" in plan.slices:
            plan.n_events = max(plan.n_events, int(forced["overlap"]["min_events"]))
        if "causal_chain" in plan.slices and not chains:
            plan.slices.discard("causal_chain")     # không còn chuỗi nào sinh được
        if "causal_chain" in plan.slices:
            plan.chain = rng.choice(chains)["name"]
            plan.n_events = max(plan.n_events, len(chain_by_name(chains, plan.chain)["sequence"]))
        plans.append(plan)
    return plans


def usable_chains(chains: list[dict], available: set[str]) -> tuple[list[dict], list[tuple[str, list[str]]]]:
    """(chuỗi dùng được, [(tên chuỗi bỏ, các lớp còn thiếu)]).

    Chuỗi nhân quả chỉ có nghĩa khi ĐỦ mọi mắt xích. Thiếu một lớp thì hoặc Scaper
    chết giữa chừng, hoặc — tệ hơn — ta lặng lẽ sinh ra chuỗi cụt mang nhãn của chuỗi
    đủ: `glass_breaking → scream` vẫn được ghi là kịch bản "đột nhập" dù thiếu hẳn
    tiếng chân chạy, và model học một định nghĩa sai về đột nhập.

    Bỏ hẳn chuỗi và NÓI RA là lựa chọn đúng: thiếu `shout_yell` thì hai kịch bản
    forced_entry và emergency không sinh được, và đó là thông tin phải biết chứ không
    phải chi tiết cần giấu.
    """
    ok, bo = [], []
    for chain in chains:
        thieu = [label for label in chain["sequence"] if label not in available]
        (bo.append((chain["name"], thieu)) if thieu else ok.append(chain))
    return ok, bo


def available_labels() -> set[str]:
    """Các lớp THẬT SỰ có clip trong bank foreground — đọc từ đĩa, không từ config."""
    if not FOREGROUND_DIR.exists():
        return set()
    return {p.name for p in FOREGROUND_DIR.iterdir() if p.is_dir() and any(p.glob("*.wav"))}


def chain_by_name(chains: list[dict], name: str) -> dict:
    for chain in chains:
        if chain["name"] == name:
            return chain
    raise SystemExit(f"❌ không có chuỗi tên `{name}` trong scaper_train.yaml")


def chain_event_times(chain: dict, rng: random.Random, duration: float,
                      event_duration: float = 1.0) -> list[float]:
    """Thời điểm bắt đầu của từng sự kiện trong chuỗi, giữ đúng THỨ TỰ và khoảng cách.

    Thứ tự mới là thứ mang nghĩa: `glass_breaking → scream → running_footsteps` là
    đột nhập, còn đảo ngược thì không. Nếu chuỗi tràn khỏi clip thì co khoảng cách
    về mức nhỏ nhất; vẫn tràn thì trả rỗng để nơi gọi bỏ qua chuỗi này, chứ không
    cắt cụt chuỗi thành một chuỗi khác nghĩa.
    """
    gaps_spec = chain.get("gaps_sec") or []
    gaps = [rng.uniform(low, high) for low, high in gaps_spec]
    need = lambda gs: event_duration * len(chain["sequence"]) + sum(gs)  # noqa: E731

    if need(gaps) > duration:
        gaps = [low for low, _ in gaps_spec]          # co về khoảng cách nhỏ nhất
    if need(gaps) > duration:
        return []

    start = rng.uniform(0, duration - need(gaps))
    times, cursor = [], start
    for position in range(len(chain["sequence"])):
        times.append(round(cursor, 3))
        if position < len(gaps):
            cursor += event_duration + gaps[position]
    return times


def slice_report(plans: list[ClipPlan], config: dict) -> str:
    lines = [f"{'lát cắt':<16}{'đích':>8}{'thực tế':>10}{'clip':>8}"]
    lines.append("─" * 42)
    for name in SLICE_NAMES:
        target = float(config["forced_slices"][name]["ratio"])
        hit = sum(1 for p in plans if name in p.slices)
        lines.append(f"{name:<16}{target:>7.0%}{hit / len(plans):>10.1%}{hit:>8}")
    silent = sum(1 for p in plans if p.n_events == 0)
    lines.append(f"\n{silent} clip không có sự kiện ({silent / len(plans):.1%}) — nhánh dạy model biết im lặng")
    return "\n".join(lines)


# ── Sinh audio ───────────────────────────────────────────────────────────────


# Nền của khu vực nào thì phải mang tiếng vang của khu vực đó. Ghép ngẫu nhiên sẽ cho
# ra clip nền nhà xe mà vang như phòng học — model học được rằng vang và bối cảnh
# không liên quan gì nhau, trong khi ngoài đời chúng đi liền.
AREA_TO_RIR_SPACE = {
    "school": "medium_room",        # hành lang, sảnh lớp
    "parking": "large_room",        # nhà xe, vang dài
    "residential": "small_room",    # nhà ở, sân nhỏ
    "factory": "large_room",        # xưởng
}


def apply_rir(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Tích chập clip với RIR, giữ NGUYÊN độ dài và mức to.

    Hai điều bắt buộc, cả hai đều âm thầm nếu làm sai:

    1. CẮT VỀ ĐÚNG ĐỘ DÀI CŨ. Tích chập cho ra tín hiệu dài thêm bằng đuôi RIR (tới 2
       giây). Không cắt thì clip 10 giây thành 12 giây, trong khi .jams vẫn mô tả một
       clip 10 giây — mọi mốc thời gian sau đó lệch.
    2. GIỮ MỨC TO. RIR đã chuẩn hoá theo năng lượng nên về lý thuyết mức giữ nguyên,
       nhưng cộng hưởng có thể đẩy đỉnh vượt 1.0. Hạ xuống theo đỉnh, KHÔNG chuẩn hoá
       lại toàn bộ — chuẩn hoá lại sẽ phá tỉ số SNR mà Scaper vừa đặt.
    """
    wet = np.convolve(audio, rir, mode="full")[: len(audio)]
    peak = float(np.abs(wet).max())
    if peak > 0.99:
        wet = wet * (0.99 / peak)
    return wet.astype(np.float32)


def pick_rir(rng: random.Random, area_type: str) -> Path | None:
    """Một RIR ngẫu nhiên thuộc loại phòng khớp với khu vực nền."""
    space = AREA_TO_RIR_SPACE.get(area_type)
    folder = RIR_DIR / space if space else None
    if not folder or not folder.exists():
        return None
    files = sorted(folder.glob("*.wav"))
    return rng.choice(files) if files else None



def class_dirs(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and any(p.glob("*.wav"))) if root.exists() else []


def build_scaper(config: dict, seed: int):
    import scaper

    generator = scaper.Scaper(
        duration=float(config["duration_sec"]),
        fg_path=str(FOREGROUND_DIR),
        bg_path=str(BACKGROUND_DIR),
        random_state=seed,
    )
    generator.ref_db = int(config["ref_db"])
    generator.sr = int(config["sample_rate"])
    return generator


def populate(generator, plan: ClipPlan, config: dict, rng: random.Random,
             chains: list[dict] | None = None) -> None:
    """Nạp nền + các sự kiện của một clip vào generator theo đúng kế hoạch."""
    chains = config["causal_chains"] if chains is None else chains
    events = config["events"]
    duration = float(config["duration_sec"])
    snr_low, snr_high = events["snr_db"]
    if "low_snr" in plan.slices:
        snr_high = float(config["forced_slices"]["low_snr"]["max_snr_db"])

    # Chỉ đích danh khu vực thay vì để Scaper bốc ngẫu nhiên: kế hoạch đã chọn khu vực
    # rồi, và RIR sắp tích chập phải khớp với chính khu vực đó.
    area = ("const", plan.area_type) if plan.area_type else ("choose", [])
    generator.add_background(label=area, source_file=("choose", []), source_time=("const", 0))

    placed = 0
    if plan.chain:
        chain = chain_by_name(chains, plan.chain)
        for label, start in zip(chain["sequence"], chain_event_times(chain, rng, duration)):
            _add_event(generator, label, ("const", start), events, (snr_low, snr_high))
            placed += 1

    for _ in range(plan.n_events - placed):
        _add_event(generator, None, ("uniform", 0, duration), events, (snr_low, snr_high))


def _add_event(generator, label: str | None, event_time, events: dict, snr: tuple[float, float]) -> None:
    pitch_low, pitch_high = events["pitch_shift_semitones"]
    stretch_low, stretch_high = events["time_stretch"]
    generator.add_event(
        label=("const", label) if label else ("choose", []),
        source_file=("choose", []),
        source_time=("const", 0),
        event_time=event_time,
        event_duration=("const", 1.0),
        snr=("uniform", float(snr[0]), float(snr[1])),
        pitch_shift=("uniform", float(pitch_low), float(pitch_high)),
        time_stretch=("uniform", float(stretch_low), float(stretch_high)),
    )


def generate_split(config: dict, split: str, plans: list[ClipPlan], seed: int,
                   chains: list[dict] | None = None) -> int:
    out = OUTPUT_DIR / split
    for sub in ("audio", "jams", "tsv"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    import soundfile

    reverb_ok = bool(class_dirs(RIR_DIR))
    if not reverb_ok:
        print("  ⚠️ chưa có bank RIR — bỏ lát cắt `reverb`, sẽ phải sinh lại khi có")

    rng = random.Random(seed)
    written = 0
    for plan in plans:
        name = f"{split}_{plan.index:06d}"
        generator = build_scaper(config, seed + plan.index)
        populate(generator, plan, config, rng, chains)
        audio_path = out / "audio" / f"{name}.wav"
        generator.generate(
            audio_path=str(audio_path),
            jams_path=str(out / "jams" / f"{name}.jams"),
            # CỐ Ý để None: `reverb` của Scaper là hiệu ứng vang tổng hợp của SoX, KHÔNG
            # phải tích chập RIR. DATA_PLAN §7 yêu cầu tích chập, và nó khác hẳn — RIR
            # đo tại phòng thật mang cả hình dạng phản xạ sớm lẫn cách phòng hút tần số
            # cao, thứ mà hiệu ứng tổng hợp không tái tạo được. Tích chập làm ở dưới.
            reverb=None,
            # BẮT BUỘC. Nền ở -23 LUFS cộng SNR tối đa +25 dB cho sự kiện ở +2 dBFS,
            # tức méo cứng. Đo thật trên 5 clip đầu: 3/5 clip méo, tới 4016 mẫu chạm
            # trần. Sinh ra train set mà chính cổng chất lượng của ta sẽ loại (>0.1%
            # mẫu cắt phẳng → `clipped`) là tự mâu thuẫn, và méo là thứ model học
            # được rất nhanh: nó sẽ dùng méo làm dấu hiệu nhận biết sự kiện to.
            # fix_clipping hạ đều cả soundscape nên GIỮ NGUYÊN tỉ số SNR đã đặt.
            fix_clipping=True,
        )

        if "reverb" in plan.slices and reverb_ok:
            rir_path = pick_rir(rng, plan.area_type)
            if rir_path:
                audio, rate = soundfile.read(str(audio_path))
                rir, _ = soundfile.read(str(rir_path))
                soundfile.write(audio_path, apply_rir(audio, rir), rate, subtype="PCM_16")
                plan.rir_used = rir_path.stem

        written += 1
        if written % 100 == 0:
            print(f"  {written}/{len(plans)}…", flush=True)
    return written


def write_plan_index(split: str, plans: list[ClipPlan]) -> None:
    """Ghi kế hoạch ra JSONL để biết clip nào thuộc lát cắt nào khi báo cáo kết quả."""
    path = OUTPUT_DIR / split / "slice_index.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for plan in plans:
            handle.write(json.dumps({
                "clip_id": f"{split}_{plan.index:06d}",
                "slices": sorted(plan.slices),
                "n_events": plan.n_events,
                "chain": plan.chain,
                "area_type": plan.area_type,
                "rir_used": plan.rir_used,
            }, ensure_ascii=False) + "\n")


# ── Chạy ─────────────────────────────────────────────────────────────────────


def main() -> int:
    enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "dev"], required=True)
    parser.add_argument("--limit", type=int, help="chỉ sinh N clip đầu (thử nhanh)")
    parser.add_argument("--plan-only", action="store_true", help="chỉ lập kế hoạch + kiểm tra, không sinh audio")
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    split_config = config["splits"][args.split]
    seed = int(config["seed"]) + int(split_config.get("seed_offset", 0))

    n_clips = int(float(split_config["target_hours"]) * 3600 / float(config["duration_sec"]))
    if args.limit:
        n_clips = min(n_clips, args.limit)

    co_san = available_labels()
    chains, bo_chuoi = usable_chains(config["causal_chains"], co_san)
    for name, thieu in bo_chuoi:
        print(f"⚠️ bỏ chuỗi `{name}` — bank chưa có lớp {thieu}")
    if bo_chuoi:
        print("   Sinh chuỗi cụt mà vẫn mang nhãn của chuỗi đủ sẽ dạy model một định")
        print("   nghĩa sai về kịch bản đó, nên bỏ hẳn và ghi lại ở đây.")
        print()

    areas = class_dirs(BACKGROUND_DIR)
    plans = plan_clips(config, n_clips, seed, chains, areas)
    print(f"▶ {args.split}: {n_clips} clip × {config['duration_sec']}s "
          f"= {n_clips * float(config['duration_sec']) / 3600:.1f} h · seed {seed}\n")
    print(slice_report(plans, config))

    foreground = sorted(FOREGROUND_DIR.rglob("*.wav"))
    if not foreground:
        print(f"\n❌ bank foreground rỗng ({FOREGROUND_DIR.relative_to(REPO_ROOT)}).")
        print("   Cần chạy auto_screen.py + người duyệt trước — DATA_PLAN §4.2–4.3.")
        return 1

    violations = check_bank_clean(foreground, bank_eligible_ids())
    if violations:
        print(f"\n❌ {len(violations)} file trong bank KHÔNG thuộc {BANK_SPLIT}:")
        for name in violations[:5]:
            print(f"      {name}")
        print("   Đây là rò rỉ vào train set. Sửa bank hoặc chạy lại make_splits.py.")
        return 1
    print(f"\n✓ {len(foreground)} clip bank đều thuộc {BANK_SPLIT}")

    if not class_dirs(BACKGROUND_DIR):
        print(f"❌ bank background rỗng ({BACKGROUND_DIR.relative_to(REPO_ROOT)}) — DATA_PLAN §5.")
        return 1

    if args.plan_only:
        print("\n(--plan-only: chưa sinh audio)")
        return 0

    written = generate_split(config, args.split, plans, seed, chains)
    # Ghi SAU khi sinh: `rir_used` chỉ có giá trị sau bước tích chập. Ghi trước thì
    # cột đó luôn rỗng và không truy ngược được clip nào dùng RIR nào khi phân tích lỗi.
    write_plan_index(args.split, plans)
    print(f"\n✓ đã sinh {written} clip → {(OUTPUT_DIR / args.split).relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
