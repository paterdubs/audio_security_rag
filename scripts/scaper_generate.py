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


def plan_clips(config: dict, n_clips: int, seed: int) -> list[ClipPlan]:
    """Kế hoạch cho n_clips, tất định theo seed.

    Mỗi slice rút Bernoulli ĐỘC LẬP theo tỉ lệ của nó, nên một clip có thể vừa
    overlap vừa reverb — đúng ý đồ: các lát cắt trong DATA_PLAN chồng lấn nhau,
    tổng tỉ lệ của chúng vượt 100%. Chia rời thành 5 nhóm loại trừ nhau sẽ làm sai
    tỉ lệ từng lát và không sinh được ca khó nhất (low_snr + overlap cùng lúc).
    """
    forced = config["forced_slices"]
    chains = config["causal_chains"]
    rng = random.Random(seed)
    probability = slice_probabilities(config)

    plans = []
    for index in range(n_clips):
        plan = ClipPlan(index=index)
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
        if "causal_chain" in plan.slices:
            plan.chain = rng.choice(chains)["name"]
            plan.n_events = max(plan.n_events, len(chain_by_name(chains, plan.chain)["sequence"]))
        plans.append(plan)
    return plans


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


def populate(generator, plan: ClipPlan, config: dict, rng: random.Random) -> None:
    """Nạp nền + các sự kiện của một clip vào generator theo đúng kế hoạch."""
    events = config["events"]
    duration = float(config["duration_sec"])
    snr_low, snr_high = events["snr_db"]
    if "low_snr" in plan.slices:
        snr_high = float(config["forced_slices"]["low_snr"]["max_snr_db"])

    generator.add_background(label=("choose", []), source_file=("choose", []), source_time=("const", 0))

    placed = 0
    if plan.chain:
        chain = chain_by_name(config["causal_chains"], plan.chain)
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


def generate_split(config: dict, split: str, plans: list[ClipPlan], seed: int) -> int:
    out = OUTPUT_DIR / split
    for sub in ("audio", "jams", "tsv"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    reverb_ok = bool(class_dirs(RIR_DIR))
    if not reverb_ok:
        print("  ⚠️ chưa có bank RIR — bỏ lát cắt `reverb`, sẽ phải sinh lại khi có")

    rng = random.Random(seed)
    written = 0
    for plan in plans:
        name = f"{split}_{plan.index:06d}"
        generator = build_scaper(config, seed + plan.index)
        populate(generator, plan, config, rng)
        generator.generate(
            audio_path=str(out / "audio" / f"{name}.wav"),
            jams_path=str(out / "jams" / f"{name}.jams"),
            reverb=0.4 if ("reverb" in plan.slices and reverb_ok) else None,
        )
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

    plans = plan_clips(config, n_clips, seed)
    print(f"▶ {args.split}: {n_clips} clip × {config['duration_sec']}s "
          f"= {n_clips * float(config['duration_sec']) / 3600:.1f} h · seed {seed}\n")
    print(slice_report(plans, config))
    write_plan_index(args.split, plans)

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

    written = generate_split(config, args.split, plans, seed)
    print(f"\n✓ đã sinh {written} clip → {(OUTPUT_DIR / args.split).relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
