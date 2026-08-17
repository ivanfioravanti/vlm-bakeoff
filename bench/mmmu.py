from __future__ import annotations

import ast
import json
import random
from pathlib import Path

from bench import IMAGES, TASKS

SEED = 42
SUBSET_N = 300

# MMMU validation split: ~900 items across 30 college subjects (art, law,
# physics, ...), each a multiple-choice question with up to 4 options and
# 1-3 images (diagrams, charts, photos). Liquid blog (vLLM): 48.4 on val.
# The config list is discovered at runtime so subject renames upstream don't
# silently drop a track.
HF_ID = "MMMU/MMMU"
SPLIT = "validation"

LIQUID_MMMU_VAL = 0.484


def build_mmmu(split: str | int = "full") -> list[dict]:
    from datasets import get_dataset_config_names, load_dataset

    configs = sorted(get_dataset_config_names(HF_ID))
    rows: list[tuple[str, int, dict]] = []
    for config in configs:
        ds = load_dataset(HF_ID, config, split=SPLIT)
        for idx in range(len(ds)):
            rows.append((config, idx, ds[idx]))
        print(f"  mmmu {config}: {len(ds)} items")

    mode, subset_n = "full", None
    if split == "subset":
        mode, subset_n = "subset", SUBSET_N
    elif isinstance(split, int) and not isinstance(split, bool) and split > 0:
        mode, subset_n = "subset", min(split, len(rows))
    elif split != "full":
        raise ValueError(f"mmmu split must be 'full', 'subset' or a positive int, got {split!r}")
    indices = list(range(len(rows)))
    if subset_n is not None:
        rng = random.Random(SEED)
        rng.shuffle(indices)
        indices = indices[:subset_n]

    out_dir = IMAGES / "mmmu"
    out_dir.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()

    tasks: list[dict] = []
    for n, idx in enumerate(indices):
        config, row_idx, row = rows[idx]
        raw_options = row.get("options")
        if isinstance(raw_options, str):
            # A slice of MMMU rows ship options as the string literal of a
            # list ("['Project A', 'Project B']"); without this the string
            # explodes into one single-character option per letter.
            try:
                raw_options = ast.literal_eval(raw_options)
            except (ValueError, SyntaxError):
                raw_options = [raw_options]
        options = [str(o) for o in (raw_options or []) if o is not None and str(o).strip()]
        answer = str(row.get("answer") or "").strip().upper()
        question = str(row.get("question") or "").strip()
        if not options or answer not in [chr(ord("A") + i) for i in range(len(options))]:
            continue
        if not question:
            continue

        images: list[str] = []
        for k in range(1, 8):
            img = row.get(f"image_{k}")
            if img is None:
                break
            fname = f"{config}_{row_idx}_{k}.jpg"
            target = out_dir / fname
            if not target.exists():
                img.convert("RGB").save(target)
            used.add(fname)
            images.append(f"mmmu/{fname}")
        # A slice of MMMU questions is text-only; keep them so the denominator
        # matches the published 900-item val accuracy (the harness handles
        # zero-image tasks on both backends).

        lines = "\n".join(f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options))
        prompt = (
            f"{question}\n{lines}\n"
            "Answer with the option's letter from the given choices directly."
        )
        meta: dict = {
            "subject": config,
            "topic": str(row.get("topic") or ""),
            "question_type": "multi_choice",
            "split_mode": mode,
        }
        if subset_n is not None:
            meta["subset_n"] = subset_n
        tasks.append(
            {
                "id": f"mmmu_val_{config}_{row_idx}",
                "category": "mmmu",
                "images": images,
                "prompt": prompt,
                "scorer": "mc_option",
                "expected": {"gold": answer, "n": len(options)},
                "max_tokens": 32,
                "meta": meta,
            }
        )
        if (n + 1) % 100 == 0 or (n + 1) == len(indices):
            print(f"  mmmu tasks {n + 1}/{len(indices)}")

    for stale in out_dir.iterdir():
        if stale.is_file() and stale.name not in used:
            stale.unlink()

    TASKS.mkdir(parents=True, exist_ok=True)
    (TASKS / "mmmu.json").write_text(json.dumps(tasks, indent=2) + "\n")
    return tasks
