from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from bench import IMAGES, TASKS

SEED = 42
PER_TASK = 16  # subset mode: 14 tasks x 16 = 224 items

# Official BLINK repo: one config per perceptual task, each with a small val
# split (117-172 items; 1,901 total — so 'full' is affordable too). Items
# carry 1-4 images, a canonical formatted prompt with lettered choices, and
# the gold answer as "(B)". Evaluated on val, overall accuracy.
HF_ID = "BLINK-Benchmark/BLINK"
TASK_CONFIGS = (
    "Art_Style",
    "Counting",
    "Forensic_Detection",
    "Functional_Correspondence",
    "IQ_Test",
    "Jigsaw",
    "Multi-view_Reasoning",
    "Object_Localization",
    "Relative_Depth",
    "Relative_Reflectance",
    "Semantic_Correspondence",
    "Spatial_Relation",
    "Visual_Correspondence",
    "Visual_Similarity",
)

# Liquid blog (vLLM 0.26): BLINK overall accuracy, normalized to 0-100.
LIQUID_BLINK = 0.615


def build_blink(split: str = "subset") -> list[dict]:
    if split not in ("full", "subset"):
        raise ValueError(f"blink split must be 'full' or 'subset', got {split!r}")
    from datasets import load_dataset

    rng = random.Random(SEED)
    out_dir = IMAGES / "blink"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[dict] = []
    for config in TASK_CONFIGS:
        ds = load_dataset(HF_ID, config, split="val")
        indices = list(range(len(ds)))
        if split == "subset":
            rng.shuffle(indices)
            indices = indices[:PER_TASK]
        for idx in indices:
            row = ds[idx]
            images = []
            for k in range(1, 5):
                img = row.get(f"image_{k}")
                if img is None:
                    continue
                fname = f"{config}_{idx}_{k}.jpg"
                image = img.convert("RGB")
                image.save(out_dir / fname)
                images.append(f"blink/{fname}")
            if not images:
                continue
            answer = str(row.get("answer") or "").strip()
            if not answer:
                continue
            tasks.append(
                {
                    "id": f"blink_{config}_{idx}",
                    "category": "blink",
                    "images": images,
                    "prompt": str(row["prompt"]).strip(),
                    "scorer": "choice_letter",
                    "expected": {"text": answer},
                    "max_tokens": 16,
                    "meta": {
                        "task": str(row.get("sub_task") or config.replace("_", " ")).strip(),
                        "config": config,
                        "answer": answer,
                        "split_mode": split,
                    },
                }
            )
        print(f"  blink {config}: {len(ds) if split == 'full' else min(PER_TASK, len(ds))} tasks")

    TASKS.mkdir(parents=True, exist_ok=True)
    (TASKS / "blink.json").write_text(json.dumps(tasks, indent=2) + "\n")
    return tasks
