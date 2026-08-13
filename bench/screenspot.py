from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from bench import IMAGES, TASKS
from bench.prompts import SCREENSPOT_PROMPT

HF_ID = "HongxinLi/ScreenSpot_v2"
SEED = 42
PER_PLATFORM = 16

# Liquid blog (vLLM 0.26). 80.7 is the unweighted mean of these three.
LIQUID_SCREENSPOT_V2 = {"desktop": 0.787, "mobile": 0.812, "web": 0.822}
LIQUID_SCREENSPOT_AVG = sum(LIQUID_SCREENSPOT_V2.values()) / 3

# Official OS-Copilot / VLMEvalKit split is the file_name prefix
# (mobile 501, desktop/pc 334, web 437 = 1,272).
_PREFIX = {
    "mobile": "mobile",
    "pc": "desktop",
    "desktop": "desktop",
    "web": "web",
}


def platform_of(file_name: str, data_source: str = "") -> str:
    prefix = Path(file_name or "").name.split("_", 1)[0].lower()
    if prefix in _PREFIX:
        return _PREFIX[prefix]
    s = (data_source or "").lower()
    if s in ("ios", "android", "mobile"):
        return "mobile"
    if s in ("macos", "windows", "linux", "desktop"):
        return "desktop"
    return "web"


def _to_1000(bbox: list[float], w: int, h: int) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    if max(x1, y1, x2, y2) <= 1.01:
        return [int(x1 * 1000), int(y1 * 1000), int(x2 * 1000), int(y2 * 1000)]
    return [
        int(x1 / w * 1000),
        int(y1 / h * 1000),
        int(x2 / w * 1000),
        int(y2 / h * 1000),
    ]


def build_screenspot(split: str = "full") -> list[dict]:
    if split not in ("full", "subset"):
        raise ValueError(f"screenspot split must be 'full' or 'subset', got {split!r}")
    from datasets import load_dataset

    ds = load_dataset(HF_ID, split="test")
    indices = list(range(len(ds)))
    if split == "subset":
        buckets: dict[str, list[int]] = defaultdict(list)
        for i in indices:
            row = ds[i]
            buckets[platform_of(row.get("file_name") or "", row.get("data_source") or "")].append(i)
        rng = random.Random(SEED)
        chosen: list[int] = []
        for name in ("mobile", "desktop", "web"):
            idxs = list(buckets.get(name, []))
            rng.shuffle(idxs)
            chosen.extend(idxs[:PER_PLATFORM])
        indices = chosen

    out_dir = IMAGES / "screenspot"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: set[str] = set()
    tasks: list[dict] = []
    for n, idx in enumerate(indices):
        row = ds[idx]
        image = row["image"].convert("RGB")
        w, h = image.size
        fname = Path(str(row.get("file_name") or f"{idx:04d}.png")).name
        if fname not in saved:
            image.save(out_dir / fname)
            saved.add(fname)
        gold = _to_1000(list(row["bbox"]), w, h)
        instruction = str(row["instruction"]).strip()
        src = str(row.get("data_source") or "")
        plat = platform_of(str(row.get("file_name") or ""), src)
        dtype = str(row.get("data_type") or "")
        tasks.append(
            {
                "id": f"screenspot_{idx:04d}_{plat}_{dtype or 'na'}",
                "category": "screenspot",
                "images": [f"screenspot/{fname}"],
                "prompt": SCREENSPOT_PROMPT.format(instruction=instruction),
                "scorer": "click_in_box",
                "expected": {"bbox": gold},
                "max_tokens": 64,
                "meta": {
                    "platform": plat,
                    "data_source": src,
                    "data_type": dtype,
                    "hf_index": idx,
                    "split": split,
                    "file_name": fname,
                    "instruction": instruction,
                },
            }
        )
        if (n + 1) % 100 == 0 or n + 1 == len(indices):
            print(f"  screenspot images {n + 1}/{len(indices)}")

    TASKS.mkdir(parents=True, exist_ok=True)
    (TASKS / "screenspot.json").write_text(json.dumps(tasks, indent=2) + "\n")
    return tasks
