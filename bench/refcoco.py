from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from bench import IMAGES, TASKS
from bench.prompts import GROUNDING_JSON_SYSTEM, REFERCOCO_USER
from bench.screenspot import _to_1000

SEED = 42
PER_SPLIT = 64  # subset mode: 8 splits x 64 = 512 items

# lmms-lab renamed their grounding sets (lmms-lab/RefCOCO → lmms-lab-encoder);
# the rows are one per region: full COCO image, gold bbox in COCO [x, y, w, h]
# pixels, and the region's referring expressions in `answer`. RefCOCO's extra
# 5,000-row `test` split is not one of the splits behind Liquid's RefCOCO-avg,
# so it is skipped.
DATASETS: dict[str, tuple[str, tuple[str, ...]]] = {
    "refcoco": ("lmms-lab-encoder/RefCOCO", ("val", "testA", "testB")),
    "refcocoplus": ("lmms-lab-encoder/RefCOCOplus", ("val", "testA", "testB")),
    "refcocog": ("lmms-lab-encoder/RefCOCOg", ("val", "test")),
}

SPLITS = tuple(f"{name}/{sp}" for name, (_, splits) in DATASETS.items() for sp in splits)

# Liquid blog (vLLM 0.26): 87.9 is the unweighted mean of precision@1 over the
# 8 splits in SPLITS. Per-split numbers are not published, so only the average
# has a reference value.
LIQUID_REFERCOCO_AVG = 0.879


def build_refcoco(split: str = "subset") -> list[dict]:
    if split not in ("full", "subset"):
        raise ValueError(f"refcoco split must be 'full' or 'subset', got {split!r}")
    from datasets import load_dataset

    rng = random.Random(SEED)
    out_dir = IMAGES / "refcoco"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: set[str] = set()
    tasks: list[dict] = []
    skipped = 0
    for name, (hf_id, split_names) in DATASETS.items():
        for split_name in split_names:
            ds = load_dataset(hf_id, split=split_name)
            indices = list(range(len(ds)))
            if split == "subset":
                rng.shuffle(indices)
                indices = indices[:PER_SPLIT]
            n_split = 0
            for idx in indices:
                row = ds[idx]
                answers = [a for a in (row.get("answer") or []) if str(a).strip()]
                if not answers:
                    skipped += 1
                    continue
                image = row["image"].convert("RGB")
                w, h = image.size
                x, y, bw, bh = [float(v) for v in list(row["bbox"])[:4]]
                x2, y2 = x + bw, y + bh
                # Rows must carry the full COCO image the bbox refers to; a box
                # escaping the frame means a cropped/mismatched image — drop it.
                if x2 > w + 1 or y2 > h + 1 or x2 <= x or y2 <= y:
                    skipped += 1
                    continue
                fname = Path(str(row.get("file_name") or f"{idx:06d}.jpg")).name
                if fname not in saved:
                    image.save(out_dir / fname)
                    saved.add(fname)
                expression = str(answers[0]).strip()
                qid = row.get("question_id")
                tasks.append(
                    {
                        "id": f"refcoco_{name}_{split_name}_{qid if qid is not None else idx}",
                        "category": "refcoco",
                        "images": [f"refcoco/{fname}"],
                        "prompt": REFERCOCO_USER.format(expression=expression),
                        "system": GROUNDING_JSON_SYSTEM,
                        "scorer": "click_in_box",
                        "expected": {"bbox": _to_1000([x, y, x2, y2], w, h)},
                        "max_tokens": 64,
                        "meta": {
                            "subset": f"{name}/{split_name}",
                            "expression": expression,
                            "question_id": qid,
                            "file_name": fname,
                            "split_mode": split,
                        },
                    }
                )
                n_split += 1
            print(f"  refcoco {name}/{split_name}: {n_split} tasks")
            if skipped:
                print(f"  (running total of dropped rows: {skipped})")

    TASKS.mkdir(parents=True, exist_ok=True)
    (TASKS / "refcoco.json").write_text(json.dumps(tasks, indent=2) + "\n")
    return tasks
