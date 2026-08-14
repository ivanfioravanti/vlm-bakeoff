from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from bench import IMAGES, TASKS
from bench.prompts import DOCVQA_PROMPT

SEED = 42
SUBSET_N = 500

# lmms-lab renamed their DocVQA mirror as well (lmms-lab/DocVQA →
# lmms-lab-encoder). Only the `validation` split carries answers (test is the
# blind split); the published number is ANLS on validation. The repo also hosts
# InfographicVQA behind the same schema — adding it later is one config entry.
HF_ID = "lmms-lab-encoder/DocVQA"
CONFIG = "DocVQA"
SPLIT = "validation"

# Liquid blog (vLLM 0.26): DocVQA (val) ANLS, normalized to 0-100 in their
# table as 91.1.
LIQUID_DOCVQA_ANLS = 0.911


def build_docvqa(split: str = "subset") -> list[dict]:
    if split not in ("full", "subset"):
        raise ValueError(f"docvqa split must be 'full' or 'subset', got {split!r}")
    from datasets import load_dataset

    ds = load_dataset(HF_ID, CONFIG, split=SPLIT)
    indices = list(range(len(ds)))
    if split == "subset":
        rng = random.Random(SEED)
        rng.shuffle(indices)
        indices = indices[:SUBSET_N]

    out_dir = IMAGES / "docvqa"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: set[str] = set()
    tasks: list[dict] = []
    for n, idx in enumerate(indices):
        row = ds[idx]
        answers = [str(a).strip() for a in (row.get("answers") or []) if str(a).strip()]
        if not answers:
            continue
        image = row["image"].convert("RGB")
        # Many questions share a page; name by document + page so each page
        # is stored once.
        doc = str(row.get("ucsf_document_id") or row.get("docId") or idx)
        page = str(row.get("ucsf_document_page_no") or 0)
        fname = f"{doc}_p{page}.png"
        if fname not in saved:
            image.save(out_dir / fname)
            saved.add(fname)
        qid = row.get("questionId", idx)
        qtypes = list(row.get("question_types") or [])
        question = str(row["question"]).strip()
        tasks.append(
            {
                "id": f"docvqa_val_{qid}",
                "category": "docvqa",
                "images": [f"docvqa/{fname}"],
                "prompt": DOCVQA_PROMPT.format(question=question),
                "scorer": "anls",
                "expected": {"answers": answers, "anls": 0.5},
                "max_tokens": 32,
                "meta": {
                    "question_type": qtypes[0] if qtypes else "other",
                    "question": question,
                    "answers": answers,
                    "doc_id": doc,
                    "page": page,
                    "split_mode": split,
                },
            }
        )
        if (n + 1) % 100 == 0 or n + 1 == len(indices):
            print(f"  docvqa tasks {n + 1}/{len(indices)}")

    TASKS.mkdir(parents=True, exist_ok=True)
    (TASKS / "docvqa.json").write_text(json.dumps(tasks, indent=2) + "\n")
    return tasks
