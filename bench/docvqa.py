from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path
from urllib.parse import urlparse

from bench import IMAGES, TASKS
from bench.prompts import DOCVQA_PROMPT

SEED = 42

# lmms-lab renamed their DocVQA mirror (lmms-lab/DocVQA → lmms-lab-encoder);
# it hosts two ANLS-scored tracks behind the same schema: DocVQA and
# InfographicVQA. Only the `validation` split carries answers (test is the
# blind split); the published numbers are ANLS on validation.
HF_ID = "lmms-lab-encoder/DocVQA"
SPLIT = "validation"

# Liquid blog (vLLM 0.26): validation ANLS, normalized to 0-100 in their
# table as 91.1 (DocVQA) and 70.2 (InfographicVQA).
LIQUID_DOCVQA_ANLS = 0.911
LIQUID_INFOGRAPHICVQA_ANLS = 0.702

TRACKS: dict[str, dict] = {
    "docvqa": {
        "config": "DocVQA",
        "category": "docvqa",
        "subset_n": 500,
        "full_n": 5_349,
    },
    "infographicvqa": {
        "config": "InfographicVQA",
        "category": "infographicvqa",
        "subset_n": 500,
        "full_n": 2_801,
    },
}


def _page_name(row: dict, idx: int) -> str:
    # DocVQA rows name their UCSF page; InfographicVQA rows carry the source
    # image URL — hash it so shared infographics are stored once.
    doc = row.get("ucsf_document_id")
    if doc:
        page = str(row.get("ucsf_document_page_no") or 0)
        return f"{doc}_p{page}.png"
    url = str(row.get("image_url") or "")
    if url:
        stem = Path(urlparse(url).path).name or "img"
        digest = hashlib.md5(url.encode()).hexdigest()[:8]
        base = stem.rsplit(".", 1)[0][:48] or "img"
        suffix = stem.rsplit(".", 1)[-1].lower()
        ext = "png" if suffix not in ("png", "jpg", "jpeg") else ("jpg" if suffix in ("jpg", "jpeg") else "png")
        return f"{digest}_{base}.{ext}"
    return f"{row.get('questionId', idx)}.png"


def build_anls_track(name: str, split: str = "subset") -> list[dict]:
    if split not in ("full", "subset"):
        raise ValueError(f"{name} split must be 'full' or 'subset', got {split!r}")
    track = TRACKS[name]
    from datasets import load_dataset

    ds = load_dataset(HF_ID, track["config"], split=SPLIT)
    indices = list(range(len(ds)))
    if split == "subset":
        rng = random.Random(SEED)
        rng.shuffle(indices)
        indices = indices[: track["subset_n"]]

    out_dir = IMAGES / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: set[str] = set()
    tasks: list[dict] = []
    category = track["category"]
    for n, idx in enumerate(indices):
        row = ds[idx]
        answers = [str(a).strip() for a in (row.get("answers") or []) if str(a).strip()]
        if not answers:
            continue
        fname = _page_name(row, idx)
        if fname not in saved:
            row["image"].convert("RGB").save(out_dir / fname)
            saved.add(fname)
        qid = row.get("questionId", idx)
        types = list(row.get("question_types") or row.get("answer_type") or [])
        question = str(row["question"]).strip()
        tasks.append(
            {
                "id": f"{category}_val_{qid}",
                "category": category,
                "images": [f"{name}/{fname}"],
                "prompt": DOCVQA_PROMPT.format(question=question),
                "scorer": "anls",
                "expected": {"answers": answers, "anls": 0.5},
                "max_tokens": 32,
                "meta": {
                    "question_type": types[0] if types else "other",
                    "question": question,
                    "answers": answers,
                    "split_mode": split,
                },
            }
        )
        if (n + 1) % 100 == 0 or n + 1 == len(indices):
            print(f"  {name} tasks {n + 1}/{len(indices)}")

    TASKS.mkdir(parents=True, exist_ok=True)
    (TASKS / f"{name}.json").write_text(json.dumps(tasks, indent=2) + "\n")
    return tasks


def build_docvqa(split: str = "subset") -> list[dict]:
    return build_anls_track("docvqa", split)


def build_infographicvqa(split: str = "subset") -> list[dict]:
    return build_anls_track("infographicvqa", split)
