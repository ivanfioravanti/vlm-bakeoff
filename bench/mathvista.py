from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from bench import IMAGES, TASKS

SEED = 42
SUBSET_N = 300

# AI4Math/MathVista testmini: 1,000 items over 10 subjects / 5 difficulty
# levels, mixing multiple-choice and short free-form (integer/decimal/float/
# text) answers. Liquid blog (vLLM): 68.5 on testmini — their pipeline uses
# CoT-style eval, so the direct-answer protocol here is a lower bound.
HF_ID = "AI4Math/MathVista"
SPLIT = "testmini"

LIQUID_MATHVISTA_MINI = 0.685


def _fname(uid: str) -> str:
    digest = hashlib.md5(uid.encode()).hexdigest()[:8]
    return f"{digest}.jpg"


def build_mathvista(split: str | int = "full") -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(HF_ID, split=SPLIT)
    indices = list(range(len(ds)))
    mode, subset_n = "full", None
    if split == "subset":
        mode, subset_n = "subset", SUBSET_N
    elif isinstance(split, int) and not isinstance(split, bool) and split > 0:
        mode, subset_n = "subset", min(split, len(ds))
    elif split != "full":
        raise ValueError(f"mathvista split must be 'full', 'subset' or a positive int, got {split!r}")
    if subset_n is not None:
        rng = random.Random(SEED)
        rng.shuffle(indices)
        indices = indices[:subset_n]

    out_dir = IMAGES / "mathvista"
    out_dir.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()

    tasks: list[dict] = []
    for n, idx in enumerate(indices):
        row = ds[idx]
        uid = str(row.get("pid") or idx)
        image = row.get("decoded_image")
        question = str(row.get("question") or "").strip()
        if image is None or not question:
            continue
        fname = _fname(uid)
        if fname not in used:
            target = out_dir / fname
            if not target.exists():
                image.convert("RGB").save(target)
            used.add(fname)

        qtype = str(row.get("question_type") or "free_form").strip()  # multi_choice | free_form
        meta_info = row.get("metadata") or {}
        meta: dict = {
            "question_type": qtype,
            "subject": str(meta_info.get("task") or "other"),
            "level": str(meta_info.get("grade") or ""),
            "split_mode": mode,
        }
        if subset_n is not None:
            meta["subset_n"] = subset_n

        if qtype == "multi_choice":
            choices = [str(c) for c in (row.get("choices") or []) if c is not None and str(c).strip()]
            answer = str(row.get("answer") or "").strip()
            if len(choices) < 2 or answer not in choices:
                continue
            gold = chr(ord("A") + choices.index(answer))
            expected: dict = {"kind": "mc", "gold": gold, "n": len(choices), "text": answer}
            meta["choices"] = len(choices)
        else:
            answer = row.get("answer")
            if answer is None or not str(answer).strip():
                continue
            expected = {
                "kind": "value",
                "answer": str(answer).strip(),
                "answer_type": str(row.get("answer_type") or "text"),
            }
        # The dataset ships its official prompt ("Hint: …\nQuestion: …" with
        # "(A)"-style choices embedded) — use it verbatim so the protocol
        # matches the published MathVista numbers (direct-answer mode).
        prompt = str(row.get("query") or "").strip()
        if not prompt:
            lines = "\n".join(
                f"({chr(ord('A') + i)}) {c}" for i, c in enumerate(row.get("choices") or [])
            )
            prompt = question + (f"\n{lines}" if lines else "")
            prompt += "\nAnswer the question using a single word, number, or phrase."

        tasks.append(
            {
                "id": f"mathvista_mini_{uid}",
                "category": "mathvista",
                "images": [f"mathvista/{fname}"],
                "prompt": prompt,
                "scorer": "mathvista",
                "expected": expected,
                "max_tokens": 32,
                "meta": meta,
            }
        )
        if (n + 1) % 100 == 0 or (n + 1) == len(indices):
            print(f"  mathvista tasks {n + 1}/{len(indices)}")

    for stale in out_dir.iterdir():
        if stale.is_file() and stale.name not in used:
            stale.unlink()

    TASKS.mkdir(parents=True, exist_ok=True)
    (TASKS / "mathvista.json").write_text(json.dumps(tasks, indent=2) + "\n")
    return tasks
