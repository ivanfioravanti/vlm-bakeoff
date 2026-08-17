from __future__ import annotations

import re
from typing import Any

# Numbers tolerate decode artifacts where a decimal point is followed by a
# space ("0.940" arriving as "0. 940"); measured on the 20260812-234554 run.
_NUM = r"-?\d+(?:\.\s*\d+)?"
_BOX = re.compile(rf"\[\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*\]")
_POINT = re.compile(rf"\[\s*({_NUM})\s*,\s*({_NUM})\s*\]")
_XY = re.compile(rf"x\s*=\s*({_NUM})\s*,\s*y\s*=\s*({_NUM})", re.I)
_CLICK_XY = re.compile(r"x=([\d.]+), y=([\d.]+)")


def parse_bbox(text: str) -> list[int] | None:
    m = _BOX.search(text or "")
    if not m:
        return None
    try:
        vals = [float(g.replace(" ", "")) for g in m.groups()]
    except ValueError:
        return None
    # Liquid's native grounding format is normalized to [0, 1]; ours is [0, 1000].
    if all(0.0 <= v <= 1.01 for v in vals):
        vals = [v * 1000 for v in vals]
    x1, y1, x2, y2 = (int(round(v)) for v in vals)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _scale_coord(v: float) -> float:
    return v * 1000 if 0 <= v <= 1.01 else v


def parse_click(text: str) -> tuple[float, float] | None:
    box = parse_bbox(text)
    if box:
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    m = _XY.search(text or "")
    if m:
        x, y = (float(g.replace(" ", "")) for g in m.groups())
        return (_scale_coord(x), _scale_coord(y))
    m = _POINT.search(text or "")
    if m:
        x, y = (float(g.replace(" ", "")) for g in m.groups())
        return (_scale_coord(x), _scale_coord(y))
    return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _contains(needle: str, haystack: str) -> bool:
    """Substring match, but only on token boundaries.

    Plain `in` makes "red" match "colored" and "no" match "know", which turns
    wrong answers into passes on the short closed-form VQA cases.
    """
    n, h = _norm(needle), _norm(haystack)
    if not n:
        return False
    left = r"(?<![0-9a-z])" if n[0].isalnum() else ""
    right = r"(?![0-9a-z])" if n[-1].isalnum() else ""
    return re.search(f"{left}{re.escape(n)}{right}", h) is not None


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return len(a) or len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _choice_letter(text: str) -> str | None:
    """Extract an A–D choice from BLINK-style answers: "(B)", "B", "B is closer"."""
    t = _norm(text)
    m = re.search(r"\(([a-d])\)", t)
    if m:
        return m.group(1)
    m = re.match(r"^\(?\s*([a-d])(?:\s|\)|$)", t)
    if m:
        return m.group(1)
    m = re.search(r"\b([a-d])\s*\)", t)
    if m:
        return m.group(1)
    letters = re.findall(r"\b([a-d])\b", t)
    return letters[-1] if letters else None


def _option_letter(text: str, n_choices: int) -> str | None:
    """Like _choice_letter but for an arbitrary option count (MathVista A–E)."""
    t = _norm(text)
    last = chr(ord("a") + max(1, n_choices) - 1)
    m = re.search(rf"\(([a-{last}])\)", t)
    if m:
        return m.group(1)
    m = re.match(rf"^\(?\s*([a-{last}])(?:\s|\)|$)", t)
    if m:
        return m.group(1)
    letters = re.findall(rf"\b([a-{last}])\b", t)
    return letters[-1] if letters else None


def _parse_number(s: str) -> float | None:
    t = re.sub(r"\s+", "", (s or "")).replace(",", "").replace("$", "")
    t = t.rstrip("%").rstrip(".")
    if not t or t in ("-", ".", "-."):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _first_number(text: str) -> float | None:
    m = re.search(_NUM.replace(r"\.", r"\.?\s*"), text or "")
    if not m:
        return None
    return _parse_number(m.group(0))


def score(task: dict[str, Any], text: str) -> dict[str, Any]:
    kind = task["scorer"]
    expected = task.get("expected") or {}
    result: dict[str, Any] = {"pass": False, "metric": None}

    if kind == "contains":
        ok = _contains(expected["text"], text)
        result["pass"] = ok
        result["metric"] = float(ok)
    elif kind == "exact":
        ok = _norm(text) == _norm(expected["text"])
        result["pass"] = ok
        result["metric"] = float(ok)
    elif kind == "bbox_iou":
        pred = parse_bbox(text)
        gold = expected["bbox"]
        thresh = float(expected.get("iou", 0.5))
        val = iou(pred, gold) if pred else 0.0
        result["metric"] = val
        result["pass"] = pred is not None and val >= thresh
        result["pred_bbox"] = pred
    elif kind == "click_in_box":
        pred = parse_bbox(text)
        click = parse_click(text)
        gold = expected["bbox"]
        ok = click is not None and gold[0] <= click[0] <= gold[2] and gold[1] <= click[1] <= gold[3]
        result["pass"] = ok
        result["metric"] = float(ok)
        result["pred_bbox"] = pred
        # Secondary criterion: IoU@0.5 against the gold box. Liquid publishes
        # RefCOCO as precision@1 without spelling out the hit rule, so both
        # this and the center rule are recorded per case and compared in the
        # report until one is pinned as their P@1.
        result["iou"] = iou(pred, gold) if pred else 0.0
        result["pred_click"] = list(click) if click else None
    elif kind == "click_point":
        # Mirrors Liquid's VLMEvalKit ScreenSpot scorer (parse_bbox_aguvis +
        # evaluate_point): parse `x=…, y=…`; coordinates > 1 are treated as
        # pixels and normalized by the image size; a missing match falls back
        # to (0, 0) and can only pass if the gold box contains the origin.
        from bench.tasks import image_paths

        gold = expected["bbox"]
        x = y = 0.0
        parsed = False
        m = _CLICK_XY.search(text or "")
        if m:
            parsed = True
            x, y = float(m.group(1)), float(m.group(2))
            if x > 1 or y > 1:
                paths = image_paths(task)
                if paths:
                    from PIL import Image

                    with Image.open(paths[0]) as im:
                        x, y = x / im.size[0], y / im.size[1]
        click = (x * 1000, y * 1000)
        ok = gold[0] <= click[0] <= gold[2] and gold[1] <= click[1] <= gold[3]
        result["pass"] = ok
        result["metric"] = float(ok)
        result["pred_click"] = list(click)
        result["parsed"] = parsed
    elif kind == "anls":
        # Official DocVQA metric: 1 - NLD (Levenshtein / max string length),
        # best over the reference answers; values below the threshold count
        # as 0. The published benchmark number is the mean of per-case values.
        answers = expected["answers"]
        thresh = float(expected.get("anls", 0.5))
        text_n = _norm(text)
        val = 0.0
        for ans in answers:
            ans_n = _norm(str(ans))
            if not ans_n:
                continue
            nld = _levenshtein(text_n, ans_n) / max(len(text_n), len(ans_n))
            val = max(val, 1.0 - nld)
        if val < thresh:
            val = 0.0
        result["metric"] = val
        result["pass"] = val >= thresh
    elif kind == "mathvista":
        # MathVista testmini: multiple-choice items score by option letter;
        # free-form items compare numbers with tolerance (answers like "3.14",
        # "50%", "$1,200") and fall back to normalized text match.
        if expected.get("kind") == "mc":
            pred = _option_letter(text or "", int(expected.get("n") or 5))
            ok = pred is not None and pred == str(expected["gold"]).lower()
            if not ok and expected.get("text"):
                # The official hint asks for the letter, but a model may emit
                # the option text itself ("145°" instead of "(B) 145°").
                ok = _contains(str(expected["text"]), text or "")
            result["pass"] = ok
            result["metric"] = float(ok)
            result["pred_letter"] = pred
        else:
            gold = str(expected.get("answer") or "")
            gold_num = _parse_number(gold)
            if gold_num is not None:
                pred_num = _first_number(text or "")
                ok = pred_num is not None and abs(gold_num - pred_num) <= max(
                    1e-6, abs(gold_num) * 1e-4
                )
            else:
                ok = _norm(text) == _norm(gold) or _contains(gold, text)
            result["pass"] = ok
            result["metric"] = float(ok)
    elif kind == "mc_option":
        # Lettered multiple choice with an arbitrary option count (MMMU A–I):
        # gold is the option letter, n the number of options shown.
        gold = str(expected.get("gold") or "").lower()
        pred = _option_letter(text or "", int(expected.get("n") or 4))
        ok = pred is not None and pred == gold
        result["pass"] = ok
        result["metric"] = float(ok)
        result["pred_letter"] = pred
    elif kind == "choice_letter":
        # BLINK-style multiple choice: gold is "(B)"; the model may emit the
        # parenthesized letter, a bare letter, or letter + choice text.
        gold = _choice_letter(str(expected["text"]))
        pred = _choice_letter(text or "")
        ok = gold is not None and pred == gold
        result["pass"] = ok
        result["metric"] = float(ok)
        result["pred_letter"] = pred
    elif kind == "choice":
        gold = _norm(str(expected["text"]))
        ok = _contains(gold, text) or _norm(text).startswith(gold)
        result["pass"] = ok
        result["metric"] = float(ok)
    else:
        raise ValueError(f"unknown scorer: {kind}")
    return result
