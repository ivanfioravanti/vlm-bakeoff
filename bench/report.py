from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench import RESULTS
from bench.blink import LIQUID_BLINK
from bench.docvqa import (
    LIQUID_DOCVQA_ANLS,
    LIQUID_INFOGRAPHICVQA_ANLS,
)
from bench.mathvista import LIQUID_MATHVISTA_MINI
from bench.mmmu import LIQUID_MMMU_VAL
from bench.refcoco import LIQUID_REFERCOCO_AVG, SPLITS
from bench.screenspot import LIQUID_SCREENSPOT_AVG, LIQUID_SCREENSPOT_V2

# ANLS tracks share the DocVQA module, scorer, and report shape.
ANLS_TRACKS = {
    "docvqa": ("DocVQA", LIQUID_DOCVQA_ANLS, 5_349),
    "infographicvqa": ("InfographicVQA", LIQUID_INFOGRAPHICVQA_ANLS, 2_801),
}

# Multiple-choice / short-answer exam tracks share the accuracy report shape.
EXAM_TRACKS = {
    "mathvista": ("MathVista", LIQUID_MATHVISTA_MINI),
    "mmmu": ("MMMU", LIQUID_MMMU_VAL),
}


def _acc(rows: list[dict]) -> float | None:
    if not rows:
        return None
    return sum(1 for r in rows if r["pass"]) / len(rows)


def _iou_acc(rows: list[dict]) -> float | None:
    if not rows:
        return None
    return sum(1 for r in rows if (r.get("iou") or 0.0) >= 0.5) / len(rows)


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def summarize(run: dict[str, Any]) -> dict[str, Any]:
    by_cat: dict[str, list] = defaultdict(list)
    by_plat: dict[str, list] = defaultdict(list)
    by_type: dict[str, list] = defaultdict(list)
    by_subset: dict[str, list] = defaultdict(list)
    by_dtype_q: dict[tuple[str, str], list] = defaultdict(list)
    by_blink_task: dict[str, list] = defaultdict(list)
    by_exam_type: dict[tuple[str, str], list] = defaultdict(list)
    by_exam_subject: dict[tuple[str, str], list] = defaultdict(list)
    for row in run["cases"]:
        by_cat[row["category"]].append(row)
        meta = row.get("meta") or {}
        plat = meta.get("platform")
        if plat:
            by_plat[plat].append(row)
        dtype = meta.get("data_type")
        if dtype:
            by_type[dtype].append(row)
        subset = meta.get("subset")
        if subset:
            by_subset[subset].append(row)
        qtype = meta.get("question_type")
        if row["category"] in ANLS_TRACKS and qtype:
            by_dtype_q[(row["category"], qtype)].append(row)
        if row["category"] in EXAM_TRACKS:
            if qtype:
                by_exam_type[(row["category"], qtype)].append(row)
            subject = meta.get("subject")
            if subject:
                by_exam_subject[(row["category"], subject)].append(row)
        btask = meta.get("task")
        if row["category"] == "blink" and btask:
            by_blink_task[btask].append(row)
    plats = {k: _acc(v) for k, v in sorted(by_plat.items())}
    macro = None
    if all(p in plats and plats[p] is not None for p in ("desktop", "mobile", "web")):
        macro = sum(plats[p] for p in ("desktop", "mobile", "web")) / 3
    subsets = {k: _acc(v) for k, v in sorted(by_subset.items())}
    subsets_iou = {k: _iou_acc(v) for k, v in sorted(by_subset.items())}
    refcoco_macro = sum(subsets.values()) / len(subsets) if subsets else None
    refcoco_iou_macro = sum(subsets_iou.values()) / len(subsets_iou) if subsets_iou else None
    docvqa_rows = by_cat.get("docvqa") or []
    anls_tracks: dict[str, dict] = {}
    for cat in ANLS_TRACKS:
        rows = by_cat.get(cat) or []
        if not rows:
            continue
        anls_tracks[cat] = {
            "anls": sum(float(r.get("metric") or 0.0) for r in rows) / len(rows),
            "acc": _acc(rows),
            "by_type": {
                q: sum(float(r.get("metric") or 0.0) for r in v) / len(v)
                for (c, q), v in sorted(by_dtype_q.items())
                if c == cat
            },
            "n": len(rows),
        }
    exam_tracks: dict[str, dict] = {}
    for cat in EXAM_TRACKS:
        rows = by_cat.get(cat) or []
        if not rows:
            continue
        exam_tracks[cat] = {
            "acc": _acc(rows),
            "by_type": {
                q: _acc(v) for (c, q), v in sorted(by_exam_type.items()) if c == cat
            },
            "by_subject": {
                s: _acc(v) for (c, s), v in sorted(by_exam_subject.items()) if c == cat
            },
            "n": len(rows),
        }
    return {
        "overall": _acc(run["cases"]),
        "n": len(run["cases"]),
        "n_pass": sum(1 for r in run["cases"] if r["pass"]),
        "elapsed_s": run.get("elapsed_s"),
        "by_category": {k: _acc(v) for k, v in sorted(by_cat.items())},
        "screenspot_by_platform": plats,
        "screenspot_by_type": {k: _acc(v) for k, v in sorted(by_type.items())},
        "screenspot_macro": macro,
        "screenspot_n": sum(len(v) for v in by_plat.values()),
        "refcoco_by_subset": subsets,
        "refcoco_iou_by_subset": subsets_iou,
        "refcoco_macro": refcoco_macro,
        "refcoco_iou_macro": refcoco_iou_macro,
        "refcoco_n": sum(len(v) for v in by_subset.values()),
        "anls_tracks": anls_tracks,
        "exam_tracks": exam_tracks,
        "blink_n": sum(len(v) for v in by_blink_task.values()),
        "blink_acc": _acc([r for r in by_cat.get("blink") or []]),
        "blink_by_task": {k: _acc(v) for k, v in sorted(by_blink_task.items())},
    }


def write_outputs(
    models: list[str],
    runs: dict[str, dict],
    extra: dict[str, Any],
    out_dir: Path | None = None,
) -> Path:
    if out_dir is None:
        out_dir = RESULTS / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created": out_dir.name,
        "models": models,
        "runs": runs,
        **extra,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    try:
        (out_dir / "REPORT.md").write_text(render_markdown(payload))
    except Exception as exc:
        print(f"markdown report failed: {exc}")
    try:
        from bench.html_report import render_html

        (out_dir / "REPORT.html").write_text(render_html(payload))
    except Exception as exc:
        print(f"html report failed: {exc}")
    return out_dir


def render_markdown(payload: dict[str, Any]) -> str:
    models = payload["models"]
    runs = payload["runs"]
    summaries = {m: summarize(runs[m]) for m in models}
    cats = sorted({c for s in summaries.values() for c in s["by_category"]})
    backends: list[str] = []
    for m in models:
        b = str(runs[m].get("backend") or "mlx")
        if b not in backends:
            backends.append(b)
    if "mlx" in backends:
        backends = ["mlx"] + [b for b in backends if b != "mlx"]
    runtime = " + ".join({"mlx": "MLX", "gguf": "GGUF", "coreai": "Core AI"}.get(b, b.upper()) for b in backends)
    has_screenspot = any(summaries[m].get("screenspot_n") for m in models)
    mlx_tiled = False
    if "mlx" in backends:
        try:
            ver = str(payload.get("mlx_vlm_version") or "0")
            mlx_tiled = tuple(int(p) for p in ver.split(".")[:3] if p.isdigit()) >= (0, 6, 14)
        except ValueError:
            mlx_tiled = False
    prep_notes: list[str] = []
    if has_screenspot:
        if "gguf" in backends:
            prep_notes.append(
                "llama.cpp (GGUF) tiles large screenshots into up to 10×512² tiles + overview, "
                "matching the official preprocessing behind Liquid's 80.7 — treat it as the accuracy reference."
            )
        if "mlx" in backends:
            if mlx_tiled:
                prep_notes.append(
                    f"the MLX path (mlx-vlm {payload.get('mlx_vlm_version')}) implements the same "
                    "image splitting — MLX and GGUF numbers are directly comparable."
                )
            else:
                prep_notes.append(
                    "the MLX path (mlx-vlm + the LiquidAI MLX repos) disables image splitting, so every "
                    "screenshot is downscaled to a single ≤512×512-equivalent view (~256 image tokens vs "
                    "up to ~2.8k tiled) — MLX ScreenSpot numbers are a lower bound, not a model-quality deficit."
                )
    protocol = str(payload.get("protocol") or "bbox")
    protocol_line = {
        "bbox": "Prompt protocol: `bbox` — model-native `[xmin, ymin, xmax, ymax]` integers in [0, 1000].",
        "pyautogui": (
            "Prompt protocol: `pyautogui` — Liquid's official ScreenSpot-v2 click wording "
            "(user message only; their harness drops the system prompt). Scorer parses `x=…, y=…`."
        ),
        "grounding_json": (
            "Prompt protocol: `grounding_json` — docs.liquid.ai grounding recipe: JSON array of "
            "`{image_id, bbox_2d, label}` items (0–1000 integer coordinates). Scorer takes the "
            "predicted box center."
        ),
        "liquid": (
            "Prompt protocol: `liquid` — the ScreenSpot-v2 prompt used in Liquid's official tests: "
            "locate the clickable element, return one tight JSON bbox_2d whose center is the click "
            "point (0–1000 integer coordinates). No system prompt."
        ),
        "liquid_reason": (
            "Prompt protocol: `liquid_reason` — as `liquid`, plus a silent-reasoning instruction "
            "(reason internally, output only the box)."
        ),
    }.get(protocol, f"Prompt protocol: `{protocol}`.")
    lines = [
        f"# LFM2.5-VL local {runtime} bench",
        "",
        f"- Created: `{payload['created']}`",
        *(
            [f"- mlx-vlm: `{payload.get('mlx_vlm_version', '?')}`"]
            if "mlx" in backends
            else []
        ),
        f"- Chip: `{payload.get('chip', '?')}`",
        *(
            [f"- llama.cpp: `{payload['llama_cpp']}`"]
            if "gguf" in backends and payload.get("llama_cpp")
            else []
        ),
        "",
        "ScreenSpot-v2 uses the official test items (full = 1,272). "
        "Liquid's 80.7 is the unweighted mean of desktop/mobile/web on vLLM, "
        "not this local run.",
        "",
        protocol_line,
        *([f"Image preprocessing: {n}" for n in prep_notes] if prep_notes else []),
        "",
    ]
    has_cases = any(summaries[m]["n"] for m in models)
    if has_cases:
        lines += [
            "## Overall",
            "",
            "| Model | Pass | Accuracy |",
            "| --- | --- | --- |",
        ]
        for m in models:
            s = summaries[m]
            lines.append(f"| `{m}` | {s['n_pass']}/{s['n']} | {_pct(s['overall'])} |")
        timed = [m for m in models if runs[m].get("elapsed_s") and summaries[m]["n"]]
        if timed:
            def _dur(x: float) -> str:
                s = int(round(x))
                h, rem = divmod(s, 3600)
                m_, sec = divmod(rem, 60)
                return f"{h}:{m_:02d}:{sec:02d}" if h else f"{m_}:{sec:02d}"

            lines += [
                "",
                "## Time & quality",
                "",
                "Wall-clock per model run (weights load + all cases), fastest first.",
                "",
                "| # | Model | Duration | Items/s | Overall | ScreenSpot avg |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for rank, m in enumerate(sorted(timed, key=lambda m: runs[m]["elapsed_s"]), 1):
                el = float(runs[m]["elapsed_s"])
                s = summaries[m]
                lines.append(
                    f"| {rank} | `{m}` | {_dur(el)} | {s['n'] / el:.2f} | "
                    f"{_pct(s['overall'])} | {_pct(s.get('screenspot_macro'))} |"
                )
        lines += ["", "## By category", "", "| Category | " + " | ".join(f"`{m}`" for m in models) + " |", "| --- | " + " | ".join("---" for _ in models) + " |"]
        for cat in cats:
            cells = [_pct(summaries[m]["by_category"].get(cat)) for m in models]
            lines.append(f"| {cat} | " + " | ".join(cells) + " |")
        plats = sorted({p for s in summaries.values() for p in s["screenspot_by_platform"]})
        if plats:
            ss_n = max(s.get("screenspot_n") or 0 for s in summaries.values())
            lines += [
                "",
                f"## ScreenSpot-v2 by platform ({ss_n} items)",
                "",
                (
                    "Click-in-box (predicted box center inside gold box)."
                    if protocol == "bbox"
                    else "Click-point-in-box (`x=…, y=…` inside gold box), as in Liquid's harness."
                )
                + " Liquid column is the published vLLM number, not this local run.",
                "",
                "| Platform | " + " | ".join(f"`{m}`" for m in models) + " | Liquid vLLM |",
                "| --- | " + " | ".join("---" for _ in models) + " | --- |",
            ]
            for p in plats:
                cells = [_pct(summaries[m]["screenspot_by_platform"].get(p)) for m in models]
                ref = _pct(LIQUID_SCREENSPOT_V2.get(p))
                lines.append(f"| {p} | " + " | ".join(cells) + f" | {ref} |")
            macros = [_pct(summaries[m].get("screenspot_macro")) for m in models]
            lines.append("| unweighted avg | " + " | ".join(macros) + f" | {_pct(LIQUID_SCREENSPOT_AVG)} |")
            types = sorted({t for s in summaries.values() for t in s.get("screenspot_by_type") or {}})
            if types:
                lines += [
                    "",
                    "## ScreenSpot-v2 by element",
                    "",
                    "| Type | " + " | ".join(f"`{m}`" for m in models) + " |",
                    "| --- | " + " | ".join("---" for _ in models) + " |",
                ]
                for t in types:
                    cells = [_pct(summaries[m]["screenspot_by_type"].get(t)) for m in models]
                    lines.append(f"| {t} | " + " | ".join(cells) + " |")
        subsets_present = sorted(
            {s for sm in summaries.values() for s in sm.get("refcoco_by_subset") or {}}
        )
        if subsets_present:
            order = [s for s in SPLITS if s in subsets_present]
            rc_n = max(s.get("refcoco_n") or 0 for s in summaries.values())
            scope = (
                "seeded subset, expect a few pp of sampling noise"
                if rc_n < 1000
                else "all 8 eval splits"
            )
            lines += [
                "",
                f"## RefCOCO grounding ({rc_n} items, {scope})",
                "",
                "Referring expressions on COCO photos — RefCOCO / RefCOCO+ / RefCOCOg, the 8 "
                "eval splits behind Liquid's published RefCOCO-avg. Two hit rules are reported "
                "because their precision@1 rule is unspecified; whichever avg lands nearer 87.9 "
                "is the one matching their scorer. Liquid column is the published vLLM number, "
                "not this local run.",
                "",
                "| Split | " + " | ".join(f"`{m}`" for m in models) + " | Liquid vLLM |",
                "| --- | " + " | ".join("---" for _ in models) + " | --- |",
            ]
            for s in order:
                cells = [_pct(summaries[m]["refcoco_by_subset"].get(s)) for m in models]
                lines.append(f"| {s} | " + " | ".join(cells) + " | |")
            cells = [_pct(summaries[m].get("refcoco_macro")) for m in models]
            lines.append("| avg — box center in gold | " + " | ".join(cells) + f" | {_pct(LIQUID_REFERCOCO_AVG)} |")
            cells = [_pct(summaries[m].get("refcoco_iou_macro")) for m in models]
            lines.append("| avg — IoU ≥ 0.5 | " + " | ".join(cells) + " | |")
        for cat, (title, ref, full_n) in ANLS_TRACKS.items():
            if not any(summaries[m]["anls_tracks"].get(cat) for m in models):
                continue
            dv_n = max(
                (s["anls_tracks"][cat]["n"] for s in summaries.values() if s["anls_tracks"].get(cat)),
                default=0,
            )
            scope = (
                "seeded subset, expect a couple pp of sampling noise"
                if dv_n < full_n * 0.9
                else "full validation split"
            )
            lines += [
                "",
                f"## {title} ({dv_n} items, {scope})",
                "",
                f"{'Document' if cat == 'docvqa' else 'Infographic'} reading comprehension — free-form short "
                "answers on the official validation split, scored by ANLS (Levenshtein, threshold "
                "0.5; errors count 0). Liquid column is their published vLLM ANLS, not this local run.",
                "",
                "| Metric | " + " | ".join(f"`{m}`" for m in models) + " | Liquid vLLM |",
                "| --- | " + " | ".join("---" for _ in models) + " | --- |",
            ]
            cells = [_pct(summaries[m]["anls_tracks"][cat].get("anls")) for m in models]
            lines.append("| ANLS | " + " | ".join(cells) + f" | {_pct(ref)} |")
            cells = [_pct(summaries[m]["anls_tracks"][cat].get("acc")) for m in models]
            lines.append("| ANLS pass rate | " + " | ".join(cells) + " | |")
            qtypes = sorted(
                {
                    q
                    for m in models
                    for q in (summaries[m]["anls_tracks"].get(cat) or {}).get("by_type", {})
                }
            )
            if qtypes:
                lines += [
                    "",
                    "| Answer/question type | " + " | ".join(f"`{m}`" for m in models) + " |",
                    "| --- | " + " | ".join("---" for _ in models) + " |",
                ]
                for q in qtypes:
                    cells = [
                        _pct((summaries[m]["anls_tracks"].get(cat) or {}).get("by_type", {}).get(q))
                        for m in models
                    ]
                    lines.append(f"| {q} | " + " | ".join(cells) + " |")
        if any(summaries[m].get("blink_n") for m in models):
            bl_n = max(s.get("blink_n") or 0 for s in summaries.values())
            scope = (
                "seeded subset (16 per task), noisy per-task"
                if bl_n < 1800
                else "full validation split (all 14 tasks)"
            )
            lines += [
                "",
                f"## BLINK ({bl_n} items, {scope})",
                "",
                "Multi-image perceptual tasks (relative depth, correspondence, jigsaw, ...) with "
                "the benchmark's canonical lettered-choice prompts. Liquid column is their "
                "published vLLM overall accuracy, not this local run.",
                "",
                "| Metric | " + " | ".join(f"`{m}`" for m in models) + " | Liquid vLLM |",
                "| --- | " + " | ".join("---" for _ in models) + " | --- |",
            ]
            cells = [_pct(summaries[m].get("blink_acc")) for m in models]
            lines.append("| Overall accuracy | " + " | ".join(cells) + f" | {_pct(LIQUID_BLINK)} |")
            btasks = sorted({t for s in summaries.values() for t in s.get("blink_by_task") or {}})
            if btasks:
                lines += [
                    "",
                    "| Task | " + " | ".join(f"`{m}`" for m in models) + " |",
                    "| --- | " + " | ".join("---" for _ in models) + " |",
                ]
                for t in btasks:
                    cells = [_pct(summaries[m]["blink_by_task"].get(t)) for m in models]
                    lines.append(f"| {t} | " + " | ".join(cells) + " |")
        for cat, (title, ref) in EXAM_TRACKS.items():
            if not any(summaries[m]["exam_tracks"].get(cat) for m in models):
                continue
            ev_n = max(
                (s["exam_tracks"][cat]["n"] for s in summaries.values() if s["exam_tracks"].get(cat)),
                default=0,
            )
            scope_note = (
                "Direct short answers (option letter or single number/word), scored by exact/"
                "tolerant match. Liquid's published number uses a CoT-style eval pipeline, so "
                "treat this local direct-answer accuracy as a lower bound vs their setup."
            )
            lines += [
                "",
                f"## {title} ({ev_n} items)",
                "",
                scope_note,
                "",
                "| Metric | " + " | ".join(f"`{m}`" for m in models) + " | Liquid vLLM |",
                "| --- | " + " | ".join("---" for _ in models) + " | --- |",
            ]
            cells = [_pct((summaries[m]["exam_tracks"].get(cat) or {}).get("acc")) for m in models]
            lines.append("| Accuracy | " + " | ".join(cells) + f" | {_pct(ref)} |")
            for label, key in (("question type", "by_type"), ("subject", "by_subject")):
                groups = sorted(
                    {
                        g
                        for m in models
                        for g in (summaries[m]["exam_tracks"].get(cat) or {}).get(key, {})
                    }
                )
                if not groups:
                    continue
                lines += [
                    "",
                    f"| {label.capitalize()} | " + " | ".join(f"`{m}`" for m in models) + " |",
                    "| --- | " + " | ".join("---" for _ in models) + " |",
                ]
                for g in groups:
                    cells = [
                        _pct((summaries[m]["exam_tracks"].get(cat) or {}).get(key, {}).get(g))
                        for m in models
                    ]
                    lines.append(f"| {g} | " + " | ".join(cells) + " |")
    if payload.get("speed"):
        lines += ["", "## Speed", "", "| Model | Setting | TTFT s | tok/s | Peak GB |", "| --- | --- | --- | --- | --- |"]
        for m, rows in payload["speed"].items():
            for row in rows:
                lines.append(
                    f"| `{m}` | {row['setting']} | {row['ttft_s']:.3f} | {row['tok_s']:.1f} | {row['peak_gb']:.2f} |"
                )
    if has_cases:
        lines += ["", "## Failures", ""]
        for m in models:
            fails = [c for c in runs[m]["cases"] if not c["pass"]]
            lines.append(f"### `{m}` ({len(fails)})")
            if not fails:
                lines.append("")
                lines.append("None.")
                lines.append("")
                continue
            lines.append("")
            for c in fails:
                preview = (c.get("output") or "").replace("\n", " ")[:160]
                expected = c.get("expected") or {}
                exp = (
                    " / ".join(str(a) for a in expected.get("answers") or [])
                    or str(expected.get("gold") or expected.get("answer") or expected.get("text") or "")
                )
                exp = f" · expected: {exp[:120]}" if exp else ""
                lines.append(f"- `{c['id']}` ({c['category']}): {preview}{exp}")
            lines.append("")
    return "\n".join(lines) + "\n"
