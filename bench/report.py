from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench import RESULTS
from bench.docvqa import LIQUID_DOCVQA_ANLS
from bench.refcoco import LIQUID_REFERCOCO_AVG, SPLITS
from bench.screenspot import LIQUID_SCREENSPOT_AVG, LIQUID_SCREENSPOT_V2


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
    by_dtype_q: dict[str, list] = defaultdict(list)
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
        if row["category"] == "docvqa" and qtype:
            by_dtype_q[qtype].append(row)
    plats = {k: _acc(v) for k, v in sorted(by_plat.items())}
    macro = None
    if all(p in plats and plats[p] is not None for p in ("desktop", "mobile", "web")):
        macro = sum(plats[p] for p in ("desktop", "mobile", "web")) / 3
    subsets = {k: _acc(v) for k, v in sorted(by_subset.items())}
    subsets_iou = {k: _iou_acc(v) for k, v in sorted(by_subset.items())}
    refcoco_macro = sum(subsets.values()) / len(subsets) if subsets else None
    refcoco_iou_macro = sum(subsets_iou.values()) / len(subsets_iou) if subsets_iou else None
    docvqa_rows = by_cat.get("docvqa") or []
    docvqa_anls = (
        sum(float(r.get("metric") or 0.0) for r in docvqa_rows) / len(docvqa_rows)
        if docvqa_rows
        else None
    )
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
        "docvqa_anls": docvqa_anls,
        "docvqa_acc": _acc(docvqa_rows),
        "docvqa_by_type": {
            k: sum(float(r.get("metric") or 0.0) for r in v) / len(v)
            for k, v in sorted(by_dtype_q.items())
        },
        "docvqa_n": len(docvqa_rows),
    }


def write_outputs(models: list[str], runs: dict[str, dict], extra: dict[str, Any]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = RESULTS / stamp
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "created": stamp,
        "models": models,
        "runs": runs,
        **extra,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    try:
        (out / "REPORT.md").write_text(render_markdown(payload))
    except Exception as exc:
        print(f"markdown report failed: {exc}")
    try:
        from bench.html_report import render_html

        (out / "REPORT.html").write_text(render_html(payload))
    except Exception as exc:
        print(f"html report failed: {exc}")
    return out


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
    runtime = " + ".join({"mlx": "MLX", "gguf": "GGUF"}.get(b, b.upper()) for b in backends)
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
        if any(summaries[m].get("docvqa_n") for m in models):
            dv_n = max(s.get("docvqa_n") or 0 for s in summaries.values())
            scope = (
                "seeded subset, expect a couple pp of sampling noise"
                if dv_n < 5000
                else "full validation split"
            )
            lines += [
                "",
                f"## DocVQA ({dv_n} items, {scope})",
                "",
                "Document reading comprehension — free-form short answers on the official "
                "validation split, scored by ANLS (Levenshtein, threshold 0.5; errors count 0). "
                "Liquid column is their published vLLM ANLS, not this local run.",
                "",
                "| Metric | " + " | ".join(f"`{m}`" for m in models) + " | Liquid vLLM |",
                "| --- | " + " | ".join("---" for _ in models) + " | --- |",
            ]
            cells = [_pct(summaries[m].get("docvqa_anls")) for m in models]
            lines.append("| ANLS | " + " | ".join(cells) + f" | {_pct(LIQUID_DOCVQA_ANLS)} |")
            cells = [_pct(summaries[m].get("docvqa_acc")) for m in models]
            lines.append("| ANLS pass rate | " + " | ".join(cells) + " | |")
            qtypes = sorted(
                {q for s in summaries.values() for q in s.get("docvqa_by_type") or {}}
            )
            if qtypes:
                lines += [
                    "",
                    "| Question type | " + " | ".join(f"`{m}`" for m in models) + " |",
                    "| --- | " + " | ".join("---" for _ in models) + " |",
                ]
                for q in qtypes:
                    cells = [_pct(summaries[m]["docvqa_by_type"].get(q)) for m in models]
                    lines.append(f"| {q} | " + " | ".join(cells) + " |")
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
                lines.append(f"- `{c['id']}` ({c['category']}): {preview}")
            lines.append("")
    return "\n".join(lines) + "\n"
