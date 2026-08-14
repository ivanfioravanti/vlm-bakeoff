from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

from bench.docvqa import build_docvqa
from bench.infer import ModelSession
from bench.models import DEFAULT_MODELS, parse_models, spec
from bench.prompts import (
    ANSWER_DIRECTLY,
    GROUNDING_JSON_SYSTEM,
    GROUNDING_JSON_USER,
    PYAUTOGUI_USER,
    liquid_user,
)
from bench.refcoco import build_refcoco
from bench.report import write_outputs
from bench.score import score
from bench.screenspot import build_screenspot
from bench.speed import run_speed
from bench.tasks import load_tasks
from bench import RESULTS


def _chip() -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor() or platform.machine()


def cmd_prepare(args: argparse.Namespace) -> int:
    tracks = [
        ("screenspot", args.screenspot, build_screenspot),
        ("refcoco", args.refcoco, build_refcoco),
        ("docvqa", args.docvqa, build_docvqa),
    ]
    if all(mode == "off" for _, mode, _ in tracks):
        raise SystemExit("nothing to prepare: every track flag is off")
    for name, mode, build in tracks:
        if mode != "off":
            tasks = build(mode)
            print(f"{name}: {len(tasks)} tasks ({mode})")
    return 0


def _apply_protocol(tasks: list[dict], protocol: str) -> list[dict]:
    """Re-ground ScreenSpot tasks under a different prompt protocol.

    ScreenSpot only — RefCOCO tasks already carry the grounding_json recipe
    baked in at prepare time and are passed through untouched.

    'bbox' keeps the tasks as prepared. 'pyautogui' swaps in Liquid's official
    ScreenSpot-v2 wording (user message only — their harness drops the system
    prompt for single images) and the click-point scorer. 'grounding_json'
    uses the docs.liquid.ai grounding recipe: system prompt specifying the
    JSON bbox_2d format plus "Provide bounding boxes for ..." — the model's
    native grounding interface.
    """
    if protocol == "bbox":
        return tasks
    out: list[dict] = []
    for task in tasks:
        if task.get("category") != "screenspot" or task.get("scorer") != "click_in_box":
            out.append(task)
            continue
        meta = task.get("meta") or {}
        instruction = meta.get("instruction")
        if not instruction:
            instruction = (
                task["prompt"].split("instruction:", 1)[-1].split("\n")[0].strip()
            )
        t = dict(task)
        if protocol == "pyautogui":
            t["prompt"] = PYAUTOGUI_USER.format(instruction=instruction) + ANSWER_DIRECTLY
            t["scorer"] = "click_point"
            t["max_tokens"] = 1024
            t.pop("system", None)
        elif protocol == "grounding_json":
            t["prompt"] = GROUNDING_JSON_USER.format(instruction=instruction)
            t["system"] = GROUNDING_JSON_SYSTEM
            t["max_tokens"] = 256
        elif protocol in ("liquid", "liquid_reason"):
            t["prompt"] = liquid_user(instruction, reason=protocol == "liquid_reason")
            t["max_tokens"] = 256
            t.pop("system", None)
        else:
            raise ValueError(f"unknown protocol: {protocol!r}")
        out.append(t)
    return out


def cmd_run(args: argparse.Namespace) -> int:
    names = parse_models(args.models)
    cats = [c.strip() for c in args.categories.split(",") if c.strip()] if args.categories else None
    tasks = _apply_protocol(load_tasks(cats), args.protocol)
    runs: dict = {}
    used_gguf = False
    for name in names:
        sp = spec(name)
        used_gguf = used_gguf or sp.backend == "gguf"
        print(f"\n== {name} ({sp.backend}: {sp.model_id}) ==")
        t0 = time.perf_counter()
        session = ModelSession(
            name,
            temperature=args.temp,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        cases = []
        try:
            for i, task in enumerate(tasks, 1):
                try:
                    result = session.generate(task)
                    text = result.text if hasattr(result, "text") else str(result)
                    scored = score(task, text)
                except Exception as exc:
                    text = ""
                    scored = {"pass": False, "metric": None, "error": f"{type(exc).__name__}: {exc}"}
                    traceback.print_exc()
                row = {
                    "id": task["id"],
                    "category": task["category"],
                    "pass": scored["pass"],
                    "metric": scored.get("metric"),
                    "iou": scored.get("iou"),
                    "pred_bbox": scored.get("pred_bbox"),
                    "output": text,
                    "expected": task.get("expected"),
                    "meta": task.get("meta"),
                    "error": scored.get("error"),
                }
                mark = "PASS" if row["pass"] else "FAIL"
                print(f"  [{i}/{len(tasks)}] {mark} {task['id']}")
                cases.append(row)
        finally:
            session.close()
        elapsed_s = time.perf_counter() - t0
        n_pass = sum(1 for c in cases if c["pass"])
        print(f"  {n_pass}/{len(cases)} passed in {elapsed_s/60:.1f} min")
        runs[name] = {
            "model_id": sp.model_id,
            "backend": sp.backend,
            "elapsed_s": round(elapsed_s, 1),
            "cases": cases,
        }

    extra = {
        "chip": _chip(),
        "mlx_vlm_version": importlib.metadata.version("mlx-vlm"),
        "python": sys.version.split()[0],
        "protocol": args.protocol,
        "sampling": {
            "temperature": args.temp,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
        },
    }
    if used_gguf:
        from bench.gguf_infer import llama_version

        extra["llama_cpp"] = llama_version()
    out = write_outputs(names, runs, extra)
    print(f"\nWrote {out / 'REPORT.html'}")
    print(f"      {out / 'REPORT.md'}")
    return 0


def cmd_speed(args: argparse.Namespace) -> int:
    names = parse_models(args.models)
    speed = {}
    used_gguf = False
    for name in names:
        sp = spec(name)
        used_gguf = used_gguf or sp.backend == "gguf"
        print(f"\n== speed {name} ({sp.backend}: {sp.model_id}) ==")
        session = ModelSession(name, temperature=0.0)
        try:
            rows = run_speed(session)
        finally:
            session.close()
        speed[name] = rows
        for row in rows:
            print(
                f"  {row['setting']}: ttft={row['ttft_s']:.3f}s  "
                f"{row['tok_s']:.1f} tok/s  peak={row['peak_gb']:.2f} GB"
            )
    extra = {
        "chip": _chip(),
        "mlx_vlm_version": importlib.metadata.version("mlx-vlm"),
        "python": sys.version.split()[0],
        "speed": speed,
    }
    if used_gguf:
        from bench.gguf_infer import llama_version

        extra["llama_cpp"] = llama_version()
    out = write_outputs(names, {n: {"model_id": spec(n).model_id, "backend": spec(n).backend, "cases": []} for n in names}, extra)
    print(f"\nWrote {out / 'REPORT.html'}")
    print(f"      {out / 'REPORT.md'}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from bench.html_report import render_html
    from bench.report import render_markdown

    src = args.path
    if not src:
        dirs = sorted(
            (p for p in RESULTS.glob("*") if p.is_dir() and (p / "results.json").exists()),
            reverse=True,
        )
        if not dirs:
            raise SystemExit("No results yet. Run: python -m bench run")
        src = str(dirs[0] / "results.json")
    path = Path(src)
    payload = json.loads(path.read_text())
    out_dir = path.parent
    (out_dir / "REPORT.html").write_text(render_html(payload))
    (out_dir / "REPORT.md").write_text(render_markdown(payload))
    print(f"Wrote {out_dir / 'REPORT.html'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Local MLX / GGUF VLM capability benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare", help="Generate ScreenSpot-v2, RefCOCO and DocVQA tasks and images")
    sp.add_argument(
        "--screenspot",
        choices=("full", "subset", "off"),
        default="full",
        help="full = 1,272-item official test set; subset = 48 seeded items (16 per platform); off = skip",
    )
    sp.add_argument(
        "--refcoco",
        choices=("full", "subset", "off"),
        default="subset",
        help="grounding on COCO photos behind Liquid's published RefCOCO-avg 87.9: "
        "full = all 8 eval splits (25,770 items, ~5 GB download, hours per model); "
        "subset = 64 seeded items per split (512 total); off = skip",
    )
    sp.add_argument(
        "--docvqa",
        choices=("full", "subset", "off"),
        default="subset",
        help="document reading comprehension behind Liquid's published DocVQA 91.1 (val, ANLS): "
        "full = 5,349 questions (~1 GB download); subset = 500 seeded items; off = skip",
    )
    sp.set_defaults(func=cmd_prepare)

    sr = sub.add_parser("run", help="Run capability tasks")
    sr.add_argument("--models", default=",".join(DEFAULT_MODELS))
    sr.add_argument("--categories", default="")
    sr.add_argument(
        "--protocol",
        choices=("grounding_json", "liquid", "liquid_reason", "bbox", "pyautogui"),
        default="grounding_json",
        help="grounding_json = docs.liquid.ai grounding recipe (JSON bbox_2d, "
        "default — 80.8%% vs 64.3%% macro locally); bbox = plain [0,1000] box; "
        "pyautogui = Liquid's ScreenSpot-v2 harness wording",
    )
    sr.add_argument(
        "--temp",
        type=float,
        default=0.2,
        help="Liquid's recommended generation params (temperature 0.2, top_k 50) "
        "are the default; sampling is on whenever temp > 0. Pass --temp 0 for greedy.",
    )
    sr.add_argument("--top-k", type=int, default=50)
    sr.add_argument("--repetition-penalty", type=float, default=None)
    sr.set_defaults(func=cmd_run)

    ss = sub.add_parser("speed", help="TTFT / tok/s / peak memory")
    ss.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ss.set_defaults(func=cmd_speed)

    so = sub.add_parser("report", help="Rebuild HTML/Markdown from results.json")
    so.add_argument("path", nargs="?", default="", help="Path to results.json (default: latest)")
    so.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
