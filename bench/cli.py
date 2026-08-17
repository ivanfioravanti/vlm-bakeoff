from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from bench.blink import build_blink
from bench.checkpoint import CheckpointStore, task_fingerprint
from bench.docvqa import build_docvqa, build_infographicvqa
from bench.mathvista import build_mathvista
from bench.mmmu import build_mmmu
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


def _split_arg(value: str) -> str | int:
    v = value.strip().lower()
    if v in ("off", "subset", "full"):
        return v
    try:
        n = int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected off, subset, full or N, got {value!r}")
    if n <= 0:
        raise argparse.ArgumentTypeError("N must be a positive item count")
    return n


def cmd_prepare(args: argparse.Namespace) -> int:
    tracks = [
        ("screenspot", args.screenspot, build_screenspot),
        ("refcoco", args.refcoco, build_refcoco),
        ("docvqa", args.docvqa, build_docvqa),
        ("infographicvqa", args.infographicvqa, build_infographicvqa),
        ("blink", args.blink, build_blink),
        ("mathvista", args.mathvista, build_mathvista),
        ("mmmu", args.mmmu, build_mmmu),
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


def _fmt_eta(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    s = int(seconds)
    if s < 3600:
        return f"{s // 60}m"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def cmd_run(args: argparse.Namespace) -> int:
    names = parse_models(args.models)
    cats = [c.strip() for c in args.categories.split(",") if c.strip()] if args.categories else None
    tasks = _apply_protocol(load_tasks(cats), args.protocol)
    if args.limit:
        tasks = tasks[: args.limit]
    fingerprint = task_fingerprint(tasks)
    if args.resume:
        run_dir = Path(args.resume).expanduser()
        if not run_dir.is_dir():
            raise SystemExit(f"--resume: not a results run dir: {run_dir}")
    elif args.run_dir:
        run_dir = Path(args.run_dir).expanduser()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = RESULTS / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    store = CheckpointStore(run_dir, fingerprint)

    runs: dict = {}
    used_gguf = False
    for name in names:
        sp = spec(name)
        used_gguf = used_gguf or sp.backend == "gguf"
        resumed = store.load(name)
        batch_note = f", batch {args.batch_size}" if args.batch_size > 1 else ""
        resume_note = f" [resuming {len(resumed)}/{len(tasks)}]" if resumed else ""
        print(f"\n== {name} ({sp.backend}: {sp.model_id}{batch_note}) =={resume_note}")
        t0 = time.perf_counter()
        session = None

        def get_session():
            # Created only if something actually needs generating, so a fully
            # resumed model skips the weight load / llama-server startup.
            nonlocal session
            if session is None:
                session = ModelSession(
                    name,
                    temperature=args.temp,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    batch_size=args.batch_size,
                )
            return session

        out_fh = store.writer(name)
        cases: dict[str, dict] = dict(resumed)
        state = {"done": len(resumed), "new": 0, "batch_fails": 0, "batch_off": False}

        def _record(task: dict, text: str, scored: dict, gen_s: float) -> None:
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
                "gen_s": round(gen_s, 3),
            }
            out_fh.write(json.dumps({"type": "row", "row": row}) + "\n")
            out_fh.flush()
            cases[task["id"]] = row
            state["done"] += 1
            state["new"] += 1
            elapsed = time.perf_counter() - t0
            rate = state["new"] / elapsed * 60 if elapsed > 0 else 0.0
            eta = (len(tasks) - state["done"]) / (state["new"] / elapsed) if state["new"] else None
            mark = "PASS" if row["pass"] else "FAIL"
            print(
                f"  [{state['done']}/{len(tasks)}] {mark} {task['id']} "
                f"({rate:.1f} it/min, eta {_fmt_eta(eta)})"
            )

        def _run_one(task: dict) -> None:
            t_gen = time.perf_counter()
            try:
                result = session.generate(task)
                text = result.text if hasattr(result, "text") else str(result)
                scored = score(task, text)
            except Exception as exc:
                text = ""
                scored = {"pass": False, "metric": None, "error": f"{type(exc).__name__}: {exc}"}
                traceback.print_exc()
            _record(task, text, scored, time.perf_counter() - t_gen)

        try:
            i = 0
            while i < len(tasks):
                task = tasks[i]
                if task["id"] in cases:
                    i += 1
                    continue
                get_session()
                if args.batch_size > 1 and not state["batch_off"] and session.batchable(task):
                    chunk = [task]
                    i += 1
                    while i < len(tasks) and len(chunk) < args.batch_size:
                        nxt = tasks[i]
                        if nxt["id"] in cases or not session.batchable(nxt):
                            break
                        chunk.append(nxt)
                        i += 1
                    t_gen = time.perf_counter()
                    try:
                        results = session.generate_batch(chunk)
                    except Exception as exc:
                        state["batch_fails"] += 1
                        traceback.print_exc()
                        print(
                            f"  batch of {len(chunk)} failed ({type(exc).__name__}); "
                            "retrying sequentially"
                        )
                        if state["batch_fails"] >= 5:
                            state["batch_off"] = True
                            print("  disabling batching for this model after repeated batch failures")
                        results = None
                    if results is not None:
                        # gen_s is each item's amortized share of the batch wall time
                        gen_s = (time.perf_counter() - t_gen) / len(chunk)
                        for item, res in zip(chunk, results):
                            text = res.text if hasattr(res, "text") else str(res)
                            try:
                                scored = score(item, text)
                            except Exception as exc:
                                text = ""
                                scored = {"pass": False, "metric": None, "error": f"{type(exc).__name__}: {exc}"}
                                traceback.print_exc()
                            _record(item, text, scored, gen_s)
                        continue
                    for item in chunk:  # sequential fallback after a failed batch
                        if item["id"] not in cases:
                            _run_one(item)
                else:
                    i += 1
                    _run_one(task)
        finally:
            elapsed_s = time.perf_counter() - t0
            if state["new"]:
                out_fh.write(json.dumps({"type": "done", "elapsed_s": round(elapsed_s, 1)}) + "\n")
                out_fh.flush()
            if session is not None:
                session.close()
            out_fh.close()
        total_s = store.load_elapsed(name)
        ordered = [cases[t["id"]] for t in tasks if t["id"] in cases]
        n_pass = sum(1 for c in ordered if c["pass"])
        print(
            f"  {n_pass}/{len(ordered)} passed | session {elapsed_s/60:.1f} min | "
            f"total {total_s/60:.1f} min (+{len(resumed)} resumed)"
        )
        runs[name] = {
            "model_id": sp.model_id,
            "backend": sp.backend,
            "elapsed_s": round(total_s, 1),
            "batch_size": args.batch_size,
            "resumed": len(resumed),
            "cases": ordered,
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
        "batch_size": args.batch_size,
    }
    if used_gguf:
        from bench.gguf_infer import llama_version

        extra["llama_cpp"] = llama_version()
    out = write_outputs(names, runs, extra, out_dir=run_dir)
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


def cmd_ui(args: argparse.Namespace) -> int:
    from bench.webui.server import serve

    serve(host=args.host, port=args.port, open_browser=not args.no_open)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Local MLX / GGUF VLM capability benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare", help="Generate benchmark tasks and images (all tracks)")
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
        type=_split_arg,
        default="subset",
        help="document reading comprehension behind Liquid's published DocVQA 91.1 (val, ANLS): "
        "full = 5,349 questions (~1 GB download); subset = 500 seeded items; N = custom "
        "seeded subset of N items (same shuffle, so 500 stays a prefix); off = skip",
    )
    sp.add_argument(
        "--infographicvqa",
        type=_split_arg,
        default="subset",
        help="infographic reading comprehension behind Liquid's published 70.2 (val, ANLS; same "
        "repo/scorer as DocVQA): full = 2,801 questions; subset = 500 seeded items; "
        "N = custom seeded subset; off = skip",
    )
    sp.add_argument(
        "--blink",
        choices=("full", "subset", "off"),
        default="subset",
        help="multi-image perceptual tasks behind Liquid's published BLINK 61.5 (val, overall "
        "accuracy; 1-4 images per item): full = all 14 tasks (1,901 items); subset = 16 seeded "
        "items per task (224); off = skip",
    )
    sp.add_argument(
        "--mathvista",
        type=_split_arg,
        default="full",
        help="visual math reasoning behind Liquid's published MathVista 68.5 (testmini, "
        "1,000 items: multiple-choice + short free-form answers): full = testmini; "
        "subset = 300 seeded; N = custom seeded subset; off = skip. Note: Liquid's "
        "number uses CoT-style eval; this suite scores direct answers, so expect "
        "a lower bound.",
    )
    sp.add_argument(
        "--mmmu",
        type=_split_arg,
        default="full",
        help="college-level multi-discipline MC behind Liquid's published MMMU 48.4 "
        "(val, ~900 items over 30 subjects, 1-3 images each incl. some text-only "
        "questions): full = val; subset = 300 seeded; N = custom; off = skip",
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
    sr.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="items in flight per model (concurrency): MLX uses mlx-vlm batch_generate "
        "(single-image tasks; BLINK stays sequential), GGUF uses parallel llama-server "
        "slots. Measured on M5 Max: MLX is vision-bound and flat across batch sizes; "
        "GGUF is ~10% faster at 2-4 slots and ~1.7x slower at 8.",
    )
    sr.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only run the first N tasks after category/protocol filters (smoke tests)",
    )
    sr.add_argument(
        "--resume",
        default="",
        help="results run dir holding checkpoints/ to resume from (results are appended "
        "incrementally, so a killed run can always be resumed)",
    )
    sr.add_argument(
        "--run-dir",
        default="",
        help="explicit output dir for this run (created if missing) instead of the "
        "auto-generated results/<UTC timestamp>",
    )
    sr.set_defaults(func=cmd_run)

    ui = sub.add_parser("ui", help="Local web UI for launching and browsing runs")
    ui.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument(
        "--no-open",
        action="store_true",
        help="don't open the browser automatically",
    )
    ui.set_defaults(func=cmd_ui)

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
