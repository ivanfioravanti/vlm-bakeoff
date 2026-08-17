"""Local web UI server — stdlib only.

Serves a single-page launcher and a small JSON API that spawns
`python -m bench run` as a subprocess (identical behavior to the terminal,
including checkpointing/resume), tails its log, and browses past reports.
One active run at a time: the GPU is serial.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bench import RESULTS, TASKS
from bench.models import GGUF_ALIASES, MLX_ALIASES, spec

STATIC = Path(__file__).parent / "static"
_DIRNAME_RE = re.compile(r"^\d{8}-\d{6}$")
# Active-run record on disk: lets a restarted server re-adopt a run started
# by a previous instance (runs live in their own process group, so a server
# restart orphans the child but does not kill it).
_STATE_FILE = RESULTS / ".ui-active.json"

_state: dict = {"proc": None, "pid": None, "cmd": [], "run_dir": None, "started": None, "exit_code": None}
_lock = threading.Lock()


# ---------------------------------------------------------------- helpers

def _track_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(TASKS.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict):
            data = [data]
        if data:
            counts[data[0].get("category", path.stem)] = len(data)
    return counts


# What `prepare --<track> full` yields per track (mmmu full-prepare keeps only
# the scored MC items; refcoco full is all 8 eval splits, ~5 GB download).
_FULL_N = {
    "docvqa": 5_349,
    "infographicvqa": 2_801,
    "blink": 1_901,
    "screenspot": 1_272,
    "refcoco": 25_770,
    "mathvista": 1_000,
    "mmmu": 847,
}
_PREPARE_FLAG = {
    "docvqa": "--docvqa",
    "infographicvqa": "--infographicvqa",
    "blink": "--blink",
    "screenspot": "--screenspot",
    "refcoco": "--refcoco",
    "mathvista": "--mathvista",
    "mmmu": "--mmmu",
}


def _prepare_shortfalls(categories: list[str], limits: dict) -> dict[str, dict]:
    """Tracks whose prepared items are fewer than requested → prepare target."""
    counts = _track_counts()
    gcap = int(limits.get("global") or 0) or None
    per = limits.get("per_track") or {}
    shortfalls: dict[str, dict] = {}
    for cat in categories or list(counts):
        prepared = counts.get(cat, 0)
        requested = per.get(cat) or gcap or _FULL_N.get(cat, prepared)
        target = min(int(requested), _FULL_N.get(cat, int(requested)))
        if target > prepared:
            shortfalls[cat] = {"prepared": prepared, "target": target, "full": _FULL_N.get(cat)}
    return shortfalls


def _checkpoint_progress(run_dir: Path) -> list[dict]:
    out = []
    ckpt = run_dir / "checkpoints"
    if not ckpt.is_dir():
        return out
    for path in sorted(ckpt.glob("*.jsonl")):
        rows = done = 0
        cur = None
        gen_total = 0.0
        try:
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") != "row":
                    continue
                row = rec.get("row") or {}
                rows += 1
                done += 1 if row.get("pass") else 0
                gen_total += float(row.get("gen_s") or 0.0)
                cur = row.get("id")
        except OSError:
            continue
        out.append(
            {
                "model": path.stem,
                "rows": rows,
                "pass": done,
                "current": cur,
                "avg_gen_s": round(gen_total / rows, 3) if rows else None,
                "mtime": path.stat().st_mtime,
            }
        )
    return out


def _run_summary(run_dir: Path) -> dict | None:
    try:
        data = json.loads((run_dir / "results.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    from bench.report import summarize

    models = data.get("models") or []
    return {
        "models": models,
        "overall": {m: summarize(data["runs"][m])["overall"] for m in models},
        "by_category": {m: summarize(data["runs"][m])["by_category"] for m in models},
        "elapsed_s": {m: data["runs"][m].get("elapsed_s") for m in models},
    }


def _save_active() -> None:
    proc = _state.get("proc")
    if proc is None:
        return
    try:
        _STATE_FILE.write_text(
            json.dumps(
                {
                    "pid": proc.pid,
                    "run_dir": str(_state.get("run_dir")),
                    "cmd": _state.get("cmd"),
                    "started": _state.get("started"),
                }
            )
        )
    except OSError:
        pass


def _clear_active() -> None:
    try:
        _STATE_FILE.unlink()
    except OSError:
        pass


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _adopt_active() -> None:
    """Re-attach to a run started by a previous server instance."""
    if not _STATE_FILE.is_file():
        return
    try:
        rec = json.loads(_STATE_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        _clear_active()
        return
    run_dir = Path(str(rec.get("run_dir") or ""))
    if _pid_alive(rec.get("pid")) and not (run_dir / "results.json").is_file():
        _state.update(
            proc=None,
            pid=rec.get("pid"),
            cmd=rec.get("cmd") or [],
            run_dir=run_dir,
            started=rec.get("started"),
            exit_code=None,
        )
        print(f"bench UI: adopted active run pid {rec.get('pid')} → {run_dir}")
    else:
        _clear_active()


def _spawn(argv: list[str], run_dir: Path, cmd_label: list[str]) -> None:
    log = open(run_dir / "run.log", "ab")
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        argv,
        cwd=str(Path(__file__).resolve().parent.parent.parent),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    _state.update(
        proc=proc, pid=proc.pid, cmd=cmd_label, run_dir=run_dir, started=time.time(), exit_code=None
    )
    _save_active()


def _bench_argv(*args: str) -> list[str]:
    return [sys.executable, "-u", "-m", "bench", *args]


def _reap() -> None:
    proc = _state.get("proc")
    if proc is not None:
        if proc.poll() is not None:
            _state["exit_code"] = proc.returncode
            _state["proc"] = None
            _state["pid"] = None
            _clear_active()
        return
    # adopted run from a previous server instance: liveness + report presence
    if _state.get("pid") and not _pid_alive(_state["pid"]):
        _state["exit_code"] = 0
        _state["pid"] = None
        _clear_active()


# ---------------------------------------------------------------- handler

class Handler(BaseHTTPRequestHandler):
    server_version = "vlm-bakeoff-ui"

    def log_message(self, fmt, *args):  # quiet default access log
        pass

    # -- responses
    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, ctype: str, status: int = 200) -> None:
        try:
            body = path.read_bytes()
        except OSError:
            self._json({"error": "not found"}, 404)
            return
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- GET
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._file(STATIC / "index.html", "text/html; charset=utf-8")
        elif path == "/app.js":
            self._file(STATIC / "app.js", "text/javascript; charset=utf-8")
        elif path == "/style.css":
            self._file(STATIC / "style.css", "text/css; charset=utf-8")
        elif path == "/api/meta":
            models = [
                {"alias": a, "backend": "mlx", "model_id": spec(a).model_id}
                for a in MLX_ALIASES
            ] + [
                {"alias": a, "backend": "gguf", "model_id": spec(a).model_id}
                for a in GGUF_ALIASES
            ]
            try:
                from bench.coreai_infer import coreai_available

                ok, _reason = coreai_available()
                if ok:
                    models.append(
                        {"alias": "coreai", "backend": "coreai", "model_id": spec("coreai").model_id}
                    )
            except Exception:
                pass
            self._json(
                {
                    "models": models,
                    "tracks": _track_counts(),
                    "defaults": {
                        "temp": 0.2,
                        "top_k": 50,
                        "batch_size": 2,
                        "protocol": "grounding_json",
                        "protocols": [
                            "grounding_json",
                            "liquid",
                            "liquid_reason",
                            "bbox",
                            "pyautogui",
                        ],
                    },
                }
            )
        elif path == "/api/status":
            self._status()
        elif path == "/api/runs":
            self._runs()
        elif path.startswith("/api/report/"):
            name = path.rsplit("/", 1)[-1]
            if not _DIRNAME_RE.match(name):
                self._json({"error": "bad run dir"}, 400)
                return
            report = RESULTS / name / "REPORT.html"
            if not report.is_file():
                self._json({"error": "no report"}, 404)
                return
            self._file(report, "text/html; charset=utf-8")
        else:
            self._json({"error": "not found"}, 404)

    # -- POST
    def do_POST(self) -> None:
        if self.path == "/api/start":
            self._start()
        elif self.path == "/api/stop":
            self._stop()
        else:
            self._json({"error": "not found"}, 404)

    # -- endpoint bodies
    def _read_body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return {}

    def _start(self) -> None:
            with _lock:
                _reap()
                if _state.get("proc") is not None or _state.get("pid"):
                    self._json({"error": "a run is already active"}, 409)
                    return
            req = self._read_body()
            models = [m.strip() for m in req.get("models") or [] if m.strip()]
            if not models:
                self._json({"error": "select at least one model"}, 400)
                return
            for m in models:
                try:
                    spec(m)
                except Exception as exc:
                    self._json({"error": f"bad model {m!r}: {exc}"}, 400)
                    return
            categories = [c.strip() for c in req.get("categories") or [] if c.strip()]
            known = set(_track_counts())
            unknown = [c for c in categories if c not in known]
            if categories and unknown:
                self._json({"error": f"unknown tracks: {', '.join(unknown)}"}, 400)
                return
            if not categories and not known:
                self._json({"error": "no task files — run: python -m bench prepare"}, 400)
                return

            resume = str(req.get("resume") or "").strip()
            if resume:
                run_dir = Path(resume).expanduser()
                if not run_dir.is_dir():
                    self._json({"error": f"resume dir not found: {resume}"}, 400)
                    return
            else:
                run_dir = RESULTS / time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            run_dir.mkdir(parents=True, exist_ok=True)

            cmd = ["run", "--models", ",".join(models)]
            if categories:
                cmd += ["--categories", ",".join(categories)]
            limits = req.get("limits") or {}
            limit_parts = []
            if limits.get("global"):
                limit_parts.append(str(int(limits["global"])))
            for cat, n in (limits.get("per_track") or {}).items():
                if cat not in known:
                    self._json({"error": f"unknown track in limits: {cat}"}, 400)
                    return
                if n:
                    limit_parts.append(f"{cat}:{int(n)}")
            if limit_parts:
                cmd += ["--limits", ",".join(limit_parts)]
            for flag, key in (
                ("--batch-size", "batch_size"),
                ("--temp", "temp"),
                ("--top-k", "top_k"),
                ("--limit", "limit"),
            ):
                val = req.get(key)
                if val not in (None, ""):
                    cmd += [flag, str(val)]
            if req.get("protocol"):
                cmd += ["--protocol", str(req["protocol"])]
            if resume:
                cmd += ["--resume", str(run_dir)]
            else:
                cmd += ["--run-dir", str(run_dir)]

            # Auto-prepare: if more items are requested per track than are on
            # disk (unlimited = the full suite), regenerate those tracks first.
            # Two-phase: report the shortfall for confirmation, then chain
            # prepare (only the short tracks, everything else off) → run.
            shortfalls = _prepare_shortfalls(categories or list(known), limits)
            if shortfalls and not resume:
                if not req.get("confirm_prepare"):
                    self._json({"ok": False, "needs_prepare": shortfalls}, 200)
                    return
                prepare_cmd = ["prepare"]
                for cat in _PREPARE_FLAG:
                    if cat in shortfalls:
                        target = shortfalls[cat]["target"]
                        prepare_cmd += [
                            _PREPARE_FLAG[cat],
                            "full" if target >= (_FULL_N.get(cat) or 0) else str(target),
                        ]
                    else:
                        prepare_cmd += [_PREPARE_FLAG[cat], "off"]
                chain = " ".join(
                    [
                        " ".join(_bench_argv(*prepare_cmd)),
                        "&&",
                        " ".join(_bench_argv(*cmd)),
                    ]
                )
                _spawn(["/bin/sh", "-c", chain], run_dir, ["prepare", *prepare_cmd[1:], "→", *cmd])
            else:
                _spawn(_bench_argv(*cmd), run_dir, cmd)
            self._json({"ok": True, "run_dir": str(run_dir), "cmd": cmd})

    def _stop(self) -> None:
        with _lock:
            _reap()
            proc = _state.get("proc")
            pid = _state.get("pid")
            if proc is None and not pid:
                self._json({"ok": True, "note": "no active run"})
                return
            target = proc.pid if proc is not None else pid
            try:
                os.killpg(os.getpgid(target), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                if proc is not None:
                    proc.terminate()
            if proc is not None:
                for _ in range(50):  # up to 5s graceful
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)
                if proc.poll() is None:
                    try:
                        os.killpg(os.getpgid(target), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        proc.kill()
                _state["exit_code"] = proc.poll()
            else:
                for _ in range(50):
                    if not _pid_alive(pid):
                        break
                    time.sleep(0.1)
                _state["exit_code"] = 0
            _state["proc"] = None
            _state["pid"] = None
            _clear_active()
            self._json({"ok": True, "note": "stopped; resume any time (checkpoints kept)"})

    def _status(self) -> None:
        with _lock:
            _reap()
            active = _state.get("proc") is not None or bool(_state.get("pid"))
            payload: dict = {
                "active": active,
                "cmd": _state.get("cmd"),
                "run_dir": str(_state.get("run_dir")) if _state.get("run_dir") else None,
                "started": _state.get("started"),
                "exit_code": _state.get("exit_code"),
            }
        cmd = _state.get("cmd") or []
        counts = _track_counts()
        gcap: int | None = None
        caps: dict[str, int] = {}
        if "--limits" in cmd:
            for part in cmd[cmd.index("--limits") + 1].split(","):
                part = part.strip()
                if not part:
                    continue
                if ":" in part:
                    cat, _, n = part.rpartition(":")
                    caps[cat] = int(n)
                elif part.isdigit():
                    gcap = int(part)
        if "--limit" in cmd:
            payload["target_total"] = int(cmd[cmd.index("--limit") + 1])
        elif "--categories" in cmd:
            cats = cmd[cmd.index("--categories") + 1].split(",")
            payload["target_total"] = sum(min(counts.get(c, 0), caps.get(c, gcap or counts.get(c, 0))) for c in cats)
        elif counts:
            payload["target_total"] = sum(min(n, caps.get(c, gcap or n)) for c, n in counts.items())
        run_dir = _state.get("run_dir")
        if run_dir and Path(run_dir).is_dir():
            payload["models_progress"] = _checkpoint_progress(Path(run_dir))
            log = Path(run_dir) / "run.log"
            if log.is_file():
                try:
                    lines = log.read_text(errors="replace").splitlines()
                    payload["log_tail"] = lines[-120:]
                except OSError:
                    payload["log_tail"] = []
        self._json(payload)

    def _runs(self) -> None:
        runs = []
        if RESULTS.is_dir():
            for d in sorted(RESULTS.iterdir(), reverse=True):
                if not d.is_dir() or not _DIRNAME_RE.match(d.name):
                    continue
                entry: dict = {
                    "dir": d.name,
                    "created": d.stat().st_mtime,
                    "report": (d / "REPORT.html").is_file(),
                    "models": [p.stem for p in sorted((d / "checkpoints").glob("*.jsonl"))]
                    if (d / "checkpoints").is_dir()
                    else [],
                }
                entry["progress"] = _checkpoint_progress(d)
                summary = _run_summary(d)
                if summary:
                    entry["overall"] = summary["overall"]
                runs.append(entry)
        self._json({"runs": runs})


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    _adopt_active()
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"bench UI: {url}  (Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbench UI: bye")
