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

_state: dict = {"proc": None, "cmd": [], "run_dir": None, "started": None, "exit_code": None}
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


def _spawn(cmd: list[str], run_dir: Path) -> None:
    log = open(run_dir / "run.log", "ab")
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "bench", *cmd],
        cwd=str(Path(__file__).resolve().parent.parent.parent),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    _state.update(proc=proc, cmd=cmd, run_dir=run_dir, started=time.time(), exit_code=None)


def _reap() -> None:
    proc = _state.get("proc")
    if proc is not None and proc.poll() is not None:
        _state["exit_code"] = proc.returncode
        _state["proc"] = None


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
            self._json(
                {
                    "models": [
                        {"alias": a, "backend": "mlx", "model_id": spec(a).model_id}
                        for a in MLX_ALIASES
                    ]
                    + [
                        {"alias": a, "backend": "gguf", "model_id": spec(a).model_id}
                        for a in GGUF_ALIASES
                    ],
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
            if _state.get("proc") is not None:
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
            _spawn(cmd, run_dir)
            self._json({"ok": True, "run_dir": str(run_dir), "cmd": cmd})

    def _stop(self) -> None:
        with _lock:
            _reap()
            proc = _state.get("proc")
            if proc is None:
                self._json({"ok": True, "note": "no active run"})
                return
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.terminate()
            for _ in range(50):  # up to 5s graceful
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
            _state["exit_code"] = proc.poll()
            _state["proc"] = None
            self._json({"ok": True, "note": "stopped; resume any time (checkpoints kept)"})

    def _status(self) -> None:
        with _lock:
            _reap()
            proc = _state.get("proc")
            payload: dict = {
                "active": proc is not None,
                "cmd": _state.get("cmd"),
                "run_dir": str(_state.get("run_dir")) if _state.get("run_dir") else None,
                "started": _state.get("started"),
                "exit_code": _state.get("exit_code"),
            }
        cmd = _state.get("cmd") or []
        counts = _track_counts()
        if "--limit" in cmd:
            payload["target_total"] = int(cmd[cmd.index("--limit") + 1])
        elif "--categories" in cmd:
            cats = cmd[cmd.index("--categories") + 1].split(",")
            payload["target_total"] = sum(counts.get(c, 0) for c in cats)
        elif counts:
            payload["target_total"] = sum(counts.values())
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
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"bench UI: {url}  (Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbench UI: bye")
