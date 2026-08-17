from __future__ import annotations

import base64
import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench.models import GGUF_REPO, ModelSpec
from bench.tasks import image_paths


@dataclass
class Generation:
    text: str


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _llama_server() -> str:
    import shutil

    path = shutil.which("llama-server")
    if not path:
        raise SystemExit(
            "llama-server not found on PATH. Install llama.cpp to run GGUF models, "
            "e.g. `brew install llama.cpp`."
        )
    return path


def llama_version() -> str | None:
    import shutil

    path = shutil.which("llama-server")
    if not path:
        return None
    try:
        out = subprocess.check_output([path, "--version"], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.strip().splitlines()[0] if out.strip() else None


def _download_gguf(spec: ModelSpec) -> tuple[str, str | None]:
    if spec.model_id.endswith(".gguf"):
        return spec.model_id, spec.mmproj_file
    if not spec.gguf_file:
        raise SystemExit(f"No GGUF filename mapped for {spec.model_id}")
    from huggingface_hub import hf_hub_download

    print(f"  downloading {spec.gguf_file} from {GGUF_REPO}")
    model = hf_hub_download(GGUF_REPO, spec.gguf_file)
    mmproj = None
    if spec.mmproj_file:
        print(f"  downloading {spec.mmproj_file} from {GGUF_REPO}")
        mmproj = hf_hub_download(GGUF_REPO, spec.mmproj_file)
    return model, mmproj


class GgufSession:
    backend = "gguf"

    def __init__(
        self,
        spec: ModelSpec,
        temperature: float = 0.0,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        batch_size: int = 1,
    ):
        self.spec = spec
        self.model_id = spec.model_id
        self.temperature = temperature
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.batch_size = max(1, batch_size)
        self.port = _free_port()
        model_path, mmproj_path = _download_gguf(spec)
        # -c is the total KV budget shared by the -np slots, so each parallel
        # request still gets the single-slot 8192 context.
        cmd = [
            _llama_server(),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "-m",
            model_path,
            "-ngl",
            "99",
            "-c",
            str(8192 * self.batch_size),
            "--no-ui",
            "--jinja",
        ]
        if self.batch_size > 1:
            cmd += ["-np", str(self.batch_size)]
        if mmproj_path:
            cmd += ["--mmproj", mmproj_path]
        print(f"  starting llama-server on 127.0.0.1:{self.port} (-np {self.batch_size})")
        self.proc = subprocess.Popen(cmd)
        try:
            self._wait_ready()
        except Exception:
            self.close()
            raise

    def _wait_ready(self, timeout: float = 300.0) -> None:
        url = f"http://127.0.0.1:{self.port}/health"
        deadline = time.time() + timeout
        last_print = 0.0
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited {self.proc.returncode}")
            try:
                with urllib.request.urlopen(url, timeout=1.5) as resp:
                    if resp.status == 200:
                        print(f"  llama-server ready on port {self.port}")
                        return
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                pass
            now = time.time()
            if now - last_print > 5:
                print(f"  waiting for llama-server on port {self.port}…")
                last_print = now
            time.sleep(0.4)
        raise TimeoutError(f"llama-server did not become ready on port {self.port}")

    def generate(self, task: dict[str, Any]) -> Generation:
        text, _, _ = self._complete(task, stream=False)
        return Generation(text=text)

    def batchable(self, task: dict[str, Any]) -> bool:
        return True

    def generate_batch(self, tasks: list[dict[str, Any]]) -> list[Generation]:
        # llama-server does the continuous batching server-side across its
        # -np slots; here we just keep that many requests in flight.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.batch_size) as pool:
            return list(pool.map(self.generate, tasks))

    def timed_generate(self, task: dict[str, Any], warmup: bool = False) -> dict[str, float]:
        if warmup:
            self.generate(task)
        t0 = time.perf_counter()
        _, ttft, n_tok = self._complete(task, stream=True)
        elapsed = time.perf_counter() - t0
        return {
            "ttft_s": ttft if ttft is not None else elapsed,
            "tok_s": (n_tok / elapsed) if elapsed else 0.0,
            "peak_gb": self._rss_gb(),
            "generation_tokens": n_tok,
        }

    def _rss_gb(self) -> float:
        try:
            out = subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(self.proc.pid)], text=True
            )
            return int(out.strip().split()[0]) / (1024 * 1024)
        except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
            return 0.0

    def _complete(self, task: dict[str, Any], stream: bool) -> tuple[str, float | None, int]:
        paths = [str(p) for p in image_paths(task)]
        user: Any
        if paths:
            parts: list[dict[str, Any]] = []
            for path in paths:
                raw = Path(path).read_bytes()
                b64 = base64.b64encode(raw).decode("ascii")
                mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    }
                )
            parts.append({"type": "text", "text": task["prompt"]})
            user = parts
        else:
            user = task["prompt"]
        messages: list[dict[str, Any]] = []
        if task.get("system"):
            messages.append({"role": "system", "content": task["system"]})
        messages.append({"role": "user", "content": user})
        body: dict[str, Any] = {
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": int(task.get("max_tokens", 128)),
            "stream": stream,
        }
        if self.top_k is not None:
            body["top_k"] = self.top_k
        if self.repetition_penalty is not None:
            body["repeat_penalty"] = self.repetition_penalty
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if stream:
            return self._read_stream(req)
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        text = payload["choices"][0]["message"]["content"] or ""
        usage = payload.get("usage") or {}
        n_tok = int(usage.get("completion_tokens") or 0)
        return text, None, n_tok

    def _read_stream(self, req: urllib.request.Request) -> tuple[str, float | None, int]:
        t0 = time.perf_counter()
        ttft: float | None = None
        chunks: list[str] = []
        n = 0
        with urllib.request.urlopen(req, timeout=180) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = ((obj.get("choices") or [{}])[0].get("delta") or {}).get("content")
                if not delta:
                    continue
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunks.append(delta)
                n += 1
        return "".join(chunks), ttft, n

    def close(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
