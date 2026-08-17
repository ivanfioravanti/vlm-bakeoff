from __future__ import annotations

import hashlib
import json
from pathlib import Path


def task_fingerprint(tasks: list[dict]) -> str:
    """Stable digest of the exact task list a run executes.

    Covers id, prompt and max_tokens so a protocol change, a split change or
    a task-file regeneration invalidates old checkpoints instead of silently
    resuming mismatched work.
    """
    h = hashlib.sha256()
    for t in tasks:
        h.update(str(t.get("id")).encode())
        h.update(b"\0")
        h.update(str(t.get("prompt")).encode())
        h.update(b"\0")
        h.update(str(t.get("max_tokens")).encode())
        h.update(b"\0")
    return h.hexdigest()


class CheckpointStore:
    """Append-only JSONL of scored rows per model, one file under run_dir/checkpoints."""

    def __init__(self, run_dir: Path, fingerprint: str):
        self.dir = run_dir / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint

    def path_for(self, model: str) -> Path:
        return self.dir / f"{model}.jsonl"

    def load(self, model: str) -> dict[str, dict]:
        """Completed rows keyed by task id; validates the header fingerprint."""
        path = self.path_for(model)
        rows: dict[str, dict] = {}
        if not path.exists():
            return rows
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # a crash mid-write can tear the last line; keep what came before
                print(f"warning: skipping unparsable line {lineno} in {path.name}")
                continue
            if rec.get("type") == "header":
                if rec.get("fingerprint") != self.fingerprint:
                    raise SystemExit(
                        f"checkpoint {path.name} was written for a different task set "
                        f"({str(rec.get('fingerprint'))[:8]}… != {self.fingerprint[:8]}…): "
                        "task files or flags changed since that run"
                    )
                continue
            row = rec.get("row")
            if row and row.get("id"):
                rows[row["id"]] = row
        return rows

    def load_elapsed(self, model: str) -> float:
        """Sum of per-session wall times recorded in done footers (0 if none)."""
        path = self.path_for(model)
        if not path.exists():
            return 0.0
        total = 0.0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "done":
                total += float(rec.get("elapsed_s") or 0.0)
        return total

    def writer(self, model: str):
        """Open the model's checkpoint for appending, creating the header if new."""
        path = self.path_for(model)
        fresh = not path.exists()
        fh = path.open("a", encoding="utf-8")
        if fresh:
            fh.write(json.dumps({"type": "header", "fingerprint": self.fingerprint}) + "\n")
        return fh
