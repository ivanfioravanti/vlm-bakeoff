from __future__ import annotations

import json
from pathlib import Path

from bench import IMAGES, TASKS


def load_tasks(categories: list[str] | None = None) -> list[dict]:
    if not TASKS.exists():
        raise FileNotFoundError("No tasks yet. Run: python -m bench prepare")
    tasks: list[dict] = []
    for path in sorted(TASKS.glob("*.json")):
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            data = [data]
        tasks.extend(data)
    if categories:
        allow = set(categories)
        tasks = [t for t in tasks if t.get("category") in allow]
    if not tasks:
        raise SystemExit("No tasks matched. Did you run prepare?")
    return tasks


def image_paths(task: dict) -> list[Path]:
    return [IMAGES / p for p in task.get("images") or []]
