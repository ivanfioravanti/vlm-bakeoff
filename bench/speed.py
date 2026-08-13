from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
from PIL import Image

from bench import IMAGES
from bench.infer import ModelSession


def _ttft_and_tps(session: ModelSession, task: dict, warmup: bool = False) -> dict:
    if getattr(session, "backend", "mlx") == "gguf":
        return session.timed_generate(task, warmup=warmup)

    reset = getattr(mx, "reset_peak_memory", None)
    if reset:
        reset()
    if warmup:
        session.generate(task)
        if reset:
            reset()

    from mlx_vlm.generate import stream_generate
    from bench.tasks import image_paths

    paths = [str(p) for p in image_paths(task)]
    messages = []
    if paths:
        content = [{"type": "image"} for _ in paths] + [{"type": "text", "text": task["prompt"]}]
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": task["prompt"]})
    prompt = session._apply(
        session.processor,
        session.model.config,
        messages,
        add_generation_prompt=True,
        num_images=len(paths),
    )
    t0 = time.perf_counter()
    ttft = None
    n = 0
    last = None
    for chunk in stream_generate(
        session.model,
        session.processor,
        prompt,
        paths or None,
        max_tokens=int(task.get("max_tokens", 64)),
        temperature=0.0,
    ):
        if ttft is None:
            ttft = time.perf_counter() - t0
        n += 1
        last = chunk
    elapsed = time.perf_counter() - t0
    gen = getattr(last, "generation_tokens", n) if last else n
    return {
        "ttft_s": ttft or elapsed,
        "tok_s": gen / elapsed if elapsed else 0.0,
        "peak_gb": mx.get_peak_memory() / 1e9,
        "generation_tokens": gen,
    }


def run_speed(session: ModelSession) -> list[dict]:
    IMAGES.mkdir(parents=True, exist_ok=True)
    one = IMAGES / "_speed_one.png"
    two = IMAGES / "_speed_two.png"
    if not one.exists():
        Image.new("RGB", (512, 512), "#8899aa").save(one)
    if not two.exists():
        Image.new("RGB", (512, 512), "#aa9988").save(two)

    text_task = {
        "prompt": "Say hello in one sentence.",
        "images": [],
        "max_tokens": 64,
    }
    one_task = {
        "prompt": "Describe this image in one sentence.",
        "images": [str(one.relative_to(IMAGES))],
        "max_tokens": 64,
    }
    two_task = {
        "prompt": "What is different between these images? One sentence.",
        "images": [str(one.relative_to(IMAGES)), str(two.relative_to(IMAGES))],
        "max_tokens": 64,
    }

    _ttft_and_tps(session, text_task, warmup=True)
    rows = []
    for name, task in (
        ("text", text_task),
        ("image_512", one_task),
        ("two_images", two_task),
    ):
        stats = _ttft_and_tps(session, task)
        stats["setting"] = name
        rows.append(stats)
    return rows
