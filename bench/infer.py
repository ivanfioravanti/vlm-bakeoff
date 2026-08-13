from __future__ import annotations

import gc
from typing import Any

import mlx.core as mx

from bench.models import ModelSpec, spec
from bench.tasks import image_paths


class MlxSession:
    backend = "mlx"

    def __init__(
        self,
        model_id: str,
        temperature: float = 0.0,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
    ):
        from mlx_vlm import apply_chat_template, generate, load

        self.model_id = model_id
        self.temperature = temperature
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.model, self.processor = load(model_id)
        self._apply = apply_chat_template
        self._generate = generate

    def generate(self, task: dict[str, Any]):
        paths = [str(p) for p in image_paths(task)]
        messages: list[dict] = []
        if task.get("system"):
            messages.append({"role": "system", "content": task["system"]})
        if paths:
            content: list[dict] = [{"type": "image"} for _ in paths]
            content.append({"type": "text", "text": task["prompt"]})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": task["prompt"]})
        prompt = self._apply(
            self.processor,
            self.model.config,
            messages,
            add_generation_prompt=True,
            num_images=len(paths),
        )
        kwargs: dict[str, Any] = {
            "max_tokens": int(task.get("max_tokens", 128)),
            "temperature": self.temperature,
            "verbose": False,
        }
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.repetition_penalty is not None:
            kwargs["repetition_penalty"] = self.repetition_penalty
        return self._generate(
            self.model,
            self.processor,
            prompt,
            paths or None,
            **kwargs,
        )

    def close(self) -> None:
        self.model = None
        self.processor = None
        gc.collect()
        mx.clear_cache()


def ModelSession(
    name: str,
    temperature: float = 0.0,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
):
    sp = name if isinstance(name, ModelSpec) else spec(name)
    if sp.backend == "gguf":
        from bench.gguf_infer import GgufSession

        return GgufSession(
            sp,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
    return MlxSession(
        sp.model_id,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
