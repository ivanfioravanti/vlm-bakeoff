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
        batch_size: int = 1,
    ):
        from mlx_vlm import apply_chat_template, batch_generate, generate, load
        from mlx_vlm.sample_utils import make_logits_processors, make_sampler

        self.model_id = model_id
        self.temperature = temperature
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.batch_size = batch_size
        self.model, self.processor = load(model_id)
        self._apply = apply_chat_template
        self._generate = generate
        self._batch_generate = batch_generate
        self._make_sampler = make_sampler
        self._make_logits_processors = make_logits_processors

    def _messages(self, task: dict[str, Any]) -> list[dict]:
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
        return messages

    def _sampling(self, max_tokens: int | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"temperature": self.temperature, "verbose": False}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.repetition_penalty is not None:
            kwargs["repetition_penalty"] = self.repetition_penalty
        return kwargs

    def batchable(self, task: dict[str, Any]) -> bool:
        # mlx-vlm's batch_generate takes exactly one image per prompt;
        # multi-image (BLINK) tasks stay on the sequential path.
        return len(task.get("images") or []) == 1

    def generate(self, task: dict[str, Any]):
        prompt = self._apply(
            self.processor,
            self.model.config,
            self._messages(task),
            add_generation_prompt=True,
            num_images=len(image_paths(task)),
        )
        return self._generate(
            self.model,
            self.processor,
            prompt,
            [str(p) for p in image_paths(task)] or None,
            **self._sampling(int(task.get("max_tokens", 128))),
        )

    def generate_batch(self, tasks: list[dict[str, Any]]) -> list[str]:
        paths = [[str(p) for p in image_paths(t)] for t in tasks]
        if any(len(p) != 1 for p in paths):
            raise ValueError("generate_batch accepts single-image tasks only")
        # batch_generate routes through BatchGenerator, which takes a sampler
        # callable instead of the temperature/top_k kwargs of generate().
        kwargs: dict[str, Any] = {
            "sampler": self._make_sampler(temp=self.temperature, top_k=self.top_k or 0),
        }
        if self.repetition_penalty is not None:
            kwargs["logits_processors"] = self._make_logits_processors(
                repetition_penalty=self.repetition_penalty
            )
        # _generate_batch applies the chat template itself, so pass the raw
        # messages and let it inject the image token per prompt.
        response = self._batch_generate(
            self.model,
            self.processor,
            images=[p[0] for p in paths],
            prompts=[self._messages(task) for task in tasks],
            max_tokens=[int(t.get("max_tokens", 128)) for t in tasks],
            **kwargs,
        )
        return list(response.texts)

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
    batch_size: int = 1,
):
    sp = name if isinstance(name, ModelSpec) else spec(name)
    if sp.backend == "gguf":
        from bench.gguf_infer import GgufSession

        return GgufSession(
            sp,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
        )
    if sp.backend == "coreai":
        from bench.coreai_infer import CoreAISession

        return CoreAISession(
            sp.model_id,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
        )
    return MlxSession(
        sp.model_id,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        batch_size=batch_size,
    )
