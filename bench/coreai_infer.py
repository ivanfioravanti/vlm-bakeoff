"""Core AI backend — LFM2.5-VL-3B via Apple's Core AI runtime (.aimodel).

Optional backend (macOS 27+, `uv sync --extra coreai`). Drives the community
two-bundle conversion (mlboydaisuke/LFM2.5-VL-3B-CoreAI): a fixed-grid
SigLIP2 vision tower (one 512x512 view -> 256 image tokens) plus the LFM2
hybrid decoder with ``image_embeds`` riding the static-input hook, exactly the
host loop the conversion's engine gate uses (john-rocky/coreai-model-zoo).

Known comparability limits, by construction of the conversion:
  * ONE 512x512 stretched view per image — no tiling — so document-heavy
    tracks see far less resolution than the tiled MLX/GGUF paths.
  * single-image items only; multi-image (BLINK) items are unsupported and
    skipped by the runner.
"""

from __future__ import annotations

import inspect
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from bench.models import ModelSpec
from bench.tasks import image_paths

COREAI_REPO = "mlboydaisuke/LFM2.5-VL-3B-CoreAI"
# Pinned commit: the repo's layout moved upstream after this was captured
# (files 404 at HEAD); this host loop is built and validated against it.
COREAI_REVISION = "aa121aa373554c65f2c4b7ae8062bfeb803211cf"
HF_TEXT_REPO = "LiquidAI/LFM2.5-VL-3B-MLX-4bit"  # tokenizer + config source
VISION_NAME = "gpu-pipelined/lfm2_5_vl_3b_vision_fp16"
DECODER_NAME = "gpu-pipelined/lfm2_5_vl_3b_decode_int8lin"
AOT_CACHE = Path.home() / ".cache" / "vlm-bakeoff" / "coreai"

TILE = 512
PATCH = 16
GRID = TILE // PATCH  # 32x32 patches -> 256 tokens after 2x downsample
N_IMAGE_TOKENS = GRID * GRID // 4
KV_SEQ = 2048


def coreai_available() -> tuple[bool, str]:
    """(ok, reason) — macOS 27+ and the optional coreai-core package."""
    try:
        import platform

        ver = platform.mac_ver()[0]
        major = int(ver.split(".")[0]) if ver else 0
        if major < 27:
            return False, f"Core AI needs macOS 27+ (this machine runs {ver or 'unknown'})"
    except Exception as exc:  # pragma: no cover
        return False, f"cannot determine macOS version: {exc}"
    try:
        import coreai.runtime  # noqa: F401
    except ImportError:
        return False, "coreai-core not installed — run: uv sync --extra coreai"
    return True, ""


class _Loop:
    """One persistent event loop for the session's async runtime calls.

    coreai-core mixes sync and async APIs (AIModel.load is awaitable,
    load_function returns a plain object), so only drive real awaitables.
    """

    def __init__(self) -> None:
        import asyncio

        self._loop = asyncio.new_event_loop()

    def run(self, x):
        if not inspect.isawaitable(x):
            return x

        import asyncio

        async def _wrap():
            return await x

        return self._loop.run_until_complete(_wrap())


_loop = _Loop()


def _run(x):
    return _loop.run(x)


def _load_decoder(rt, raw: Path):
    """Load the AOT decoder, trying each compiled architecture until one fits.

    The bundle's input_ids dim is fixed [1,1] (token-at-a-time contract), so
    prefill runs as S=1 steps — the AOT build with --expect-frequent-reshapes
    keeps that from re-specializing per position length.
    """
    opts = rt.SpecializationOptions.default()
    out_dir = AOT_CACHE / f"{raw.stem}_aotc"
    if not out_dir.is_dir() or not sorted(out_dir.glob("*.aimodelc")):
        _compile_aot(raw, out_dir)
    candidates = sorted(out_dir.glob("*.aimodelc"))
    # newest GPU families first — the first that loads is the right one
    candidates.sort(key=lambda p: tuple(-int(c[1:]) for c in re.findall(r"h(\d+)", p.name)))
    last_exc = None
    for cand in candidates:
        try:
            dm = _run(rt.AIModel.load(str(cand), opts))
            fn = _run(dm.load_function(dm.function_names[0]))
            print(f"  coreai: decoder {cand.name} ready")
            return fn
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"no AOT specialization loaded for {raw.name}: {last_exc}")


def _compile_aot(raw: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  coreai: AOT-compiling decoder (one-time, ~30 min) → {out_dir}")
    t0 = time.perf_counter()
    subprocess.run(
        [
            "xcrun",
            "coreai-build",
            "compile",
            str(raw),
            "--platform",
            "macOS",
            "--preferred-compute",
            "gpu",
            "--expect-frequent-reshapes",
            # no --architecture: compile for all supported ones (a single
            # hand-picked arch can fail to load with AIModelError error 0)
            "--output",
            str(out_dir),
        ],
        check=True,
    )
    if not sorted(out_dir.glob("*.aimodelc")):
        raise FileNotFoundError(f"coreai-build produced no .aimodelc in {out_dir}")
    print(f"  coreai: AOT compile done in {(time.perf_counter()-t0)/60:.1f} min")


class _Model:
    """Loaded bundles + the token-stepping host loop (asyncio-free surface)."""

    # 9 = the chunk size the conversion author verified (verify_s9); the fp16
    # chunked-scan inverse goes unstable past ~32 tokens and SIGTRAPs on this
    # runtime at 32 in our sweep (9 and 16 held 40/40 token-exact vs S=1).
    CHUNK = 9

    def __init__(self) -> None:
        import coreai.runtime as rt
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        self.rt = rt
        vision_raw = Path(
            hf_hub_download(
                COREAI_REPO,
                f"{VISION_NAME}/{Path(VISION_NAME).name}.aimodel",
                revision=COREAI_REVISION,
            )
        )
        # prefer a locally re-exported decoder with a dynamic input_ids seq dim
        # (chunked prefill — see README "Core AI backend"); fall back to the
        # pinned HF bundle (token-at-a-time)
        chunked = AOT_CACHE / "exports" / "lfm2_5_vl_3b_decode_int8lin_chunked" / "lfm2_5_vl_3b_decode_int8lin_chunked.aimodel"
        decoder_raw = chunked if chunked.exists() else Path(
            hf_hub_download(
                COREAI_REPO,
                f"{DECODER_NAME}/{Path(DECODER_NAME).name}.aimodel",
                revision=COREAI_REVISION,
            )
        )

        cfg_raw = json.loads(Path(hf_hub_download(HF_TEXT_REPO, "config.json")).read_text())
        text = cfg_raw["text_config"]
        layer_types = text.get("layer_types") or []
        self.num_full = sum(1 for t in layer_types if t == "full_attention")
        self.num_conv = text["num_hidden_layers"] - self.num_full
        self.hidden = text["hidden_size"]
        self.vocab = text["vocab_size"]
        self.head_dim = self.hidden // text["num_attention_heads"]
        self.n_kv = text.get("num_key_value_heads", 8)
        self.conv_w = text.get("conv_L_cache", 3) - 1
        self.image_token_id = int(cfg_raw.get("image_token_id", 124907))
        self.resample = Image.BICUBIC  # 3B processor: resample 3

        self.tok = AutoTokenizer.from_pretrained(HF_TEXT_REPO)

        opts = rt.SpecializationOptions.default()
        vm = _run(rt.AIModel.load(str(vision_raw), opts))
        self.vfn = _run(vm.load_function(vm.function_names[0]))
        self.dfn = _load_decoder(rt, decoder_raw)
        self.stop_ids = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")}
        # set once the first prefill probes the graph: True for dynamic-seq
        # exports (chunked prefill), False for the pinned [1,1] HF bundle
        self.chunk_ok = None

    def preprocess(self, img: Image.Image) -> np.ndarray:
        """RGB → 512² stretch → (x-0.5)/0.5 → [1024, 768] patches, channel-fastest."""
        x = img.convert("RGB").resize((TILE, TILE), self.resample)
        a = np.asarray(x, dtype=np.float32) / 255.0
        a = (a - 0.5) / 0.5
        p = a.reshape(GRID, PATCH, GRID, PATCH, 3).transpose(0, 2, 1, 3, 4)
        return np.ascontiguousarray(p.reshape(GRID * GRID, 3 * PATCH * PATCH), dtype=np.float16)

    def vision(self, img: Image.Image) -> np.ndarray:
        out = _run(self.vfn(inputs={"patches": self.rt.NDArray(self.preprocess(img))}))
        return np.ascontiguousarray(np.asarray(out["image_embeds"].numpy()).astype(np.float16))

    def fresh_state(self) -> dict:
        rt = self.rt
        kshape = (self.num_full, 1, self.n_kv, KV_SEQ, self.head_dim)
        return {
            "keyCache": rt.NDArray(np.zeros(kshape, dtype=np.float16)),
            "valueCache": rt.NDArray(np.zeros(kshape, dtype=np.float16)),
            "convState": rt.NDArray(
                np.zeros((self.num_conv, 1, self.hidden, self.conv_w), dtype=np.float16)
            ),
        }

    def step(self, token: int, pos: int, image_embeds, state) -> np.ndarray:
        inputs = {
            "input_ids": self.rt.NDArray(np.array([[token]], dtype=np.int32)),
            "position_ids": self.rt.NDArray(np.arange(pos + 1, dtype=np.int32)[None]),
            "image_embeds": image_embeds,
        }
        res = _run(self.dfn(inputs=inputs, state=state))
        return np.asarray(res["logits"].numpy())[0, -1]

    def step_chunk(self, tokens: list[int], pos_end: int, image_embeds, state) -> np.ndarray:
        """Feed a block of tokens at once (dynamic-seq bundle export only)."""
        inputs = {
            "input_ids": self.rt.NDArray(np.array([tokens], dtype=np.int32)),
            "position_ids": self.rt.NDArray(np.arange(pos_end + 1, dtype=np.int32)[None]),
            "image_embeds": image_embeds,
        }
        res = _run(self.dfn(inputs=inputs, state=state))
        return np.asarray(res["logits"].numpy())[0, -1]

    def prefill(self, ids: list[int], image_embeds, state) -> np.ndarray:
        """Chunked when the bundle's input_ids dim is dynamic, else S=1 steps."""
        start = 0
        if self.chunk_ok is None:
            try:
                probe = ids[: self.CHUNK]
                logits = self.step_chunk(probe, len(probe) - 1, image_embeds, state)
                self.chunk_ok = True
                start = len(probe)
            except RuntimeError:
                self.chunk_ok = False
        if self.chunk_ok:
            for c0 in range(start, len(ids), self.CHUNK):
                chunk = ids[c0 : c0 + self.CHUNK]
                logits = self.step_chunk(chunk, c0 + len(chunk) - 1, image_embeds, state)
            return logits
        logits = None
        for i, token in enumerate(ids):
            logits = self.step(int(token), i, image_embeds, state)
        return logits


class Generation:
    text: str

    def __init__(self, text: str):
        self.text = text


class CoreAISession:
    backend = "coreai"

    def __init__(
        self,
        model_id: str,
        temperature: float = 0.0,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        batch_size: int = 1,
    ):
        ok, reason = coreai_available()
        if not ok:
            raise SystemExit(f"coreai backend unavailable: {reason}")
        self.temperature = temperature
        self.top_k = top_k
        self.model = _Model()

    def supports_task(self, task: dict[str, Any]) -> bool:
        # the conversion bakes a single 512x512 view: one image per item only
        return len(task.get("images") or []) == 1

    def batchable(self, task: dict[str, Any]) -> bool:
        return False

    def _prompt_ids(self, task: dict) -> np.ndarray:
        from mlx_vlm import apply_chat_template

        prompt = apply_chat_template(
            self.model.tok,
            {"model_type": "lfm2_vl"},
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": task["prompt"]},
                    ],
                }
            ],
            add_generation_prompt=True,
            num_images=1,
        )
        ids = np.asarray(
            self.model.tok(prompt, add_special_tokens=False)["input_ids"], dtype=np.int64
        )
        pos = np.nonzero(ids == self.model.image_token_id)[0]
        if pos.size == 1:
            p = int(pos[0])
            ids = np.concatenate(
                [ids[:p], np.full(N_IMAGE_TOKENS, self.model.image_token_id, dtype=ids.dtype), ids[p + 1 :]]
            )
        else:
            raise ValueError(f"expected 1 image marker in prompt, found {pos.size}")
        return ids

    def _sample(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        if self.temperature == 0:
            return int(logits.argmax())
        logits = logits.astype(np.float64) / self.temperature
        if self.top_k:
            k = min(self.top_k, logits.size)
            cutoff = np.partition(logits, -k)[-k]
            logits[logits < cutoff] = -np.inf
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return int(rng.choice(logits.size, p=probs))

    def generate(self, task: dict[str, Any]) -> Generation:
        if not self.supports_task(task):
            raise ValueError("coreai backend supports single-image items only")
        paths = image_paths(task)
        image_embeds = self.model.rt.NDArray(self.model.vision(Image.open(paths[0])))
        ids = self._prompt_ids(task)
        ids = ids.copy()
        img_pos = np.nonzero(ids == self.model.image_token_id)[0]
        ids[img_pos] = self.model.vocab + np.arange(img_pos.size)

        state = self.model.fresh_state()
        rng = np.random.default_rng()

        all_ids = ids.tolist()
        logits = self.model.prefill(all_ids, image_embeds, state)
        stop = self.model.stop_ids
        out_ids: list[int] = []
        for k in range(int(task.get("max_tokens", 128))):
            nxt = self._sample(logits, rng)
            if nxt in stop:
                break
            out_ids.append(nxt)
            logits = self.model.step(nxt, len(all_ids) + k, image_embeds, state)
        return Generation(self.model.tok.decode(out_ids).strip())

    def generate_batch(self, tasks: list[dict[str, Any]]) -> list[Generation]:
        return [self.generate(t) for t in tasks]

    def close(self) -> None:
        self.model = None


def CoreAISessionFactory(spec: ModelSpec, **kwargs) -> CoreAISession:
    return CoreAISession(spec.model_id, **kwargs)
