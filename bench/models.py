from __future__ import annotations

from dataclasses import dataclass

GGUF_REPO = "LiquidAI/LFM2.5-VL-3B-GGUF"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backend: str  # "mlx" | "gguf"
    model_id: str
    gguf_file: str | None = None
    mmproj_file: str | None = None


MLX_ALIASES = {
    "4bit": "LiquidAI/LFM2.5-VL-3B-MLX-4bit",
    "5bit": "LiquidAI/LFM2.5-VL-3B-MLX-5bit",
    "6bit": "LiquidAI/LFM2.5-VL-3B-MLX-6bit",
    "8bit": "LiquidAI/LFM2.5-VL-3B-MLX-8bit",
    "bf16": "LiquidAI/LFM2.5-VL-3B-MLX-bf16",
}

_MMPROJ_Q8 = "mmproj-LFM2.5-VL-3B-Q8_0.gguf"
GGUF_FILES = {
    "Q4_0": ("LFM2.5-VL-3B-Q4_0.gguf", _MMPROJ_Q8),
    "Q4_K_M": ("LFM2.5-VL-3B-Q4_K_M.gguf", _MMPROJ_Q8),
    "Q5_K_M": ("LFM2.5-VL-3B-Q5_K_M.gguf", _MMPROJ_Q8),
    "Q6_K": ("LFM2.5-VL-3B-Q6_K.gguf", _MMPROJ_Q8),
    "Q8_0": ("LFM2.5-VL-3B-Q8_0.gguf", _MMPROJ_Q8),
    "BF16": ("LFM2.5-VL-3B-BF16.gguf", "mmproj-LFM2.5-VL-3B-BF16.gguf"),
    "F16": ("LFM2.5-VL-3B-F16.gguf", "mmproj-LFM2.5-VL-3B-F16.gguf"),
}

GGUF_ALIASES = {
    "gguf-q4": "Q4_0",
    "gguf-q4km": "Q4_K_M",
    "gguf-q5": "Q5_K_M",
    "gguf-q6": "Q6_K",
    "gguf-q8": "Q8_0",
    "gguf-bf16": "BF16",
    "gguf-f16": "F16",
}

ALIASES = {
    **{k: v for k, v in MLX_ALIASES.items()},
    **{k: f"{GGUF_REPO}:{q}" for k, q in GGUF_ALIASES.items()},
}

DEFAULT_MODELS = ["6bit", "8bit", "bf16"]


def _gguf_spec(name: str, quant: str) -> ModelSpec:
    files = GGUF_FILES.get(quant)
    gguf, mmproj = files if files else (None, None)
    return ModelSpec(name, "gguf", f"{GGUF_REPO}:{quant}", gguf, mmproj)


def spec(name: str) -> ModelSpec:
    key = name.strip()
    if key in MLX_ALIASES:
        return ModelSpec(key, "mlx", MLX_ALIASES[key])
    if key == "coreai":
        # optional backend — Apple Core AI conversion, macOS 27+ only
        return ModelSpec(key, "coreai", "mlboydaisuke/LFM2.5-VL-3B-CoreAI")
    if key in GGUF_ALIASES:
        return _gguf_spec(key, GGUF_ALIASES[key])
    if key.endswith(".gguf"):
        return ModelSpec(key, "gguf", key)
    if "GGUF" in key or key.lower().startswith("gguf:"):
        hid = key[5:] if key.lower().startswith("gguf:") else key
        quant = hid.rsplit(":", 1)[-1] if ":" in hid else "Q8_0"
        if ":" not in hid:
            hid = f"{hid}:{quant}"
        files = GGUF_FILES.get(quant)
        gguf, mmproj = files if files else (None, None)
        return ModelSpec(key, "gguf", hid, gguf, mmproj)
    return ModelSpec(key, "mlx", key)


def resolve(name: str) -> str:
    return spec(name).model_id


def parse_models(spec_str: str) -> list[str]:
    return [p.strip() for p in spec_str.split(",") if p.strip()]
