"""Prompt wording for benchmark tasks and protocol variants.

Echo history: on the original untiled stack, a bracketed literal like
"[0, 1000]" in a plain-bbox prompt could come back verbatim as the answer and
be read as a prediction (14.6% vs 58.3%, 48-item slice, MLX 8-bit). That does
not reproduce on the current tiled stack — the model's [0, 0, 1000, 1000]
outputs on unfound targets are its own whole-image failure box, present with
or without any literal, and the official grounding system prompt below
contains bracketed literals by design. Avoiding numeric bracketed literals in
custom plain-box prompts is still cheap hygiene: the bbox parser will read
any echoed bracket.
"""

from __future__ import annotations

# `bbox` protocol wording, also baked into prepared ScreenSpot tasks at
# prepare time (bench/screenspot.py).
BBOX_FMT = (
    "Reply with only the bounding box as xmin, ymin, xmax, ymax, "
    "using integers from 0 to 1000."
)

SCREENSPOT_PROMPT = "Detect the UI element for this instruction: {instruction}\n" + BBOX_FMT

# `pyautogui` protocol — Liquid's ScreenSpot-v2 wording, verbatim from their
# VLMEvalKit fork (Liquid4All/VLMEvalKit_Liquid, vlmeval/dataset/GUI/
# screenspot.py) plus the "answer directly" suffix their model wrapper
# appends. Their harness drops the system prompt for single-image items, so
# this protocol sends the user message alone (which is why PYAUTOGUI_SYSTEM
# below is reference-only, deliberately NOT sent). Every variant tested
# scored at most 6.7% locally (user-alone: 0.0%, answers like "Yes" never
# parse; sys+user with coords re-read 0-1000: 6.7%) vs 75.6% for
# grounding_json on the same 45-item slice. It exists for comparability with
# the published 80.7, not for accuracy.
PYAUTOGUI_SYSTEM = (
    "You are a GUI agent. You are given a task and a screenshot of the screen. "
    "You need to perform pyautogui click/moveTo action to complete the task. "
    "The answer format is `pyautogui.click(x=?, y=?), x and y is necessary`"
)
PYAUTOGUI_USER = (
    "Please complete the following tasks by clicking using `pyautogui.click`:\n{instruction}"
)
ANSWER_DIRECTLY = (
    "\nPlease answer directly with only the final answer, do not give any explanation."
)

# `grounding_json` protocol (default) — official recipe from docs.liquid.ai
# (lfm/key-concepts/vision-capabilities, "Object detection and grounding").
# The JSON bbox_2d format is the model's native grounding output — it shows
# up spontaneously even when prompted for plain boxes. Full 1,272-item set,
# MLX bf16: 80.8% vs 64.3% macro for the plain bbox prompt.
GROUNDING_JSON_SYSTEM = (
    "When asked for bounding boxes for objects, return a valid JSON array.\n"
    "Each array item must be an object with:\n"
    "- image_id: the 0-based index of the image\n"
    "- bbox_2d: [xmin, ymin, xmax, ymax] normalized integer coordinates in [0, 1000]\n"
    "- label: a concise label you choose for the predicted object or region\n"
    "Return one item per visible matching object or region. Return [] if none are visible."
)
GROUNDING_JSON_USER = "Provide bounding boxes for the UI element for this instruction: {instruction}"
