from __future__ import annotations

import html
from typing import Any

from bench.blink import LIQUID_BLINK
from bench.docvqa import LIQUID_DOCVQA_ANLS, LIQUID_INFOGRAPHICVQA_ANLS
from bench.refcoco import LIQUID_REFERCOCO_AVG, SPLITS
from bench.screenspot import LIQUID_SCREENSPOT_AVG, LIQUID_SCREENSPOT_V2

ANLS_TRACKS = {
    "docvqa": ("DocVQA", LIQUID_DOCVQA_ANLS, 5_349),
    "infographicvqa": ("InfographicVQA", LIQUID_INFOGRAPHICVQA_ANLS, 2_801),
}

_CSS = """
:root {
  --ink: #100e0c;
  --paper: #191613;
  --raised: #211e1a;
  --line: #322c26;
  --copper: #d08a45;
  --copper-dim: #8a5a2c;
  --sage: #9aa57f;
  --clay: #c45c3e;
  --fog: #d4cbc0;
  --mute: #7d746a;
  --hair: rgba(208, 138, 69, 0.22);
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--ink); color: var(--fog); }
body {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 14px;
  line-height: 1.5;
  min-height: 100vh;
  background-image:
    radial-gradient(1200px 600px at 10% -10%, rgba(208,138,69,.08), transparent 55%),
    radial-gradient(900px 500px at 110% 0%, rgba(154,165,127,.05), transparent 50%);
}
body::before {
  content: "";
  position: fixed; inset: 0; pointer-events: none; z-index: 8;
  opacity: .045;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='140' height='140'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2' stitchTiles='stitch'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>");
}
a { color: var(--copper); }
.wrap { max-width: 1180px; margin: 0 auto; padding: 48px 28px 96px; }
.mast {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 24px;
  align-items: end;
  border-bottom: 1px solid var(--line);
  padding-bottom: 28px;
  margin-bottom: 36px;
}
h1 {
  font-family: Fraunces, Georgia, serif;
  font-weight: 360;
  font-size: clamp(40px, 6vw, 72px);
  line-height: .92;
  letter-spacing: -.03em;
  margin: 0 0 12px;
  color: #f3ece3;
}
.kicker {
  color: var(--copper);
  letter-spacing: .22em;
  text-transform: uppercase;
  font-size: 11px;
  margin-bottom: 10px;
}
.lede { max-width: 46ch; color: var(--mute); font-size: 13px; }
.meta {
  text-align: right;
  color: var(--mute);
  font-size: 12px;
  display: grid;
  gap: 4px;
}
.meta b { color: var(--fog); font-weight: 500; }
.plate { margin: 8px 0 40px; }
.plate-head, .plate-row {
  display: grid;
  grid-template-columns: 44px repeat(var(--cols), minmax(0, 1fr));
  gap: 16px;
  align-items: stretch;
}
.plate-head { margin-bottom: 8px; }
.plate-row + .plate-row { margin-top: 16px; }
.plate-col {
  color: var(--mute);
  font-size: 11px;
  letter-spacing: .16em;
  text-transform: uppercase;
  padding-left: 4px;
}
.plate-lab {
  color: var(--copper);
  letter-spacing: .18em;
  text-transform: uppercase;
  font-size: 11px;
  writing-mode: vertical-rl;
  transform: rotate(180deg);
  display: flex;
  align-items: center;
  justify-content: center;
}
.models {
  display: contents;
}
.card.ghost {
  border-style: dashed;
  background: transparent;
  min-height: 168px;
  animation: none;
}
.card.ghost::after { content: none; }
.card {
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 22px 22px 18px;
  position: relative;
  overflow: hidden;
  animation: rise .55s cubic-bezier(.16,1,.3,1) both;
}
.card:nth-child(2) { animation-delay: .07s; }
.card:nth-child(3) { animation-delay: .14s; }
.card:nth-child(4) { animation-delay: .21s; }
@keyframes rise {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: none; }
}
@media (prefers-reduced-motion: reduce) {
  .card { animation: none; }
}
.card::after {
  content: "";
  position: absolute; right: -40px; top: -40px;
  width: 140px; height: 140px; border-radius: 50%;
  background: radial-gradient(circle, var(--hair), transparent 70%);
}
.alias { color: var(--copper); font-size: 12px; letter-spacing: .14em; text-transform: uppercase; }
.hid { color: var(--mute); font-size: 11px; margin: 6px 0 18px; word-break: break-all; }
.score {
  font-family: Fraunces, Georgia, serif;
  font-size: 64px;
  font-weight: 350;
  letter-spacing: -.04em;
  line-height: .9;
  color: #f6efe6;
}
.score small { font-size: 22px; color: var(--mute); }
.frac { margin-top: 10px; color: var(--mute); font-size: 12px; }
.ring {
  width: 54px; height: 54px; border-radius: 50%;
  background: conic-gradient(var(--copper) calc(var(--p) * 1%), var(--line) 0);
  mask: radial-gradient(farthest-side, transparent 62%, #000 63%);
  -webkit-mask: radial-gradient(farthest-side, transparent 62%, #000 63%);
  float: right;
}
h2 {
  font-family: Fraunces, Georgia, serif;
  font-weight: 420;
  font-size: 28px;
  margin: 48px 0 16px;
  color: #f3ece3;
}
details.cases > summary {
  font-family: Fraunces, Georgia, serif;
  font-weight: 420;
  font-size: 28px;
  margin: 48px 0 16px;
  color: #f3ece3;
  cursor: pointer;
  list-style: none;
  display: flex;
  align-items: baseline;
  gap: 10px;
}
details.cases > summary::-webkit-details-marker { display: none; }
details.cases > summary::before { content: "▸"; color: var(--copper); font-size: 20px; }
details.cases[open] > summary::before { content: "▾"; }
details.cases > summary .count { color: var(--mute); font-size: 13px; }
.note { color: var(--mute); font-size: 12px; margin: -8px 0 18px; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--line); vertical-align: middle; }
th { color: var(--mute); font-size: 11px; letter-spacing: .16em; text-transform: uppercase; font-weight: 500; }
.dim { color: var(--mute); font-size: 10px; letter-spacing: .08em; text-transform: none; }
.heat {
  display: inline-block; min-width: 72px; padding: 3px 8px; border-radius: 999px;
  text-align: center; font-size: 12px;
  background: color-mix(in srgb, var(--copper) calc(var(--p) * .7%), var(--raised));
  color: #f6efe6;
}
.bar {
  height: 7px; background: var(--line); border-radius: 99px; overflow: hidden; min-width: 88px;
}
.bar > i { display: block; height: 100%; background: var(--copper); width: var(--p); }
td.ref { color: var(--mute); font-size: 12px; }
.speed { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
.tq { display: grid; gap: 10px; margin-top: 8px; }
.tq-head, .tq-row {
  display: grid;
  grid-template-columns: 34px minmax(130px, 210px) 1fr 128px 82px 82px;
  gap: 14px;
  align-items: center;
}
.tq-head { color: var(--mute); font-size: 11px; letter-spacing: .16em; text-transform: uppercase; padding: 0 14px; }
.tq-row {
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 10px 14px;
}
.tq-rank {
  font-family: Fraunces, Georgia, serif;
  font-size: 20px;
  color: var(--copper);
  text-align: center;
}
.tq-name { color: #f3ece3; font-size: 13px; }
.tq-name .cat { margin-left: 6px; }
.tq-time { color: #f3ece3; font-size: 13px; text-align: right; white-space: nowrap; }
.tq-time .u { color: var(--mute); font-size: 11px; margin-left: 6px; }
@media (max-width: 720px) {
  .tq-head, .tq-row { grid-template-columns: 28px 1fr 110px; }
  .hide-sm { display: none; }
}
.dial {
  background: var(--paper); border: 1px solid var(--line); border-radius: 14px; padding: 16px;
}
.dial .v {
  font-family: Fraunces, Georgia, serif; font-size: 28px; color: #f3ece3;
}
.dial .u { color: var(--mute); font-size: 11px; }
.filters { display: flex; gap: 8px; flex-wrap: wrap; margin: 0 0 16px; }
.filters label {
  border: 1px solid var(--line); border-radius: 999px; padding: 6px 12px; cursor: pointer; color: var(--mute);
}
.filters input { display: none; }
.filters input:checked + label { border-color: var(--copper); color: var(--copper); }
.log { display: grid; gap: 8px; }
.case {
  background: var(--paper); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px;
}
.case.fail { border-color: color-mix(in srgb, var(--clay) 45%, var(--line)); }
.row1 { display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }
.pill {
  font-size: 10px; letter-spacing: .14em; text-transform: uppercase;
  padding: 2px 7px; border-radius: 999px;
}
.pass .pill { background: color-mix(in srgb, var(--sage) 25%, transparent); color: var(--sage); }
.fail .pill { background: color-mix(in srgb, var(--clay) 22%, transparent); color: var(--clay); }
.id { color: #f3ece3; }
.cat { color: var(--mute); }
.out {
  margin: 8px 0 0; color: var(--mute); white-space: pre-wrap; word-break: break-word;
  font-size: 12px; max-height: 7.5em; overflow: auto;
}
.out .exp { color: var(--copper-dim); }
body[data-filter="fail"] .case.pass { display: none; }
body[data-filter="pass"] .case.fail { display: none; }
@media (max-width: 720px) {
  .mast { grid-template-columns: 1fr; }
  .meta { text-align: left; }
  .score { font-size: 48px; }
  .plate-head, .plate-row {
    grid-template-columns: 28px repeat(var(--cols), minmax(0, 1fr));
    gap: 8px;
  }
}
@media print {
  body { background: #fff; color: #111; }
  .card, .dial, .case { break-inside: avoid; }
}
"""


def _esc(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def _pct_num(x: float | None) -> float:
    return 0.0 if x is None else 100.0 * x


def _fmt_dur(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


_FAMILY_RANK = ("4", "5", "6", "8", "bf16", "f16")
_FAMILY_LABEL = {
    "4": "4-bit",
    "5": "5-bit",
    "6": "6-bit",
    "8": "8-bit",
    "bf16": "bf16",
    "f16": "f16",
}
_FAMILY_NEEDLES = (
    ("bf16", ("bf16",)),
    ("f16", ("fp16", "f16")),
    ("4", ("4bit", "q4_k", "q4km", "q4_0", "q4")),
    ("5", ("5bit", "q5_k", "q5_0", "q5")),
    ("6", ("6bit", "q6_k", "q6_0", "q6")),
    ("8", ("8bit", "q8_0", "q8")),
)


_BACKEND_LABEL = {"mlx": "MLX", "gguf": "GGUF", "coreai": "Core AI"}
_BACKEND_ENGINE = {"mlx": "mlx-vlm", "gguf": "llama.cpp", "coreai": "Core AI (.aimodel)"}


def _backend_of(name: str, run: dict[str, Any]) -> str:
    backend = run.get("backend")
    if backend:
        return str(backend)
    blob = f"{name} {run.get('model_id') or ''}".lower()
    return "gguf" if "gguf" in blob else "mlx"


def _backends_used(models: list[str], runs: dict[str, Any]) -> list[str]:
    seen: list[str] = []
    for m in models:
        backend = _backend_of(m, runs[m])
        if backend not in seen:
            seen.append(backend)
    if "mlx" in seen:
        seen = ["mlx"] + [b for b in seen if b != "mlx"]
    return seen


def _join(parts: list[str]) -> str:
    if len(parts) <= 1:
        return parts[0] if parts else ""
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def _quant_family(name: str, model_id: str = "") -> str:
    blob = f"{name} {model_id}".lower().replace("-", "_")
    for family, needles in _FAMILY_NEEDLES:
        if any(n in blob for n in needles):
            return family
    return name


def _plate_columns(models: list[str], runs: dict[str, Any]) -> list[str]:
    seen: list[str] = []
    for m in models:
        fam = _quant_family(m, runs[m].get("model_id") or "")
        if fam not in seen:
            seen.append(fam)
    known = [f for f in _FAMILY_RANK if f in seen]
    rest = [f for f in seen if f not in _FAMILY_RANK]
    return known + rest


def _plate_rows(
    models: list[str], runs: dict[str, Any], columns: list[str]
) -> list[tuple[str, list[str | None]]]:
    by_backend: dict[str, dict[str, list[str]]] = {}
    for m in models:
        backend = _backend_of(m, runs[m])
        fam = _quant_family(m, runs[m].get("model_id") or "")
        if backend not in by_backend:
            by_backend[backend] = {c: [] for c in columns}
        by_backend[backend].setdefault(fam, []).append(m)
    rows: list[tuple[str, list[str | None]]] = []
    for backend in _backends_used(models, runs):
        slots = by_backend[backend]
        depth = max((len(slots.get(c) or []) for c in columns), default=1) or 1
        for i in range(depth):
            cells = [(slots.get(c) or [None] * depth)[i] if i < len(slots.get(c) or []) else None for c in columns]
            name = _BACKEND_LABEL.get(backend, backend.upper())
            rows.append((name if depth == 1 else f"{name} {i + 1}", cells))
    return rows


def _model_card(m: str | None, summaries: dict[str, Any], runs: dict[str, Any]) -> str:
    if m is None:
        return '<article class="card ghost" aria-hidden="true"></article>'
    s = summaries[m]
    p = _pct_num(s["overall"])
    hid = _esc(runs[m].get("model_id") or m)
    backend = runs[m].get("backend")
    if backend:
        hid = f"{backend} · {hid}"
    if s["overall"] is None:
        score_html = '<div class="score">—</div>'
    else:
        score_html = f'<div class="score">{p:.1f}<small>%</small></div>'
    return f"""<article class="card">
  <div class="ring" style="--p:{p:.1f}"></div>
  <div class="alias">{_esc(m)}</div>
  <div class="hid">{hid}</div>
  {score_html}
  <div class="frac">{s["n_pass"]}/{s["n"]} passed</div>
</article>"""


def render_html(payload: dict[str, Any]) -> str:
    from bench.report import summarize

    models = payload["models"]
    runs = payload["runs"]
    summaries = {m: summarize(runs[m]) for m in models}
    cats = sorted({c for s in summaries.values() for c in s["by_category"]})
    plats = sorted({p for s in summaries.values() for p in s["screenspot_by_platform"]})
    has_cases = any(summaries[m]["n"] for m in models)

    columns = _plate_columns(models, runs)
    plate_rows = _plate_rows(models, runs, columns)
    head = "".join(
        f'<div class="plate-col">{_esc(_FAMILY_LABEL.get(c, c))}</div>' for c in columns
    )
    row_html = []
    for label, cells in plate_rows:
        cards = "".join(_model_card(m, summaries, runs) for m in cells)
        row_html.append(
            f'<div class="plate-row"><div class="plate-lab">{_esc(label)}</div>'
            f'<div class="models">{cards}</div></div>'
        )

    cat_rows = []
    for cat in cats:
        cells = []
        for m in models:
            val = summaries[m]["by_category"].get(cat)
            p = _pct_num(val)
            cells.append(
                f'<td><span class="heat" style="--p:{p:.0f}">{_pct(val)}</span></td>'
            )
        cat_rows.append(f"<tr><th>{_esc(cat)}</th>{''.join(cells)}</tr>")

    plat_rows = []
    for plat in plats:
        cells = []
        for m in models:
            val = summaries[m]["screenspot_by_platform"].get(plat)
            p = _pct_num(val)
            cells.append(
                f'<td><div class="bar" title="{_pct(val)}"><i style="--p:{p:.0f}%"></i></div></td>'
            )
        ref = LIQUID_SCREENSPOT_V2.get(plat)
        ref_cell = f'<td class="ref">{_pct(ref)}</td>' if ref is not None else "<td></td>"
        plat_rows.append(f"<tr><th>{_esc(plat)}</th>{''.join(cells)}{ref_cell}</tr>")
    if any(summaries[m].get("screenspot_macro") is not None for m in models):
        cells = []
        for m in models:
            val = summaries[m].get("screenspot_macro")
            p = _pct_num(val)
            cells.append(
                f'<td><div class="bar" title="{_pct(val)}"><i style="--p:{p:.0f}%"></i></div></td>'
            )
        plat_rows.append(
            f"<tr><th>unweighted avg</th>{''.join(cells)}"
            f'<td class="ref">{_pct(LIQUID_SCREENSPOT_AVG)}</td></tr>'
        )

    types = sorted({t for s in summaries.values() for t in s.get("screenspot_by_type") or {}})
    type_rows = []
    for t in types:
        cells = []
        for m in models:
            val = summaries[m]["screenspot_by_type"].get(t)
            p = _pct_num(val)
            cells.append(
                f'<td><span class="heat" style="--p:{p:.0f}">{_pct(val)}</span></td>'
            )
        type_rows.append(f"<tr><th>{_esc(t)}</th>{''.join(cells)}</tr>")

    subsets_present = sorted(
        {s for s in summaries.values() for s in s.get("refcoco_by_subset") or {}}
    )
    refcoco_rows = []
    for s in [x for x in SPLITS if x in subsets_present]:
        cells = []
        for m in models:
            val = summaries[m]["refcoco_by_subset"].get(s)
            p = _pct_num(val)
            cells.append(
                f'<td><span class="heat" style="--p:{p:.0f}">{_pct(val)}</span></td>'
            )
        refcoco_rows.append(f"<tr><th>{_esc(s)}</th>{''.join(cells)}<td></td></tr>")
    for label, key, ref in (
        ("avg — box center in gold", "refcoco_macro", LIQUID_REFERCOCO_AVG),
        ("avg — IoU ≥ 0.5", "refcoco_iou_macro", None),
    ):
        cells = []
        for m in models:
            val = summaries[m].get(key)
            p = _pct_num(val)
            cells.append(
                f'<td><div class="bar" title="{_pct(val)}"><i style="--p:{p:.0f}%"></i></div></td>'
            )
        ref_cell = (
            f'<td class="ref">{_pct(ref)}</td>' if ref is not None else "<td></td>"
        )
        refcoco_rows.append(
            f"<tr><th>{_esc(label)}</th>{''.join(cells)}{ref_cell}</tr>"
        )

    speed_html = ""
    if payload.get("speed"):
        dials = []
        for m, rows in payload["speed"].items():
            for row in rows:
                dials.append(
                    f"""<div class="dial">
  <div class="alias">{_esc(m)} · {_esc(row["setting"])}</div>
  <div class="v">{row["tok_s"]:.1f} <span class="u">tok/s</span></div>
  <div class="u">TTFT {row["ttft_s"]:.3f}s · peak {row["peak_gb"]:.2f} GB</div>
</div>"""
                )
        speed_html = (
            "<h2>Speed</h2>"
            '<p class="note">Wall-clock including prefill. Not Liquid’s published decode-only figure.</p>'
            f'<div class="speed">{"".join(dials)}</div>'
        )

    cases_html = ""
    if has_cases:
        def _expected_str(expected: dict | None) -> str:
            if not expected:
                return ""
            if expected.get("answers"):
                return " / ".join(str(a) for a in expected["answers"][:4])
            if expected.get("gold") is not None:
                gold = str(expected["gold"])
                if expected.get("text"):
                    gold = f"{gold} — {expected['text']}"
                return gold
            if expected.get("answer") is not None:
                return str(expected["answer"])
            if expected.get("text"):
                return str(expected["text"])
            if expected.get("bbox"):
                return f"box {expected['bbox']}"
            return ""

        ss_n = max(s.get("screenspot_n") or 0 for s in summaries.values())
        omit_ss_pass = ss_n > 48
        items = []
        omitted = 0
        for m in models:
            for c in runs[m]["cases"]:
                if omit_ss_pass and c.get("category") == "screenspot" and c["pass"]:
                    omitted += 1
                    continue
                kind = "pass" if c["pass"] else "fail"
                out = _esc((c.get("output") or "")[:800])
                err = c.get("error")
                extra = f"<div class='out'>{_esc(err)}</div>" if err else ""
                # expected only on failures, inline after the output
                expected = "" if c["pass"] else _expected_str(c.get("expected"))
                exp_html = (
                    f"<span class='exp'> · expected: {_esc(expected[:300])}</span>"
                    if expected
                    else ""
                )
                items.append(
                    f"""<article class="case {kind}">
  <div class="row1">
    <span class="pill">{kind}</span>
    <span class="id">{_esc(c["id"])}</span>
    <span class="cat">{_esc(m)} · {_esc(c["category"])}</span>
  </div>
  <div class="out">{out}{exp_html}</div>{extra}
</article>"""
                )
        omit_note = (
            f'<p class="note">{omitted} ScreenSpot passes omitted from this log.</p>'
            if omitted
            else ""
        )
        n_cases = sum(len(runs[m]["cases"]) for m in models)
        cases_html = f"""
<details class="cases">
  <summary>Cases <span class="count">{n_cases:,} items · collapsed by default — click to expand</span></summary>
  {omit_note}
  <div class="filters">
  <input id="f-all" type="radio" name="flt" checked onchange="document.body.dataset.filter='all'">
  <label for="f-all">All</label>
  <input id="f-fail" type="radio" name="flt" onchange="document.body.dataset.filter='fail'">
  <label for="f-fail">Fails</label>
  <input id="f-pass" type="radio" name="flt" onchange="document.body.dataset.filter='pass'">
  <label for="f-pass">Passes</label>
  </div>
  <div class="log">{''.join(items)}</div>
</details>"""

    cat_table = ""
    if cats:
        heads = "".join(f"<th>{_esc(m)}</th>" for m in models)
        cat_table = f"<h2>By category</h2><table><thead><tr><th>Category</th>{heads}</tr></thead><tbody>{''.join(cat_rows)}</tbody></table>"

    tq_html = ""
    tq_models = [m for m in models if runs[m].get("elapsed_s") and summaries[m]["n"]]
    if tq_models:
        ranked = sorted(tq_models, key=lambda m: runs[m]["elapsed_s"])
        max_el = max(runs[m]["elapsed_s"] for m in tq_models)
        rows = []
        for rank, m in enumerate(ranked, 1):
            el = float(runs[m]["elapsed_s"])
            n = summaries[m]["n"]
            width = 100.0 * el / max_el if max_el else 0.0
            overall = summaries[m]["overall"]
            macro = summaries[m].get("screenspot_macro")
            rows.append(
                f"""<div class="tq-row">
  <div class="tq-rank">{rank}</div>
  <div class="tq-name">{_esc(m)} <span class="cat">{_esc(runs[m].get("backend") or "")}</span></div>
  <div class="bar hide-sm" title="{_fmt_dur(el)}"><i style="--p:{width:.0f}%"></i></div>
  <div class="tq-time">{_fmt_dur(el)}<span class="u">{n / el:.2f} it/s</span></div>
  <div class="heat hide-sm" style="--p:{_pct_num(overall):.0f}">{_pct(overall)}</div>
  <div class="heat hide-sm" style="--p:{_pct_num(macro):.0f}">{_pct(macro)}</div>
</div>"""
            )
        tq_html = (
            "<h2>Time &amp; quality</h2>"
            '<p class="note">Wall-clock per model run — weights load + all cases — fastest first. '
            "Right columns: overall pass rate, ScreenSpot-v2 unweighted avg.</p>"
            '<div class="tq">'
            '<div class="tq-head"><div>#</div><div>Model</div><div class="hide-sm">Relative</div>'
            '<div>Duration</div><div class="hide-sm">Overall</div><div class="hide-sm">ScreenSpot</div></div>'
            f"{''.join(rows)}</div>"
        )

    plat_table = ""
    if plats:
        ss_n = max(s.get("screenspot_n") or 0 for s in summaries.values())
        heads = "".join(f"<th>{_esc(m)}</th>" for m in models)
        backends = _backends_used(models, runs)
        protocol = str(payload.get("protocol") or "bbox")
        criterion = (
            "Click-in-box (predicted box center inside gold box)."
            if protocol == "bbox"
            else "Click-point-in-box (x=…, y=… inside gold box), as in Liquid’s harness."
        )
        prep_bits = []
        if "gguf" in backends:
            prep_bits.append(
                "llama.cpp (GGUF) tiles large screenshots (up to 10×512² + overview) like the "
                "official pipeline behind 80.7 — the accuracy reference."
            )
        if "mlx" in backends:
            try:
                mlx_ver = str(payload.get("mlx_vlm_version") or "0")
                mlx_tiled = (
                    tuple(int(p) for p in mlx_ver.split(".")[:3] if p.isdigit()) >= (0, 6, 14)
                )
            except ValueError:
                mlx_tiled = False
            if mlx_tiled:
                prep_bits.append(
                    f"the MLX path (mlx-vlm {mlx_ver}) implements the same image splitting — "
                    "MLX and GGUF numbers are directly comparable."
                )
            else:
                prep_bits.append(
                    "the MLX path disables image splitting (single ≤512²-equivalent view, ~256 tokens), "
                    "so MLX ScreenSpot numbers are a lower bound."
                )
        prep = " Image preprocessing: " + " ".join(prep_bits) if prep_bits else ""
        note = (
            f"{ss_n} items. {criterion} Liquid column is published vLLM 0.26, not this local run. "
            f"Unweighted avg is how they report 80.7.{prep}"
        )
        plat_table = (
            f"<h2>ScreenSpot-v2</h2><p class='note'>{_esc(note)}</p>"
            f"<table><thead><tr><th>Platform</th>{heads}<th>Liquid vLLM</th></tr></thead>"
            f"<tbody>{''.join(plat_rows)}</tbody></table>"
        )
        if type_rows:
            theads = "".join(f"<th>{_esc(m)}</th>" for m in models)
            plat_table += (
                "<h2>ScreenSpot-v2 by element</h2>"
                f"<table><thead><tr><th>Type</th>{theads}</tr></thead>"
                f"<tbody>{''.join(type_rows)}</tbody></table>"
            )

    refcoco_table = ""
    if subsets_present:
        rc_n = max(s.get("refcoco_n") or 0 for s in summaries.values())
        heads = "".join(f"<th>{_esc(m)}</th>" for m in models)
        note = (
            f"{rc_n} items — RefCOCO / RefCOCO+ / RefCOCOg referring expressions on COCO "
            "photos, the 8 eval splits behind Liquid’s published RefCOCO-avg. Two hit rules "
            "reported (precision@1 rule unspecified); whichever avg lands nearer 87.9 matches "
            "their scorer. Liquid column is published vLLM 0.26, not this local run."
        )
        refcoco_table = (
            "<h2>RefCOCO grounding</h2>"
            f"<p class='note'>{_esc(note)}</p>"
            f"<table><thead><tr><th>Split</th>{heads}<th>Liquid vLLM</th></tr></thead>"
            f"<tbody>{''.join(refcoco_rows)}</tbody></table>"
        )

    anls_tables = []
    for cat, (title, ref, full_n) in ANLS_TRACKS.items():
        if not any(summaries[m]["anls_tracks"].get(cat) for m in models):
            continue
        track_n = max(
            (s["anls_tracks"][cat]["n"] for s in summaries.values() if s["anls_tracks"].get(cat)),
            default=0,
        )
        rows = []
        for label, key in (("ANLS", "anls"), ("ANLS pass rate", "acc")):
            cells = []
            for m in models:
                val = (summaries[m]["anls_tracks"].get(cat) or {}).get(key)
                p = _pct_num(val)
                cells.append(
                    f'<td><span class="heat" style="--p:{p:.0f}">{_pct(val)}</span></td>'
                )
            ref_cell = f'<td class="ref">{_pct(ref)}</td>' if key == "anls" else "<td></td>"
            rows.append(f"<tr><th>{_esc(label)}</th>{''.join(cells)}{ref_cell}</tr>")
        qtypes = sorted(
            {
                q
                for m in models
                for q in (summaries[m]["anls_tracks"].get(cat) or {}).get("by_type", {})
            }
        )
        for q in qtypes:
            cells = []
            for m in models:
                val = (summaries[m]["anls_tracks"].get(cat) or {}).get("by_type", {}).get(q)
                p = _pct_num(val)
                cells.append(
                    f'<td><span class="heat" style="--p:{p:.0f}">{_pct(val)}</span></td>'
                )
            rows.append(f"<tr><th>{_esc(q)}</th>{''.join(cells)}<td></td></tr>")
        heads = "".join(f"<th>{_esc(m)}</th>" for m in models)
        kind = "Document" if cat == "docvqa" else "Infographic"
        note = (
            f"{track_n} items — {kind.lower()} reading comprehension on the official validation "
            "split, free-form short answers scored by ANLS (threshold 0.5; errors count 0). "
            "Liquid column is published vLLM 0.26 ANLS, not this local run."
        )
        anls_tables.append(
            f"<h2>{_esc(title)}</h2>"
            f"<p class='note'>{_esc(note)}</p>"
            f"<table><thead><tr><th>Metric</th>{heads}<th>Liquid vLLM</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    blink_table = ""
    if any(summaries[m].get("blink_n") for m in models):
        bl_n = max(s.get("blink_n") or 0 for s in summaries.values())
        heads = "".join(f"<th>{_esc(m)}</th>" for m in models)
        cells = []
        for m in models:
            val = summaries[m].get("blink_acc")
            p = _pct_num(val)
            cells.append(
                f'<td><span class="heat" style="--p:{p:.0f}">{_pct(val)}</span></td>'
            )
        row = (
            f"<tr><th>Overall accuracy</th>{''.join(cells)}"
            f'<td class="ref">{_pct(LIQUID_BLINK)}</td></tr>'
        )
        btasks = sorted({t for s in summaries.values() for t in s.get("blink_by_task") or {}})
        for t in btasks:
            cells = []
            for m in models:
                val = summaries[m]["blink_by_task"].get(t)
                p = _pct_num(val)
                cells.append(
                    f'<td><span class="heat" style="--p:{p:.0f}">{_pct(val)}</span></td>'
                )
            row += f"<tr><th>{_esc(t)}</th>{''.join(cells)}<td></td></tr>"
        note = (
            f"{bl_n} items — multi-image perceptual tasks (relative depth, correspondence, "
            "jigsaw, …) with the benchmark's canonical lettered-choice prompts. Liquid column "
            "is published vLLM 0.26 overall accuracy, not this local run."
        )
        blink_table = (
            "<h2>BLINK</h2>"
            f"<p class='note'>{_esc(note)}</p>"
            f"<table><thead><tr><th>Metric</th>{heads}<th>Liquid vLLM</th></tr></thead>"
            f"<tbody>{row}</tbody></table>"
        )

    from bench.report import EXAM_TRACKS

    exam_tables = []
    for cat, (title, ref) in EXAM_TRACKS.items():
        if not any(summaries[m]["exam_tracks"].get(cat) for m in models):
            continue
        heads = "".join(f"<th>{_esc(m)}</th>" for m in models)
        ev_n = max(
            (s["exam_tracks"][cat]["n"] for s in summaries.values() if s["exam_tracks"].get(cat)),
            default=0,
        )
        cells = []
        for m in models:
            val = (summaries[m]["exam_tracks"].get(cat) or {}).get("acc")
            p = _pct_num(val)
            cells.append(
                f'<td><span class="heat" style="--p:{p:.0f}">{_pct(val)}</span></td>'
            )
        row = (
            f"<tr><th>Accuracy</th>{''.join(cells)}<td class='ref'>{_pct(ref)}</td></tr>"
        )
        for label, key in (("question type", "by_type"), ("subject", "by_subject")):
            groups = sorted(
                {
                    g
                    for m in models
                    for g in (summaries[m]["exam_tracks"].get(cat) or {}).get(key, {})
                }
            )
            for g in groups:
                cells = []
                for m in models:
                    val = (summaries[m]["exam_tracks"].get(cat) or {}).get(key, {}).get(g)
                    p = _pct_num(val)
                    cells.append(
                        f'<td><span class="heat" style="--p:{p:.0f}">{_pct(val)}</span></td>'
                    )
                row += f"<tr><th>{_esc(g)} <span class='dim'>· {label}</span></th>{''.join(cells)}<td></td></tr>"
        note = (
            f"{ev_n} items — multiple-choice / short-answer accuracy with direct answers "
            "(option letter or single number/word). Liquid column is their published vLLM "
            "number, which uses a CoT-style eval pipeline — treat this local direct-answer "
            "accuracy as a lower bound vs their setup."
        )
        exam_tables.append(
            f"<h2>{_esc(title)}</h2>"
            f"<p class='note'>{_esc(note)}</p>"
            f"<table><thead><tr><th>Metric</th>{heads}<th>Liquid vLLM</th></tr></thead>"
            f"<tbody>{row}</tbody></table>"
        )

    sampling = payload.get("sampling") or {}
    backends = _backends_used(models, runs)
    runtime = " + ".join(_BACKEND_LABEL.get(b, b.upper()) for b in backends) or "local"
    engines = _join([_BACKEND_ENGINE.get(b, b) for b in backends])
    ss_n = max((s.get("screenspot_n") or 0) for s in summaries.values()) if summaries else 0
    rc_n = max((s.get("refcoco_n") or 0) for s in summaries.values()) if summaries else 0
    if ss_n >= 1000:
        lede = (
            "The full 1,272-item ScreenSpot-v2 test set. "
            f"Same items as Liquid’s 80.7; {engines}, not vLLM."
        )
    elif ss_n:
        lede = (
            f"A {ss_n}-item ScreenSpot-v2 subset. "
            "Not Liquid’s full 80.7 number."
        )
    else:
        lede = f"Local {runtime} capability and speed measurements."
    if rc_n:
        full_rc = rc_n >= 25000
        lede += (
            " Plus RefCOCO grounding: "
            + ("all 8 eval splits" if full_rc else f"a {rc_n}-item seeded subset")
            + " vs Liquid’s published 87.9."
        )
    anls_present = [t for t in ANLS_TRACKS if any(s["anls_tracks"].get(t) for s in summaries.values())]
    if anls_present:
        refs = " and ".join(f"{ANLS_TRACKS[t][0]} ({_pct(ANLS_TRACKS[t][1])})" for t in anls_present)
        lede += f" Plus {refs} vs the published ANLS."
    bl_n = max((s.get("blink_n") or 0) for s in summaries.values()) if summaries else 0
    if bl_n:
        lede += f" Plus BLINK multi-image ({bl_n} items) vs Liquid’s published 61.5."
    versions = []
    if "mlx" in backends and payload.get("mlx_vlm_version"):
        versions.append(f'mlx-vlm {_esc(payload["mlx_vlm_version"])}')
    if "gguf" in backends and payload.get("llama_cpp"):
        versions.append(f'llama.cpp {_esc(payload["llama_cpp"])}')
    versions.append(f'py {_esc(payload.get("python"))}')
    version_html = "".join(f"<div>{v}</div>" for v in versions)
    protocol = str(payload.get("protocol") or "bbox")
    protocol_html = (
        f"<div>protocol {_esc(protocol)}</div>" if protocol != "bbox" else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LFM2.5-VL bench · {_esc(payload.get("created"))}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,360;9..144,420&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>{_CSS}</style>
</head>
<body data-filter="all">
  <main class="wrap">
    <header class="mast">
      <div>
        <div class="kicker">Local {_esc(runtime)} · vision-language</div>
        <h1>Capability<br>plate.</h1>
        <p class="lede">{_esc(lede)}</p>
      </div>
      <div class="meta">
        <div><b>{_esc(payload.get("chip") or "unknown chip")}</b></div>
        {version_html}
        <div>{_esc(payload.get("created"))} · temp { _esc(sampling.get("temperature", 0)) }</div>
        {protocol_html}
      </div>
    </header>
    <section class="plate" style="--cols:{len(columns)}">
      <div class="plate-head"><div></div>{head}</div>
      {''.join(row_html)}
    </section>
    {tq_html}
    {cat_table}
    {plat_table}
    {refcoco_table}
    {''.join(anls_tables)}
    {blink_table}
    {''.join(exam_tables)}
    {speed_html}
    {cases_html}
  </main>
</body>
</html>
"""
