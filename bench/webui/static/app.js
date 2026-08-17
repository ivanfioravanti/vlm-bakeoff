/* vlm-bakeoff bench UI — vanilla JS, no build step. */

const $ = (id) => document.getElementById(id);
let META = null;
let selected = { models: new Set(["4bit", "8bit", "bf16"]), tracks: new Set() };
let lastLogSig = "";

async function api(path, opts) {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || res.statusText);
  return body;
}

/* ---------------------------------------------------------------- meta */

async function loadMeta() {
  META = await api("/api/meta");
  const mk = (chip, cls) => {
    chip.className = "chip" + (selected.models.has(chip.dataset.model) ? " on" : "");
  };
  for (const [host, group] of [["mlx-chips", "mlx"], ["gguf-chips", "gguf"]]) {
    const box = $(host);
    box.innerHTML = "";
    for (const m of META.models.filter((x) => x.backend === group)) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "chip";
      b.dataset.model = m.alias;
      b.title = m.model_id;
      b.innerHTML = `${m.alias}<small>${m.model_id.split("/").pop().replace("LFM2.5-VL-3B-", "")}</small>`;
      b.onclick = () => {
        selected.models.has(m.alias) ? selected.models.delete(m.alias) : selected.models.add(m.alias);
        mk(b);
        updateTotals();
      };
      box.appendChild(b);
      mk(b);
    }
  }
  const tracks = $("tracks");
  tracks.innerHTML = "";
  for (const [name, n] of Object.entries(META.tracks)) {
    const label = document.createElement("label");
    label.className = "track";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = true;
    cb.onchange = () => {
      cb.checked ? selected.tracks.add(name) : selected.tracks.delete(name);
      updateTotals();
    };
    selected.tracks.add(name);
    label.append(cb, Object.assign(document.createElement("span"), { className: "name", textContent: name }),
      Object.assign(document.createElement("span"), { className: "n", textContent: `${n} items` }));
    tracks.appendChild(label);
  }
  const caps = $("per-track-limits");
  caps.innerHTML = "";
  for (const name of Object.keys(META.tracks)) {
    const f = document.createElement("div");
    f.className = "field";
    f.innerHTML = `<label>${name}</label><input type="number" min="1" placeholder="global" id="cap-${name}">`;
    f.querySelector("input").oninput = updateTotals;
    caps.appendChild(f);
  }
  $("opt-trackcap").oninput = updateTotals;
  const d = META.defaults;
  $("opt-temp").value = d.temp;
  $("opt-topk").value = d.top_k;
  $("opt-batch").value = d.batch_size;
  $("opt-trackcap").value = 1000; // sensible quick-run default; clear for full suites
  $("opt-protocol").innerHTML = d.protocols.map((p) => `<option ${p === d.protocol ? "selected" : ""}>${p}</option>`).join("");
  $("all-tracks").onclick = (e) => { e.preventDefault(); setAllTracks(true); };
  $("none-tracks").onclick = (e) => { e.preventDefault(); setAllTracks(false); };
  updateTotals();
}

function setAllTracks(on) {
  selected.tracks = new Set(on && META ? Object.keys(META.tracks) : []);
  document.querySelectorAll(".track input").forEach((cb) => (cb.checked = on));
  updateTotals();
}

function currentCaps() {
  const g = Number($("opt-trackcap").value) || null;
  const per = {};
  for (const name of Object.keys(META?.tracks || {})) {
    const v = Number((document.getElementById(`cap-${name}`) || {}).value) || null;
    if (v) per[name] = v;
  }
  return { global: g, per_track: per };
}

function updateTotals() {
  const caps = currentCaps();
  let total = 0;
  let chosen = 0;
  for (const [t, n] of Object.entries(META?.tracks || {})) {
    total += n;
    if (selected.tracks.has(t)) chosen += Math.min(n, caps.per_track[t] ?? caps.global ?? n);
  }
  $("track-total").textContent = `${chosen.toLocaleString()} / ${total.toLocaleString()} items`;
}

/* --------------------------------------------------------------- start */

async function startRun() {
  const models = [...selected.models];
  const custom = $("custom-model").value.trim();
  if (custom) models.push(custom);
  if (!models.length) return note("select at least one model");
  const categories = [...selected.tracks];
  if (!categories.length) return note("select at least one track");
  const req = { models, categories, protocol: $("opt-protocol").value };
  const caps = currentCaps();
  if (caps.global || Object.keys(caps.per_track).length) req.limits = caps;
  for (const [key, id] of [["batch_size", "opt-batch"], ["temp", "opt-temp"], ["top_k", "opt-topk"], ["limit", "opt-limit"]]) {
    const v = $(id).value.trim();
    if (v !== "") req[key] = Number(v);
  }
  const resume = $("opt-resume").value;
  if (resume) req.resume = resume;
  try {
    const out = await api("/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    note(`started → results/${out.run_dir.split("/").pop()}`);
    refreshStatusLoop(true);
    refreshRuns();
  } catch (e) {
    note(e.message);
  }
}

function note(text) { $("launch-note").textContent = text; }

/* --------------------------------------------------------------- status */

async function refreshStatus(fast) {
  let st;
  try { st = await api("/api/status"); } catch { return; }

  const active = st.active;
  $("btn-start").disabled = active;
  $("btn-stop").disabled = !active;
  const live = $("live");
  if (active || (st.models_progress || []).length) {
    live.style.display = "";
    $("live-dir").textContent = st.run_dir ? st.run_dir.split("/").slice(-2).join("/") : "";
    renderProgress(st);
    renderLog(st.log_tail || []);
  } else {
    live.style.display = "none";
  }
  if (!fast) refreshRuns();
}

function renderProgress(st) {
  const total = st.target_total || 0;
  const box = $("progress");
  box.innerHTML = "";
  for (const m of st.models_progress || []) {
    const capped = total ? Math.min(m.rows, total) : m.rows;
    const pct = total ? Math.round((capped / total) * 100) : 0;
    const div = document.createElement("div");
    div.className = "modelbar";
    div.innerHTML = `
      <div class="top">
        <span class="name">${m.model}</span>
        <span class="stat">${m.rows.toLocaleString()}${total ? ` / ${total.toLocaleString()}` : ""} done · ${m.pass.toLocaleString()} pass
          ${m.avg_gen_s ? `· ${(1 / m.avg_gen_s).toFixed(1)} it/s` : ""}${m.current ? ` · ${m.current}` : ""}</span>
      </div>
      <div class="bar ${m.rows ? "" : "idle"}"><div style="width:${pct}%"></div></div>`;
    box.appendChild(div);
  }
}

function renderLog(lines) {
  const el = $("log");
  // the server always sends the last N lines, so the count alone can't detect
  // change once the log is long enough — key on length + last line instead
  const sig = `${lines.length}|${lines[lines.length - 1] || ""}`;
  if (sig === lastLogSig) return;
  lastLogSig = sig;
  const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
  el.innerHTML = lines.map((l) => {
    let cls = "sys";
    if (l.includes(" PASS ")) cls = "pass";
    else if (l.includes(" FAIL ")) cls = "fail";
    return `<div class="${cls}">${l.replace(/&/g, "&amp;").replace(/</g, "&lt;")}</div>`;
  }).join("");
  if (nearBottom) el.scrollTop = el.scrollHeight;
}

let statusTimer = null;
function refreshStatusLoop(immediate) {
  clearTimeout(statusTimer);
  const tick = async () => {
    await refreshStatus(false);
    let st = { active: false };
    try { st = await api("/api/status"); } catch {}
    statusTimer = setTimeout(tick, st.active ? 2000 : 10000);
  };
  if (immediate) tick();
  else statusTimer = setTimeout(tick, 2000);
}

/* ----------------------------------------------------------------- runs */

async function refreshRuns() {
  let data;
  try { data = await api("/api/runs"); } catch { return; }
  const body = $("runs-body");
  body.innerHTML = "";
  const resumeSel = $("opt-resume");
  const currentResume = resumeSel.value;
  const options = ['<option value="">— fresh —</option>'];
  for (const r of data.runs.slice(0, 30)) {
    const done = (r.progress || []).reduce((a, p) => a + p.rows, 0);
    const isDone = !!r.report;
    const overall = r.overall
      ? Object.entries(r.overall).map(([m, v]) => `${m} ${(100 * v).toFixed(1)}%`).join(" · ")
      : "";
    const tr = document.createElement("tr");
    tr.className = r.report ? "runclick" : "runclick noreport";
    const date = new Date(r.created * 1000);
    tr.innerHTML = `
      <td>${r.dir}</td>
      <td>${(r.models || []).join(", ") || "—"}</td>
      <td>${done.toLocaleString()}</td>
      <td>${r.report ? '<span class="badge done">complete</span>' : '<span class="badge live">partial</span>'}</td>
      <td class="score">${overall}</td>`;
    tr.onclick = () => openReport(r);
    body.appendChild(tr);
    if (!isDone && (r.progress || []).length) {
      const val = `results/${r.dir}`;
      options.push(`<option value="${val}">${r.dir} (${done.toLocaleString()} done)</option>`);
    }
  }
  resumeSel.innerHTML = options.join("");
  if (currentResume) resumeSel.value = currentResume;
}

function openReport(r) {
  if (!r.report) { note(`run ${r.dir} has no report yet (still partial — resume it)`); return; }
  $("dlg-title").textContent = `results/${r.dir}/REPORT.html`;
  $("report-frame").src = `/api/report/${r.dir}`;
  $("report-dlg").showModal();
}

/* ----------------------------------------------------------------- boot */

$("btn-start").onclick = startRun;
$("btn-stop").onclick = async () => {
  try {
    const out = await api("/api/stop", { method: "POST" });
    note(out.note || "stopped");
  } catch (e) { note(e.message); }
};

loadMeta().catch((e) => note(`meta failed: ${e.message}`));
refreshStatusLoop(true); // status renders the live card immediately, independent of meta
