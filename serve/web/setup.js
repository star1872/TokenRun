// serve/web/setup.js - the model manager (serve/setup_web.py): this PC, the download source, every model and size with
// whether it runs here, setup.py's install and the model's start, and what quantization a GGUF really is.
"use strict";

const $ = (id) => document.getElementById(id);
const SPRITE = "web/sprite.svg";
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? "–" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}));
const store = {
  get(k, d) { try { const v = localStorage.getItem("strata." + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem("strata." + k, JSON.stringify(v)); } catch (e) { /* this page only */ } },
};
applyI18n();

function toast(kind, title, text = "", ms = 4000) {
  const el = document.createElement("div");
  el.className = `st-toast st-toast--${kind}`;
  el.innerHTML = `<svg class="st-icon"><use href="${SPRITE}#i-${{success: "check", warn: "warning", error: "error"}[kind] || "info"}"/></svg>` +
    `<div><div class="st-toast__title"></div><div class="t-text"></div></div>`;
  el.querySelector(".st-toast__title").textContent = title;
  el.querySelector(".t-text").textContent = text;
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), ms);
}
// the API: serve/setup_web.py's, at manage/api/ both on its own (START-HERE.bat --web) and inside a model's server
async function api(path, body) {
  const r = await fetch("manage/" + path, body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw Object.assign(new Error(j.error || `HTTP ${r.status}`), {status: r.status, body: j});
  return j;
}

// ------------------------------------------------------------------ theme and language
function setTheme(theme, save) {
  document.documentElement.dataset.theme = theme;
  if (save) try { localStorage.setItem("strata.theme", theme); } catch (e) { /* ignore */ }
  $("theme-icon").setAttribute("href", `${SPRITE}#i-${theme === "dark" ? "sun" : "moon"}`);
}
$("theme-btn").onclick = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true);
setTheme(document.documentElement.dataset.theme || "light", false);
$("lang-text").textContent = LANG === "zh-CN" ? "EN" : "中";
$("lang-btn").onclick = () => setLang(LANG === "zh-CN" ? "en" : "zh-CN");

// ------------------------------------------------------------------ this PC
let hw = null, whatIf = null;
async function loadHardware(refresh = false) {
  hw = await api("api/hardware" + (refresh ? "?refresh=1" : ""));
  const gpus = hw.gpus.length ? hw.gpus.map((g) => `${g.name}, ${fmt(g.vram_gb, 0)} GB` +
    (g.problem ? ` – ${t("cannot be used")}: ${g.problem}` : "")).join("\n") : t("none found (Strata needs an NVIDIA RTX 20 series or newer, or a supported AMD card)");
  const dl = $("hw");
  dl.innerHTML = [[t("GPU"), gpus], [t("RAM"), `${fmt(hw.ram_gb, 0)} GB`],
    [t("CPU"), `${hw.cpu}${hw.avx512 ? " (AVX-512)" : hw.avx2 ? " (AVX2)" : ""}`],
    [t("Model folder"), `${hw.models_dir} · ${t("{n} GB free", {n: fmt(hw.free_gb, 0)})}`], [t("System"), hw.os]]
    .map(([k, v]) => `<dt>${esc(k)}</dt><dd class="pre">${esc(v)}</dd>`).join("");
  $("dl-dir").placeholder = hw.data_dir;
  if (!$("dl-dir").dataset.loaded) { $("dl-dir").dataset.loaded = "1"; $("dl-dir").value = store.get("dataDir", ""); checkFolder(false); }
  $("wi-ram").placeholder = fmt(hw.ram_gb, 0);
  const usable = hw.gpus.filter((g) => !g.problem);
  $("wi-vram").placeholder = usable.length ? fmt(Math.max(...usable.map((g) => g.vram_gb)), 0) : "0";
}
$("hw-refresh").onclick = async () => { await loadHardware(true); await loadModels(); toast("success", t("Checked again")); };
$("wi-apply").onclick = () => {
  const ram = +$("wi-ram").value, vram = +$("wi-vram").value;
  if (!(ram > 0) || !(vram >= 0) || $("wi-vram").value === "") { toast("warn", t("Enter both numbers")); return; }
  whatIf = {ram, vram};
  loadModels();
};
$("wi-reset").onclick = () => { whatIf = null; $("wi-ram").value = ""; $("wi-vram").value = ""; loadModels(); };

// ------------------------------------------------------------------ download source
const pickSource = () => (document.querySelector("input[name=src]:checked") || {}).value || "auto";
// the download folder: empty = the default (setup's data folder); kept in this browser
const dataDir = () => $("dl-dir").value.trim();
const dirQuery = () => (dataDir() ? `dir=${encodeURIComponent(dataDir())}` : "");
async function checkFolder(reload) {
  const info = $("dl-info");
  try {
    const r = await api(`api/folder${dataDir() ? "?path=" + encodeURIComponent(dataDir()) : ""}`);
    info.dataset.state = r.writable ? "ok" : "err";
    info.textContent = r.writable
      ? t("{path}: {n} GB free", {path: r.path, n: fmt(r.free_gb, 0)}) + (r.exists ? "" : " · " + t("it will be created"))
      : t("{path}: no permission to write there", {path: r.path});
    store.set("dataDir", dataDir());
  } catch (e) {
    info.dataset.state = "err";
    info.textContent = e.message;
  }
  if (reload) { loadModels(); if (llRepo) listVariants(llRepo); }
}
$("dl-check").onclick = () => checkFolder(true);
$("dl-dir").addEventListener("change", () => checkFolder(true));
for (const r of document.querySelectorAll("input[name=src]")) {
  r.checked = r.value === store.get("source", "auto");
  r.onchange = () => store.set("source", pickSource());
}
$("src-test").onclick = async () => {
  for (const k of ["auto", "modelscope", "huggingface"]) { $(`st-${k}`).className = "st-badge"; $(`st-${k}`).textContent = t("Testing…"); }
  try {
    const s = await api("api/sources");
    for (const k of ["modelscope", "huggingface"]) {
      $(`st-${k}`).className = `st-badge ${s[k] ? "st-badge--generating" : "st-badge--error"}`;
      $(`st-${k}`).textContent = t(s[k] ? "Reachable" : "Not reachable");
    }
    $("st-auto").className = `st-badge ${s.auto ? "st-badge--generating" : "st-badge--error"}`;
    $("st-auto").textContent = s.auto ? t("Will use {name}", {name: s.auto === "modelscope" ? "ModelScope" : "Hugging Face"}) : t("Not reachable");
  } catch (e) {
    toast("error", t("The test failed"), e.message);
  }
};

// ------------------------------------------------------------------ models
const FAMILY_ABOUT = {
  qwen: L("The original model, quantized by ISTA-DASLab (GSQ-RCO). The general choice.",
          "原版模型，由 ISTA-DASLab 量化（GSQ-RCO）。通用，推荐。"),
  swift: L("UkisAI's fine-tune: thinks much shorter (-63% thinking tokens, answers 1.8x sooner by its authors' numbers). Its own license.",
           "UkisAI 的微调版：思考时间短得多（作者数据：思考 token 少 63%，出答案快 1.8 倍）。有自己的许可证。"),
  coder: L("ISTA-DASLab's coding version: half the experts (code, tools, images kept). Fits 32 GB of RAM and runs faster; weaker outside code.",
           "ISTA-DASLab 的编程版：只保留一半专家（代码、工具调用、看图用的那些）。32 GB 内存就能跑，速度更快；编程以外的能力较弱。"),
  unsloth: L("Unsloth's ~4-bit files: closer to the full model, bigger. With less than ~80 GB of RAM part of the experts is read from the SSD.",
             "Unsloth 的约 4 位版本：更接近完整模型，文件更大。内存不到约 80 GB 时，部分专家要从硬盘读取。"),
};
const SIZE_ABOUT = {
  Q2_0: L("2-bit, the fastest", "2 位，最快"),
  IQ2_XS: L("2-bit i-quant: a little better quality, close in speed. Recommended when unsure", "2 位 i-quant：质量略好，速度接近。不确定就选它"),
  IQ3_XXS: L("3-bit i-quant: better quality, slower (more CPU work per token)", "3 位 i-quant：质量更好，较慢（每个 token 的 CPU 计算更多）"),
  IQ3_S: L("3.5-bit i-quant: the best of these (matches the full model on the published benchmarks), the slowest; 64 GB of RAM with little else open",
           "3.5 位 i-quant：这几种里质量最好（公开评测与完整模型相当），最慢；需要 64 GB 内存且少开其他程序"),
  IQ1_M: L("The Coder's only size. Named IQ1_M, but its experts are stored at about 3.3 bits", "Coder 唯一的大小。名字叫 IQ1_M，但专家实际约 3.3 位"),
  "UD-IQ4_XS": L("~4-bit i-quant (Unsloth Dynamic): between IQ3_S and UD-Q4_K_XL in quality", "约 4 位 i-quant（Unsloth Dynamic）：质量介于 IQ3_S 和 UD-Q4_K_XL 之间"),
  "UD-Q4_K_XL": L("4-bit (Unsloth Dynamic), experimental: the closest to the full model, but on a 64 GB PC most experts come from the SSD (7-8.5 tokens/s measured by Strata)",
                  "4 位（Unsloth Dynamic），实验性：最接近完整模型，但 64 GB 内存的电脑上大部分专家从硬盘读（Strata 官方实测 7～8.5 token/秒）"),
};
const LEVEL = {ok: ["st-badge--generating", "Runs well"], slow: ["st-badge--queued", "Runs, slower"], no: ["st-badge--error", "Will not run"]};
function reasonText(r) {
  const n = (x) => fmt(x, 0);
  switch (r.code) {
    case "fits": return t("Fits: {experts} GB of experts in {ram} GB of RAM.", {experts: n(r.experts), ram: n(r.ram)});
    case "no_gpu": return t("No graphics card Strata can use (NVIDIA RTX 20 series or newer, or a supported AMD card).");
    case "vram_tiny": return t("{vram} GB of VRAM: Strata needs at least {need} GB.", {vram: n(r.vram), need: r.need});
    case "vram_small": return t("{vram} GB of VRAM: it starts, but fewer experts fit on the card, so it is slower ({want} GB or more recommended).", {vram: n(r.vram), want: r.want});
    case "nvidia_only": return t("This size runs on NVIDIA cards only.");
    case "ram_budget": return t("{ram} GB of RAM: part of its {experts} GB of experts is read from the SSD while it answers (about {need} GB of RAM holds them all).", {ram: n(r.ram), experts: n(r.experts), need: r.need});
    case "low_ram": return t("{ram} GB of RAM is less than its {experts} GB of experts + 10 GB: the low-RAM mode reads experts from the SSD, much slower (about {need} GB of RAM runs it normally).", {ram: n(r.ram), experts: n(r.experts), need: r.need});
    case "low_ram_resident": return t("{ram} GB of RAM holds its {experts} GB of experts only together with the graphics card: setup uses the low-RAM mode and copies the experts the card does not hold into RAM once.", {ram: n(r.ram), experts: n(r.experts)});
    case "ram_short": return t("{ram} GB of RAM: it needs about {need} GB. The model will not start.", {ram: n(r.ram), need: r.need});
    case "disk": return t("Only {free} GB free in the model folder; the download needs about {need} GB (choose a bigger folder under Download folder above).", {free: n(r.free), need: r.need});
    case "experimental": return t("Experimental.");
    default: return r.code;
  }
}
const klass = (c) => (c ? t(c) : "");
let models = [], installedMap = {};
async function loadModels() {
  const q = whatIf ? `?ram=${whatIf.ram}&vram=${whatIf.vram}` : dataDir() ? "?" + dirQuery() : "";
  const r = await api("api/models" + q);
  models = r.models; installedMap = r.installed;
  $("models-note").textContent = whatIf ? t("For a PC with {ram} GB of RAM and {vram} GB of VRAM", {ram: whatIf.ram, vram: whatIf.vram}) : t("For this PC");
  const fams = [...new Set(models.map((m) => m.family))];
  $("models").innerHTML = fams.map((f) => {
    const rows = models.filter((m) => m.family === f);
    return `<div class="fam"><div class="fam__head"><b>${esc(rows[0].family_title)}</b><span class="muted small">${esc(FAMILY_ABOUT[f] || rows[0].by)}</span></div>` +
      rows.map((m) => {
        const [cls, label] = LEVEL[m.fit.level];
        const key = `${m.family}/${m.model}`;
        const reasons = m.fit.reasons.map((r) => `<li class="r-${r.level}">${esc(reasonText(r))}</li>`).join("");
        const action = m.installed
          ? `<span class="st-badge st-badge--reading">${esc(t("Installed"))}</span><button class="st-btn st-btn--primary" data-start="${esc(key)}">${esc(t("Start"))}</button>`
          : `<button class="st-btn st-btn--secondary" data-install="${esc(key)}">${esc(t("Download and install"))}</button>`;
        return `<div class="size"><div class="size__main"><div class="size__name"><b>${esc(m.model)}</b>` +
          (m.expert_bits ? `<span class="chip q">${esc(t("{b} bits", {b: fmt(m.expert_bits, 2)}))} · ${esc(klass(m.class))}</span>` : "") +
          (m.experimental ? `<span class="chip">${esc(t("Experimental"))}</span>` : "") + `</div>` +
          `<div class="muted small">${esc(SIZE_ABOUT[m.model] || m.about)}</div>` +
          `<div class="size__nums small">${esc(t("Download {d} GB · experts {e} GB in RAM", {d: fmt(m.download_gb, 1), e: fmt(m.experts_gb, 1)}))}</div>` +
          `<ul class="reasons small">${reasons}</ul></div>` +
          `<div class="size__side"><span class="st-badge ${cls}">${esc(t(label))}</span>${action}</div></div>`;
      }).join("") + `</div>`;
  }).join("");
}
$("models").addEventListener("click", async (e) => {
  const ins = e.target.closest("[data-install]"), st = e.target.closest("[data-start]");
  if (ins) {
    const m = models.find((x) => `${x.family}/${x.model}` === ins.dataset.install);
    if (whatIf) { toast("warn", t("Showing another PC"), t("Switch back to this PC before installing.")); return; }
    if (m.fit.level === "no" && !confirm(t("This PC does not meet the requirements for {model}: after the download the model will not start.\n\nDownload anyway?", {model: m.model}))) return;
    if (m.fit.level === "slow" && !confirm(t("{model} will run on this PC, but slower:\n\n{why}\n\nDownload it ({gb} GB)?", {model: m.model, gb: fmt(m.download_gb, 1), why: m.fit.reasons.filter((r) => r.level === "slow").map(reasonText).join("\n")}))) return;
    // setup stops before downloading when the models folder is short (it counts a download it resumes): ask first
    const disk = m.fit.reasons.find((r) => r.code === "disk");
    if (disk && !confirm(t("Not enough free disk space for {model}: about {need} GB is needed, the model folder has {free} GB. The install will most likely stop at its disk check.\n\nTry anyway?", {model: m.model, need: disk.need, free: fmt(disk.free, 0)}))) return;
    try {
      await api("api/install", {family: m.family, model: m.model, source: pickSource(), data_dir: dataDir() || null});
      pollJobs();
    } catch (err) {
      toast("error", t(err.status === 409 ? "Something is already running" : "Could not start the install"), err.status === 409 ? t("Wait for it to finish, or stop it.") : err.message);
    }
  }
  if (st) {
    startModel(st.dataset.start);
  }
});

// ------------------------------------------------------------------ start a model, then its chat
// The chat is the model's own page on its own port.  A tab opened by the Start click itself (a later window.open is
// stopped by popup blockers) waits for the model and then turns into the chat; the progress card's button is there
// as well, in case the browser blocked the tab or it was closed.
let chatTab = null, chatArmed = false;          // armed: the server took this Start (an older job is not it)
const chatUrl = (port) => `${location.protocol}//${location.hostname}:${port}/`;
function waitingTab() {
  let w = null;
  try { w = window.open("", "_blank"); } catch (e) { /* blocked */ }
  if (!w) return null;
  try {
    w.document.title = t("Loading the model…");
    w.document.body.innerHTML = `<p style="font:16px/1.6 system-ui,sans-serif;max-width:560px;margin:80px auto;padding:0 16px;color:#555">` +
      `${esc(t("Loading the model (1-3 minutes)…"))}<br>${esc(t("The chat opens in this tab by itself when the model is ready."))}</p>`;
  } catch (e) { /* the tab works without the note */ }
  return w;
}
async function startModel(key) {
  if (chatTab && !chatTab.closed) chatTab.close();
  chatArmed = false;
  chatTab = waitingTab();
  try {
    await api("api/start", {key});
    chatArmed = true;
    pollJobs();
    $("job-card").scrollIntoView({behavior: "smooth", block: "start"});
  } catch (err) {
    if (chatTab) { chatTab.close(); chatTab = null; }
    if (err.status === 409 && err.body.error === "running") toast("warn", t("A model is already running"), t("Close its window (or press Ctrl+C in it) first: the graphics card holds one model."), 7000);
    else toast("error", t(err.status === 409 ? "Something is already running" : "Could not start the model"), err.status === 409 ? t("Wait for it to finish, or stop it.") : err.message);
  }
}
function chatReady(port) {
  showChat(port);
  $("job-chat").href = chatUrl(port);
  $("job-chat").hidden = false;
  if (chatTab && chatArmed) {
    try { if (!chatTab.closed) chatTab.location.href = chatUrl(port); } catch (e) { /* closed meanwhile */ }
    chatTab = null;
  }
}
function chatFailed() {
  if (!chatTab || !chatArmed) return;
  try {
    if (!chatTab.closed) chatTab.document.body.innerHTML = `<p style="font:16px/1.6 system-ui,sans-serif;max-width:560px;margin:80px auto;padding:0 16px;color:#b42318">` +
      `${esc(t("The model did not start: see the log in the model manager."))}</p>`;
  } catch (e) { /* closed */ }
  chatTab = null;
}

// ------------------------------------------------------------------ install / start progress
let current = null, pollTimer = null, lastDone = null;
async function pollJobs() {
  clearTimeout(pollTimer);
  let jobs = {};
  try { jobs = await api("api/job"); } catch (e) { pollTimer = setTimeout(pollJobs, 3000); return; }
  const list = Object.entries(jobs).sort((a, b) => (b[1].running - a[1].running) || (a[1].seconds - b[1].seconds));
  const [name, j] = list[0] || [];
  $("job-card").hidden = !j;
  if (!j) return;
  current = name;
  const what = j.kind === "install" ? `${(models.find((m) => m.family === j.what.family) || {}).family_title || j.what.family} ${j.what.model}` : j.what.model_name;
  $("job-title").textContent = j.kind === "start" && j.running && j.ready ? t("{what} is running", {what})
                            : j.running ? t(j.kind === "install" ? "Installing {what}" : "Starting {what}", {what})
                                         : t(j.kind === "install" ? "Install: {what}" : "Start: {what}", {what});
  const mins = Math.floor(j.seconds / 60), secs = j.seconds % 60;
  let status;
  if (j.kind === "start" && j.running) status = j.ready ? t("Running: open the chat.") : t("Loading the model (1-3 minutes)…");
  else if (j.running) status = t("Running for {m} min {s} s. The download resumes where it stopped if it is interrupted.", {m: mins, s: secs});
  else status = j.code === 0 ? t(j.kind === "install" ? "Installed. Start it from the list above." : "The model stopped.") : t("Stopped with an error (code {c}): see the log below.", {c: j.code});
  $("job-status").textContent = status;
  $("job-status").dataset.state = j.running ? "run" : j.code === 0 ? "ok" : "err";
  $("job-progress").textContent = j.progress || "";
  $("job-log").textContent = j.lines.join("\n");
  $("job-log").scrollTop = $("job-log").scrollHeight;
  $("job-stop").hidden = !j.running;
  const ready = j.kind === "start" && j.running && j.ready;
  if (ready) chatReady(j.what.port);
  else $("job-chat").hidden = true;
  if (j.kind === "start" && !j.running) chatFailed();
  if (!j.running && lastDone !== name + j.code) { lastDone = name + j.code; loadModels(); loadLlama(); }
  if (j.running) { lastDone = null; pollTimer = setTimeout(pollJobs, 1500); }
}
$("job-stop").onclick = async () => {
  if (!confirm(t("Stop it? A download can be resumed later."))) return;
  await api("api/stop", {job: current}).catch(() => {});
  setTimeout(pollJobs, 800);
};

// ------------------------------------------------------------------ which quantization is this file
const GROUPS = {"routed experts": "Routed experts", "shared experts": "Shared experts", "router": "Router", "norms": "Norms",
  "hyper-connections": "Hyper-connections", "n-gram/per-layer embeddings": "N-gram / per-layer embeddings", "linear attn/SSM": "Linear attention / SSM",
  "attention (incl. GDN qkv/gate)": "Attention (incl. GDN qkv/gate)", "dense FFN": "Dense FFN", "token embedding": "Token embedding",
  "output head": "Output head", "MTP / nextn": "MTP / nextn", "other": "Other"};
const VERDICT = {known: ["st-badge--generating", "Strata runs it"], layout: ["st-badge--queued", "May run (not tested)"],
                 untested: ["st-badge--queued", "Not tested"], no: ["st-badge--error", "Strata cannot run it"]};
async function inspect(source, variant) {
  const box = $("inspect");
  box.innerHTML = `<p class="muted">${esc(t("Reading the headers…"))}</p>`;
  $("in-go").disabled = true;
  try {
    const r = await api("api/inspect", {source, variant});
    if (r.error) {
      box.innerHTML = `<p class="msg-error">${esc(t(r.error === "no_gguf" ? "No model .gguf files there." : "No variant matches."))}</p>`;
    } else if (r.variants) {
      box.innerHTML = `<p class="muted small">${esc(t("{n} variants: click one to inspect it.", {n: r.variants.length}))}</p>` +
        `<div class="variants">${r.variants.map((v) => {
          const [dir, file] = v.name.includes("/") ? v.name.split("/") : [v.name, ""];
          return `<button class="variant" data-variant="${esc(v.name)}"><b>${esc(dir)}</b>` +
            (file && file !== dir ? `<span class="small variant__file">${esc(file)}</span>` : "") +
            `<span class="muted small">${esc(t("{gb} GB · {n} file(s)", {gb: fmt(v.gb, 1), n: v.files}))}</span></button>`;
        }).join("")}</div>`;
      box.dataset.source = source;
    } else {
      const [cls, label] = VERDICT[r.verdict] || ["", r.verdict];
      const known = r.known ? t("This is {family} {model}, a file setup installs.", {family: (models.find((m) => m.family === r.known.family) || {}).family_title || r.known.family, model: r.known.model}) : "";
      box.innerHTML = `<div class="verdict"><div><b>${esc(r.name)}</b><div class="muted small">${esc(t("{gb} GB · {bpw} bits per weight overall", {gb: fmt(r.gb, 1), bpw: fmt(r.bpw, 2)}))}</div></div>` +
        `<div class="verdict__big">${r.experts_bpw ? `<span class="chip q">${esc(t("Experts {b} bits", {b: fmt(r.experts_bpw, 2)}))} · ${esc(klass(r.class))}</span>` : ""}` +
        `<span class="st-badge ${cls}">${esc(t(label))}</span></div></div>` +
        (known ? `<p>${esc(known)}</p>` : "") + `<p class="muted small">${esc(r.sentence)}</p>` +
        `<table class="opt-table"><thead><tr><th>${esc(t("Weight group"))}</th><th class="num">GB</th><th class="num">${esc(t("Bits"))}</th><th>${esc(t("Quantization types (share of bytes)"))}</th></tr></thead><tbody>` +
        r.groups.map((g) => `<tr${g.group === "routed experts" ? ' class="hl"' : ""}><td>${esc(t(GROUPS[g.group] || g.group))}</td><td class="num">${fmt(g.gb, 2)}</td>` +
          `<td class="num">${g.bpw == null ? "–" : fmt(g.bpw, 2)}</td><td><code>${esc(g.types.map(([ty, p]) => `${ty} ${p}%`).join(", "))}</code></td></tr>`).join("") +
        `</tbody></table>`;
    }
  } catch (e) {
    box.innerHTML = `<p class="msg-error">${esc(t("Could not read it: {err}", {err: e.message}))}</p>`;
  }
  $("in-go").disabled = false;
}
$("in-go").onclick = () => {
  const s = $("in-source").value.trim();
  if (!s) { $("in-source").focus(); return; }
  inspect(s, $("in-variant").value.trim());
};
$("in-source").addEventListener("keydown", (e) => { if (e.key === "Enter") $("in-go").click(); });
$("inspect").addEventListener("click", (e) => {
  const v = e.target.closest("[data-variant]");
  if (v) { $("in-variant").value = v.dataset.variant; inspect($("inspect").dataset.source, v.dataset.variant); }
});
for (const b of document.querySelectorAll("[data-ex]")) b.onclick = () => {
  const [s, v] = b.dataset.ex.split("|");
  $("in-source").value = s; $("in-variant").value = v || "";
  inspect(s, v || "");
};

// "Open the chat": a model running on this PC, started here or anywhere else (run-<model>, START-HERE.bat)
function showChat(port) {
  const chat = $("open-chat");
  chat.hidden = !port;
  if (port) chat.href = `${location.protocol}//${location.hostname}:${port}/`;   // this page's host: also from another PC
}
async function checkRunning() {
  try { const r = await api("api/running"); showChat(r.ports[0]); } catch (e) { /* the manager stopped */ }
  setTimeout(checkRunning, 5000);
}

// ------------------------------------------------------------------ other models: llama.cpp
const WHERE = {gpu: "Fits in VRAM: fast.", "gpu+ram": "Split between VRAM and RAM.", cpu: "No GPU: the CPU runs it from RAM.",
               ssd: "Does not fit in VRAM and RAM: it would be read from the disk, very slowly."};
let llRepo = "", llVariants = [];
// the Models card's two engines: Strata's own model (the default) and any GGUF model through llama.cpp
function showPane(name, save) {
  for (const b of document.querySelectorAll(".model-tabs .st-tab")) b.setAttribute("aria-selected", String(b.dataset.pane === name));
  $("pane-strata").hidden = name !== "strata";
  $("pane-llama").hidden = name !== "llama";
  if (save) store.set("modelTab", name);
}
for (const b of document.querySelectorAll(".model-tabs .st-tab")) b.onclick = () => showPane(b.dataset.pane, true);
showPane(store.get("modelTab", "strata") === "llama" ? "llama" : "strata");
async function loadLlama() {
  let r;
  try { r = await api("api/llama/models"); } catch (e) { $("mt-llama").hidden = true; showPane("strata"); return; }
  $("ll-build").textContent = t("llama.cpp {b}", {b: r.build});
  $("ll-suggest").innerHTML = r.suggested.map((x) => `<button type="button" class="link" data-repo="${esc(x.repo)}" title="${esc(x.about)}">${esc(x.title)}</button>`).join(" · ");
  const inst = Object.entries(r.installed);
  $("ll-installed").innerHTML = inst.length ? `<div class="fam"><div class="fam__head"><b>${esc(t("Installed"))}</b></div>` + inst.map(([k, c]) =>
    `<div class="size"><div class="size__main"><div class="size__name"><b>${esc(c.variant ? c.variant.split("/").pop() : k)}</b></div>` +
    `<div class="muted small">${esc(c.repo || "")} · ${esc(fmt(c.gb, 1))} GB</div></div>` +
    `<div class="size__side"><span class="st-badge st-badge--reading">${esc(t("Installed"))}</span><button class="st-btn st-btn--primary" data-start="llama/${esc(k)}">${esc(t("Start"))}</button></div></div>`).join("") + `</div>` : "";
}
// what the box takes: owner/name, ms:/hf: owner/name, a ModelScope or Hugging Face page URL, or a model name to search for
const REPO_NAME = /^[A-Za-z0-9][\w.-]*\/[A-Za-z0-9][\w.-]*$/;
function repoOf(text) {
  text = text.trim().replace(/^(ms|hf):/, "");
  const m = text.match(/^https?:\/\/[^/]+\/(?:models\/)?([^/?#]+\/[^/?#]+)/);
  return m ? m[1] : text;
}
async function searchRepos(query) {
  const box = $("ll-variants");
  box.innerHTML = `<p class="muted">${esc(t("Searching for {q}…", {q: query}))}</p>`;
  $("ll-go").disabled = true;
  try {
    const r = await api(`api/llama/search?q=${encodeURIComponent(query)}&source=${encodeURIComponent(pickSource())}`);
    if (!r.repos.length) { box.innerHTML = `<p class="msg-error">${esc(t("No GGUF repository found for {q}. Check the name, or type a repository as owner/name.", {q: query}))}</p>`; return; }
    box.innerHTML = `<div class="fam"><div class="fam__head"><b>${esc(t("GGUF repositories for {q}", {q: query}))}</b><span class="muted small">${esc(t(r.source === "modelscope" ? "from ModelScope" : "from Hugging Face"))}</span></div>` +
      `<p class="muted small">${esc(t("Choose one to list its versions. Common publishers first, then by downloads."))}</p>` +
      r.repos.map((x) => `<div class="size"><div class="size__main"><div class="size__name"><button type="button" class="link" data-repo="${esc(x.repo)}"><b>${esc(x.repo)}</b></button></div>` +
        `<div class="size__nums small">${esc(t("{n} downloads", {n: x.downloads.toLocaleString()}))}</div></div>` +
        `<div class="size__side">${x.known ? `<span class="st-badge st-badge--reading">${esc(t("Common publisher"))}</span>` : ""}` +
        `<button class="st-btn st-btn--secondary" data-repo="${esc(x.repo)}">${esc(t("List the versions"))}</button></div></div>`).join("") + `</div>`;
  } catch (err) {
    box.innerHTML = `<p class="msg-error">${esc(t("Could not search: {e}", {e: err.message}))}</p>`;
  } finally { $("ll-go").disabled = false; }
}
async function listVariants(repo) {
  llRepo = repoOf(repo);
  if (!llRepo) return;
  if (!REPO_NAME.test(llRepo)) { const q = llRepo; llRepo = ""; return searchRepos(q); }
  $("ll-repo").value = llRepo;
  const box = $("ll-variants");
  box.innerHTML = `<p class="muted">${esc(t("Reading the repository…"))}</p>`;
  $("ll-go").disabled = true;
  try {
    const r = await api(`api/llama/variants?repo=${encodeURIComponent(llRepo)}&source=${encodeURIComponent(pickSource())}${dataDir() ? "&" + dirQuery() : ""}`);
    llVariants = r.variants;
    if (!llVariants.length) { box.innerHTML = `<p class="msg-error">${esc(t("No model .gguf files there."))}</p>`; return; }
    box.innerHTML = `<div class="fam"><div class="fam__head"><b>${esc(llRepo)}</b><span class="muted small">${esc(t(r.source === "modelscope" ? "from ModelScope" : "from Hugging Face"))}</span></div>` +
      llVariants.map((v, i) => {
        const [cls, label] = LEVEL[v.fit.level];
        const notes = [t(WHERE[v.fit.where])];
        if (v.fit.need_ram) notes.push(t("It needs about {n} GB of RAM.", {n: v.fit.need_ram}));
        if (v.disk) notes.push(t("Only {free} GB free in the model folder; the download needs about {need} GB (choose a bigger folder under Download folder above).", {free: fmt(r.free_gb, 0), need: fmt((v.left_gb ?? v.gb) + 2, 0)}));
        return `<div class="size"><div class="size__main"><div class="size__name"><b>${esc(v.label)}</b></div>` +
          `<div class="size__nums small">${esc(t("Download {d} GB", {d: fmt(v.gb, 1)}))}${v.files > 1 ? " · " + esc(t("{n} files", {n: v.files})) : ""}</div>` +
          `<ul class="reasons small">${notes.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div>` +
          `<div class="size__side"><span class="st-badge ${cls}">${esc(t(label))}</span><button class="st-btn st-btn--secondary" data-ll-install="${i}">${esc(t("Download and install"))}</button></div></div>`;
      }).join("") + `</div>`;
  } catch (err) {
    box.innerHTML = `<p class="msg-error">${esc(t("Could not read the repository: {e}", {e: err.message}))}</p>`;
  } finally { $("ll-go").disabled = false; }
}
$("ll-go").onclick = () => listVariants($("ll-repo").value);
$("ll-repo").addEventListener("keydown", (e) => { if (e.key === "Enter") listVariants($("ll-repo").value); });
$("ll-suggest").addEventListener("click", (e) => { const b = e.target.closest("[data-repo]"); if (b) listVariants(b.dataset.repo); });
$("ll-variants").addEventListener("click", async (e) => {
  const pick = e.target.closest("[data-repo]");
  if (pick) return listVariants(pick.dataset.repo);
  const b = e.target.closest("[data-ll-install]");
  if (!b) return;
  const v = llVariants[+b.dataset.llInstall];
  if (v.fit.level === "no" && !confirm(t("{model} does not fit in this PC's VRAM and RAM: it would run from the disk, very slowly.\n\nDownload anyway?", {model: v.label}))) return;
  if (v.disk && !confirm(t("Not enough free disk space for {model}: about {need} GB is needed, the model folder has {free} GB. The install will most likely stop at its disk check.\n\nTry anyway?", {model: v.label, need: fmt((v.left_gb ?? v.gb) + 2, 0), free: "?"}))) return;
  try { await api("api/llama/install", {repo: llRepo, variant: v.name, source: pickSource(), data_dir: dataDir() || null}); pollJobs(); }
  catch (err) { toast("error", t(err.status === 409 ? "Something is already running" : "Could not start the install"), err.status === 409 ? t("Wait for it to finish, or stop it.") : err.message); }
});
$("ll-installed").addEventListener("click", async (e) => {
  const st = e.target.closest("[data-start]");
  if (!st) return;
  startModel(st.dataset.start);
});

// ------------------------------------------------------------------ start
let embedded = false;
(async () => {
  try { embedded = (await api("api/context")).embedded; } catch (e) { /* an older manager */ }
  $("tabs").hidden = !embedded;                     // inside a model's server: its tabs; on its own: "Open the chat"
  $("standalone-title").hidden = embedded;
  $("note-standalone").hidden = embedded;
  $("note-embedded").hidden = !embedded;
  if (!embedded) checkRunning();
  loadLlama();
  try { await loadHardware(); await loadModels(); } catch (e) { toast("error", t("The model manager is not reachable"), e.message, 8000); }
  pollJobs();
})();
