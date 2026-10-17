// app.js — state, SSE client, stage cards, results, lightbox. See demo/PLAN.md §4/§7.
import { $, $$, el, clamp, countUp, makeTimer, statusGlyph, debounce, prefersReducedMotion, enter } from "./ui.js";
import { MeshViewer } from "./meshviewer.js";
import { OrbitDiagram, VIEW_GRID, AZ_LIST, EL_LIST } from "./orbit.js";
import { NOTES, organNote, attachNote, noteDot } from "./notes.js";
import { ReconViewer } from "./reconviewer.js";

const BROWSE_HOME = "/Users/jasminavulovic/Documents/Masters/TEZAAA/Totalsegmentator_dataset_v201";
const QUICK_PICKS = ["s0006", "s0005", "s0018", "s0020", "s0055"];
const HERO_DEFAULT_VIEW = "v02_az+0_el+0";
// From EVALUATION/v25_gbuf_glint/speed.json — the actual thesis benchmark, measured on
// the server (NVIDIA L4), never to be confused with this laptop's own demo-run numbers.
const SERVER_REF = {
  cycles_s_per_frame: 53.025,
  net_ms_per_frame_l4: 98.348,
  speedup_x: 539.2,
  total_ms_per_frame_end_to_end: 2349.8,
  speedup_x_end_to_end: 22.6,
};

const ACCENT_HEX = { ct: "#C9B8F0", masks: "#F9C6D9", mesh: "#BDE8D4", gbuffer: "#BCDDF7", buffers: "#FFD9B8", infer: "#E9C6F2", recon: "#F5E3A8" };
const STAGE_META = {
  ct:      { n: 1, title: "CT scan",                what: "Load the volume, window it, and check it can go through the pipeline." },
  masks:   { n: 2, title: "Segmentation masks",      what: "29 TotalSegmentator organ masks, coloured and measured." },
  mesh:    { n: 3, title: "Mesh extraction",         what: "Marching cubes turns each mask into a watertight surface mesh." },
  gbuffer: { n: 4, title: "G-buffer rendering",      what: "Blender renders cheap geometry buffers from all 20 camera angles." },
  buffers: { n: 5, title: "Texture & glint buffers", what: "Procedural tissue texture and vessel glints, packed into the 17-channel network input." },
  infer:   { n: 6, title: "Neural rendering",        what: "The trained generator turns each 17-channel stack into a photoreal frame." },
  recon:   { n: 7, title: "3-D view",                what: "The 20 neural renders projected onto the organ meshes. Exact at the 20 cameras, blended in between." },
};
// Small line icons for the stepper dots (24x24, stroke=currentColor so they inherit
// the dot's active/done colour for free).
const STAGE_ICON_PATH = {
  ct: '<rect x="4" y="4" width="16" height="16" rx="4"/><line x1="4" y1="12" x2="20" y2="12"/><circle cx="12" cy="12" r="1.3" fill="currentColor" stroke="none"/>',
  masks: '<polygon points="12,3.5 20.5,8.5 12,13.5 3.5,8.5"/><polyline points="3.5,12.5 12,17.5 20.5,12.5"/><polyline points="3.5,16.5 12,21.5 20.5,16.5"/>',
  mesh: '<polygon points="12,3 21,19.5 3,19.5"/><line x1="12" y1="3" x2="12" y2="19.5"/><line x1="7.3" y1="11.6" x2="16.7" y2="11.6"/>',
  gbuffer: '<rect x="3" y="7" width="18" height="12.5" rx="2.2"/><path d="M8.2 7 L10 4.2 H14 L15.8 7"/><circle cx="12" cy="13.2" r="3.4"/>',
  buffers: '<rect x="3" y="3" width="7.5" height="7.5" rx="1.4"/><rect x="13.5" y="3" width="7.5" height="7.5" rx="1.4"/><rect x="3" y="13.5" width="7.5" height="7.5" rx="1.4"/><rect x="13.5" y="13.5" width="7.5" height="7.5" rx="1.4"/>',
  infer: '<path d="M12 2.5 L13.9 10.1 L21.5 12 L13.9 13.9 L12 21.5 L10.1 13.9 L2.5 12 L10.1 10.1 Z"/>',
  recon: '<polygon points="12,3 20,7.5 20,16.5 12,21 4,16.5 4,7.5"/><polyline points="4,7.5 12,12 20,7.5"/><line x1="12" y1="12" x2="12" y2="21"/>',
};
function stageIconSvg(stage) {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">${STAGE_ICON_PATH[stage] || ""}</svg>`;
}
const TIMELINE_STAGES = ["masks", "mesh", "gbuffer", "buffers", "infer", "recon"];
const ALL_STAGES = ["ct", "masks", "mesh", "gbuffer", "buffers", "infer", "recon"];
const VIEW_BY_NAME = Object.fromEntries(VIEW_GRID.map(v => [v.view, v]));
const ARROW_SVG = '<svg viewBox="0 0 46 16"><path d="M2 8 H37"/><polygon points="35,3 44,8 35,13"/></svg>';
const TEX_TILE_LABELS = { albedo: "Albedo", nmap: "Normal map", rough: "Roughness", glint: "Vessel glints", highlight: "Highlights" };
// The 17-channel network input, grouped exactly as infer.load_input packs it (see
// demo/PLAN.md channel layout comment) — this grouping drives both the tensor-stack
// legend and the fixed back-to-front layer order.
const TENSOR_GROUPS = [
  { title: "G-buffers", ch: 8, rows: [
    { key: "seg", stage: "gbuffer", label: "Seg RGB", ch: 3 },
    { key: "depth", stage: "gbuffer", label: "Depth", ch: 1 },
    { key: "normals", stage: "gbuffer", label: "Normals", ch: 3 },
    { key: "segid", stage: "gbuffer", label: "Seg ID", ch: 1 },
  ] },
  { title: "Texture", ch: 7, rows: [
    { key: "albedo", stage: "buffers", label: "Albedo", ch: 3 },
    { key: "nmap", stage: "buffers", label: "Normal map", ch: 3 },
    { key: "rough", stage: "buffers", label: "Roughness", ch: 1 },
  ] },
  { title: "Glints", ch: 2, rows: [
    { key: "glint", stage: "buffers", label: "Vessel glints", ch: 1 },
    { key: "highlight", stage: "buffers", label: "Highlights", ch: 1 },
  ] },
];
const TENSOR_LAYERS = TENSOR_GROUPS.flatMap(g => g.rows);
const GROUP_ACCENT = { "G-buffers": "#BCDDF7", "Texture": "#FFD9B8", "Glints": "#E9C6F2" };
const LAYER_ACCENT = Object.fromEntries(TENSOR_GROUPS.flatMap(g => g.rows.map(r => [r.key, GROUP_ACCENT[g.title]])));
const INPUT_TABS = [
  { key: "seg", stage: "gbuffer", label: "Seg" }, { key: "depth", stage: "gbuffer", label: "Depth" },
  { key: "normals", stage: "gbuffer", label: "Normals" }, { key: "segid", stage: "gbuffer", label: "Seg ID" },
  { key: "albedo", stage: "buffers", label: "Albedo" }, { key: "nmap", stage: "buffers", label: "Normal map" },
  { key: "rough", stage: "buffers", label: "Roughness" }, { key: "glint", stage: "buffers", label: "Glints" },
  { key: "highlight", stage: "buffers", label: "Highlights" },
];

const fmtDecimals = { int: 0, float1: 1, float2: 2, ms: 1, s: 1 };

function fmtBytes(n) {
  if (n == null) return "";
  if (n > 1e9) return (n / 1e9).toFixed(2) + " GB";
  if (n > 1e6) return (n / 1e6).toFixed(1) + " MB";
  return (n / 1e3).toFixed(0) + " KB";
}
function round2(x) { return Math.round(x * 100) / 100; }

// ───────────────────────── global state ─────────────────────────
const state = {
  status: null,
  ctPath: "",
  ctData: null,
  runId: null,
  running: false,
  followAlong: true,
  timings: {},
  lastStat: {},             // "stage:key" -> raw stat event (source of truth, not DOM — count-up animates the DOM)
  views: {},               // view -> {gbuffer:{}, buffers:{}, infer:{}}
  galleryMode: "output",
  meshViewer: null,
  orbitDiagram: null,
  bufFanView: null,
  gbufTileView: null,
  heroSet: false,
  lastEventSource: null,
  hadRun: false,
  lightboxView: null,
  selectedView: null,      // view the user clicked; null = cards follow the latest artifact
  latest: {},              // stage -> last view that produced artifacts
  volumes: null,           // masks/volumes.json, for organ notes
  lightboxTab: "render",
  browsePath: BROWSE_HOME,
  // recon (3-D view) card
  reconViewer: null,
  reconViewerUrl: null,     // recon/viewer.json url once known (live artifact or backfill)
  reconLoadedUrl: null,
  reconLoading: false,
  reconCardVisible: false,
  reconMode: "ulr",
  reconCamerasOn: false,
  reconShowUnseen: false,
  reconSelftestRequested: new URLSearchParams(location.search).get("recon_selftest") === "1",
  reconSelftestDone: false,
  reconForceNoFloat: new URLSearchParams(location.search).get("recon_nofloat") === "1",
  meshBuffers: {},          // organ name -> ArrayBuffer, reused by the recon viewer (no re-download)
};

const cards = {};   // stage -> {root, pill, timerCtl, progressText, progressFill, statTilesEl, detailsBody, ...}
const galleryTiles = {}; // view -> {root, img}

// ───────────────────────── boot ─────────────────────────
document.addEventListener("DOMContentLoaded", init);

function init() {
  buildStepper();
  buildChips();
  buildTimeline();
  buildGalleryGrid();
  wireChooseCt();
  wireLightbox();
  wireDrawer();
  wireBrowseModal();
  wireFollowToggle();

  state.meshViewer = new MeshViewer($("#meshCanvas"));
  state.orbitDiagram = new OrbitDiagram($("#orbitCanvas"));
  state.orbitDiagram.onSelect = (view) => selectView(view);
  state.reconViewer = new ReconViewer($("#reconCanvas"));
  window.__reconViewer = state.reconViewer; // debug hook for headless verification, mirrors window.__errs

  pollStatus();
  setInterval(pollStatus, 2500);
  refreshDrawerList();
  showEmptyState();
  openRunFromHash();
}

// #run=<run_id>[&speed=N] opens a finished run directly (instant, or re-timed replay),
// so a finished run can be bookmarked for a presentation.
function openRunFromHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  const runId = h.get("run");
  if (!runId) return;
  const speed = parseFloat(h.get("speed") || "0");
  resetRunUI();
  connectEvents(runId, speed > 0 ? { replay: true, speed } : {});
}

// ───────────────────────── model status ─────────────────────────
async function pollStatus() {
  try {
    const r = await fetch("/api/status");
    const j = await r.json();
    state.status = j;
    const dot = $("#modelDot"), text = $("#modelChipText");
    dot.className = "status-dot " + (j.model === "ready" ? "ready" : j.model === "error" ? "error" : "loading");
    let runName = "gbuf_glint", epoch = "?";
    const m1 = /training_runs\/([^\/]+)\//.exec(j.ckpt || "");
    if (m1) runName = m1[1];
    const m2 = /epoch_(\d+)/.exec(j.ckpt || "");
    if (m2) epoch = parseInt(m2[1], 10);
    const dev = (j.device || "").toUpperCase();
    text.textContent = j.model === "ready" ? `${runName} · epoch ${epoch} · ${dev}`
      : j.model === "error" ? "model error" : "loading model…";
  } catch (e) { /* server not reachable — leave last known state */ }
  updateRunButtonState();
}

// ───────────────────────── stepper ─────────────────────────
function buildStepper() {
  const wrap = $("#stepper");
  wrap.innerHTML = "";
  ALL_STAGES.forEach((stage, i) => {
    const dot = el("div", { class: "dot", html: stageIconSvg(stage) });
    const node = el("div", { class: "stepper-node", id: `step-${stage}`, style: { "--accent": ACCENT_HEX[stage] } }, [
      dot,
      el("div", { class: "label", text: STAGE_META[stage].title }),
    ]);
    node.onclick = () => {
      const target = stage === "ct" ? $("#ctCard") : cards[stage]?.root;
      target?.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" });
    };
    wrap.appendChild(node);
    if (i < ALL_STAGES.length - 1) wrap.appendChild(el("div", { class: "stepper-seg", id: `stepseg-${stage}` }));
  });
}
function setStepperStatus(stage, status) {
  const node = $(`#step-${stage}`);
  if (!node) return;
  node.classList.remove("active", "done", "error");
  if (status === "running") node.classList.add("active");
  if (status === "done") node.classList.add("done");
  if (status === "error") node.classList.add("error");
  const idx = ALL_STAGES.indexOf(stage);
  if (status === "done" && idx < ALL_STAGES.length - 1) $(`#stepseg-${stage}`)?.classList.add("filled");
}

// ───────────────────────── choose a CT ─────────────────────────
function buildChips() {
  const row = $("#chipsRow");
  for (const sid of QUICK_PICKS) {
    const chip = el("button", { class: "chip", "data-sid": sid, text: sid });
    chip.onclick = () => {
      const path = `${BROWSE_HOME}/${sid}`;
      $("#pathField").value = path;
      doInspect(path);
    };
    row.appendChild(chip);
  }
}
function highlightActiveChip(path) {
  $$(".chip").forEach(c => c.classList.toggle("active", path === `${BROWSE_HOME}/${c.dataset.sid}`));
}

function wireChooseCt() {
  const field = $("#pathField");
  field.addEventListener("keydown", (e) => { if (e.key === "Enter") doInspect(field.value.trim()); });
  field.addEventListener("blur", () => { if (field.value.trim()) doInspect(field.value.trim()); });

  $("#browseNativeBtn").onclick = async () => {
    try {
      const r = await fetch("/api/pick", { method: "POST" });
      const j = await r.json();
      if (j.path) { field.value = j.path; doInspect(j.path); }
    } catch (e) { toast("native picker failed: " + e, true); }
  };
  $("#browseAppBtn").onclick = () => openBrowseModal(state.browsePath);
}

function showInspectError(msg) {
  const b = $("#inspectError");
  b.textContent = msg; b.classList.add("shown");
}
function clearInspectError() { $("#inspectError").classList.remove("shown"); }

async function doInspect(path) {
  if (!path) return;
  clearInspectError();
  try {
    const r = await fetch("/api/inspect", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path }) });
    const j = await r.json();
    if (!r.ok) { showInspectError(j.error || "inspect failed"); return; }
    state.ctPath = path;
    state.ctData = {
      sid: j.sid, split: j.split, shape: j.shape, spacing: j.spacing_mm, extent: j.extent_cm,
      hu_range: [j.hu_min, j.hu_max], orientation: j.orientation, load_time: j.load_time_s,
      file_size: j.file_size, dtype: j.dtype, n_tissues: j.n_tissues, n_tissues_total: j.n_tissues_total,
      gt_available: j.gt_available, checks: j.checks, urls: j.urls, spriteMeta: { n: j.sprite_idx.length, w: 256, h: 256 },
    };
    renderCtCard();
    highlightActiveChip(path);
  } catch (e) { showInspectError(String(e)); }
}

function splitBadgeClass(split) {
  if (split === "train") return "badge-train";
  if (split === "external") return "badge-external";
  return "badge-test";
}

function renderCtCard() {
  const d = state.ctData;
  if (!d) return;
  const card = $("#ctCard");
  card.classList.add("shown");
  card.style.setProperty("--wash", "var(--ct)");
  card.innerHTML = "";

  const splitCheck = (d.checks || []).find(c => c.name === "split");
  const splitLabel = splitCheck ? splitCheck.reason : (d.split || "");

  const head = el("div", { class: "ct-card-head" }, [
    el("div", { class: "stage-badge", style: { position: "static", background: "var(--ct)", borderColor: "var(--ct)", color: "#2E2A3B" }, text: "1" }),
    el("h3", { text: `${d.sid}` }),
    el("span", { class: `badge ${splitBadgeClass(d.split)}`, text: splitLabel }),
    d.gt_available ? el("span", { class: "badge badge-test", text: "reference render available" }) : null,
  ]);
  card.appendChild(head);

  const tiles = el("div", { class: "stat-tiles" });
  card.appendChild(tiles);
  const addTile = (key, label, valueText) => upsertStatTileText(tiles, key, label, valueText);
  if (d.shape) addTile("shape", "shape", d.shape.join(" × ") + " voxels");
  if (d.spacing) addTile("spacing", "spacing", d.spacing.map(v => round2(v)).join(" × ") + " mm");
  if (d.extent) addTile("extent", "extent", d.extent.join(" × ") + " cm");
  if (d.hu_range) addTile("hu_range", "HU range", `${Math.round(d.hu_range[0])} to ${Math.round(d.hu_range[1])} HU`);
  if (d.orientation) addTile("orientation", "orientation", d.orientation);
  if (d.file_size) addTile("size", "file size", fmtBytes(d.file_size));
  if (d.dtype) addTile("dtype", "dtype", d.dtype);
  if (d.load_time != null) addTile("load_time", "load time", `${round2(d.load_time)} s`);

  if (d.checks && d.checks.length) {
    const checksEl = el("div", { class: "checks" });
    for (const c of d.checks) {
      checksEl.appendChild(el("div", { class: "check-row" }, [
        statusGlyph(c.status),
        el("span", { class: "name", text: c.name }),
        el("span", { class: "reason", text: c.reason }),
      ]));
    }
    card.appendChild(checksEl);
  }

  if (d.urls) {
    const slices = el("div", { class: "slices-row" });
    for (const axis of ["axial", "coronal", "sagittal"]) {
      if (!d.urls[axis]) continue;
      slices.appendChild(el("div", { class: "slice-well" }, [el("img", { src: d.urls[axis], alt: axis })]));
    }
    card.appendChild(slices);

    if (d.urls.axial_stack) {
      const view = el("div", { class: "scrubber-view", id: "scrubberView" });
      const range = el("input", { type: "range", min: "0", max: String(d.spriteMeta.n - 1), value: String(Math.floor(d.spriteMeta.n / 2)) });
      const idxLabel = el("span", { class: "idx tnum" });
      const masksBtn = el("button", { class: "toggle", id: "masksScrubToggle", type: "button" });
      let showMasks = false;
      const setImg = () => { view.style.backgroundImage = `url(${showMasks && state.masksOverlayStackUrl ? state.masksOverlayStackUrl : d.urls.axial_stack})`; };
      setImg();
      const update = () => {
        const idx = parseInt(range.value, 10);
        const w = view.clientWidth || 256;
        view.style.backgroundSize = `${w}px ${w * d.spriteMeta.n}px`;
        view.style.backgroundPosition = `0 ${-idx * w}px`;
        idxLabel.textContent = `${idx + 1}/${d.spriteMeta.n}`;
      };
      range.addEventListener("input", update);
      window.addEventListener("resize", debounce(update, 100));
      masksBtn.onclick = () => {
        if (!state.masksOverlayStackUrl) return;
        showMasks = !showMasks;
        masksBtn.classList.toggle("on", showMasks);
        setImg(); update();
      };
      update();
      card.appendChild(el("div", { class: "scrubber-wrap" }, [
        view,
        el("div", { class: "scrubber-controls" }, [range, idxLabel]),
        el("div", { class: "masks-toggle" }, [masksBtn, el("span", { text: "show masks (once segmented)" })]),
      ]));
    }
  }

  const fails = (d.checks || []).filter(c => c.status === "fail");
  const runBtn = el("button", { class: "btn btn-primary", id: "runBtn", text: state.running ? "Running…" : "Run pipeline", style: { marginTop: "18px" } });
  runBtn.onclick = onRunClick;
  card.appendChild(runBtn);
  updateRunButtonState();
}

function updateRunButtonState() {
  const btn = $("#runBtn");
  if (!btn) return;
  const d = state.ctData;
  const fails = d ? (d.checks || []).some(c => c.status === "fail") : true;
  const modelReady = state.status && state.status.model === "ready";
  btn.disabled = fails || !modelReady || state.running;
  btn.textContent = state.running ? "Running…" : "Run pipeline";
}

async function onRunClick() {
  if (state.running || !state.ctPath) return;
  const btn = $("#runBtn");
  btn.disabled = true; btn.textContent = "Starting…";
  try {
    const r = await fetch("/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path: state.ctPath }) });
    const j = await r.json();
    if (!r.ok) {
      toast(j.error || `run failed (${r.status})`, true);
      updateRunButtonState();
      return;
    }
    resetRunUI();
    state.running = true;
    connectEvents(j.run_id, {});
  } catch (e) {
    toast(String(e), true);
    updateRunButtonState();
  }
}

// ───────────────────────── browse-in-app modal ─────────────────────────
function wireBrowseModal() {
  $("#browseModalClose").onclick = closeBrowseModal;
  $("#browseModal").addEventListener("click", (e) => { if (e.target.id === "browseModal") closeBrowseModal(); });
}
function openBrowseModal(path) {
  state.browsePath = path;
  $("#browseModal").classList.add("shown");
  loadBrowse(path);
}
function closeBrowseModal() { $("#browseModal").classList.remove("shown"); }

async function loadBrowse(path) {
  const list = $("#browseList");
  list.innerHTML = '<div class="browse-row"><span class="ic">…</span><span class="name">loading…</span></div>';
  try {
    const r = await fetch(`/api/browse?path=${encodeURIComponent(path)}`);
    const j = await r.json();
    if (!r.ok) { list.innerHTML = `<div class="browse-row"><span class="name">${j.error || "error"}</span></div>`; return; }
    state.browsePath = j.path;
    renderBreadcrumb(j.path);
    list.innerHTML = "";
    if (j.parent) {
      const up = el("div", { class: "browse-row" }, [el("span", { class: "ic", text: "↑" }), el("span", { class: "name", text: ".. (up)" })]);
      up.onclick = () => loadBrowse(j.parent);
      list.appendChild(up);
    }
    for (const d of j.dirs) {
      const row = el("div", { class: "browse-row" + (d.has_ct ? " has-ct" : "") }, [
        el("span", { class: "ic", text: d.has_ct ? "⌘" : "\u{1F4C1}" }),
        el("span", { class: "name", text: d.name }),
        d.has_ct ? el("span", { class: "pick", text: "ct.nii.gz" }) : null,
      ]);
      row.onclick = () => { if (d.has_ct) selectPath(d.path); else loadBrowse(d.path); };
      list.appendChild(row);
    }
    for (const f of j.files) {
      const row = el("div", { class: "browse-row nii" }, [el("span", { class: "ic", text: "\u{1F9EC}" }), el("span", { class: "name", text: f.name })]);
      row.onclick = () => selectPath(f.path);
      list.appendChild(row);
    }
  } catch (e) {
    list.innerHTML = `<div class="browse-row"><span class="name">${e}</span></div>`;
  }
}
function renderBreadcrumb(path) {
  const bc = $("#browseBreadcrumb");
  bc.innerHTML = "";
  const parts = path.split("/").filter(Boolean);
  let acc = "";
  const root = el("span", { text: "/" });
  root.onclick = () => loadBrowse("/");
  bc.appendChild(root);
  parts.forEach((p, i) => {
    acc += "/" + p;
    const full = acc;
    bc.appendChild(el("span", { class: "sep", text: " / " }));
    const s = el("span", { text: p });
    s.onclick = () => loadBrowse(full);
    bc.appendChild(s);
  });
}
function selectPath(path) {
  $("#pathField").value = path;
  closeBrowseModal();
  doInspect(path);
}

// ───────────────────────── timeline / stage cards ─────────────────────────
function statSuffix(fmt) { return fmt in fmtDecimals; }

function upsertStatTileText(container, key, label, valueText) {
  let tile = container.querySelector(`[data-key="${key}"]`);
  if (!tile) {
    tile = el("div", { class: "stat-tile", "data-key": key }, [el("div", { class: "k", text: label }), el("div", { class: "v" })]);
    container.appendChild(tile);
    enter(tile);
  }
  tile.querySelector(".v").textContent = valueText;
  return tile;
}

const ARRAY_LABELS = { shape: "voxels", spacing: "mm", extent: "cm" };
function upsertStatTile(container, ev) {
  let tile = container.querySelector(`[data-key="${ev.key}"]`);
  if (!tile) {
    tile = el("div", { class: "stat-tile", "data-key": ev.key }, [el("div", { class: "k", text: ev.label }), el("div", { class: "v" })]);
    container.appendChild(tile);
    enter(tile);
  }
  const vEl = tile.querySelector(".v");
  if (Array.isArray(ev.value)) {
    if (ev.key === "hu_range") vEl.textContent = `${Math.round(ev.value[0])} to ${Math.round(ev.value[1])} ${ev.unit}`;
    else vEl.textContent = ev.value.map(x => typeof x === "number" ? round2(x) : x).join(" × ") + (ev.unit ? " " + ev.unit : "");
  } else if (typeof ev.value === "number" && statSuffix(ev.fmt)) {
    vEl.innerHTML = "";
    const num = el("span", { class: "tnum" });
    vEl.appendChild(num);
    if (ev.unit) vEl.appendChild(el("span", { class: "unit", text: " " + ev.unit }));
    countUp(num, ev.value, { decimals: fmtDecimals[ev.fmt] });
  } else {
    vEl.textContent = String(ev.value) + (ev.unit ? " " + ev.unit : "");
  }
}

function makeIoWell(labelText) {
  return el("div", { class: "io-well" }, [el("img", {}), el("span", { class: "well-label", text: labelText })]);
}
function makeArrow() {
  const a = el("div", { class: "io-arrow" });
  a.innerHTML = ARROW_SVG;
  return a;
}

// The 17-channel network input as a readable isometric layer stack (CSS 3-D:
// translateZ per layer), with an optional legend column grouped G-buffers/Texture/
// Glints — hovering a legend row lifts its layer. `compact` drops the legend for use
// as the neural-rendering card's input well.
function buildTensorStack({ compact = false } = {}) {
  const imgs = {};
  const layerEls = {};
  const rowEls = {};
  const layers = TENSOR_LAYERS.map((f, i) => {
    const img = el("img", {});
    imgs[f.key] = img;
    const layer = el("div", { class: "tensor-layer", "data-key": f.key, style: { "--i": String(i), "--accent": LAYER_ACCENT[f.key] } }, [img]);
    layerEls[f.key] = layer;
    return layer;
  });
  const inner = el("div", { class: "tensor-stack-inner" }, layers);
  const stack = el("div", { class: "tensor-stack" }, [inner]);
  const root = el("div", { class: "tensor-wrap" + (compact ? " compact" : "") }, [stack]);

  // Hovering (or clicking, to pin) a layer or its legend row lifts that layer out of the
  // stack and pops a note on what the channel does and why it is there.
  const lift = (key) => {
    Object.values(layerEls).forEach(l => l.classList.toggle("lifted", l.dataset.key === key));
    Object.values(rowEls).forEach(r => r.classList.toggle("active", r.dataset.key === key));
    stack.classList.add("has-lift");
  };
  const drop = () => {
    Object.values(layerEls).forEach(l => l.classList.remove("lifted"));
    Object.values(rowEls).forEach(r => r.classList.remove("active"));
    stack.classList.remove("has-lift");
  };
  const noteFor = (key) => () => {
    const src = imgs[key].getAttribute("src");
    return { ...NOTES[key], extra: src ? el("img", { class: "spy-img", src }) : null };
  };
  for (const f of TENSOR_LAYERS) {
    attachNote(layerEls[f.key], noteFor(f.key), { accent: LAYER_ACCENT[f.key], onShow: () => lift(f.key), onHide: drop });
  }

  if (!compact) {
    const legend = el("div", { class: "tensor-legend" });
    for (const g of TENSOR_GROUPS) {
      const rows = g.rows.map((r) => {
        const row = el("div", { class: "legend-row", "data-key": r.key, style: { "--accent": GROUP_ACCENT[g.title] } }, [
          el("span", { text: r.label }), el("span", { class: "ch", text: `${r.ch} ch` }),
        ]);
        rowEls[r.key] = row;
        attachNote(row, noteFor(r.key), { accent: GROUP_ACCENT[g.title], onShow: () => lift(r.key), onHide: drop });
        return row;
      });
      legend.appendChild(el("div", { class: "legend-group" }, [el("div", { class: "legend-title", text: `${g.title} · ${g.ch} ch` }), ...rows]));
    }
    legend.appendChild(el("div", { class: "legend-hint", text: "hover a layer to see what it does · click to pin" }));
    root.appendChild(legend);
  }
  return { root, imgs };
}

function buildStageCardShell(stage) {
  const meta = STAGE_META[stage];
  const root = el("div", { class: "card stage-card", id: `stage-${stage}`, style: { "--accent": ACCENT_HEX[stage], "--wash": ACCENT_HEX[stage] } });
  root.appendChild(el("div", { class: "stage-badge", text: meta.n }));
  const pill = el("div", { class: "stage-pill" }, [el("span", { class: "pilltext", text: "waiting" })]);
  const headDots = el("span", { class: "head-dots" });
  const viewChipText = el("span", {});
  const viewChipBtn = el("button", { text: "↺ follow latest" });
  viewChipBtn.onclick = () => clearSelection();
  const viewChip = el("div", { class: "view-chip" }, [viewChipText, viewChipBtn]);
  root.appendChild(el("div", { class: "stage-head" }, [
    el("div", {}, [el("h3", {}, [meta.title, headDots]), el("div", { class: "stage-what", text: meta.what }), viewChip]),
    pill,
  ]));
  const progressText = el("div", { class: "ptext tnum" });
  const progressFill = el("span", {});
  const progressWrap = el("div", { class: "stage-progress", style: { display: "none" } }, [
    el("div", { class: "bar" }, [progressFill]), progressText,
  ]);
  root.appendChild(progressWrap);
  const errBox = el("div", { class: "error-banner" });
  root.appendChild(errBox);

  const ioRow = el("div", { class: "io-row" });
  root.appendChild(ioRow);
  const statTiles = el("div", { class: "stat-tiles" });
  root.appendChild(statTiles);
  const extra = el("div", {});
  root.appendChild(extra);
  const detailsBody = el("div", { class: "details-body" });
  root.appendChild(el("details", { class: "details-box" }, [el("summary", { text: "Details" }), detailsBody]));

  cards[stage] = {
    root, pill, pillText: pill.querySelector(".pilltext"), timerCtl: null,
    progressWrap, progressText, progressFill, errBox, ioRow, statTiles, extra, detailsBody,
    headDots, viewChip, viewChipText,
  };
  return root;
}

function setStageStatus(stage, status, seconds) {
  setStepperStatus(stage, status);
  const c = cards[stage];
  if (!c) return;
  c.root.classList.remove("running", "done", "error");
  if (status !== "waiting") c.root.classList.add(status);
  c.root.querySelectorAll(".io-arrow").forEach(a => a.classList.toggle("flowing", status === "running"));
  if (status === "running") {
    c.pillText.textContent = "running";
    const timerNode = el("span", { class: "timer tnum" });
    c.pill.innerHTML = ""; c.pill.appendChild(el("span", { class: "pilltext", text: "running" })); c.pill.appendChild(timerNode);
    c.pillText = c.pill.querySelector(".pilltext");
    c.timerCtl = makeTimer(timerNode);
    c.timerCtl.start();
    c.progressWrap.style.display = "flex";
  } else if (status === "done") {
    if (c.timerCtl) c.timerCtl.stop(seconds);
    c.pill.innerHTML = "";
    c.pill.appendChild(el("span", { class: "pilltext", text: "done" }));
    c.pill.appendChild(el("span", { class: "timer tnum", text: seconds != null ? `${seconds.toFixed(1)} s` : "" }));
  } else if (status === "error") {
    if (c.timerCtl) c.timerCtl.stop(seconds);
    c.pill.innerHTML = "";
    c.pill.appendChild(el("span", { class: "pilltext", text: "error" }));
  }
  if (state.followAlong && status === "running") {
    c.root.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" });
  }
}

function setProgress(stage, done, total, label) {
  const c = cards[stage];
  if (!c) return;
  c.progressWrap.style.display = "flex";
  c.progressText.textContent = `${label} — ${done}/${total}`;
  c.progressFill.style.transform = `scaleX(${clamp(done / Math.max(1, total), 0, 1)})`;
}

function buildTimeline() {
  const tl = $("#timeline");
  tl.innerHTML = "";

  // masks
  const masksCard = buildStageCardShell("masks");
  cards.masks.ioRow.appendChild(makeIoWell("CT slice")); cards.masks.ioRow.children[0].querySelector(".well-label").textContent = "CT slice";
  cards.masks.ioRow.appendChild(makeArrow());
  cards.masks.ioRow.appendChild(makeIoWell("Segmentation overlay"));
  cards.masks.organChips = el("div", { class: "organ-chips", style: { marginTop: "6px" } });
  cards.masks.extra.appendChild(cards.masks.organChips);
  tl.appendChild(masksCard);

  // mesh
  const meshCard = buildStageCardShell("mesh");
  cards.mesh.ioRow.appendChild(makeIoWell("Segmentation overlay"));
  cards.mesh.ioRow.appendChild(makeArrow());
  const meshDot = noteDot(NOTES.meshMethod, { accent: ACCENT_HEX.mesh });
  meshDot.classList.add("floating");
  const meshCanvasWrap = el("div", { class: "mesh-canvas-wrap" }, [el("canvas", { id: "meshCanvas" }), meshDot, el("div", { class: "mesh-hint", text: "drag to orbit · scroll to zoom · double-click to auto-rotate" })]);
  cards.mesh.ioRow.appendChild(meshCanvasWrap);
  cards.mesh.headDots.appendChild(noteDot(NOTES.meshStats, { accent: ACCENT_HEX.mesh }));
  cards.mesh.organRow = el("div", { class: "mesh-organs" });
  cards.mesh.extra.appendChild(cards.mesh.organRow);
  cards.mesh.organGroups = {};
  cards.mesh.detailsBody.appendChild(el("table", { class: "details-table" }, [
    el("thead", {}, [el("tr", {}, [el("th", { text: "organ" }), el("th", { text: "verts" }), el("th", { text: "tris" }), el("th", { text: "ms" })])]),
    el("tbody", { id: "meshTableBody" }),
  ]));
  tl.appendChild(meshCard);

  // gbuffer
  const gbufferCard = buildStageCardShell("gbuffer");
  cards.gbuffer.ioRow.style.display = "none"; // uses the custom .gbuffer-row layout below instead
  const row = el("div", { class: "gbuffer-row" });
  const orbitDot = noteDot(NOTES.orbit, { accent: ACCENT_HEX.gbuffer });
  orbitDot.classList.add("floating");
  row.appendChild(el("div", { class: "orbit-canvas-wrap" }, [el("canvas", { id: "orbitCanvas" }), orbitDot, el("div", { class: "orbit-hint", text: "click a camera to inspect that view" })]));
  const tiles2x2 = el("div", { class: "tiles-2x2" });
  const tileRefs = {};
  const GBUF_TAG = { seg: "Seg", depth: "Depth", normals: "Normals", segid: "Seg ID" };
  for (const k of ["seg", "depth", "normals", "segid"]) {
    const img = el("img", {});
    const tile = el("div", { class: "tile" }, [img, el("span", { class: "tag", text: GBUF_TAG[k] })]);
    attachNote(tile, NOTES[k], { accent: ACCENT_HEX.gbuffer });
    tiles2x2.appendChild(tile);
    tileRefs[k] = img;
  }
  cards.gbuffer.headDots.appendChild(noteDot(NOTES.gbufferSpeed, { accent: ACCENT_HEX.gbuffer }));
  row.appendChild(tiles2x2);
  cards.gbuffer.extra.appendChild(row);
  const filmstrip = el("div", { class: "filmstrip" });
  cards.gbuffer.extra.appendChild(filmstrip);
  cards.gbuffer.tiles2x2 = tileRefs;
  cards.gbuffer.filmstrip = filmstrip;
  tl.appendChild(gbufferCard);

  // buffers — small "input" well (seg + geometry) next to a real-size grid of the
  // five texture/glint outputs, then the full 17-channel tensor stack + legend below.
  const buffersCard = buildStageCardShell("buffers");
  const bIo = cards.buffers.ioRow;
  bIo.classList.add("io-row-buffers");
  const segThumbImg = el("img", {});
  bIo.appendChild(el("div", { class: "io-well io-well-input" }, [segThumbImg, el("span", { class: "well-label", text: "Input · seg + geometry" })]));
  bIo.appendChild(makeArrow());
  const tiles5 = el("div", { class: "tiles-5" });
  const tile5Refs = {};
  for (const k of ["albedo", "nmap", "rough", "glint", "highlight"]) {
    const img = el("img", {});
    const tile = el("div", { class: "tile-sm" }, [img, el("span", { class: "tag", text: TEX_TILE_LABELS[k] })]);
    attachNote(tile, NOTES[k], { accent: LAYER_ACCENT[k] });
    tiles5.appendChild(tile);
    tile5Refs[k] = img;
  }
  bIo.appendChild(tiles5);
  cards.buffers.segThumbImg = segThumbImg;
  cards.buffers.tiles5 = tile5Refs;

  const tensor = buildTensorStack({ compact: false });
  cards.buffers.extra.appendChild(el("div", { style: { marginTop: "18px" } }, [tensor.root]));
  cards.buffers.fanImgs = tensor.imgs;
  tl.appendChild(buffersCard);

  // infer — the compact tensor stack as the (square) input, developing into the
  // (equally square) output, vertically aligned.
  const inferCard = buildStageCardShell("infer");
  cards.infer.ioRow.classList.add("io-row-square");
  const stackWell = el("div", { class: "io-well io-well-tensor" });
  const compactTensor = buildTensorStack({ compact: true });
  stackWell.appendChild(compactTensor.root);
  stackWell.appendChild(el("span", { class: "well-label", text: "Input · 17-channel stack" }));
  cards.infer.ioRow.appendChild(stackWell);
  cards.infer.ioRow.appendChild(makeArrow());
  const qualityDot = noteDot(NOTES.quality, { accent: ACCENT_HEX.infer });
  qualityDot.classList.add("floating");
  const outWell = el("div", { class: "infer-output-well" }, [el("img", { class: "develop-img" }), qualityDot]);
  cards.infer.headDots.appendChild(noteDot(NOTES.network, { accent: ACCENT_HEX.infer }));
  cards.infer.headDots.appendChild(noteDot(NOTES.speed, { accent: ACCENT_HEX.infer }));
  cards.infer.ioRow.appendChild(outWell);
  cards.infer.outputImg = outWell.querySelector("img");
  cards.infer.stackImgs = compactTensor.imgs;
  tl.appendChild(inferCard);

  // recon — card 7, "3-D view" (after Neural rendering, before Results)
  tl.appendChild(buildReconCard());

  $("#emptyState")?.remove();
  const emptyState = el("div", { class: "card empty-state", id: "emptyState" }, [
    el("div", { class: "icon", text: "◈" }),
    el("h3", { text: "No run yet" }),
    el("p", { text: "Choose a CT above and run the pipeline, or load the most recent finished run to explore the demo." }),
    (() => { const b = el("button", { class: "btn btn-primary", text: "Load last run" }); b.onclick = loadLastRun; return b; })(),
  ]);
  tl.parentNode.insertBefore(emptyState, tl);
  tl.style.display = "none";
}

// ───────────────────────── card 7: 3-D view (recon) ─────────────────────────
// The WebGL2 viewer (reconviewer.js) is constructed once in init(), but it only grabs a
// GL context / loads data once this card enters the viewport AND the run has produced a
// `viewer` artifact (or a backfill has built one) — see maybeLoadReconViewer().
function buildReconCard() {
  const reconCard = buildStageCardShell("recon");
  const c = cards.recon;
  c.ioRow.style.display = "none"; // custom .recon-row layout below

  const canvas = el("canvas", { id: "reconCanvas" });
  const hudTxt = el("span", { class: "txt" });
  const hudExact = el("span", { class: "exact-pill", text: "exact" });
  const hud = el("div", { class: "recon-hud" }, [hudTxt, hudExact]);
  const hint = el("div", { class: "recon-hint", text: "drag to orbit · scroll to zoom · double-click resets to v02" });
  const compareImg = el("img", {});
  const compareHandle = el("div", { class: "compare-handle", style: { left: "50%" } });
  const compareOv = el("div", { class: "recon-compare-ov" }, [
    compareImg, el("span", { class: "compare-tag left", text: "actual render" }),
    el("span", { class: "compare-tag right", text: "3-D view" }), compareHandle,
  ]);
  const canvasWrap = el("div", { class: "recon-canvas-wrap" }, [canvas, compareOv, hud, hint]);
  wireHandleDragWipe(compareHandle, compareImg, canvasWrap);

  const modeBtns = {};
  const segctl = el("div", { class: "segctl" });
  for (const [mode, noteKey, label] of [["ulr", "reconBlended", "Blended"], ["baked", "reconBaked", "Baked"], ["nearest", "reconNearest", "Nearest"]]) {
    const btn = el("button", { text: label, "data-mode": mode, class: mode === "ulr" ? "active" : "" });
    attachNote(btn, NOTES[noteKey], { accent: ACCENT_HEX.recon });
    btn.onclick = () => setReconMode(mode);
    modeBtns[mode] = btn;
    segctl.appendChild(btn);
  }
  const camBtn = el("button", { class: "toggle toggle-light", type: "button" });
  camBtn.onclick = () => {
    state.reconCamerasOn = !state.reconCamerasOn;
    camBtn.classList.toggle("on", state.reconCamerasOn);
    state.reconViewer?.showCameras(state.reconCamerasOn);
  };
  const unseenBtn = el("button", { class: "toggle toggle-light", type: "button" });
  const unseenNote = {
    kicker: "3-D view · toggle", title: "Never-seen surface",
    what: "Tints, flat lilac, the surface no camera ever photographed — about 28% of the mesh by area, mostly in the pockets between organs.",
    why: "Those patches are filled in from neighbouring captured colour, not shown as flat placeholder. About 95% of what a typical new viewpoint shows was genuinely captured by one of the 20 cameras; this toggle is where the rest is.",
  };
  const unseenWrap = el("div", { class: "recon-cameras-toggle" }, [unseenBtn, el("span", { text: "never-seen" })]);
  attachNote(unseenWrap, unseenNote, { accent: ACCENT_HEX.recon });
  unseenBtn.onclick = () => {
    state.reconShowUnseen = !state.reconShowUnseen;
    unseenBtn.classList.toggle("on", state.reconShowUnseen);
    state.reconViewer?.setShowUnseen(state.reconShowUnseen);
  };
  const show3dBtn = el("button", { class: "btn btn-secondary btn-sm recon-show3d-btn", text: "↳ Show in 3-D" });
  show3dBtn.onclick = () => {
    if (!state.selectedView) return;
    reconSnapByView(state.selectedView);
    reconCard.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" });
  };
  const buildBtn = el("button", { class: "btn btn-primary btn-sm", text: "Build 3-D view", style: { display: "none" } });
  buildBtn.onclick = onBuildReconClick;
  const modesRow = el("div", { class: "recon-modes-row" }, [
    segctl, el("div", { class: "recon-cameras-toggle" }, [camBtn, el("span", { text: "cameras" })]), unseenWrap, show3dBtn, buildBtn,
  ]);

  const snapGrid = el("div", { class: "recon-snapgrid" });
  const snapTiles = {};
  let si = 0;
  for (const elv of EL_LIST) {
    for (const az of AZ_LIST) {
      const v = VIEW_GRID.find((g) => g.az === az && g.el === elv);
      const tile = el("div", { class: "recon-snap-tile", style: { animationDelay: `${si * 20}ms` } });
      tile.onclick = () => reconSnapByView(v.view);
      snapGrid.appendChild(tile);
      snapTiles[v.view] = tile;
      si++;
    }
  }
  const snapWrap = el("div", { class: "recon-snapgrid-wrap" }, [
    el("div", { class: "recon-snapgrid-title", text: "snap to a capture" }), snapGrid,
  ]);

  const controls = el("div", { class: "recon-controls" }, [modesRow, snapWrap]);
  const row = el("div", { class: "recon-row" }, [canvasWrap, controls]);
  c.extra.appendChild(row);

  const selftestEl = el("div", { class: "recon-selftest-line", text: "self-test: not run (add ?recon_selftest=1 to the URL)" });
  const caveats = el("ul", { class: "recon-caveats" }, [
    el("li", { text: "Not a new neural render — the network only ran at the 20 captured cameras; in between, this is image-based interpolation over the CT-derived mesh, never “reconstructed by the network.”" }),
    el("li", { text: "Highlights are view-dependent: between captures, glints and wet highlights cross-fade or ghost (Blended), or smear into a matte patch (Baked)." }),
    el("li", { text: "The views are sparse — 72° azimuth steps put the nearest capture up to about 38° away. About 28% of the mesh surface, by area, sits in pockets between organs that none of the 20 cameras ever sees; that colour is filled in from the nearest surface the cameras did capture, not shown flat. Still, about 95% of what a typical new viewpoint actually shows on screen was genuinely captured — the “never-seen” toggle marks the rest honestly." }),
  ]);
  c.detailsBody.appendChild(el("div", {}, [caveats, selftestEl]));

  c.headDots.appendChild(noteDot(NOTES.recon, { accent: ACCENT_HEX.recon }));

  Object.assign(c, { canvas, canvasWrap, hudTxt, hudExact, modeBtns, camBtn, unseenBtn, show3dBtn, buildBtn, snapTiles, compareOv, compareImg, compareHandle, selftestEl });

  const io = new IntersectionObserver((entries) => {
    state.reconCardVisible = entries[0]?.isIntersecting ?? false;
    if (state.reconCardVisible) maybeLoadReconViewer();
  }, { threshold: 0.01 });
  io.observe(canvasWrap);
  c.io = io;

  return reconCard;
}

// Drag-to-wipe compare: the handle alone captures the pointer, so the canvas underneath
// stays draggable (orbit) everywhere else on the card.
function wireHandleDragWipe(handle, img, container) {
  let dragging = false;
  const setPct = (clientX) => {
    const rect = container.getBoundingClientRect();
    const pct = clamp(((clientX - rect.left) / rect.width) * 100, 0, 100);
    handle.style.left = pct + "%";
    img.style.clipPath = `inset(0 0 0 ${pct}%)`;
  };
  handle.addEventListener("pointerdown", (e) => {
    dragging = true; e.stopPropagation(); setPct(e.clientX);
    try { handle.setPointerCapture(e.pointerId); } catch { /* Safari/edge cases can reject this */ }
  });
  handle.addEventListener("pointermove", (e) => { if (dragging) { e.stopPropagation(); setPct(e.clientX); } });
  handle.addEventListener("pointerup", () => { dragging = false; });
}

function setReconMode(mode) {
  state.reconMode = mode;
  const c = cards.recon;
  if (c) Object.entries(c.modeBtns).forEach(([m, b]) => b.classList.toggle("active", m === mode));
  state.reconViewer?.setMode(mode);
}

function reconSnapByView(view) {
  const viewer = state.reconViewer;
  if (!viewer || !viewer.ready) { toast("the 3-D view is not ready yet", true); return; }
  const idx = viewer.cams.findIndex((cm) => cm.view === view);
  if (idx < 0) return;
  viewer.snapTo(idx);
}

function onReconHud(text, exact) {
  const c = cards.recon;
  if (!c) return;
  c.hudTxt.textContent = text;
  c.hudExact.classList.toggle("on", exact);
  const viewer = state.reconViewer;
  const view = exact && viewer && viewer.snappedIndex != null ? viewer.cams[viewer.snappedIndex].view : null;
  updateReconCompare(view);
  updateReconSnapHighlight(view);
}
function updateReconCompare(view) {
  const c = cards.recon;
  if (!c) return;
  const v = view && state.views[view];
  if (v && v.infer && v.infer.output) {
    if (c.compareImg.getAttribute("src") !== v.infer.output) c.compareImg.src = v.infer.output;
    c.compareOv.classList.add("shown");
  } else {
    c.compareOv.classList.remove("shown");
  }
}
function updateReconSnapHighlight(view) {
  const c = cards.recon;
  if (!c) return;
  Object.entries(c.snapTiles).forEach(([v, t]) => t.classList.toggle("active", v === view));
}

async function maybeLoadReconViewer() {
  if (!state.reconViewerUrl || !state.reconCardVisible) return;
  if (state.reconLoading || state.reconLoadedUrl === state.reconViewerUrl) return;
  state.reconLoading = true;
  try {
    await state.reconViewer.load(state.reconViewerUrl, { meshBuffers: state.meshBuffers, forceNoFloat: state.reconForceNoFloat });
    state.reconLoadedUrl = state.reconViewerUrl;
    onReconViewerReady();
  } catch (e) {
    toast("3-D view failed to load: " + e, true);
  } finally {
    state.reconLoading = false;
  }
}

function onReconViewerReady() {
  const viewer = state.reconViewer, c = cards.recon;
  if (!c) return;
  if (viewer.noWebGL2) {
    c.canvasWrap.classList.add("no-webgl");
    if (!c.noWebglNote) { c.noWebglNote = el("div", { class: "no-webgl-note", text: "the 3-D view needs WebGL2" }); c.canvasWrap.appendChild(c.noWebglNote); }
    return;
  }
  if (c.noWebglNote) c.noWebglNote.remove(); c.noWebglNote = null;
  viewer.onHud = onReconHud;
  viewer.setMode(state.reconMode);
  viewer.showCameras(state.reconCamerasOn);
  viewer.setShowUnseen(state.reconShowUnseen);
  if (viewer.noFloat) {
    c.modeBtns.ulr.style.display = "none";
    c.modeBtns.nearest.style.display = "none";
    setReconMode("baked");
  }
  updateViewChips();
  maybeRunReconSelftest();
}

async function maybeRunReconSelftest() {
  if (!state.reconSelftestRequested || state.reconSelftestDone) return;
  const viewer = state.reconViewer;
  if (!viewer || !viewer.ready || !state.runId) return;
  state.reconSelftestDone = true;
  try {
    const res = await viewer.selftest(`/runs/${state.runId}/recon/selftest`);
    if (cards.recon) cards.recon.selftestEl.textContent = "self-test (PSNR dB): " + JSON.stringify(res);
  } catch (e) {
    if (cards.recon) cards.recon.selftestEl.textContent = "self-test failed: " + e;
  }
}

function renderReconStatsFromBackfill(stats) {
  const c = cards.recon;
  if (!c) return;
  const t = c.statTiles;
  upsertStatTileText(t, "views", "views", String(stats.views));
  upsertStatTileText(t, "verts", "vertices", stats.vertices.toLocaleString("en-US"));
  upsertStatTileText(t, "seen_pct", "surface seen", `${(stats.seen_frac_area_weighted * 100).toFixed(1)} %`);
  upsertStatTileText(t, "spread_net", "cross-view spread · network", `${stats.spread_net_rms8.toFixed(1)} RMS/255`);
  if (stats.spread_cycles_rms8 != null) upsertStatTileText(t, "spread_cycles", "cross-view spread · Cycles", `${stats.spread_cycles_rms8.toFixed(1)} RMS/255`);
  upsertStatTileText(t, "build_s", "build time", `${stats.seconds.toFixed(2)} s`);
}

async function onBuildReconClick() {
  if (!state.runId) return;
  const btn = cards.recon.buildBtn;
  btn.disabled = true; btn.textContent = "Building…";
  try {
    const r = await fetch(`/api/runs/${state.runId}/recon`, { method: "POST" });
    const j = await r.json();
    if (!r.ok) { toast(j.error || "3-D view build failed", true); btn.disabled = false; btn.textContent = "Build 3-D view"; return; }
    state.reconViewerUrl = j.viewer;
    btn.style.display = "none";
    if (j.stats) { renderReconStatsFromBackfill(j.stats); setStageStatus("recon", "done", j.stats.seconds); }
    // The user just clicked a button inside this card, so it is on screen by definition —
    // don't depend on the IntersectionObserver's callback having already fired (it can lag
    // a frame behind a click that happens right as the card scrolls into view).
    state.reconCardVisible = true;
    maybeLoadReconViewer();
  } catch (e) {
    toast(String(e), true);
    btn.disabled = false; btn.textContent = "Build 3-D view";
  }
}

async function loadLastRun() {
  try {
    const r = await fetch("/api/runs");
    const j = await r.json();
    const runs = (j.runs || []).filter(r => r.status === "done");
    if (!runs.length) { toast("no finished runs yet", true); return; }
    resetRunUI();
    connectEvents(runs[0].run_id, {});
  } catch (e) { toast(String(e), true); }
}

function showEmptyState() {
  const es = $("#emptyState");
  if (es) es.style.display = state.hadRun ? "none" : "block";
}

function wireFollowToggle() {
  const btn = $("#followToggle");
  btn.onclick = () => {
    state.followAlong = !state.followAlong;
    btn.classList.toggle("on", state.followAlong);
  };
}

// ───────────────────────── gallery grid ─────────────────────────
function buildGalleryGrid() {
  const body = $("#galleryBody");
  body.innerHTML = "";
  const azHeader = el("div", { class: "gallery-axis-col" }, AZ_LIST.map(az => el("span", { text: `az ${az}°` })));
  body.appendChild(el("div", { class: "gallery-row gallery-row-header" }, [el("span", {}), azHeader]));
  let idx = 0;
  for (const elv of EL_LIST) {
    const cells = el("div", { class: "cells" });
    for (const az of AZ_LIST) {
      const v = VIEW_GRID.find(g => g.az === az && g.el === elv);
      const tile = el("div", { class: "gallery-tile empty", style: { animationDelay: `${idx * 30}ms` }, "data-view": v.view });
      tile.onclick = () => { if (state.views[v.view]?.infer?.output) openLightbox(v.view, "render"); };
      cells.appendChild(tile);
      galleryTiles[v.view] = { root: tile, img: null };
      idx++;
    }
    body.appendChild(el("div", { class: "gallery-row" }, [el("span", { class: "el-label", text: `el ${elv >= 0 ? "+" : ""}${elv}°` }), cells]));
  }
  $("#gallerySeg").querySelectorAll("button").forEach(btn => {
    btn.onclick = () => {
      state.galleryMode = btn.dataset.mode;
      $("#gallerySeg").querySelectorAll("button").forEach(b => b.classList.toggle("active", b === btn));
      Object.keys(galleryTiles).forEach(updateGalleryTile);
    };
  });
}
function getModeUrl(view, mode) {
  const v = state.views[view];
  if (!v) return null;
  if (mode === "output") return (v.infer && (v.infer.thumb || v.infer.output)) || null;
  if (mode === "seg") return (v.gbuffer && v.gbuffer.seg) || null;
  if (mode === "normals") return (v.gbuffer && v.gbuffer.normals) || null;
  if (mode === "glint") return (v.buffers && v.buffers.glint) || null;
  return null;
}
function updateGalleryTile(view) {
  const t = galleryTiles[view];
  if (!t) return;
  const url = getModeUrl(view, state.galleryMode);
  if (!url) { t.root.classList.add("empty"); return; }
  t.root.classList.remove("empty");
  if (!t.img) { t.img = el("img", {}); t.root.appendChild(t.img); }
  if (t.img.src !== location.origin + url && t.img.getAttribute("src") !== url) t.img.src = url;
  t.root.classList.add("loaded");
}

// ───────────────────────── results hero ─────────────────────────
function maybeSetHero(view) {
  const v = state.views[view];
  if (!v || !v.infer || !v.infer.output) return;
  if (state.heroSet && view !== HERO_DEFAULT_VIEW) return;
  if (view === HERO_DEFAULT_VIEW || !state.heroSet) {
    $("#resultsTitle").style.display = "flex";
    $("#resultsHero").style.display = "block";
    const img = $("#heroImg");
    img.src = v.infer.output;
    const info = VIEW_BY_NAME[view];
    $("#heroCaption").innerHTML = `<b>${view}</b> &nbsp;·&nbsp; az ${info.az}°, el ${info.el >= 0 ? "+" : ""}${info.el}°`;
    $("#heroWell").onclick = () => openLightbox(view, "render");
    state.heroSet = view === HERO_DEFAULT_VIEW ? true : state.heroSet;
    if (view === HERO_DEFAULT_VIEW) state.heroSet = true;
  }
}

// ───────────────────────── summary bar ─────────────────────────
function renderSummary() {
  const card = $("#summaryCard");
  card.style.display = "block";
  card.style.setProperty("--wash", "var(--buffers)");
  const total = ALL_STAGES.reduce((s, k) => s + (state.timings[k] || 0), 0) || 1;
  card.innerHTML = "";
  const bars = el("div", { class: "summary-bars" });
  const legend = el("div", { class: "summary-legend" });
  for (const stage of ALL_STAGES) {
    const secs = state.timings[stage];
    if (secs == null) continue;
    const frac = secs / total;
    const seg = el("div", { class: "summary-bar-seg", style: { width: `${frac * 100}%`, background: ACCENT_HEX[stage] } });
    bars.appendChild(seg);
    requestAnimationFrame(() => { seg.style.transform = "scaleX(1)"; });
    legend.appendChild(el("div", { class: "item" }, [el("span", { class: "sw", style: { background: ACCENT_HEX[stage] } }), `${STAGE_META[stage].title} — ${secs.toFixed(1)} s`]));
  }
  card.appendChild(el("h3", { text: "Summary", style: { fontSize: "18px", marginBottom: "14px" } }));
  card.appendChild(bars);
  card.appendChild(legend);

  const totals = el("div", { class: "summary-totals" });
  totals.appendChild(el("div", { class: "t" }, [el("div", { class: "k", text: "total wall-clock" }), el("div", { class: "v tnum", text: `${total.toFixed(1)} s` })]));
  const inferMs = state.lastStat["infer:ms_per_view"];
  if (inferMs) totals.appendChild(el("div", { class: "t" }, [el("div", { class: "k", text: "neural ms/view" }), el("div", { class: "v tnum", text: `${inferMs.value.toFixed(1)} ${inferMs.unit}` })]));
  card.appendChild(totals);

  card.appendChild(el("div", { class: "server-ref" }, [
    el("div", { class: "server-ref-title", text: "Thesis benchmark (server, NVIDIA L4)" }),
    el("div", { class: "server-ref-stats" }, [
      el("b", { text: "Cycles" }), ` ${SERVER_REF.cycles_s_per_frame.toFixed(1)} s/view  ·  `,
      el("b", { text: "network" }), ` ${SERVER_REF.net_ms_per_frame_l4.toFixed(1)} ms/view (${SERVER_REF.speedup_x.toFixed(0)}× faster)  ·  `,
      el("b", { text: "incl. input preparation" }), ` ${(SERVER_REF.total_ms_per_frame_end_to_end / 1000).toFixed(2)} s/view (${SERVER_REF.speedup_x_end_to_end.toFixed(1)}× faster)`,
    ]),
    el("div", { class: "server-ref-note", text: "The laptop numbers above are this demo run, not the thesis benchmark." }),
  ]));
}

// ───────────────────────── SSE ─────────────────────────
function resetRunUI() {
  state.timings = {};
  state.lastStat = {};
  state.views = {};
  state.heroSet = false;
  state.bufFanView = null;
  state.gbufTileView = null;
  state.selectedView = null;
  state.latest = {};
  state.volumes = null;
  state.masksOverlayStackUrl = null;
  state.meshViewer?.reset();
  state.orbitDiagram?.reset();
  state.reconViewer?.dispose();
  state.reconViewerUrl = null;
  state.reconLoadedUrl = null;
  state.reconSelftestDone = false;
  state.reconCamerasOn = false;
  state.reconShowUnseen = false;
  state.reconMode = "ulr";
  state.meshBuffers = {};
  if (cards.recon) {
    cards.recon.buildBtn.style.display = "none";
    cards.recon.buildBtn.disabled = false; cards.recon.buildBtn.textContent = "Build 3-D view";
    cards.recon.camBtn.classList.remove("on");
    cards.recon.unseenBtn.classList.remove("on");
    cards.recon.compareOv.classList.remove("shown");
    cards.recon.hudTxt.textContent = "";
    cards.recon.hudExact.classList.remove("on");
    cards.recon.canvasWrap.classList.remove("no-webgl");
    cards.recon.selftestEl.textContent = "self-test: not run (add ?recon_selftest=1 to the URL)";
    Object.values(cards.recon.snapTiles).forEach((t) => t.classList.remove("active"));
    Object.entries(cards.recon.modeBtns).forEach(([m, b]) => { b.classList.toggle("active", m === "ulr"); b.style.display = ""; });
  }
  state.hadRun = true;
  showEmptyState();
  $("#timeline").style.display = "block";
  $("#resultsTitle").style.display = "none";
  $("#resultsHero").style.display = "none";
  $("#galleryHead").style.display = "none";
  $("#galleryBody").style.display = "none";
  $("#summaryCard").style.display = "none";
  for (const stage of ALL_STAGES) {
    setStageStatus(stage, "waiting");
    if (cards[stage]) {
      cards[stage].progressWrap.style.display = "none";
      cards[stage].errBox.classList.remove("shown");
      cards[stage].statTiles.innerHTML = "";
    }
  }
  $$(".stepper-seg").forEach(s => s.classList.remove("filled"));
  cards.masks.organChips.innerHTML = "";
  $("#meshTableBody").innerHTML = "";
  cards.gbuffer.filmstrip.innerHTML = "";
  cards.mesh.organRow.innerHTML = "";
  cards.mesh.organGroups = {};
  updateViewChips();
  Object.values(galleryTiles).forEach(t => { t.root.classList.add("empty"); t.root.classList.remove("loaded"); if (t.img) { t.img.remove(); t.img = null; } });
}

function connectEvents(runId, { replay, speed } = {}) {
  if (state.lastEventSource) state.lastEventSource.close();
  state.runId = runId;
  let url = `/api/runs/${runId}/events`;
  if (replay) url += `?replay=1&speed=${speed || 1}`;
  const es = new EventSource(url);
  state.lastEventSource = es;
  es.onmessage = (ev) => {
    let obj;
    try { obj = JSON.parse(ev.data); } catch { return; }
    handleEvent(obj);
  };
  es.onerror = () => { /* stream ended (finished run) or dropped — fine either way */ };
}

function handleEvent(ev) {
  switch (ev.type) {
    case "run_start": return onRunStart(ev);
    case "stage_start": return setStageStatus(ev.stage, "running");
    case "progress": return onProgress(ev);
    case "stat": return onStat(ev);
    case "artifact": return onArtifact(ev);
    case "log": return onLog(ev);
    case "stage_done": return onStageDone(ev);
    case "error": return onError(ev);
    case "run_done": return onRunDone(ev);
  }
}

function onRunStart(ev) {
  state.ctPath = ev.ct_path;
  if (!state.ctData || state.ctData.sid !== ev.subject) {
    state.ctData = { sid: ev.subject, checks: [], urls: {}, spriteMeta: { n: 64, w: 256, h: 256 } };
  }
}

function onProgress(ev) {
  setProgress(ev.stage, ev.done, ev.total, ev.label);
  if (ev.stage === "gbuffer") state.orbitDiagram?.setCurrent(ev.label);
}

const CT_STAT_FIELD = { shape: "shape", spacing: "spacing", extent: "extent", hu_range: "hu_range", orientation: "orientation", split: "split", load_time: "load_time" };
function onStat(ev) {
  state.lastStat[`${ev.stage}:${ev.key}`] = ev;
  if (ev.stage === "ct") {
    const field = CT_STAT_FIELD[ev.key];
    if (field) state.ctData[field] = ev.value;
    renderCtCardIfReady();
    return;
  }
  upsertStatTile(cards[ev.stage].statTiles, ev);
}

function renderCtCardIfReady() {
  if (state.ctData && state.ctData.shape) renderCtCard();
}

function onLog(ev) {
  if (ev.stage === "ct" || !cards[ev.stage]) return;
  const body = cards[ev.stage].detailsBody;
  let logWrap = body.querySelector(".log-lines");
  if (!logWrap) { logWrap = el("div", { class: "log-lines" }); body.insertBefore(logWrap, body.firstChild); }
  logWrap.appendChild(el("div", { text: ev.line }));
  logWrap.scrollTop = logWrap.scrollHeight;
}

function ensureView(view) {
  if (!state.views[view]) state.views[view] = { gbuffer: {}, buffers: {}, infer: {} };
  return state.views[view];
}

function onArtifact(ev) {
  const { stage, key, url, view, meta } = ev;
  if (stage === "ct") return onCtArtifact(key, url, meta);
  if (stage === "masks") return onMasksArtifact(key, url);
  if (stage === "mesh") return onMeshArtifact(key, url, meta);
  if (stage === "gbuffer") return onGbufferArtifact(key, url, view, meta);
  if (stage === "buffers") return onBuffersArtifact(key, url, view);
  if (stage === "infer") return onInferArtifact(key, url, view);
  if (stage === "recon") return onReconArtifact(key, url, meta);
}

function onReconArtifact(key, url, meta) {
  if (key !== "viewer") return; // the UI only needs the viewer manifest (PLAN_3D.md §4.3)
  state.reconViewerUrl = url;
  if (cards.recon) cards.recon.buildBtn.style.display = "none";
  maybeLoadReconViewer();
}

function onCtArtifact(key, url, meta) {
  if (!state.ctData) state.ctData = { sid: "", checks: [], urls: {}, spriteMeta: { n: 64, w: 256, h: 256 } };
  if (key === "axial_stack") {
    state.ctData.urls.axial_stack = url;
    if (meta) state.ctData.spriteMeta = { n: meta.n, w: meta.w, h: meta.h };
  } else if (key === "info") {
    fetch(url).then(r => r.json()).then(info => {
      Object.assign(state.ctData, {
        split: info.split, gt_available: info.gt_available, file_size: info.file_size, dtype: info.dtype,
        n_tissues: info.n_tissues, n_tissues_total: info.n_tissues_total, checks: info.checks,
      });
      renderCtCard();
    }).catch(() => {});
  } else {
    state.ctData.urls[key] = url;
  }
  renderCtCardIfReady();
}

function onMasksArtifact(key, url) {
  if (key === "volumes") {
    fetch(url).then(r => r.json()).then(volumes => renderMasksLists(volumes)).catch(() => {});
  } else if (key === "overlay_axial") {
    const img = cards.masks.ioRow.children[2].querySelector("img");
    img.src = url;
  } else if (key === "overlay_stack") {
    state.masksOverlayStackUrl = url;
  }
  if (cards.masks.ioRow.children[0]) {
    const leftImg = cards.masks.ioRow.children[0].querySelector("img");
    if (!leftImg.src && state.ctData?.urls?.axial) leftImg.src = state.ctData.urls.axial;
  }
}

function renderMasksLists(volumes) {
  state.volumes = volumes;
  for (const g of Object.values(cards.mesh.organGroups || {})) {
    const d = volumes[g.names[0]]?.display;
    if (d && g.sw) g.sw.style.background = d;
  }
  const chips = cards.masks.organChips;
  chips.innerHTML = "";
  // `present` already means volume_ml >= 1 (server-side), and `display` is the bright
  // same-hue pastel used for the regenerated overlays — chips must match those exactly.
  const entries = Object.entries(volumes).filter(([, v]) => v.present);
  const maxVol = Math.max(1, ...entries.map(([, v]) => v.volume_ml));
  entries.sort((a, b) => b[1].volume_ml - a[1].volume_ml);
  entries.forEach(([name, v], i) => {
    const swatchColor = v.display || v.color;
    const bar = el("span", { style: { background: swatchColor } });
    const chip = el("div", { class: "organ-chip" }, [
      el("div", { class: "row1" }, [el("span", { class: "swatch", style: { background: swatchColor } }), el("span", { class: "name", text: name.replace(/_/g, " ") }), el("span", { class: "vol", text: `${v.volume_ml >= 1000 ? (v.volume_ml / 1000).toFixed(2) + " L" : v.volume_ml.toFixed(0) + " mL"}` })]),
      el("div", { class: "vol-bar" }, [bar]),
    ]);
    chips.appendChild(chip);
    requestAnimationFrame(() => { bar.style.transform = `scaleX(${v.volume_ml / maxVol})`; });
  });

  const body = cards.masks.detailsBody;
  body.innerHTML = "";
  const table = el("table", { class: "details-table" }, [
    el("thead", {}, [el("tr", {}, [el("th", { text: "tissue" }), el("th", { text: "present" }), el("th", { text: "volume" })])]),
  ]);
  const tbody = el("tbody");
  for (const [name, v] of Object.entries(volumes)) {
    tbody.appendChild(el("tr", {}, [
      el("td", { text: name }),
      el("td", { text: v.present ? "yes" : "no" }),
      el("td", { text: v.present && v.volume_ml >= 1 ? `${v.volume_ml.toFixed(1)} mL` : "—" }),
    ]));
  }
  table.appendChild(tbody);
  body.appendChild(table);
}

function onMeshArtifact(key, url, meta) {
  if (key !== "organ" || !meta) return;
  // Cache the raw ArrayBuffer too — the recon (3-D view) card reuses it instead of
  // re-downloading the same web mesh (PLAN_3D.md §4.3/§4.5).
  fetch(meta.web).then(r => r.arrayBuffer()).then(buf => {
    state.meshViewer?.addOrgan(meta, buf);
    state.meshBuffers[meta.name] = buf;
  }).catch(() => {});
  addOrganChip(meta);
  const tbody = $("#meshTableBody");
  tbody.appendChild(el("tr", {}, [
    el("td", { text: meta.name }), el("td", { text: String(meta.verts) }), el("td", { text: String(meta.tris) }), el("td", { text: `${meta.ms.toFixed(0)} ms` }),
  ]));
  const leftImg = cards.mesh.ioRow.children[0]?.querySelector("img");
  if (leftImg && !leftImg.src && cards.masks.ioRow.children[2]) {
    const overlaySrc = cards.masks.ioRow.children[2].querySelector("img").src;
    if (overlaySrc) leftImg.src = overlaySrc;
  }
}

// Organ chips under the 3-D viewer — left/right pairs and the vertebrae grouped. Hover
// ghosts every other organ in the viewer and pops the organ's note; click pins it.
function organGroupKey(name) {
  if (name.startsWith("vertebrae_")) return "vertebrae";
  if (name.startsWith("lung_")) return "lungs";
  if (name.startsWith("autochthon_")) return "back muscles";
  if (name.startsWith("kidney_")) return "kidneys";
  if (name === "portal_vein_and_splenic_vein") return "portal & splenic vein";
  return name.replace(/_/g, " ");
}
function addOrganChip(meta) {
  const key = organGroupKey(meta.name);
  const groups = cards.mesh.organGroups;
  let g = groups[key];
  if (!g) {
    g = groups[key] = { key, names: [], tris: 0, ms: 0 };
    const vol = state.volumes?.[meta.name];
    g.countEl = el("span", { class: "n" });
    g.sw = el("span", { class: "swatch", style: { background: (vol && vol.display) || meta.color } });
    g.chip = el("div", { class: "mesh-organ" }, [g.sw, el("span", { text: key }), g.countEl]);
    attachNote(g.chip, () => organNoteFor(g), {
      accent: ACCENT_HEX.mesh,
      onShow: () => { g.chip.classList.add("active"); state.meshViewer?.setHighlight(g.names); },
      onHide: () => { g.chip.classList.remove("active"); state.meshViewer?.setHighlight(null); },
    });
    cards.mesh.organRow.appendChild(g.chip);
  }
  g.names.push(meta.name);
  g.tris += meta.tris;
  g.ms += meta.ms;
  g.countEl.textContent = g.names.length > 1 ? `×${g.names.length}` : "";
}
function organNoteFor(g) {
  const vol = g.names.reduce((acc, n) => acc + (state.volumes?.[n]?.volume_ml || 0), 0);
  const facts = el("div", { class: "spy-facts" }, [
    el("div", {}, [el("b", { text: vol >= 1000 ? `${(vol / 1000).toFixed(2)} L` : `${vol.toFixed(0)} mL` }), el("span", { text: "volume" })]),
    el("div", {}, [el("b", { text: g.tris.toLocaleString("en-US") }), el("span", { text: "triangles" })]),
    el("div", {}, [el("b", { text: `${Math.round(g.ms)} ms` }), el("span", { text: "to extract" })]),
  ]);
  const what = g.names.length > 1
    ? `${g.names.length} separate meshes: ${g.names.map(n => n.replace(/_/g, " ")).join(", ")}.`
    : "One watertight surface mesh from its TotalSegmentator mask.";
  return { kicker: "Organ", title: g.key.charAt(0).toUpperCase() + g.key.slice(1), what, why: organNote(g), extra: facts };
}

function onGbufferArtifact(key, url, view, meta) {
  if (key === "cameras") {
    fetch(url).then(r => r.json()).then(j => {
      for (const c of (j.cameras || [])) state.orbitDiagram?.setDone(c.view, { az: c.az, el: c.el });
    }).catch(() => {});
    return;
  }
  const v = ensureView(view);
  v.gbuffer[key] = url;
  state.latest.gbuffer = view;
  if (!state.selectedView || state.selectedView === view) setImg(cards.gbuffer.tiles2x2[key], url);
  if (key === "seg") {
    const thumb = el("img", { src: url, "data-view": view, title: `${view} — click to inspect` });
    thumb.onclick = () => selectView(view);
    cards.gbuffer.filmstrip.appendChild(thumb);
  }
  updateGalleryTile(view);
}

function onBuffersArtifact(key, url, view) {
  const v = ensureView(view);
  v.buffers[key] = url;
  state.latest.buffers = view;
  if (!state.selectedView || state.selectedView === view) showBuffersView(view);
  updateGalleryTile(view);
}

function onInferArtifact(key, url, view) {
  const v = ensureView(view);
  if (key === "output") {
    v.infer.output = url;
    state.latest.infer = view;
    if (!state.selectedView || state.selectedView === view) showInferView(view);
    maybeSetHero(view);
  } else if (key === "thumb") {
    v.infer.thumb = url;
  } else if (key === "gt") {
    v.infer.gt = url;
  }
  updateGalleryTile(view);
  updateReconSnapThumb(view);
}

function updateReconSnapThumb(view) {
  const c = cards.recon;
  const v = state.views[view];
  if (!c || !v?.infer) return;
  const tile = c.snapTiles[view];
  if (!tile) return;
  const src = v.infer.thumb || v.infer.output;
  if (!src) return;
  if (!tile.firstChild) tile.appendChild(el("img", {}));
  const img = tile.querySelector("img");
  if (img.getAttribute("src") !== src) img.src = src;
}

// ───────────────────────── selected view ─────────────────────────
// Clicking a camera on the orbit (or a filmstrip thumb) pins that view: the G-buffer
// tiles, the texture/glint tiles, both tensor stacks and the neural output all switch
// to it. "follow latest" hands the cards back to the incoming artifacts.
function viewLabel(view) {
  const g = VIEW_BY_NAME[view];
  return `<b>${view.slice(0, 3)}</b> · az ${g.az}°, el ${g.el >= 0 ? "+" : ""}${g.el}°`;
}
function selectView(view) {
  if (!state.views[view]) return;
  state.selectedView = view;
  state.orbitDiagram?.setSelected(view);
  showGbufferView(view);
  showBuffersView(view);
  showInferView(view);
  updateViewChips();
}
function clearSelection() {
  state.selectedView = null;
  state.orbitDiagram?.setSelected(null);
  if (state.latest.gbuffer) showGbufferView(state.latest.gbuffer);
  if (state.latest.buffers) showBuffersView(state.latest.buffers);
  if (state.latest.infer) showInferView(state.latest.infer);
  updateViewChips();
}
function updateViewChips() {
  for (const st of ["gbuffer", "buffers", "infer"]) {
    const c = cards[st];
    if (!c) continue;
    c.viewChip.classList.toggle("shown", !!state.selectedView);
    if (state.selectedView) c.viewChipText.innerHTML = `Showing ${viewLabel(state.selectedView)}`;
  }
  cards.gbuffer.filmstrip.querySelectorAll("img").forEach(t => t.classList.toggle("active", t.dataset.view === state.selectedView));
  // "Show in 3-D" only makes sense once the viewer is up and that view exists in it.
  if (cards.recon) cards.recon.show3dBtn.classList.toggle("shown", !!state.selectedView && !!state.reconViewer?.ready);
}
function setImg(img, url) {
  if (!img || !url) return;
  if (img.getAttribute("src") === url) { img.classList.add("shown"); return; }
  img.classList.remove("shown");
  img.onload = () => img.classList.add("shown");
  img.src = url;
}
function showGbufferView(view) {
  const v = state.views[view];
  if (!v) return;
  for (const k of ["seg", "depth", "normals", "segid"]) setImg(cards.gbuffer.tiles2x2[k], v.gbuffer[k]);
}
function showBuffersView(view) {
  const v = state.views[view];
  if (!v) return;
  for (const k of Object.keys(cards.buffers.tiles5)) setImg(cards.buffers.tiles5[k], v.buffers[k]);
  if (v.gbuffer.seg) cards.buffers.segThumbImg.src = v.gbuffer.seg;
  for (const f of TENSOR_LAYERS) {
    const src = v[f.stage][f.key];
    if (src) cards.buffers.fanImgs[f.key].src = src;
  }
}
function showInferView(view) {
  const v = state.views[view];
  if (!v) return;
  for (const f of TENSOR_LAYERS) {
    const src = v[f.stage][f.key];
    if (src) cards.infer.stackImgs[f.key].src = src;
  }
  const img = cards.infer.outputImg;
  if (v.infer.output && img.getAttribute("src") !== v.infer.output) {
    img.classList.remove("sharp");
    img.onload = () => requestAnimationFrame(() => img.classList.add("sharp"));
    img.src = v.infer.output;
  }
}

function onStageDone(ev) {
  state.timings[ev.stage] = ev.seconds;
  setStageStatus(ev.stage, "done", ev.seconds);
  if (ev.stage === "gbuffer") state.orbitDiagram?.setCurrent(null);
  if (cards[ev.stage]) cards[ev.stage].progressWrap.style.display = "none";
  if (["ct", "masks", "mesh", "gbuffer", "buffers"].includes(ev.stage)) {
    renderSummary();
  }
  if (ev.stage === "gbuffer" || ev.stage === "buffers") {
    $("#galleryHead").style.display = "flex";
    $("#galleryBody").style.display = "block";
  }
}

function onError(ev) {
  setStageStatus(ev.stage, "error");
  const box = cards[ev.stage]?.errBox;
  if (box) { box.textContent = `${ev.message}${ev.hint ? " — " + ev.hint : ""}`; box.classList.add("shown"); }
  toast(`${ev.stage}: ${ev.message}`, true);
  state.running = false;
  updateRunButtonState();
}

function onRunDone(ev) {
  state.running = false;
  updateRunButtonState();
  renderSummary();
  refreshDrawerList();
  // A run made before the `recon` stage existed (or whose recon stage failed) never sent
  // a `viewer` artifact — offer to build it now (replay path, PLAN_3D.md §4.6/§4.4).
  if (!state.reconViewerUrl && cards.recon) cards.recon.buildBtn.style.display = "";
}

// ───────────────────────── lightbox ─────────────────────────
function wireLightbox() {
  $("#lightboxClose").onclick = closeLightbox;
  $("#lightboxBackdrop").onclick = closeLightbox;
  $("#lbPrev").onclick = () => stepLightbox(-1);
  $("#lbNext").onclick = () => stepLightbox(1);
  $("#lightboxTabs").querySelectorAll("button").forEach(btn => {
    btn.onclick = () => { state.lightboxTab = btn.dataset.tab; renderLightboxTabs(); renderLightboxContent(); };
  });
  document.addEventListener("keydown", (e) => {
    if (!$("#lightbox").classList.contains("shown")) return;
    if (e.key === "Escape") closeLightbox();
    else if (e.key === "ArrowLeft") stepLightbox(-1);
    else if (e.key === "ArrowRight") stepLightbox(1);
  });
}
function openLightbox(view, tab) {
  state.lightboxView = view;
  state.lightboxTab = tab || "render";
  $("#lightbox").classList.add("shown");
  renderLightboxTabs();
  renderLightboxContent();
}
function closeLightbox() { $("#lightbox").classList.remove("shown"); }
function renderedViews() { return VIEW_GRID.filter(v => state.views[v.view]?.infer?.output).map(v => v.view); }
function stepLightbox(dir) {
  const list = renderedViews();
  if (!list.length) return;
  let idx = list.indexOf(state.lightboxView);
  idx = (idx + dir + list.length) % list.length;
  state.lightboxView = list[idx];
  renderLightboxTabs();
  renderLightboxContent();
}
function renderLightboxTabs() {
  const v = state.views[state.lightboxView];
  const hasGt = !!(v && v.infer && v.infer.gt);
  $("#lightboxTabs").querySelectorAll("button").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.tab === state.lightboxTab);
    if (btn.dataset.tab === "compare") btn.style.display = hasGt ? "" : "none";
  });
  if (state.lightboxTab === "compare" && !hasGt) { state.lightboxTab = "render"; renderLightboxTabs(); }
}
function renderLightboxContent() {
  const view = state.lightboxView;
  const v = state.views[view] || { gbuffer: {}, buffers: {}, infer: {} };
  const info = VIEW_BY_NAME[view];
  const content = $("#lightboxContent");
  content.innerHTML = "";
  if (state.lightboxTab === "render") {
    content.appendChild(el("img", { class: "lightbox-main-img", src: v.infer.output || "" }));
  } else if (state.lightboxTab === "inputs") {
    const grid = el("div", { class: "inputs-grid" });
    for (const t of INPUT_TABS) {
      const src = v[t.stage][t.key];
      grid.appendChild(el("div", { class: "cell" }, [src ? el("img", { src }) : el("div", {}), el("span", { class: "lbl", text: t.label })]));
    }
    content.appendChild(grid);
  } else if (state.lightboxTab === "compare") {
    const wrap = el("div", {});
    const cmp = el("div", { class: "compare-wrap" });
    const before = el("img", { src: v.infer.gt || "" });
    const after = el("img", { class: "compare-after", src: v.infer.output || "" });
    const handle = el("div", { class: "compare-handle", style: { left: "50%" } });
    cmp.appendChild(before); cmp.appendChild(after);
    cmp.appendChild(el("span", { class: "compare-tag left", text: "Cycles (reference)" }));
    cmp.appendChild(el("span", { class: "compare-tag right", text: "Neural (ours)" }));
    cmp.appendChild(handle);
    let dragging = false;
    const setPct = (clientX) => {
      const rect = cmp.getBoundingClientRect();
      const pct = clamp(((clientX - rect.left) / rect.width) * 100, 0, 100);
      handle.style.left = pct + "%";
      after.style.clipPath = `inset(0 0 0 ${pct}%)`;
    };
    cmp.addEventListener("pointerdown", (e) => {
      dragging = true; setPct(e.clientX);
      try { cmp.setPointerCapture(e.pointerId); } catch { /* Safari/edge cases can reject this; drag still works without capture */ }
    });
    cmp.addEventListener("pointermove", (e) => { if (dragging) setPct(e.clientX); });
    cmp.addEventListener("pointerup", () => { dragging = false; });
    setPct((cmp.getBoundingClientRect().left || 0) + 1e9); // will clamp to 100 until layout; corrected below
    requestAnimationFrame(() => { const r = cmp.getBoundingClientRect(); setPct(r.left + r.width / 2); });
    wrap.appendChild(cmp);
    wrap.appendChild(el("div", { class: "compare-note", text: "drag to compare · Cycles reference render vs. this network's output, same G-buffers" }));
    content.appendChild(wrap);
  }
  $("#lightboxCaption").innerHTML = `<span><b>${view}</b> · az ${info.az}°, el ${info.el >= 0 ? "+" : ""}${info.el}°</span><span>${renderedViews().indexOf(view) + 1} / ${renderedViews().length}</span>`;
}

// ───────────────────────── recent runs drawer ─────────────────────────
function wireDrawer() {
  $("#openDrawerBtn").onclick = openDrawer;
  $("#drawerClose").onclick = closeDrawer;
  $("#drawerBackdrop").onclick = closeDrawer;
}
function openDrawer() { $("#drawer").classList.add("shown"); $("#drawerBackdrop").classList.add("shown"); refreshDrawerList(); }
function closeDrawer() { $("#drawer").classList.remove("shown"); $("#drawerBackdrop").classList.remove("shown"); }

async function refreshDrawerList() {
  try {
    const r = await fetch("/api/runs");
    const j = await r.json();
    const list = $("#drawerList");
    list.innerHTML = "";
    for (const run of (j.runs || [])) {
      const row = el("div", { class: "run-row" }, [
        el("div", { class: "id", text: run.run_id }),
        el("div", { class: "meta", text: `${run.subject} · ${run.status} · ${run.seconds_total != null ? run.seconds_total.toFixed(1) + " s" : ""}` }),
      ]);
      const actions = el("div", { class: "actions" });
      const mk = (label, opts) => { const b = el("button", { text: label }); b.onclick = () => { closeDrawer(); resetRunUI(); connectEvents(run.run_id, opts); }; return b; };
      actions.appendChild(mk("Replay instant", {}));
      actions.appendChild(mk("1×", { replay: 1, speed: 1 }));
      actions.appendChild(mk("4×", { replay: 1, speed: 4 }));
      row.appendChild(actions);
      list.appendChild(row);
    }
  } catch (e) { /* ignore */ }
}

// ───────────────────────── toast ─────────────────────────
function toast(msg, bad) {
  const stack = $("#toastStack");
  const t = el("div", { class: "toast" + (bad ? " bad" : ""), text: msg });
  stack.appendChild(t);
  setTimeout(() => { t.style.transition = "opacity 300ms, transform 300ms"; t.style.opacity = "0"; t.style.transform = "translateY(8px)"; setTimeout(() => t.remove(), 320); }, 3800);
}
