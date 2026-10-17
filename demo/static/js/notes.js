// notes.js — the "spy gadget" info bubbles: a callout that pops out of whatever you hover
// (or click, to pin it), with a short note on what that thing is and why it matters in
// the thesis. One bubble element is reused for the whole page.
//
// Every number quoted here comes from the thesis evaluation (EVALUATION/v25_gbuf_glint,
// thesis_context.md) or from this pipeline's own code — nothing is illustrative.

import { el, prefersReducedMotion } from "./ui.js";

// ───────────────────────── content ─────────────────────────
// kicker = small label above the title; what = what it is; why = why it matters here.
export const NOTES = {
  // 17-channel input, one per layer group
  seg: {
    kicker: "G-buffer · 3 ch", title: "Seg RGB",
    what: "Every organ drawn in one flat colour by Blender's EEVEE rasteriser — no lights, no shadows.",
    why: "Tells the network which tissue it is looking at and where each silhouette runs, so it can learn every organ's own look.",
  },
  depth: {
    kicker: "G-buffer · 1 ch", title: "Depth",
    what: "Distance from the camera plane at every pixel, normalised per view.",
    why: "Shows the network where organs overlap, so it can place the occlusion and soft shadowing that the path tracer gets from light bouncing between surfaces.",
  },
  normals: {
    kicker: "G-buffer · 3 ch", title: "Normals",
    what: "Which way each surface faces, in world space.",
    why: "The main shading cue: with the fixed light rig it decides how bright each surface is and where the highlights sit.",
  },
  segid: {
    kicker: "G-buffer · 1 ch", title: "Seg ID",
    what: "The tissue class — one of 29 — as a number rather than a colour.",
    why: "A stable label that does not depend on the display colour, so organs with similar flat colours stay distinct.",
  },
  albedo: {
    kicker: "Texture · 3 ch", title: "Albedo",
    what: "The exact texture colour the path tracer uses at that surface point: the same 80 mm tissue tiles and box (triplanar) projection, looked up per pixel.",
    why: "Gives the network surface detail that is fixed to the anatomy, instead of asking it to invent that detail differently in every view.",
  },
  nmap: {
    kicker: "Texture · 3 ch", title: "Normal map",
    what: "The tissue's normal-map tile at that point — tiny bumps that bend the light.",
    why: "In the ground truth the vessels live only here: they appear as fine specular glints, not as colour. That is why the vessels were not learned from texture alone.",
  },
  rough: {
    kicker: "Texture · 1 ch", title: "Roughness",
    what: "How glossy the tissue is at that point, from the same texture tiles.",
    why: "Controls how sharp or spread-out the highlights are — wet, shiny serosa versus matte muscle.",
  },
  glint: {
    kicker: "Glints · 1 ch", title: "Vessel glints",
    what: "Our analytic glint model: GGX specular from the scene's five lights, using the normal bent by the normal map — the vessel sparkle, computed ahead of time.",
    why: "The key fix of the thesis. The earlier G-buffer model reproduced 0.6 % of the ground truth's vessel-streak energy; with this channel it reaches 97 % (30 test subjects).",
  },
  highlight: {
    kicker: "Glints · 1 ch", title: "Highlights",
    what: "The same specular model with the flat, unperturbed normal — the broad shine.",
    why: "The glints are expressed relative to it (log-scaled), so the network can tell vessel sparkle apart from ordinary shine.",
  },

  // G-buffer stage
  orbit: {
    kicker: "Camera manifold", title: "20 cameras around the body",
    what: "5 azimuths (every 72°) × 4 elevations (−50°, 0°, 30°, 60°), each with a small jitter seeded by the subject ID. Click a camera to inspect that view.",
    why: "Because the seed is the subject ID, these are exactly the training dataset's cameras: this Mac's G-buffers match the server-rendered s0006 to within half-float precision.",
  },
  gbufferSpeed: {
    kicker: "Why G-buffers", title: "Cheap on purpose",
    what: "All four buffers come from EEVEE, a rasteriser: about half a second per view on this Mac, for all four together.",
    why: "A single Cycles path-traced frame takes about 53 s on the thesis server. The network's job is to close that gap.",
  },

  // mesh stage
  meshMethod: {
    kicker: "Mesh extraction", title: "From masks to surfaces",
    what: "Each mask becomes a surface by marching cubes on a float32 field, then Laplacian smoothing — 20 passes, or 12 gentler ones for thin vessels — keeping only the largest connected piece.",
    why: "Casting the masks to uint8 left a voxel staircase that showed up as corduroy stripes on every organ in early renders; the float32 field removes it.",
  },
  meshStats: {
    kicker: "Two copies", title: "Full meshes for Blender, light ones for you",
    what: "Blender renders the full-resolution meshes; this viewer shows a copy decimated to at most 15 000 triangles per organ.",
    why: "The G-buffers — and so the network input — always come from the full meshes.",
  },

  // neural stage
  network: {
    kicker: "The generator", title: "pix2pixHD · 182.8 M parameters",
    what: "A global generator at 512 px plus a local enhancer at 1024 px, trained against multi-scale PatchGAN discriminators with perceptual (VGG), feature-matching and LSGAN losses, and no L1 term.",
    why: "It upsamples by resize + convolution instead of transposed convolution, which is what printed a 32-px checkerboard grid in the earlier models. Trained for 61 epochs on 4× NVIDIA L4.",
  },
  quality: {
    kicker: "Test set · 1 920 views", title: "How close to Cycles?",
    what: "Against the previous G-buffer model: PSNR 35.47 → 37.30 dB, SSIM 0.973 → 0.981, LPIPS 0.021 → 0.013, FID 13.2 → 6.3.",
    why: "It also stays as consistent across views as the ground truth itself: over 61 camera pairs the consistency gap is +0.00 dB.",
  },
  speed: {
    kicker: "Speed", title: "Milliseconds, not minutes",
    what: "About a third of a second per view on this MacBook's GPU; 98 ms on an NVIDIA L4.",
    why: "The path tracer needs about 53 s per frame on that server — 539× slower than the network alone, 22.6× including input preparation.",
  },

  // recon (3-D view) stage
  recon: {
    kicker: "3-D view", title: "The renders, wrapped onto the anatomy",
    what: "The 20 neural renders projected onto the CT organ meshes and blended by viewing angle (unstructured lumigraph). Exact at the 20 cameras; in between it interpolates.",
    why: "Leave-one-out, predicting each view from the other 19 scores 27.4 dB for the network's renders and 27.6 dB for Cycles' own — the network's views are as mutually consistent as the path tracer's. It is a visualisation of the 20 renders, not a new neural render.",
  },
  reconBlended: {
    kicker: "3-D view · mode", title: "Blended (ULR)",
    what: "The nearest 3 renders weighted by angle to the current view.",
    why: "Keeps sharpness and highlights, but between captures a glint can ghost — cross-fade between two cameras' versions of it.",
  },
  reconBaked: {
    kicker: "3-D view · mode", title: "Baked",
    what: "One colour per vertex, averaged over every view that saw it.",
    why: "Stable from any angle, but highlights and vessel glints get smeared into a soft, matte average.",
  },
  reconNearest: {
    kicker: "3-D view · mode", title: "Nearest",
    what: "The single angularly closest captured render, with no blending at all.",
    why: "Sharpest of the three, but the seams between cameras are visible as the view moves.",
  },
};

// Short thesis notes for specific organs (shown under the organ's own stats).
const THIN = ["aorta", "inferior_vena_cava", "superior_vena_cava", "portal_vein_and_splenic_vein", "esophagus"];
export function organNote(group) {
  const names = group.names;
  if (names.includes("liver")) return "The largest textured surface, and where the vessel glints are easiest to see: liver vessel streaks went from 1.2 % to 106 % of the ground truth with the glint channel.";
  if (names.some(n => n.startsWith("kidney"))) return "Kidney vessel streaks went from 13–34 % to 94–95 % of the ground truth with the glint channel.";
  if (names.some(n => THIN.includes(n))) return "A thin structure: smoothed more gently (12 × 0.15) so a two-voxel-wide vessel is not thinned or broken apart.";
  return "";
}

// ───────────────────────── the bubble ─────────────────────────
let bubble = null;       // {root, kicker, title, what, why, extra, close, tail}
let current = null;      // {target, pinned, onHide}
let hideTimer = null;

function ensureBubble() {
  if (bubble) return bubble;
  const kicker = el("div", { class: "spy-kicker" });
  const title = el("div", { class: "spy-title" });
  const what = el("p", { class: "spy-what" });
  const whyText = el("span", {});
  const why = el("div", { class: "spy-why" }, [el("span", { class: "spy-why-label", text: "In the thesis" }), whyText]);
  const extra = el("div", { class: "spy-extra" });
  const close = el("button", { class: "spy-close", "aria-label": "close", html: "&times;" });
  const tail = el("div", { class: "spy-tail" });
  const gadget = el("div", { class: "spy-gadget", html: '<svg viewBox="0 0 24 24"><path d="M12 2.5 L13.9 10.1 L21.5 12 L13.9 13.9 L12 21.5 L10.1 13.9 L2.5 12 L10.1 10.1 Z"/></svg>' });
  const root = el("div", { class: "spy-bubble", role: "tooltip" }, [
    tail, el("div", { class: "spy-head" }, [gadget, el("div", {}, [kicker, title])]), close, what, extra, why,
  ]);
  root.addEventListener("pointerenter", () => clearTimeout(hideTimer));
  root.addEventListener("pointerleave", () => { if (current && !current.pinned) scheduleHide(); });
  close.addEventListener("click", () => hideNote(true));
  document.body.appendChild(root);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") hideNote(true); });
  document.addEventListener("pointerdown", (e) => {
    if (!current || !current.pinned) return;
    if (root.contains(e.target) || (current.target instanceof Element && current.target.contains(e.target))) return;
    hideNote(true);
  });
  window.addEventListener("scroll", () => { if (current) place(); }, { passive: true });
  bubble = { root, kicker, title, what, why, whyText, extra, close, tail };
  return bubble;
}

function rectOf(target) {
  return typeof target.getBoundingClientRect === "function" ? target.getBoundingClientRect() : target.rect();
}

function place() {
  const b = bubble, r = rectOf(current.target);
  const bw = b.root.offsetWidth, bh = b.root.offsetHeight, gap = 14, pad = 12;
  const vw = window.innerWidth, vh = window.innerHeight;
  let side = "right";
  if (r.right + gap + bw > vw - pad) side = r.left - gap - bw > pad ? "left" : (r.top - gap - bh > pad ? "top" : "bottom");
  let x, y;
  if (side === "right" || side === "left") {
    x = side === "right" ? r.right + gap : r.left - gap - bw;
    y = Math.min(Math.max(r.top + r.height / 2 - bh / 2, pad), vh - bh - pad);
  } else {
    x = Math.min(Math.max(r.left + r.width / 2 - bw / 2, pad), vw - bw - pad);
    y = side === "top" ? r.top - gap - bh : r.bottom + gap;
  }
  b.root.style.left = `${Math.round(x)}px`;
  b.root.style.top = `${Math.round(y)}px`;
  b.root.dataset.side = side;
  // the tail points at the anchor's centre, clamped to the bubble's edge
  if (side === "right" || side === "left") {
    b.tail.style.top = `${Math.min(Math.max(r.top + r.height / 2 - y, 22), bh - 22)}px`;
    b.tail.style.left = "";
  } else {
    b.tail.style.left = `${Math.min(Math.max(r.left + r.width / 2 - x, 22), bw - 22)}px`;
    b.tail.style.top = "";
  }
}

/** Show `note` ({kicker,title,what,why,extra?}) next to `target` (an Element, or
 *  {rect: () => DOMRect} for things drawn on a canvas). */
export function showNote(target, note, { pinned = false, accent = "#C9B8F0", onHide = null } = {}) {
  const b = ensureBubble();
  clearTimeout(hideTimer);
  const same = current && current.target === target;
  if (current && !same && current.onHide) current.onHide();
  current = { target, pinned: pinned || (same && current.pinned), onHide };
  b.kicker.textContent = note.kicker || "";
  b.title.textContent = note.title || "";
  b.what.textContent = note.what || "";
  b.whyText.textContent = note.why || "";
  b.why.style.display = note.why ? "" : "none";
  b.extra.innerHTML = "";
  if (note.extra) b.extra.appendChild(note.extra);
  b.extra.style.display = note.extra ? "" : "none";
  b.root.style.setProperty("--spy-accent", accent);
  b.root.classList.toggle("pinned", current.pinned);
  if (!same) {
    b.root.classList.remove("shown");
    void b.root.offsetWidth;            // restart the pop animation
  }
  b.root.classList.add("shown");
  if (prefersReducedMotion()) b.root.classList.add("no-motion");
  place();
}

function scheduleHide() {
  clearTimeout(hideTimer);
  hideTimer = setTimeout(() => hideNote(false), 140);
}

export function hideNote(force) {
  if (!bubble || !current) return;
  if (current.pinned && !force) return;
  if (current.onHide) current.onHide();
  current = null;
  bubble.root.classList.remove("shown", "pinned");
}

/** Hover shows the note, click pins it (click again or ✕/Esc/outside closes). `note`
 *  may be a function so the content can be computed when it opens. */
export function attachNote(target, note, { accent, onShow, onHide } = {}) {
  target.classList.add("has-note");
  const get = () => (typeof note === "function" ? note() : note);
  const open = (pinned) => { onShow?.(); showNote(target, get(), { pinned, accent, onHide }); };
  target.addEventListener("pointerenter", () => { if (!current || !current.pinned) open(false); });
  target.addEventListener("pointerleave", () => { if (current && current.target === target && !current.pinned) scheduleHide(); });
  target.addEventListener("click", (e) => {
    e.stopPropagation();
    if (current && current.target === target && current.pinned) hideNote(true);
    else open(true);
  });
}

/** A small round "i" hotspot that opens a note — for places without a natural hover target. */
export function noteDot(note, opts = {}) {
  const dot = el("button", { class: "note-dot", "aria-label": "more info", html: '<svg viewBox="0 0 16 16"><circle cx="8" cy="4.2" r="1.3"/><rect x="6.9" y="6.6" width="2.2" height="6.2" rx="1.1"/></svg>' });
  attachNote(dot, note, opts);
  return dot;
}
