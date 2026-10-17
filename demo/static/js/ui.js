// ui.js — small generic helpers shared by app.js / meshviewer.js / orbit.js.
// No DOM assumptions beyond what's passed in; no state.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (k === "style" && typeof v === "object") {
      for (const [sk, sv] of Object.entries(v)) {
        if (sk.startsWith("--")) node.style.setProperty(sk, sv);
        else node.style[sk] = sv;
      }
    }
    else if (v === true) node.setAttribute(k, "");
    else node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c === null || c === undefined) continue;
    node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return node;
}

export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
export const lerp = (a, b, t) => a + (b - a) * t;

export const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);
export const easeOutBack = (t) => {
  const c1 = 1.70158, c3 = c1 + 1;
  return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
};

let _reduced = null;
export function prefersReducedMotion() {
  if (_reduced === null) {
    try { _reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches; }
    catch { _reduced = false; }
  }
  return _reduced;
}

export function formatMMSS(seconds) {
  seconds = Math.max(0, seconds);
  const m = Math.floor(seconds / 60);
  const s = (seconds - m * 60).toFixed(1);
  return `${m}:${s.padStart(4, "0")}`;
}

// Count a number up from its current displayed value to `to`. Cancels any animation
// already running on this node. Stores state on the node itself so repeated calls
// (e.g. stat updates arriving twice) don't stack rAF loops.
export function countUp(node, to, { decimals = 0, duration = 700, suffix = "", prefix = "" } = {}) {
  const from = node._countFrom ?? 0;
  if (node._countRaf) cancelAnimationFrame(node._countRaf);
  if (prefersReducedMotion() || duration <= 0) {
    node.textContent = prefix + to.toFixed(decimals) + suffix;
    node._countFrom = to;
    return;
  }
  const t0 = performance.now();
  const step = (now) => {
    const t = clamp((now - t0) / duration, 0, 1);
    const v = lerp(from, to, easeOutCubic(t));
    node.textContent = prefix + v.toFixed(decimals) + suffix;
    if (t < 1) {
      node._countRaf = requestAnimationFrame(step);
    } else {
      node._countFrom = to;
      node._countRaf = null;
    }
  };
  node._countRaf = requestAnimationFrame(step);
}

// A live mm:ss.s timer driven by wall clock (works the same under SSE replay speed-up,
// since the server paces event arrival — we just watch real elapsed time between events).
export function makeTimer(node) {
  let t0 = null, iv = null;
  return {
    start() {
      t0 = performance.now();
      if (iv) clearInterval(iv);
      node.textContent = "0:00.0";
      iv = setInterval(() => {
        node.textContent = formatMMSS((performance.now() - t0) / 1000);
      }, 97);
    },
    stop(finalSeconds) {
      if (iv) { clearInterval(iv); iv = null; }
      if (finalSeconds !== undefined) node.textContent = formatMMSS(finalSeconds);
    },
  };
}

// Small drawn check/warn/fail glyph (stroke-dashoffset animates in via CSS class .draw).
export function statusGlyph(status) {
  const svgns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgns, "svg");
  svg.setAttribute("viewBox", "0 0 20 20");
  svg.setAttribute("class", `glyph glyph-${status}`);
  const path = document.createElementNS(svgns, "path");
  if (status === "pass") {
    path.setAttribute("d", "M4 10.5 L8.2 14.5 L16 5.5");
  } else if (status === "warn") {
    path.setAttribute("d", "M10 4 L10 12 M10 15.4 L10 15.5");
  } else {
    path.setAttribute("d", "M5 5 L15 15 M15 5 L5 15");
  }
  path.setAttribute("class", "glyph-path");
  svg.appendChild(path);
  return svg;
}

export function debounce(fn, wait = 200) {
  let h = null;
  return (...args) => {
    clearTimeout(h);
    h = setTimeout(() => fn(...args), wait);
  };
}

export function hexToRgba(hex, a = 1) {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16), g = parseInt(h.slice(2, 4), 16), b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${a})`;
}

// Fade + rise entrance for a freshly-inserted node (honours reduced motion).
export function enter(node, { delay = 0, rise = 8 } = {}) {
  if (prefersReducedMotion()) return;
  node.style.opacity = "0";
  node.style.transform = `translateY(${rise}px)`;
  node.style.transition = "none";
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      node.style.transition = `opacity 480ms cubic-bezier(.2,.8,.2,1) ${delay}ms, transform 480ms cubic-bezier(.2,.8,.2,1) ${delay}ms`;
      node.style.opacity = "1";
      node.style.transform = "translateY(0)";
    });
  });
}
