// orbit.js — canvas-2D camera-orbit schematic: a fixed oblique 3-D-projected view of the
// "organ sphere" with 4 elevation rings (el -50/0/30/60) and the 20 camera positions on
// them, placed at their REAL az/el from gbuffers/cameras.json. Pending = hollow ring,
// finished = filled with the gbuffer accent (just-finished pulses once), only the
// current/last camera draws a line to the centre, back-hemisphere dots are fainter.
//
// The rAF loop is gated by IntersectionObserver + document.visibilitychange: it fully
// stops when the canvas is offscreen or the tab is hidden (see demo/PLAN.md — this is a
// 16 GB laptop that has frozen before from unbounded background work).

import { clamp, prefersReducedMotion } from "./ui.js";

export const AZ_LIST = [0, 72, 144, 216, 288];
export const EL_LIST = [-50, 0, 30, 60];

/** Canonical 20 view names in emission order, e.g. "v01_az+0_el-50". */
export function buildViewGrid() {
  const views = [];
  let n = 1;
  for (const az of AZ_LIST) {
    for (const el of EL_LIST) {
      const azS = `${az}`;
      const elS = (el >= 0 ? `+${el}` : `${el}`);
      const view = `v${String(n).padStart(2, "0")}_az+${azS}_el${elS}`;
      views.push({ view, az, el, n });
      n++;
    }
  }
  return views;
}

export const VIEW_GRID = buildViewGrid();

const ACCENT = "#7FB2E0";      // a slightly deeper sky than --gbuffer, reads better as a fill
const ACCENT_LINE = "#2E2A3B";
const PITCH = 20 * Math.PI / 180;   // fixed oblique viewing angle
const PULSE_MS = 650;

function rot(az, el, yaw, pitch) {
  const azR = (az * Math.PI) / 180, elR = (el * Math.PI) / 180;
  const x0 = Math.cos(elR) * Math.sin(azR);
  const y0 = Math.sin(elR);
  const z0 = Math.cos(elR) * Math.cos(azR);
  // yaw around Y (vertical)
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const x1 = x0 * cy + z0 * sy;
  const z1 = -x0 * sy + z0 * cy;
  // pitch around X (fixed oblique tilt)
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const y2 = y0 * cp - z1 * sp;
  const z2 = y0 * sp + z1 * cp;
  return { x: x1, y: y2, z: z2 }; // z>0 = far side (behind sphere), z<0 = near side
}

export class OrbitDiagram {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.done = new Set();          // view -> real {az, el} from cameras.json
    this.doneMeta = new Map();
    this.current = null;
    this.selected = null;           // view the user clicked (drives every card below)
    this.hover = null;
    this.onSelect = null;           // callback(view)
    this._dots = [];                // last drawn dot positions, for picking
    this.pulseAt = new Map();       // view -> performance.now() when it finished
    this._raf = null;
    this._visible = false;
    this._yaw = 0;

    this._ro = new ResizeObserver(() => { this._resize(); this._kick(); });
    this._ro.observe(canvas);
    this._resize();

    this._io = new IntersectionObserver((entries) => {
      this._visible = entries[0]?.isIntersecting ?? false;
      if (this._visible) this._kick();
    }, { threshold: 0.01 });
    this._io.observe(canvas);

    this._onVis = () => { if (!document.hidden) this._kick(); };
    document.addEventListener("visibilitychange", this._onVis);

    canvas.addEventListener("pointermove", (e) => {
      const v = this.pick(e.clientX, e.clientY);
      if (v !== this.hover) { this.hover = v; this.canvas.style.cursor = v ? "pointer" : "default"; this._kick(); }
    });
    canvas.addEventListener("pointerleave", () => { if (this.hover) { this.hover = null; this._kick(); } });
    canvas.addEventListener("click", (e) => {
      const v = this.pick(e.clientX, e.clientY);
      if (v && this.done.has(v) && this.onSelect) this.onSelect(v);
    });

    this._kick();
  }

  /** Nearest finished-or-pending camera dot within 13 px of a client point (near side wins ties). */
  pick(clientX, clientY) {
    const r = this.canvas.getBoundingClientRect();
    const x = clientX - r.left, y = clientY - r.top;
    let best = null, bestD = 13;
    for (const d of this._dots) {
      const dist = Math.hypot(d.x - x, d.y - y) + (d.z > 0.08 ? 3 : 0);
      if (dist < bestD) { bestD = dist; best = d.view; }
    }
    return best;
  }
  setSelected(viewName) { this.selected = viewName; this._kick(); }

  // setDone now takes the real {az, el} (falls back to nominal grid values if omitted).
  setDone(viewName, meta) {
    const wasNew = !this.done.has(viewName);
    this.done.add(viewName);
    if (meta) this.doneMeta.set(viewName, meta);
    if (wasNew) this.pulseAt.set(viewName, performance.now());
    this._kick();
  }
  setCurrent(viewName) { this.current = viewName; this._kick(); }
  reset() { this.done.clear(); this.doneMeta.clear(); this.pulseAt.clear(); this.current = null; this.selected = null; this.hover = null; this._kick(); }

  _resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, this.canvas.clientWidth), h = Math.max(1, this.canvas.clientHeight);
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = w; this.h = h;
  }

  /** (Re)start the render loop if it isn't already running and we're allowed to run. */
  _kick() {
    if (this._raf || document.hidden || !this._visible) { this._draw(performance.now()); return; }
    this._raf = requestAnimationFrame((t) => this._loop(t));
  }

  _loop(t) {
    this._raf = null;
    if (document.hidden || !this._visible) return; // fully stop — no more rAF scheduled
    if (!prefersReducedMotion()) this._yaw += 0.00012;
    const stillPulsing = [...this.pulseAt.values()].some(t0 => t - t0 < PULSE_MS);
    this._draw(t);
    if (stillPulsing || !prefersReducedMotion()) {
      this._raf = requestAnimationFrame((tt) => this._loop(tt));
    }
    // else: idle (no drift wanted, nothing pulsing) — loop stays stopped until setDone/
    // setCurrent/resize calls _kick() again.
  }

  _project(az, el, R) {
    const p = rot(az, el, this._yaw, PITCH);
    return { sx: p.x * R, sy: -p.y * R, z: p.z };
  }

  _ringPath(el, R, steps = 72) {
    const pts = [];
    for (let i = 0; i <= steps; i++) {
      const az = (360 * i) / steps;
      pts.push(this._project(az, el, R));
    }
    return pts;
  }

  _draw(t) {
    const ctx = this.ctx, w = this.w, h = this.h;
    if (!w || !h) return;
    ctx.clearRect(0, 0, w, h);
    const cx = w / 2, cy = h / 2 + h * 0.03;
    const R = Math.min(w, h) * 0.40;

    // central sphere — orthographic silhouette of a sphere is always a circle
    const sphereR = R * 0.46;
    const grad = ctx.createRadialGradient(cx - sphereR * 0.32, cy - sphereR * 0.38, sphereR * 0.08, cx, cy, sphereR);
    grad.addColorStop(0, "rgba(189,232,212,0.5)");
    grad.addColorStop(1, "rgba(189,232,212,0.08)");
    ctx.beginPath(); ctx.arc(cx, cy, sphereR, 0, Math.PI * 2);
    ctx.fillStyle = grad; ctx.fill();
    ctx.lineWidth = 1; ctx.strokeStyle = "rgba(107,101,128,0.16)"; ctx.stroke();

    // elevation rings, each an ellipse traced from sampled points; label at its rightmost point
    const labelSpots = [];
    for (const elv of EL_LIST) {
      const pts = this._ringPath(elv, R);
      ctx.beginPath();
      pts.forEach((p, i) => { const x = cx + p.sx, y = cy + p.sy; if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
      ctx.strokeStyle = "rgba(107,101,128,0.20)";
      ctx.lineWidth = 1;
      ctx.setLineDash([1, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
      // rightmost sampled point -> label anchor
      let best = pts[0];
      for (const p of pts) if (p.sx > best.sx) best = p;
      labelSpots.push({ elv, x: cx + best.sx, y: cy + best.sy });
    }
    // declutter labels vertically (min 14px apart), keep near their ring
    labelSpots.sort((a, b) => a.y - b.y);
    for (let i = 1; i < labelSpots.length; i++) {
      if (labelSpots[i].y - labelSpots[i - 1].y < 14) labelSpots[i].y = labelSpots[i - 1].y + 14;
    }
    ctx.font = "10.5px ui-monospace, 'SF Mono', monospace";
    ctx.fillStyle = "rgba(107,101,128,0.8)";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    for (const s of labelSpots) {
      ctx.fillText(`el ${s.elv >= 0 ? "+" : ""}${s.elv}°`, s.x + 8, s.y);
    }

    // camera dots, depth-sorted so the front hemisphere paints on top
    const dots = VIEW_GRID.map(({ view, az, el }) => {
      const meta = this.doneMeta.get(view);
      const a = meta ? meta.az : az, e = meta ? meta.el : el;
      const p = this._project(a, e, R);
      return { view, x: cx + p.sx, y: cy + p.sy, z: p.z };
    }).sort((d1, d2) => d2.z - d1.z); // far (z>0) first, near last (drawn on top)

    this._dots = dots;
    // the selected (else current/last-finished) camera gets a line to the centre
    const lineView = this.selected || this.current || (this.done.size ? [...this.done][this.done.size - 1] : null);
    const lineDot = dots.find(d => d.view === lineView);
    if (lineDot) {
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(lineDot.x, lineDot.y);
      ctx.strokeStyle = "rgba(46,42,59,0.4)";
      ctx.lineWidth = 1.3;
      ctx.stroke();
    }

    for (const d of dots) {
      const isDone = this.done.has(d.view);
      const isCurrent = this.current === d.view;
      const behind = d.z > 0.08;
      const fade = behind ? 0.38 : 1;
      const pulseT0 = this.pulseAt.get(d.view);
      const pulse = pulseT0 != null ? clamp((t - pulseT0) / PULSE_MS, 0, 1) : 1;

      if (pulseT0 != null && pulse < 1 && !prefersReducedMotion()) {
        ctx.beginPath();
        ctx.arc(d.x, d.y, 4 + pulse * 11, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(127,178,224,${(1 - pulse) * 0.6})`;
        ctx.lineWidth = 1.6;
        ctx.stroke();
      }

      const isSel = this.selected === d.view, isHover = this.hover === d.view;
      if (isSel) {
        ctx.beginPath();
        ctx.arc(d.x, d.y, 9.5, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(127,178,224,0.22)";
        ctx.fill();
        ctx.lineWidth = 1.6; ctx.strokeStyle = "#5E97CC"; ctx.stroke();
      }
      const r = isCurrent || isSel ? 5.4 : isHover ? 5.2 : 4.2;
      ctx.beginPath();
      ctx.arc(d.x, d.y, r, 0, Math.PI * 2);
      if (isDone) {
        ctx.globalAlpha = fade;
        ctx.fillStyle = ACCENT;
        ctx.fill();
        ctx.globalAlpha = 1;
      } else {
        ctx.lineWidth = 1.3;
        ctx.strokeStyle = behind ? "rgba(163,157,181,0.28)" : "rgba(163,157,181,0.55)";
        ctx.stroke();
      }
      if (isCurrent) {
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = ACCENT_LINE;
        ctx.stroke();
      }
    }

    // label for the hovered (else selected) camera, in a small pill
    const labelView = this.hover || this.selected;
    const ld = labelView && dots.find(d => d.view === labelView);
    if (ld) {
      const g = VIEW_GRID.find(v => v.view === labelView);
      const text = `${labelView.slice(0, 3)} · az ${g.az}° · el ${g.el >= 0 ? "+" : ""}${g.el}°` + (this.done.has(labelView) ? "" : " · pending");
      ctx.font = "600 11px -apple-system, system-ui, sans-serif";
      const tw = ctx.measureText(text).width + 16;
      let lx = ld.x + 12, ly = ld.y - 26;
      if (lx + tw > w - 6) lx = ld.x - 12 - tw;
      if (ly < 6) ly = ld.y + 12;
      ctx.fillStyle = "rgba(46,42,59,0.86)";
      ctx.beginPath(); ctx.roundRect ? ctx.roundRect(lx, ly, tw, 20, 10) : ctx.rect(lx, ly, tw, 20); ctx.fill();
      ctx.fillStyle = "#fff"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText(text, lx + 8, ly + 10.5);
    }
  }

  destroy() {
    if (this._raf) cancelAnimationFrame(this._raf);
    if (this._ro) this._ro.disconnect();
    if (this._io) this._io.disconnect();
    document.removeEventListener("visibilitychange", this._onVis);
  }
}
