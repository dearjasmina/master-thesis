// meshviewer.js — vanilla WebGL organ mesh viewer. No three.js (offline, no CDN, no
// build step — see demo/PLAN.md §1). Loads the little-endian .bin web meshes written by
// viz.write_web_mesh_bin: u32 nV, u32 nI, f32 pos[3nV] (mm, centred on the CT volume
// centre), f32 nrm[3nV], u32 idx[nI].
//
// Usage:
//   const mv = new MeshViewer(canvas);
//   mv.addOrgan(organMeta, arrayBuffer);   // organMeta = the `mesh organ` artifact's meta
//   mv.reset();

import { clamp, easeOutBack, easeOutCubic, prefersReducedMotion } from "./ui.js";

const VERT_SRC = `
  attribute vec3 aPos;
  attribute vec3 aNormal;
  uniform mat4 uMVP;
  uniform mat4 uModel;
  varying vec3 vNormal;
  varying vec3 vWorldPos;
  void main() {
    vec4 wp = uModel * vec4(aPos, 1.0);
    vWorldPos = wp.xyz;
    vNormal = mat3(uModel) * aNormal;
    gl_Position = uMVP * vec4(aPos, 1.0);
  }
`;

const FRAG_SRC = `
  precision mediump float;
  varying vec3 vNormal;
  varying vec3 vWorldPos;
  uniform vec3 uColor;
  uniform vec3 uLight1;
  uniform vec3 uLight2;
  uniform vec3 uEye;
  uniform float uOpacity;
  void main() {
    vec3 n = normalize(vNormal);
    if (!gl_FrontFacing) n = -n;
    vec3 l1 = normalize(uLight1);
    vec3 l2 = normalize(uLight2);
    float d1 = max(dot(n, l1), 0.0);
    float d2 = max(dot(n, l2), 0.0) * 0.45;
    vec3 v = normalize(uEye - vWorldPos);
    vec3 h = normalize(l1 + v);
    float spec = pow(max(dot(n, h), 0.0), 24.0) * 0.18;
    float ambient = 0.42;
    float rim = pow(1.0 - max(dot(n, v), 0.0), 2.2) * 0.16;
    vec3 col = uColor * (ambient + d1 * 0.62 + d2) + vec3(spec) + vec3(rim) * vec3(0.75, 0.68, 0.95);
    gl_FragColor = vec4(col, uOpacity);
  }
`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(s);
    gl.deleteShader(s);
    throw new Error("shader compile failed: " + info);
  }
  return s;
}

function mat4Identity() { return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]); }

function mat4Multiply(a, b) {
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      out[i * 4 + j] = a[0*4+j]*b[i*4+0] + a[1*4+j]*b[i*4+1] + a[2*4+j]*b[i*4+2] + a[3*4+j]*b[i*4+3];
    }
  }
  return out;
}

function mat4Perspective(fovy, aspect, near, far) {
  const f = 1.0 / Math.tan(fovy / 2), nf = 1 / (near - far);
  const out = new Float32Array(16);
  out[0] = f / aspect; out[5] = f; out[10] = (far + near) * nf; out[11] = -1;
  out[14] = 2 * far * near * nf;
  return out;
}

function mat4LookAt(eye, target, up) {
  const z = normalize3(sub3(eye, target));
  const x = normalize3(cross3(up, z));
  const y = cross3(z, x);
  const out = new Float32Array(16);
  out[0]=x[0]; out[4]=x[1]; out[8]=x[2];  out[12]=-dot3(x,eye);
  out[1]=y[0]; out[5]=y[1]; out[9]=y[2];  out[13]=-dot3(y,eye);
  out[2]=z[0]; out[6]=z[1]; out[10]=z[2]; out[14]=-dot3(z,eye);
  out[3]=0; out[7]=0; out[11]=0; out[15]=1;
  return out;
}
function sub3(a,b){return [a[0]-b[0],a[1]-b[1],a[2]-b[2]];}
function cross3(a,b){return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];}
function dot3(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
function normalize3(a){const l=Math.hypot(a[0],a[1],a[2])||1; return [a[0]/l,a[1]/l,a[2]/l];}

function mat4ScaleTranslate(scale, tx, ty, tz) {
  const out = mat4Identity();
  out[0] = scale; out[5] = scale; out[10] = scale;
  out[12] = tx; out[13] = ty; out[14] = tz;
  return out;
}

export class MeshViewer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl", { antialias: true, alpha: true, preserveDrawingBuffer: false })
             || canvas.getContext("experimental-webgl");
    if (!gl) { this._noWebGL = true; return; }
    this.gl = gl;
    // WebGL1 needs this extension for gl.UNSIGNED_INT index buffers (our decimated
    // meshes stay under 15k tris, so Uint16 would likely fit too, but don't rely on it).
    this._uintIdx = !!gl.getExtension("OES_element_index_uint");
    this.organs = [];        // {name,color:[r,g,b],vbo,nbo,ibo,count,addedAt,bboxMin,bboxMax}
    this.unionMin = null; this.unionMax = null;
    this.camYaw = 0.72; this.camPitch = 0.36;
    this.camDist = 260; this.targetDist = 260;
    this.camCenter = [0, 0, 0]; this.targetCenter = [0, 0, 0];
    this.reframeStart = null; this.reframeFrom = { dist: 260, center: [0, 0, 0] };
    this.reframeTo = { dist: 260, center: [0, 0, 0] };
    this._autoRotate = true;
    this._dragging = false;
    this._hlNames = null;     // Set of organ names to keep solid; null = everything solid
    this._hlFrom = 0; this._hlTo = 0; this._hlStart = null;   // animated 0..1 "ghost the rest" amount
    this._lastPointer = null;
    this._visible = false;

    this._initGL();
    this._bindInput();
    this._ro = new ResizeObserver(() => { this._resize(); this._kick(); });
    this._ro.observe(canvas);
    this._resize();

    // Stop rendering entirely when the canvas is scrolled offscreen or the tab is
    // hidden — a continuous WebGL rAF loop is real GPU/CPU cost we don't need paying
    // for a card the user isn't looking at (see demo/PLAN.md memory-safety notes).
    this._io = new IntersectionObserver((entries) => {
      this._visible = entries[0]?.isIntersecting ?? false;
      if (this._visible) this._kick();
    }, { threshold: 0.01 });
    this._io.observe(canvas);
    this._onVis = () => { if (!document.hidden) this._kick(); };
    document.addEventListener("visibilitychange", this._onVis);

    this._kick();
  }

  /** (Re)start the render loop if stopped, allowed to run, and something can change. */
  _kick() {
    if (this._raf || this._noWebGL || document.hidden || !this._visible) return;
    this._raf = requestAnimationFrame((t) => this._tick(t));
  }

  _isAnimating() {
    if (this._dragging) return true;
    if (this._autoRotate) return true;
    if (Math.abs(this.camDist - this.targetDist) > 0.4) return true;
    if (this.reframeStart !== null) return true;
    if (this._hlStart !== null) return true;
    const now = performance.now();
    for (const o of this.organs) {
      if (o.popDuration > 0 && now - o.addedAt < o.popDuration) return true;
    }
    return false;
  }

  _initGL() {
    const gl = this.gl;
    const vs = compile(gl, gl.VERTEX_SHADER, VERT_SRC);
    const fs = compile(gl, gl.FRAGMENT_SHADER, FRAG_SRC);
    const prog = gl.createProgram();
    gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    this.prog = prog;
    this.loc = {
      aPos: gl.getAttribLocation(prog, "aPos"),
      aNormal: gl.getAttribLocation(prog, "aNormal"),
      uMVP: gl.getUniformLocation(prog, "uMVP"),
      uModel: gl.getUniformLocation(prog, "uModel"),
      uColor: gl.getUniformLocation(prog, "uColor"),
      uLight1: gl.getUniformLocation(prog, "uLight1"),
      uLight2: gl.getUniformLocation(prog, "uLight2"),
      uEye: gl.getUniformLocation(prog, "uEye"),
      uOpacity: gl.getUniformLocation(prog, "uOpacity"),
    };
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(0, 0, 0, 0);
  }

  _bindInput() {
    const c = this.canvas;
    c.addEventListener("pointerdown", (e) => {
      this._dragging = true; this._autoRotate = false;
      this._lastPointer = [e.clientX, e.clientY];
      try { c.setPointerCapture(e.pointerId); } catch { /* Safari/edge cases can reject this; drag still works without capture */ }
      this._kick();
    });
    c.addEventListener("pointermove", (e) => {
      if (!this._dragging) return;
      const dx = e.clientX - this._lastPointer[0], dy = e.clientY - this._lastPointer[1];
      this._lastPointer = [e.clientX, e.clientY];
      this.camYaw -= dx * 0.008;
      this.camPitch = clamp(this.camPitch - dy * 0.008, -1.45, 1.45);
    });
    const stop = () => { this._dragging = false; };
    c.addEventListener("pointerup", stop);
    c.addEventListener("pointerleave", stop);
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.targetDist = clamp(this.targetDist * (1 + e.deltaY * 0.0012), 40, 2000);
      this._kick();
    }, { passive: false });
    c.addEventListener("dblclick", () => { this._autoRotate = true; this._kick(); });
  }

  _resize() {
    const gl = this.gl;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, this.canvas.clientWidth), h = Math.max(1, this.canvas.clientHeight);
    const pw = Math.round(w * dpr), ph = Math.round(h * dpr);
    if (this.canvas.width !== pw || this.canvas.height !== ph) {
      this.canvas.width = pw; this.canvas.height = ph;
    }
    gl.viewport(0, 0, pw, ph);
  }

  /** Keep `names` solid and ghost every other organ (null clears). Animated. */
  setHighlight(names) {
    const next = names && names.length ? new Set(names) : null;
    this._hlFrom = this._hlAmount(performance.now());
    this._hlTo = next ? 1 : 0;
    if (next) this._hlNames = next;          // keep the old set while fading back out
    this._hlStart = performance.now();
    this._kick();
  }

  _hlAmount(now) {
    if (this._hlStart === null) return this._hlTo;
    const t = clamp((now - this._hlStart) / (prefersReducedMotion() ? 1 : 260), 0, 1);
    const v = lerp(this._hlFrom, this._hlTo, easeOutCubic(t));
    if (t >= 1) { this._hlStart = null; if (this._hlTo === 0) this._hlNames = null; }
    return v;
  }

  reset() {
    const gl = this.gl;
    for (const o of this.organs) {
      gl.deleteBuffer(o.vbo); gl.deleteBuffer(o.nbo); gl.deleteBuffer(o.ibo);
    }
    this.organs = [];
    this.unionMin = null; this.unionMax = null;
    this.camDist = this.targetDist = 260;
    this.camCenter = [0, 0, 0]; this.targetCenter = [0, 0, 0];
    this._draw(performance.now()); // clear the canvas even if the loop is currently stopped
  }

  /** meta: {name, color:"#rrggbb", bbox_min:[3], bbox_max:[3]}; buf: ArrayBuffer of the .bin */
  addOrgan(meta, buf) {
    if (this._noWebGL) return;
    const gl = this.gl;
    const dv = new DataView(buf);
    const nV = dv.getUint32(0, true), nI = dv.getUint32(4, true);
    const posOff = 8, nrmOff = posOff + nV * 12, idxOff = nrmOff + nV * 12;
    const pos = new Float32Array(buf, posOff, nV * 3);
    const nrm = new Float32Array(buf, nrmOff, nV * 3);
    const idx32 = new Uint32Array(buf, idxOff, nI);
    const useUint32 = this._uintIdx && nV > 65535;
    const idx = useUint32 ? idx32 : new Uint16Array(idx32);

    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
    const nbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, nbo); gl.bufferData(gl.ARRAY_BUFFER, nrm, gl.STATIC_DRAW);
    const ibo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);

    const hex = (meta.color || "#BDE8D4").replace("#", "");
    const color = [parseInt(hex.slice(0,2),16)/255, parseInt(hex.slice(2,4),16)/255, parseInt(hex.slice(4,6),16)/255];

    this.organs.push({
      name: meta.name, color, vbo, nbo, ibo, count: nI,
      idxType: useUint32 ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT,
      addedAt: performance.now(), popDuration: prefersReducedMotion() ? 0 : 560,
    });

    const bmin = meta.bbox_min, bmax = meta.bbox_max;
    if (!this.unionMin) { this.unionMin = bmin.slice(); this.unionMax = bmax.slice(); }
    else {
      for (let i = 0; i < 3; i++) {
        this.unionMin[i] = Math.min(this.unionMin[i], bmin[i]);
        this.unionMax[i] = Math.max(this.unionMax[i], bmax[i]);
      }
    }
    this._reframe();
    this._kick();
  }

  _reframe() {
    if (!this.unionMin) return;
    const center = [0, 1, 2].map(i => (this.unionMin[i] + this.unionMax[i]) / 2);
    const extent = [0, 1, 2].map(i => this.unionMax[i] - this.unionMin[i]);
    const radius = Math.max(60, Math.hypot(extent[0], extent[1], extent[2]) / 2);
    const dist = radius / Math.tan((45 * Math.PI / 180) / 2) * 1.15;
    this.reframeFrom = { dist: this.camDist, center: this.camCenter.slice() };
    this.reframeTo = { dist, center };
    this.reframeStart = performance.now();
    this.reframeDur = prefersReducedMotion() ? 0 : 650;
  }

  _tick(now) {
    this._raf = null;
    if (this._noWebGL || document.hidden || !this._visible) return; // fully stop, no reschedule
    this._resize();
    if (this._autoRotate && !this._dragging) this.camYaw += 0.0014 * 16.6;
    this.camDist = lerp(this.camDist, this.targetDist, 0.12);

    if (this.reframeStart !== null) {
      const t = clamp((now - this.reframeStart) / Math.max(1, this.reframeDur), 0, 1);
      const e = easeOutCubic(t);
      this.camDist = lerp(this.reframeFrom.dist, this.reframeTo.dist, e);
      this.targetDist = this.reframeTo.dist;
      this.camCenter = [0,1,2].map(i => lerp(this.reframeFrom.center[i], this.reframeTo.center[i], e));
      if (t >= 1) this.reframeStart = null;
    }

    this._draw(now);
    if (this._isAnimating()) this._raf = requestAnimationFrame((t) => this._tick(t));
    // else: settled and idle — loop stays stopped until input, addOrgan or reset kicks it again.
  }

  _draw(now) {
    const gl = this.gl;
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!this.organs.length) return;

    const aspect = this.canvas.width / this.canvas.height;
    const proj = mat4Perspective(45 * Math.PI / 180, aspect, 1, 5000);
    const cx = this.camCenter[0] + this.camDist * Math.cos(this.camPitch) * Math.sin(this.camYaw);
    const cy = this.camCenter[1] + this.camDist * Math.sin(this.camPitch);
    const cz = this.camCenter[2] + this.camDist * Math.cos(this.camPitch) * Math.cos(this.camYaw);
    const eye = [cx, cy, cz];
    const view = mat4LookAt(eye, this.camCenter, [0, 1, 0]);
    const vp = mat4Multiply(proj, view);

    const l1 = normalize3([0.55, 0.85, 0.35]);
    const l2 = normalize3([-0.6, 0.25, -0.5]);

    gl.useProgram(this.prog);
    gl.uniform3fv(this.loc.uLight1, l1);
    gl.uniform3fv(this.loc.uLight2, l2);
    gl.uniform3fv(this.loc.uEye, eye);

    // Highlight: solid organs first with depth writes, then the ghosted rest on top
    // without depth writes, so the highlighted organ shows through them.
    const hl = this._hlAmount(now);
    const names = this._hlNames;
    const isSolid = (o) => !names || names.has(o.name);
    const ordered = names ? [...this.organs.filter(isSolid), ...this.organs.filter(o => !isSolid(o))] : this.organs;
    const GHOST = [0.80, 0.77, 0.88];
    for (const o of ordered) {
      let scale = 1, opacity = 1;
      if (o.popDuration > 0) {
        const t = clamp((now - o.addedAt) / o.popDuration, 0, 1);
        scale = t < 1 ? Math.max(0.001, easeOutBack(t)) : 1;
        opacity = clamp(t * 1.4, 0, 1);
      }
      let color = o.color;
      const ghost = names && !isSolid(o) ? hl : 0;
      if (ghost > 0) {
        color = [0, 1, 2].map(i => lerp(o.color[i], GHOST[i], ghost * 0.8));
        opacity *= lerp(1, 0.13, ghost);
      }
      gl.depthMask(ghost < 0.5);
      const model = mat4ScaleTranslate(scale, 0, 0, 0);
      const mvp = mat4Multiply(vp, model);

      gl.bindBuffer(gl.ARRAY_BUFFER, o.vbo);
      gl.enableVertexAttribArray(this.loc.aPos);
      gl.vertexAttribPointer(this.loc.aPos, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, o.nbo);
      gl.enableVertexAttribArray(this.loc.aNormal);
      gl.vertexAttribPointer(this.loc.aNormal, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, o.ibo);

      gl.uniformMatrix4fv(this.loc.uMVP, false, mvp);
      gl.uniformMatrix4fv(this.loc.uModel, false, model);
      gl.uniform3fv(this.loc.uColor, color);
      gl.uniform1f(this.loc.uOpacity, opacity);
      gl.drawElements(gl.TRIANGLES, o.count, o.idxType, 0);
    }
    gl.depthMask(true);
  }

  destroy() {
    if (this._raf) cancelAnimationFrame(this._raf);
    if (this._ro) this._ro.disconnect();
    if (this._io) this._io.disconnect();
    document.removeEventListener("visibilitychange", this._onVis);
  }
}

function lerp(a, b, t) { return a + (b - a) * t; }
