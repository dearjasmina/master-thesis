// reconviewer.js — WebGL2 3-D view: the 20 neural renders projected onto the organ
// meshes and blended by viewing angle ("view-dependent projective texturing" / the
// unstructured lumigraph, Buehler et al. 2001), with a per-vertex "baked" fallback and
// a "nearest" mode. Vanilla WebGL2, no libraries, works offline. See demo/PLAN_3D.md §4.5
// — the maths mirrors demo/research_3d/fuse_eval.py::blend and webref.py exactly (same
// weights, same visibility test), which is what the self-test checks against.
//
// API: new ReconViewer(canvas); await load(viewerJsonUrl, {meshBuffers?, forceNoFloat?});
// setMode("ulr"|"baked"|"nearest"); snapTo(i); showCameras(bool); dispose(); selftest(url).

import { clamp, easeOutCubic, prefersReducedMotion } from "./ui.js";

const NCAM = 20;

// ── small vec3 helpers (kept local — no shared 3-D math module in this repo) ──────────
const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm3 = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

/** Orbit camera around `target`, Z-up (the 20 capture cameras orbit world Z — see
 *  PLAN_3D.md §4.5), matching the generator's spherical convention. Returns
 *  {R: [9] row-major, t: [3], center: [3]}. */
function orbitCamera(azDeg, elDeg, d, target) {
  const az = (azDeg * Math.PI) / 180, el = (elDeg * Math.PI) / 180;
  const C = [
    target[0] + d * Math.cos(el) * Math.cos(az),
    target[1] + d * Math.cos(el) * Math.sin(az),
    target[2] + d * Math.sin(el),
  ];
  const f = norm3(sub3(target, C));
  let r = cross3(f, [0, 0, 1]);
  const rl = Math.hypot(r[0], r[1], r[2]);
  r = rl > 1e-6 ? [r[0] / rl, r[1] / rl, r[2] / rl] : [-Math.sin(az), Math.cos(az), 0];
  const u = cross3(r, f);
  const R = [r[0], r[1], r[2], -u[0], -u[1], -u[2], f[0], f[1], f[2]];
  const t = [-dot3(R.slice(0, 3), C), -dot3(R.slice(3, 6), C), -dot3(R.slice(6, 9), C)];
  return { R, t, center: C };
}
function azElDistFromCenter(center, target) {
  const d3 = sub3(center, target);
  const d = Math.hypot(d3[0], d3[1], d3[2]) || 1;
  return { az: (Math.atan2(d3[1], d3[0]) * 180) / Math.PI,
           el: (Math.asin(clamp(d3[2] / d, -1, 1)) * 180) / Math.PI, d };
}

// ── shaders ─────────────────────────────────────────────────────────────────────────
const XFORM = `
vec3 Xc = aPos * uCurR + uCurT;
vec2 px = uCurK.xy * Xc.xy / Xc.z + uCurK.zw;
float zn = 2.0 * (Xc.z - uNF.x) / (uNF.y - uNF.x) - 1.0;`;

const VIS_VS = `#version 300 es
layout(location=0) in vec3 aPos;
uniform mat3 uCurR; uniform vec3 uCurT; uniform vec4 uCurK; uniform vec2 uNF; uniform vec2 uViewport;
out float vZ;
void main() {
  ${XFORM}
  vZ = Xc.z;
  vec2 ndc = 2.0 * px / uViewport - 1.0;
  gl_Position = vec4(ndc * Xc.z, zn * Xc.z, Xc.z);
}`;
const VIS_FS = `#version 300 es
precision highp float;
in float vZ;
out float o;
void main() { o = vZ; }`;

const MAIN_VS = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
layout(location=2) in vec4 aBake;
uniform mat3 uCurR; uniform vec3 uCurT; uniform vec4 uCurK; uniform vec2 uNF; uniform vec2 uViewport;
out vec3 vPos; out vec3 vNrm; out vec4 vBake;
void main() {
  vPos = aPos; vNrm = aNrm; vBake = aBake;
  ${XFORM}
  vec2 ndc = vec2(2.0 * px.x / uViewport.x - 1.0, 1.0 - 2.0 * px.y / uViewport.y);
  gl_Position = vec4(ndc * Xc.z, zn * Xc.z, Xc.z);
}`;
// The GPU twin of research_3d/fuse_eval.py::blend + webref.py hole handling (PLAN_3D.md §4.5).
// Extra over the plan's reference shader (see the white-speckles fix): vBake.rgb for a
// never-seen (vBake.a≈0) vertex is now demo/recon.py's geodesic infill, not a flat colour
// — dimmed slightly (x0.92) so it reads as surface-in-shadow rather than a lit patch, and
// an optional uShowUnseen tint (flat lilac) so a viewer can ask "what did the 20 cameras
// never see" honestly instead of it being invisible.
const MAIN_FS = `#version 300 es
precision highp float;
precision highp sampler2DArray;
#define NCAM_ ${NCAM}
uniform mat3 uR[NCAM_]; uniform vec3 uT[NCAM_]; uniform vec4 uK[NCAM_]; uniform vec3 uC[NCAM_];
uniform sampler2DArray uImg; uniform sampler2DArray uVis; uniform vec2 uImgSize;
uniform vec3 uEye; uniform int uMode; uniform float uTol; uniform int uShowUnseen;
in vec3 vPos; in vec3 vNrm; in vec4 vBake; out vec4 frag;
void main() {
  vec3 fallback = vBake.rgb * (vBake.a < 0.5 ? 0.92 : 1.0);
  vec3 result; bool done = false;
  if (uMode == 1) { result = fallback; done = true; }
  if (!done) {
    vec3 n = normalize(vNrm), dEye = normalize(uEye - vPos);
    float th[4]; int id[4]; vec2 uv4[4]; float cs[4];
    for (int k = 0; k < 4; k++) { th[k] = 1e9; id[k] = -1; }
    for (int i = 0; i < NCAM_; i++) {
      vec3 Xc = vPos * uR[i] + uT[i];
      if (Xc.z < 1.0) continue;
      vec2 uv = (uK[i].xy * Xc.xy / Xc.z + uK[i].zw) / uImgSize;
      if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) continue;
      if (abs(texture(uVis, vec3(uv, float(i))).r - Xc.z) > uTol) continue;
      vec3 dC = normalize(uC[i] - vPos);
      float a = acos(clamp(dot(dC, dEye), -1.0, 1.0)), c = max(dot(dC, n), 0.0);
      for (int k = 0; k < 4; k++) if (a < th[k]) {
        for (int m = 3; m > k; m--) { th[m] = th[m-1]; id[m] = id[m-1]; uv4[m] = uv4[m-1]; cs[m] = cs[m-1]; }
        th[k] = a; id[k] = i; uv4[k] = uv; cs[k] = c; break;
      }
    }
    if (id[0] < 0) { result = fallback; done = true; }
    if (!done) {
      vec3 c0 = texture(uImg, vec3(uv4[0], float(id[0]))).rgb;
      if (uMode == 2) { result = c0; done = true; }
      if (!done) {
        float th4 = id[3] >= 0 ? th[3] : 3.14159265;
        vec3 acc = vec3(0.0); float ws = 0.0;
        for (int k = 0; k < 3; k++) {
          if (id[k] < 0) break;
          float w = clamp(1.0 - th[k] / th4, 0.0, 1.0) / (th[k] + 1e-3) * max(cs[k], 0.05);
          acc += w * texture(uImg, vec3(uv4[k], float(id[k]))).rgb; ws += w;
        }
        result = ws > 0.0 ? acc / ws : c0;
      }
    }
  }
  if (uShowUnseen == 1) {
    float unseen = 1.0 - vBake.a;
    result = mix(result, vec3(0.79, 0.72, 0.94), unseen * 0.55);
  }
  frag = vec4(result, 1.0);
}`;
const LINE_VS = `#version 300 es
layout(location=0) in vec3 aPos;
uniform mat3 uCurR; uniform vec3 uCurT; uniform vec4 uCurK; uniform vec2 uNF; uniform vec2 uViewport;
void main() {
  ${XFORM}
  vec2 ndc = vec2(2.0 * px.x / uViewport.x - 1.0, 1.0 - 2.0 * px.y / uViewport.y);
  gl_Position = vec4(ndc * Xc.z, zn * Xc.z, Xc.z);
}`;
const LINE_FS = `#version 300 es
precision mediump float;
uniform vec4 uColor;
out vec4 frag;
void main() { frag = uColor; }`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(s);
    gl.deleteShader(s);
    throw new Error("recon shader compile failed: " + info + "\n" + src);
  }
  return s;
}
function link(gl, vsSrc, fsSrc) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error("recon program link failed: " + gl.getProgramInfoLog(p));
  return p;
}

const EL_COLOR = { "-50": [0.949, 0.612, 0.639], "0": [0.498, 0.698, 0.878], "30": [0.741, 0.910, 0.831], "60": [0.914, 0.776, 0.949] };

export class ReconViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = null;
    this.ready = false;
    this.noWebGL2 = false;
    this.noFloat = false;
    this.mode = "ulr";           // "ulr" | "baked" | "nearest"
    this.showFrustums = false;
    this.showUnseen = false;     // tint never-seen-by-a-camera surface flat lilac
    this.az = 0; this.el = 0; this.d = 1500;
    this.target = [0, 0, 0];
    this._exactCam = null;       // {R,t,center} when snapped and undragged
    this._snapAnim = null;       // {from:{az,el,d}, to:{az,el,d}, t0, exact}
    this._dragging = false;
    this._lastPointer = null;
    this._visible = false;
    this.onHud = null;           // (text, exact) => void
    this._organs = [];           // {vao stuff per program, count, ...}
    this._raf = null;
    this._destroyed = false;

    this._ro = new ResizeObserver(() => { this._resizeCanvas(); this._kick(); });
    this._ro.observe(canvas);
    this._io = new IntersectionObserver((entries) => {
      this._visible = entries[0]?.isIntersecting ?? false;
      if (this._visible) this._kick();
    }, { threshold: 0.01 });
    this._io.observe(canvas);
    this._onVis = () => { if (!document.hidden) this._kick(); };
    document.addEventListener("visibilitychange", this._onVis);
    this._bindInput();
  }

  // ── loading ───────────────────────────────────────────────────────────────────────
  async load(viewerJsonUrl, { meshBuffers = {}, forceNoFloat = false } = {}) {
    this.ready = false;
    const gl = this.canvas.getContext("webgl2", { antialias: true, alpha: false, preserveDrawingBuffer: true });
    if (!gl) { this.noWebGL2 = true; return this; }
    this.gl = gl;
    gl.getExtension("EXT_color_buffer_float");
    this.noFloat = forceNoFloat || !gl.getExtension("EXT_color_buffer_float");
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

    const viewer = await (await fetch(viewerJsonUrl)).json();
    this._base = viewerJsonUrl.replace(/\/recon\/viewer\.json.*$/, "");
    const [manifest, camDoc] = await Promise.all([
      fetch(viewer.manifest).then((r) => r.json()),
      fetch(viewer.cameras).then((r) => r.json()),
    ]);
    const bakeBuf = await (await fetch(viewer.bake)).arrayBuffer();

    this.nearMm = camDoc.near_mm ?? 500; this.farMm = camDoc.far_mm ?? 3000;
    this.tolMm = camDoc.tol_mm ?? 3.0;
    this.target = camDoc.organ_center_web || [0, 0, 0];
    this.cams = camDoc.cameras;
    this.v02Index = Math.max(0, this.cams.findIndex((c) => c.view.startsWith("v02_")));
    const dists = this.cams.map((c) => Math.hypot(...sub3(c.center, this.target)));
    this.meanDist = dists.reduce((a, b) => a + b, 0) / dists.length;
    this.snapFx = this.cams[this.v02Index].fx;

    this._buildPrograms();
    await this._buildOrgans(manifest, bakeBuf, meshBuffers);
    await this._buildPhotoArray(this.cams);
    if (!this.noFloat) this._buildVisibilityArray();
    this._buildFrustumLines(this.cams);
    this._buildCamUniforms(this.cams);

    this.ready = true;
    this.snapTo(this.v02Index, { instant: true });
    this._resizeCanvas();
    this._kick();
    return this;
  }

  _buildPrograms() {
    const gl = this.gl;
    this.progMain = link(gl, MAIN_VS, MAIN_FS);
    this.progLine = link(gl, LINE_VS, LINE_FS);
    this.uMain = this._locs(this.progMain, ["uCurR", "uCurT", "uCurK", "uNF", "uViewport",
      "uImg", "uVis", "uImgSize", "uEye", "uMode", "uTol", "uShowUnseen"]);
    this.uMain.uR = gl.getUniformLocation(this.progMain, "uR[0]");
    this.uMain.uT = gl.getUniformLocation(this.progMain, "uT[0]");
    this.uMain.uK = gl.getUniformLocation(this.progMain, "uK[0]");
    this.uMain.uC = gl.getUniformLocation(this.progMain, "uC[0]");
    this.uLine = this._locs(this.progLine, ["uCurR", "uCurT", "uCurK", "uNF", "uViewport", "uColor"]);
    if (!this.noFloat) {
      this.progVis = link(gl, VIS_VS, VIS_FS);
      this.uVisP = this._locs(this.progVis, ["uCurR", "uCurT", "uCurK", "uNF", "uViewport"]);
    }
  }
  _locs(prog, names) {
    const o = {};
    for (const n of names) o[n] = this.gl.getUniformLocation(prog, n);
    return o;
  }

  async _buildOrgans(manifest, bakeBuf, meshBuffers) {
    const gl = this.gl;
    const dvBake = new DataView(bakeBuf);
    const nVTotal = dvBake.getUint32(0, true);
    const bakeBytes = new Uint8Array(bakeBuf, 4, nVTotal * 4);
    this._organs.forEach((o) => this._freeOrgan(o));
    this._organs = [];
    // Fetch every organ's buffer first (order across these is irrelevant — Promise.all
    // resolves them keyed to `manifest.organs`' own order regardless of network timing),
    // THEN walk them in a single synchronous loop to accumulate `off`. Interleaving the
    // fetch and the offset accumulation in one async map (as this used to) is a race: an
    // organ whose buffer had to be fetched (not cache-hit) would resume after organs
    // later in the manifest had already advanced `off`, slicing bake.bin at the wrong
    // vertex range and painting organs with each other's colours.
    const loaded = await Promise.all(manifest.organs.map(async (meta) => {
      let buf = meshBuffers[meta.name];
      if (!buf) buf = await (await fetch(meta.web)).arrayBuffer();
      return { meta, buf };
    }));
    let off = 0;
    for (const { meta, buf } of loaded) {
      const dv = new DataView(buf);
      const nV = dv.getUint32(0, true), nI = dv.getUint32(4, true);
      const pos = new Float32Array(buf, 8, nV * 3);
      const nrm = new Float32Array(buf, 8 + 12 * nV, nV * 3);
      const idx = new Uint32Array(buf, 8 + 24 * nV, nI);
      const bakeSlice = bakeBytes.subarray(off * 4, (off + nV) * 4);
      off += nV;

      const posBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
      const nrmBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, nrmBuf); gl.bufferData(gl.ARRAY_BUFFER, nrm, gl.STATIC_DRAW);
      const bakeBufGl = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, bakeBufGl); gl.bufferData(gl.ARRAY_BUFFER, bakeSlice, gl.STATIC_DRAW);
      const idxBuf = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);

      const vaoMain = gl.createVertexArray();
      gl.bindVertexArray(vaoMain);
      gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, nrmBuf); gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, bakeBufGl); gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 4, gl.UNSIGNED_BYTE, true, 0, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
      gl.bindVertexArray(null);

      let vaoVis = null;
      if (!this.noFloat) {
        vaoVis = gl.createVertexArray();
        gl.bindVertexArray(vaoVis);
        gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
        gl.bindVertexArray(null);
      }
      this._organs.push({ name: meta.name, posBuf, nrmBuf, bakeBufGl, idxBuf, vaoMain, vaoVis, count: nI });
    }
  }
  _freeOrgan(o) {
    const gl = this.gl;
    gl.deleteBuffer(o.posBuf); gl.deleteBuffer(o.nrmBuf); gl.deleteBuffer(o.bakeBufGl); gl.deleteBuffer(o.idxBuf);
    gl.deleteVertexArray(o.vaoMain); if (o.vaoVis) gl.deleteVertexArray(o.vaoVis);
  }

  async _buildPhotoArray(cams) {
    const gl = this.gl;
    if (this.texImg) gl.deleteTexture(this.texImg);
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex);
    gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, gl.RGBA8, 1024, 1024, NCAM);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const bitmaps = await Promise.all(cams.map((c) =>
      fetch(c.image).then((r) => r.blob()).then((b) => createImageBitmap(b, { colorSpaceConversion: "none" }))));
    for (let i = 0; i < bitmaps.length; i++) {
      gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, i, 1024, 1024, 1, gl.RGBA, gl.UNSIGNED_BYTE, bitmaps[i]);
      bitmaps[i].close?.();
    }
    this.texImg = tex;
  }

  _buildVisibilityArray() {
    const gl = this.gl;
    if (this.texVis) gl.deleteTexture(this.texVis);
    if (this.visFbo) gl.deleteFramebuffer(this.visFbo);
    if (this.visDepth) gl.deleteRenderbuffer(this.visDepth);
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex);
    gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, gl.R32F, 1024, 1024, NCAM);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.texVis = tex;

    const depth = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, depth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, 1024, 1024);
    this.visDepth = depth;

    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth);
    this.visFbo = fbo;

    gl.viewport(0, 0, 1024, 1024);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LESS); gl.disable(gl.CULL_FACE);
    gl.useProgram(this.progVis);
    for (let i = 0; i < this.cams.length; i++) {
      gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, tex, 0, i);
      gl.clearBufferfv(gl.COLOR, 0, new Float32Array([1e9, 0, 0, 0]));
      gl.clear(gl.DEPTH_BUFFER_BIT);
      this._setCurCam(this.uVisP, this.cams[i], 1024, 1024);
      for (const o of this._organs) {
        gl.bindVertexArray(o.vaoVis);
        gl.drawElements(gl.TRIANGLES, o.count, gl.UNSIGNED_INT, 0);
      }
    }
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  _buildCamUniforms(cams) {
    const R = new Float32Array(NCAM * 9), T = new Float32Array(NCAM * 3);
    const K = new Float32Array(NCAM * 4), C = new Float32Array(NCAM * 3);
    cams.forEach((c, i) => {
      const r = [].concat(...c.R);
      R.set(r, i * 9); T.set(c.t, i * 3); C.set(c.center, i * 3);
      K.set([c.fx, c.fy, c.cx, c.cy], i * 4);
    });
    this._camR = R; this._camT = T; this._camK = K; this._camC = C;
  }

  _buildFrustumLines(cams) {
    const gl = this.gl;
    if (this.frustBuf) gl.deleteBuffer(this.frustBuf);
    const L = 55; // mm, visual length only
    const pts = [];
    this._frustSeg = []; // {start,count,color,view}
    for (const c of cams) {
      const Rrows = [c.R[0], c.R[1], c.R[2]];
      const worldDir = (ld) => [0, 1, 2].map((k) => ld[0] * Rrows[0][k] + ld[1] * Rrows[1][k] + ld[2] * Rrows[2][k]);
      const corners = [[0, 0], [1024, 0], [0, 1024], [1024, 1024]].map(([px, py]) => {
        const ld = norm3([(px - c.cx) / c.fx, (py - c.cy) / c.fy, 1]);
        const wd = worldDir(ld);
        return [0, 1, 2].map((k) => c.center[k] + wd[k] * L);
      });
      const start = pts.length / 3;
      const segs = [c.center, corners[0], c.center, corners[1], c.center, corners[2], c.center, corners[3],
                    corners[0], corners[1], corners[1], corners[3], corners[3], corners[2], corners[2], corners[0]];
      for (const p of segs) pts.push(p[0], p[1], p[2]);
      const col = EL_COLOR[String(Math.round(c.el))] || [0.5, 0.5, 0.5];
      this._frustSeg.push({ start, count: segs.length, color: col, view: c.view });
    }
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pts), gl.STATIC_DRAW);
    this.frustBuf = buf;
    this.frustVao = gl.createVertexArray();
    gl.bindVertexArray(this.frustVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
  }

  // ── camera control ───────────────────────────────────────────────────────────────
  _bindInput() {
    const c = this.canvas;
    c.addEventListener("pointerdown", (e) => {
      if (!this.ready) return;
      this._dragging = true; this._snapAnim = null; this._exactCam = null;
      this._lastPointer = [e.clientX, e.clientY];
      try { c.setPointerCapture(e.pointerId); } catch { /* Safari/edge cases can reject this */ }
      this._kick();
    });
    c.addEventListener("pointermove", (e) => {
      if (!this._dragging) return;
      const dx = e.clientX - this._lastPointer[0], dy = e.clientY - this._lastPointer[1];
      this._lastPointer = [e.clientX, e.clientY];
      this.az = (this.az - dx * 0.35 + 360) % 360;
      this.el = clamp(this.el + dy * 0.28, -89, 89);
      this._kick();
    });
    const stop = () => { this._dragging = false; };
    c.addEventListener("pointerup", stop);
    c.addEventListener("pointerleave", stop);
    c.addEventListener("wheel", (e) => {
      if (!this.ready) return;
      e.preventDefault();
      this._exactCam = null; this._snapAnim = null;
      const lo = 0.6 * this.meanDist, hi = 2.5 * this.meanDist;
      this.d = clamp(this.d * (1 + e.deltaY * 0.0012), lo, hi);
      this._kick();
    }, { passive: false });
    c.addEventListener("dblclick", () => { if (this.ready) this.snapTo(this.v02Index); });
  }

  setMode(mode) { this.mode = mode; this._kick(); }
  showCameras(on) { this.showFrustums = on; this._kick(); }
  setShowUnseen(on) { this.showUnseen = on; this._kick(); }

  /** Snap the camera exactly to capture index `i`. Animates az/el/d for a smooth motion,
   *  then locks onto the camera's exact stored (R,t,center) so the render matches its PNG
   *  to the pixel (see PLAN_3D.md §4.5). */
  snapTo(i, { instant = false } = {}) {
    if (!this.ready || !this.cams[i]) return;
    const cam = this.cams[i];
    const to = azElDistFromCenter(cam.center, this.target);
    const from = { az: this.az, el: this.el, d: this.d };
    this.snappedIndex = i;
    this.snapFx = cam.fx;
    if (instant || prefersReducedMotion()) {
      this.az = to.az; this.el = to.el; this.d = to.d;
      this._exactCam = { R: cam.R, t: cam.t, center: cam.center };
      this._snapAnim = null;
    } else {
      this._exactCam = null;
      this._snapAnim = { from, to, t0: performance.now(), dur: 400, cam };
    }
    this._kick();
  }

  _stepSnapAnim(now) {
    const a = this._snapAnim;
    if (!a) return;
    const t = clamp((now - a.t0) / a.dur, 0, 1);
    const e = easeOutCubic(t);
    // shortest-path azimuth interpolation
    let dAz = a.to.az - a.from.az;
    dAz = ((dAz + 540) % 360) - 180;
    this.az = a.from.az + dAz * e;
    this.el = a.from.el + (a.to.el - a.from.el) * e;
    this.d = a.from.d + (a.to.d - a.from.d) * e;
    if (t >= 1) {
      this._snapAnim = null;
      this.az = a.to.az; this.el = a.to.el; this.d = a.to.d;
      this._exactCam = { R: a.cam.R, t: a.cam.t, center: a.cam.center };
    }
  }

  _currentCam() {
    if (this._exactCam) return { R: [].concat(...[this._exactCam.R[0], this._exactCam.R[1], this._exactCam.R[2]]), t: this._exactCam.t, center: this._exactCam.center };
    const cam = orbitCamera(this.az, this.el, this.d, this.target);
    return cam;
  }

  _setCurCam(u, cam, W, H, S) {
    const gl = this.gl;
    const scale = (S || W) / 1024;
    const cx = this.cams ? this.cams[0].cx : 563.2, cy = this.cams ? this.cams[0].cy : 532.48;
    const fx = cam.fx != null ? cam.fx : this.snapFx;
    gl.uniformMatrix3fv(u.uCurR, false, new Float32Array(Array.isArray(cam.R[0]) ? [].concat(...cam.R) : cam.R));
    gl.uniform3fv(u.uCurT, cam.t);
    gl.uniform4f(u.uCurK, fx * scale, fx * scale, cx * scale, cy * scale);
    gl.uniform2f(u.uNF, this.nearMm, this.farMm);
    gl.uniform2f(u.uViewport, W, H);
  }

  // ── render loop ──────────────────────────────────────────────────────────────────
  _isAnimating() { return this._dragging || this._snapAnim !== null; }
  _kick() {
    if (this._raf || this._destroyed || !this.ready || document.hidden || !this._visible) return;
    this._raf = requestAnimationFrame((t) => this._tick(t));
  }
  _tick(now) {
    this._raf = null;
    if (this._destroyed || document.hidden || !this._visible) return;
    this._stepSnapAnim(now);
    this._draw();
    this._updateHud();
    if (this._isAnimating()) this._raf = requestAnimationFrame((t) => this._tick(t));
  }
  _resizeCanvas() {
    if (!this.gl) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, this.canvas.clientWidth), h = Math.max(1, this.canvas.clientHeight);
    const s = Math.round(Math.min(w, h) * dpr);
    if (this.canvas.width !== s || this.canvas.height !== s) { this.canvas.width = s; this.canvas.height = s; }
    this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
  }

  _draw() {
    const gl = this.gl;
    if (!gl || !this._organs.length) return;
    this._resizeCanvas();
    const S = this.canvas.width;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, S, S);
    gl.clearColor(0, 0, 0, 1);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LESS); gl.depthMask(true);
    gl.disable(gl.CULL_FACE);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const cam = this._currentCam();
    const modeInt = this.mode === "baked" ? 1 : this.mode === "nearest" ? 2 : 0;
    const useUlrPass = modeInt !== 1 && !this.noFloat;
    gl.useProgram(this.progMain);
    this._setCurCam(this.uMain, cam, S, S, S);
    gl.uniform1i(this.uMain.uMode, useUlrPass ? modeInt : 1);
    gl.uniform1f(this.uMain.uTol, this.tolMm);
    gl.uniform1i(this.uMain.uShowUnseen, this.showUnseen ? 1 : 0);
    gl.uniform3fv(this.uMain.uEye, cam.center);
    gl.uniform2f(this.uMain.uImgSize, 1024, 1024);
    if (useUlrPass) {
      gl.uniformMatrix3fv(this.uMain.uR, false, this._camR);
      gl.uniform3fv(this.uMain.uT, this._camT);
      gl.uniform4fv(this.uMain.uK, this._camK);
      gl.uniform3fv(this.uMain.uC, this._camC);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.texImg); gl.uniform1i(this.uMain.uImg, 0);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.texVis); gl.uniform1i(this.uMain.uVis, 1);
    }
    for (const o of this._organs) {
      gl.bindVertexArray(o.vaoMain);
      gl.drawElements(gl.TRIANGLES, o.count, gl.UNSIGNED_INT, 0);
    }
    gl.bindVertexArray(null);

    if (this.showFrustums) this._drawFrustums(cam, S);
  }

  _drawFrustums(cam, S) {
    const gl = this.gl;
    gl.useProgram(this.progLine);
    this._setCurCam(this.uLine, cam, S, S, S);
    gl.bindVertexArray(this.frustVao);
    gl.depthMask(false);
    const nearestView = this._nearestView(cam);
    for (const seg of this._frustSeg) {
      const isNear = seg.view === nearestView;
      gl.uniform4f(this.uLine.uColor, ...(isNear ? [0.996, 0.925, 0.667, 1] : [...seg.color, 0.55]));
      gl.drawArrays(gl.LINES, seg.start, seg.count);
    }
    gl.depthMask(true);
    gl.bindVertexArray(null);
  }

  _nearestView(cam) {
    if (!this.cams) return null;
    const dir = norm3(sub3(cam.center, this.target));
    let best = null, bestDot = -Infinity;
    for (const c of this.cams) {
      const cd = norm3(sub3(c.center, this.target));
      const dp = dot3(dir, cd);
      if (dp > bestDot) { bestDot = dp; best = c; }
    }
    return best ? best.view : null;
  }

  _updateHud() {
    if (!this.onHud || !this.cams) return;
    const cam = this._currentCam();
    const nv = this._nearestView(cam);
    const c = this.cams.find((x) => x.view === nv);
    let angleDeg = 0;
    if (c) {
      const a = norm3(sub3(cam.center, this.target)), b = norm3(sub3(c.center, this.target));
      angleDeg = (Math.acos(clamp(dot3(a, b), -1, 1)) * 180) / Math.PI;
    }
    const exact = !!this._exactCam;
    const label = nv ? nv.slice(0, 3) : "?";
    this.onHud(`nearest render ${label} · ${angleDeg.toFixed(0)}°`, exact);
  }

  // ── self-test (R3): render the same 3 test cameras x {ulr,baked} the CPU oracle
  // (research_3d/webref.py) produced, and compare pixel-for-pixel via PSNR. ─────────────
  async selftest(baseUrl) {
    if (!this.ready) return { error: "viewer not ready" };
    const gl = this.gl;
    const camDoc = await (await fetch(`${baseUrl}/cameras.json`)).json();
    if (!this._selfFbo) {
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, 1024, 1024);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      const depth = gl.createRenderbuffer();
      gl.bindRenderbuffer(gl.RENDERBUFFER, depth);
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, 1024, 1024);
      const fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth);
      this._selfFbo = fbo; this._selfTex = tex; this._selfDepth = depth;
    }
    const renderTo = (cam, modeInt) => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this._selfFbo);
      gl.viewport(0, 0, 1024, 1024);
      gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LESS); gl.depthMask(true); gl.disable(gl.CULL_FACE);
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.useProgram(this.progMain);
      this._setCurCam(this.uMain, cam, 1024, 1024, 1024);
      const useUlr = modeInt !== 1 && !this.noFloat;
      gl.uniform1i(this.uMain.uMode, useUlr ? modeInt : 1);
      gl.uniform1f(this.uMain.uTol, this.tolMm);
      gl.uniform1i(this.uMain.uShowUnseen, 0); // the CPU oracle never applies the tint
      gl.uniform3fv(this.uMain.uEye, cam.center);
      gl.uniform2f(this.uMain.uImgSize, 1024, 1024);
      if (useUlr) {
        gl.uniformMatrix3fv(this.uMain.uR, false, this._camR);
        gl.uniform3fv(this.uMain.uT, this._camT);
        gl.uniform4fv(this.uMain.uK, this._camK);
        gl.uniform3fv(this.uMain.uC, this._camC);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.texImg); gl.uniform1i(this.uMain.uImg, 0);
        gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.texVis); gl.uniform1i(this.uMain.uVis, 1);
      }
      for (const o of this._organs) { gl.bindVertexArray(o.vaoMain); gl.drawElements(gl.TRIANGLES, o.count, gl.UNSIGNED_INT, 0); }
      gl.bindVertexArray(null);
      const px = new Uint8Array(1024 * 1024 * 4);
      gl.readPixels(0, 0, 1024, 1024, gl.RGBA, gl.UNSIGNED_BYTE, px);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return flipRows(px, 1024, 1024);
    };
    const decodePng = async (url) => {
      const bmp = await createImageBitmap(await (await fetch(url)).blob(), { colorSpaceConversion: "none" });
      if (!this._psnrCanvas) { this._psnrCanvas = document.createElement("canvas"); this._psnrCanvas.width = 1024; this._psnrCanvas.height = 1024; }
      const ctx = this._psnrCanvas.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(bmp, 0, 0);
      bmp.close?.();
      return ctx.getImageData(0, 0, 1024, 1024).data;
    };

    const results = {};
    for (const c of camDoc.cameras) {
      for (const [modeName, modeInt] of [["ulr", 0], ["baked", 1]]) {
        const rendered = renderTo(c, modeInt);
        const ref = await decodePng(`${baseUrl}/${c.view}_${modeName}.png`);
        results[`${c.view}_${modeName}`] = round2(psnr(rendered, ref));
      }
    }
    // extra R3 check: ULR at the captured v02 vs the ACTUAL neural render, on organ pixels
    const v02 = this.cams[this.v02Index];
    const rendered = renderTo(v02, 0);
    const refImg = await decodePng(v02.image);
    const fg = fgMask(refImg);
    results["v02_ulr_vs_output_fg"] = round2(psnr(rendered, refImg, fg));
    console.info("[recon-selftest]", results);
    return results;
  }

  /** Free every GPU resource this run's data owns (textures, buffers, VAOs, programs) —
   *  called when another run is loaded (PLAN_3D.md §4.6). Keeps the WebGL2 context itself
   *  alive (context loss/restore is asynchronous and not worth the fragility here — see
   *  meshviewer.js, which uses the same "clear and reuse the context" pattern via reset()),
   *  so a following load() just rebuilds into the same context. `destroy()` below is the
   *  final teardown (page/canvas going away) and does drop the context. */
  dispose() {
    this.ready = false;
    const gl = this.gl;
    if (gl) {
      this._organs.forEach((o) => this._freeOrgan(o));
      [this.texImg, this.texVis, this._selfTex].forEach((t) => t && gl.deleteTexture(t));
      [this.visFbo, this._selfFbo].forEach((f) => f && gl.deleteFramebuffer(f));
      [this.visDepth, this._selfDepth].forEach((r) => r && gl.deleteRenderbuffer(r));
      if (this.frustBuf) gl.deleteBuffer(this.frustBuf);
      if (this.frustVao) gl.deleteVertexArray(this.frustVao);
      [this.progMain, this.progLine, this.progVis].forEach((p) => p && gl.deleteProgram(p));
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    }
    this.progMain = this.progLine = this.progVis = null;
    this.texImg = this.texVis = this._selfTex = null;
    this.visFbo = this._selfFbo = this.visDepth = this._selfDepth = null;
    this.frustBuf = this.frustVao = null;
    this._organs = []; this.cams = null; this._exactCam = null; this._snapAnim = null;
  }

  /** Final teardown (the canvas itself is going away) — this does drop the context. */
  destroy() {
    this._destroyed = true;
    if (this._raf) cancelAnimationFrame(this._raf);
    if (this._ro) this._ro.disconnect();
    if (this._io) this._io.disconnect();
    document.removeEventListener("visibilitychange", this._onVis);
    this.dispose();
    if (this.gl) this.gl.getExtension("WEBGL_lose_context")?.loseContext();
    this.gl = null;
  }
}

function flipRows(px, W, H) {
  const out = new Uint8Array(px.length);
  const stride = W * 4;
  for (let y = 0; y < H; y++) out.set(px.subarray(y * stride, y * stride + stride), (H - 1 - y) * stride);
  return out;
}
// Foreground mask for the R3 organ-pixel check, eroded by one pixel (3x3, all 8
// neighbours) — the same "no silhouette px" step research_3d/export_colmap.py's own
// --verify takes before comparing across two different renderers/rasterisers. Without
// it, the thin antialiased fringe where the browser's WebGL edge and the reference
// PNG's own (differently antialiased) edge disagree by a pixel dominates the metric:
// on v02 it alone drops the masked PSNR from ~80 dB (organs agree almost exactly) to
// 33 dB, even though the un-masked full-frame PSNR (~42 dB) shows nothing is wrong.
function fgMask(rgba, W = 1024, H = 1024) {
  const n = W * H;
  const raw = new Uint8Array(n);
  for (let i = 0; i < n; i++) raw[i] = Math.max(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2]) > 8 ? 1 : 0;
  const out = new Uint8Array(n);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (!raw[i]) continue;
      let ok = 1;
      for (let dy = -1; dy <= 1 && ok; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const yy = y + dy, xx = x + dx;
          if (yy < 0 || yy >= H || xx < 0 || xx >= W || !raw[yy * W + xx]) { ok = 0; break; }
        }
      }
      out[i] = ok;
    }
  }
  return out;
}
function psnr(a, b, mask) {
  let se = 0, n = 0;
  const N = a.length / 4;
  for (let i = 0; i < N; i++) {
    if (mask && !mask[i]) continue;
    for (let c = 0; c < 3; c++) { const d = a[i * 4 + c] - b[i * 4 + c]; se += d * d; }
    n += 3;
  }
  const mse = Math.max(se / Math.max(n, 1), 1e-6);
  return 10 * Math.log10((255 * 255) / mse);
}
function round2(x) { return Math.round(x * 100) / 100; }
