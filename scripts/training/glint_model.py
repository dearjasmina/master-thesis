"""
glint_model.py — the vessel glints v25 renders, computed analytically from the dataset.

In v25 the vessels exist ONLY in the normal-map tiles, and the ground truth shows them as
thin specular glints. Where a glint lands depends on the normal-map perturbation, the
tangent frame it is applied in, the camera and the lights. The pre-flight showed a
network does not infer that response from the raw tile colours (held-out M3 0.003 vs GT
0.066), so this module computes it. Everything is recoverable from the dataset:

  camera     meta.json + camera_model (lens shift, emission gain)
  lights     v25 setup_lights(): 5 world-fixed area lights around meta organ_center,
             positions and sizes in units of meta scene_scale, energy ~ (scene_scale/0.78)^2
  material   v25 TISSUES + REFERENCE_LOOK, loaded from the source without Blender;
             roughness = t.rough * (0.6 + 0.8 * rough tile), normal map strength 0.8
  tiles      texture_lookup (the Cycles-exact BOX projection)

The one unknown is the TANGENT FRAME: the meshes have no UVs, and the Normal Map node is
in tangent space. Cycles then derives tangents with MikkTSpace from a sphere mapping of
generated coordinates. This module implements that and alternatives as named hypotheses,
plus a tangent-free upper bound; `texture_learnability.py glint` measures which one
matches the GT before anything is trained on it.

Specular model: GGX with Smith masking and Schlick Fresnel, area lights widened by
alpha' = alpha + size / (4 d). Radiometry is relative, which is all a correlation needs.
Base specular always uses the perturbed normal. Whether the COAT does is a hypothesis too:
make_material() leaves 'Coat Normal' unlinked, and with coat roughness ~0.02-0.03 on the
glossy organs the coat is the only lobe sharp enough to draw one-pixel streaks.

Note: generate_training_dataset.py applies its per-subject key-energy jitter only to a
light named "KeyLight", and v25 never names one — so meta's subj_key_energy_factor was
recorded but never applied. It is ignored here for that reason.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V25_PATH = ROOT / "scripts" / "render_pair_totalseg_v25_textured.py"
NORMAL_MAP_STRENGTH = 0.8
COAT_IOR = 1.41

# (name, offset from organ_center in scene_scale units, base watts, colour, size in scene_scale)
V25_LIGHTS = (
    ("key",    (1.4, 0.3, 0.8),   90.0, (1.00, 0.98, 0.96), 0.25),
    ("fill",   (-1.0, -1.0, 0.5), 15.0, (0.96, 0.97, 1.00), 1.10),
    ("rim1",   (-0.5, 1.4, 0.3),  20.0, (1.00, 0.94, 0.88), 0.02),
    ("rim2",   (0.4, -1.4, -0.2), 21.0, (0.95, 0.95, 1.00), 0.08),
    ("bounce", (0.2, 0.1, -1.1),  30.0, (1.00, 0.34, 0.22), 1.20),   # feat["env"] is True
)


# ── material ──────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def tissue_params(v25_path=str(V25_PATH)):
    """{tissue name: params} exactly as v25 renders them (TISSUES + REFERENCE_LOOK).

    The tables are plain data between `_DEFAULTS = dict(` and the module-level
    `apply_reference_look()` call, so they execute without bpy. The one module constant
    they read (AORTA_REALISTIC) is parsed from the source rather than assumed.
    """
    src = Path(v25_path).read_text()
    ns = {"AORTA_REALISTIC": re.search(r"^AORTA_REALISTIC\s*=\s*(True|False)", src, re.M).group(1) == "True"}
    a = src.index("_DEFAULTS = dict(")
    b = src.index("\napply_reference_look()\n") + len("\napply_reference_look()\n")
    exec(compile(src[a:b], "v25_tissue_tables", "exec"), ns)
    return {t["name"]: t for t in ns["TISSUES"]}


def f0_from_ior(ior):
    return ((ior - 1.0) / (ior + 1.0)) ** 2


# ── lights ────────────────────────────────────────────────────────────────────
def light_rig(meta):
    """[(position_m, watts, rgb, size_m, emitter_axis)] as setup_lights() placed them."""
    c = np.asarray(meta["organ_center"], np.float64)
    sc = float(meta["scene_scale"])
    e = (sc / 0.78) ** 2
    rig = []
    for _, off, watts, rgb, size in V25_LIGHTS:
        pos = c + sc * np.asarray(off, np.float64)
        axis = (c - pos) / np.linalg.norm(c - pos)          # _track_to: -Z faces the centre
        rig.append((pos, watts * e, np.asarray(rgb), size * sc, axis))
    return rig


# ── tangent frames ────────────────────────────────────────────────────────────
TANGENT_FAMILIES = {
    # Cycles, no UV map: MikkTSpace on map_to_sphere(generated coordinates), i.e. a sphere
    # mapping centred on the object's texture space (mesh bounds), normalised per axis.
    "sphere_bbox": "centre = mesh bbox centre, per-axis normalised by bbox half-size",
    # Alternatives, in case generated coordinates are not re-centred the way recalled.
    "sphere_origin": "centre = object origin (CT volume corner), no normalisation",
    "sphere_centre": "centre = mesh bbox centre, isotropic",
}


def sphere_uv(P_mm, family, bbox=None):
    if family == "sphere_bbox":
        c, hs = bbox
        q = (P_mm - c) / np.maximum(hs, 1e-6)
    elif family == "sphere_centre":
        q = P_mm - bbox[0]
    elif family == "sphere_origin":
        q = P_mm
    else:
        raise ValueError(family)
    r = np.linalg.norm(q, axis=-1)
    u = -np.arctan2(q[..., 0], q[..., 1]) / (2 * np.pi)
    v = -np.arcsin(np.clip(q[..., 2] / np.maximum(r, 1e-9), -1, 1)) / np.pi
    return u, v


def screen_tangents(P, N, u, v):
    """Per-pixel MikkTSpace-style frame from screen-space derivatives: dP = T du + B dv.

    Returns unit T orthogonal to N, and B = sign * cross(N, T) as Cycles' Normal Map node
    builds it. u wraps at the atan2 seam, so its differences are wrapped into [-0.5, 0.5].
    Pixels next to silhouettes get garbage frames; callers score eroded masks.
    """
    def d(a, axis):
        g = np.zeros_like(a)
        if axis == 1:
            g[:, 1:-1] = (a[:, 2:] - a[:, :-2]) / 2.0
        else:
            g[1:-1] = (a[2:] - a[:-2]) / 2.0
        return g
    wrap = lambda x: x - np.round(x)
    dPx, dPy = d(P, 1), d(P, 0)
    dux, duy = wrap(d(u, 1)), wrap(d(u, 0))
    dvx, dvy = d(v, 1), d(v, 0)
    det = dux * dvy - duy * dvx
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)[..., None]
    T = (dPx * dvy[..., None] - dPy * dvx[..., None]) / det
    B_raw = (dPy * dux[..., None] - dPx * duy[..., None]) / det
    T = T - N * np.sum(T * N, -1, keepdims=True)
    T = T / np.maximum(np.linalg.norm(T, axis=-1, keepdims=True), 1e-12)
    sign = np.sign(np.sum(np.cross(N, T) * B_raw, -1, keepdims=True))
    sign[sign == 0] = 1.0
    return T, sign * np.cross(N, T)


def perturb(N, T, B, color, sT=1.0, sB=1.0, strength=NORMAL_MAP_STRENGTH):
    """Cycles svm_node_normal_map (tangent space), including its strength blend."""
    Nm = color[..., :1] * sT * T + color[..., 1:2] * sB * B + color[..., 2:3] * N
    Nm = Nm / np.maximum(np.linalg.norm(Nm, axis=-1, keepdims=True), 1e-12)
    Np = N + (Nm - N) * strength
    return Np / np.maximum(np.linalg.norm(Np, axis=-1, keepdims=True), 1e-12)


def max_tilt(color, strength=NORMAL_MAP_STRENGTH):
    """Angle between N and the perturbed normal, independent of the tangent frame."""
    c = color / np.maximum(np.linalg.norm(color, axis=-1, keepdims=True), 1e-12)
    th = np.arccos(np.clip(c[..., 2], -1, 1))
    return np.arctan2(strength * np.sin(th), 1 - strength + strength * np.cos(th))


# ── shading ───────────────────────────────────────────────────────────────────
def _g1(ndotx, a):
    return 2 * ndotx / np.maximum(ndotx + np.sqrt(a * a + (1 - a * a) * ndotx * ndotx), 1e-12)


def specular(Nn, V, P_m, rig, alpha, f0, tilt=None):
    """Relative GGX specular luminance summed over the rig.

    tilt: if given, the tangent-free best case — the normal is assumed tilted toward each
    light's half vector by up to `tilt` radians (an upper bound on where a glint CAN be).
    """
    alpha = np.maximum(alpha, 1e-3)
    nv = np.clip(np.sum(Nn * V, -1), 1e-4, 1.0)
    out = np.zeros(Nn.shape[:-1])
    lum = lambda rgb: 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    for pos, watts, rgb, size, axis in rig:
        Lv = pos - P_m
        dist = np.linalg.norm(Lv, axis=-1)
        l = Lv / dist[..., None]
        cos_e = np.clip(-np.sum(l * axis, -1), 0.0, 1.0)
        h = l + V
        h = h / np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-12)
        nl = np.clip(np.sum(Nn * l, -1), 0.0, 1.0)
        nh = np.clip(np.sum(Nn * h, -1), 0.0, 1.0)
        if tilt is not None:
            nh = np.cos(np.maximum(np.arccos(nh) - tilt, 0.0))
        vh = np.clip(np.sum(V * h, -1), 0.0, 1.0)
        a = np.minimum(alpha + size / (4.0 * dist), 1.0)
        a2 = a * a
        D = a2 / (np.pi * (nh * nh * (a2 - 1) + 1) ** 2)
        F = f0 + (1 - f0) * (1 - vh) ** 5
        G = _g1(nl, a) * _g1(nv, a)
        out += D * F * G / (4 * nv) * watts * cos_e / (dist * dist) * lum(rgb) * (nl > 0)
    return out


def camera_dirs(P_mm, meta):
    C = np.asarray(meta["camera_pos"], np.float64)
    P_m = P_mm / 1000.0
    V = C - P_m
    return V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-12), P_m


# ── the validated model ───────────────────────────────────────────────────────
# texture_learnability.py `glint` on the server (s0098, s0746, s0194; 60 views) ranked 24
# tangent hypotheses + a tangent-free bound by texture-locked excess. Winner:
# sphere_origin, signs (+,-), coat unperturbed — excess +0.119 (partial r 0.149, half-tile
# null 0.030), 2.8x the raw-ridge proxy. It is exactly Cycles' own construction:
#   * blender/mesh.cpp stores generated = co*size - loc, and MikkMeshWrapper undoes it with
#     (orco + orco_loc) * inv_orco_size, so map_to_sphere() sees RAW object coordinates —
#     a sphere mapping about the object origin, not the texture-space box.
#   * Cycles' map_to_sphere has v = 1 - acos(z/|co|)/pi, whose derivative is the opposite
#     of the -asin(z/|co|)/pi used by sphere_uv(), so the hypothesis search needed sB = -1.
#     With Cycles' own v below, both signs are +1.
#   * Principled BSDF leaves an unlinked Coat Normal at the unperturbed shading normal.
# cycles_uv() + perturb(sT=+1, sB=+1) + coat on N reproduces sphere_origin+-_C0 exactly.
GLINT_LOG_RANGE = 6.0     # log1p(G / view median of plain highlights) mapped to [0, 1]


def cycles_uv(P_mm):
    """Cycles map_to_sphere() on raw object coordinates (util/math: u, v conventions)."""
    r = np.linalg.norm(P_mm, axis=-1)
    u = 0.5 - np.arctan2(P_mm[..., 0], P_mm[..., 1]) / (2 * np.pi)
    v = 1.0 - np.arccos(np.clip(P_mm[..., 2] / np.maximum(r, 1e-9), -1, 1)) / np.pi
    return u, v


def glint_buffer(view_dir, meta, pid_to_gid, organ_names, bank, tex_class_fn, box_weights_fn,
                 box_project_fn, tex_shift_mm=0.0):
    """HxWx2 float32 in [-1, 1]: [vessel glints, plain geometric highlights]. Background -1.

    Channel 0 shades the Cycles-exact perturbed normal (normal-map tile, MikkTSpace frame on
    map_to_sphere of object coordinates, strength 0.8) for the base lobe and the
    unperturbed normal for the coat; channel 1 shades the unperturbed normal for both. The
    network gets the vessel term and the geometric term separately, so the difference is
    the vessel contribution. Both use the view's median plain highlight as reference, so
    their ratio is preserved. Texture helpers are passed in to avoid a circular import.
    tex_shift_mm offsets only the tile lookups — preflight.py uses a half tile as a null.
    """
    import camera_model as cm
    view_dir = Path(view_dir)
    P = cm.unproject(cm.depth_m(view_dir), meta)
    N = cm.normals_world(view_dir)
    lut = np.asarray(pid_to_gid).astype(np.int64)
    gid = lut[np.clip(cm.segid_pid(view_dir), 0, len(lut) - 1)]
    Vd, P_m = camera_dirs(P, meta)
    rig = light_rig(meta)
    tp = tissue_params()
    T, B = screen_tangents(P, N, *cycles_uv(P))
    H, W = gid.shape
    vessel, flat = np.zeros((H, W)), np.zeros((H, W))
    organ_px = gid > 0
    for g in np.unique(gid):
        if g <= 0:
            continue
        organ = organ_names[int(g) - 1]
        t = tp.get(organ)
        if t is None:
            continue
        m = gid == g
        Nn, Vn, Pn = N[m], Vd[m], P_m[m]
        cls = tex_class_fn(organ)
        nimg, ns = bank.get(cls, "normal") if cls else (None, 0.0)
        rimg, rs = bank.get(cls, "rough") if cls else (None, 0.0)
        w = box_weights_fn(Nn)
        Pt = P[m] + tex_shift_mm
        rough = t["rough"] * ((0.6 + 0.8 * box_project_fn(rimg, Pt, Nn, tex_mm=bank.tex_mm, w=w, scale=rs))
                              if rimg is not None else 1.0)
        a_b = np.broadcast_to(np.asarray(rough, np.float64) ** 2, (len(Pn),))
        a_c = np.full(len(Pn), t["coat_rough"] ** 2)
        f0b, f0c = f0_from_ior(t["ior"]), f0_from_ior(COAT_IOR)
        coat = t["coat"] * specular(Nn, Vn, Pn, rig, a_c, f0c)
        flat[m] = specular(Nn, Vn, Pn, rig, a_b, f0b) + coat
        if nimg is not None:
            col = box_project_fn(nimg, Pt, Nn, tex_mm=bank.tex_mm, w=w, scale=ns) * 2.0 - 1.0
            Np = perturb(Nn, T[m], B[m], col)
            vessel[m] = specular(Np, Vn, Pn, rig, a_b, f0b) + coat
        else:
            vessel[m] = flat[m]
    ref = max(float(np.median(flat[organ_px])), 1e-12) if organ_px.any() else 1.0
    out = np.full((H, W, 2), -1.0, dtype=np.float32)
    for k, G in enumerate((vessel, flat)):
        ch = np.clip(np.log1p(G / ref) / GLINT_LOG_RANGE, 0.0, 1.0) * 2.0 - 1.0
        out[..., k] = np.where(organ_px, ch, -1.0)
    return out
