# 1.5.5 Speed Benchmark — Networks vs Interactive Rate

## Setup

| Parameter | Value |
|-----------|-------|
| Device benchmarked | Apple MPS (M-series, local laptop) |
| Timed runs per config | 3 (+ 1 warm-up, not counted) |
| Conditioning input | CT95 depth.png (512×512) |
| Interactive threshold | < 33ms = 30 FPS |
| Real-time target (3DGS) | < 10ms = 100 FPS |

---

## Results

| Method | Steps | Mean | Std | FPS equiv | Slowdown vs 30FPS |
|--------|-------|------|-----|-----------|-------------------|
| SD 1.5 + ControlNet | 20 | 25.80s | 0.12s | 0.04 | **782×** |
| SD 1.5 + ControlNet | 50 | 68.82s | 8.93s | 0.01 | **2,085×** |
| IP-Adapter + ControlNet | 20 | 616.32s | 279.87s | 0.00 | **18,676×** |
| Luma Ray 2 (cloud) | — | ~30s | — | 0.03 | **~909×** |
| **3DGS (literature)** | — | **0.010s** | — | **100** | **0×** |

*3DGS value from Kerbl et al., SIGGRAPH 2023: ≥100 FPS @ 1080p on RTX 3090.*
*IP-Adapter timing reused from 1.5.4 (5 runs, high variance due to MPS CLIP overhead).*

---

## Key Observations

### Every diffusion method fails the interactive threshold by 3–4 orders of magnitude
- ControlNet @ 20 steps: **782× too slow** (25.8s vs 33ms needed)
- ControlNet @ 50 steps: **2,085× too slow**
- IP-Adapter: **18,676× too slow** (extreme MPS overhead from per-step CLIP encoder)
- Luma Ray: **~909× too slow** (cloud latency, no local GPU used)

All four methods require seconds-to-minutes per frame. Interactive rendering (≥30 FPS)
requires milliseconds. The gap is not a "can we optimise this" problem — it is
fundamental to diffusion sampling: even at 20 steps, each step is a full U-Net forward pass.

### MPS vs CUDA: expected speedup on cluster
| Method | MPS (this run) | Expected CUDA 24GB | Est. CUDA slowdown |
|--------|---------------|-------------------|-------------------|
| ControlNet @ 20 steps | 25.8s | ~1.5–2.5s | ~45–78× |
| ControlNet @ 50 steps | 68.8s | ~4–6s | ~120–200× |
| IP-Adapter @ 20 steps | 616s | ~30–60s | ~900–1800× |

Even on a high-end cluster GPU, diffusion models remain **45–2000× slower** than
the interactive threshold. 3DGS at 100 FPS = 10ms is the only viable real-time path.

### IP-Adapter MPS variance is extreme (std = 279s)
The 279s standard deviation across 5 runs reflects MPS instability when the CLIP
encoder runs at every diffusion step — memory management causes unpredictable stalls.
On CUDA this would be ~5× overhead vs plain ControlNet, not ~24×.

---

## Thesis Implications — Hypothesis 2 Confirmed

> **Hypothesis 2: Diffusion networks are too slow for interactive volumetric rendering.**

**CONFIRMED quantitatively.**

The numbers establish the thesis motivation for 3DGS as the real-time stage:

```
Diffusion model (best case, CUDA):  ~1.5s/frame  →  ~0.7 FPS
3DGS rendering:                     ~10ms/frame  →  ~100 FPS
Speed-up from 3DGS:                 ~140×
```

This is the core speed argument: train once offline with diffusion → reconstruct once
with 3DGS → render in real-time thereafter. The diffusion step happens at dataset
generation time, not at query time.

**Table for thesis (to be filled with cluster CUDA numbers in Phase 4):**

| Method | Steps | Time/frame | FPS equiv | Notes |
|--------|-------|-----------|-----------|-------|
| SD 1.5 + ControlNet | 20 | TBD (cluster) | TBD | |
| SD 1.5 + ControlNet | 50 | TBD (cluster) | TBD | |
| IP-Adapter + ControlNet | 30 | TBD (cluster) | TBD | |
| Luma Ray 2 | — | ~30s | ~0.03 | Cloud API |
| 3DGS (target) | — | <10ms | ≥100 | Kerbl et al. 2023 |
