# Evaluation report

## Rendering cost

- network: **98.824 ms/frame** on NVIDIA L4
- **speed-up not computed** (no Cycles reference supplied)

## Distribution realism

- FID **6.26**, KID **0.00011 ± 0.00158** (n = 1920, in-repo fallback (torchvision Inception-v3, bilinear 299))

## Multi-view consistency

- reprojection PSNR, ground truth (floor): **15.42 dB**
- reprojection PSNR, model: **15.46 dB**
- **consistency gap: -0.04 dB** (lower is better, n = 35 pairs)

> GT reprojection PSNR is the FLOOR (interpolation, depth quantisation and genuinely view-dependent shading). The gap is the inconsistency the model ADDS. A gap near 0 dB means the model is as multi-view consistent as the renderer it imitates; a large gap means it invents per-view detail, which would appear as flicker under camera motion.

## Per-organ quality (worst first)

| organ | n | pixel frac | PSNR | 95% CI |
|---|---|---|---|---|
| liver | 300 | 0.0240 | 24.85 | [24.54, 25.15] |
| aorta | 301 | 0.0040 | 25.65 | [25.26, 26.03] |
| kidney_left | 137 | 0.0034 | 25.84 | [25.13, 26.52] |
| kidney_right | 119 | 0.0038 | 26.01 | [25.26, 26.69] |
| vertebrae_L4 | 128 | 0.0019 | 26.02 | [25.46, 26.58] |
| duodenum | 74 | 0.0013 | 26.09 | [25.36, 26.88] |
| superior_vena_cava | 82 | 0.0010 | 26.44 | [25.49, 27.47] |
| pancreas | 76 | 0.0019 | 26.45 | [25.24, 27.72] |
| esophagus | 247 | 0.0020 | 26.49 | [26.10, 26.93] |
| gallbladder | 50 | 0.0009 | 26.55 | [23.47, 32.02] |
| small_bowel | 258 | 0.0189 | 26.60 | [26.29, 26.91] |
| stomach | 194 | 0.0061 | 26.65 | [26.11, 27.16] |
| vertebrae_T12 | 146 | 0.0013 | 26.69 | [26.24, 27.13] |
| vertebrae_L5 | 127 | 0.0034 | 27.04 | [26.66, 27.42] |
| heart | 231 | 0.0102 | 27.06 | [26.72, 27.42] |
| spleen | 215 | 0.0061 | 27.14 | [26.67, 27.62] |
| vertebrae_L3 | 110 | 0.0012 | 27.15 | [26.61, 27.67] |
| vertebrae_L2 | 114 | 0.0012 | 27.42 | [26.90, 27.91] |
| vertebrae_L1 | 138 | 0.0012 | 27.43 | [26.94, 27.88] |
| colon | 295 | 0.0175 | 27.60 | [27.34, 27.84] |
| inferior_vena_cava | 148 | 0.0014 | 27.69 | [26.88, 28.60] |
| portal_vein_and_splenic_vein | 40 | 0.0009 | 28.84 | [26.78, 31.04] |
| urinary_bladder | 175 | 0.0066 | 28.96 | [28.53, 29.40] |
| lung_lower_lobe_left | 307 | 0.0150 | 32.00 | [31.55, 32.47] |
| lung_lower_lobe_right | 318 | 0.0159 | 32.12 | [31.71, 32.53] |
| lung_upper_lobe_left | 336 | 0.0246 | 32.24 | [31.87, 32.60] |
| lung_upper_lobe_right | 275 | 0.0242 | 32.35 | [31.87, 32.86] |
| autochthon_right | 380 | 0.0133 | 32.38 | [32.02, 32.78] |
| autochthon_left | 366 | 0.0126 | 32.59 | [32.15, 33.02] |
