# AI-enhanced volumetric CT rendering: a complete thesis research guide

**This guide covers the full pipeline from CT volume acquisition through 3D Gaussian Splatting, providing specific tools, papers, code repositories,https://github.com/dearjasmina/master-thesis.git and benchmarks for each stage.** The pipeline—CT volume → fast ray-cast conditioning signals → diffusion model enhancement → 3DGS real-time rendering—is novel in its combination and represents a viable master's thesis contribution. The most critical decision points are (1) choosing Mitsuba 3 for ground truth generation, (2) ControlNet on SD 1.5 as the primary model architecture, and (3) original 3DGS or 2D Gaussian Splatting for final reconstruction. Each stage has well-documented open-source tooling, but the main thesis risks lie in cross-view consistency of diffusion outputs and the domain gap between Stable Diffusion's natural image priors and medical volume rendering aesthetics.

---

## Section A: datasets and preprocessing for CT volume rendering

### Head/skull and thorax datasets with open access

For head/skull CT (the primary focus), the two most valuable datasets are the **RSNA 2019 Brain CT Hemorrhage dataset** (~25,000 head CT exams, 874,035 DICOM slices from four institutions; available at https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/ and AWS Open Data Registry) and **CQ500** (491 non-contrast head CTs, ~28.66 GB in DICOM format, CC BY-NC-SA 4.0 license; download at http://headctstudy.qure.ai/dataset; Chilamkurthy et al., arXiv:1803.05854). RSNA is the largest public head CT collection and provides multi-scanner diversity essential for generalization. CQ500 is smaller but includes radiologist annotations for fractures and hemorrhage, making it useful for validating anatomical correctness of renderings.

For thorax/torso data, **TotalSegmentator v2** is the standout choice: 1,228 CT volumes with 117 anatomical structure segmentations (27 organs, 59 bones, 10 muscles, 8 vessels) in NIfTI format under CC BY 4.0 license (Wasserthal et al., Radiology: AI 2023;5:e230024; download at https://zenodo.org/records/10047292; GitHub: https://github.com/wasserth/TotalSegmentator). It includes both skull bones and thoracic structures, making it uniquely suitable for both anatomy targets. **LIDC-IDRI** provides 1,018 thoracic CT scans under CC BY 3.0 (https://www.cancerimagingarchive.net/collection/lidc-idri/; Armato et al., Medical Physics 2011;38(2):915-931), while **AMOS** offers 500 CT plus 100 MRI multi-organ scans under CC BY 4.0 (https://zenodo.org/records/7262581; Ji et al., arXiv:2206.08023, NeurIPS 2022).

Additional useful datasets include CT-ORG (140 CTs with brain and body organs; https://www.cancerimagingarchive.net/collection/ct-org/), BTCV/Synapse (50 abdominal CTs; https://www.synapse.org/#!Synapse:syn3193805), and the Medical Segmentation Decathlon (10 tasks, CC BY-SA 4.0; http://medicaldecathlon.com/). The SAT-DS collection (https://github.com/zhaoziheng/SAT-DS) aggregates 72+ public segmentation datasets with shortcut download links.

### Loading, windowing, and normalizing CT volumes

CT values are measured in Hounsfield Units (HU), where air = −1000 and water = 0. Standard clinical windows are defined by a center level (L) and width (W), computed as HU_min = L − W/2 and HU_max = L + W/2. The critical windows for this project are **bone (W:2000, L:500, range −500 to 1500 HU)**, soft tissue (W:400, L:40, range −160 to 240 HU), and lung (W:1500, L:−600). For rendering, keep raw HU values and apply windowing via transfer functions at render time rather than baking windows into the data.

For NIfTI loading with nibabel: `img = nib.load('volume.nii.gz')`, `data = img.get_fdata()`, `voxel_spacing = img.header.get_zooms()`. Convert to canonical RAS orientation with `nib.as_closest_canonical(img)`. For DICOM with SimpleITK: use `sitk.ImageSeriesReader()` with `GetGDCMSeriesFileNames()`, noting that `sitk.GetArrayFromImage()` returns (Z,Y,X) axis order versus nibabel's (X,Y,Z). Convert DICOM to NIfTI early in your pipeline with `sitk.WriteImage(image, 'output.nii.gz')` to standardize formats.

Resample to isotropic voxels (recommend **1.0 mm³** for general use or **0.5 mm** for high-resolution skull rendering) using `sitk.Resample()` with `sitk.sitkLinear` interpolation for CT images and `sitk.sitkNearestNeighbor` for masks. Set the default fill value to −1000 (air). Be aware that nibabel always uses RAS+ coordinates while SimpleITK uses LPS internally; converting between them requires negating the X and Y components.

### Paired dataset structure for ControlNet training

The HuggingFace diffusers ControlNet training script expects a directory with `conditioning_images/` (inputs) and `images/` (targets) plus a `metadata.jsonl` file where each line contains `{"image": "images/00001.png", "conditioning_image": "conditioning_images/00001.png", "text": "prompt"}`. The recommended naming convention for this project is `{volume_id}_view{angle_idx}_{window_type}.png`, with metadata recording azimuth, elevation, and volume ID.

A critical design rule: **split train/val/test at the volume level, not the image level**, to prevent data leakage. If images from the same CT volume appear in both training and test sets, the model will appear to perform much better than it actually generalizes. Render at the target resolution (512×512 for SD 1.5, 1024×1024 for SDXL) or render at higher resolution and downsample. Store camera parameters in metadata for exact reproducibility of viewpoints across rendering systems.

### Validated augmentation strategies

For paired render-to-render training, augmentations must be applied identically to input and target images to maintain geometric alignment. Safe augmentations include random rotation (±15–30°), random cropping with resize, random scaling (0.8–1.2×), and horizontal flipping for thorax data (bilateral symmetry). **Avoid flipping for head CT**—Kebaili et al. (arXiv:2307.13125) and Chlap et al. (J Med Imaging Radiat Oncol 2021;65:545-563) note that flipping causes anatomical inconsistencies since brain laterality is clinically significant.

Input-only augmentations (applied to conditioning images but not targets) can improve robustness: Gaussian noise, slight intensity perturbation, and random gamma correction simulate rendering parameter variations. Volume-level augmentations applied before rendering—random HU window variation (±5–10%), small camera jitter (±2–5° azimuth/elevation), and transfer function randomization—are the most principled approach because they generate genuinely different but valid training pairs.

---

## Section B: generating conditioning signals from CT volumes

### Which visual buffers matter most for diffusion model conditioning

The original ControlNet paper (Zhang & Agrawala, "Adding Conditional Control to Text-to-Image Diffusion Models," arXiv:2302.05543, ICCV 2023) trained 8 conditioning types. For this pipeline, the most relevant are **depth maps** (global 3D structure), **surface normal maps** (fine geometric detail—the ControlNet documentation states normals are "a bit better at preserving geometry" than depth because "minor details are not salient in depth maps, but are salient in normal maps"), **Canny edge maps** (structural boundaries), and **transfer-function-colored renders** (tissue-type color information).

Multi-condition approaches are well-supported. Multi-ControlNet composes multiple ControlNets by summing their output features with per-condition weight λ. Uni-ControlNet (Zhao et al., NeurIPS 2023; https://github.com/ShihaoZhaoZSH/Uni-ControlNet) uses only 2 adapters (local + global) for any number of conditions, finding that HED edges are "the most powerful" local condition. T2I-Adapter (Mou et al., AAAI 2024; https://github.com/TencentARC/T2I-Adapter) achieves conditioning with only ~77M parameters (~300MB) versus ControlNet's ~361M. DC-ControlNet (arXiv:2502.14779) provides a key insight: **"edge conditions regulate image details and textures, while depth and normal conditions define geometric properties."**

No published work specifically addresses conditioning signals for volumetric medical rendering with ControlNet-like models. This represents a genuine research gap that the thesis can fill.

### Rendering buffers with PyVista and VTK

PyVista's `add_volume()` handles the core volume rendering, supporting colormaps, opacity curves, Phong shading, and multiple blending modes (composite, additive, maximum). Key parameters include `shade=True` for gradient-based lighting that reveals surface detail, `opacity` accepting predefined curves (`"sigmoid"`, `"sigmoid_6"`) or custom arrays, and `clim=[min_val, max_val]` for scalar range mapping.

For **depth buffer extraction**, use `plotter.get_image_depth()` which returns a normalized 0–1 numpy array for surface rendering. VTK's volume ray caster does not write to the standard z-buffer, so for volumetric depth you must either extract an isosurface first or use the SSAO pass's internal depth mechanism. For **surface normals**, extract an isosurface at the bone threshold (~300 HU) using `grid.contour(isosurfaces=[300], method='flying_edges')` followed by `isosurface.compute_normals(consistent_normals=True)`. Map normals from [-1,1] to [0,255] RGB using the standard convention (blue=Z+, red=X+, green=Y+) to match ControlNet v1.1's `control_v11p_sd15_normalbae` model.

**Ambient occlusion** is available natively via `vtkSSAOPass` since VTK 9.0, extended to volumes circa 2023 (documented at https://www.kitware.com/screen-space-ambient-occlusion-for-volumes/). PyVista exposes `plotter.enable_ssao()` for surfaces. For volumetric SSAO, access the underlying VTK renderer and configure radius, bias, kernel size, and blur parameters manually; volume property must have `ShadeOn()`. **Edge maps** require post-processing: render the volume, then apply `cv2.Canny()` on the 2D image. **Curvature maps** can be computed natively from isosurface meshes with `mesh.curvature(curv_type='mean')`.

### Transfer function design for bone visualization

Cortical bone spans **+700 to +3000 HU**, cancellous bone +300 to +700 HU, soft tissue +40 to +100 HU, and air −1000 HU. A skull-optimized transfer function should use a steep opacity ramp at ~200–400 HU to sharply delineate bone surfaces, full transparency below ~100–200 HU to eliminate soft tissue, and Phong shading (`shade=True`) to reveal surface microstructure. In VTK, configure `vtkPiecewiseFunction` with opacity points (−1000→0.0, −200→0.0, 100→0.0, 200→0.05, 300→0.15, 500→0.5, 700→0.85, 1500→1.0) and `vtkColorTransferFunction` mapping from tissue-brown through bone-cream to dense-bone white.

### Multi-viewpoint camera sampling

**Fibonacci sphere sampling** produces near-uniform point distributions on a sphere and is the gold standard for multi-view capture. The algorithm distributes N points with approximately equal solid angle per point using golden-ratio increments for azimuth and uniform spacing in cos(θ) for elevation (González, arXiv:0912.4540). For objects with a natural "ground plane," restrict to an upper hemisphere with elevation range 10°–80° to avoid extreme top-down or grazing angles.

Typical view counts in the literature: NeRF training uses 50–100 views, 3DGS training 50–200+, and multi-view diffusion models (MVDream, Zero123) use 4–8 canonical views. For this pipeline, **32–64 views** provide an effective balance of coverage versus computation, with diminishing returns beyond ~62 views (arXiv:2505.24162 on 3D symmetry detection systematically tested [6, 14, 26, 42, 62, 86, 144] views).

In PyVista, set camera positions by computing Fibonacci sphere coordinates at a fixed radius from `grid.center`, then for each point set `plotter.camera.position`, `plotter.camera.focal_point = center`, `plotter.camera.up = (0,0,1)`, and `plotter.camera.view_angle = 30`. Capture each viewpoint with `plotter.screenshot()`.

### Setting up the conditioning signal ablation

Use four to seven configurations in an additive ablation: (A) depth only, (B) depth + normals, (C) depth + normals + edges, (D) depth + normals + edges + transfer-function color, and optionally (E) all buffers including AO and silhouette. Train the same ControlNet architecture with identical data and hyperparameters, changing only the conditioning input. Evaluate with PSNR (pixel-level; `skimage.metrics.peak_signal_noise_ratio`), SSIM (structural; `skimage.metrics.structural_similarity`), LPIPS (perceptual; `pip install lpips`, using AlexNet backbone, input normalized to [-1,1]; https://github.com/richzhang/PerceptualSimilarity), and FID for distributional quality (use clean-fid: `pip install clean-fid`; Parmar et al., CVPR 2022; https://github.com/GaParmar/clean-fid). Run each configuration with ≥3 random seeds and report mean ± 95% confidence intervals over ≥100 test image pairs.

---

## Section C: generating photorealistic ground truth via path tracing

### Mitsuba 3 is the clear winner for CT volume path tracing

Among the four candidates, **Mitsuba 3** (https://github.com/mitsuba-renderer/mitsuba3) is optimal for this use case. It provides full support for heterogeneous participating media via the `gridvolume` plugin, accepts numpy arrays directly through `mi.VolumeGrid(mi.TensorXf(array))`, offers GPU acceleration via CUDA/OptiX backends (`cuda_rgb` variant), and has an excellent Python API where scenes are defined as Python dictionaries via `mi.load_dict()`. Rendering is a single call: `mi.render(scene, spp=512)`. The `volpath` integrator handles volumetric path tracing with Woodcock tracking for heterogeneous media.

Blender Cycles is the runner-up, offering production-quality volumetric rendering with artistic control. It requires converting medical data to OpenVDB format, which **Bioxel Nodes** (https://github.com/OmooLab/BioxelNodes) automates—it imports DICOM/NIfTI directly into Blender and provides Geometry Nodes presets for medical visualization. Batch scripting uses `bpy` with `bpy.ops.render.render(write_still=True)`. However, the extra format-conversion step and heavier rendering overhead make Mitsuba 3 preferable for systematic batch generation.

Intel OSPRay (https://github.com/RenderKit/ospray) is CPU-focused with limited Python integration, making it unsuitable for rapid prototyping. NVIDIA OptiX is a low-level API requiring custom CUDA implementation of delta tracking—powerful but far too much engineering overhead for a thesis.

### Mapping HU values to volumetric optical properties

The standard approach is NOT to derive physically exact optical properties from X-ray attenuation. Instead, design a transfer function mapping HU → (extinction σ_t, albedo, color). Normalize HU to [0,1] via `(HU + 1000) / 4096.0`, clipped, then scale by a density multiplier (40–200, depending on volume physical size). Bone (HU > 300) should have high extinction and lower albedo (0.5–0.7); soft tissue (HU 0–300) gets medium extinction and higher albedo (0.8–0.9); air (HU < −500) gets near-zero extinction. The Render-FM paper (arXiv:2505.17338) confirms this transfer-function approach for generating ground truth from CT volumes. For phase functions, use Henyey-Greenstein with g = 0 (isotropic) for bone and g = 0.7–0.9 (forward scattering) for soft tissue. Key references include Nelson Max, "Optical Models for Direct Volume Rendering" (IEEE TVCG, 1995) and Dappa et al., "Cinematic rendering" (Insights into Imaging 2016;7(6):849–856, DOI:10.1007/s13244-016-0518-1).

### Mitsuba 3 scene configuration

A complete Mitsuba 3 scene for CT path tracing uses a `cube` shape with `null` BSDF (so light enters the volume freely), a `heterogeneous` interior medium referencing a `gridvolume` for spatially varying extinction, and a `volpath` integrator. The gridvolume's `to_world` transform should translate by (−0.5, −0.5, −0.5) and scale to center the volume at the origin. Gridvolume values must be in [0,1] for Woodcock tracking; use the `scale` parameter to multiply. Set albedo to 0.8 (uniform) or use a separate gridvolume for spatially varying albedo. For lighting, HDR environment maps from Polyhaven produce the most photorealistic cinematic rendering look, consistent with Siemens syngo.via cinematic rendering. A `perspective` sensor with `hdrfilm` output at 512×512 completes the setup.

### Samples per pixel and denoising for training data

For volumetric path tracing convergence: **1 SPP** is extremely noisy (usable only with neural denoisers), **128–256 SPP** is suitable for denoised final frames, **512 SPP** is the standard reference quality used in Mitsuba tutorials and the Render-FM paper, and **1024–4096 SPP** approaches convergence for complex scenes. Render at **512–1024 SPP** for ground truth training data, optionally applying Intel Open Image Denoise (OIDN; https://github.com/RenderKit/oidn) as a gentle cleanup. OIDN uses deep learning to approximate converged results from noisy input and achieves millisecond-scale denoising on GPU. At 512 SPP, remaining noise is minimal, so OIDN serves as cleanup rather than aggressive reconstruction.

Hofmann et al. ("Neural Denoising for Path Tracing of Medical Volumetric Data," ACM PACMCGIT/HPG 2020, DOI:10.1145/3406181; https://github.com/nihofm/ndptmvd) demonstrated that domain-specific denoising for medical VPT significantly outperforms generic denoisers, achieving good quality from 2–8 SPP. However, for training data, **avoid relying on heavily denoised low-SPP renders** as sole ground truth, since denoiser bias can introduce systematic artifacts.

### Camera alignment between PyVista and Mitsuba 3

VTK/PyVista camera parameters (position, focal_point, view_up, view_angle) map directly to Mitsuba 3's `look_at` transform: `mi.ScalarTransform4f().look_at(origin=position, target=focal_point, up=view_up)`. VTK's `view_angle` is vertical FOV in degrees; set Mitsuba's `fov_axis: 'y'` accordingly. Both systems are right-handed with Y-up, X-right, camera looking along −Z, so no coordinate transform is needed. For Blender (which uses Z-up), swap Y and Z axes and set the camera to look along −Z local.

Verify alignment by rendering identical geometry in both systems and computing a pixel-level difference image. Place spheres at volume corners as fiducial markers and confirm matching pixel positions. Target SSIM > 0.999 between PyVista's basic render and Mitsuba's equivalent unlit render as confirmation of correct alignment.

### Evaluating ground truth convergence

Use an SPP-doubling test: render at N and 2N SPP, compute SSIM between them; when SSIM > 0.999, the lower SPP is effectively converged. Watch for fireflies (fix with MIS integrator `volpathmis` or contribution clamping), color banding (use HDR EXR output, not 8-bit PNG), and dark regions (adjust the scale parameter). Always output in EXR format and tone-map to sRGB only for final display or training.

---

## Section D: ControlNet is the recommended architecture, with specific caveats

### Architecture comparison and selection rationale

**ControlNet on SD 1.5** (Zhang & Agrawala, ICCV 2023, arXiv:2302.05543; https://github.com/lllyasviel/ControlNet) is the recommended primary architecture. It creates a trainable copy of SD's encoder blocks connected via zero-initialized convolutions, ensuring the pretrained backbone is preserved during fine-tuning. It natively supports all required conditioning types (depth, normal, Canny), has mature SD 1.5 checkpoints (ControlNet 1.1 with 14 models), is robust with small datasets (<50K pairs), and has extensive community documentation. The ~361M trainable parameters can be trained on a single RTX 3090 with fp16 and gradient checkpointing (~20 GB VRAM).

**T2I-Adapter** (Mou et al., AAAI 2024, arXiv:2302.08453; https://github.com/TencentARC/T2I-Adapter) is the lightweight alternative at only ~77M parameters. It does not copy the encoder, instead learning small adapter networks. Lower control fidelity than ControlNet but significantly faster training—consider this if VRAM is constrained.

**Pix2pix** (Isola et al., CVPR 2017) is the essential baseline for paired image translation. Its U-Net generator with PatchGAN discriminator is computationally cheap and deterministic. Expect competitive structural fidelity but worse perceptual quality than diffusion models. **CycleGAN** (Zhu et al., ICCV 2017) is relevant only if unpaired data becomes necessary.

**InstructPix2Pix** (Brooks et al., CVPR 2023, arXiv:2211.09800) is NOT suitable—it's designed for semantic text-instruction edits, not precise geometric conditioning. **LoRA alone** cannot provide spatial conditioning; however, **LoRA + ControlNet** is a powerful combination where ControlNet handles structure while LoRA adapts appearance domain to medical rendering aesthetics.

Among 2024–2025 improvements: **ControlNet++** (arXiv:2404.07987) adds pixel-level cycle-consistency loss for improved conditioning fidelity. **ControlNet-XS** (Zavadski et al., ECCV 2024) achieves better FID with only **1% of base model parameters** (~14M for SD 1.5) via bidirectional encoder connections. **Ctrl-X** (Lin et al., NeurIPS 2024, arXiv:2406.07540) is training-free and guidance-free, supporting arbitrary conditioning modalities—valuable for rapid prototyping. **Perturbed-Attention Guidance (PAG)** (ECCV 2024; https://github.com/cvlab-kaist/Perturbed-Attention-Guidance) enhances sample quality without text prompts, directly addressing the medical domain's prompt problem. **IC-Light** (by Lvmin Zhang; https://github.com/lllyasviel/IC-Light) enables lighting manipulation, relevant for consistent illumination across views.

### Known limitations and mitigations

ControlNet's primary limitations for this pipeline are (1) **view inconsistency**—each view is generated independently with no cross-view coherence mechanism, (2) **hallucination** of details not present in the conditioning signal, (3) **domain gap**—SD was trained on natural images, not medical renderings, and (4) **stochastic outputs** inherent to diffusion sampling.

For view inconsistency, architectural solutions exist: **SyncDreamer** (Liu et al., ICLR 2024, arXiv:2309.03453) generates multi-view-consistent images by modeling the joint probability distribution of all views simultaneously using 3D-aware feature attention. **MVDream** (Shi et al., ICLR 2024, arXiv:2308.16512) learns generalizable 3D priors from both 2D and 3D data. **Zero-1-to-3** (Liu et al., ICCV 2023, arXiv:2303.11328) conditions on input image plus relative camera pose. A practical compromise: since the CT volume geometry is known, the conditioning signal (depth, normals) already enforces structural consistency; the remaining inconsistency is in texture/appearance, mitigatable by using **low classifier-free guidance (CFG) scale** (1.0–3.0), **high conditioning strength**, and **fixed random seeds** across views.

### Fine-tuning hyperparameters validated in the literature

The HuggingFace diffusers training script (`train_controlnet.py`) provides the reference implementation. Based on the original paper and community best practices:

- **Dataset size**: 1,000–5,000 paired renders should suffice given highly structured conditioning signals. The original paper demonstrated robustness with <50K pairs. Community reports indicate 200–500 pairs for proof-of-concept.
- **Learning rate**: **1e-5** (confirmed optimal by HuggingFace blog; range 1e-4 to 2e-6)
- **Batch size**: 4 per device with gradient accumulation of 1–4 for SD 1.5
- **Training steps**: ~50K steps; convergence can happen suddenly—monitor visually
- **Optimizer**: AdamW with betas=(0.9, 0.999), weight_decay=1e-2
- **Mixed precision**: fp16 strongly recommended (halves VRAM)
- **Prompt dropout**: **50% is critical**—randomly replacing half the prompts with empty strings enables prompt-free inference
- **Loss**: Standard diffusion noise-prediction MSE loss; optionally add Min-SNR weighting (--snr_gamma=5.0) for faster convergence
- **VRAM**: ~20 GB with fp16 + gradient checkpointing on SD 1.5; ~30–40 GB for SDXL
- **Training time**: ~12–48 hours on a single A100 or RTX 3090

### Text prompt strategy for out-of-distribution medical content

The recommended approach is **empty/null prompt** with PAG enhancement. ControlNet explicitly supports no-prompt generation—the paper shows coherent outputs across all conditioning types with empty strings. During training, the standard 50% prompt dropout enables this. At inference, use empty prompt "" with PAG (pag_scale=2.0–4.0, guidance_scale=0.0) via `StableDiffusionControlNetPAGPipeline` in diffusers. As a complementary approach, train a **LoRA adapter** (~1–6 MB) on the target photorealistic render domain to shift SD's output distribution toward medical rendering aesthetics without interfering with ControlNet's structural control.

### Expected metric values for this translation task

Based on comparable image-to-image translation benchmarks: **PSNR > 25 dB** is good, > 30 dB is excellent. **SSIM > 0.85** indicates good structural preservation for conditioned translation. **LPIPS < 0.2** is good, < 0.1 excellent (LPIPS is the single most reliable perceptual metric per recent studies). **FID < 50** is good, < 20 excellent. ControlNet-based methods on COCO achieve FID 20–60. For this CT task with strong geometric conditioning, target the good-to-excellent range on SSIM and LPIPS; PSNR may be lower due to the domain gap.

---

## Section E: 3D Gaussian Splatting from AI-generated multi-view images

### Reconstructing from generated rather than photographed images

When input images come from a diffusion model rather than cameras, three problems dominate: different views may show **contradictory textures** (hallucinated details that differ across angles), **inconsistent lighting** (each view may have different shadows and specular highlights), and COLMAP may fail to find correspondences in synthetic images lacking natural feature distributions. The SDS-based approaches (DreamGaussian, Tang et al., arXiv:2309.16653; GaussianDreamer, Yi et al., CVPR 2024, arXiv:2310.08529) bypass multi-view consistency by using diffusion models as optimization guidance. However, for this pipeline where you have pre-generated multi-view images with known camera poses, direct 3DGS optimization is more appropriate—you can **skip COLMAP entirely** by initializing from known camera poses and depth-based point clouds.

### Wonderland and its limitations for medical data

"Wonderland: Navigating 3D Scenes from a Single Image" (Liang et al., CVPR 2025, arXiv:2412.12091; project page: https://snap-research.github.io/wonderland/) uses a camera-guided video diffusion model with dual-branch Plücker embedding conditioning to generate 3D-aware video latents, which a Latent-based Large Reconstruction Model (LaLRM) converts to 3DGS in a feed-forward manner. Code is **not yet released** (GitHub repo at https://github.com/snap-research/wonderland contains only assets). For medical CT data, Wonderland has critical limitations: it was trained on natural scene datasets (RealEstate10K, DL3DV), creating a severe domain gap; it's designed for exterior scene navigation rather than interior volumetric structures; and its feed-forward architecture may lack the fine-grained control needed for anatomical accuracy. It is not recommended as a core component but could serve as an interesting comparison point.

### Choosing the right 3DGS variant

**Original 3DGS** (Kerbl et al., ACM TOG/SIGGRAPH 2023, arXiv:2308.04079; https://github.com/graphdeco-inria/gaussian-splatting) achieves PSNR ~27.43, SSIM ~0.814, LPIPS ~0.257 on Mip-NeRF 360 with ~23 minutes training time—the most robust and well-tested option. **2D Gaussian Splatting** (Huang et al., SIGGRAPH 2024, arXiv:2403.17888; https://github.com/hbb1/2d-gaussian-splatting) uses flat 2D disk primitives that naturally align with surfaces, making it **preferable for CT data where surface quality matters for anatomical structures**. **Mip-Splatting** (Yu et al., CVPR 2024 Best Student Paper, arXiv:2311.16493; https://github.com/autonomousvision/mip-splatting) adds anti-aliasing critical for medical visualization where users zoom across scales. **Gaussian Opacity Fields** (Yu et al., 2024, arXiv:2404.10772; https://github.com/autonomousvision/gaussian-opacity-fields) provides the best mesh extraction for unbounded scenes. **SuGaR** (Guédon & Lepetit, CVPR 2024, arXiv:2311.12775; https://github.com/Anttwo/SuGaR) enables mesh extraction and Blender integration for downstream editing.

For 32–128 synthetic views of a CT volume, the recommendation is **2DGS** (for surface quality) or **original 3DGS with Mip-Splatting** (for robust rendering quality and anti-aliasing).

### View count requirements and diminishing returns

With **8 views**, sparse-view methods like DNGaussian (Li et al., CVPR 2024, arXiv:2403.06912; https://github.com/Fictionarry/DNGaussian) achieve usable results with depth priors (PSNR ~20 dB on LLFF 3-view). At **32 views**, reconstruction approaches dense-view quality for bounded objects; **64 views** provides near-peak quality with clear diminishing returns beyond this point. At **128+ views**, improvements are marginal, primarily benefiting view-dependent effects. For this pipeline, **64 views is the recommended target**, with 32 as a minimum viable count. With fewer than 32 views, add monocular depth regularization using Depth Anything V2 as a geometric prior.

### Handling view inconsistency from AI-generated images

View-inconsistent inputs cause blurring (3DGS averages conflicting supervision), floaters (inconsistent depth cues spawn free-floating Gaussians), and popping artifacts (conflicting spherical harmonics cause abrupt appearance changes). Three mitigation strategies are essential:

**Per-image appearance embeddings** separate intrinsic (material) from dynamic (environmental) appearance per Gaussian. GS-W (Zhang et al., ECCV 2024, arXiv:2403.15704; https://eastbeanzhang.github.io/GS-W/) achieves this with **1000× faster rendering** than NeRF-W. WildGaussians (Kulhanek et al., NeurIPS 2024; https://wild-gaussians.github.io/) adds a DINO-based uncertainty predictor for transient artifact removal. Splatfacto-W (https://kevinxu02.github.io/splatfactow/) integrates into nerfstudio with **5.3 dB PSNR improvement** over vanilla 3DGS on in-the-wild data.

**Confidence-aware loss** (ReconX, NeurIPS 2024) down-weights per-pixel loss in regions likely to be hallucinated. **Depth regularization** using known CT geometry prevents floaters by constraining Gaussian positions to plausible surfaces.

### Real-time rendering performance

Original 3DGS achieves **≥100 FPS at 1080p** on high-end GPUs. On an RTX 4090, expect 200+ FPS for <1M Gaussians and 100–150 FPS for 1–2M. An RTX 3070 maintains 60+ FPS for scenes under ~1M Gaussians. For a single CT volume reconstruction from 32–128 views, expect **500K–2M Gaussians**, yielding comfortable real-time performance on any modern GPU. Model sizes range 50–200 MB uncompressed, compressible to 5–50 MB. Available viewers include the original SIBR viewer (OpenGL), gsplat (CUDA rasterization for nerfstudio), web-based viewers (GaussianSplats3D/WebGL), and game engine integrations (UnityGaussianSplatting).

---

## Section F: designing the evaluation and ablation framework

### Three ablation axes to test systematically

**Axis 1: Conditioning signals → enhancement quality.** Train identical ControlNet architecture with incrementally richer inputs: depth only → depth + normals → depth + normals + edges → all buffers. This follows the standard additive ablation pattern formalized as "Leave-One-Component-Out" (LOCO) by Sheikholeslami et al. (AutoAblation, EuroMLSys 2021) and described in Meyes et al. (arXiv:1901.08644, "Ablation Studies in Artificial Neural Networks"). Change only one factor at a time. Report PSNR, SSIM, LPIPS, and optionally FID. Run each configuration with ≥3 random seeds and report mean ± 95% confidence intervals over ≥100 test pairs.

**Axis 2: Training set size → model quality.** Train with 100, 500, 1,000, and 5,000 pairs. Plot log-log learning curves (training set size vs. error) and fit a power-law model: ε(m) ∝ α·m^β, where typical exponents range β = −0.07 to −0.35 (Hestness et al., arXiv:1712.00409, "Deep Learning Scaling is Predictable, Empirically"; Hoiem et al., "Learning Curves for Analysis of Deep Networks," ICML 2021). Identify the knee point where diminishing returns begin and report what fraction of full training data achieves 90% of full performance.

**Axis 3: Viewpoint coverage → 3DGS quality.** Test with 8, 16, 32, 64, and 128 input views. Evaluate PSNR/SSIM/LPIPS at held-out viewpoints. Key references for sparse-view benchmarking include CoR-GS (Zhang et al., ECCV 2024, arXiv:2405.12110) testing 3/6/9 views on DTU, and SparseGS testing 12/24 views on Mip-NeRF 360.

### Essential baselines and how to present them

Four baseline comparisons are non-negotiable for this thesis: (1) path-traced ground truth vs. AI-enhanced output (establishes upper bound), (2) simple render vs. AI-enhanced output (quantifies improvement—the core contribution), (3) ControlNet vs. pix2pix (diffusion vs. GAN baseline), and (4) AI-enhanced 3DGS vs. direct volume rendering (practical value proposition).

Present results in a table with rows = methods, columns = metrics (PSNR↑, SSIM↑, LPIPS↓, FID↓, inference time), best values bolded, with ± standard deviation. Include a visual comparison grid showing 3–5 representative examples (easy/medium/hard cases) as side-by-side crops: input → each method → ground truth. For statistical significance, use a **paired t-test** when differences are normally distributed (verify with Shapiro-Wilk test, p > 0.05), otherwise a **Wilcoxon signed-rank test**. Apply Bonferroni correction (α_adjusted = 0.05/k) when making multiple pairwise comparisons.

### User study protocol for photorealism assessment

Use **Two-Alternative Forced Choice (2AFC)**: show participants a reference path-traced image and two alternatives (AI-enhanced vs. simple render, or AI-enhanced vs. pix2pix baseline), asking "which looks more similar to the reference?" The LPIPS metric itself was calibrated using 484K 2AFC judgments (Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric," CVPR 2018; https://github.com/richzhang/PerceptualSimilarity). For absolute quality rating, use **Mean Opinion Score (MOS)** on a 1–5 scale per ITU-R BT.500.

**15–20 participants** is the minimum viable sample for a master's thesis (sufficient for 80% power to detect medium effects per Journal of Cognition tutorial, DOI:10.5334/joc.72). Run the study via a custom web interface (Flask + JavaScript) or Prolific.co for recruitment. Report inter-rater agreement using **Krippendorff's alpha** (α ≥ 0.80 indicates satisfactory reliability; tool at k-alpha.org; Zapf et al., BMC Med Res Methodology 2016, DOI:10.1186/s12874-016-0200-9). Convert 2AFC results to Thurstone scores and report preference rates with 95% binomial confidence intervals. Most universities classify image-viewing user studies as exempt from full IRB review, but still require informed consent and data anonymization.

### Proof-of-concept versus full systematic evaluation

A successful proof-of-concept thesis requires: 1 anatomy type (head/skull), 3 core metrics (PSNR, SSIM, LPIPS), 1–2 ablation axes (conditioning signals + data efficiency), and 2 baselines (simple render, pix2pix). The novel contribution is the pipeline itself plus quantitative evidence that it works. A full systematic evaluation adds: 2 anatomies (head + thorax), 5+ metrics, all 3 ablation axes with ≥3 seeds, 4+ baselines, a user study with 20+ participants, cross-dataset generalization testing, and comprehensive statistical significance analysis.

Pragmatic scoping for a ~6-month timeline: spend months 1–2 on pipeline implementation and data generation, month 3 on training the main model plus 1–2 baselines, month 4 on the conditioning ablation and data efficiency study, month 5 on viewpoint coverage ablation and optional user study, and month 6 on writing. **A thorough evaluation on 1 anatomy with 2 ablation axes and 2 baselines is far more valuable than a shallow evaluation on 3 anatomies with no ablations.**

### Recommended evaluation toolkit

For a unified metrics workflow, use **pyiqa** (https://github.com/chaofengc/IQA-PyTorch) which wraps 40+ metrics in a single interface: `metric = pyiqa.create_metric('lpips', device='cuda')`. For FID specifically, always use **clean-fid** (https://github.com/GaParmar/clean-fid) which addresses critical resizing artifacts that cause >6 FID difference between implementations. For individual metrics, torchmetrics provides PyTorch Lightning integration. Compute bootstrap confidence intervals by resampling test scores 10,000 times and reporting 2.5th/97.5th percentiles. For seed-level variance, report mean ± std across 3–5 training runs.

---

## Conclusion: critical reading order and highest-risk failure modes

The five papers to read first are: (1) **ControlNet** (Zhang & Agrawala, arXiv:2302.05543)—the core architecture, (2) **3D Gaussian Splatting** (Kerbl et al., arXiv:2308.04079)—the reconstruction method, (3) **Cinematic Rendering** (Dappa et al., Insights into Imaging 2016)—the domain and visual target, (4) **ControlNet++** (arXiv:2404.07987)—the likely upgrade path, and (5) **GS-W** (arXiv:2403.15704)—the solution for view-inconsistent 3DGS inputs.

The highest-risk failure modes are: **cross-view inconsistency** producing blurry or artifact-laden 3DGS reconstructions (mitigate with appearance embeddings and high conditioning scale), **domain gap** causing SD to hallucinate natural-image textures onto medical renders (mitigate with LoRA domain adaptation and empty prompts), **camera misalignment** between PyVista and Mitsuba producing misregistered training pairs (verify with fiducial markers and SSIM > 0.999), and **path-tracing convergence** issues where insufficient SPP produces noisy ground truth that trains the model to reproduce noise (validate with SPP-doubling SSIM tests). The pipeline's novelty lies precisely in combining these established components for a new application domain; the thesis contribution is demonstrating it works and characterizing when and why it fails.