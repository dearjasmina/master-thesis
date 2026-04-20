#!/usr/bin/env bash
set -e

PYTHON=${PYTHON:-python3.10}
VENV_DIR="venv"

# ── Detect CUDA before touching Python ───────────────────────────────────────
# Uses nvidia-smi (always present when a GPU driver is installed).
# Falls back to nvcc if smi is missing but toolkit is present.
has_cuda() {
    command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null && return 0
    command -v nvcc &>/dev/null && return 0
    return 1
}

if has_cuda; then
    CUDA=true
    # Read CUDA major.minor from nvcc or nvidia-smi
    CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || \
               nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || \
               echo "12.1")
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    # Map to a torch whl tag (cu118 / cu121 / cu124)
    if   [ "$CUDA_MAJOR" -le 11 ]; then TORCH_INDEX="cu118"
    elif [ "$CUDA_MAJOR" -eq 12 ]; then TORCH_INDEX="cu121"
    else                                 TORCH_INDEX="cu124"
    fi
    echo "CUDA $CUDA_VER detected — will install GPU PyTorch (index: $TORCH_INDEX)"
else
    CUDA=false
    echo "No CUDA detected — will install CPU PyTorch"
fi

# ── 1. Create venv ────────────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtualenv with $PYTHON ..."
    $PYTHON -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

# ── 2. PyTorch ────────────────────────────────────────────────────────────────
if [ "$CUDA" = true ]; then
    echo "Installing GPU PyTorch ..."
    pip install torch torchvision \
        --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}" --quiet
else
    echo "Installing CPU PyTorch ..."
    pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu --quiet
fi

# ── 3. Main requirements ──────────────────────────────────────────────────────
echo "Installing requirements ..."
pip install -r requirements.txt --quiet

# ── 4. CUDA-only extras ───────────────────────────────────────────────────────
if [ "$CUDA" = true ]; then
    echo "Installing CUDA extras (xformers) ..."
    pip install xformers --quiet
fi

# ── 5. Smoke test ─────────────────────────────────────────────────────────────
echo ""
echo "Running smoke test ..."
python - <<'PYEOF'
import sys, importlib

pkgs = [
    ("torch",        lambda m: m.__version__),
    ("pyvista",      lambda m: m.__version__),
    ("nibabel",      lambda m: m.__version__),
    ("SimpleITK",    lambda m: m.Version_VersionString()),
    ("diffusers",    lambda m: m.__version__),
    ("transformers", lambda m: m.__version__),
    ("mitsuba",      lambda _: "ok"),
    ("lpips",        lambda _: "ok"),
    ("cleanfid",     lambda _: "ok"),
    ("pyiqa",        lambda m: m.__version__),
    ("cv2",          lambda m: m.__version__),
    ("matplotlib",   lambda m: m.__version__),
]

ok = True
for name, ver_fn in pkgs:
    try:
        m = importlib.import_module(name)
        print(f"  ok  {name:<20} {ver_fn(m)}")
    except Exception as e:
        print(f"  FAIL {name:<20} {e}")
        ok = False

import torch
cuda = torch.cuda.is_available()
print(f"\n  CUDA available: {cuda}")
if cuda:
    print(f"  GPU:            {torch.cuda.get_device_name(0)}")

print()
if ok:
    print("All imports OK.")
else:
    print("Some imports failed — check errors above.")
    sys.exit(1)
PYEOF
