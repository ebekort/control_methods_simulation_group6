#!/bin/bash
#
# One-time setup of the training venv. Run on the login node (compute nodes
# may have no internet):
#
#   bash hpc/setup_hpc.sh
#
# uv downloads its own Python, so the venv does not depend on a Python
# module, and the jax[cuda12] wheels bring the CUDA libraries, so no CUDA
# module is needed either. The versions match the local PC so that runs are
# comparable. They need an NVIDIA driver that supports CUDA 12: the job log
# prints the driver version, and if JAX falls back to the CPU, an older
# driver is the first suspect.
#
# Settings are environment variables with the defaults below.

VENV=${VENV:-$HOME/venvs/pinn}
PYTHON=${PYTHON:-3.13}
JAX=${JAX:-jax[cuda12]==0.12.0.1}
OPTAX=${OPTAX:-optax==0.2.8}
NUMPY=${NUMPY:-numpy==2.5.4}

set -euo pipefail

if ! command -v uv >/dev/null; then
    # official installer, puts uv in ~/.local/bin
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
export UV_LINK_MODE=copy  # cache and venv may be on different filesystems

mkdir -p "$(dirname "$VENV")"
uv venv --python "$PYTHON" "$VENV"
# --no-cache: the CUDA wheels are a few GB, too much for a home quota
uv pip install --no-cache --python "$VENV/bin/python" \
    "$JAX" "$OPTAX" "$NUMPY"

# the login node has no GPU driver, so the CUDA plugin is skipped here; the
# GPU is checked by hpc_train.sh on the compute node
JAX_PLATFORMS=cpu "$VENV/bin/python" - <<'EOF'
from importlib.metadata import version
import jax
import optax
print(*(f'{p} {version(p)}' for p in ('jax', 'jaxlib', 'optax', 'numpy')))
print('devices:', jax.devices())
EOF
echo "venv ready: $VENV"
