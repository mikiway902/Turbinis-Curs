#!/usr/bin/env bash
# User-local WSL2 JAX CUDA environment for Turbinis-Curs.
#
# This script never uses sudo/apt and never modifies system Python.  It creates
# (or reuses, only with explicit consent) one virtual environment under the
# current user's home directory and installs the optional JAX CUDA wheels
# there.  It expects the WSL NVIDIA driver plumbing (nvidia-smi and
# /usr/lib/wsl/lib/libcuda.so.1) to be provided by the host/WSL installation.
#
# Usage, inside WSL from the repository root:
#   bash scripts/setup_wsl_jax_cuda.sh
#   REUSE_EXISTING=1 bash scripts/setup_wsl_jax_cuda.sh
#
# Overrides:
#   VENV=~/.venvs/turbinis-gpu
#   PYTHON=python3
#   JAX_VERSION=0.12.0.1
#   JAX_EXTRA=cuda12
#
# The default CUDA 12 extra bundles the CUDA runtime libraries in user space
# and only uses the WSL-provided driver library.  No wheel source is piped to a
# shell; installation goes through the venv's pip resolver.

set -euo pipefail

JAX_VERSION="${JAX_VERSION:-0.12.0.1}"
JAX_EXTRA="${JAX_EXTRA:-cuda12}"
VENV="${VENV:-$HOME/.venvs/turbinis-gpu}"
PYTHON="${PYTHON:-python3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Turbinis-Curs WSL JAX CUDA setup (user-local)"
echo "Repository: $REPO_ROOT"
echo "Venv:       $VENV"
echo "JAX:        jax[$JAX_EXTRA]==$JAX_VERSION"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version \
        --format=csv,noheader || true
else
    echo "WARNING: nvidia-smi not found on PATH; GPU validation will report it." >&2
fi
if [ -e /usr/lib/wsl/lib/libcuda.so.1 ]; then
    echo "Driver library: /usr/lib/wsl/lib/libcuda.so.1 present"
else
    echo "WARNING: /usr/lib/wsl/lib/libcuda.so.1 not found; GPU may fail." >&2
fi

if [ -e "$VENV" ]; then
    if [ ! -x "$VENV/bin/python" ]; then
        echo "Refusing to modify existing non-venv path: $VENV" >&2
        echo "Inspect it or choose a different VENV=... path." >&2
        exit 2
    fi
    echo "Existing venv detected; inspecting without changing it:"
    "$VENV/bin/python" - <<'PY'
import platform
import sys

print("  executable:", sys.executable)
print("  python:    ", sys.version.split()[0])
print("  platform:  ", platform.platform())
PY
    if [ "${REUSE_EXISTING:-0}" != "1" ]; then
        echo "Leaving the existing venv unchanged."
        echo "Set REUSE_EXISTING=1 to install/update JAX in this venv."
        exit 0
    fi
    echo "REUSE_EXISTING=1: installing/updating JAX in the existing venv."
else
    "$PYTHON" - <<'PY'
import sys
import venv  # noqa: F401

if sys.version_info < (3, 12):
    raise SystemExit(
        "JAX 0.12 requires Python >= 3.12; use PYTHON=/path/to/python3.12 "
        "or create a managed Python with uv first"
    )
print("Creating venv with Python", sys.version.split()[0])
PY
    "$PYTHON" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install --no-cache-dir \
    "jax[$JAX_EXTRA]==$JAX_VERSION"

echo
echo "Installed package versions:"
"$VENV/bin/python" -m pip list 2>/dev/null \
    | grep -E '^(jax|jaxlib|jax-cuda|nvidia-)' || true

echo
echo "Explicit GPU self-test (jax.devices('gpu')):"
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
    "$VENV/bin/python" - <<'PY'
import jax
import jax.numpy as jnp

print("jax", jax.__version__)
print("x64", jax.config.jax_enable_x64)
print("all devices", jax.devices())
gpu_devices = jax.devices("gpu")
print("gpu devices", gpu_devices)
jax.config.update("jax_enable_x64", True)
value = jnp.arange(8, dtype=jnp.float64)
print("gpu dtype", value.dtype, "sum", float(value.sum().block_until_ready()))
PY

echo
echo "Validation command:"
echo "  \"$VENV/bin/python\" \"$REPO_ROOT/scripts/validate_backends.py\" --devices cpu,gpu"
echo "No system packages, sudo, or user data outside $VENV were changed."
