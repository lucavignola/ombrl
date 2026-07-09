#!/usr/bin/env bash

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile
fi

OMBRL_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${OMBRL_REPO_ROOT}:${PYTHONPATH:-}"

export XLA_FLAGS=--xla_gpu_triton_gemm_any=true
export WANDB_CACHE_DIR=/cluster/scratch/lvignola/wandb
export MUJOCO_GL=osmesa
export WANDB_API_KEY='your_key'
export OMBRL_VENV=/cluster/home/lvignola/path/to/venv

module load stack/2024-06
module load gcc/12.2.0
module load eth_proxy

module load libx11/1.8.4-ns5x2da
module load libxrandr/1.5.3-acspwjp
module load libxinerama/1.1.3
module load libxcursor/1.2.1
module load libxi/1.7.6-qeazdpn
module load mesa/23.0.3
module load libxrender/0.9.10-kss2t7k
module load libxext/1.3.3-e74gj2z
module load libxfixes/5.0.2-5fbeidb

module load python/3.11.6

if [ -n "${OMBRL_VENV:-}" ]; then
  source "${OMBRL_VENV}/bin/activate"
fi

python - <<'PY'
import importlib.metadata as md
pkgs = ["jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt"]
versions = {}
for pkg in pkgs:
    try:
        versions[pkg] = md.version(pkg)
    except md.PackageNotFoundError:
        pass

jaxlib_version = versions.get("jaxlib")
plugin_version = versions.get("jax-cuda12-plugin")
jax_version = versions.get("jax")
if jaxlib_version and plugin_version and jaxlib_version != plugin_version:
    raise RuntimeError(
        "JAX CUDA package mismatch: "
        f"jaxlib=={jaxlib_version}, jax-cuda12-plugin=={plugin_version}. "
        "Reinstall matching JAX CUDA packages in the venv."
    )
if jax_version:
    major, minor, *_ = [int(part) for part in jax_version.split(".")[:2]]
    if (major, minor) >= (0, 5):
        raise RuntimeError(
            "This repo's jaxrl/tensorflow-probability stack is not compatible "
            f"with jax=={jax_version}. Install the pinned JAX 0.4.x CUDA stack."
        )
print("JAX packages:", versions)
PY
