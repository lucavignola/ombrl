#!/usr/bin/env bash
set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${1:-/cluster/home/lvignola/venvs/ombrl-sombrl}"

rm -rf "${REPO_ROOT}/ombrl.egg-info"
if [ -n "${PYTHONPATH:-}" ]; then
  PYTHONPATH="${PYTHONPATH//${REPO_ROOT}:/}"
  PYTHONPATH="${PYTHONPATH//:${REPO_ROOT}/}"
  export PYTHONPATH
fi

if [ -n "${VIRTUAL_ENV:-}" ]; then
  deactivate || true
  PATH="${PATH//${VIRTUAL_ENV}\/bin:/}"
  PATH="${PATH//:${VIRTUAL_ENV}\/bin/}"
  unset VIRTUAL_ENV
fi
hash -r

module load stack/2024-06
module load gcc/12.2.0
module load eth_proxy
module load python/3.11.6

python -m venv --clear "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"
hash -r

python -m pip install --upgrade pip setuptools wheel
python -m pip uninstall -y ombrl || true
if python -m pip show ombrl >/dev/null 2>&1; then
  echo "ERROR: stale ombrl package metadata is still visible in ${VIRTUAL_ENV}" >&2
  echo "Remove ${REPO_ROOT}/ombrl.egg-info and ensure PYTHONPATH does not contain ${REPO_ROOT} while bootstrapping." >&2
  python -m pip show ombrl >&2
  exit 1
fi

python -m pip install --no-cache-dir \
  --constraint "${REPO_ROOT}/utility_scripts/euler_constraints.txt" \
  "jax[cuda12]==0.4.34"

python -m pip install --no-cache-dir -r "${REPO_ROOT}/utility_scripts/euler_constraints.txt"

python -m pip install --no-cache-dir --constraint "${REPO_ROOT}/utility_scripts/euler_constraints.txt" \
  pandas \
  jaxtyping \
  gymnasium==0.29.1 \
  tensorboardX \
  tqdm \
  wandb \
  dm-control \
  mujoco \
  "maxinforl_jax @ git+https://github.com/sukhijab/maxinforl_jax.git"

python - <<'PY'
import importlib.metadata as md
for pkg in [
    "numpy",
    "jax",
    "jaxlib",
    "jax-cuda12-plugin",
    "jax-cuda12-pjrt",
    "nvidia-cudnn-cu12",
    "nvidia-cublas-cu12",
    "tensorflow-probability",
    "maxinforl_jax",
    "gymnasium",
]:
    try:
        print(pkg, md.version(pkg))
    except md.PackageNotFoundError as exc:
        raise SystemExit(f"Missing required package after install: {pkg}") from exc

import maxinforl_jax
print("maxinforl_jax import ok")
PY

echo
echo "Created SOMBRL Euler venv at: ${VENV_PATH}"
echo "Set this in utility_scripts/setup_sombrl_euler.bash:"
echo "export OMBRL_VENV=${VENV_PATH}"
echo
echo "OMBRL itself is intentionally not pip-installed; setup_sombrl_euler.bash"
echo "adds the repo root to PYTHONPATH so this reproduction env can omit"
echo "unrelated setup.py dependencies such as humanoid-bench and metaworld."
