#!/usr/bin/env bash
set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${1:-/cluster/home/lvignola/venvs/ombrl-sombrl}"

module load stack/2024-06
module load gcc/12.2.0
module load eth_proxy
module load python/3.11.6

python -m venv --clear "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
pip uninstall -y ombrl || true

pip install --no-cache-dir -r "${REPO_ROOT}/utility_scripts/euler_constraints.txt"

pip install --no-cache-dir --constraint "${REPO_ROOT}/utility_scripts/euler_constraints.txt" \
  pandas \
  jaxtyping \
  gymnasium==0.29.1 \
  tensorboardX \
  tqdm \
  wandb \
  dm-control \
  mujoco \
  "maxinforl_jax @ git+https://github.com/sukhijab/maxinforl_jax.git"

echo
echo "Created SOMBRL Euler venv at: ${VENV_PATH}"
echo "Set this in utility_scripts/setup_sombrl_euler.bash:"
echo "export OMBRL_VENV=${VENV_PATH}"
echo
echo "OMBRL itself is intentionally not pip-installed; setup_sombrl_euler.bash"
echo "adds the repo root to PYTHONPATH so this reproduction env can omit"
echo "unrelated setup.py dependencies such as humanoid-bench and metaworld."
