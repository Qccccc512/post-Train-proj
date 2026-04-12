#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="${1:-${REPO_ROOT_DEFAULT}}"
CONDA_ROOT_DEFAULT="${HOME}/miniforge3"
CONDA_ROOT="${CONDA_ROOT:-${CONDA_ROOT_DEFAULT}}"
ENV_NAME="${2:-post-train-local}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
MINIFORGE_URL="${MINIFORGE_URL:-https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh}"

if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
  echo "Miniforge not found at ${CONDA_ROOT}. Installing it now..."
  INSTALLER="/tmp/miniforge-installer.sh"
  curl -L "${MINIFORGE_URL}" -o "${INSTALLER}"
  bash "${INSTALLER}" -b -p "${CONDA_ROOT}"
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
fi

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
conda activate "${ENV_NAME}"
export PYTHONNOUSERSITE=1

echo "Using python: $(which python)"
echo "Using pip: $(python -s -m pip --version)"

python -s -m pip install --upgrade pip

LOCKFILE="${REPO_ROOT}/requirements-colab.lock.txt"
BASE_LOCKFILE="/tmp/post-train-local-base-lock.txt"
grep -v '^causal_conv1d==' "${LOCKFILE}" > "${BASE_LOCKFILE}"
CAUSAL_CONV_VERSION="$(awk -F'==' '/^causal_conv1d==/ {print $2}' "${LOCKFILE}" | tail -n 1)"

resolve_cuda_home_from_nvcc() {
  local nvcc_bin="$1"
  local candidate
  candidate="$(cd "$(dirname "${nvcc_bin}")/.." && pwd)"
  if [[ -x "${candidate}/bin/nvcc" ]]; then
    printf '%s\n' "${candidate}"
    return 0
  fi
  return 1
}

python -s -m pip install -r "${BASE_LOCKFILE}"

if [[ -n "${CAUSAL_CONV_VERSION}" ]] && command -v nvcc >/dev/null 2>&1; then
  NVCC_BIN="$(command -v nvcc)"
  CUDA_HOME_CANDIDATE="$(resolve_cuda_home_from_nvcc "${NVCC_BIN}" || true)"
  if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    DETECTED_ARCH="$(python - <<'PY'
import torch

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(f"{major}.{minor}")
PY
)"
    if [[ -n "${DETECTED_ARCH}" ]]; then
      export TORCH_CUDA_ARCH_LIST="${DETECTED_ARCH}"
      echo "Detected local GPU arch ${TORCH_CUDA_ARCH_LIST}; constraining causal_conv1d build to this architecture."
    fi
  fi
  if [[ -n "${CUDA_HOME_CANDIDATE}" ]]; then
    export CUDA_HOME="${CUDA_HOME_CANDIDATE}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
    echo "Using CUDA toolkit from ${CUDA_HOME} for causal_conv1d build."
  else
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    echo "Warning: detected nvcc at ${NVCC_BIN}, but could not derive a valid CUDA_HOME from it."
    echo "Attempting causal_conv1d build with the current PATH only."
  fi
  if python -s -m pip install --no-build-isolation "causal_conv1d==${CAUSAL_CONV_VERSION}"; then
    echo "Installed full high-performance dependency stack."
  else
    echo
    echo "Warning: failed to build optional accelerator causal_conv1d on the local runtime."
    echo "Continuing without it because flash-linear-attention provides Triton conv1d implementations."
  fi
elif [[ -z "${CAUSAL_CONV_VERSION}" ]]; then
  echo
  echo "No causal_conv1d pin found in ${LOCKFILE}. Skipping local source build."
else
  echo
  echo "Local fallback: nvcc was not found, so causal_conv1d cannot be built from source here."
  echo "Keeping the rest of the stack installed without causal_conv1d."
fi

echo
echo "Conda environment is ready."
echo "Activate with:"
echo "  source ${CONDA_ROOT}/etc/profile.d/conda.sh && conda activate ${ENV_NAME}"
