#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniforge3}"
BENCHMARK_ENV_NAME="post-train-benchmark"
LOCK_FILE="${REPO_ROOT}/requirements-benchmark.lock.txt"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"

# Bootstrap benchmark frameworks (idempotent)
python "${REPO_ROOT}/scripts/bench/bootstrap_benchmarks.py"

install_from_lock_file() {
    if [[ ! -f "${LOCK_FILE}" ]]; then
        echo "Benchmark lock file is missing: ${LOCK_FILE}" >&2
        exit 1
    fi
    echo "Installing benchmark dependencies from ${LOCK_FILE}..."
    python -m pip install --use-deprecated=legacy-resolver -r "${LOCK_FILE}"
}

# Check if conda env exists
if conda env list | grep -q "^${BENCHMARK_ENV_NAME} "; then
    echo "Conda environment '${BENCHMARK_ENV_NAME}' already exists."
    echo "To recreate, run: conda env remove -n ${BENCHMARK_ENV_NAME}"
    echo "Activating existing environment..."
    conda activate "${BENCHMARK_ENV_NAME}"
    if ! python -c "import vllm, lm_eval, bitsandbytes" >/dev/null 2>&1; then
        echo "Existing benchmark environment is missing required packages; reconciling from lock file."
        install_from_lock_file
    fi
else
    echo "Creating conda environment '${BENCHMARK_ENV_NAME}'..."
    conda create -n "${BENCHMARK_ENV_NAME}" python=3.11 -y
    conda activate "${BENCHMARK_ENV_NAME}"
    install_from_lock_file
fi

if ! python -c "import bitsandbytes" >/dev/null 2>&1; then
    echo "bitsandbytes is still missing after lock-file installation." >&2
    exit 1
fi

echo
echo "Benchmark conda environment is ready: ${BENCHMARK_ENV_NAME}"
echo "Activate with: conda activate ${BENCHMARK_ENV_NAME}"
echo
echo "Note: You may see a warning about opencv-python-headless requiring numpy>=2."
echo "      This can be safely ignored - opencv is an optional transitive dependency"
echo "      and is not used in our evaluation pipeline."
