#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/content/post-Train-proj}"
VENV_DIR="${BENCHMARK_VENV_DIR:-${REPO_ROOT}/.venv}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"
SETUP_MODE="${BENCHMARK_SETUP_MODE:-global}"
LOCK_FILE="${REPO_ROOT}/requirements-benchmark.lock.txt"

cd "${REPO_ROOT}"

install_benchmark_packages() {
  local python_bin="$1"
  if [[ ! -f "${LOCK_FILE}" ]]; then
    echo "Benchmark lock file is missing: ${LOCK_FILE}" >&2
    exit 1
  fi
  "${python_bin}" -m pip install --use-deprecated=legacy-resolver -r "${LOCK_FILE}"
}

echo "=> Installing benchmark dependencies..."
echo "Repo root: ${REPO_ROOT}"
echo "Benchmark setup mode: ${SETUP_MODE}"
echo "Benchmark lock file: ${LOCK_FILE}"
"${BOOTSTRAP_PYTHON}" --version

create_benchmark_venv() {
  if "${BOOTSTRAP_PYTHON}" -m venv "${VENV_DIR}"; then
    return 0
  fi

  echo "=> python -m venv failed; falling back to virtualenv..."
  rm -rf "${VENV_DIR}"
  "${BOOTSTRAP_PYTHON}" -m pip install --upgrade virtualenv
  "${BOOTSTRAP_PYTHON}" -m virtualenv "${VENV_DIR}"
}

ensure_benchmark_venv() {
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "=> Creating benchmark venv..."
    create_benchmark_venv
    return 0
  fi

  if ! "${VENV_DIR}/bin/python" -m pip --version >/dev/null 2>&1; then
    echo "=> Existing benchmark venv is incomplete (pip missing); recreating..."
    rm -rf "${VENV_DIR}"
    create_benchmark_venv
  fi
}

if [[ "${SETUP_MODE}" == "venv" ]]; then
  echo "Benchmark venv: ${VENV_DIR}"
  ensure_benchmark_venv
  install_benchmark_packages "${VENV_DIR}/bin/python"
  echo "=> Bootstrapping benchmark frameworks..."
  "${VENV_DIR}/bin/python" scripts/bench/bootstrap_benchmarks.py
  echo "Colab benchmark setup completed."
  echo "Benchmark runtime: ${VENV_DIR}"
elif [[ "${SETUP_MODE}" == "global" ]]; then
  echo "Benchmark python: ${BOOTSTRAP_PYTHON}"
  install_benchmark_packages "${BOOTSTRAP_PYTHON}"
  echo "=> Bootstrapping benchmark frameworks..."
  "${BOOTSTRAP_PYTHON}" scripts/bench/bootstrap_benchmarks.py
  echo "Colab benchmark setup completed."
  echo "Benchmark runtime: current global Python"
else
  echo "Unsupported BENCHMARK_SETUP_MODE: ${SETUP_MODE}" >&2
  exit 1
fi
