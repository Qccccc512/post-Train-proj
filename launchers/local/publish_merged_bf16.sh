#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="${REPO_ROOT_DEFAULT}"
if [[ "${1:-}" != "" && "${1:0:2}" != "--" ]]; then
  REPO_ROOT="$1"
  shift
fi

RUN_ARGS=()
SKIP_SETUP=0
for arg in "$@"; do
  if [[ "${arg}" == "--skip-setup" ]]; then
    SKIP_SETUP=1
    continue
  fi
  RUN_ARGS+=("${arg}")
done

CONDA_ROOT_DEFAULT="${HOME}/miniforge3"
CONDA_ROOT="${CONDA_ROOT:-${CONDA_ROOT_DEFAULT}}"
TRAIN_ENV_NAME="${TRAIN_ENV_NAME:-post-train-local}"

source_conda() {
  if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi
  echo "conda is not available and conda.sh was not found under ${CONDA_ROOT}" >&2
  exit 1
}

cd "${REPO_ROOT}"

if [[ "${SKIP_SETUP}" != "1" ]]; then
  bash launchers/local/setup_train_env.sh "${REPO_ROOT}" "${TRAIN_ENV_NAME}"
fi

source_conda

echo "========================================"
echo "Publish Merged BF16"
echo "========================================"
echo "Repo root: ${REPO_ROOT}"
echo "Runtime mode: conda (${TRAIN_ENV_NAME})"
echo

CMD=(
  conda run
  --no-capture-output
  -n "${TRAIN_ENV_NAME}"
  python
  -u
  scripts/hub/publish_merged_bf16_from_adapter_repo.py
)
echo "+ ${CMD[*]} ${RUN_ARGS[*]}"
"${CMD[@]}" "${RUN_ARGS[@]}"
