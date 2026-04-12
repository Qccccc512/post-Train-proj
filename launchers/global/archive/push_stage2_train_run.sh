#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/content/post-Train-proj"
if [[ "${1:-}" != "" && "${1:0:2}" != "--" ]]; then
  REPO_ROOT="$1"
  shift
fi

RUN_NAME="${STAGE2_RUN_NAME:-}"
if [[ -z "${RUN_NAME}" && "${1:-}" != "" && "${1:0:2}" != "--" ]]; then
  RUN_NAME="$1"
  shift
fi

cd "${REPO_ROOT}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Export it before pushing Stage 2 run artifacts." >&2
  exit 1
fi

if [[ -z "${RUN_NAME}" ]]; then
  POINTER_FILE="${REPO_ROOT}/runs/_launcher/latest_stage2_train_run_name.txt"
  if [[ ! -f "${POINTER_FILE}" ]]; then
    echo "Run name not provided and no launcher pointer found: ${POINTER_FILE}" >&2
    exit 1
  fi
  RUN_NAME="$(<"${POINTER_FILE}")"
fi

if [[ ! -d "${REPO_ROOT}/runs/${RUN_NAME}" ]]; then
  echo "Run directory not found: ${REPO_ROOT}/runs/${RUN_NAME}" >&2
  exit 1
fi

if [[ -n "${HF_ENDPOINT:-}" ]]; then
  echo "HF_ENDPOINT is set to ${HF_ENDPOINT}; clearing it for official Hub upload."
  unset HF_ENDPOINT
fi

echo "========================================"
echo "Push Stage 2 Run Artifacts"
echo "========================================"
echo "Repo root: ${REPO_ROOT}"
echo "Run name: ${RUN_NAME}"
echo "Source dir: runs/${RUN_NAME}"
echo

bash scripts-for-colab/setup_colab.sh "${REPO_ROOT}"
python scripts/hf_repo_sync.py upload-run --run-name "${RUN_NAME}" "$@"
