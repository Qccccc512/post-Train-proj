#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/content/post-Train-proj"
if [[ "${1:-}" != "" && "${1:0:2}" != "--" ]]; then
  REPO_ROOT="$1"
  shift
fi

cd "${REPO_ROOT}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Export it before launching detached Stage 2 training." >&2
  exit 1
fi

LAUNCH_DIR="${REPO_ROOT}/runs/_launcher"
mkdir -p "${LAUNCH_DIR}"

STAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_PATH="${LAUNCH_DIR}/${STAMP}_stage2_train.log"
PID_PATH="${LAUNCH_DIR}/${STAMP}_stage2_train.pid"
CMD_PATH="${LAUNCH_DIR}/${STAMP}_stage2_train.cmd"

{
  printf 'bash scripts-for-colab/run_stage2_train.sh %q' "${REPO_ROOT}"
  for arg in "$@"; do
    printf ' %q' "${arg}"
  done
  printf '\n'
} > "${CMD_PATH}"

nohup bash scripts-for-colab/run_stage2_train.sh "${REPO_ROOT}" "$@" > "${LOG_PATH}" 2>&1 &
PID=$!
echo "${PID}" > "${PID_PATH}"

echo "Detached Stage 2 training launched."
echo "PID: ${PID}"
echo "Log: ${LOG_PATH}"
echo "PID file: ${PID_PATH}"
echo "Command file: ${CMD_PATH}"
echo "Latest run name will be written to:"
echo "  ${LAUNCH_DIR}/latest_stage2_train_run_name.txt"
echo "Watch logs with:"
echo "  tail -f ${LOG_PATH}"
echo "Stop it with:"
echo "  kill ${PID}"
