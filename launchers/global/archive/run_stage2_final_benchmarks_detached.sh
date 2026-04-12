#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/content/post-Train-proj"
if [[ "${1:-}" != "" && "${1:0:2}" != "--" ]]; then
  REPO_ROOT="$1"
  shift
fi

RUN_NAME="${STAGE2_FINAL_RUN_NAME:-}"
if [[ -z "${RUN_NAME}" && "${1:-}" != "" && "${1:0:2}" != "--" ]]; then
  RUN_NAME="$1"
  shift
fi

cd "${REPO_ROOT}"

normalize_run_name() {
  local raw="$1"
  raw="${raw%/}"
  raw="${raw##*/runs/}"
  raw="${raw##*/}"
  printf '%s\n' "${raw}"
}

if [[ -n "${RUN_NAME}" ]]; then
  RUN_NAME="$(normalize_run_name "${RUN_NAME}")"
fi

LAUNCH_DIR="${REPO_ROOT}/runs/_launcher"
mkdir -p "${LAUNCH_DIR}"

STAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_PATH="${LAUNCH_DIR}/${STAMP}_stage2_final_benchmark.log"
PID_PATH="${LAUNCH_DIR}/${STAMP}_stage2_final_benchmark.pid"
CMD_PATH="${LAUNCH_DIR}/${STAMP}_stage2_final_benchmark.cmd"

{
  printf 'bash scripts-for-colab/run_stage2_final_benchmarks.sh %q' "${REPO_ROOT}"
  if [[ -n "${RUN_NAME}" ]]; then
    printf ' %q' "${RUN_NAME}"
  fi
  for arg in "$@"; do
    printf ' %q' "${arg}"
  done
  printf '\n'
} > "${CMD_PATH}"

if [[ -n "${RUN_NAME}" ]]; then
  nohup bash scripts-for-colab/run_stage2_final_benchmarks.sh "${REPO_ROOT}" "${RUN_NAME}" "$@" > "${LOG_PATH}" 2>&1 &
else
  nohup bash scripts-for-colab/run_stage2_final_benchmarks.sh "${REPO_ROOT}" "$@" > "${LOG_PATH}" 2>&1 &
fi
PID=$!
echo "${PID}" > "${PID_PATH}"

echo "Detached Stage 2 final benchmark launched."
echo "PID: ${PID}"
echo "Log: ${LOG_PATH}"
echo "PID file: ${PID_PATH}"
echo "Command file: ${CMD_PATH}"
echo "Watch logs with:"
echo "  tail -f ${LOG_PATH}"
echo "Stop it with:"
echo "  kill ${PID}"
