#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${1:-}" != "" && "${1:0:2}" != "--" ]]; then
  REPO_ROOT="$1"
  shift
fi

SKIP_SETUP=0
for arg in "$@"; do
  if [[ "${arg}" == "--skip-setup" ]]; then
    SKIP_SETUP=1
    break
  fi
done

if [[ "${SKIP_SETUP}" != "1" ]]; then
  bash launchers/local/setup_train_env.sh "${REPO_ROOT}" "${TRAIN_ENV_NAME:-post-train-local}"
  bash launchers/local/setup_bench_env.sh "${REPO_ROOT}"
fi

export STAGE2_FINAL_TRAIN_RUNTIME=conda
export STAGE2_FINAL_BENCHMARK_RUNTIME=conda
bash launchers/global/run_benchmark.sh "${REPO_ROOT}" --skip-setup "$@"
