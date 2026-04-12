#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="${1:-${REPO_ROOT_DEFAULT}}"

cd "${REPO_ROOT}"
bash scripts/bench/setup_benchmark_env.sh
