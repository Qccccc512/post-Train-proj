#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/content/post-Train-proj"

if [[ "${1:-}" != "" && "${1:0:2}" != "--" ]]; then
  REPO_ROOT="$1"
  shift
fi

cd "${REPO_ROOT}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Export it before running training." >&2
  exit 1
fi

bash scripts-for-colab/setup_colab.sh "${REPO_ROOT}"
python scripts/train_sft.py "$@"
