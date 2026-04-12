#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/content/post-Train-proj"
if [[ "${1:-}" != "" && "${1:0:2}" != "--" ]]; then
  REPO_ROOT="$1"
  shift
fi

cd "${REPO_ROOT}"

RUN_ARGS=()
SKIP_SETUP=0
for arg in "$@"; do
  if [[ "${arg}" == "--skip-setup" ]]; then
    SKIP_SETUP=1
    continue
  fi
  RUN_ARGS+=("${arg}")
done

if [[ "${SKIP_SETUP}" != "1" ]]; then
  bash launchers/global/setup_train_env.sh "${REPO_ROOT}"
fi

echo "========================================"
echo "Publish Merged BF16"
echo "========================================"
echo "Repo root: ${REPO_ROOT}"
echo "Runtime mode: global"
echo

CMD=(python -u scripts/hub/publish_merged_bf16_from_adapter_repo.py)
echo "+ ${CMD[*]} ${RUN_ARGS[*]}"
"${CMD[@]}" "${RUN_ARGS[@]}"
