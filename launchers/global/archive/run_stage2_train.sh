#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/content/post-Train-proj"
if [[ "${1:-}" != "" && "${1:0:2}" != "--" ]]; then
  REPO_ROOT="$1"
  shift
fi

TRAIN_CONFIG="${STAGE2_TRAIN_CONFIG:-configs/train/stage2_qwen3_8b_lora.yaml}"
if [[ "${1:-}" != "" && "${1:0:2}" != "--" ]]; then
  TRAIN_CONFIG="$1"
  shift
fi

cd "${REPO_ROOT}"

contains_flag() {
  local needle="$1"
  shift
  for arg in "$@"; do
    if [[ "${arg}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

extract_option_value() {
  local option="$1"
  shift
  local previous=""
  for arg in "$@"; do
    if [[ "${previous}" == "${option}" ]]; then
      printf '%s\n' "${arg}"
      return 0
    fi
    previous="${arg}"
  done
  return 1
}

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Export it before running Stage 2 training." >&2
  exit 1
fi

if [[ ! -f "${TRAIN_CONFIG}" ]]; then
  echo "Train config not found: ${TRAIN_CONFIG}" >&2
  exit 1
fi

DATASET_CONFIG="${STAGE2_DATASET_CONFIG:-configs/datasets/stage2_final_fixed_60k.yaml}"
GROUP="${STAGE2_GROUP:-S2}"
PHASE="${STAGE2_PHASE:-stage2final}"
AUTO_UPLOAD="${STAGE2_AUTO_UPLOAD:-1}"
EXTRA_ARGS=("$@")

USER_PHASE="$(extract_option_value --phase "${EXTRA_ARGS[@]}" || true)"
if [[ -n "${USER_PHASE}" ]]; then
  PHASE="${USER_PHASE}"
fi

RUN_STAMP="$(date '+%Y%m%d_%H%M%S')"
CONFIG_SLUG="$(basename "${TRAIN_CONFIG}" .yaml)"
RUN_LABEL="${STAGE2_TRAIN_LABEL:-stage2train_${RUN_STAMP}}"
RUN_NAME="${STAGE2_RUN_NAME:-${RUN_LABEL}_${CONFIG_SLUG}}"
USER_RUN_NAME="$(extract_option_value --run-name "${EXTRA_ARGS[@]}" || true)"
if [[ -n "${USER_RUN_NAME}" ]]; then
  RUN_NAME="${USER_RUN_NAME}"
fi

LAUNCH_DIR="${REPO_ROOT}/runs/_launcher"
mkdir -p "${LAUNCH_DIR}"
printf '%s\n' "${RUN_NAME}" > "${LAUNCH_DIR}/latest_stage2_train_run_name.txt"
cat > "${LAUNCH_DIR}/latest_stage2_train_launch.env" <<EOF
RUN_NAME=${RUN_NAME}
TRAIN_CONFIG=${TRAIN_CONFIG}
DATASET_CONFIG=${DATASET_CONFIG}
GROUP=${GROUP}
PHASE=${PHASE}
AUTO_UPLOAD=${AUTO_UPLOAD}
EOF

echo "========================================"
echo "Stage 2 Training"
echo "========================================"
echo "Repo root: ${REPO_ROOT}"
echo "Dataset config: ${DATASET_CONFIG}"
echo "Train config: ${TRAIN_CONFIG}"
echo "Group: ${GROUP}"
echo "Phase: ${PHASE}"
echo "Run name: ${RUN_NAME}"
if [[ "${AUTO_UPLOAD}" == "1" ]]; then
  echo "Upload mode: auto upload after training"
else
  echo "Upload mode: manual push after training"
fi
echo

bash scripts-for-colab/setup_colab.sh "${REPO_ROOT}"

TRAIN_ARGS=(
  --dataset-config "${DATASET_CONFIG}"
  --train-config "${TRAIN_CONFIG}"
  --group "${GROUP}"
)

if ! contains_flag --phase "${EXTRA_ARGS[@]}"; then
  TRAIN_ARGS+=(--phase "${PHASE}")
fi

if ! contains_flag --run-name "${EXTRA_ARGS[@]}"; then
  TRAIN_ARGS+=(--run-name "${RUN_NAME}")
fi

if [[ "${AUTO_UPLOAD}" != "1" ]] && ! contains_flag --skip-auto-upload "${EXTRA_ARGS[@]}"; then
  TRAIN_ARGS+=(--skip-auto-upload)
fi

TRAIN_ARGS+=("${EXTRA_ARGS[@]}")

python scripts/train_sft.py "${TRAIN_ARGS[@]}"
