#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/content/post-Train-proj"
if [[ "${1:-}" != "" && "${1:0:2}" != "--" ]]; then
  REPO_ROOT="$1"
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

is_global_runtime() {
  [[ -n "${COLAB_RELEASE_TAG:-}" || -n "${COLAB_GPU:-}" || "${REPO_ROOT}" == /content/* ]]
}

source_conda() {
  local conda_root="${CONDA_ROOT:-${HOME}/miniforge3}"
  if [[ -f "${conda_root}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${conda_root}/etc/profile.d/conda.sh"
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi
  echo "conda is not available and conda.sh was not found under ${conda_root}" >&2
  exit 1
}

RUN_ARGS=()
SKIP_SETUP=0
for arg in "$@"; do
  if [[ "${arg}" == "--skip-setup" ]]; then
    SKIP_SETUP=1
    continue
  fi
  RUN_ARGS+=("${arg}")
done

TRAIN_CONFIG="${TRAIN_CONFIG:-${STAGE2_TRAIN_CONFIG:-configs/train/stage2_search_lr1e4_r16_e1_ms500.yaml}}"
DATASET_CONFIG="${DATASET_CONFIG:-${STAGE2_DATASET_CONFIG:-configs/datasets/stage2_search_fixed_10k.yaml}}"
GROUP="${TRAIN_GROUP:-${STAGE2_GROUP:-S2}}"
PHASE="${TRAIN_PHASE:-${STAGE2_PHASE:-stage2search}}"
AUTO_UPLOAD="${TRAIN_AUTO_UPLOAD:-${STAGE2_AUTO_UPLOAD:-1}}"
TRAIN_ENV_NAME="${TRAIN_ENV_NAME:-post-train-local}"
TRAIN_RUNTIME_MODE="${TRAIN_RUNTIME_MODE:-auto}"

if [[ "${TRAIN_RUNTIME_MODE}" == "auto" ]]; then
  if is_global_runtime; then
    TRAIN_RUNTIME_MODE="global"
  else
    TRAIN_RUNTIME_MODE="conda"
  fi
fi

if [[ "${TRAIN_RUNTIME_MODE}" != "global" && "${TRAIN_RUNTIME_MODE}" != "conda" ]]; then
  echo "Unsupported TRAIN_RUNTIME_MODE: ${TRAIN_RUNTIME_MODE}" >&2
  exit 1
fi

USER_TRAIN_CONFIG="$(extract_option_value --train-config "${RUN_ARGS[@]}" || true)"
if [[ -n "${USER_TRAIN_CONFIG}" ]]; then
  TRAIN_CONFIG="${USER_TRAIN_CONFIG}"
fi

USER_DATASET_CONFIG="$(extract_option_value --dataset-config "${RUN_ARGS[@]}" || true)"
if [[ -n "${USER_DATASET_CONFIG}" ]]; then
  DATASET_CONFIG="${USER_DATASET_CONFIG}"
fi

USER_GROUP="$(extract_option_value --group "${RUN_ARGS[@]}" || true)"
if [[ -n "${USER_GROUP}" ]]; then
  GROUP="${USER_GROUP}"
fi

USER_PHASE="$(extract_option_value --phase "${RUN_ARGS[@]}" || true)"
if [[ -n "${USER_PHASE}" ]]; then
  PHASE="${USER_PHASE}"
fi

RUN_STAMP="$(date '+%Y%m%d_%H%M%S')"
CONFIG_SLUG="$(basename "${TRAIN_CONFIG}" .yaml)"
RUN_LABEL="${TRAIN_RUN_LABEL:-stage2search_${RUN_STAMP}}"
RUN_NAME="${TRAIN_RUN_NAME:-${RUN_LABEL}_${CONFIG_SLUG}}"
USER_RUN_NAME="$(extract_option_value --run-name "${RUN_ARGS[@]}" || true)"
if [[ -n "${USER_RUN_NAME}" ]]; then
  RUN_NAME="${USER_RUN_NAME}"
fi

if [[ ! -f "${TRAIN_CONFIG}" ]]; then
  echo "Train config not found: ${TRAIN_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${DATASET_CONFIG}" ]]; then
  echo "Dataset config not found: ${DATASET_CONFIG}" >&2
  exit 1
fi

LAUNCH_DIR="${REPO_ROOT}/runs/_launcher"
mkdir -p "${LAUNCH_DIR}"
printf '%s\n' "${RUN_NAME}" > "${LAUNCH_DIR}/latest_training_run_name.txt"
printf '%s\n' "${RUN_NAME}" > "${LAUNCH_DIR}/latest_stage2_train_run_name.txt"
cat > "${LAUNCH_DIR}/latest_training_launch.env" <<EOF
RUN_NAME=${RUN_NAME}
TRAIN_CONFIG=${TRAIN_CONFIG}
DATASET_CONFIG=${DATASET_CONFIG}
GROUP=${GROUP}
PHASE=${PHASE}
AUTO_UPLOAD=${AUTO_UPLOAD}
TRAIN_RUNTIME_MODE=${TRAIN_RUNTIME_MODE}
EOF
cp "${LAUNCH_DIR}/latest_training_launch.env" "${LAUNCH_DIR}/latest_stage2_train_launch.env"

echo "========================================"
echo "Training"
echo "========================================"
echo "Repo root: ${REPO_ROOT}"
echo "Dataset config: ${DATASET_CONFIG}"
echo "Train config: ${TRAIN_CONFIG}"
echo "Group: ${GROUP}"
echo "Phase: ${PHASE}"
echo "Run name: ${RUN_NAME}"
echo "Runtime mode: ${TRAIN_RUNTIME_MODE}"
if [[ "${AUTO_UPLOAD}" == "1" ]]; then
  echo "Upload mode: auto upload after training"
else
  echo "Upload mode: manual push after training"
fi
echo

if [[ "${SKIP_SETUP}" != "1" ]]; then
  if [[ "${TRAIN_RUNTIME_MODE}" == "global" ]]; then
    bash launchers/global/setup_train_env.sh "${REPO_ROOT}"
  else
    bash launchers/local/setup_train_env.sh "${REPO_ROOT}" "${TRAIN_ENV_NAME}"
  fi
fi

TRAIN_CMD=()
if [[ "${TRAIN_RUNTIME_MODE}" == "global" ]]; then
  TRAIN_CMD=(python)
else
  source_conda
  TRAIN_CMD=(conda run -n "${TRAIN_ENV_NAME}" python)
fi

TRAIN_ARGS=(
  --dataset-config "${DATASET_CONFIG}"
  --train-config "${TRAIN_CONFIG}"
)

if ! contains_flag --group "${RUN_ARGS[@]}"; then
  TRAIN_ARGS+=(--group "${GROUP}")
fi
if ! contains_flag --phase "${RUN_ARGS[@]}"; then
  TRAIN_ARGS+=(--phase "${PHASE}")
fi
if ! contains_flag --run-name "${RUN_ARGS[@]}"; then
  TRAIN_ARGS+=(--run-name "${RUN_NAME}")
fi
if [[ "${AUTO_UPLOAD}" != "1" ]] && ! contains_flag --skip-auto-upload "${RUN_ARGS[@]}"; then
  TRAIN_ARGS+=(--skip-auto-upload)
fi

TRAIN_ARGS+=("${RUN_ARGS[@]}")

echo "+ ${TRAIN_CMD[*]} scripts/train/train_sft.py ${TRAIN_ARGS[*]}"
"${TRAIN_CMD[@]}" scripts/train/train_sft.py "${TRAIN_ARGS[@]}"
