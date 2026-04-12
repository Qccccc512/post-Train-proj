#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/content/post-Train-proj"
if [[ "${1:-}" != "" && "${1:0:2}" != "--" ]]; then
  REPO_ROOT="$1"
  shift
fi

cd "${REPO_ROOT}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Export it before launching Stage 2 search." >&2
  exit 1
fi

DATASET_CONFIG="${STAGE2_DATASET_CONFIG:-configs/datasets/stage2_search_fixed_10k.yaml}"
GROUP="${STAGE2_GROUP:-S2}"
SEARCH_LABEL="${STAGE2_SEARCH_LABEL:-stage2search_$(date '+%Y%m%d_%H%M%S')}"

TRAIN_CONFIGS=(
  # Previous default matrix (kept for reference):
  # "configs/train/stage2_search_lr1e5_r16_e1_ms500.yaml"
  # "configs/train/stage2_search_lr2e5_r16_e1_ms500.yaml"
  # "configs/train/stage2_search_lr5e5_r16_e1_ms500.yaml"
  # "configs/train/stage2_search_lr2e5_r32_e1_ms500.yaml"
  # "configs/train/stage2_search_lr2e5_r64_e1_ms500.yaml"
  # "configs/train/stage2_search_lr2e5_r8_e1_ms500.yaml"

  # Current default experiment matrix:
  "configs/train/stage2_search_lr1e4_r16_e1_ms500.yaml"
  "configs/train/stage2_search_lr2e4_r16_e1_ms500.yaml"
  "configs/train/stage2_search_lr5e5_r16_e1_ms500_cosine.yaml"
  "configs/train/stage2_search_lr5e5_r32_e1_ms500_cosine.yaml"
)

if [[ "$#" -gt 0 ]]; then
  TRAIN_CONFIGS=("$@")
fi

echo "========================================"
echo "Stage 2 Hyperparameter Search"
echo "========================================"
echo "Repo root: ${REPO_ROOT}"
echo "Dataset config: ${DATASET_CONFIG}"
echo "Group: ${GROUP}"
echo "Search label: ${SEARCH_LABEL}"
echo "Upload mode: auto upload after each run (default train_sft behavior)"
echo "Runs: ${#TRAIN_CONFIGS[@]}"
printf ' - %s\n' "${TRAIN_CONFIGS[@]}"
echo

bash scripts-for-colab/setup_colab.sh "${REPO_ROOT}"

RUN_NAMES=()
for idx in "${!TRAIN_CONFIGS[@]}"; do
  train_config="${TRAIN_CONFIGS[$idx]}"
  if [[ ! -f "${train_config}" ]]; then
    echo "Train config not found: ${train_config}" >&2
    exit 1
  fi
  config_slug="$(basename "${train_config}" .yaml)"
  run_name="${SEARCH_LABEL}_${config_slug}"
  RUN_NAMES+=("${run_name}")

  echo "========================================"
  echo "[$((idx + 1))/${#TRAIN_CONFIGS[@]}] ${train_config}"
  echo "Run name: ${run_name}"
  echo "========================================"
  python scripts/train_sft.py \
    --dataset-config "${DATASET_CONFIG}" \
    --train-config "${train_config}" \
    --group "${GROUP}" \
    --phase stage2search \
    --run-name "${run_name}"
done

SUMMARY_DIR="runs/_stage2_search_summaries"
mkdir -p "${SUMMARY_DIR}"
SUMMARY_MD="${SUMMARY_DIR}/${SEARCH_LABEL}_eval_loss_summary.md"
SUMMARY_JSON="${SUMMARY_DIR}/${SEARCH_LABEL}_eval_loss_summary.json"

python scripts/summarize_stage2_search_eval_loss.py \
  --output-md "${SUMMARY_MD}" \
  --output-json "${SUMMARY_JSON}" \
  "${RUN_NAMES[@]}"

echo
echo "Stage 2 search completed."
echo "Eval loss summary (markdown): ${SUMMARY_MD}"
echo "Eval loss summary (json): ${SUMMARY_JSON}"
