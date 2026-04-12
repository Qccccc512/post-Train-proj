#!/bin/bash
# Re-run benchmarks for base model and Group H to verify results

set -e

cd /content/post-Train-proj
source /root/miniconda/etc/profile.d/conda.sh
conda activate post-train-benchmark
export HF_ENDPOINT="https://hf-mirror.com"

BASE_DIR="/content/post-Train-proj/runs/_benchmark_runtime/model_cache/adapters"
BASE_MODEL="Qwen/Qwen3-8B"

echo "========================================"
echo "Re-evaluation: Base Model + Group H"
echo "========================================"
echo ""

# Base model only (no adapter)
echo "[1/2] Re-evaluating Base Model (no adapter)..."
python scripts/run_benchmarks.py \
  --suite all \
  --base-model $BASE_MODEL \
  --label stage1_base_model_rerun \
  --max-samples 50

echo ""
echo "[2/2] Re-evaluating Group H..."
python scripts/run_benchmarks.py \
  --suite all \
  --base-model $BASE_MODEL \
  --adapter $BASE_DIR/2026-04-04_stage1_H_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/runs/stage1_qwen3_8b/2026-04-04_stage1_H_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/checkpoints/final \
  --label stage1_group_H_rerun \
  --max-samples 50

echo ""
echo "========================================"
echo "Re-evaluation completed!"
echo "========================================"
echo "Results saved to:"
echo "  - analysis/stage1/benchmarks/stage1_base_model_rerun/"
echo "  - analysis/stage1/benchmarks/stage1_group_H_rerun/"
