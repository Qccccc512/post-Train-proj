#!/bin/bash
# Stage 1 Benchmark Evaluation Script
# Runs benchmarks for groups B-I (skip A which is already running) with 50 samples each

set -e

cd /content/post-Train-proj
source /root/miniconda/etc/profile.d/conda.sh
conda activate post-train-benchmark
export HF_ENDPOINT="https://hf-mirror.com"

BASE_DIR="/content/post-Train-proj/runs/_benchmark_runtime/model_cache/adapters"
BASE_MODEL="Qwen/Qwen3-8B"

echo "========================================"
echo "Stage 1 Benchmark Evaluation (B-I)"
echo "========================================"
echo "Base model: $BASE_MODEL"
echo "Samples per task: 50"
echo "Adapters: B-I (8 groups, A already running)"
echo ""

# Group B
echo "[1/8] Evaluating Group B..."
python scripts/run_benchmarks.py \
  --suite all \
  --base-model $BASE_MODEL \
  --adapter $BASE_DIR/2026-04-04_stage1_B_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/runs/stage1_qwen3_8b/2026-04-04_stage1_B_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/checkpoints/final \
  --label stage1_group_B \
  --max-samples 50

# Group C
echo "[2/8] Evaluating Group C..."
python scripts/run_benchmarks.py \
  --suite all \
  --base-model $BASE_MODEL \
  --adapter $BASE_DIR/2026-04-03_stage1_C_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/runs/stage1_qwen3_8b/2026-04-03_stage1_C_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/checkpoints/final \
  --label stage1_group_C \
  --max-samples 50

# Group D
echo "[3/8] Evaluating Group D..."
python scripts/run_benchmarks.py \
  --suite all \
  --base-model $BASE_MODEL \
  --adapter $BASE_DIR/2026-04-04_stage1_D_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/runs/stage1_qwen3_8b/2026-04-04_stage1_D_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/checkpoints/final \
  --label stage1_group_D \
  --max-samples 50

# Group E
echo "[4/8] Evaluating Group E..."
python scripts/run_benchmarks.py \
  --suite all \
  --base-model $BASE_MODEL \
  --adapter $BASE_DIR/2026-04-04_stage1_E_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/runs/stage1_qwen3_8b/2026-04-04_stage1_E_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/checkpoints/final \
  --label stage1_group_E \
  --max-samples 50

# Group F
echo "[5/8] Evaluating Group F..."
python scripts/run_benchmarks.py \
  --suite all \
  --base-model $BASE_MODEL \
  --adapter $BASE_DIR/2026-04-03_stage1_F_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/runs/stage1_qwen3_8b/2026-04-03_stage1_F_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/checkpoints/final \
  --label stage1_group_F \
  --max-samples 50

# Group G
echo "[6/8] Evaluating Group G..."
python scripts/run_benchmarks.py \
  --suite all \
  --base-model $BASE_MODEL \
  --adapter $BASE_DIR/2026-04-04_stage1_G_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/runs/stage1_qwen3_8b/2026-04-04_stage1_G_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/checkpoints/final \
  --label stage1_group_G \
  --max-samples 50

# Group H
echo "[7/8] Evaluating Group H..."
python scripts/run_benchmarks.py \
  --suite all \
  --base-model $BASE_MODEL \
  --adapter $BASE_DIR/2026-04-04_stage1_H_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/runs/stage1_qwen3_8b/2026-04-04_stage1_H_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/checkpoints/final \
  --label stage1_group_H \
  --max-samples 50

# Group I
echo "[8/8] Evaluating Group I..."
python scripts/run_benchmarks.py \
  --suite all \
  --base-model $BASE_MODEL \
  --adapter $BASE_DIR/2026-04-04_stage1_I_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/runs/stage1_qwen3_8b/2026-04-04_stage1_I_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/checkpoints/final \
  --label stage1_group_I \
  --max-samples 50

echo ""
echo "========================================"
echo "All benchmarks completed!"
echo "========================================"
echo "Results saved to: analysis/stage1/benchmarks/"
