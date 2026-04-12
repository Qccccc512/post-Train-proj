#!/bin/bash
set -euo pipefail

REPO_ROOT="/content/post-Train-proj"
# Support local test overrides
if [[ -d "/home/fan/workspace/post-Train-proj" ]]; then
  REPO_ROOT="/home/fan/workspace/post-Train-proj"
fi
cd "${REPO_ROOT}"

echo "=> Installing dependencies for Colab (No Conda)..."
# Colab comes with a pre-installed Python 3.10+, we install dependencies in phases
# to avoid pip resolver 'resolution-too-deep' errors as noted in the lock file.
pip install torch==2.10.0 vllm==0.19.0 transformers==4.57.6 huggingface_hub==0.36.2
pip install lm_eval==0.4.11 bfcl-eval==2025.12.17
pip install numpy==1.26.4 tokenizers==0.22.2 langdetect==1.0.9 immutabledict==4.2.1 soundfile==0.13.1

echo "=> Bootstrapping benchmark frameworks..."
python3 scripts/bootstrap_benchmarks.py

export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONPATH=scripts

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set! Fetching private models may fail."
fi

BASE_MODEL="Qwen/Qwen3-8B"
echo "========================================"
echo "1. Fetching Top Adapter Weights"
echo "========================================"
RUN1="stage2search_20260407_173210_stage2_search_lr2e4_r16_e1_ms500"
RUN2="stage2search_20260407_173210_stage2_search_lr1e4_r16_e1_ms500"

python3 scripts/hf_repo_sync.py fetch-run --run-name "$RUN1"
python3 scripts/hf_repo_sync.py fetch-run --run-name "$RUN2"

echo "========================================"
echo "2. Preparing Benchmark Config"
echo "========================================"
CONFIG_FILE="configs/benchmark/colab_full_stage2.yaml"
cat << 'YAML' > "$CONFIG_FILE"
output_base_dir: analysis/stage2_best_benchmarks
max_samples: 999999
base_model: Qwen/Qwen3-8B
concurrent: 256
server:
  host: 127.0.0.1
  port: 8010
  # No conda needed for colab, vllm will run in current environment
  vllm_env_dir: ""
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.95
  max_model_len: 32768
  max_num_seqs: 256
  dtype: bfloat16
  max_lora_rank: 64
  repetition_penalty: 1.1
lm_eval:
  model: local-completions
  tokenizer_backend: huggingface
  batch_size: 256
  suites:
    ceval:
      task: ceval-valid
      limit: 999999
    ifeval:
      task: ifeval
      max_gen_toks: 4096
      limit: 999999
bfcl:
  model_name: Qwen/Qwen3-8B
  max_samples: 999999
  max_output_tokens: 4096
  test_categories: []
YAML

export BENCHMARK_CONFIG="$CONFIG_FILE"

echo "========================================"
echo "3. Running Full Benchmarks"
echo "========================================"

get_adapter_path() {
  local run_name="$1"
  if [[ -d "runs/$run_name/checkpoints/best" ]]; then
    echo "runs/$run_name/checkpoints/best"
  else
    echo "runs/$run_name/checkpoints/final"
  fi
}

echo "=> Benchmarking Base Model (Qwen3-8B)..."
python3 scripts/run_benchmarks.py \
  --suite all \
  --base-model "$BASE_MODEL" \
  --label stage2_best_base_model \
  --max-samples 999999

echo "=> Benchmarking LR 2e-4 (Target 1)..."
python3 scripts/run_benchmarks.py \
  --suite all \
  --base-model "$BASE_MODEL" \
  --adapter "$(get_adapter_path $RUN1)" \
  --label stage2_best_lr2e4_r16 \
  --max-samples 999999

echo "=> Benchmarking LR 1e-4 (Target 2)..."
python3 scripts/run_benchmarks.py \
  --suite all \
  --base-model "$BASE_MODEL" \
  --adapter "$(get_adapter_path $RUN2)" \
  --label stage2_best_lr1e4_r16 \
  --max-samples 999999

echo "========================================"
echo "4. Uploading Benchmark Results to HF"
echo "========================================"

python3 << 'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi
from hf_repo_sync import ensure_remote_repo, load_hf_config, upload_folder_with_retry
from runtime_utils import get_hf_token, resolve_hf_repo_id

# Mirrors are fine for downloads, but uploads must go to the official Hub API.
mirror_endpoint = os.environ.pop("HF_ENDPOINT", None)
if mirror_endpoint:
    print(f"HF_ENDPOINT was set to {mirror_endpoint}; switching back to the official Hugging Face Hub for upload.")

hf_cfg = load_hf_config("configs/hf/default.yaml")
token = get_hf_token(hf_cfg)
repo_id = resolve_hf_repo_id(hf_cfg)
api = HfApi(endpoint="https://huggingface.co", token=token)
ensure_remote_repo(api, hf_cfg, repo_id)

local_folder = "analysis/stage2_best_benchmarks"
if Path(local_folder).exists():
    print(f"Uploading {local_folder} to {repo_id} under analysis/...")
    upload_folder_with_retry(
        api=api,
        hf_config=hf_cfg,
        repo_id=repo_id,
        repo_type=hf_cfg.get("repo_type", "model"),
        folder_path=local_folder,
        path_in_repo="analysis/stage2_best_benchmarks",
        allow_patterns=None,
        commit_message="Upload full benchmark results for stage 2 best adapters"
    )
    print("Upload complete!")
else:
    print("No benchmark folder found to upload.")
PY

echo "Done!"
