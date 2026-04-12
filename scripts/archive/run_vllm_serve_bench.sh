#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" != "" && "${1:0:2}" != "--" ]]; then
  REPO_ROOT="$1"
  shift
fi

cd "${REPO_ROOT}"

CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniforge3}"
CONDA_ENV_NAME="${VLLM_BENCH_CONDA_ENV:-post-train-benchmark}"
RUN_STAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_TAG="${VLLM_BENCH_RUN_TAG:-vllm_serve_bench_${RUN_STAMP}}"
OUTPUT_BASE_DIR="${VLLM_BENCH_OUTPUT_BASE_DIR:-analysis/vllm_serve_benchmarks}"
HF_CACHE_ROOT="${VLLM_BENCH_HF_HOME:-${REPO_ROOT}/runs/_benchmark_runtime/hf_cache}"

HOST="${VLLM_BENCH_HOST:-127.0.0.1}"
PORT_BASE="${VLLM_BENCH_PORT_BASE:-8100}"
TENSOR_PARALLEL_SIZE="${VLLM_BENCH_TP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${VLLM_BENCH_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${VLLM_BENCH_MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${VLLM_BENCH_MAX_NUM_SEQS:-128}"
NUM_WARMUPS="${VLLM_BENCH_NUM_WARMUPS:-3}"
READY_TIMEOUT_SEC="${VLLM_BENCH_READY_TIMEOUT_SEC:-900}"
SEED="${VLLM_BENCH_SEED:-42}"
SAVE_DETAILED="${VLLM_BENCH_SAVE_DETAILED:-0}"
PLOT_TIMELINE="${VLLM_BENCH_PLOT_TIMELINE:-0}"
DRY_RUN="${VLLM_BENCH_DRY_RUN:-0}"
EXTRA_SERVER_ARGS="${VLLM_BENCH_EXTRA_SERVER_ARGS:-}"
EXTRA_BENCH_ARGS="${VLLM_BENCH_EXTRA_BENCH_ARGS:-}"

read -r -d '' DEFAULT_MODEL_MATRIX <<'EOF' || true
qwen3_8b|Qwen/Qwen3-8B|bfloat16||
qwen3_8b_fp8|Qwen/Qwen3-8B-FP8|auto||
qwen3_8b_awq|Qwen/Qwen3-8B-AWQ|half||
EOF

read -r -d '' DEFAULT_PROFILE_MATRIX <<'EOF' || true
short|512|128|256|128|inf
medium|2048|256|128|64|inf
long|4096|512|64|32|inf
EOF

MODEL_MATRIX="${VLLM_BENCH_MODEL_MATRIX:-${DEFAULT_MODEL_MATRIX}}"
PROFILE_MATRIX="${VLLM_BENCH_PROFILE_MATRIX:-${DEFAULT_PROFILE_MATRIX}}"

if [[ "${OUTPUT_BASE_DIR}" = /* ]]; then
  OUTPUT_DIR="${OUTPUT_BASE_DIR}/${RUN_TAG}"
else
  OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_BASE_DIR}/${RUN_TAG}"
fi

RESULTS_DIR="${OUTPUT_DIR}/results"
SUMMARY_DIR="${OUTPUT_DIR}/summary"
ENV_DIR="${OUTPUT_DIR}/environment"

mkdir -p "${RESULTS_DIR}" "${SUMMARY_DIR}" "${ENV_DIR}" "${HF_CACHE_ROOT}"

contains_conda_env() {
  local env_name="$1"
  conda env list | awk 'NF >= 1 {print $1}' | grep -Fxq "${env_name}"
}

slugify() {
  python - "$1" <<'PY'
import re
import sys

value = sys.argv[1].strip().lower()
value = re.sub(r"[^a-z0-9]+", "_", value)
value = value.strip("_")
print(value or "item")
PY
}

pick_port() {
  python - "$1" <<'PY'
import socket
import sys

preferred = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", preferred))
        print(preferred)
    except OSError:
        sock.bind(("127.0.0.1", 0))
        print(sock.getsockname()[1])
PY
}

wait_for_server() {
  local base_url="$1"
  local server_pid="$2"
  local timeout_sec="$3"
  local server_log="$4"
  local deadline=$((SECONDS + timeout_sec))

  while (( SECONDS < deadline )); do
    if python - "${base_url}" <<'PY'
import json
import sys
from urllib.error import URLError
from urllib.request import urlopen

base_url = sys.argv[1].rstrip("/")
try:
    with urlopen(f"{base_url}/models", timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("data"):
        raise SystemExit(0)
except URLError:
    pass
except Exception:
    pass
raise SystemExit(1)
PY
    then
      return 0
    fi

    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "vLLM server exited before becoming ready. See ${server_log}" >&2
      tail -n 80 "${server_log}" >&2 || true
      return 1
    fi
    sleep 2
  done

  echo "Timed out waiting for ${base_url}/models. See ${server_log}" >&2
  tail -n 80 "${server_log}" >&2 || true
  return 1
}

run_logged_command() {
  local log_path="$1"
  shift
  mkdir -p "$(dirname "${log_path}")"
  {
    printf '$'
    for arg in "$@"; do
      printf ' %q' "${arg}"
    done
    printf '\n\n'
  } > "${log_path}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    cat "${log_path}"
    return 0
  fi

  "$@" >> "${log_path}" 2>&1
}

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "conda.sh not found under ${CONDA_ROOT}. Set CONDA_ROOT first." >&2
  exit 1
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"

if ! contains_conda_env "${CONDA_ENV_NAME}"; then
  if [[ "${CONDA_ENV_NAME}" != "post-train-benchmark" ]]; then
    echo "Conda env '${CONDA_ENV_NAME}' does not exist. Create it first." >&2
    exit 1
  fi
  bash scripts/setup_benchmark_env.sh
fi

conda activate "${CONDA_ENV_NAME}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export HF_HOME="${HF_CACHE_ROOT}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm command is unavailable in conda env ${CONDA_ENV_NAME}." >&2
  exit 1
fi

python - <<'PY' "${ENV_DIR}/benchmark_env.json"
from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

output = Path(sys.argv[1])
report = {
    "python": sys.version,
    "platform": platform.platform(),
    "conda_prefix": os.environ.get("CONDA_PREFIX"),
    "hf_home": os.environ.get("HF_HOME"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}

for module_name in ["torch", "vllm", "transformers", "huggingface_hub"]:
    module = importlib.import_module(module_name)
    report[module_name] = getattr(module, "__version__", "unknown")

try:
    import torch
    report["cuda_available"] = torch.cuda.is_available()
    report["cuda_device_count"] = torch.cuda.device_count()
    if torch.cuda.is_available():
        report["cuda_device_name"] = torch.cuda.get_device_name(0)
        report["bf16_supported"] = torch.cuda.is_bf16_supported()
except Exception as exc:
    report["torch_cuda_probe_error"] = repr(exc)

report["vllm_version_command"] = subprocess.check_output(
    ["vllm", "--version"], text=True
).strip()

output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

nvidia-smi > "${ENV_DIR}/nvidia-smi.txt" 2>&1 || true
conda list > "${ENV_DIR}/conda-list.txt" 2>&1 || true
printf '%s\n' "${MODEL_MATRIX}" > "${ENV_DIR}/model_matrix.txt"
printf '%s\n' "${PROFILE_MATRIX}" > "${ENV_DIR}/profile_matrix.txt"

cat > "${OUTPUT_DIR}/run_manifest.json" <<EOF
{
  "run_tag": "${RUN_TAG}",
  "output_dir": "${OUTPUT_DIR}",
  "conda_env": "${CONDA_ENV_NAME}",
  "host": "${HOST}",
  "port_base": ${PORT_BASE},
  "tensor_parallel_size": ${TENSOR_PARALLEL_SIZE},
  "gpu_memory_utilization": ${GPU_MEMORY_UTILIZATION},
  "max_model_len": ${MAX_MODEL_LEN},
  "max_num_seqs": ${MAX_NUM_SEQS},
  "num_warmups": ${NUM_WARMUPS},
  "ready_timeout_sec": ${READY_TIMEOUT_SEC},
  "seed": ${SEED},
  "dry_run": ${DRY_RUN}
}
EOF

echo "========================================"
echo "vLLM Serve Benchmark"
echo "========================================"
echo "Repo root: ${REPO_ROOT}"
echo "Conda env: ${CONDA_ENV_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
echo "HF cache: ${HF_HOME}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run: enabled"
fi
echo

CURRENT_SERVER_PID=""
cleanup_server() {
  if [[ -n "${CURRENT_SERVER_PID}" ]] && kill -0 "${CURRENT_SERVER_PID}" 2>/dev/null; then
    kill "${CURRENT_SERVER_PID}" 2>/dev/null || true
    wait "${CURRENT_SERVER_PID}" 2>/dev/null || true
  fi
  CURRENT_SERVER_PID=""
}
trap cleanup_server EXIT

CURRENT_PORT="${PORT_BASE}"
while IFS='|' read -r MODEL_LABEL MODEL_ID MODEL_DTYPE MODEL_QUANTIZATION MODEL_EXTRA_ARGS; do
  [[ -z "${MODEL_LABEL}" ]] && continue

  MODEL_SLUG="$(slugify "${MODEL_LABEL}")"
  MODEL_DIR="${RESULTS_DIR}/${MODEL_SLUG}"
  SERVER_DIR="${MODEL_DIR}/server"
  mkdir -p "${SERVER_DIR}"

  PORT="$(pick_port "${CURRENT_PORT}")"
  CURRENT_PORT="$((PORT + 1))"
  SERVED_MODEL_NAME="${VLLM_BENCH_SERVED_MODEL_PREFIX:-bench_}${MODEL_SLUG}"
  BASE_URL="http://${HOST}:${PORT}/v1"
  SERVER_LOG="${SERVER_DIR}/server.log"

  SERVER_CMD=(
    vllm serve "${MODEL_ID}"
    --host "${HOST}"
    --port "${PORT}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --trust-remote-code
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --seed "${SEED}"
  )
  if [[ -n "${MODEL_DTYPE}" ]]; then
    SERVER_CMD+=(--dtype "${MODEL_DTYPE}")
  fi
  if [[ -n "${MODEL_QUANTIZATION}" ]]; then
    SERVER_CMD+=(--quantization "${MODEL_QUANTIZATION}")
  fi
  if [[ -n "${EXTRA_SERVER_ARGS}" ]]; then
    read -r -a EXTRA_SERVER_ARR <<< "${EXTRA_SERVER_ARGS}"
    SERVER_CMD+=("${EXTRA_SERVER_ARR[@]}")
  fi
  if [[ -n "${MODEL_EXTRA_ARGS}" ]]; then
    read -r -a MODEL_EXTRA_ARR <<< "${MODEL_EXTRA_ARGS}"
    SERVER_CMD+=("${MODEL_EXTRA_ARR[@]}")
  fi

  cat > "${MODEL_DIR}/model_manifest.json" <<EOF
{
  "model_label": "${MODEL_LABEL}",
  "model_id": "${MODEL_ID}",
  "model_dtype": "${MODEL_DTYPE}",
  "model_quantization": "${MODEL_QUANTIZATION}",
  "served_model_name": "${SERVED_MODEL_NAME}",
  "base_url": "${BASE_URL}"
}
EOF

  echo "========================================"
  echo "Model: ${MODEL_LABEL}"
  echo "HF repo/path: ${MODEL_ID}"
  echo "Served model name: ${SERVED_MODEL_NAME}"
  echo "Base URL: ${BASE_URL}"
  echo "========================================"

  if [[ "${DRY_RUN}" == "1" ]]; then
    run_logged_command "${SERVER_LOG}" "${SERVER_CMD[@]}"
  else
    {
      printf '$'
      for arg in "${SERVER_CMD[@]}"; do
        printf ' %q' "${arg}"
      done
      printf '\n\n'
    } > "${SERVER_LOG}"
    "${SERVER_CMD[@]}" >> "${SERVER_LOG}" 2>&1 &
    CURRENT_SERVER_PID=$!
    wait_for_server "${BASE_URL}" "${CURRENT_SERVER_PID}" "${READY_TIMEOUT_SEC}" "${SERVER_LOG}"
  fi

  while IFS='|' read -r PROFILE_LABEL INPUT_LEN OUTPUT_LEN NUM_PROMPTS MAX_CONCURRENCY REQUEST_RATE; do
    [[ -z "${PROFILE_LABEL}" ]] && continue

    PROFILE_SLUG="$(slugify "${PROFILE_LABEL}")"
    PROFILE_DIR="${MODEL_DIR}/profiles/${PROFILE_SLUG}"
    PROFILE_LOG="${PROFILE_DIR}/bench.log"
    mkdir -p "${PROFILE_DIR}"

    cat > "${PROFILE_DIR}/profile_manifest.json" <<EOF
{
  "model_label": "${MODEL_LABEL}",
  "model_id": "${MODEL_ID}",
  "served_model_name": "${SERVED_MODEL_NAME}",
  "profile_label": "${PROFILE_LABEL}",
  "input_len": ${INPUT_LEN},
  "output_len": ${OUTPUT_LEN},
  "num_prompts": ${NUM_PROMPTS},
  "max_concurrency": ${MAX_CONCURRENCY},
  "request_rate": "${REQUEST_RATE}",
  "result_json": "${PROFILE_DIR}/result.json"
}
EOF

    BENCH_CMD=(
      vllm bench serve
      --backend openai
      --base-url "${BASE_URL}"
      --endpoint /v1/completions
      --model "${SERVED_MODEL_NAME}"
      --served-model-name "${SERVED_MODEL_NAME}"
      --tokenizer "${MODEL_ID}"
      --dataset-name random
      --input-len "${INPUT_LEN}"
      --output-len "${OUTPUT_LEN}"
      --num-prompts "${NUM_PROMPTS}"
      --request-rate "${REQUEST_RATE}"
      --max-concurrency "${MAX_CONCURRENCY}"
      --num-warmups "${NUM_WARMUPS}"
      --seed "${SEED}"
      --trust-remote-code
      --ready-check-timeout-sec "${READY_TIMEOUT_SEC}"
      --disable-tqdm
      --save-result
      --result-dir "${PROFILE_DIR}"
      --result-filename result.json
    )

    if [[ "${SAVE_DETAILED}" == "1" ]]; then
      BENCH_CMD+=(--save-detailed)
    fi
    if [[ "${PLOT_TIMELINE}" == "1" ]]; then
      BENCH_CMD+=(--plot-timeline)
    fi
    if [[ -n "${EXTRA_BENCH_ARGS}" ]]; then
      read -r -a EXTRA_BENCH_ARR <<< "${EXTRA_BENCH_ARGS}"
      BENCH_CMD+=("${EXTRA_BENCH_ARR[@]}")
    fi

    echo "Profile: ${PROFILE_LABEL} | input=${INPUT_LEN} output=${OUTPUT_LEN} prompts=${NUM_PROMPTS} concurrency=${MAX_CONCURRENCY} request_rate=${REQUEST_RATE}"
    run_logged_command "${PROFILE_LOG}" "${BENCH_CMD[@]}"
  done <<< "${PROFILE_MATRIX}"

  cleanup_server
done <<< "${MODEL_MATRIX}"

if [[ "${DRY_RUN}" != "1" ]]; then
  python scripts/summarize_vllm_serve_bench.py \
    --run-dir "${OUTPUT_DIR}" \
    --output-json "${SUMMARY_DIR}/vllm_serve_bench_summary.json" \
    --output-csv "${SUMMARY_DIR}/vllm_serve_bench_summary.csv" \
    --output-md "${SUMMARY_DIR}/vllm_serve_bench_summary.md"
fi

echo
echo "Finished. Results saved under:"
echo "  ${OUTPUT_DIR}"
