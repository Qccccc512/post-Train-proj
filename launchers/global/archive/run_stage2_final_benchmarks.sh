#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/content/post-Train-proj"
if [[ "${1:-}" != "" && "${1:0:2}" != "--" ]]; then
  REPO_ROOT="$1"
  shift
fi

RUN_NAME="${STAGE2_FINAL_RUN_NAME:-}"
if [[ -z "${RUN_NAME}" && "${1:-}" != "" && "${1:0:2}" != "--" ]]; then
  RUN_NAME="$1"
  shift
fi

cd "${REPO_ROOT}"

DEFAULT_STAGE2_FINAL_RUN_REF="${STAGE2_FINAL_DEFAULT_RUN_REF:-yyyyFan/final_proj/runs/stage2train_20260408_095122_stage2_qwen3_8b_lora}"
DEFAULT_STAGE2_FINAL_MERGED_BF16_REPO="${STAGE2_FINAL_DEFAULT_MERGED_BF16_REPO:-yyyyFan/final_proj-stage2-stage2train_20260408_095122_stage2_qwen3_8b_lora-best-merged-bf16}"
DEFAULT_STAGE2_FINAL_MERGED_4BIT_REPO="${STAGE2_FINAL_DEFAULT_MERGED_4BIT_REPO:-yyyyFan/final_proj-stage2-stage2train_20260408_095122_stage2_qwen3_8b_lora-best-merged-4bit}"

normalize_run_name() {
  local raw="$1"
  raw="${raw%/}"
  raw="${raw##*/runs/}"
  raw="${raw##*/}"
  printf '%s\n' "${raw}"
}

discover_latest_stage2_run() {
  local pointer_file="${REPO_ROOT}/runs/_launcher/latest_stage2_train_run_name.txt"
  if [[ -n "${DEFAULT_STAGE2_FINAL_RUN_REF}" ]]; then
    printf '%s\n' "${DEFAULT_STAGE2_FINAL_RUN_REF}"
    return 0
  fi
  if [[ -f "${pointer_file}" ]]; then
    cat "${pointer_file}"
    return 0
  fi
  find "${REPO_ROOT}/runs" -mindepth 1 -maxdepth 1 -type d -name 'stage2train_*' -printf '%T@ %f\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | awk '{print $2}'
}

if [[ -z "${RUN_NAME}" ]]; then
  RUN_NAME="$(discover_latest_stage2_run || true)"
fi
RUN_NAME="$(normalize_run_name "${RUN_NAME}")"

if [[ -z "${RUN_NAME}" ]]; then
  echo "Stage 2 final benchmark requires a run name." >&2
  echo "Usage: bash scripts-for-colab/run_stage2_final_benchmarks.sh <repo_root> <run_name>" >&2
  echo "Or set STAGE2_FINAL_RUN_NAME=<run_name>." >&2
  echo "If omitted, the script will default to ${DEFAULT_STAGE2_FINAL_RUN_REF}." >&2
  exit 1
fi

CHECKPOINT_KIND="${STAGE2_FINAL_CHECKPOINT_KIND:-best}"
if [[ "${CHECKPOINT_KIND}" != "best" && "${CHECKPOINT_KIND}" != "final" ]]; then
  echo "STAGE2_FINAL_CHECKPOINT_KIND must be 'best' or 'final', got: ${CHECKPOINT_KIND}" >&2
  exit 1
fi

BASE_MODEL="${STAGE2_FINAL_BASE_MODEL:-Qwen/Qwen3-8B}"
BENCHMARK_TAG="${STAGE2_FINAL_BENCHMARK_TAG:-${RUN_NAME}_${CHECKPOINT_KIND}}"
OUTPUT_BASE_DIR="${STAGE2_FINAL_OUTPUT_BASE_DIR:-analysis/stage2_final_benchmarks/${BENCHMARK_TAG}}"
SUMMARY_DIR="${OUTPUT_BASE_DIR}/summary"
BENCHMARK_ROOT="${OUTPUT_BASE_DIR}/benchmarks"
CONFIG_FILE="${OUTPUT_BASE_DIR}/generated_benchmark_config.yaml"

BASE_LABEL="${STAGE2_FINAL_BASE_LABEL:-stage2_final_base_model}"
MERGED_BF16_LABEL="${STAGE2_FINAL_MERGED_BF16_LABEL:-${STAGE2_FINAL_MODEL_LABEL:-stage2_final_merged_bf16_model}}"
MERGED_4BIT_LABEL="${STAGE2_FINAL_MERGED_4BIT_LABEL:-stage2_final_merged_4bit_model}"

UPLOAD_RESULTS="${STAGE2_FINAL_UPLOAD_RESULTS:-0}"
UPLOAD_MERGED_MODELS="${STAGE2_FINAL_UPLOAD_MERGED_MODELS:-1}"
UPLOAD_MERGED_BF16="${STAGE2_FINAL_UPLOAD_MERGED_BF16:-0}"
UPLOAD_MERGED_4BIT="${STAGE2_FINAL_UPLOAD_MERGED_4BIT:-1}"
USE_REMOTE_MERGED_MODELS="${STAGE2_FINAL_USE_REMOTE_MERGED_MODELS:-1}"
FETCH_IF_MISSING="${STAGE2_FINAL_FETCH_IF_MISSING:-1}"
FORCE_FETCH="${STAGE2_FINAL_FORCE_FETCH:-0}"
SKIP_BASE="${STAGE2_FINAL_SKIP_BASE:-1}"
SKIP_MERGED_BF16_BENCH="${STAGE2_FINAL_SKIP_MERGED_BF16_BENCH:-0}"
SKIP_MERGED_4BIT_BENCH="${STAGE2_FINAL_SKIP_MERGED_4BIT_BENCH:-0}"
FORCE_BENCHMARK_SETUP="${STAGE2_FINAL_FORCE_BENCHMARK_SETUP:-0}"
FORCE_EXPORT="${STAGE2_FINAL_FORCE_EXPORT:-0}"
DRY_RUN="${STAGE2_FINAL_DRY_RUN:-0}"

MAX_SAMPLES="${STAGE2_FINAL_MAX_SAMPLES:-999999}"
BFCL_MAX_SAMPLES="${STAGE2_FINAL_BFCL_MAX_SAMPLES:-${MAX_SAMPLES}}"
CONCURRENT="${STAGE2_FINAL_CONCURRENT:-128}"
PORT="${STAGE2_FINAL_PORT:-8010}"
GPU_MEMORY_UTILIZATION="${STAGE2_FINAL_GPU_MEMORY_UTILIZATION:-0.95}"
MAX_MODEL_LEN="${STAGE2_FINAL_MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${STAGE2_FINAL_MAX_NUM_SEQS:-256}"
BASE_DTYPE="${STAGE2_FINAL_BASE_DTYPE:-bfloat16}"
MERGED_BF16_DTYPE="${STAGE2_FINAL_MERGED_BF16_DTYPE:-bfloat16}"
MERGED_4BIT_DTYPE="${STAGE2_FINAL_MERGED_4BIT_DTYPE:-auto}"
MAX_LORA_RANK="${STAGE2_FINAL_MAX_LORA_RANK:-64}"
REPETITION_PENALTY="${STAGE2_FINAL_REPETITION_PENALTY:-1.1}"
LM_EVAL_BATCH_SIZE="${STAGE2_FINAL_LM_EVAL_BATCH_SIZE:-256}"
REMOTE_ANALYSIS_DIR="${STAGE2_FINAL_REMOTE_ANALYSIS_DIR:-analysis/stage2_final_benchmarks/${BENCHMARK_TAG}}"

TRAIN_ENV_NAME="${STAGE2_FINAL_TRAIN_ENV:-post-train-local}"
BENCHMARK_ENV_NAME="${STAGE2_FINAL_BENCHMARK_ENV:-post-train-benchmark}"
BENCHMARK_VENV_DIR="${STAGE2_FINAL_BENCHMARK_VENV_DIR:-${REPO_ROOT}/.venv}"
TRAIN_RUNTIME_MODE="${STAGE2_FINAL_TRAIN_RUNTIME:-auto}"
BENCHMARK_RUNTIME_MODE="${STAGE2_FINAL_BENCHMARK_RUNTIME:-auto}"
TRAIN_PYTHON_BIN="${STAGE2_FINAL_TRAIN_PYTHON_BIN:-python}"
BENCHMARK_PYTHON_BIN="${STAGE2_FINAL_BENCHMARK_PYTHON_BIN:-python3}"
BENCHMARK_HF_ENDPOINT="${STAGE2_FINAL_HF_ENDPOINT:-}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniforge3}"
EXPORT_MAX_MEMORY_USAGE="${STAGE2_FINAL_EXPORT_MAX_MEMORY_USAGE:-0.75}"

RUN_DIR="${REPO_ROOT}/runs/${RUN_NAME}"
ADAPTER_PATH="${STAGE2_FINAL_ADAPTER_PATH:-${RUN_DIR}/checkpoints/${CHECKPOINT_KIND}}"
MERGED_OUTPUT_ROOT="${STAGE2_FINAL_MERGED_OUTPUT_ROOT:-${RUN_DIR}/merged_models/${CHECKPOINT_KIND}}"
MERGED_MANIFEST_PATH="${STAGE2_FINAL_MERGED_MANIFEST_PATH:-${MERGED_OUTPUT_ROOT}/merged_model_exports.json}"
MERGED_BF16_DIR="${STAGE2_FINAL_MERGED_BF16_DIR:-${MERGED_OUTPUT_ROOT}/merged_bf16}"
MERGED_4BIT_DIR="${STAGE2_FINAL_MERGED_4BIT_DIR:-${MERGED_OUTPUT_ROOT}/merged_4bit}"
REMOTE_MERGED_BF16_REPO="${STAGE2_FINAL_REMOTE_MERGED_BF16_REPO:-${DEFAULT_STAGE2_FINAL_MERGED_BF16_REPO}}"
REMOTE_MERGED_4BIT_REPO="${STAGE2_FINAL_REMOTE_MERGED_4BIT_REPO:-${DEFAULT_STAGE2_FINAL_MERGED_4BIT_REPO}}"

MERGED_BF16_REPO_ID_OVERRIDE="${STAGE2_FINAL_MERGED_BF16_REPO_ID:-}"
MERGED_4BIT_REPO_ID_OVERRIDE="${STAGE2_FINAL_MERGED_4BIT_REPO_ID:-}"
EXPORT_ORDER="${STAGE2_FINAL_EXPORT_ORDER:-bf16-first}"

export CONDA_ROOT
export PYTHONPATH="scripts${PYTHONPATH:+:${PYTHONPATH}}"
export STAGE2_FINAL_UPLOAD_LOCAL_DIR="${OUTPUT_BASE_DIR}"
export STAGE2_FINAL_UPLOAD_REMOTE_DIR="${REMOTE_ANALYSIS_DIR}"
export STAGE2_FINAL_UPLOAD_RUN_NAME="${RUN_NAME}"

run_cmd() {
  echo "+ $*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@"
}

is_colab_runtime() {
  [[ -n "${COLAB_RELEASE_TAG:-}" || -n "${COLAB_GPU:-}" || "${REPO_ROOT}" == /content/* ]]
}

resolve_train_runtime_mode() {
  local raw="$1"
  if [[ "${raw}" == "auto" ]]; then
    if is_colab_runtime; then
      printf 'global\n'
    else
      printf 'conda\n'
    fi
    return 0
  fi
  if [[ "${raw}" != "conda" && "${raw}" != "global" ]]; then
    echo "Unsupported runtime mode: ${raw}" >&2
    exit 1
  fi
  printf '%s\n' "${raw}"
}

resolve_benchmark_runtime_mode() {
  local raw="$1"
  if [[ "${raw}" == "auto" ]]; then
    if is_colab_runtime; then
      printf 'global\n'
    else
      printf 'conda\n'
    fi
    return 0
  fi
  if [[ "${raw}" != "conda" && "${raw}" != "global" && "${raw}" != "venv" ]]; then
    echo "Unsupported benchmark runtime mode: ${raw}" >&2
    exit 1
  fi
  printf '%s\n' "${raw}"
}

if [[ -n "${BENCHMARK_HF_ENDPOINT}" ]]; then
  export HF_ENDPOINT="${BENCHMARK_HF_ENDPOINT}"
else
  unset HF_ENDPOINT || true
fi

source_conda() {
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi
  if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    return 0
  fi
  echo "conda is not available and conda.sh was not found under ${CONDA_ROOT}" >&2
  exit 1
}

contains_conda_env() {
  local env_name="$1"
  conda env list | awk 'NF >= 1 {print $1}' | grep -Fxq "${env_name}"
}

run_train_python() {
  if [[ "${TRAIN_RUNTIME_MODE_RESOLVED}" == "conda" ]]; then
    run_cmd conda run -n "${TRAIN_ENV_NAME}" python "$@"
  else
    run_cmd "${TRAIN_PYTHON_BIN}" "$@"
  fi
}

run_benchmark_python() {
  if [[ "${BENCHMARK_RUNTIME_MODE_RESOLVED}" == "conda" ]]; then
    run_cmd conda run -n "${BENCHMARK_ENV_NAME}" python "$@"
  elif [[ "${BENCHMARK_RUNTIME_MODE_RESOLVED}" == "venv" ]]; then
    run_cmd "${BENCHMARK_VENV_DIR}/bin/python" "$@"
  else
    run_cmd "${BENCHMARK_PYTHON_BIN}" "$@"
  fi
}

TRAIN_RUNTIME_MODE_RESOLVED="$(resolve_train_runtime_mode "${TRAIN_RUNTIME_MODE}")"
BENCHMARK_RUNTIME_MODE_RESOLVED="$(resolve_benchmark_runtime_mode "${BENCHMARK_RUNTIME_MODE}")"
if [[ "${BENCHMARK_RUNTIME_MODE_RESOLVED}" == "conda" ]]; then
  BENCHMARK_VLLM_ENV_SPEC="${BENCHMARK_ENV_NAME}"
elif [[ "${BENCHMARK_RUNTIME_MODE_RESOLVED}" == "venv" ]]; then
  BENCHMARK_VLLM_ENV_SPEC="${BENCHMARK_VENV_DIR}"
else
  BENCHMARK_VLLM_ENV_SPEC=""
fi

mkdir -p "${OUTPUT_BASE_DIR}" "${SUMMARY_DIR}"

cat > "${CONFIG_FILE}" <<YAML
output_base_dir: ${OUTPUT_BASE_DIR}
max_samples: ${MAX_SAMPLES}
base_model: ${BASE_MODEL}
concurrent: ${CONCURRENT}
server:
  host: 127.0.0.1
  port: ${PORT}
  vllm_env_dir: ${BENCHMARK_VLLM_ENV_SPEC}
  tensor_parallel_size: 1
  gpu_memory_utilization: ${GPU_MEMORY_UTILIZATION}
  max_model_len: ${MAX_MODEL_LEN}
  max_num_seqs: ${MAX_NUM_SEQS}
  dtype: ${BASE_DTYPE}
  max_lora_rank: ${MAX_LORA_RANK}
  repetition_penalty: ${REPETITION_PENALTY}
lm_eval:
  model: local-completions
  tokenizer_backend: huggingface
  batch_size: ${LM_EVAL_BATCH_SIZE}
  suites:
    ceval:
      task: ceval-valid
      limit: ${MAX_SAMPLES}
    ifeval:
      task: ifeval
      max_gen_toks: 4096
      limit: ${MAX_SAMPLES}
bfcl:
  model_name: ${BASE_MODEL}
  max_samples: ${BFCL_MAX_SAMPLES}
  max_output_tokens: 4096
  test_categories: []
YAML

export BENCHMARK_CONFIG="${CONFIG_FILE}"

needs_fetch=0
if [[ "${USE_REMOTE_MERGED_MODELS}" != "1" && "${FORCE_FETCH}" == "1" ]]; then
  needs_fetch=1
elif [[ "${USE_REMOTE_MERGED_MODELS}" != "1" && ! -d "${RUN_DIR}" ]]; then
  needs_fetch=1
fi

if [[ "${needs_fetch}" == "1" ]]; then
  if [[ "${FETCH_IF_MISSING}" != "1" ]]; then
    echo "Run directory is missing and STAGE2_FINAL_FETCH_IF_MISSING!=1: ${RUN_DIR}" >&2
    exit 1
  fi
  if [[ "${DRY_RUN}" != "1" && -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is required to fetch run artifacts from HF." >&2
    exit 1
  fi
fi

if [[ "${USE_REMOTE_MERGED_MODELS}" != "1" && "${UPLOAD_MERGED_MODELS}" == "1" && ( "${UPLOAD_MERGED_BF16}" == "1" || "${UPLOAD_MERGED_4BIT}" == "1" ) && "${DRY_RUN}" != "1" && -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required because STAGE2_FINAL_UPLOAD_MERGED_MODELS=1." >&2
  exit 1
fi

if [[ "${UPLOAD_RESULTS}" == "1" && "${DRY_RUN}" != "1" && -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required because STAGE2_FINAL_UPLOAD_RESULTS=1." >&2
  exit 1
fi

if [[ "${DRY_RUN}" != "1" && ( "${needs_fetch}" == "1" || "${USE_REMOTE_MERGED_MODELS}" != "1" ) ]]; then
  if [[ "${TRAIN_RUNTIME_MODE_RESOLVED}" == "conda" || "${BENCHMARK_RUNTIME_MODE_RESOLVED}" == "conda" ]]; then
    source_conda
  fi
  if [[ "${TRAIN_RUNTIME_MODE_RESOLVED}" == "conda" ]]; then
    if ! contains_conda_env "${TRAIN_ENV_NAME}"; then
      echo "Required conda env is missing: ${TRAIN_ENV_NAME}" >&2
      exit 1
    fi
  elif ! "${TRAIN_PYTHON_BIN}" -c "import unsloth" >/dev/null 2>&1; then
    echo "Training/export runtime is missing Unsloth; preparing the current Colab environment."
    run_cmd bash scripts-for-colab/setup_colab.sh "${REPO_ROOT}"
  fi
fi

if [[ "${needs_fetch}" == "1" ]]; then
  echo "========================================"
  echo "1. Fetching Stage 2 Final Training Run"
  echo "========================================"
  run_train_python scripts/hf_repo_sync.py fetch-run --run-name "${RUN_NAME}"
fi

if [[ "${USE_REMOTE_MERGED_MODELS}" != "1" && ! -d "${ADAPTER_PATH}" ]]; then
  echo "Adapter path not found: ${ADAPTER_PATH}" >&2
  if [[ -d "${RUN_DIR}/checkpoints" ]]; then
    echo "Available checkpoints under ${RUN_DIR}/checkpoints:" >&2
    find "${RUN_DIR}/checkpoints" -maxdepth 2 -type d | sort >&2
  fi
  exit 1
fi

echo "========================================"
echo "Stage 2 Final Benchmarks"
echo "========================================"
echo "Repo root: ${REPO_ROOT}"
echo "Run name: ${RUN_NAME}"
echo "Checkpoint kind: ${CHECKPOINT_KIND}"
echo "Adapter path: ${ADAPTER_PATH}"
echo "Base model: ${BASE_MODEL}"
echo "Benchmark tag: ${BENCHMARK_TAG}"
echo "Output base dir: ${OUTPUT_BASE_DIR}"
echo "Merged output root: ${MERGED_OUTPUT_ROOT}"
echo "Benchmark config: ${CONFIG_FILE}"
echo "Use remote merged models: ${USE_REMOTE_MERGED_MODELS}"
echo "Train runtime: ${TRAIN_RUNTIME_MODE_RESOLVED}"
if [[ "${TRAIN_RUNTIME_MODE_RESOLVED}" == "conda" ]]; then
  echo "Train env: ${TRAIN_ENV_NAME}"
else
  echo "Train python: ${TRAIN_PYTHON_BIN}"
fi
echo "Benchmark runtime: ${BENCHMARK_RUNTIME_MODE_RESOLVED}"
if [[ "${BENCHMARK_RUNTIME_MODE_RESOLVED}" == "conda" ]]; then
  echo "Benchmark env: ${BENCHMARK_ENV_NAME}"
elif [[ "${BENCHMARK_RUNTIME_MODE_RESOLVED}" == "venv" ]]; then
  echo "Benchmark venv: ${BENCHMARK_VENV_DIR}"
else
  echo "Benchmark python: ${BENCHMARK_PYTHON_BIN}"
fi
echo "Benchmark vllm_env_dir: ${BENCHMARK_VLLM_ENV_SPEC:-<system-python>}"
echo "Benchmark HF endpoint: ${HF_ENDPOINT:-<official-default>}"
if [[ "${USE_REMOTE_MERGED_MODELS}" == "1" ]]; then
  echo "Remote merged bf16 repo: ${REMOTE_MERGED_BF16_REPO}"
  echo "Remote merged 4bit repo: ${REMOTE_MERGED_4BIT_REPO}"
elif [[ "${UPLOAD_MERGED_MODELS}" == "1" ]]; then
  echo "Upload merged models: enabled"
  echo "Upload merged bf16: ${UPLOAD_MERGED_BF16}"
  echo "Upload merged 4bit: ${UPLOAD_MERGED_4BIT}"
else
  echo "Upload merged models: disabled"
fi
echo "Requested export order: ${EXPORT_ORDER}"
echo "Effective export flow: merged bf16 -> merged 4bit"
if [[ "${UPLOAD_RESULTS}" == "1" ]]; then
  echo "Upload benchmark results: enabled"
  echo "Remote analysis dir: ${REMOTE_ANALYSIS_DIR}"
else
  echo "Upload benchmark results: disabled"
fi
if [[ -n "${MERGED_BF16_REPO_ID_OVERRIDE}" ]]; then
  echo "Merged bf16 repo override: ${MERGED_BF16_REPO_ID_OVERRIDE}"
fi
if [[ -n "${MERGED_4BIT_REPO_ID_OVERRIDE}" ]]; then
  echo "Merged 4bit repo override: ${MERGED_4BIT_REPO_ID_OVERRIDE}"
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run: enabled"
fi
echo

MERGED_BF16_REPO_ID="${REMOTE_MERGED_BF16_REPO}"
MERGED_4BIT_REPO_ID="${REMOTE_MERGED_4BIT_REPO}"
MERGED_BF16_BENCH_MODEL="${REMOTE_MERGED_BF16_REPO}"
MERGED_4BIT_BENCH_MODEL="${REMOTE_MERGED_4BIT_REPO}"
if [[ "${USE_REMOTE_MERGED_MODELS}" == "1" ]]; then
  echo "========================================"
  echo "2. Using Remote Merged Model Repos"
  echo "========================================"
  echo "Skipping local export and benchmarking directly from Hugging Face repos."
else
  echo "========================================"
  echo "2. Exporting Merged Model Variants"
  echo "========================================"
  EXPORT_CMD=(
    scripts/export_merged_model_variants.py
    --adapter-dir "${ADAPTER_PATH}"
    --run-name "${RUN_NAME}"
    --checkpoint-kind "${CHECKPOINT_KIND}"
    --output-root "${MERGED_OUTPUT_ROOT}"
    --bf16-output-dir "${MERGED_BF16_DIR}"
    --fourbit-output-dir "${MERGED_4BIT_DIR}"
    --bf16-dtype "${MERGED_BF16_DTYPE}"
    --fourbit-compute-dtype "${MERGED_4BIT_DTYPE}"
    --maximum-memory-usage "${EXPORT_MAX_MEMORY_USAGE}"
    --export-order "${EXPORT_ORDER}"
  )
  if [[ "${UPLOAD_MERGED_MODELS}" != "1" ]]; then
    EXPORT_CMD+=(--skip-upload)
  fi
  if [[ "${UPLOAD_MERGED_BF16}" != "1" ]]; then
    EXPORT_CMD+=(--no-upload-bf16)
  fi
  if [[ "${UPLOAD_MERGED_4BIT}" != "1" ]]; then
    EXPORT_CMD+=(--no-upload-fourbit)
  fi
  if [[ "${FORCE_EXPORT}" == "1" ]]; then
    EXPORT_CMD+=(--force)
  fi
  if [[ -n "${MERGED_BF16_REPO_ID_OVERRIDE}" ]]; then
    EXPORT_CMD+=(--bf16-repo-id "${MERGED_BF16_REPO_ID_OVERRIDE}")
  fi
  if [[ -n "${MERGED_4BIT_REPO_ID_OVERRIDE}" ]]; then
    EXPORT_CMD+=(--fourbit-repo-id "${MERGED_4BIT_REPO_ID_OVERRIDE}")
  fi
  run_train_python "${EXPORT_CMD[@]}"

  MERGED_BF16_REPO_ID="<not-uploaded>"
  MERGED_4BIT_REPO_ID="<not-uploaded>"
  if [[ "${DRY_RUN}" != "1" && -f "${MERGED_MANIFEST_PATH}" ]]; then
    mapfile -t _MERGED_REPOS < <(python3 - <<PY
import json
from pathlib import Path

payload = json.loads(Path(${MERGED_MANIFEST_PATH@Q}).read_text(encoding="utf-8"))
exports = payload.get("exports", {})
print((exports.get("merged_bf16") or {}).get("repo_id") or "")
print((exports.get("merged_4bit") or {}).get("repo_id") or "")
PY
    )
    MERGED_BF16_REPO_ID="${_MERGED_REPOS[0]:-<not-uploaded>}"
    MERGED_4BIT_REPO_ID="${_MERGED_REPOS[1]:-<not-uploaded>}"
  fi
  MERGED_BF16_BENCH_MODEL="${MERGED_BF16_DIR}"
  MERGED_4BIT_BENCH_MODEL="${MERGED_4BIT_DIR}"
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  if [[ "${BENCHMARK_RUNTIME_MODE_RESOLVED}" == "conda" ]]; then
    source_conda
  fi
  if [[ "${BENCHMARK_RUNTIME_MODE_RESOLVED}" == "conda" ]]; then
    if ! contains_conda_env "${BENCHMARK_ENV_NAME}" || [[ "${FORCE_BENCHMARK_SETUP}" == "1" ]]; then
      echo "========================================"
      echo "3. Preparing Benchmark Runtime"
      echo "========================================"
      run_cmd bash scripts/setup_benchmark_env.sh
    else
      echo "========================================"
      echo "3. Benchmark Runtime Already Present"
      echo "========================================"
      echo "Reusing conda env ${BENCHMARK_ENV_NAME}"
    fi
  elif [[ "${BENCHMARK_RUNTIME_MODE_RESOLVED}" == "venv" ]]; then
    if [[ "${FORCE_BENCHMARK_SETUP}" == "1" ]] || ! "${BENCHMARK_VENV_DIR}/bin/python" -c "import vllm, lm_eval, bitsandbytes" >/dev/null 2>&1; then
      echo "========================================"
      echo "3. Preparing Benchmark Runtime"
      echo "========================================"
      BENCHMARK_VENV_DIR="${BENCHMARK_VENV_DIR}" run_cmd bash scripts-for-colab/setup_benchmark.sh "${REPO_ROOT}"
    else
      echo "========================================"
      echo "3. Benchmark Runtime Already Present"
      echo "========================================"
      echo "Reusing benchmark venv ${BENCHMARK_VENV_DIR}"
    fi
  else
    if [[ "${FORCE_BENCHMARK_SETUP}" == "1" ]] || ! "${BENCHMARK_PYTHON_BIN}" -c "import vllm, lm_eval, bitsandbytes" >/dev/null 2>&1; then
      echo "========================================"
      echo "3. Preparing Benchmark Runtime"
      echo "========================================"
      BENCHMARK_SETUP_MODE=global BOOTSTRAP_PYTHON="${BENCHMARK_PYTHON_BIN}" run_cmd bash scripts-for-colab/setup_benchmark.sh "${REPO_ROOT}"
    else
      echo "========================================"
      echo "3. Benchmark Runtime Already Present"
      echo "========================================"
      echo "Reusing current Python benchmark environment"
    fi
  fi
else
  echo "========================================"
  echo "3. Preparing Benchmark Runtime"
  echo "========================================"
  if [[ "${BENCHMARK_RUNTIME_MODE_RESOLVED}" == "conda" ]]; then
    echo "+ bash scripts/setup_benchmark_env.sh"
  elif [[ "${BENCHMARK_RUNTIME_MODE_RESOLVED}" == "venv" ]]; then
    echo "+ BENCHMARK_VENV_DIR=${BENCHMARK_VENV_DIR} bash scripts-for-colab/setup_benchmark.sh ${REPO_ROOT}"
  else
    echo "+ BENCHMARK_SETUP_MODE=global BOOTSTRAP_PYTHON=${BENCHMARK_PYTHON_BIN} bash scripts-for-colab/setup_benchmark.sh ${REPO_ROOT}"
  fi
fi

if [[ "${SKIP_MERGED_4BIT_BENCH}" != "1" ]]; then
  echo "========================================"
  echo "3.1 Verifying 4bit Benchmark Dependency"
  echo "========================================"
  if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ "${BENCHMARK_RUNTIME_MODE_RESOLVED}" == "conda" ]] && ! conda run -n "${BENCHMARK_ENV_NAME}" python -c "import bitsandbytes" >/dev/null 2>&1; then
      echo "bitsandbytes is missing in ${BENCHMARK_ENV_NAME}; reconciling benchmark environment."
      run_cmd bash scripts/setup_benchmark_env.sh
    fi
    if [[ "${BENCHMARK_RUNTIME_MODE_RESOLVED}" == "venv" ]] && ! "${BENCHMARK_VENV_DIR}/bin/python" -c "import bitsandbytes" >/dev/null 2>&1; then
      echo "bitsandbytes is missing in benchmark venv ${BENCHMARK_VENV_DIR}; reconciling benchmark environment."
      BENCHMARK_VENV_DIR="${BENCHMARK_VENV_DIR}" run_cmd bash scripts-for-colab/setup_benchmark.sh "${REPO_ROOT}"
    fi
    if [[ "${BENCHMARK_RUNTIME_MODE_RESOLVED}" == "global" ]] && ! "${BENCHMARK_PYTHON_BIN}" -c "import bitsandbytes" >/dev/null 2>&1; then
      echo "bitsandbytes is missing in the current Python environment; reconciling benchmark environment."
      BENCHMARK_SETUP_MODE=global BOOTSTRAP_PYTHON="${BENCHMARK_PYTHON_BIN}" run_cmd bash scripts-for-colab/setup_benchmark.sh "${REPO_ROOT}"
    fi
  fi
  run_benchmark_python -c "import bitsandbytes; print(bitsandbytes.__version__)"
fi

echo "========================================"
echo "4. Running Benchmarks"
echo "========================================"

SUMMARY_RUNS=()
if [[ "${SKIP_BASE}" != "1" ]]; then
  SUMMARY_RUNS+=("${BASE_LABEL}")
  run_benchmark_python scripts/run_benchmarks.py \
    --suite all \
    --base-model "${BASE_MODEL}" \
    --label "${BASE_LABEL}" \
    --max-samples "${MAX_SAMPLES}" \
    --bfcl-max-samples "${BFCL_MAX_SAMPLES}" \
    --dtype "${BASE_DTYPE}"
else
  echo "Skipping base benchmark because STAGE2_FINAL_SKIP_BASE=1"
fi

if [[ "${SKIP_MERGED_BF16_BENCH}" != "1" ]]; then
  SUMMARY_RUNS+=("${MERGED_BF16_LABEL}")
  run_benchmark_python scripts/run_benchmarks.py \
    --suite all \
    --base-model "${MERGED_BF16_BENCH_MODEL}" \
    --served-model-name "${BASE_MODEL}" \
    --label "${MERGED_BF16_LABEL}" \
    --max-samples "${MAX_SAMPLES}" \
    --bfcl-max-samples "${BFCL_MAX_SAMPLES}" \
    --dtype "${MERGED_BF16_DTYPE}"
else
  echo "Skipping merged bf16 benchmark because STAGE2_FINAL_SKIP_MERGED_BF16_BENCH=1"
fi

if [[ "${SKIP_MERGED_4BIT_BENCH}" != "1" ]]; then
  SUMMARY_RUNS+=("${MERGED_4BIT_LABEL}")
  run_benchmark_python scripts/run_benchmarks.py \
    --suite all \
    --base-model "${MERGED_4BIT_BENCH_MODEL}" \
    --served-model-name "${BASE_MODEL}" \
    --label "${MERGED_4BIT_LABEL}" \
    --max-samples "${MAX_SAMPLES}" \
    --bfcl-max-samples "${BFCL_MAX_SAMPLES}" \
    --dtype "${MERGED_4BIT_DTYPE}"
else
  echo "Skipping merged 4bit benchmark because STAGE2_FINAL_SKIP_MERGED_4BIT_BENCH=1"
fi

if (( ${#SUMMARY_RUNS[@]} >= 2 )); then
  echo "========================================"
  echo "5. Summarizing Final Model Variants"
  echo "========================================"
  SUMMARY_CMD=(
    scripts/summarize_benchmark_candidates.py
    --benchmark-root "${BENCHMARK_ROOT}"
    --output-md "${SUMMARY_DIR}/final_train_model_variants_summary.md"
    --output-json "${SUMMARY_DIR}/final_train_model_variants_summary.json"
  )
  for run_label in "${SUMMARY_RUNS[@]}"; do
    SUMMARY_CMD+=("${run_label}")
  done
  if [[ "${SKIP_BASE}" != "1" ]]; then
    SUMMARY_CMD+=(--reference "${BASE_LABEL}")
  fi
  run_benchmark_python "${SUMMARY_CMD[@]}"
else
  echo "Skipping summary because fewer than two benchmark targets were enabled."
fi

if [[ "${UPLOAD_RESULTS}" == "1" ]]; then
  echo "========================================"
  echo "6. Uploading Benchmark Results"
  echo "========================================"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "+ upload benchmark folder ${OUTPUT_BASE_DIR} -> ${REMOTE_ANALYSIS_DIR}"
  else
    if [[ "${TRAIN_RUNTIME_MODE_RESOLVED}" == "conda" ]]; then
      conda run -n "${TRAIN_ENV_NAME}" python <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi
from hf_repo_sync import ensure_remote_repo, load_hf_config, upload_folder_with_retry
from runtime_utils import OFFICIAL_HF_ENDPOINT, get_hf_token, resolve_hf_repo_id

local_folder = Path(os.environ["STAGE2_FINAL_UPLOAD_LOCAL_DIR"])
remote_dir = os.environ["STAGE2_FINAL_UPLOAD_REMOTE_DIR"]

mirror_endpoint = os.environ.pop("HF_ENDPOINT", None)
if mirror_endpoint:
    print(f"HF_ENDPOINT was set to {mirror_endpoint}; switching back to the official Hugging Face Hub for upload.")

hf_cfg = load_hf_config("configs/hf/default.yaml")
token = get_hf_token(hf_cfg)
repo_id = resolve_hf_repo_id(hf_cfg)
api = HfApi(endpoint=OFFICIAL_HF_ENDPOINT, token=token)
ensure_remote_repo(api, hf_cfg, repo_id)

if not local_folder.exists():
    raise FileNotFoundError(f"Benchmark output folder is missing: {local_folder}")

print(f"Uploading {local_folder} to {repo_id} under {remote_dir}...")
upload_folder_with_retry(
    api=api,
    hf_config=hf_cfg,
    repo_id=repo_id,
    repo_type=hf_cfg.get("repo_type", "model"),
    folder_path=str(local_folder),
    path_in_repo=remote_dir,
    allow_patterns=None,
    commit_message=f"Upload Stage 2 final benchmark results for {os.environ['STAGE2_FINAL_UPLOAD_RUN_NAME']}",
)
print("Upload complete!")
PY
    else
      python <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi
from hf_repo_sync import ensure_remote_repo, load_hf_config, upload_folder_with_retry
from runtime_utils import OFFICIAL_HF_ENDPOINT, get_hf_token, resolve_hf_repo_id

local_folder = Path(os.environ["STAGE2_FINAL_UPLOAD_LOCAL_DIR"])
remote_dir = os.environ["STAGE2_FINAL_UPLOAD_REMOTE_DIR"]

mirror_endpoint = os.environ.pop("HF_ENDPOINT", None)
if mirror_endpoint:
    print(f"HF_ENDPOINT was set to {mirror_endpoint}; switching back to the official Hugging Face Hub for upload.")

hf_cfg = load_hf_config("configs/hf/default.yaml")
token = get_hf_token(hf_cfg)
repo_id = resolve_hf_repo_id(hf_cfg)
api = HfApi(endpoint=OFFICIAL_HF_ENDPOINT, token=token)
ensure_remote_repo(api, hf_cfg, repo_id)

if not local_folder.exists():
    raise FileNotFoundError(f"Benchmark output folder is missing: {local_folder}")

print(f"Uploading {local_folder} to {repo_id} under {remote_dir}...")
upload_folder_with_retry(
    api=api,
    hf_config=hf_cfg,
    repo_id=repo_id,
    repo_type=hf_cfg.get("repo_type", "model"),
    folder_path=str(local_folder),
    path_in_repo=remote_dir,
    allow_patterns=None,
    commit_message=f"Upload Stage 2 final benchmark results for {os.environ['STAGE2_FINAL_UPLOAD_RUN_NAME']}",
)
print("Upload complete!")
PY
    fi
  fi
fi

cat > "${SUMMARY_DIR}/benchmark_context.md" <<EOF
# Stage 2 Final Benchmark Context

- run_name: \`${RUN_NAME}\`
- checkpoint_kind: \`${CHECKPOINT_KIND}\`
- adapter_path: \`${ADAPTER_PATH}\`
- base_model: \`${BASE_MODEL}\`
- benchmark_tag: \`${BENCHMARK_TAG}\`
- output_base_dir: \`${OUTPUT_BASE_DIR}\`
- benchmark_config: \`${CONFIG_FILE}\`
- benchmark_hf_endpoint: \`${HF_ENDPOINT:-}\`
- merged_output_root: \`${MERGED_OUTPUT_ROOT}\`
- merged_manifest_path: \`${MERGED_MANIFEST_PATH}\`
- merged_bf16_local_dir: \`${MERGED_BF16_DIR}\`
- merged_4bit_local_dir: \`${MERGED_4BIT_DIR}\`
- merged_bf16_benchmark_source: \`${MERGED_BF16_BENCH_MODEL}\`
- merged_4bit_benchmark_source: \`${MERGED_4BIT_BENCH_MODEL}\`
- merged_bf16_repo_id: \`${MERGED_BF16_REPO_ID}\`
- merged_4bit_repo_id: \`${MERGED_4BIT_REPO_ID}\`
- use_remote_merged_models: \`${USE_REMOTE_MERGED_MODELS}\`
- upload_merged_models: \`${UPLOAD_MERGED_MODELS}\`
- upload_results: \`${UPLOAD_RESULTS}\`
- remote_analysis_dir: \`${REMOTE_ANALYSIS_DIR}\`
EOF

echo
echo "Stage 2 final benchmark workflow completed."
echo "Output base dir: ${OUTPUT_BASE_DIR}"
echo "Summary dir: ${SUMMARY_DIR}"
if (( ${#SUMMARY_RUNS[@]} >= 2 )); then
  echo "Comparison summary: ${SUMMARY_DIR}/final_train_model_variants_summary.md"
fi
