# Reference: Runtime and Reproduction

## 1. Execution Model

这个仓库当前固定成两种入口语义：

- `launchers/global/*`
  - 面向 Colab 或已经准备好的全局 Python 环境
  - 本质是“先配置全局 Python，再在其中运行”
- `launchers/local/*`
  - 面向本地 / 服务器 conda 环境
  - 本质是“用 conda 管理环境，再通过 `conda run` 执行”

默认环境约定：

- 训练、数据处理、HF 发布、QA、GGUF 导出：`post-train-local`
- benchmark：`post-train-benchmark`

HF token 读取优先级：

1. `HF_TOKEN`
2. 仓库根目录 `keys.json` 里的 `hf_token`

## 2. Canonical Task Map

### 2.1 Train

默认训练就是最终采用的 short-run winner：

```bash
bash launchers/local/run_training.sh .
```

Colab / 全局 Python：

```bash
bash launchers/global/run_training.sh .
```

显式复现 full-length 正式训练反例：

```bash
bash launchers/local/run_training.sh . \
  --dataset-config configs/datasets/stage2_final_fixed_60k.yaml \
  --train-config configs/train/stage2_qwen3_8b_lora.yaml
```

核心训练 CLI：

```bash
conda run -n post-train-local python scripts/train/train_sft.py --help
```

### 2.2 Benchmark

通用 benchmark 用 `scripts/bench/run_benchmarks.py`，支持：

- `--suite all|ceval|ifeval|bfcl_v4`
- `--base-model`
- `--adapter`
- `--served-model-name`

查看帮助：

```bash
conda run -n post-train-benchmark python scripts/bench/run_benchmarks.py --help
```

评测 base model：

```bash
conda run -n post-train-benchmark python scripts/bench/run_benchmarks.py \
  --suite all \
  --base-model Qwen/Qwen3-8B \
  --label qwen3_8b_base
```

评测 adapter：

```bash
conda run -n post-train-benchmark python scripts/bench/run_benchmarks.py \
  --suite all \
  --base-model Qwen/Qwen3-8B \
  --adapter runs/<run_name>/checkpoints/best \
  --label <run_name>_best
```

评测 merged BF16：

```bash
conda run -n post-train-benchmark python scripts/bench/run_benchmarks.py \
  --suite all \
  --base-model runs/hf_publish/final_proj-stage2-best-lr1e4-r16-merged-bf16/merged_bf16 \
  --served-model-name Qwen/Qwen3-8B \
  --label stage2_best_merged_bf16
```

full-length final benchmark / failure analysis 复现：

```bash
bash launchers/local/run_benchmark.sh .
```

```bash
bash launchers/global/run_benchmark.sh .
```

### 2.3 HF Publish

最终 adapter 发布：

```bash
conda run -n post-train-local python scripts/hub/publish_final_stage2_adapter_low_traffic.py --help
```

默认有 `--dry-run` 可先跑检查：

```bash
conda run -n post-train-local python scripts/hub/publish_final_stage2_adapter_low_traffic.py --dry-run
```

merged BF16 发布：

```bash
conda run -n post-train-local python scripts/hub/publish_merged_bf16_from_adapter_repo.py --help
```

只做本地导出、不上传：

```bash
conda run -n post-train-local python scripts/hub/publish_merged_bf16_from_adapter_repo.py --skip-upload
```

### 2.4 Validate merged BF16

先补兼容资产：

```bash
conda run -n post-train-local python scripts/qa/sync_base_model_compat_assets.py
```

再跑 merged BF16 smoke test：

```bash
conda run -n post-train-local python scripts/qa/smoke_test_merged_model.py
```

### 2.5 Export and validate GGUF

完整导出：

```bash
conda run -n post-train-local python scripts/convert/export_gguf_from_merged_model.py
```

只做 GGUF smoke test：

```bash
conda run -n post-train-local python scripts/qa/smoke_test_gguf.py
```

### 2.6 Rebuild datasets

标准数据重建：

```bash
conda run -n post-train-local python scripts/data/prepare_datasets.py build-standard
```

Step 维护入口：

```bash
conda run -n post-train-local python scripts/data/prepare_datasets.py download-step-shards --help
conda run -n post-train-local python scripts/data/prepare_datasets.py sample-step-clean-general --help
conda run -n post-train-local python scripts/data/prepare_datasets.py extract-step-toolcall --help
```

## 3. Important Paths

- 训练产物：
  - `runs/<run_name>/trainer_output`
  - `runs/<run_name>/checkpoints/best`
  - `runs/<run_name>/checkpoints/final`
  - `runs/<run_name>/logs`
  - `runs/<run_name>/meta`
  - `runs/<run_name>/configs`
- benchmark 产物：
  - `analysis/stage2_final_benchmarks/<run>_<checkpoint_kind>/`
- merged smoke test：
  - `analysis/smoke_inference/final_proj-stage2-best-lr1e4-r16-merged-bf16/`
- GGUF 产物：
  - `runs/gguf_exports/final_proj-stage2-best-lr1e4-r16-merged-bf16/`
- HF 发布产物：
  - `runs/hf_publish/final_proj-stage2-best-lr1e4-r16-merged-bf16/`
  - `yyyyFan/final_proj-stage2-best-lr1e4-r16`
  - `yyyyFan/final_proj-stage2-best-lr1e4-r16-merged-bf16`

## 4. Runtime Notes

- `run_benchmark.sh` 是面向 full-length final benchmark / failure-case 的工作流前门，不是通用 `--help` 入口。
- `publish_merged_bf16_from_adapter_repo.py` 会先拉 adapter、再合并成 BF16、最后按需上传。
- `export_gguf_from_merged_model.py` 会依次补兼容资产、跑 merged BF16 smoke test、构建 `llama.cpp`、导出 `f16.gguf` 和 `q4_k_m.gguf`、再跑 GGUF smoke test。
- 当前 merged BF16 / GGUF 链路明确不依赖 Unsloth Studio。
- `llama.cpp` 的 `llama-cli` 对聊天模型默认可能进入 conversation mode，因此 smoke test 会显式规避这个行为。
- 某些服务器存在双 conda 根目录并存问题，`conda run -n ...` 不一定总是指向预期环境，必要时要先确认 `which conda` 和 `conda env list`。

## 5. What to Read Together

- [project_closeout_report.md](project_closeout_report.md)
- [reference_data_and_training.md](reference_data_and_training.md)
- [final_selection_recommendation.md](../analysis/stage2_best_benchmarks/summary/final_selection_recommendation.md)
- [final_failure_analysis.md](../analysis/stage2_final_benchmarks/summary/final_failure_analysis.md)
