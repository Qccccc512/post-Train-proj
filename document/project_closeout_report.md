# Qwen3 Tool Calling Project Closeout Report

## 1. Executive Summary

这个仓库已经完成收敛，当前最终采用的 checkpoint 是 `stage2_best_lr1e4_r16`，对应 `lr=1e-4, LoRA r=16` 的 500-step short-run best adapter。

项目现在的定位是：

- 以 `Qwen/Qwen3-8B` 为基座，维护一套可复现的 tool-calling 微调、评测、发布与 GGUF 导出流水线。
- 以 `stage2_best_lr1e4_r16` 作为最终采用对象。
- 以 `stage2train_20260408_095122_stage2_qwen3_8b_lora` 作为 full-length 正式训练失败案例留档。

关键信息继续以分析产物为准：

- 最终选型依据见 [final_selection_recommendation.md](../analysis/stage2_best_benchmarks/summary/final_selection_recommendation.md)
- full-length 失败复盘见 [final_failure_analysis.md](../analysis/stage2_final_benchmarks/summary/final_failure_analysis.md)

## 2. Project Arc

这个项目大致经历了四个阶段：

1. 数据准备与格式统一
2. Stage 1 数据配方消融
3. Stage 2 冻结配方与超参数搜索
4. Stage 2 full-length 正式训练、最终失败分析、HF 发布与 GGUF 验证

最终留下来的主线不是“继续扩实验”，而是“把结论、复现和发布固定下来”。

## 3. Final Decisions

### 3.1 Final checkpoint

- 最终采用对象：`stage2_best_lr1e4_r16`
- 对应 search run：`stage2search_20260407_173210_stage2_search_lr1e4_r16_e1_ms500`
- 默认复现配置：
  - `configs/datasets/stage2_search_fixed_10k.yaml`
  - `configs/train/stage2_search_lr1e4_r16_e1_ms500.yaml`

### 3.2 Why the full-length run is not the final model

full-length Stage 2 正式训练最终被判定为失败案例，而不是最终模型。更详细的证据链见失败分析，但结论可以简化为：

- 通用能力回退
- 指令遵循回退
- tool-calling 行为整体崩坏

因此，现在主线应当把 full-length run 当作反例归档，而不是继续围绕它做后续模型选择。

### 3.3 Final data mix

Stage 2 最终冻结的数据配方是：

- Hermes 70%
- Step tool-call 20%
- Step general 10%

对应冻结数据：

- search：`10k`
- final：`60k`

## 4. Detailed Reproduction Recipe

### 4.1 Prerequisites

先准备以下内容：

- Hugging Face token，或仓库根目录 `keys.json`
- `post-train-local`
- `post-train-benchmark`
- 对 `Qwen/Qwen3-8B`、目标 HF 命名空间、`llama.cpp` 有读写或拉取权限

推荐阅读顺序：

1. [reference_data_and_training.md](reference_data_and_training.md)
2. [reference_runtime_and_repro.md](reference_runtime_and_repro.md)
3. [final_selection_recommendation.md](../analysis/stage2_best_benchmarks/summary/final_selection_recommendation.md)
4. [final_failure_analysis.md](../analysis/stage2_final_benchmarks/summary/final_failure_analysis.md)

### 4.2 Data preparation

默认训练主链路直接复用 `datasets/processed/`，不要求先重建数据。若需要重建，先看 [reference_data_and_training.md](reference_data_and_training.md) 中的数据取舍与准备流程，然后执行：

```bash
conda run -n post-train-local python scripts/data/prepare_datasets.py build-standard
```

Step 相关维护入口同样在 `reference_data_and_training.md` 里有完整说明。

### 4.3 Train the final winner

默认训练就是最终采用的 short-run winner：

```bash
bash launchers/local/run_training.sh .
```

Colab / 全局 Python 场景：

```bash
bash launchers/global/run_training.sh .
```

也可以直接走核心 CLI：

```bash
conda run -n post-train-local python scripts/train/train_sft.py \
  --dataset-config configs/datasets/stage2_search_fixed_10k.yaml \
  --train-config configs/train/stage2_search_lr1e4_r16_e1_ms500.yaml
```

### 4.4 Reproduce the full-length failure case

显式覆写到 full-length 配置：

```bash
bash launchers/local/run_training.sh . \
  --dataset-config configs/datasets/stage2_final_fixed_60k.yaml \
  --train-config configs/train/stage2_qwen3_8b_lora.yaml
```

或者直接调用训练 CLI：

```bash
conda run -n post-train-local python scripts/train/train_sft.py \
  --dataset-config configs/datasets/stage2_final_fixed_60k.yaml \
  --train-config configs/train/stage2_qwen3_8b_lora.yaml
```

### 4.5 Benchmark

通用 benchmark 用 `scripts/bench/run_benchmarks.py`，full-length failure-case 复现用 `launchers/*/run_benchmark.sh`。详细命令口径见 [reference_runtime_and_repro.md](reference_runtime_and_repro.md)。

### 4.6 Publish and validate

最终 adapter 发布、merged BF16 发布、merged BF16 smoke test、GGUF 导出和 GGUF smoke test 的完整命令与默认产物路径，也统一放在 [reference_runtime_and_repro.md](reference_runtime_and_repro.md)。

## 5. Artifact Map

- 训练产物：`runs/<run_name>/...`
- final benchmark / failure analysis：`analysis/stage2_final_benchmarks/<run>_<checkpoint_kind>/`
- merged BF16 smoke test：`analysis/smoke_inference/final_proj-stage2-best-lr1e4-r16-merged-bf16/`
- GGUF 导出：`runs/gguf_exports/final_proj-stage2-best-lr1e4-r16-merged-bf16/`
- final adapter repo：`yyyyFan/final_proj-stage2-best-lr1e4-r16`
- merged BF16 public repo：`yyyyFan/final_proj-stage2-best-lr1e4-r16-merged-bf16`

## 6. What to Read Next

- [reference_data_and_training.md](reference_data_and_training.md)
- [reference_runtime_and_repro.md](reference_runtime_and_repro.md)
- [final_selection_recommendation.md](../analysis/stage2_best_benchmarks/summary/final_selection_recommendation.md)
- [final_failure_analysis.md](../analysis/stage2_final_benchmarks/summary/final_failure_analysis.md)
