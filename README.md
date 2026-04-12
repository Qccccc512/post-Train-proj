# Qwen3 Tool Calling Fine-tuning

这个仓库当前已经收敛到最终结论：**Stage 2 就是最后一步**。  
当前唯一采用的最终 checkpoint 是 `stage2_best_lr1e4_r16`，也就是 `lr=1e-4, LoRA r=16` 的 500-step short-run best adapter。

当前最终采用相关对象：

- 最终采用的 adapter 结论：`stage2_best_lr1e4_r16`
- 最终 adapter repo：`yyyyFan/final_proj-stage2-best-lr1e4-r16`
- 目标 merged BF16 public repo：`yyyyFan/final_proj-stage2-best-lr1e4-r16-merged-bf16`
- 归档用 full-length 正式训练 run：
  - `yyyyFan/final_proj/runs/stage2train_20260408_095122_stage2_qwen3_8b_lora`

当前主报告与引用文档见：

- [project_closeout_report.md](document/project_closeout_report.md)
- [reference_data_and_training.md](document/reference_data_and_training.md)
- [reference_runtime_and_repro.md](document/reference_runtime_and_repro.md)
- [final_selection_recommendation.md](analysis/stage2_best_benchmarks/summary/final_selection_recommendation.md)
- [final_failure_analysis.md](analysis/stage2_final_benchmarks/summary/final_failure_analysis.md)

## for_human

### 快速了解

这是一个围绕 `Qwen/Qwen3-8B` 的 tool-calling 微调、评测、发布与 GGUF 导出仓库，当前最终采用的 checkpoint 是 `stage2_best_lr1e4_r16`。

### 你需要先准备

1. Hugging Face 账号，以及对 `Qwen/Qwen3-8B` 和你的目标 HF 命名空间的访问权限。
2. 仓库根目录的 `keys.json` 或环境变量 `HF_TOKEN`。
3. `conda` 环境 `post-train-local` 和 `post-train-benchmark`。

如果需要了解更多，可以先浏览完 README.md 的内容，再选择性阅读以下文档：

- [project_closeout_report.md](document/project_closeout_report.md)
- [reference_data_and_training.md](document/reference_data_and_training.md)
- [reference_runtime_and_repro.md](document/reference_runtime_and_repro.md)
- [final_selection_recommendation.md](analysis/stage2_best_benchmarks/summary/final_selection_recommendation.md)
- [final_failure_analysis.md](analysis/stage2_final_benchmarks/summary/final_failure_analysis.md)

更推荐的做法是完成基本的配置后，向 agent 发送以下提示词：

```
从 README.md 开始，探索、理解当前工作目录，了解其结构和工作内容，做好回答项目相关问题、使用该项目、进行二次开发的准备。
```

然后，让 agent 帮你干活吧！

## for_agent

### 当前项目结论

当前项目已经完成收敛，保留的最终结论是：

- 最终采用对象：`stage2_best_lr1e4_r16`
- 基座模型：`Qwen/Qwen3-8B`
- 当前固定数据配方：
  - `Hermes 70% + Step tool-call 20% + Step general 10%`
- full-length 正式训练 run 只保留为失败案例归档，不作为最终模型采用

### 项目能力

当前仓库主要支持 5 类工作：

1. 训练 Qwen3-8B 的 LoRA tool-calling 模型。
2. 运行 benchmark，比较 base model、adapter、merged model。
3. 把最终 adapter 整理并发布到 Hugging Face。
4. 把最终 adapter 合并成 merged BF16 full model 并发布到 Hugging Face。
5. 不依赖 Unsloth Studio，直接对 merged BF16 做 smoke test、导出 GGUF、再做 GGUF 推理验证。

### 必要配置与环境

项目优先使用 `conda` 管理工作环境。

默认约定：

- 训练、数据处理、HF 发布、QA、GGUF 导出：使用 `post-train-local`
- benchmark：使用 `post-train-benchmark`
- 除 Colab 或明确的全局运行场景外，不建议直接依赖系统 Python 或临时裸环境

当前入口分两类：

- `launchers/global/*`
  - 适合 Colab 或计划使用全局 Python 环境而不担心造成破坏
- `launchers/local/*`
  - 适合本地 / 服务器 conda 环境

本地常用 setup 入口：

- [setup_train_env.sh](launchers/local/setup_train_env.sh)
- [setup_bench_env.sh](launchers/local/setup_bench_env.sh)

全局环境常用 setup 入口：

- [setup_train_env.sh](launchers/global/setup_train_env.sh)
- [setup_bench_env.sh](launchers/global/setup_bench_env.sh)

仓库当前主路径只要求一个秘钥：

- `hf_token`

`keys.json` 最小示例：

```json
{
  "hf_token": "hf_xxx"
}
```

当前代码读取秘钥的优先级是：

1. 先读环境变量 `HF_TOKEN`
2. 如果没设置，再 fallback 到仓库根目录的 `keys.json`

相关实现见：

- [runtime_utils.py](scripts/common/runtime_utils.py)
- [default.yaml](configs/hf/default.yaml)

### 上手指引

如果你需要快速建立上下文，建议按这个顺序：

1. 先读：
   - [project_closeout_report.md](document/project_closeout_report.md)
   - [reference_data_and_training.md](document/reference_data_and_training.md)
   - [reference_runtime_and_repro.md](document/reference_runtime_and_repro.md)
   - [final_selection_recommendation.md](analysis/stage2_best_benchmarks/summary/final_selection_recommendation.md)
   - [final_failure_analysis.md](analysis/stage2_final_benchmarks/summary/final_failure_analysis.md)
2. 再用 `--help` 看 CLI：
   - `conda run -n post-train-local python scripts/data/prepare_datasets.py --help`
   - `conda run -n post-train-local python scripts/train/train_sft.py --help`
   - `conda run -n post-train-benchmark python scripts/bench/run_benchmarks.py --help`
   - `conda run -n post-train-local python scripts/hub/publish_final_stage2_adapter_low_traffic.py --help`
   - `conda run -n post-train-local python scripts/hub/publish_merged_bf16_from_adapter_repo.py --help`
   - `conda run -n post-train-local python scripts/convert/export_gguf_from_merged_model.py --help`
3. 再决定是训练、benchmark、HF 发布，还是 GGUF 导出。

### 目录地图

当前仓库主要目录：

- [analysis](analysis)
  - 冻结的分析结果、发布记录、评测产物
- [benchmark](benchmark)
  - 第三方 benchmark 框架源码
- [configs](configs)
  - 训练、数据、HF、benchmark 配置
- [datasets](datasets)
  - 当前训练入口使用 `datasets/processed/`
  - `datasets/raw/` 只作为可选本地重建缓存，不是训练/评测主链路必需
- [document](document)
  - 主报告、引用文档、少量专题说明
- [launchers](launchers)
  - 面向 human 的平台入口
- [scripts](scripts)
  - 主要实现代码

`scripts/` 当前分域如下：

- [scripts/common](scripts/common)
- [scripts/train](scripts/train)
- [scripts/bench](scripts/bench)
- [scripts/data](scripts/data)
- [scripts/hub](scripts/hub)
- [scripts/analysis](scripts/analysis)
- [scripts/qa](scripts/qa)
- [scripts/convert](scripts/convert)

### 代码入口

训练主入口：

- [run_training.sh](launchers/global/run_training.sh)
- [run_training.sh](launchers/local/run_training.sh)
- [train_sft.py](scripts/train/train_sft.py)

benchmark 主入口：

- [run_benchmark.sh](launchers/global/run_benchmark.sh)
- [run_benchmark.sh](launchers/local/run_benchmark.sh)
- [run_benchmarks.py](scripts/bench/run_benchmarks.py)

说明：

- `run_benchmarks.py`
  - 通用 benchmark CLI，用于评测任意 `base` / `adapter` / `merged model`
- `launchers/*/run_benchmark.sh`
  - 归档的 full-length final benchmark / failure-analysis 复现工作流

HF 发布入口：

- [publish_final_stage2_adapter_low_traffic.py](scripts/hub/publish_final_stage2_adapter_low_traffic.py)
- [publish_merged_bf16_from_adapter_repo.py](scripts/hub/publish_merged_bf16_from_adapter_repo.py)
- [publish_merged_bf16.sh](launchers/local/publish_merged_bf16.sh)
- [publish_merged_bf16.sh](launchers/global/publish_merged_bf16.sh)

非 Unsloth 验证与 GGUF 导出入口：

- [model_artifacts.py](scripts/common/model_artifacts.py)
- [sync_base_model_compat_assets.py](scripts/qa/sync_base_model_compat_assets.py)
- [smoke_test_merged_model.py](scripts/qa/smoke_test_merged_model.py)
- [smoke_test_gguf.py](scripts/qa/smoke_test_gguf.py)
- [export_gguf_from_merged_model.py](scripts/convert/export_gguf_from_merged_model.py)

### 默认配置与默认路径

默认训练配置：

- dataset config：`configs/datasets/stage2_search_fixed_10k.yaml`
- train config：`configs/train/stage2_search_lr1e4_r16_e1_ms500.yaml`

当前 canonical 数据目录：

- 处理后训练数据：`datasets/processed`
- HF 数据远端目录：`datasets`
- HF 本地缓存目录：`runs/_hf_cache/datasets`

当前不再把处理后数据按 `v1/v2/v3/v4` 分层落目录；历史 `v4` 产物已归并到 `datasets/processed/`。

当前建议保留并继续复用的配置集合：

- benchmark：`configs/benchmark/default.yaml`
- datasets：`configs/datasets/stage1_default.yaml`、`configs/datasets/stage2_default.yaml`、`configs/datasets/stage2_final_fixed_60k.yaml`、`configs/datasets/stage2_search_fixed_10k.yaml`
- train：`configs/train/stage1_qwen3_8b_lora.yaml`、`configs/train/stage2_qwen3_8b_lora.yaml`、`configs/train/stage2_search_*.yaml`

当前默认采用的训练入口组合：

- `configs/datasets/stage2_search_fixed_10k.yaml`
- `configs/train/stage2_search_lr1e4_r16_e1_ms500.yaml`

原因：

- 这是项目最终采用的 short-run winner 对应配置
- `stage2 final 60k + stage2_qwen3_8b_lora` 作为 full-length 正式训练反例保留，但不再作为默认训练入口

已经在清理中移除的历史配置类型：

- 无引用的 benchmark smoke 配置
- 只存在于早期训练草案中的 fulltrain 配置
- 只配套 archive 本地 smoke 脚本的 0.6B smoke 训练配置

默认 HF 配置：

- [default.yaml](configs/hf/default.yaml)

默认 merged BF16 输入目录（按仓库根目录解析）：

- `runs/hf_publish/final_proj-stage2-best-lr1e4-r16-merged-bf16/merged_bf16`
- 如果你不在项目根目录下阅读或执行，请先切回仓库根目录再使用这个相对路径

默认 smoke test 输出目录：

- `analysis/smoke_inference/final_proj-stage2-best-lr1e4-r16-merged-bf16`

默认 GGUF 输出目录：

- `runs/gguf_exports/final_proj-stage2-best-lr1e4-r16-merged-bf16`

默认 `llama.cpp` checkout 目录：

- `tools/llama.cpp`

### Agent 最小检索路径

如果 agent 需要快速建立上下文，优先跑：

```bash
rg --files launchers scripts configs document analysis | sort
```

```bash
rg -n "argparse|build_parser|main\\(" scripts
```

```bash
rg -n "stage2_best_lr1e4_r16|final_proj-stage2-best-lr1e4-r16|merged_bf16|gguf" README.md document analysis scripts
```

如果 agent 只想看当前支持的 CLI：

```bash
rg -n "build_parser\\(|argparse\\.ArgumentParser" scripts
```

### CLI 索引

训练：

- `bash launchers/global/run_training.sh .`
- `bash launchers/local/run_training.sh .`
- `conda run -n post-train-local python scripts/train/train_sft.py --help`

数据准备 / 重建：

- `conda run -n post-train-local python scripts/data/prepare_datasets.py --help`
- `document/reference_data_and_training.md` 记录了数据选择理由、Step shard baseline 与复现逻辑

benchmark：

- `conda run -n post-train-benchmark python scripts/bench/run_benchmarks.py --help`
- `bash launchers/global/run_benchmark.sh .`
- `bash launchers/local/run_benchmark.sh .`

HF 发布：

- `conda run -n post-train-local python scripts/hub/publish_final_stage2_adapter_low_traffic.py --help`
- `conda run -n post-train-local python scripts/hub/publish_merged_bf16_from_adapter_repo.py --help`
- `bash launchers/local/publish_merged_bf16.sh .`
- `bash launchers/global/publish_merged_bf16.sh .`

非 Unsloth merged model 验证：

- `conda run -n post-train-local python scripts/qa/sync_base_model_compat_assets.py --help`
- `conda run -n post-train-local python scripts/qa/smoke_test_merged_model.py --help`

GGUF 导出与验证：

- `conda run -n post-train-local python scripts/convert/export_gguf_from_merged_model.py --help`
- `conda run -n post-train-local python scripts/qa/smoke_test_gguf.py --help`

服务器推荐调用方式：

- `conda run -n post-train-local python scripts/qa/smoke_test_merged_model.py`
- `conda run -n post-train-local python scripts/convert/export_gguf_from_merged_model.py`
- `conda run -n post-train-local python scripts/qa/smoke_test_gguf.py`

### 环境注意事项

1. 当前主流程只要求 HF token；秘钥读取优先级是 `HF_TOKEN`，其次是仓库根目录 `keys.json` 中的 `hf_token`。
2. `Unsloth Studio` 当前在目标服务器上存在加载 bug，因此 merged BF16 验证和 GGUF 导出链路明确不依赖 Unsloth Studio。
3. `llama.cpp` 的 `llama-cli` 对聊天模型默认会进入 conversation mode；当前仓库的 GGUF smoke test 已显式加上 `-no-cnv`，避免跑完 prompt 后停在聊天框。
4. 某些服务器存在双 conda 并存问题；`conda run -n post-train-local ...` 不一定总是指向预期环境。

### 整理后保留的参考文档

- [project_closeout_report.md](document/project_closeout_report.md)
- [reference_data_and_training.md](document/reference_data_and_training.md)
- [reference_runtime_and_repro.md](document/reference_runtime_and_repro.md)
- [final_selection_recommendation.md](analysis/stage2_best_benchmarks/summary/final_selection_recommendation.md)
- [final_failure_analysis.md](analysis/stage2_final_benchmarks/summary/final_failure_analysis.md)
