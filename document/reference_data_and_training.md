# Reference: Data Selection and Training Design

## 1. Data Selection Principles

这套项目的数据策略不是“找一个看起来最大的集合”，而是围绕目标能力做组合：

- 让模型保住通用语言能力
- 让模型学会稳定的 tool calling
- 让模型在长上下文、agentic、多轮交互上有覆盖
- 控制清洗成本、模板转换成本和训练成本

因此，数据源的取舍优先级不是原始规模，而是：

1. 格式和监督信号是否足够接近 Qwen3 目标模板
2. 数据质量是否足够稳
3. 是否能覆盖当前主任务缺口
4. 转换与维护成本是否可接受

## 2. Why These Datasets

| 数据源 | 角色 | 选择理由 | 主要取舍 |
| --- | --- | --- | --- |
| Hermes Reasoning Tool Use | 主干 multi-turn tool calling | Atropos RL / BFCL 语义对齐，多轮复杂度适中，和当前目标能力最贴近 | 数据量不算最大，但质量和结构最稳 |
| xLAM Function Calling 60K | 简单单轮 FC 补充 | 格式相对规整、单轮调用清晰，适合补简单函数调用 | 容易把训练重心拉向“简单、短、好优化”的方向 |
| Step-3.5-Flash-SFT general | 通用 rehearsal / 抗遗忘 | 作为非 tool-calling 通用表达补充，控制模型别只向工具调用分布过拟合 | 序列更长，训练成本更高，不能占太大比例 |
| Step-3.5-Flash-SFT tool-call | 长上下文 / agentic 补充 | 覆盖 Hermes 不那么密集的长轨迹、web-style、memory-style、工具密集场景 | 噪声更高、成本更高，适合小比例定向补充 |
| Qwen3.5-toolcalling-v2 | 对照线 | 便于比较开箱即用的 tool-calling 风格数据 | 社区拼接感更重、长对话比例高，不适合作主干 |
| Glaive Function Calling v2 | 历史基线 / 对照线 | 经典课程推荐数据集，便于做 baseline 对比 | 风格较旧、转换成本高、质量不如 Hermes 贴近当前目标 |

### 简短结论

- **Hermes** 是最稳的主干。
- **xLAM** 适合作简单单轮补充，但不适合单独放大成主干。
- **Step general** 主要承担抗遗忘，不应成为主混合的大头。
- **Step tool-call** 是后续 Stage 2 收口里最有价值的补充源，但必须控制比例。
- **Qwen3.5 / Glaive** 更适合作对照，不适合作当前主线。

## 3. Historical Decision Path

### 3.1 Stage 1

Stage 1 用 5k / 1 epoch 的方式做了 A-I 九组配方消融，重点不是找“绝对真理”，而是排除明显不合适的方向。

阶段一给出的直接经验是：

- xLAM 和 Glaive 这类“短、整齐、好优化”的数据，训练内 loss 往往更好看，但不等于外部 benchmark 真正更好。
- Step general 会显著抬高长度和训练成本，但没有换来相应收益。
- Step tool-call 的独立价值存在，但单独成组时成本和噪声都太高，更适合小比例补充。
- Hermes 仍然是最可信的主干。

### 3.2 Stage 2

Stage 2 把阶段一的结论进一步收束成最终配方：

- Hermes 70%
- Step tool-call 20%
- Step general 10%

冻结数据大小为：

- search：10k
- final：60k

Stage 2 search 的最终赢家是 `lr=1e-4, r=16`，也就是 `stage2_best_lr1e4_r16`。

## 4. Data Preparation Workflow

### 4.1 Canonical locations

- 原始或临时重建材料：`datasets/raw/`
- 标准处理后训练数据：`datasets/processed/`
- 默认训练链路优先复用处理后数据，不要求每次都重新生成

### 4.2 Main entrypoint

```bash
conda run -n post-train-local python scripts/data/prepare_datasets.py build-standard
```

这个入口会把标准训练数据统一落到 `datasets/processed/`，并产出归一化摘要。

### 4.3 Step-specific helpers

```bash
conda run -n post-train-local python scripts/data/prepare_datasets.py download-step-shards --help
```

```bash
conda run -n post-train-local python scripts/data/prepare_datasets.py sample-step-clean-general --help
```

```bash
conda run -n post-train-local python scripts/data/prepare_datasets.py extract-step-toolcall --help
```

`sample-step-clean-general` 的当前关键行为是：

- `--only-shards`：严格按命令行给出的 shard 列表
- `--all-local-shards`：使用本地已下载的全部 general shards，并按 `chunk_<n>.json` 顺序排序
- 不传上面两个参数：使用历史固定的 10-shard baseline

历史固定的 10 个 shard 是：

```text
json/general/chunk_0.json
json/general/chunk_4.json
json/general/chunk_47.json
json/general/chunk_31.json
json/general/chunk_63.json
json/general/chunk_5.json
json/general/chunk_29.json
json/general/chunk_18.json
json/general/chunk_79.json
json/general/chunk_56.json
```

这意味着当前 Step general 复现逻辑的核心其实是：

1. 先固定 shard 集合
2. 再筛 no-tool 且长度过阈值的候选
3. 最后用全局 `sample-seed` 抽样

## 5. Normalization Rules

统一处理后的训练数据遵循这些规则：

- 统一成 `messages` + `tools` 结构
- Hermes / Step / Qwen3.5 的工具定义默认放在 system prompt 里，`tools = null`
- xLAM 保留顶层 `tools`
- 删除 assistant `<think>` 不平衡样本
- 删除不是以 assistant 收尾的样本
- 对所有 assistant 回复补齐空的 `<think>\n</think>` 前缀
- 统一对齐 Qwen3 chat template
- 不能被模板正确渲染的样本会被跳过并写入诊断信息

### `<think>` 训练策略

当前主线采用的是：

- 保留 `<think>` / `</think>` 标签本身的 loss
- 仅对 `<think>` 内部内容做 mask
- 不把标签和内部内容一股脑全 mask 掉

这样做的目标是保留 Qwen3 thinking mode 的触发语义，同时避免异源 reasoning 内容被直接当作必须模仿的目标。

## 6. Training Design

### 6.1 Stage 1

阶段一的主要设置：

- 总样本数：5000
- `lr = 2e-5`
- `LoRA rank = 16`
- `epoch = 1`
- `max_seq_length = 8192`
- `packing = False`
- `per_device_train_batch_size = 1`
- `per_device_eval_batch_size = 1`
- `gradient_accumulation_steps = 16`
- `dataset_num_proc = 16`
- `dataloader_num_workers = 16`
- train/val：95/5

阶段一的作用是做数据配方消融，不是做最终胜负宣判。

### 6.2 Stage 2 search

Stage 2 search 固定配方、搜索超参数：

- 数据：10k frozen mix
- `max_steps = 500`
- `max_seq_length = 8192`
- `packing = False`
- `eval_steps = 50`
- `save_steps = 50`
- `warmup_steps = 20`
- `load_best_model_at_end = True`
- `metric_for_best_model = eval_loss`
- `greater_is_better = False`
- train/val：95/5

Stage 2 search 的最终赢家是 `stage2_best_lr1e4_r16`。

### 6.3 Stage 2 final

最终 full-length 正式训练使用：

- 数据：60k frozen mix
- `configs/train/stage2_qwen3_8b_lora.yaml`
- `train/val` 按 98/2 口径（95/5的 eval 成本较高）

这个运行后来被失败分析否定，因此它只保留为反例归档。

## 7. What to Use for Selection

当前选型信号优先级是：

1. 外部 benchmark，尤其 BFCL live / overall
2. IFEval
3. C-Eval
4. 训练期 eval_loss

不要把 `eval_loss` 单独当作最终胜负判据。Stage 1 和 Stage 2 都已经证明，训练内指标只能做辅助。

## 8. Where to Read More

- [project_closeout_report.md](project_closeout_report.md)
- [final_selection_recommendation.md](../analysis/stage2_best_benchmarks/summary/final_selection_recommendation.md)
- [final_failure_analysis.md](../analysis/stage2_final_benchmarks/summary/final_failure_analysis.md)
