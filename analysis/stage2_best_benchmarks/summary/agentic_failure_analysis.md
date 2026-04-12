# Stage 2 Agentic Subscore Failure Analysis

- generated_at: 2026-04-08 16:10:00
- scope: `analysis/stage2_best_benchmarks`
- primary_target: `stage2_best_lr1e4_r16`
- comparison_runs:
  - `stage2_best_base_model`
  - `stage2_best_lr2e4_r16`
  - `stage2_best_lr1e4_r16`

## Executive Summary

本次 `BFCL agentic` 里的低分，`web_search` 和 `memory` 不是同一种问题：

1. `web_search` 低分的主因是搜索后端大面积报错，模型退化只是第二层问题。
2. `memory` 低分更像 benchmark 运行态或 fixture 注入异常：很多题目的 system prompt 明确写着 memory 为空，但题目本身又要求从 memory 中读出具体答案。
3. 因为这两个子项都被运行环境或评测状态强烈污染，所以它们更适合作为“诊断信号”，不适合作为本次 Stage 2 最终超参数选择的主依据。

这份分析不改变最终选型结论：`lr1e4-r16` 仍然是当前 Stage 2 的正式训练配置。见 `final_selection_recommendation.md`。

## Source Files

- 汇总分数：
  - `analysis/stage2_best_benchmarks/benchmarks/stage2_best_lr1e4_r16/bfcl_v4/project_root/score/data_agentic.csv`
  - `analysis/stage2_best_benchmarks/benchmarks/stage2_best_lr1e4_r16/bfcl_v4/bfcl_evaluate.log`
- 典型 `web_search` 失败样本：
  - `analysis/stage2_best_benchmarks/benchmarks/stage2_best_lr1e4_r16/bfcl_v4/project_root/score/Qwen_Qwen3-8B/agentic/BFCL_v4_web_search_base_score.json`
- 典型 `memory` 失败样本：
  - `analysis/stage2_best_benchmarks/benchmarks/stage2_best_lr1e4_r16/bfcl_v4/project_root/score/Qwen_Qwen3-8B/agentic/memory/kv/BFCL_v4_memory_kv_score.json`
  - `analysis/stage2_best_benchmarks/benchmarks/stage2_best_lr1e4_r16/bfcl_v4/project_root/score/Qwen_Qwen3-8B/agentic/memory/vector/BFCL_v4_memory_vector_score.json`
  - `analysis/stage2_best_benchmarks/benchmarks/stage2_best_lr1e4_r16/bfcl_v4/project_root/result/Qwen_Qwen3-8B/agentic/memory/kv/BFCL_v4_memory_kv_result.json`
  - `analysis/stage2_best_benchmarks/benchmarks/stage2_best_lr1e4_r16/bfcl_v4/project_root/result/Qwen_Qwen3-8B/agentic/memory/vector/BFCL_v4_memory_vector_result.json`
  - `analysis/stage2_best_benchmarks/benchmarks/stage2_best_lr1e4_r16/bfcl_v4/project_root/result/Qwen_Qwen3-8B/agentic/memory/rec_sum/BFCL_v4_memory_rec_sum_result.json`

## Current Scores

以最终选中的 `lr1e4-r16` 为例：

| Subsection | Accuracy |
| --- | ---: |
| Web Search Summary | 4.00% |
| Web Search Base | 3.00% |
| Web Search No Snippet | 5.00% |
| Memory Summary | 1.08% |
| Memory KV | 1.94% |
| Memory Vector | 1.29% |
| Memory Recursive Summarization | 0.00% |

对应来源见 `data_agentic.csv` 与 `bfcl_evaluate.log`。

## Web Search Analysis

### Main Finding

`web_search` 的主要问题不是模型完全不会调用搜索，而是搜索工具本身在大多数样本里返回：

`Failed to retrieve the search results from server. Please try again later.`

这会把模型推到两种失败路径：

- 工具调用后拿不到结果，只能硬猜或输出 `I cannot answer`
- 干脆不调工具，直接用过时常识做时间判断

### Cross-Run Evidence

三组 run 都有大面积同类错误，说明这不是某个 adapter 的特有退化：

| run | web_base server error rows | web_base no-tool rows | web_base temporal confusion rows | web_no_snippet server error rows | web_no_snippet no-tool rows | web_no_snippet temporal confusion rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `stage2_best_base_model` | 73 / 100 | 27 / 100 | 13 / 100 | 70 / 100 | 30 / 100 | 12 / 100 |
| `stage2_best_lr2e4_r16` | 71 / 100 | 29 / 100 | 12 / 100 | 71 / 100 | 29 / 100 | 16 / 100 |
| `stage2_best_lr1e4_r16` | 77 / 100 | 23 / 100 | 17 / 100 | 73 / 100 | 27 / 100 | 18 / 100 |

解释：

- `server error rows`：raw inference log 里出现搜索后端报错的样本数
- `no-tool rows`：完全没有走 tool call，直接输出 final answer 的样本数
- `temporal confusion rows`：出现 “2024 还没发生”/“as of 2023” 之类时间判断错误的样本数

### Representative Failures

#### 1. 工具调用后端报错，模型转为硬猜

样本：`web_search_base_0`

- 模型先调用：
  - `search_engine_query(keywords="most expensive tea producing country")`
  - `search_engine_query(keywords="richest billionaire India Forbes 2025")`
- 两个 tool 都返回搜索失败
- 最终模型自己猜成 `Mukesh Ambani`
- 评测标准答案是 `Zhang Yiming`

这个样本说明：即使模型愿意调用工具，只要搜索后端不可用，最终分数也会迅速塌掉。

#### 2. 不调工具，直接用过时时间常识拒答

样本：`web_search_base_1`

- 用户问题问的是 2024 Super Bowl halftime show performer 对应城市 NFL 战绩
- 模型没有发起任何工具调用
- 直接回答：
  - “The 2024 Super Bowl has not yet occurred...”

这类错误不是搜索接口问题，而是模型 fallback 策略不可靠；但它在总体低分中的占比仍明显小于搜索后端报错。

### Web Search Conclusion

对当前归档来说，`web_search` 低分主要受 benchmark runtime 的搜索接口异常污染。模型层面仍有两个次级问题：

- fallback 时会过度自信地猜答案
- 一部分样本会直接触发时间错觉或“无工具可用”判断

但在没有先修复搜索后端前，不能把这组分数当作纯模型能力结论。

## Memory Analysis

### Main Finding

`memory` 低分比 `web_search` 更值得警惕，因为它看起来不只是模型没学会，而是 **评测输入状态本身就有问题**。

最强证据：

- `memory_kv` 的 155 / 155 个样本，system prompt 都写着：
  - `There is no content in the core memory at this point.`
- `memory_rec_sum` 的 155 / 155 个样本，system prompt 都写着：
  - `There is no content in the memory at this point.`
- `memory_vector` 的 155 / 155 个样本里：
  - prompt 只展示了 `Core Memory`
  - `Core Memory` 内容是 `{}`
  - 没有出现 `Here is the content of your Archival Memory from previous interactions:`

如果 benchmark 题目期待模型回答例如用户姓名、任务时间、订单信息，那么在 prompt 里明确把 memory 设为空，会让这些题天然不可答。

### Cross-Run Evidence

这同样不是 `lr1e4-r16` 独有问题，三组 run 都高度一致：

| run | mem_kv no-tool rows | mem_kv `Memory is empty` | mem_kv `division by zero` | mem_rec no-tool rows | mem_rec empty-memory prompt rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stage2_best_base_model` | 150 / 155 | 83 | 5 | 145 / 155 | 155 / 155 |
| `stage2_best_lr2e4_r16` | 152 / 155 | 85 | 3 | 144 / 155 | 155 / 155 |
| `stage2_best_lr1e4_r16` | 152 / 155 | 88 | 3 | 144 / 155 | 155 / 155 |

附加观察：

- `memory_vector` 三组 run 都几乎不调工具
- 少数真的调了 vector retrieve 的样本，返回也常是 `{"result": []}`

### Representative Failures

#### 1. prompt 明示 core memory 为空，但题目要回忆姓名

样本：`memory_kv_0-customer-0`

- 用户问：`What is my first name?`
- 标准答案：`Michael`
- 但 system prompt 里写着：
  - `Here is the content of your Core Memory from previous interactions:`
  - `There is no content in the core memory at this point.`
- 模型最终答：
  - `I cannot answer this question`

这类失败更像是 benchmark 输入状态没有按预期注入，而不是模型真正“忘了”。

#### 2. vector memory 根本没有 archival context

样本：`memory_vector_0-customer-0`

- 用户同样问：`What is my first name?`
- 标准答案：`Michael`
- prompt 里只出现：
  - `Here is the content of your Core Memory from previous interactions:`
  - `{}`
- 没有任何 archival memory 内容段落
- 模型只能回答：
  - `I do not know`

也就是说，vector memory 这批题里，检索目标所在的记忆空间看起来没有被预载进 prompt。

#### 3. 少数实际工具调用也在报运行时错误

三个典型例子：

- `memory_kv_134-notetaker-4`
  - 调了 `archival_memory_key_search` 和 `core_memory_key_search`
  - 两个 tool 都返回 `Error during execution: division by zero`
- `memory_vector_8-customer-8`
  - 调了 `core_memory_retrieve(query="espresso machine event", top_k=1)`
  - tool 返回 `{"result": []}`
- `memory_rec_sum_17-customer-17`
  - 调了 `memory_retrieve()`
  - tool 返回 `{"error": "Memory is empty."}`

这说明 memory 低分不只是“模型没调工具”，而是 **就算调了，memory backend 也常常给出空状态或异常**。

### Memory Conclusion

当前这批 `memory` 结果不能被视为模型真实 memory 能力评估，更像是：

- prompt state 注入不完整
- memory backend 初始化为空
- 少数检索接口还有运行时错误

在这类前提下，`1%-2%` 的 memory 分数几乎没有超参数选择价值。

## What This Means for Hyperparameter Selection

这次 Stage 2 最终选型仍然成立，原因是：

- `web_search` 和 `memory` 的异常具有很强的跨 run 一致性
- 它们更像 benchmark runtime/fixture 问题，而不是 adapter 间的稳定能力差异
- 真正更可靠的选择信号仍然来自：
  - `BFCL Overall`
  - `BFCL Live`
  - `IFEval`
  - `C-Eval`

因此，不建议因为 `memory` 或 `web_search` 的绝对低分去推翻 `lr1e4-r16` 的正式训练决策。

## Recommended Follow-Ups

### Priority 1: Fix benchmark runtime before using these scores again

1. 修 `web_search` 的搜索后端错误
   - 先做 5-10 条 canary task
   - 确认 `search_engine_query` 不再大面积返回 server error
2. 修 `memory` 的 fixture/state 注入
   - `memory_kv` 不应在所有 recall 题里都显示 core memory empty
   - `memory_rec_sum` 不应在所有题里都显示 memory empty
   - `memory_vector` 应明确检查 archival memory 是否被正确加载

### Priority 2: Add guardrails to the benchmark pipeline

建议在 BFCL 入口前加两个 fail-fast 检查：

1. `web_search` 预检
   - 随机发一个 `search_engine_query`
   - 若连续失败则中断评测
2. `memory` 预检
   - 在每个 memory suite 开始前验证 prompt 中确实存在非空 memory state
   - 若为空则直接标记 benchmark setup failure，而不是继续产出误导性 accuracy

### Priority 3: Re-run only the contaminated subsections

如果后续要做更细的 agentic 结论，建议只重跑：

- `BFCL web_search_base`
- `BFCL web_search_no_snippet`
- `BFCL memory_kv`
- `BFCL memory_vector`
- `BFCL memory_rec_sum`

不需要重跑整套 BFCL。

## Bottom Line

本次归档里的低 `web_search` / `memory` 分数，不能简单解读为“模型 agentic 能力很差”。

- `web_search`：主要被搜索服务异常污染
- `memory`：更像 benchmark memory state 没有正确注入或 backend 初始化异常

因此，这两项目前更适合用来暴露评测链路问题，而不是用来判断 `lr1e4-r16` 和 `lr2e4-r16` 的真实优劣。
