# Stage 2 Final Failure Analysis

- generated_at: `2026-04-09`
- scope: `final bf16 merged model` vs `base` vs `stage2 search best`
- final_snapshot: `analysis/stage2_final_benchmarks/stage2train_20260408_095122_stage2_qwen3_8b_lora_best`
- final_model_label: `stage2_final_merged_bf16_model_partial`
- plan_status: `completed`

## Executive Summary

这次 Stage 2 正式训练的最终 `bf16` 模型已经可以判定为**明显失败**，不需要再继续进行完整 BF16 或 INT4 评测。

失败结论不是建立在单一 benchmark 上，而是三类证据同时成立：

1. `C-Eval` 从约 `79.7%` 级别直接掉到 `72.29%`，通用知识与基础推理明显退化。
2. `IFEval` 也同步下降，尤其 `Loose` 从 `42.70%` 掉到 `38.45%`，指令遵循没有因为长训变强，反而更差。
3. BFCL 即便只看已恢复出的官方 `partial-eval`，也已经出现：
   - `Non-Live = 13.29%`
   - `Live = 4.59%`
   - `Multi-Turn = 0.00%`

更重要的是，short run 并不是失败的。`500-step` 的 `lr1e4-r16` 相对 base 有小幅、方向一致的改善，说明：

- 这套 Stage 2 冻结配方不是天然坏掉的
- 当前训练/模板/合并/评测链路也不像一开始就全错
- 真正出问题的是**继续训到 full-length 终点之后的行为漂移**

当前最可信的结论是：**full-length Stage 2 训练把模型行为过度拉向窄的 Stage 2 数据分布，破坏了原始 Qwen3-8B 的通用推理、指令约束和 tool-calling 稳定性。**


## Metric Comparison

| run | C-Eval | IFEval Strict | IFEval Loose | BFCL Overall | BFCL Non-Live | BFCL Live | BFCL Multi-Turn | Web Search | Memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 79.79% | 34.38% | 40.48% | 34.95% | 88.38% | 78.90% | 31.87% | 2.00% | 1.08% |
| lr2e4-r16 | 79.57% | 35.67% | 41.59% | 35.21% | 88.35% | 79.50% | 33.38% | 0.50% | 1.08% |
| lr1e4-r16 | 79.72% | 35.67% | 42.70% | 35.67% | 88.73% | 80.24% | 32.25% | 4.00% | 1.08% |
| final bf16 partial | 72.29% | 31.98% | 38.45% | 11.76%* | 13.29%* | 4.59%* | 0.00%* | 3.50%* | N/A |

\* `final bf16` 的 BFCL 是官方 `partial-eval` 结果，不是完整 official run。  
\* 缺失 `memory/kv`、`memory/vector`、`memory/rec_sum` 三项。  
\* `multi-turn` 四个子类都已开跑，但内部覆盖仍不完整，见下文。

## Why Short Training Helped

short run 不是“完全没效果”，而是呈现了一个很典型的**轻度行为对齐成功**信号。

相对 base，最终选中的 short-run `lr1e4-r16` 表现为：

- `C-Eval`: `79.79% -> 79.72%`
- `IFEval Strict`: `34.38% -> 35.67%`
- `IFEval Loose`: `40.48% -> 42.70%`
- `BFCL Overall`: `34.95% -> 35.67%`
- `BFCL Live`: `78.90% -> 80.24%`
- `BFCL Multi-Turn`: `31.87% -> 32.25%`

更细一点看：

- `C-Eval` 只掉了 `0.07pt`，远在 `±1.07pt` 的标准误差内，基本可以看作持平
- `IFEval Loose` 提升了 `2.22pt`，已经接近 / 略超该项 `±2.13pt` 的标准误差
- BFCL 主指标也都是小幅正向

这说明在 `500-step` 区间内，当前固定配方更像是在对 base model 做**温和 steering**：

- LoRA 更新步数有限，还没来得及大幅改写 base 的先验能力
- tool / instruction 数据足以提醒模型“按项目目标输出”
- 但还没有覆盖掉原始 Qwen3-8B 的通用知识、格式纪律和多轮稳态

因此，short run 的改善本身就是一个强信号：**数据、模板、mask、评测链路并不是一上来就错的；问题出在 full-length 继续往下训之后。**

## BFCL Coverage Note

这次 final BFCL 的“不完整”不只是缺了 memory 三项；`multi-turn` 组内部也是 partial：

- `multi_turn_base`: `200 / 200`
- `multi_turn_long_context`: `200 / 200`
- `multi_turn_miss_func`: `188 / 200`
- `multi_turn_miss_param`: `33 / 200`

所以更准确的表述应该是：

- `live` 与 `non_live` 是完整生成并补评分的
- `multi-turn` 的四个子类都开始跑了
- 但其中 `miss_func` 和尤其 `miss_param` 没有完整生成

即便如此，已恢复出的官方 `partial-eval` 仍然给出：

- `Multi Turn Overall Acc = 0.00%`
- `Base = 0.00%`
- `Miss Func = 0.00%`
- `Miss Param = 0.00%`
- `Long Context = 0.00%`

所以 `multi-turn` 的不完整只会让结论更保守，不会把“final 模型在 multi-turn 上灾难性退化”的判断翻回来。

## What Collapsed

### 1. General reasoning regressed, not improved

相对最终选中的 search-best `lr1e4-r16`：

- `C-Eval`: `79.72% -> 72.29%`，下降 `7.43pt`
- `IFEval Strict`: `35.67% -> 31.98%`，下降 `3.70pt`
- `IFEval Loose`: `42.70% -> 38.45%`，下降 `4.25pt`

这说明问题不是“只是 tool benchmark 出了岔子”，而是**基础能力层本身就已经明显回退**。

### 2. BFCL tool-calling behavior collapsed across the board

相对 `lr1e4-r16`：

- `BFCL Non-Live`: `88.73% -> 13.29%`
- `BFCL Live`: `80.24% -> 4.59%`
- `BFCL Multi-Turn`: `32.25% -> 0.00%`

这里最关键的是：

- `live`、`non_live` 都是完整生成并完成补评分的组
- `multi_turn` 虽然内部 partial，但四个子类都已启动，而且已完成部分的官方补评分仍然是 `0.00%`
- 即使把缺失 memory 和未跑完的 multi-turn 子样本造成的低估全部扣掉，**final 模型也已经在 BFCL 主体能力上崩盘**

### 3. Web Search 没有给最终模型洗白

`Web Search = 3.50%` 看起来没有比历史最差更坏，但这不构成正面证据：

- 它只覆盖 `web_search_base` 与 `web_search_no_snippet`
- 历史上 `web_search` 本来就受运行时与搜索后端稳定性影响较大
- 真正更有判别力的 `live / non_live / multi_turn` 已经同时崩坏

所以这次不能把 `web_search` 的相对稳定解读为“final 模型至少保住了一定的 agentic 能力”。

## Failure Examples

### C-Eval: 同题回退，base 和 search-best 都能做对

样本：`environmental_impact_assessment_engineer`, `doc_id=17`

- 题目：`根据《污水综合排放标准》(GB8978—1996)，为判定下列污染物是否达标，可在排污单位总排放口采样的是____。`
- 选项：
  - `A. 苯并[a]芘`
  - `B. 六价铬`
  - `C. 总铜`
  - `D. 总镉`
- 正确答案：`C`
- final bf16 预测：`A`
- base 预测：`C`
- `lr1e4-r16` search-best 预测：`C`

这类错误说明 final 模型不是只在“复杂工具任务”上失手，而是在基础选择题上就已经出现稳定回退。

来源：`analysis/stage2_final_benchmarks/summary/final_failure_analysis.json`

### IFEval: 输出不受控，擅自加入额外要求与元叙述

样本：`doc_id=0`

- 指令要求：
  - 不要使用逗号
  - 至少 3 个 markdown 高亮 section
  - 至少 300 词
- final 输出开头却变成：
  - `The rest of the text should be written normally.`
  - `Additionally, include an emoji ...`
  - `The user wants to ensure ...`

这不是简单的“少遵守了一条格式规则”，而是模型明显进入了**元解释 / 自我指挥**的坏状态：它先给自己加了额外任务，再开始写答案，导致原始 instruction set 被污染。

来源：`analysis/stage2_final_benchmarks/summary/final_failure_analysis.json`

### BFCL Example 1: 单轮并行调用出现语法级崩坏

样本：`live_parallel_0-0-0`

用户问题是同时查询北京和上海天气，final 输出为：

```text
[get_current_weather(location="Beijing, China"), get_current_weather(location="Shanghai, China"))]
```

这里多了一个右括号，直接导致：

- `error_type = ast_decoder:decoder_failed`
- `closing parenthesis ')' does not match opening parenthesis '['`

更值得注意的是，base / search-best 在同一题上虽然也不完美，但至少还是**可解析的函数调用结构**；final 则已经退化到语法级别不可执行。

来源：

- `.../bfcl_v4/project_root/score/Qwen_Qwen3-8B/live/BFCL_v4_live_parallel_score.json`
- `.../bfcl_v4/project_root/result/Qwen_Qwen3-8B/live/BFCL_v4_live_parallel_result.json`

### BFCL Example 2: 多轮任务几乎失去可执行工具行为

样本：`multi_turn_base_0`

历史 base / search-best 在这题上至少会输出成串的工具调用，例如：

- `cd(...)`
- `mkdir(...)`
- `mv(...)`
- `grep(...)`
- `diff(...)`

而 final 模型的首轮输出已经变成了近乎乱码的重复文本：

```text
"document" "file"file"file"file ...
```

对应评分错误是：

- `error_type = multi_turn:empty_turn_model_response`

也就是说，这不是“多轮策略稍弱”，而是**连第一轮可执行 action 序列都不能稳定给出**，因此 `multi_turn` 全组直接掉到 `0.00%`。

来源：

- `.../bfcl_v4/project_root/score/Qwen_Qwen3-8B/multi_turn/BFCL_v4_multi_turn_base_score.json`
- `.../bfcl_v4/project_root/result/Qwen_Qwen3-8B/multi_turn/BFCL_v4_multi_turn_base_result.json`

### BFCL Example 3: 长上下文子类直接撞上 32k 上限

样本：`multi_turn_miss_func_97`

快照里直接记录了服务端报错：

```text
This model's maximum context length is 32768 tokens.
However, you requested 4096 output tokens and your prompt contains at least 28673 input tokens,
for a total of at least 32769 tokens.
```

这说明 final BFCL 的 `multi-turn` 失败里至少有一部分不是“纯策略错误”，而是**真实触发了 32k 上下文边界**。  
这和当前训练配置形成了明显错位：

- 训练 `max_seq_length = 8192`
- 训练集平均序列长度约 `1948`
- BFCL multi-turn 服务端却允许滚到 `32768`

所以，`multi-turn` 的灾难性表现至少有一部分是被**长上下文分布错位**进一步放大的。

### BFCL Example 4: 这不是 final 独有问题，base / search-best 也会撞上 32k

这一点需要特别说明，避免过度归因。

BFCL V4，尤其 `multi_turn_long_context`，对只有 `32k` 上下文窗口的 8B 小模型本来就是一个很强的评测集。  
这次不是只有 final 才触发上下文上限；历史 `base` 和 `search-best` 的结果文件里也能直接看到同类报错，例如：

- `stage2_best_base_model/.../BFCL_v4_multi_turn_long_context_result.json`
- `stage2_best_lr2e4_r16/.../BFCL_v4_multi_turn_long_context_result.json`
- `stage2_best_lr1e4_r16/.../BFCL_v4_multi_turn_long_context_result.json`

里面同样记录了：

```text
This model's maximum context length is 32768 tokens ...
```

所以更准确的判断应该是：

- **BFCL V4 multi-turn 对 32k 8B 模型本来就难**
- **长上下文撞限不是 final 独有现象**
- final 的问题在于：它在这个共同困难之上，又额外出现了更严重的语法退化、空响应和基础能力回退

也就是说，32k 上限问题解释了“为什么 multi-turn 普遍难”，但解释不了“为什么 final 比 base/search-best 差这么多”。

## Likely Root Cause

### 1. 不是 BFCL 不完整导致“看起来很差”

这次 BFCL 确实不完整，缺少 memory 三项，而且 multi-turn 内部也不是满覆盖。  
但这不能解释当前结果，因为：

- `CEval` 已经先掉了 `7.5pt`
- `IFEval` 也同步下降
- BFCL 的 `live / non_live` 两个完整覆盖组已经同时崩坏
- multi-turn 已完成部分的官方补评分仍然是 `0.00%`

所以“BFCL partial”只会让 `overall` 更保守，不会改变最终模型失败的结论。

### 2. 数据集本身不像主因，但“长时间优化这套目标”很像主因

先说不太像的部分：

- 正式训练和 search-best 使用的是**同一套冻结配方思路**
- short run 已经证明这套配方能带来小幅正向收益
- 正式训练的 train/val 预处理没有出现大规模异常：
  - `58800 / 58800` 训练样本全部保留
  - `1200 / 1200` 验证样本全部保留
  - sampled check 里 assistant 终止、chat template 渲染、监督 token 非空都正常

所以更合理的表述不是“数据烂”，而是：

- **这套数据目标很窄**
- **在 500 steps 内，它能帮模型做轻度对齐**
- **在 1 epoch 内，它开始把模型过度拉向这种窄分布**

这一点和当前 final mixture 的组成是吻合的：

- `Hermes 42000`
- `Step tool-call 12000`
- `Step general 6000`

这不是随便拼的坏数据，但它的 supervision surface 高度集中在：

- tool-using assistant 响应
- `<think> ... </think>` 前缀化 assistant 输出
- 比 base model 原始分布更重的 agentic / reasoning / 格式化行为

短训时这会像“提醒模型按项目需要说话”；长训时它更像“持续覆盖 base 的原始行为先验”。

### 3. 混合策略本身未必错，但 heterogeneous tool format 在长训下可能放大格式漂移

项目文档里原本就把“混合不同来源数据集”作为推荐路线，这一点本身没有问题。  
真正值得警惕的是：**混合在短训下有利，不代表在长训下仍然安全**。

当前 final mixture 虽然只包含三部分，但其 supervision surface 并不完全同构：

- Hermes：大量 reasoning/tool-use 对话，工具定义常内嵌在 system prompt
- Step tool-call：更偏 agentic 轨迹，消息里可能带 `tool_calls` / `tool_call_id`
- Step general：不要求严谨工具格式，但仍按 Qwen3 thinking 模式统一成 `<think>` 前缀

这类混合在早期可能是优势，因为它增强了覆盖面；但在长训下，也可能把模型往几种互相不完全一致的“工具表达表面形式”上同时拉扯，最终表现为：

- tool call 语法不稳定
- 非必要的 verbose / meta narration 增多
- 多轮里更容易掉到空响应或乱码式退化

所以问题不一定是“不能混”，更像是：**当前这套 mix 在 full-length 训练下已经超过了 base model 能安全吸收的剂量。**

### 4. `think mask` / 模板流程不像灾难主因；之前那个 synthetic probe 已确认是检查器 bug

从当前证据看，训练前处理链路总体是自洽的：

- assistant 回复统一补 `<think>` 前缀
- `<think>` 标签本身保留监督
- `<think>` 内部内容做 loss mask
- `completion_only_loss=True`

这条策略和项目设计是一致的，而且 short run 没有出问题，本身就是对“模板/掩码灾难性错误”的反证。

之前我在仓库里看到的那条“`synthetic_think_mask` 不完美”证据，具体来自：

- 文件：[compat_training_readiness_summary.json](/home/fan/workspace/post-Train-proj/analysis/quality_checks/compat_training_readiness_summary.json)
- 生成脚本：[recheck_training_readiness.py](/home/fan/workspace/post-Train-proj/scripts/recheck_training_readiness.py)
- 样例：脚本内 `assert_synthetic_think_mask(...)` 手工构造的最小 case
  - user: `Please help.`
  - assistant: `<think>\nsecret planning\n</think>\n\nVisible answer.`

我这次重新按当前代码展开 token/mask 后确认，原来的问题不在训练逻辑本身，而在 probe 的定位方法：

- 它原来用“逐步截断 assistant 前缀再重新走 chat template”的 token 长度差来估算 `<think>` 内容区间
- 在 Qwen3 模板下，这会把边界算偏
- 例如它会得到像 `after_open = 18`、`before_close = 22` 这样的区间，但这个切片实际上已经落到答案尾部附近，不再对应真实的 `secret planning` span

也就是说，旧的 `synthetic_think_mask=false` 是**检查器本身的假阴性**，不是训练 mask 逻辑的直接证据。

这个 probe 我已经顺手修掉了：

- 现在它和 `final_mask_format_check.py` 一样，改成在完整渲染文本上用 offset span 直接定位 `<think>` 内容 token
- 修复后重新生成的 [compat_training_readiness_summary.json](/home/fan/workspace/post-Train-proj/analysis/quality_checks/compat_training_readiness_summary.json) 已经变成：
  - `think_content_all_masked = true`
  - `think_tag_supervised = true`
  - `answer_supervised = true`

相比之下，当前更可信的证据是：

- 文件：[final_mask_format_check.json](/home/fan/workspace/post-Train-proj/analysis/quality_checks/final_mask_format_check.json)
- 脚本：[final_mask_format_check.py](/home/fan/workspace/post-Train-proj/scripts/final_mask_format_check.py)

它是直接在真实数据抽样上，用 offset span 去检查 `<think>` 内容 token 是否被 mask。该检查在各个数据集样本上都基本为全绿。

所以这里我把结论修正为：

- 现阶段没有足够证据表明 `think mask` 实现本身是这次失败的主因
- 那条旧的 `synthetic_think_mask=false` 已被确认是 probe 脚本的定位 bug，并已修复
- 模板 / mask 仍可继续做代码级清理，但不应作为当前失败复盘的核心解释

### 5. 很不像是 merge、基座错配或合并到了 `8B-Base`

这条怀疑当前更倾向于排除：

- 正式训练配置写死的模型是 `Qwen/Qwen3-8B`
- 训练摘要里记录的模型也是 `Qwen/Qwen3-8B`
- exporter 的 `detect_base_model_id(...)` 会优先接受合法 HF repo id；如果 adapter 里只是失效的本地路径，就回退到 `train_config_resolved.json` 里的 `model_name`

也就是说：

- adapter 目录里虽然残留了一个旧的本地绝对路径
- 但当前合并实现不会因此切到别的基座，更没有证据切到 `Qwen3-8B-Base`
- 最终 benchmark 用的也是远端 merged full-weight repo，不是评测时再把 LoRA 动态挂到一个不确定的 base 上

所以“权重合并到了不适配的基座模型上”目前没有证据支持。

### 6. 评测环境和推理超参不是完全一模一样，但差异不足以解释这次崩盘

final snapshot 和 search-best 的 benchmark 口径大体一致：

- `dtype = bfloat16`
- `max_model_len = 32768`
- `max_num_seqs = 256`
- `BFCL max_output_tokens = 4096`
- `IFEval max_gen_toks = 4096`
- `repetition_penalty = 1.1`

确实存在一个已知差异：

- search-best 的并发是 `256`
- final snapshot 的并发是 `128`

但这个差异主要影响的是：

- GPU 利用率
- 吞吐和总耗时

它不太可能把：

- `C-Eval -7.43pt`
- `IFEval Loose -4.25pt`
- `BFCL Live 80.24% -> 4.59%`
- `BFCL Multi-Turn 32.25% -> 0.00%`

这种量级的退化“凭空制造出来”。  
如果评测环境是主因，我们更可能看到的是局部波动，而不是三类 benchmark 同时系统性退化。

### 7. 真正更可疑的，是 full-length 训练目标和 checkpoint 选择都在持续奖励“更像训练分布”

这次正式训练的一个关键信号是：

- 训练内 `eval_loss` 从 `0.4208@250` 一路单调下降到 `0.3367@3500`
- best checkpoint 就按这个同分布 `eval_loss` 选在了 `step=3500`

这意味着训练器做的其实是：

- 在同一冻结 mixture 上持续优化
- 并在“最会拟合这套 mixture”的点停下来

这对项目的外部目标并不一定等价。  
short run 的 `500-step best` 很可能恰好处在：

- 已经吸收了项目需要的部分 tool / instruction prior
- 但还没有覆盖 base model 原始能力

而 full run 的 `step=3500` 则更像是：

- 对训练分布适应得更彻底
- 对外部 benchmark 所代表的“广义能力面”反而更差

### 8. 长上下文分布错位是 BFCL multi-turn 灾难的重要放大器，但不是 final 独有

这个因素不能解释 `CEval/IFEval` 的下降，但能很好解释为什么 `multi-turn` 特别惨：

- 训练时 `max_seq_length = 8192`
- 训练集平均序列长度只有 `1948`
- 训练完全没有覆盖接近 `32k` 的上下文滚动场景
- 但 final BFCL server 端是按 `max_model_len = 32768` 在跑 multi-turn

快照里的直接证据是：

- `multi_turn_miss_func_97`
- `multi_turn_miss_func_109`
- `multi_turn_miss_func_131`

都明确报了 `maximum context length is 32768` 的错误。

但这里要再加一个重要限定：

- base 也会撞 `32768`
- search-best 也会撞 `32768`

所以一个更完整的解释应该是：

- **广义退化**：来自 full-length 训练对窄分布的过拟合
- **multi-turn 普遍难**：因为 BFCL V4 对 32k 小模型本来就硬
- **final 的 multi-turn 特别惨**：是在这个共同困难之上，又额外叠加了 long-context 分布错位和行为退化

### 9. 其他可能的放大因素

还有两个因素值得保留为次要怀疑项：

- LoRA 不只挂在 `q/k/v/o`，还挂在 `gate/up/down_proj`
  - 这让它有足够 capacity 去改写风格和行为，不只是轻微调 attention
- 正式训练总共走了约 `3675` optimizer steps
  - 对当前这种高格式化、窄目标分布来说，这个步数已经足够让 `r=16` 的 LoRA 把行为明显拉偏

### 10. 也不像是 merge 或 repetition_penalty 造成的假退化

如果只是 merge、服务端参数或评测噪声问题，通常更像：

- 某一两个 benchmark 波动
- tool 类任务掉分，但 `CEval/IFEval` 仍接近原水平

这次不是这样。现在看到的是：

- 选择题能力掉
- instruction following 掉
- tool syntax 掉
- multi-turn stateful behavior 直接清零

这更像**模型内部行为模式整体被拉坏**，而不是单一评测配置问题。

### 11. 最可能的解释：full-length Stage 2 训练过头

目前最符合证据的解释是：

- `500-step` search-best checkpoint 仍处于“局部适配但没明显伤筋动骨”的区间
- 正式 full-length Stage 2 训练把模型进一步推向 Stage 2 语料分布
- 结果是模型学到了更强的某些局部模式，但损失了 Qwen3-8B 原本更稳的：
  - 基础知识判断
  - 指令约束遵循
  - 工具调用格式稳定性
  - 多轮任务状态保持

换句话说，这更像**长训造成的行为漂移与能力覆盖面收缩**。

## Final Judgment

1. 这轮正式 Stage 2 最终模型不应继续作为候选发布版本。
2. 没必要继续完成：
   - 最终 BF16 剩余 memory 三项
   - 最终 INT4 全量评测
3. 从当前证据看，真正有效的终点更可能接近 search 阶段的 `500-step best checkpoint`，而不是完整 1 epoch 的正式训练终点。

## Artifacts

- plan:
  - `analysis/stage2_final_benchmarks/summary/final_failure_analysis_plan.md`
- structured summary:
  - `analysis/stage2_final_benchmarks/summary/final_failure_analysis.json`
- recovered BFCL scores:
  - `analysis/stage2_final_benchmarks/stage2train_20260408_095122_stage2_qwen3_8b_lora_best/benchmarks/stage2_final_merged_bf16_model/bfcl_v4/project_root/score/data_overall.csv`
  - `analysis/stage2_final_benchmarks/stage2train_20260408_095122_stage2_qwen3_8b_lora_best/benchmarks/stage2_final_merged_bf16_model/bfcl_v4/project_root/score/data_live.csv`
  - `analysis/stage2_final_benchmarks/stage2train_20260408_095122_stage2_qwen3_8b_lora_best/benchmarks/stage2_final_merged_bf16_model/bfcl_v4/project_root/score/data_non_live.csv`
  - `analysis/stage2_final_benchmarks/stage2train_20260408_095122_stage2_qwen3_8b_lora_best/benchmarks/stage2_final_merged_bf16_model/bfcl_v4/project_root/score/data_multi_turn.csv`
- snapshot status note:
  - `analysis/stage2_final_benchmarks/stage2train_20260408_095122_stage2_qwen3_8b_lora_best/summary/snapshot_status.md`
