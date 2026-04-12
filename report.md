# Qwen3 Tool Calling 项目报告

## 1. 项目背景

本项目围绕 `Qwen/Qwen3-8B` 做后训练，目标不是单纯把模型“训得更像工具调用模型”，而是尽量同时保住三类能力：

1. 基础知识与推理能力
2. 指令遵循与格式约束
3. 多轮、带工具、带状态的 agentic 行为

因此，整个项目的核心问题其实是一个平衡问题：**如何在增强 tool calling 的同时，不把原生模型的通用能力拉坏**。  
本仓库的评测主线也始终围绕这三类目标展开，主要用 `C-Eval`、`IFEval`、`BFCL v4` 来判断模型是否真的在“可用的 agent 能力”上变强，而不是只在训练损失上变好。

从当前结论看，项目最终已经收敛到两个重要节点：

- 最终可采用的短训胜者是 `stage2_best_lr1e4_r16`
- full-length 正式训练 `stage2train_20260408_095122_stage2_qwen3_8b_lora` 则应视为失败案例

这意味着本项目不是一个“持续堆更长训练就更好”的故事，而是一个**通过数据调研、数据消融和短训搜索，找到有效干预点后，又用 full-length 训练验证其边界**的故事。

## 2. 数据集调研与选择理由

在真正做 stage1 消融之前，仓库已经先做了数据源层面的调研。这里的目标不是找“最大数据集”，而是找**最适合 Qwen3-8B 做 tool calling 后训练的监督信号组合**。

### 2.1 备选数据源及预期作用

| 数据源 | 预期角色 | 选择理由 | 风险 / 代价 |
| --- | --- | --- | --- |
| Hermes Reasoning Tool Use | 主干 multi-turn tool calling | 与 BFCL / Atropos RL 风格更接近，结构稳，质量高 | 覆盖面不算最大 |
| xLAM Function Calling 60K | 简单单轮函数调用补充 | 格式整齐，易优化，适合补“短、稳、清楚”的调用 | 容易把训练重心拉向简单分布 |
| Qwen3.5 ToolCalling v2 | 对照线 | 介绍是针对 qwen 系列模式，格式已经调整好 | 社区拼接，未严格清洗 |
| Glaive Function Calling v2 | 历史对照线 | 经典 baseline，便于比较 | 风格偏旧，转换成本高 |
| Step-3.5-Flash-SFT general | 通用 rehearsal | 补通用表达，防止只向工具调用分布过拟合 | 序列更长，成本更高 |
| Step-3.5-Flash-SFT tool-call | 长轨迹 / agentic 补充 | 覆盖长上下文、多轮和工具密集场景 | 噪声更高，不能放太大比例 |

这里的 `step` 指的是 **StepFun 开源的用于真实商业级大模型 Step-3.5-Flash-SFT 后训练的 SFT 数据集 `Step-3.5-Flash-SFT`**。  
它在本项目里不是“普通的通用文本数据”，而是专门承担以下两类补位：

- `step tool-call`：抽取其中 tool-call 类的数据，补长轨迹、agentic、多轮工具调用
- `step general`：抽取非 tool-call 的数据，补通用 rehearsal，避免模型只朝工具调用分布偏移

### 2.2 调研阶段的预期

在 stage1 之前，比较合理的预期是：

- Hermes 作为主干，应该是最稳的通用 tool calling 来源
- xLAM / Glaive 这类格式整齐、短样本多的数据，可能让训练 loss 更好看，但未必真正提升外部 benchmark
- Step tool-call 可能对长轨迹和 agent 场景更有帮助，但会明显增加序列长度和训练成本
- Step general 主要是抗遗忘和分布平衡，不应该成为主混合的大头

这些判断后来基本被 stage1 的结果部分证实，也部分推翻：

- 证实：Hermes 仍然是最稳的主干
- 证实：xLAM/Glaive 的训练损失更容易优化，但不等于外部 benchmark 更强
- 证实：Step 数据确实带来更长序列和更高成本
- 推翻：Step 数据并不天然比 Hermes 更“直接有效”，至少不能单独放大成主干

## 3. 方法

### 3.1 数据来源与处理

项目中的数据统一处理成 `messages + tools` 结构，并尽量对齐 Qwen3 的 chat template。

主要规则如下：

- Hermes / Step / Qwen3.5 系列的工具定义默认放在 system prompt 中，`tools = null`
- xLAM 保留顶层 `tools`
- assistant 回复里的 reasoning 内容统一包成 `<think> ... </think>`
- 删除 `<think>` 开闭不平衡样本
- 删除不是以 assistant 收尾的样本
- 对所有 assistant 回复补齐空的 `<think>` 前缀
- 训练时保留 `<think>` 标签本身的监督，只 mask `think` 内部内容

这一套处理逻辑的关键点是：**模型需要学会“thinking mode”的外壳和输出格式，但不必机械模仿异源数据的具体推理文本**。  
这样做的目的，是尽量保留 Qwen3 的思维模式触发语义，同时避免把异源 reasoning 内容当成必须复述的目标。

### 3.2 Step 数据的抽取方式

`Step-3.5-Flash-SFT` 在本项目里被拆成两个不同角色：

- `step tool-call`
  - 从包含 schema / `tool_calls` / `tool_call_id` / tool role 的轨迹里提取
  - 用于增强 agentic 和多轮工具调用
- `step general`
  - 仅保留没有 tool role、没有 tool_calls、没有 tool_call_id、没有 tools 字段的对话
  - 再按 Qwen3 token 长度过滤后抽样

其中 `step general` 的构造不是简单“随便抽样”，而是先固定历史 shard baseline，再做长度过滤和全局采样。  
这件事很重要，因为它意味着这不是一个任意扩容的数据集，而是一个**为了控制成本和分布偏移而设计的受控采样**。

### 3.3 训练配置

stage1 与 stage2 search 都采用 LoRA 微调，核心配置是一致的：

- `Qwen/Qwen3-8B`
- `bf16`
- `LoRA r=16`
- `target_modules = q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- `packing = false`
- `max_seq_length = 8192`
- `optim = adamw_8bit`
- `gradient_accumulation_steps = 16`

分阶段设计如下：

| 阶段 | 数据 | 训练设定 | 目的 |
| --- | --- | --- | --- |
| stage1 | 5k / 1 epoch | `lr=2e-5`, `r=16`, 重点看训练和外部 benchmark | 数据消融 |
| stage2 search | 10k frozen mix / 500 steps | 搜索 `lr` 和 `r` 组合 | 找到短训最优点 |
| stage2 final | 60k frozen mix / 1 epoch | `lr=1e-4`, `r=16`，正式 full-length | 验证是否能稳定放大 |

stage1 的意义不是“找绝对最强模型”，而是先判断**什么样的数据配方方向是对的**。  
stage2 search 则是在固定数据配方后，找一个最适合后续收敛的短训超参。  
full-length 正式训练本来是要验证“短训信号能否继续扩大”，但最终事实证明：**不能简单线性外推**。

## 4. 实验与结果

### 4.1 Stage1：数据消融与预期对照

stage1 的关键不是单个分数，而是“训练难易度”和“外部 benchmark”是否一致。

这里需要特别说明，stage1 的各评测框架都只做了 50 条左右的快速抽样评测，`C-Eval` 则是按每个子类最多抽 50 条来跑。  
因此，这里的分数更适合看**相对趋势**，不能作为唯一判断指标；同一配置在 rerun 前后也会出现一定波动，尤其是样本较少时更明显。

下面挑几组最有代表性的结果：

| 组别 | 配方 | 平均序列长度 | 训练步速 | eval_loss | C-Eval | IFEval Strict | IFEval Loose | BFCL Overall | BFCL Live | BFCL Multi-Turn |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline prompt | Qwen3-8B (Prompt) | - | - | - | 79.79% | 34.00% | 42.00% | 42.58% | 80.00% | 50.00% |
| A | Hermes only | 1351.46 | 0.304 | 0.4943 | 79.79% | 34.00% | 40.00% | 43.17% | 90.00% | 50.00% |
| E | xLAM only | 465.68 | 0.587 | 0.0304 | 79.64% | 36.00% | 44.00% | 38.83% | 80.00% | 37.50% |
| H | Step tool-call only | 3740.20 | 0.117 | 0.4169 | 79.49% | 36.00% | 44.00% | 35.08% | 80.00% | 25.00% |
| I | Step tool-call + Step general (4:1) | 3489.42 | 0.125 | 0.4891 | 79.64% | 36.00% | 44.00% | 38.83% | 80.00% | 37.50% |

#### 4.1.1 分项解读

从这组抽样结果看，stage1 更像是在回答“哪类数据适合做主干”，而不是“哪组数据能把所有指标一起拉满”。

- `C-Eval` 基本都落在 `79.49%~79.79%` 之间，变化幅度很小，说明 stage1 的短训主要在修正 tool-calling 行为和输出习惯，并没有明显破坏原生知识面。这一点很重要，因为它表明我们在做能力注入时，没有把模型推离 `Qwen3-8B` 原来的知识先验。
- `IFEval` 的变化比 `C-Eval` 更敏感。`xLAM` 和 `Step` 配方的 `Strict / Loose` 都略高于 baseline，说明这些数据确实在帮助模型学习更强的格式控制、约束响应和输出边界；但这种提升没有同步转化为更好的 `BFCL`，所以它更像是“更会遵守指令”，而不是“更会完成 agent 任务”。
- `BFCL` 最能区分不同数据源的实际价值。`Hermes only` 的 `BFCL Overall` 略高于 baseline，同时 `BFCL Live` 明显更高，说明它是比较稳的主干信号；`xLAM only` 虽然 `eval_loss` 最低，但 `BFCL Overall` 和 `Multi-Turn` 都回落，说明训练损失在这里更像是在奖励“短、整齐、容易拟合”的样本分布，而不是完整的工具能力；`Step tool-call only` 则把序列长度和训练成本推得最高，但 `Multi-Turn` 反而明显下降，说明长轨迹数据更适合作为补充能力来源，而不是单独放大就能赢。

这组结果最值得写进报告的点有三个：

1. **xLAM-only 最容易把 loss 做漂亮**，但这种“漂亮”主要来自样本结构更短、更规整，并没有自动换成更强的 `C-Eval`、`IFEval` 或 `BFCL`。
2. **Hermes-only 的 loss 并不最低，但外部表现最稳**。它没有把某一项指标推到极端，却更像一个能守住整体下限的主干数据源。
3. **Step-heavy 配方显著拉长序列、拖慢训练，但没有自动换来更好的外部能力**。`Step tool-call` 和 `Step general` 确实给模型补进了更长轨迹和更多 agentic 形态，但在这个阶段，它们还更像补位材料，不是主干替代品。

这和一开始的预期形成了很清楚的对照：

- “短、整齐、好优化”的数据不一定最适合做主干
- “看起来更 agentic”的数据不一定一加就涨分
- Hermes 更像一个稳定的能力锚点，Step 更适合作补充，而不是替代

换句话说，stage1 的真正收获不是某个配方“赢了”，而是：

- Hermes 是主线
- Step 小比例补充更合理
- xLAM / Glaive 更适合做对照，不适合单独放大

### 4.2 Stage2 search：短训超参搜索

stage2 search 固定了 10k 的 frozen mix，然后做 500-step 短训搜索。  
这一步的目的不是长跑，而是看“哪组 LoRA / 学习率更容易把 tool-calling 和 instruction following 拉起来，同时不明显破坏 base model”。

在训练损失层面，6 组短训结果里，`lr=5e-5, r=16` 的 `eval_loss` 最低，说明短训的确能被 loss 指标分出优劣。  
但后续完整 benchmark 证明，**最小的 eval_loss 并不等于最好的最终模型**。

最终真正入选的是 `stage2_best_lr1e4_r16`。  
它相对 `lr2e4-r16` 的优势不只是一个总分，而是更接近项目目标的“实用 agent 能力”：

- `BFCL Overall` 更高
- `BFCL Live` 更高
- `BFCL Web Search` 更高
- `IFEval Loose` 更高
- `C-Eval` 基本持平
- 平均 latency 更低

而 `lr2e4-r16` 只在 `BFCL Multi-Turn` 上略占优势，但不足以抵消其他维度的损失。  
所以这里最终采用的是**更广泛、面向真实工具调用场景的综合收益**，而不是单点 loss。

### 4.3 最终对比：base、search-best、full-length

最终需要对比的其实是四个对象：

| run | C-Eval | IFEval Strict | IFEval Loose | BFCL Overall | Non-Live | Live | Multi-Turn | Web Search | 说明 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline prompt | 79.79% | 34.00% | 40.00% | 42.58% | 95.83% | 80.00% | 50.00% | 0.00% | 原生 prompt 基线 |
| `lr2e4-r16` | 79.57% | 35.67% | 41.59% | 35.21% | 88.35% | 79.50% | 33.38% | 0.50% | backup candidate |
| `lr1e4-r16` | 79.72% | 35.67% | 42.70% | 35.67% | 88.73% | 80.24% | 32.25% | 4.00% | 最终 short-run winner |
| full-length bf16 partial | 72.29% | 31.98% | 38.45% | 11.76%* | 13.29%* | 4.59%* | 0.00%* | 3.50%* | partial, 19/22 类别 |

注：

- `Non-Live` 可以视为单步函数调用准确率的核心信号
- `Multi-Turn` 可以视为多步任务完成率的核心信号
- `Live`、`Web Search`、`Memory` 更像 agent 场景的辅助诊断项
- full-length BFCL 只是 partial recovery，缺失 `memory/kv`、`memory/vector`、`memory/rec_sum`

从这个表里最清楚的一件事是：

- short-run 的 `lr1e4-r16` 仍然比 base 更像“对齐后的可用工具模型”
- 但 full-length 训练并没有把这种优势继续放大
- 相反，它让基础能力和 tool-calling 都明显回退

### 4.4 失败样本

这次 full-length 失败并不是“只在某一个 benchmark 上翻车”，而是多条证据同时出现。

#### C-Eval

在 `environmental_impact_assessment_engineer` 的一道题上，final 模型把正确选项错选成了另一项，而 base 和 short-run winner 都答对了。  
这说明 final 的问题不是只发生在工具链上，而是已经回到**基础选择题层面的稳定回退**。

#### IFEval

在一个要求“不能用逗号、至少 3 个 markdown section、至少 300 词”的任务里，final 输出却主动加了额外的自我指挥内容和表情符号，属于明显的指令污染和元叙述失控。  
这不是单纯漏掉一条格式规则，而是模型把原始约束打散后又自我加戏。

#### BFCL

在一个并行天气查询任务里，final 输出了带语法错误的函数调用表达式，括号不配对，直接导致 AST 解码失败。  
这说明问题已经不是“调用得不够好”，而是**连可执行函数调用的语法都不稳定了**。

#### BFCL multi-turn

在多轮文件操作任务里，final 出现了空响应或重复碎片化文本，无法稳定给出第一轮可执行 action 序列。  
这就是为什么 `Multi-Turn` 最终掉到 `0.00%`。

## 5. 分析与讨论

### 5.1 为什么 short-run 成功了

`500-step` 的 `lr1e4-r16` 不是“完全没问题”，而是一个很典型的轻度 steering 成功信号：

- `C-Eval` 基本持平
- `IFEval Loose` 上升
- `BFCL Overall / Live` 小幅上升
- 说明数据、模板、mask、评测链路本身不是一开始就错

换句话说，短训阶段的正向结果说明：**项目的干预方向是对的，只是强度和训练时长不能无脑放大**。

### 5.2 为什么 full-length 失败

最可信的解释是：full-length 正式训练把模型过度拉向了窄的 stage2 分布，导致行为漂移。

支持这个判断的证据有几个：

- full-length 的 `eval_loss` 继续下降，但外部 benchmark 同步恶化
- `C-Eval`、`IFEval`、`BFCL live / non-live / multi-turn` 都一起掉
- `BFCL` 的崩坏不是单点，而是语法、格式、状态保持一起出问题

这说明 final 问题不是简单的 merge 误差、不是基座错配，也不是单一 benchmark 噪声，而更像：

> 模型已经在 full-length 训练中学会了更强地拟合训练分布，但失去了原始 Qwen3-8B 更稳的通用先验。

### 5.3 哪些坑不能当主因

#### `web_search` / `memory` 结果要谨慎解释

BFCL 的 `web_search` 和 `memory` 子项在本项目里有明显运行态污染：

- `web_search` 经常受搜索后端报错影响
- `memory` 子项有 fixture / prompt state 注入异常

所以它们更适合作为诊断信号，而不适合作为唯一选型依据。

#### 32k 长上下文确实难，但不是唯一原因

BFCL V4 的 multi-turn 对 32k 8B 模型本来就很硬，而本项目训练时的 `max_seq_length = 8192` 也没有覆盖到同等长度。  
因此长上下文错位会放大 multi-turn 问题，但它解释不了：

- `C-Eval` 为什么会掉
- `IFEval` 为什么会掉
- `live / non-live` 为什么会一起崩

所以它是放大器，不是唯一根因。

#### `<think>` 掩码不是这次失败的主因

仓库里最初对 `<think>` 掩码有过怀疑，但后来已经确认那条异常来自检查器定位方式的 bug，而不是训练逻辑本身。  
也就是说，`think` mask 可以继续做代码清理，但不能拿来解释这次 full-length 崩盘。

## 6. 总结与展望

本项目的核心结论已经很明确：

1. `Hermes` 是最稳的主干
2. `Step-3.5-Flash-SFT` 适合做关键补充，但不能无上限放大
3. `xLAM / Glaive / Qwen3.5` 更适合做对照，而不是最终主线
4. `stage2_best_lr1e4_r16` 是当前最合适的短训胜者
5. full-length `stage2train_20260408_095122_stage2_qwen3_8b_lora` 应视为失败案例

如果后续继续迭代，最值得做的方向不是继续盲目延长训练，而是：

- 重新审视 full-length 的训练剂量和停止点
- 更细地控制 Step 数据比例，避免行为漂移
- 在真正可信的 benchmark 运行态下再看 `web_search` / `memory`
- 如果目标偏向 multi-turn agent，再补更长上下文和更严格的状态保持训练

总体而言，这次项目已经证明：**Qwen3 Tool Calling 的有效提升，来自精心设计的数据组合和适度的短训 steering，而不是简单把训练拉长到最后一刻。**
