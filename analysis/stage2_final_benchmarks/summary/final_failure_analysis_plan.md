# Stage 2 最终评测快照整理、BFCL 补评分与失败复盘执行计划

## Summary
先把 `stage2_eval_snapshot (1).zip` 解压并整理到 `analysis/stage2_final_benchmarks/stage2train_20260408_095122_stage2_qwen3_8b_lora_best`，去掉多余的 `content/stage2_eval_snapshot/` 父级目录。  
然后在本地 `post-train-benchmark` 环境里，对快照中的 BFCL 已生成结果执行官方 `partial-eval` 补评分；若官方流程失败，再按 BFCL 现有输出口径做最小兼容汇总。  
最后生成最终分析报告，系统对比 `base / lr2e4-r16 / lr1e4-r16 / final bf16`，明确说明 final BFCL 仅完成 22 个子类中的 19 个，但这一部分已经足够证明最终训练显著退化。

## Execution Order
1. 先将本计划原样落盘到：
   - `analysis/stage2_final_benchmarks/summary/final_failure_analysis_plan.md`
2. 解压 `stage2_eval_snapshot (1).zip` 到 `analysis/`。
3. 清理无用嵌套目录，只保留最终有效根：
   - `analysis/stage2_final_benchmarks/stage2train_20260408_095122_stage2_qwen3_8b_lora_best`
4. 核验快照完整性：
   - `ceval` 结果文件存在
   - `ifeval` 结果文件存在
   - `bfcl_v4/project_root/result/Qwen_Qwen3-8B/**/_result.json` 存在
5. 在 `post-train-benchmark` 环境里进入快照对应的 `bfcl_v4/project_root`，设置 `BFCL_PROJECT_ROOT` 后执行官方 partial-eval。
6. 若官方 BFCL 评分成功，直接使用生成的 `score/data_*.csv` 与 `score/**/_score.json`。
7. 若官方 BFCL 评分失败，新增一个最小 fallback 汇总脚本，只对已完成的 19 个子类生成 partial score 表，不伪造缺失的 memory 三项。
8. 基于历史 `stage2_best_benchmarks` 汇总和 final snapshot 结果，生成最终报告：
   - `analysis/stage2_final_benchmarks/summary/final_failure_analysis.md`
9. 如有需要，再生成一个机器可读的对比 JSON：
   - `analysis/stage2_final_benchmarks/summary/final_failure_analysis.json`

## Key Changes
- 快照整理
  - 解压后移除 `content/stage2_eval_snapshot/` 外壳。
  - 若目标目录已有占位内容，以快照中的真实 benchmark 产物为准。
  - 在快照目录下补一份简短上下文说明，写明这是“CU 耗尽前保存的 bf16 final partial snapshot”。

- BFCL 补评分
  - 优先复用 BFCL 官方能力：
    - `bfcl evaluate --model Qwen/Qwen3-8B --partial-eval`
  - 运行目录与环境变量都指向快照中的 `project_root`。
  - 接受 BFCL 官方 partial-eval 语义：
    - 只对已生成 ID 计分
    - 未评估类别记为 `N/A`
    - 汇总列可能将缺失类别按 0 处理
  - fallback 仅在官方评估失败时启用：
    - 读取快照里的 19 个 `_result.json`
    - 对齐 BFCL 现有 `score/data_live.csv`、`data_non_live.csv`、`data_multi_turn.csv`、`data_overall.csv` 的字段口径
    - 输出文件名中显式标注 `partial`

- 最终分析报告
  - 对比对象固定为：
    - `stage2_best_base_model`
    - `stage2_best_lr2e4_r16`
    - `stage2_best_lr1e4_r16`
    - `stage2_final_merged_bf16_model_partial`
  - 定量部分必须覆盖：
    - `C-Eval`
    - `IFEval Strict / Loose`
    - `BFCL Overall / Non-Live / Live / Multi-Turn`
    - 明确标注 final BFCL 是 partial，且仅完成 19/22 子类，缺 `memory/kv`、`memory/vector`、`memory/rec_sum`
  - 失败原因分析必须分三层：
    - `CEval` 退化：知识/推理准确率整体下滑
    - `IFEval` 退化：格式约束、长度约束、指令遵循下降
    - `BFCL` 退化：工具调用行为、多轮保持、搜索/agentic 质量下降
  - 抽样举例：
    - 至少 1 个 `CEval` 错题样本
    - 至少 1 个 `IFEval` 指令不遵循样本
    - 至少 2 个 `BFCL` 失败样本
  - 结论必须明确：
    - 最终 full-length Stage 2 训练相对 500-step search best 出现了广泛退化
    - 这更像长训把模型行为过度拉向 Stage 2 分布，而不是单纯的 merge、量化或 benchmark 配置噪声
    - BFCL 即便不完整，也只是进一步强化这个结论，不是唯一依据

## Known Facts To Preserve
- final snapshot 已有：
  - `CEval` 结果：`72.29%`
  - `IFEval Strict`：`31.98%`
  - `IFEval Loose`：`38.45%`
- 历史最优 search 结果：
  - `base`: `C-Eval 79.79% / IFEval Loose 40.48% / BFCL Overall 34.95%`
  - `lr2e4-r16`: `C-Eval 79.57% / IFEval Loose 41.59% / BFCL Overall 35.21%`
  - `lr1e4-r16`: `C-Eval 79.72% / IFEval Loose 42.70% / BFCL Overall 35.67%`
- final BFCL 快照现状：
  - 完整 benchmark 应有 22 个 BFCL 子类结果
  - 当前快照已有 19 个 `_result.json`
  - 缺失的是 memory 三项
  - 快照中没有现成 `bfcl_evaluate.log` 或 `score/data_*.csv`，说明大概率只完成了生成、未完成评估

## Test Plan
- 快照整理校验
  - 解压后目标根目录必须直接含有 `benchmarks/` 与 `summary/`
  - 不得残留 `content/stage2_eval_snapshot/` 作为分析根目录
- BFCL 评分校验
  - 官方 partial-eval 成功时，必须产出：
    - `score/data_overall.csv`
    - `score/data_live.csv`
    - `score/data_non_live.csv`
    - `score/data_multi_turn.csv`
  - 评分后确认只覆盖 19 个子类，不虚构 memory 三项
- 报告校验
  - final `CEval/IFEval` 数值必须与快照结果文件完全一致
  - base/search best 数值必须与 `analysis/stage2_best_benchmarks/summary/final_candidate_summary.json` 一致
  - 报告中必须显式写出“BFCL partial，不是完整 official score”
  - 报告中必须给出“为何 final 明显差于 search best”的定量与样例证据

## Assumptions
- 默认不再继续任何新的在线评测，只基于本地快照和本地补评分完成全部收尾。
- 默认只分析 `bf16 final`，不再纳入 `int4`。
- 默认优先采用 BFCL 官方 `partial-eval` 输出；只有它失败时才启用 fallback 脚本。
- 默认最终计划执行时允许写入 `analysis/` 下的新文件与整理后的快照目录。
