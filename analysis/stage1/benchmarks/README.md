# Stage1 Benchmark Outputs

`analysis/stage1/benchmarks/` 存放阶段一的 benchmark 评测结果。

## 文件管理策略

- Smoke test 结果：运行完验证后手动删除，不提交到 git
- 正式评测结果：保留完整输出，与代码一起提交到 git
- 包含评分的关键文件：
  - `*/summary/benchmark_run_summary.json` - 运行摘要
  - `*/summary/benchmark_run_summary.md` - 人类可读摘要
  - `*/bfcl_v4/project_root/score/data_overall.csv` - BFCL 总体评分
  - `*/ceval/*/results*.json` - C-Eval 详细结果
  - `*/ifeval/*/results*.json` - IFEval 详细结果
