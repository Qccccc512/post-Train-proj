# Benchmark Candidate Summary

- generated_at: 2026-04-08 15:47:03
- total_runs: 3
- reference_run: stage2_best_base_model

## Metrics

| run_name | kind | C-Eval | IFEval Strict | IFEval Loose | BFCL Overall | BFCL Non-Live | BFCL Live | BFCL Multi-Turn | Web Search | Memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stage2_best_base_model | base | 79.79% | 34.38% | 40.48% | 34.95% | 88.38% | 78.90% | 31.87% | 2.00% | 1.08% |
| stage2_best_lr2e4_r16 | adapter | 79.57% | 35.67% | 41.59% | 35.21% | 88.35% | 79.50% | 33.38% | 0.50% | 1.08% |
| stage2_best_lr1e4_r16 | adapter | 79.72% | 35.67% | 42.70% | 35.67% | 88.73% | 80.24% | 32.25% | 4.00% | 1.08% |

## Deltas vs Reference

| run_name | dC-Eval | dIFEval Strict | dIFEval Loose | dBFCL Overall | dBFCL Live | dBFCL Multi-Turn | dWeb Search | dMemory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stage2_best_base_model | +0.00pt | +0.00pt | +0.00pt | +0.00pt | +0.00pt | +0.00pt | +0.00pt | +0.00pt |
| stage2_best_lr2e4_r16 | -0.22pt | +1.29pt | +1.11pt | +0.26pt | +0.60pt | +1.51pt | -1.50pt | +0.00pt |
| stage2_best_lr1e4_r16 | -0.07pt | +1.29pt | +2.22pt | +0.72pt | +1.34pt | +0.38pt | +2.00pt | +0.00pt |

## Selection Lens

- 优先看 `BFCL Live` 与 `BFCL Multi-Turn` 是否相对 base 有稳定提升，这两项最贴近当前项目的 tool-calling 目标。
- `C-Eval` 与 `IFEval` 更适合作为护栏指标：如果 adapter 带来明显回退，需要谨慎。
- 如果两个 adapter 的全量 benchmark 几乎持平，再回到 Stage 2 搜索结果，用更低的 `eval_loss` 作为 tie-breaker。

## BFCL-Oriented Ranking

1. stage2_best_lr1e4_r16: BFCL Overall=35.67%, Live=80.24%, Multi-Turn=32.25%, C-Eval=79.72%
2. stage2_best_lr2e4_r16: BFCL Overall=35.21%, Live=79.50%, Multi-Turn=33.38%, C-Eval=79.57%
3. stage2_best_base_model: BFCL Overall=34.95%, Live=78.90%, Multi-Turn=31.87%, C-Eval=79.79%
