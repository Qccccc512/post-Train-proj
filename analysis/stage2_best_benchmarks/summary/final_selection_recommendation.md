# Stage 2 Final Hyperparameter Recommendation

- generated_at: 2026-04-08 15:48:30
- benchmark_archive: `analysis/stage2_search_benchmark.zip`
- archive_status: complete
- benchmark_root: `analysis/stage2_best_benchmarks/benchmarks`

## Completeness Check

- `stage2_best_base_model`: `ceval` / `ifeval` / `bfcl_v4` / `summary` present
- `stage2_best_lr2e4_r16`: `ceval` / `ifeval` / `bfcl_v4` / `summary` present
- `stage2_best_lr1e4_r16`: `ceval` / `ifeval` / `bfcl_v4` / `summary` present

## Recommendation

**Choose `lr=1e-4, r=16` as the final Stage 2 winner.**

This is no longer a tie that needs `eval_loss` as a tie-breaker. Compared with `lr2e4-r16`, `lr1e4-r16` is better on the broader final benchmark surface:

- BFCL Overall: `35.67%` vs `35.21%`
- BFCL Live: `80.24%` vs `79.50%`
- BFCL Web Search: `4.00%` vs `0.50%`
- IFEval Loose: `42.70%` vs `41.59%`
- C-Eval: `79.72%` vs `79.57%`
- Latency Mean: `76.68s` vs `78.86s`

`lr2e4-r16` keeps one meaningful advantage:

- BFCL Multi-Turn: `33.38%` vs `32.25%`

But that single win is not enough to offset `lr1e4-r16` leading on overall BFCL, live tool-calling, web-search-like agentic behavior, instruction following, and latency.

## Interpretation

- `C-Eval` differences are tiny and within the benchmark standard error scale, so they should be treated as a guardrail, not the main decider.
- Both adapters improve `IFEval Strict` by the same amount over base.
- `lr1e4-r16` shows the better end-to-end tradeoff for the current project goal: stronger practical tool-calling behavior without giving up general capability.
- `lr2e4-r16` should still be kept as the backup candidate if later work decides to optimize specifically for BFCL multi-turn recovery.

## Final Decision

1. Final Stage 2 winner: `lr1e4-r16`
2. Backup candidate: `lr2e4-r16`
3. No need to fall back to Stage 2 search `eval_loss` for tie-breaking in the current result set
