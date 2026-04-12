# Stage 2 Final BF16 Snapshot Status

- source_archive: `stage2_eval_snapshot (1).zip`
- extracted_to: `analysis/stage2_final_benchmarks/stage2train_20260408_095122_stage2_qwen3_8b_lora_best`
- status: `partial benchmark snapshot recovered from Colab before CU exhaustion`

## Contents

- `ceval` results are complete.
- `ifeval` results are complete.
- `bfcl_v4` contains generated results for `19 / 22` expected BFCL V4 sub-categories.
- within BFCL `multi_turn`, all four sub-categories were started, but generation coverage is partial:
  - `multi_turn_base`: `200 / 200`
  - `multi_turn_long_context`: `200 / 200`
  - `multi_turn_miss_func`: `188 / 200`
  - `multi_turn_miss_param`: `33 / 200`
- missing BFCL categories:
  - `agentic/memory/kv/BFCL_v4_memory_kv_result.json`
  - `agentic/memory/vector/BFCL_v4_memory_vector_result.json`
  - `agentic/memory/rec_sum/BFCL_v4_memory_rec_sum_result.json`

## Local Recovery Work

- On `2026-04-09`, BFCL official `partial-eval` was rerun locally in the `post-train-benchmark` conda environment against this snapshot's `project_root`.
- The recovered score artifacts now live under:
  - `benchmarks/stage2_final_merged_bf16_model/bfcl_v4/project_root/score/`

## Interpretation Note

- This snapshot is sufficient for final failure analysis.
- BFCL `Overall Acc` from `partial-eval` is not directly comparable to a full official run because missing categories are treated as unevaluated / zeroed in summary columns.
- `Multi-Turn` is also a partial recovery inside the already-partial BFCL snapshot, but the recovered official score is still `0.00%`, so incompleteness does not rescue the final model.
- Even with that caveat, the fully completed BFCL groups (`live`, `non_live`, `web_search`) and the recovered partial `multi_turn` score already show severe regression relative to the earlier Stage 2 search-best benchmarks.
