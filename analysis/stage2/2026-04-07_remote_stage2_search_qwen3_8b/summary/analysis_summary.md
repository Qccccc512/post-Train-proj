# Stage2 Search Remote Analysis

- generated_at: 2026-04-07 23:06:23
- total_runs: 6
- summary_csv: analysis/stage2/2026-04-07_remote_stage2_search_qwen3_8b/summary/stage2_search_runs_comparison.csv
- plot_best_eval_loss: analysis/stage2/2026-04-07_remote_stage2_search_qwen3_8b/plots/best_eval_loss_by_run.png
- plot_final_eval_loss: analysis/stage2/2026-04-07_remote_stage2_search_qwen3_8b/plots/final_eval_loss_by_run.png
- plot_train_loss_curve: analysis/stage2/2026-04-07_remote_stage2_search_qwen3_8b/plots/train_loss_over_steps_by_run.png
- plot_eval_loss_curve: analysis/stage2/2026-04-07_remote_stage2_search_qwen3_8b/plots/eval_loss_over_steps_by_run.png

## Ranking (by best_eval_loss)

1. stage2search_20260407_093111_stage2_search_lr5e5_r16_e1_ms500 | best_eval_loss=0.419397 | final_eval_loss=0.4193972647190094 | lr=5e-05 | r=16
2. stage2search_20260407_093111_stage2_search_lr2e5_r64_e1_ms500 | best_eval_loss=0.422835 | final_eval_loss=0.4228348433971405 | lr=2e-05 | r=64
3. stage2search_20260407_093111_stage2_search_lr2e5_r32_e1_ms500 | best_eval_loss=0.457781 | final_eval_loss=0.45778143405914307 | lr=2e-05 | r=32
4. stage2search_20260407_093111_stage2_search_lr2e5_r16_e1_ms500 | best_eval_loss=0.486248 | final_eval_loss=0.48624780774116516 | lr=2e-05 | r=16
5. stage2search_20260407_093111_stage2_search_lr2e5_r8_e1_ms500 | best_eval_loss=0.513555 | final_eval_loss=0.5135548710823059 | lr=2e-05 | r=8
6. stage2search_20260407_093111_stage2_search_lr1e5_r16_e1_ms500 | best_eval_loss=0.534415 | final_eval_loss=0.5344148874282837 | lr=1e-05 | r=16

## Quick Notes

- 仅基于训练日志指标（eval_loss/train_loss）排序，不包含外部 benchmark。
- 该分析目录只包含 logs/meta/configs，不包含 checkpoints/adapters。
