# Stage2 Search Remote Analysis

- generated_at: 2026-04-08 08:44:17
- total_runs: 4
- summary_csv: analysis/stage2/2026-04-08_remote_stage2_search_qwen3_8b_round2/summary/stage2_search_runs_comparison.csv
- plot_best_eval_loss: analysis/stage2/2026-04-08_remote_stage2_search_qwen3_8b_round2/plots/best_eval_loss_by_run.png
- plot_final_eval_loss: analysis/stage2/2026-04-08_remote_stage2_search_qwen3_8b_round2/plots/final_eval_loss_by_run.png
- plot_train_loss_curve: analysis/stage2/2026-04-08_remote_stage2_search_qwen3_8b_round2/plots/train_loss_over_steps_by_run.png
- plot_eval_loss_curve: analysis/stage2/2026-04-08_remote_stage2_search_qwen3_8b_round2/plots/eval_loss_over_steps_by_run.png

## Ranking (by best_eval_loss)

1. stage2search_20260407_173210_stage2_search_lr2e4_r16_e1_ms500 | best_eval_loss=0.376867 | final_eval_loss=0.3768673837184906 | lr=0.0002 | r=16
2. stage2search_20260407_173210_stage2_search_lr1e4_r16_e1_ms500 | best_eval_loss=0.391938 | final_eval_loss=0.3919379413127899 | lr=0.0001 | r=16
3. stage2search_20260407_173210_stage2_search_lr5e5_r32_e1_ms500_cosine | best_eval_loss=0.400865 | final_eval_loss=0.4008652865886688 | lr=5e-05 | r=32
4. stage2search_20260407_173210_stage2_search_lr5e5_r16_e1_ms500_cosine | best_eval_loss=0.420284 | final_eval_loss=0.42028406262397766 | lr=5e-05 | r=16

## Quick Notes

- 仅基于训练日志指标（eval_loss/train_loss）排序，不包含外部 benchmark。
- 该分析目录只包含 logs/meta/configs，不包含 checkpoints/adapters。
