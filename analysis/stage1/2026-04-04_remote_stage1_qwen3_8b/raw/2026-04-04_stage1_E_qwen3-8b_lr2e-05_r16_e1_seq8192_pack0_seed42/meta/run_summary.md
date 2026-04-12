# Run Summary: 2026-04-04_stage1_E_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42

- Timestamp: 2026-04-04 04:30:55
- Phase: stage1
- Group: E
- Model: Qwen/Qwen3-8B
- Max seq length: 8192
- Packing: False
- Best global step: 200
- Best checkpoint: /content/post-Train-proj/runs/2026-04-04_stage1_E_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/trainer_output/checkpoint-200
- Last saved checkpoint: /content/post-Train-proj/runs/2026-04-04_stage1_E_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/trainer_output/checkpoint-297

## Dataset

- Train rows: 4750
- Val rows: 250
- Train avg seq len: 465.68
- Val avg seq len: 472.23

## Metrics

- Train result: `{"train_runtime": 505.6061, "train_samples_per_second": 9.395, "train_steps_per_second": 0.587, "total_flos": 1.010257903451136e+17, "train_loss": 0.11072738671844656, "epoch": 1.0}`
- Eval result: `{"eval_loss": 0.030375516042113304, "eval_runtime": 10.9395, "eval_samples_per_second": 22.853, "eval_steps_per_second": 22.853, "epoch": 1.0}`
