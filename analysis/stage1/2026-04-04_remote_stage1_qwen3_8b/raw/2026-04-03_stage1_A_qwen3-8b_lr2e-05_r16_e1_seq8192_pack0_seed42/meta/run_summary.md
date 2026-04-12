# Run Summary: 2026-04-03_stage1_A_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42

- Timestamp: 2026-04-03 18:14:34
- Phase: stage1
- Group: A
- Model: Qwen/Qwen3-8B
- Max seq length: 8192
- Packing: False
- Best checkpoint: /content/post-Train-proj/runs/2026-04-03_stage1_A_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/trainer_output/checkpoint-200
- Last saved checkpoint: /content/post-Train-proj/runs/2026-04-03_stage1_A_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/trainer_output/checkpoint-297

## Dataset

- Train rows: 4750
- Val rows: 250
- Train avg seq len: 1351.46
- Val avg seq len: 1367.6

## Metrics

- Train result: `{"train_runtime": 976.5155, "train_samples_per_second": 4.864, "train_steps_per_second": 0.304, "total_flos": 2.931906850332672e+17, "train_loss": 0.6168801961121736, "epoch": 1.0}`
- Eval result: `{"eval_loss": 0.4943479001522064, "eval_runtime": 22.1385, "eval_samples_per_second": 11.293, "eval_steps_per_second": 11.293, "epoch": 1.0}`
