# Run Summary: 2026-04-04_stage1_G_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42

- Timestamp: 2026-04-04 04:44:26
- Phase: stage1
- Group: G
- Model: Qwen/Qwen3-8B
- Max seq length: 8192
- Packing: False
- Best global step: 200
- Best checkpoint: /content/post-Train-proj/runs/2026-04-04_stage1_G_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/trainer_output/checkpoint-200
- Last saved checkpoint: /content/post-Train-proj/runs/2026-04-04_stage1_G_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/trainer_output/checkpoint-297

## Dataset

- Train rows: 4750
- Val rows: 250
- Train avg seq len: 507.79
- Val avg seq len: 525.3

## Metrics

- Train result: `{"train_runtime": 532.4565, "train_samples_per_second": 8.921, "train_steps_per_second": 0.558, "total_flos": 1.1016098407815168e+17, "train_loss": 0.5690837733271948, "epoch": 1.0}`
- Eval result: `{"eval_loss": 0.3389001488685608, "eval_runtime": 11.7263, "eval_samples_per_second": 21.32, "eval_steps_per_second": 21.32, "epoch": 1.0}`
