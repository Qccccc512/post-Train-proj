# Run Summary: 2026-04-04_stage1_B_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42

- Timestamp: 2026-04-04 03:59:41
- Phase: stage1
- Group: B
- Model: Qwen/Qwen3-8B
- Max seq length: 8192
- Packing: False
- Best global step: 200
- Best checkpoint: /content/post-Train-proj/runs/2026-04-04_stage1_B_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/trainer_output/checkpoint-200
- Last saved checkpoint: /content/post-Train-proj/runs/2026-04-04_stage1_B_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/trainer_output/checkpoint-297

## Dataset

- Train rows: 4749
- Val rows: 251
- Train avg seq len: 1239.9
- Val avg seq len: 1169.42

## Metrics

- Train result: `{"train_runtime": 915.6392, "train_samples_per_second": 5.187, "train_steps_per_second": 0.324, "total_flos": 2.6893224603558912e+17, "train_loss": 0.601061690937389, "epoch": 1.0}`
- Eval result: `{"eval_loss": 0.41620779037475586, "eval_runtime": 19.7581, "eval_samples_per_second": 12.704, "eval_steps_per_second": 12.704, "epoch": 1.0}`
