# Run Summary: 2026-04-04_stage1_I_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42

- Timestamp: 2026-04-04 09:04:55
- Phase: stage1
- Group: I
- Model: Qwen/Qwen3-8B
- Max seq length: 8192
- Packing: False
- Best global step: 200
- Best checkpoint: /content/post-Train-proj/runs/2026-04-04_stage1_I_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/trainer_output/checkpoint-200
- Last saved checkpoint: /content/post-Train-proj/runs/2026-04-04_stage1_I_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/trainer_output/checkpoint-297

## Dataset

- Train rows: 4750
- Val rows: 250
- Train avg seq len: 3489.42
- Val avg seq len: 3489.25

## Metrics

- Train result: `{"train_runtime": 2372.7361, "train_samples_per_second": 2.002, "train_steps_per_second": 0.125, "total_flos": 7.570061049217229e+17, "train_loss": 0.7246365338463574, "epoch": 1.0}`
- Eval result: `{"eval_loss": 0.4891101121902466, "eval_runtime": 53.2111, "eval_samples_per_second": 4.698, "eval_steps_per_second": 4.698, "epoch": 1.0}`
