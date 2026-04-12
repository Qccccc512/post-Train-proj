# Run Summary: 2026-04-04_stage1_D_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42

- Timestamp: 2026-04-04 04:21:03
- Phase: stage1
- Group: D
- Model: Qwen/Qwen3-8B
- Max seq length: 8192
- Packing: False
- Best global step: 200
- Best checkpoint: /content/post-Train-proj/runs/2026-04-04_stage1_D_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/trainer_output/checkpoint-200
- Last saved checkpoint: /content/post-Train-proj/runs/2026-04-04_stage1_D_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/trainer_output/checkpoint-297

## Dataset

- Train rows: 4750
- Val rows: 250
- Train avg seq len: 1616.4
- Val avg seq len: 1469.76

## Metrics

- Train result: `{"train_runtime": 1134.7411, "train_samples_per_second": 4.186, "train_steps_per_second": 0.262, "total_flos": 3.506684255357645e+17, "train_loss": 0.7950680906122382, "epoch": 1.0}`
- Eval result: `{"eval_loss": 0.5757744312286377, "eval_runtime": 23.6881, "eval_samples_per_second": 10.554, "eval_steps_per_second": 10.554, "epoch": 1.0}`
