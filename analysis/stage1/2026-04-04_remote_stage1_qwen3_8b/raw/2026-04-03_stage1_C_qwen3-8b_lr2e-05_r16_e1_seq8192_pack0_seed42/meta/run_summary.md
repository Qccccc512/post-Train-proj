# Run Summary: 2026-04-03_stage1_C_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42

- Timestamp: 2026-04-03 17:23:39
- Phase: stage1
- Group: C
- Model: Qwen/Qwen3-8B
- Max seq length: 8192
- Packing: False
- Best checkpoint: /content/post-Train-proj/runs/2026-04-03_stage1_C_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/trainer_output/checkpoint-200
- Last saved checkpoint: /content/post-Train-proj/runs/2026-04-03_stage1_C_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/trainer_output/checkpoint-297

## Dataset

- Train rows: 4750
- Val rows: 250
- Train avg seq len: 1512.06
- Val avg seq len: 1424.26

## Metrics

- Train result: `{"train_runtime": 1082.8224, "train_samples_per_second": 4.387, "train_steps_per_second": 0.274, "total_flos": 3.280320844733645e+17, "train_loss": 0.8023134160924841, "epoch": 1.0}`
- Eval result: `{"eval_loss": 0.5123333930969238, "eval_runtime": 23.209, "eval_samples_per_second": 10.772, "eval_steps_per_second": 10.772, "epoch": 1.0}`
