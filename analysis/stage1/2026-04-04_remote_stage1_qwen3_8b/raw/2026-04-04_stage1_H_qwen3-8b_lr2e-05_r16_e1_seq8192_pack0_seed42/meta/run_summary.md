# Run Summary: 2026-04-04_stage1_H_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42

- Timestamp: 2026-04-04 08:10:26
- Phase: stage1
- Group: H
- Model: Qwen/Qwen3-8B
- Max seq length: 8192
- Packing: False
- Best global step: 200
- Best checkpoint: /content/post-Train-proj/runs/2026-04-04_stage1_H_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/trainer_output/checkpoint-200
- Last saved checkpoint: /content/post-Train-proj/runs/2026-04-04_stage1_H_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42/trainer_output/checkpoint-297

## Dataset

- Train rows: 4750
- Val rows: 250
- Train avg seq len: 3740.2
- Val avg seq len: 3480.21

## Metrics

- Train result: `{"train_runtime": 2540.5421, "train_samples_per_second": 1.87, "train_steps_per_second": 0.117, "total_flos": 8.11412062542336e+17, "train_loss": 0.5658154311003508, "epoch": 1.0}`
- Eval result: `{"eval_loss": 0.41692090034484863, "eval_runtime": 53.0224, "eval_samples_per_second": 4.715, "eval_steps_per_second": 4.715, "epoch": 1.0}`
