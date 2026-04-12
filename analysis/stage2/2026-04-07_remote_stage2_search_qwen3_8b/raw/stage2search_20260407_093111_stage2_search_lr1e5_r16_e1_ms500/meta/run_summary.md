# Run Summary: stage2search_20260407_093111_stage2_search_lr1e5_r16_e1_ms500

- Timestamp: 2026-04-07 10:29:55
- Phase: stage2search
- Group: S2
- Model: Qwen/Qwen3-8B
- Max seq length: 8192
- Packing: False
- Best global step: 500
- Best checkpoint: /content/post-Train-proj/runs/stage2search_20260407_093111_stage2_search_lr1e5_r16_e1_ms500/trainer_output/checkpoint-500
- Last saved checkpoint: /content/post-Train-proj/runs/stage2search_20260407_093111_stage2_search_lr1e5_r16_e1_ms500/trainer_output/checkpoint-500

## Dataset

- Train rows: 9500
- Val rows: 500
- Train avg seq len: 1904.54
- Val avg seq len: 1999.21

## Metrics

- Train result: `{"train_runtime": 2773.3164, "train_samples_per_second": 2.885, "train_steps_per_second": 0.18, "total_flos": 6.990681245021798e+17, "train_loss": 0.7426694736480713, "epoch": 0.8421052631578947}`
- Eval result: `{"eval_loss": 0.5344148874282837, "eval_runtime": 61.6591, "eval_samples_per_second": 8.109, "eval_steps_per_second": 8.109, "epoch": 0.8421052631578947}`
