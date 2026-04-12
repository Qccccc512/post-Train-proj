# Run Summary: stage2search_20260407_173210_stage2_search_lr5e5_r32_e1_ms500_cosine

- Timestamp: 2026-04-07 21:00:29
- Phase: stage2search
- Group: S2
- Model: Qwen/Qwen3-8B
- Max seq length: 8192
- Packing: False
- Best global step: 500
- Best checkpoint: /content/post-Train-proj/runs/stage2search_20260407_173210_stage2_search_lr5e5_r32_e1_ms500_cosine/trainer_output/checkpoint-500
- Last saved checkpoint: /content/post-Train-proj/runs/stage2search_20260407_173210_stage2_search_lr5e5_r32_e1_ms500_cosine/trainer_output/checkpoint-500

## Dataset

- Train rows: 9500
- Val rows: 500
- Train avg seq len: 1904.54
- Val avg seq len: 1999.21

## Metrics

- Train result: `{"train_runtime": 2765.521, "train_samples_per_second": 2.893, "train_steps_per_second": 0.181, "total_flos": 7.030765321862676e+17, "train_loss": 0.559985764503479, "epoch": 0.8421052631578947}`
- Eval result: `{"eval_loss": 0.4008652865886688, "eval_runtime": 61.7297, "eval_samples_per_second": 8.1, "eval_steps_per_second": 8.1, "epoch": 0.8421052631578947}`
