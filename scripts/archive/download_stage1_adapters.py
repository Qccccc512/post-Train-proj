#!/usr/bin/env python3
"""Download Stage 1 adapters from yyyyFan/final_proj repository."""

import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

# Read HF token
with open('/content/post-Train-proj/keys.json') as f:
    token = json.load(f).get('hf_token')

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Stage 1 runs
STAGE1_RUNS = [
    "2026-04-03_stage1_A_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42",
    "2026-04-04_stage1_B_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42",
    "2026-04-03_stage1_C_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42",
    "2026-04-04_stage1_D_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42",
    "2026-04-04_stage1_E_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42",
    "2026-04-03_stage1_F_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42",
    "2026-04-04_stage1_G_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42",
    "2026-04-04_stage1_H_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42",
    "2026-04-04_stage1_I_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42",
]

# Target directory
cache_dir = Path('/content/post-Train-proj/runs/_benchmark_runtime/model_cache/adapters')
cache_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading to: {cache_dir}")
print(f"Using mirror: {os.environ.get('HF_ENDPOINT')}")
print()

# Download each run
for i, run_name in enumerate(STAGE1_RUNS, 1):
    print(f"[{i}/9] Downloading {run_name}...")
    
    # Download the entire repository but filter for this specific run
    target_dir = cache_dir / run_name
    
    # Use allow_patterns to only download files for this run
    patterns = [f"runs/stage1_qwen3_8b/{run_name}/*"]
    
    try:
        snapshot_download(
            repo_id='yyyyFan/final_proj',
            repo_type='model',
            local_dir=str(target_dir),
            token=token,
            allow_patterns=patterns,
            resume_download=True,
        )
        print(f"  ✓ Downloaded to: {target_dir}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        raise

print("\n" + "="*60)
print("All adapters downloaded successfully!")
print("="*60)
print("\nLocal paths:")
for run_name in STAGE1_RUNS:
    target_dir = cache_dir / run_name / 'runs' / 'stage1_qwen3_8b' / run_name
    print(f"  {run_name[:20]}... -> {target_dir}")
