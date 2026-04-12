#!/usr/bin/env python3
"""Download all Stage 1 adapters from Hugging Face to local cache."""

import os
from pathlib import Path

from huggingface_hub import snapshot_download


def get_hf_token():
    """Get HF token from environment or keys.json."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        keys_path = Path(__file__).parent.parent / "keys.json"
        if keys_path.exists():
            import json
            with open(keys_path) as f:
                data = json.load(f)
                token = data.get("hf_token")
    return token


# Stage 1 adapter HF paths
STAGE1_ADAPTERS = {
    "A": "yyyyFan/final_proj/runs/stage1_qwen3_8b/2026-04-03_stage1_A_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42",
    "B": "yyyyFan/final_proj/runs/stage1_qwen3_8b/2026-04-04_stage1_B_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42",
    "C": "yyyyFan/final_proj/runs/stage1_qwen3_8b/2026-04-03_stage1_C_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42",
    "D": "yyyyFan/final_proj/runs/stage1_qwen3_8b/2026-04-04_stage1_D_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42",
    "E": "yyyyFan/final_proj/runs/stage1_qwen3_8b/2026-04-04_stage1_E_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42",
    "F": "yyyyFan/final_proj/runs/stage1_qwen3_8b/2026-04-03_stage1_F_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42",
    "G": "yyyyFan/final_proj/runs/stage1_qwen3_8b/2026-04-04_stage1_G_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42",
    "H": "yyyyFan/final_proj/runs/stage1_qwen3_8b/2026-04-04_stage1_H_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42",
    "I": "yyyyFan/final_proj/runs/stage1_qwen3_8b/2026-04-04_stage1_I_qwen3-8b_lr2e-05_r16_e1_seq8192_pack0_seed42",
}


def main():
    # Set HF endpoint for mirror
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    token = get_hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN not found. Set it in environment or keys.json")
    
    cache_dir = Path(__file__).parent.parent / "runs" / "_benchmark_runtime" / "model_cache" / "adapters"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading adapters to: {cache_dir}")
    print(f"Using HF endpoint: {os.environ.get('HF_ENDPOINT', 'default')}")
    print()
    
    for group, hf_path in STAGE1_ADAPTERS.items():
        print(f"\n[Group {group}] Downloading {hf_path}...")
        local_name = hf_path.replace("/", "--")
        dest_dir = cache_dir / local_name
        
        try:
            snapshot_download(
                repo_id=hf_path,
                repo_type="model",
                local_dir=str(dest_dir),
                token=token,
                resume_download=True,
            )
            print(f"✓ Group {group} downloaded to: {dest_dir}")
        except Exception as e:
            print(f"✗ Failed to download Group {group}: {e}")
            raise
    
    print("\n" + "="*60)
    print("All adapters downloaded successfully!")
    print("="*60)
    print("\nLocal adapter paths:")
    for group, hf_path in STAGE1_ADAPTERS.items():
        local_name = hf_path.replace("/", "--")
        dest_dir = cache_dir / local_name
        print(f"  {group}: {dest_dir}")


if __name__ == "__main__":
    main()
