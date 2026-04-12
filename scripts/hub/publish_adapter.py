#!/usr/bin/env python3
from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

from huggingface_hub import HfApi

from scripts.common.runtime_utils import get_hf_token, now_timestamp, write_json
from scripts.hub.adapter_metadata import detect_base_model_id, stage_adapter_for_upload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload a local PEFT adapter directory to a Hugging Face model repo.")
    parser.add_argument("--source-dir", required=True, help="Local adapter directory, e.g. runs/<run>/checkpoints/final")
    parser.add_argument("--repo-id", required=True, help="Target HF model repo id.")
    parser.add_argument("--base-model-id", help="Override the base model repo id recorded in adapter metadata.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--manifest-path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_dir = Path(args.source_dir).expanduser().resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Adapter source directory does not exist: {source_dir}")
    base_model_id = detect_base_model_id(source_dir, args.base_model_id)
    staging_dir = stage_adapter_for_upload(source_dir, base_model_id)

    api = HfApi(token=get_hf_token())
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(staging_dir),
        revision=args.revision,
        commit_message=f"Upload adapter from {source_dir.name}",
    )

    manifest = {
        "timestamp": now_timestamp(),
        "repo_id": args.repo_id,
        "source_dir": str(source_dir),
        "staging_dir": str(staging_dir),
        "base_model_id": base_model_id,
        "revision": args.revision,
    }
    manifest_path = Path(args.manifest_path).expanduser().resolve() if args.manifest_path else source_dir / "hf_adapter_publish_manifest.json"
    write_json(manifest_path, manifest)
    print(f"Published adapter to {args.repo_id}")


if __name__ == "__main__":
    main()
