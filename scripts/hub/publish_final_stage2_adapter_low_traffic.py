#!/usr/bin/env python3
from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from scripts.common.runtime_utils import (
    OFFICIAL_HF_ENDPOINT,
    ensure_dir,
    get_hf_token,
    load_yaml,
    now_timestamp,
    resolve_path,
    write_json,
)
from scripts.hub.adapter_metadata import (
    patch_adapter_config,
    patch_readme_base_model,
)


DEFAULT_HF_CONFIG = "configs/hf/default.yaml"
DEFAULT_PROJECT_REPO = "yyyyFan/final_proj"
DEFAULT_FINAL_REPO = "yyyyFan/final_proj-stage2-best-lr1e4-r16"
DEFAULT_WINNER_RUN = "stage2search_20260407_173210_stage2_search_lr1e4_r16_e1_ms500"
DEFAULT_WINNER_CHECKPOINT = "best"
DEFAULT_FAILED_RUN = "stage2train_20260408_095122_stage2_qwen3_8b_lora"
DEFAULT_CANONICAL_BASE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_MANIFEST_PATH = "analysis/hf_publish/final_stage2_adapter_low_traffic_manifest.json"

FINAL_REPO_FILES = (
    "adapter_model.safetensors",
    "tokenizer.json",
    "adapter_config.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "README.md",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Low-traffic HF maintenance: patch archived adapter metadata in final_proj/runs "
            "and publish the final winner adapter as a standalone repo."
        )
    )
    parser.add_argument("--hf-config", default=DEFAULT_HF_CONFIG)
    parser.add_argument("--project-repo", default=DEFAULT_PROJECT_REPO)
    parser.add_argument("--final-repo", default=DEFAULT_FINAL_REPO)
    parser.add_argument("--winner-run", default=DEFAULT_WINNER_RUN)
    parser.add_argument("--winner-checkpoint", default=DEFAULT_WINNER_CHECKPOINT)
    parser.add_argument("--failed-run", default=DEFAULT_FAILED_RUN)
    parser.add_argument("--canonical-base-model", default=DEFAULT_CANONICAL_BASE_MODEL)
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def load_hf_config(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def build_api(hf_config: dict[str, Any]) -> tuple[HfApi, str]:
    token = get_hf_token(hf_config)
    api = HfApi(endpoint=OFFICIAL_HF_ENDPOINT, token=token)
    return api, token


def checkpoint_dir(run_name: str, checkpoint_kind: str) -> str:
    return f"runs/{run_name}/checkpoints/{checkpoint_kind}"


def default_patch_targets(winner_run: str, failed_run: str) -> list[str]:
    return [
        checkpoint_dir(winner_run, "best"),
        checkpoint_dir(winner_run, "final"),
        checkpoint_dir(failed_run, "best"),
        checkpoint_dir(failed_run, "final"),
    ]


def download_repo_file(
    repo_id: str,
    remote_path: str,
    *,
    token: str,
    revision: str,
    destination: Path,
) -> Path:
    cached_path = hf_hub_download(
        repo_id=repo_id,
        filename=remote_path,
        repo_type="model",
        revision=revision,
        token=token,
        endpoint=OFFICIAL_HF_ENDPOINT,
    )
    ensure_dir(destination.parent)
    shutil.copy2(cached_path, destination)
    return destination


def patch_readme(readme_path: Path, canonical_base_model: str) -> dict[str, Any]:
    return patch_readme_base_model(readme_path, canonical_base_model)


def upload_file_if_needed(
    api: HfApi,
    *,
    repo_id: str,
    local_path: Path,
    path_in_repo: str,
    revision: str,
    commit_message: str,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        commit_message=commit_message,
    )


def patch_existing_run_metadata(
    api: HfApi,
    token: str,
    *,
    repo_id: str,
    revision: str,
    checkpoint_dirs: list[str],
    canonical_base_model: str,
    dry_run: bool,
) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="hf_patch_run_metadata_") as temp_dir:
        temp_root = Path(temp_dir)
        for checkpoint_remote_dir in checkpoint_dirs:
            operation: dict[str, Any] = {
                "checkpoint_remote_dir": checkpoint_remote_dir,
                "files": [],
            }
            adapter_remote_path = f"{checkpoint_remote_dir}/adapter_config.json"
            if not api.file_exists(
                repo_id=repo_id,
                filename=adapter_remote_path,
                repo_type="model",
                revision=revision,
                token=token,
            ):
                operation["missing"] = True
                operations.append(operation)
                continue

            local_dir = temp_root / checkpoint_remote_dir
            adapter_local_path = download_repo_file(
                repo_id,
                adapter_remote_path,
                token=token,
                revision=revision,
                destination=local_dir / "adapter_config.json",
            )
            adapter_patch = patch_adapter_config(adapter_local_path, canonical_base_model)
            operation["files"].append(adapter_patch)
            if adapter_patch["changed"]:
                upload_file_if_needed(
                    api,
                    repo_id=repo_id,
                    local_path=adapter_local_path,
                    path_in_repo=adapter_remote_path,
                    revision=revision,
                    commit_message=f"Fix base model metadata for {checkpoint_remote_dir}",
                    dry_run=dry_run,
                )

            readme_remote_path = f"{checkpoint_remote_dir}/README.md"
            if api.file_exists(
                repo_id=repo_id,
                filename=readme_remote_path,
                repo_type="model",
                revision=revision,
                token=token,
            ):
                readme_local_path = download_repo_file(
                    repo_id,
                    readme_remote_path,
                    token=token,
                    revision=revision,
                    destination=local_dir / "README.md",
                )
                readme_patch = patch_readme(readme_local_path, canonical_base_model)
                operation["files"].append(readme_patch)
                if readme_patch["changed"]:
                    upload_file_if_needed(
                        api,
                        repo_id=repo_id,
                        local_path=readme_local_path,
                        path_in_repo=readme_remote_path,
                        revision=revision,
                        commit_message=f"Fix README base model metadata for {checkpoint_remote_dir}",
                        dry_run=dry_run,
                    )

            operations.append(operation)
    return operations


def stage_final_adapter_repo(
    token: str,
    *,
    project_repo: str,
    winner_run: str,
    winner_checkpoint: str,
    revision: str,
    canonical_base_model: str,
) -> tuple[Path, list[dict[str, Any]]]:
    temp_root = Path(tempfile.mkdtemp(prefix="hf_final_adapter_publish_"))
    staging_dir = temp_root / "final_adapter"
    ensure_dir(staging_dir)
    source_prefix = checkpoint_dir(winner_run, winner_checkpoint)

    staged_files: list[dict[str, Any]] = []
    for filename in FINAL_REPO_FILES:
        remote_path = f"{source_prefix}/{filename}"
        local_path = staging_dir / filename
        download_repo_file(
            project_repo,
            remote_path,
            token=token,
            revision=revision,
            destination=local_path,
        )
        staged_files.append(
            {
                "source_repo": project_repo,
                "source_path": remote_path,
                "staged_path": str(local_path),
                "size_bytes": local_path.stat().st_size,
            }
        )

    patch_adapter_config(staging_dir / "adapter_config.json", canonical_base_model)
    patch_readme(staging_dir / "README.md", canonical_base_model)
    return staging_dir, staged_files


def publish_final_adapter_repo(
    api: HfApi,
    *,
    final_repo: str,
    private: bool,
    revision: str,
    staging_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    uploaded_files = [path.name for path in sorted(staging_dir.iterdir()) if path.is_file()]
    if not dry_run:
        api.create_repo(repo_id=final_repo, repo_type="model", private=private, exist_ok=True)
        for filename in uploaded_files:
            api.upload_file(
                path_or_fileobj=str(staging_dir / filename),
                path_in_repo=filename,
                repo_id=final_repo,
                repo_type="model",
                revision=revision,
                commit_message="Publish final stage2 adapter",
            )

    return {
        "repo_id": final_repo,
        "revision": revision,
        "uploaded_files": uploaded_files,
        "staging_dir": str(staging_dir),
        "dry_run": dry_run,
    }


def main() -> None:
    args = build_parser().parse_args()
    hf_config = load_hf_config(args.hf_config)
    api, token = build_api(hf_config)

    checkpoint_dirs = default_patch_targets(args.winner_run, args.failed_run)
    metadata_patch_operations = patch_existing_run_metadata(
        api,
        token,
        repo_id=args.project_repo,
        revision=args.revision,
        checkpoint_dirs=checkpoint_dirs,
        canonical_base_model=args.canonical_base_model,
        dry_run=args.dry_run,
    )

    staging_dir, staged_files = stage_final_adapter_repo(
        token,
        project_repo=args.project_repo,
        winner_run=args.winner_run,
        winner_checkpoint=args.winner_checkpoint,
        revision=args.revision,
        canonical_base_model=args.canonical_base_model,
    )
    final_repo_publish = publish_final_adapter_repo(
        api,
        final_repo=args.final_repo,
        private=args.private,
        revision=args.revision,
        staging_dir=staging_dir,
        dry_run=args.dry_run,
    )

    manifest = {
        "timestamp": now_timestamp(),
        "dry_run": args.dry_run,
        "project_repo": args.project_repo,
        "final_repo": args.final_repo,
        "winner_run": args.winner_run,
        "winner_checkpoint": args.winner_checkpoint,
        "failed_run": args.failed_run,
        "canonical_base_model": args.canonical_base_model,
        "metadata_patch_operations": metadata_patch_operations,
        "final_repo_staged_files": staged_files,
        "final_repo_publish": final_repo_publish,
        "skipped_large_repo_migrations": [
            "yyyyFan/final_proj-stage2-stage2train_20260408_095122_stage2_qwen3_8b_lora-best-merged-bf16",
            "yyyyFan/final_proj-stage2-stage2train_20260408_095122_stage2_qwen3_8b_lora-best-merged-4bit",
        ],
    }

    manifest_path = resolve_path(args.manifest_path)
    write_json(manifest_path, manifest)

    print(f"Manifest written to {manifest_path}")
    print(f"Patched metadata for {len(metadata_patch_operations)} checkpoint directories")
    if args.dry_run:
        print(f"[dry-run] Would publish final adapter repo: {args.final_repo}")
    else:
        print(f"Published final adapter repo: {args.final_repo}")


if __name__ == "__main__":
    main()
