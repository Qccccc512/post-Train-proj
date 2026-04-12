#!/usr/bin/env python3
from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

from scripts.common.runtime_utils import (
    OFFICIAL_HF_ENDPOINT,
    ensure_dir,
    get_hf_token,
    now_timestamp,
    sanitize_name,
    write_json,
)
from scripts.hub.adapter_metadata import patch_local_adapter_metadata
from scripts.hub.hf_repo_sync import load_hf_config


DEFAULT_HF_CONFIG = "configs/hf/default.yaml"
DEFAULT_SOURCE_ADAPTER_REPO = "yyyyFan/final_proj-stage2-best-lr1e4-r16"
DEFAULT_TARGET_REPO_ID = "yyyyFan/final_proj-stage2-best-lr1e4-r16-merged-bf16"
DEFAULT_BASE_MODEL_ID = "Qwen/Qwen3-8B"


def stage_log(title: str, detail: str | None = None) -> None:
    timestamp = now_timestamp()
    print("=" * 72, flush=True)
    print(f"[publish_merged_bf16] {timestamp} | {title}", flush=True)
    if detail:
        print(detail, flush=True)


def describe_base_model_source(base_model_source: str, base_model_id: str) -> str:
    source_path = Path(base_model_source).expanduser()
    if source_path.is_absolute():
        return (
            "Resolved base model source: local cache/directory\n"
            f"- base model id: {base_model_id}\n"
            f"- local path: {source_path}"
        )
    return (
        "Resolved base model source: Hugging Face repo/cache\n"
        f"- base model id: {base_model_id}\n"
        f"- source: {base_model_source}\n"
        "- If the base model is not already cached locally, the next load may stay quiet while "
        "downloading several large files."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pull the final LoRA adapter repo from Hugging Face, export a merged BF16 full model "
            "locally, and optionally publish it to a public Hugging Face model repo."
        )
    )
    parser.add_argument("--hf-config", default=DEFAULT_HF_CONFIG)
    parser.add_argument("--source-adapter-repo", default=DEFAULT_SOURCE_ADAPTER_REPO)
    parser.add_argument("--target-repo-id", default=DEFAULT_TARGET_REPO_ID)
    parser.add_argument("--base-model-id", default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-seq-length", type=int, default=32768)
    parser.add_argument("--bf16-dtype", choices=["auto", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--maximum-memory-usage", type=float, default=0.75)
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_api(hf_config: dict[str, Any]) -> tuple[HfApi, str]:
    token = get_hf_token(hf_config)
    api = HfApi(endpoint=OFFICIAL_HF_ENDPOINT, token=token)
    return api, token


def output_root_for_target(target_repo_id: str) -> Path:
    repo_slug = sanitize_name(target_repo_id.split("/")[-1])
    return ensure_dir(Path("runs") / "hf_publish" / repo_slug)


def write_public_merged_readme(
    output_dir: Path,
    *,
    source_adapter_repo: str,
    target_repo_id: str,
    base_model_id: str,
) -> None:
    readme = "\n".join(
        [
            "---",
            f"base_model: {base_model_id}",
            "library_name: transformers",
            "pipeline_tag: text-generation",
            "tags:",
            "- merged-lora",
            "- bf16",
            "- qwen3",
            "- tool-calling",
            "---",
            "",
            f"# {target_repo_id}",
            "",
            "Merged BF16 full-model export of the project's final adopted LoRA adapter.",
            "",
            "## Export Source",
            "",
            f"- Source adapter repo: `{source_adapter_repo}`",
            f"- Base model: `{base_model_id}`",
            "- Export type: `merged BF16`",
            "",
            "## Usage",
            "",
            "Load this repo directly as a standard Hugging Face causal language model checkpoint.",
            "",
        ]
    )
    (output_dir / "README.md").write_text(readme + "\n", encoding="utf-8")


def download_source_adapter(
    *,
    source_adapter_repo: str,
    revision: str,
    token: str,
    destination: Path,
    force: bool,
) -> Path:
    if destination.exists() and force:
        shutil.rmtree(destination)
    ensure_dir(destination)
    snapshot_download(
        repo_id=source_adapter_repo,
        repo_type="model",
        revision=revision,
        token=token,
        endpoint=OFFICIAL_HF_ENDPOINT,
        local_dir=str(destination),
    )
    return destination


def main() -> None:
    args = build_parser().parse_args()
    stage_log(
        "Initialising merged BF16 publish pipeline",
        "\n".join(
            [
                f"- source adapter repo: {args.source_adapter_repo}",
                f"- target repo: {args.target_repo_id}",
                f"- base model: {args.base_model_id}",
                f"- revision: {args.revision}",
                f"- skip upload: {args.skip_upload}",
                f"- force export: {args.force_export}",
            ]
        ),
    )
    hf_config = load_hf_config(args.hf_config)
    api, token = build_api(hf_config)

    output_root = output_root_for_target(args.target_repo_id)
    source_adapter_dir = output_root / "source_adapter"
    merged_output_dir = output_root / "merged_bf16"
    manifest_path = output_root / "publish_manifest.json"

    dry_run_manifest = {
        "timestamp": now_timestamp(),
        "dry_run": True,
        "source_adapter_repo": args.source_adapter_repo,
        "target_repo_id": args.target_repo_id,
        "base_model_id": args.base_model_id,
        "source_adapter_dir": str(source_adapter_dir.resolve()),
        "merged_output_dir": str(merged_output_dir.resolve()),
        "source_repo_exists": api.repo_exists(args.source_adapter_repo, repo_type="model", token=token),
        "target_repo_exists": api.repo_exists(args.target_repo_id, repo_type="model", token=token),
    }

    if args.dry_run:
        write_json(manifest_path, dry_run_manifest)
        stage_log(
            "Dry-run summary",
            "\n".join(
                [
                    f"- source adapter repo: {args.source_adapter_repo}",
                    f"- target repo: {args.target_repo_id}",
                    f"- base model: {args.base_model_id}",
                    f"- local adapter staging dir: {source_adapter_dir.resolve()}",
                    f"- local merged output dir: {merged_output_dir.resolve()}",
                    f"- manifest: {manifest_path.resolve()}",
                ]
            ),
        )
        return

    stage_log(
        "Step 1/5: Downloading source adapter repo",
        "\n".join(
            [
                f"- repo: {args.source_adapter_repo}",
                f"- revision: {args.revision}",
                f"- destination: {source_adapter_dir.resolve()}",
                "- This step is usually small, but it may still take a bit if the local cache is cold.",
            ]
        ),
    )
    adapter_dir = download_source_adapter(
        source_adapter_repo=args.source_adapter_repo,
        revision=args.revision,
        token=token,
        destination=source_adapter_dir,
        force=args.force_export,
    )
    stage_log(
        "Step 1/5 complete",
        "\n".join(
            [
                f"- local adapter dir: {adapter_dir.resolve()}",
                f"- adapter config present: {(adapter_dir / 'adapter_config.json').exists()}",
                f"- readme present: {(adapter_dir / 'README.md').exists()}",
            ]
        ),
    )

    stage_log(
        "Step 2/5: Normalising adapter metadata",
        "\n".join(
            [
                f"- target base model id: {args.base_model_id}",
                "- This step patches adapter_config.json / README.md locally before export.",
            ]
        ),
    )
    metadata_patch = patch_local_adapter_metadata(adapter_dir, args.base_model_id)
    stage_log(
        "Step 2/5 complete",
        "\n".join(
            [
                f"- previous base model: {metadata_patch.get('old_base_model_name_or_path')}",
                f"- new base model: {metadata_patch.get('new_base_model_name_or_path')}",
                f"- adapter_config changed: {metadata_patch.get('changed_adapter_config')}",
                f"- README changed: {metadata_patch.get('changed_readme')}",
            ]
        ),
    )

    stage_log(
        "Step 3/5: Importing export helpers",
        "- Loading the merged-export pipeline from scripts/train/export_merged_model_variants.py",
    )

    from scripts.train.export_merged_model_variants import (
        export_variant,
        resolve_base_model_source,
        upload_exported_model,
    )

    stage_log(
        "Step 3/5 complete",
        "- Export helpers loaded successfully.",
    )
    stage_log(
        "Step 4/5: Resolving base model source",
        "- We are now deciding whether to reuse a local cache/directory or pull Qwen/Qwen3-8B from the Hub.",
    )
    base_model_source = resolve_base_model_source(adapter_dir, args.base_model_id, hf_config)
    stage_log(
        "Step 4/5 complete",
        describe_base_model_source(base_model_source, args.base_model_id),
    )
    stage_log(
        "Step 5/5: Exporting merged BF16 model",
        "\n".join(
            [
                f"- output dir: {merged_output_dir.resolve()}",
                f"- max seq length: {args.max_seq_length}",
                f"- bf16 dtype: {args.bf16_dtype}",
                "- This is the longest step. If the base model is not cached, it may first download 8B weights.",
                "- After that, Unsloth will load the base model, apply the adapter, and save merged BF16 weights.",
            ]
        ),
    )
    variant_manifest = export_variant(
        variant_name=f"{sanitize_name(args.target_repo_id.split('/')[-1])}_merged_bf16",
        save_method="merged_16bit",
        load_in_16bit=True,
        load_in_4bit=False,
        compute_dtype_name=args.bf16_dtype,
        local_dir=merged_output_dir,
        repo_id=None,
        base_model_id=args.base_model_id,
        base_model_source=base_model_source,
        adapter_dir=adapter_dir,
        source_model_dir=None,
        run_name=args.source_adapter_repo.split("/")[-1],
        checkpoint_kind="repo-root",
        max_seq_length=args.max_seq_length,
        maximum_memory_usage=args.maximum_memory_usage,
        trust_remote_code=False,
        force=args.force_export,
        skip_upload=True,
        hf_config=hf_config,
        revision=args.revision,
        private=args.private,
    )
    stage_log(
        "Step 5/5 complete",
        "\n".join(
            [
                f"- merged output dir: {merged_output_dir.resolve()}",
                f"- export manifest: {merged_output_dir / 'hf_merged_export_manifest.json'}",
            ]
        ),
    )

    stage_log(
        "Refreshing merged model card",
        f"- target readme: {(merged_output_dir / 'README.md').resolve()}",
    )
    write_public_merged_readme(
        merged_output_dir,
        source_adapter_repo=args.source_adapter_repo,
        target_repo_id=args.target_repo_id,
        base_model_id=args.base_model_id,
    )

    upload_completed = False
    if not args.skip_upload:
        stage_log(
            "Uploading merged BF16 repo to Hugging Face",
            "\n".join(
                [
                    f"- target repo: {args.target_repo_id}",
                    f"- visibility: {'private' if args.private else 'public'}",
                    "- This step may stay quiet while large safetensors shards are being uploaded.",
                ]
            ),
        )
        upload_exported_model(
            local_dir=merged_output_dir,
            repo_id=args.target_repo_id,
            hf_config=hf_config,
            revision=args.revision,
            private=args.private,
            commit_message=f"Publish merged BF16 model from {args.source_adapter_repo}",
        )
        api.update_repo_settings(
            repo_id=args.target_repo_id,
            repo_type="model",
            private=args.private,
            token=token,
        )
        upload_completed = True
        stage_log(
            "Upload complete",
            f"- published repo: {args.target_repo_id}",
        )
    else:
        stage_log(
            "Upload skipped",
            f"- merged model remains local at: {merged_output_dir.resolve()}",
        )

    manifest = {
        "timestamp": now_timestamp(),
        "dry_run": False,
        "source_adapter_repo": args.source_adapter_repo,
        "target_repo_id": args.target_repo_id,
        "base_model_id": args.base_model_id,
        "source_adapter_dir": str(source_adapter_dir.resolve()),
        "merged_output_dir": str(merged_output_dir.resolve()),
        "metadata_patch": metadata_patch,
        "variant_manifest": variant_manifest,
        "upload_completed": upload_completed,
        "private": args.private,
        "revision": args.revision,
    }
    write_json(manifest_path, manifest)
    stage_log(
        "Publish manifest written",
        f"- manifest: {manifest_path.resolve()}",
    )
    if upload_completed:
        stage_log(
            "Merged BF16 publish finished",
            f"- published merged repo: {args.target_repo_id}",
        )
    else:
        stage_log(
            "Merged BF16 export finished without upload",
            f"- local merged dir: {merged_output_dir.resolve()}",
        )


if __name__ == "__main__":
    main()
