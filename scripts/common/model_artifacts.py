#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from scripts.common.runtime_utils import (
    ensure_dir,
    get_hf_token,
    now_timestamp,
    repo_root,
    resolve_path,
    write_json,
)


FINAL_MODEL_SLUG = "final_proj-stage2-best-lr1e4-r16-merged-bf16"
DEFAULT_BASE_MODEL_ID = "Qwen/Qwen3-8B"
DEFAULT_SERVER_MERGED_MODEL_DIR = Path(
    "/content/post-Train-proj/runs/hf_publish/final_proj-stage2-best-lr1e4-r16-merged-bf16/merged_bf16"
)
DEFAULT_LOCAL_MERGED_MODEL_DIR_REL = Path(
    "runs/hf_publish/final_proj-stage2-best-lr1e4-r16-merged-bf16/merged_bf16"
)
DEFAULT_SMOKE_OUTPUT_DIR_REL = Path(
    "analysis/smoke_inference/final_proj-stage2-best-lr1e4-r16-merged-bf16"
)
DEFAULT_GGUF_OUTPUT_DIR_REL = Path(
    "runs/gguf_exports/final_proj-stage2-best-lr1e4-r16-merged-bf16"
)
DEFAULT_LLAMA_CPP_DIR_REL = Path("tools/llama.cpp")
COMPATIBILITY_ASSET_FILENAMES = (
    "generation_config.json",
    "LICENSE",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
)


def default_merged_model_dir() -> Path:
    if DEFAULT_SERVER_MERGED_MODEL_DIR.exists():
        return DEFAULT_SERVER_MERGED_MODEL_DIR
    return (repo_root() / DEFAULT_LOCAL_MERGED_MODEL_DIR_REL).resolve()


def default_smoke_output_dir() -> Path:
    return (repo_root() / DEFAULT_SMOKE_OUTPUT_DIR_REL).resolve()


def default_gguf_output_dir() -> Path:
    return (repo_root() / DEFAULT_GGUF_OUTPUT_DIR_REL).resolve()


def default_llama_cpp_dir() -> Path:
    return (repo_root() / DEFAULT_LLAMA_CPP_DIR_REL).resolve()


def resolve_model_dir(path: str | Path | None = None) -> Path:
    if path is None:
        return default_merged_model_dir()
    return resolve_path(path).resolve()


def resolve_output_dir(path: str | Path | None, *, default_dir: Path) -> Path:
    if path is None:
        return ensure_dir(default_dir)
    return ensure_dir(resolve_path(path))


def resolve_base_model_asset(
    *,
    filename: str,
    base_model_id: str,
    revision: str = "main",
) -> Path | None:
    from huggingface_hub import hf_hub_download

    candidate_path = Path(base_model_id).expanduser()
    if candidate_path.is_absolute():
        local_file = candidate_path / filename
        if local_file.exists():
            return local_file

    token = None
    try:
        token = get_hf_token({"allow_keys_json_fallback": True})
    except Exception:
        token = None

    try:
        downloaded = hf_hub_download(
            repo_id=base_model_id,
            filename=filename,
            repo_type="model",
            revision=revision,
            token=token,
        )
    except Exception:
        return None
    return Path(downloaded)


def sync_base_model_compat_assets(
    *,
    model_dir: str | Path,
    base_model_id: str = DEFAULT_BASE_MODEL_ID,
    output_manifest_path: str | Path | None = None,
    revision: str = "main",
) -> dict[str, Any]:
    resolved_model_dir = resolve_model_dir(model_dir)
    if not resolved_model_dir.exists():
        raise FileNotFoundError(f"Merged model directory does not exist: {resolved_model_dir}")

    copied: list[str] = []
    already_present: list[str] = []
    missing_upstream: list[str] = []
    source_paths: dict[str, str] = {}

    for filename in COMPATIBILITY_ASSET_FILENAMES:
        destination = resolved_model_dir / filename
        if destination.exists():
            already_present.append(filename)
            continue

        source = resolve_base_model_asset(
            filename=filename,
            base_model_id=base_model_id,
            revision=revision,
        )
        if source is None:
            missing_upstream.append(filename)
            continue

        shutil.copy2(source, destination)
        copied.append(filename)
        source_paths[filename] = str(source)

    manifest = {
        "timestamp": now_timestamp(),
        "model_dir": str(resolved_model_dir),
        "base_model_id": base_model_id,
        "copied": copied,
        "already_present": already_present,
        "missing_upstream": missing_upstream,
        "source_paths": source_paths,
    }
    if output_manifest_path is not None:
        write_json(output_manifest_path, manifest)
    return manifest
