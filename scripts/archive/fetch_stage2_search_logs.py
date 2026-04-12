#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from hf_repo_sync import load_hf_config
from runtime_utils import ensure_dir, get_hf_token, now_timestamp, resolve_hf_repo_id, write_json


DEFAULT_HF_CONFIG = "configs/hf/default.yaml"
DEFAULT_ANALYSIS_DIR = "analysis/stage2/2026-04-07_remote_stage2_search_qwen3_8b"
ALLOWED_PREFIXES = ("logs/", "meta/", "configs/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch Stage2 search run logs/meta/configs from HF without checkpoints.")
    parser.add_argument("--hf-config", default=DEFAULT_HF_CONFIG)
    parser.add_argument("--analysis-dir", default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument(
        "--run-pattern",
        default="stage2search",
        help="Only fetch runs whose names contain this substring (case-insensitive).",
    )
    parser.add_argument(
        "--run-name",
        action="append",
        dest="run_names",
        help="Explicit run name to fetch. Can be specified multiple times.",
    )
    return parser


def discover_stage2search_runs(files: list[str], marker: str) -> list[str]:
    marker_l = marker.lower()
    runs: set[str] = set()
    for path in files:
        if not path.startswith("runs/"):
            continue
        rel = path[len("runs/") :]
        if "/" not in rel:
            continue
        run_name = rel.split("/", 1)[0]
        if marker_l in run_name.lower():
            runs.add(run_name)
    return sorted(runs)


def should_download(relative_path: str) -> bool:
    return any(relative_path.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def main() -> None:
    args = build_parser().parse_args()

    hf_config = load_hf_config(args.hf_config)
    token = get_hf_token(hf_config)
    repo_id = resolve_hf_repo_id(hf_config)
    repo_type = hf_config.get("repo_type", "model")
    revision = hf_config.get("revision", "main")

    api = HfApi(token=token)
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)

    if args.run_names:
        run_names = sorted(set(args.run_names))
    else:
        run_names = discover_stage2search_runs(repo_files, args.run_pattern)

    analysis_dir = ensure_dir(args.analysis_dir)
    raw_dir = ensure_dir(analysis_dir / "raw")
    summary_dir = ensure_dir(analysis_dir / "summary")

    manifest: dict[str, Any] = {
        "timestamp": now_timestamp(),
        "repo_id": repo_id,
        "repo_type": repo_type,
        "revision": revision,
        "analysis_dir": str(analysis_dir),
        "allowed_prefixes": list(ALLOWED_PREFIXES),
        "run_names": run_names,
        "runs": {},
    }

    for run_name in run_names:
        remote_prefix = f"runs/{run_name}/"
        remote_files = sorted(path for path in repo_files if path.startswith(remote_prefix))
        selected_files = []
        run_out_dir = ensure_dir(raw_dir / run_name)

        # Keep raw run dir clean before re-fetching.
        if run_out_dir.exists():
            for child in run_out_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()

        for remote_path in remote_files:
            relative = remote_path[len(remote_prefix) :]
            if not should_download(relative):
                continue
            cached_path = hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
                repo_type=repo_type,
                revision=revision,
                token=token,
            )
            destination = run_out_dir / relative
            ensure_dir(destination.parent)
            shutil.copy2(cached_path, destination)
            selected_files.append(relative)

        manifest["runs"][run_name] = {
            "remote_files_total": len(remote_files),
            "downloaded_files": len(selected_files),
            "downloaded_relative_paths": selected_files,
        }
        print(f"{run_name}: downloaded {len(selected_files)} files (from {len(remote_files)} remote files)")

    manifest_path = summary_dir / "hf_stage2_search_fetch_manifest.json"
    write_json(manifest_path, manifest)
    print(f"manifest={manifest_path}")
    print(f"runs={len(run_names)}")


if __name__ == "__main__":
    main()
