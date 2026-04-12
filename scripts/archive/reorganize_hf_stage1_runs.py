#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import CommitOperationCopy, CommitOperationDelete, HfApi

from hf_repo_sync import load_hf_config
from runtime_utils import get_hf_token, now_timestamp, resolve_hf_repo_id, write_json


@dataclass
class MovePlan:
    run_name: str
    source_files: list[str]
    operations_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reorganize stage1 run artifacts in Hugging Face repo from runs/<run> to runs/stage1_qwen3_8b/<run>."
    )
    parser.add_argument("--hf-config", default="configs/hf/default.yaml")
    parser.add_argument("--run-marker", default="stage1", help="Only runs containing this marker in run name are moved.")
    parser.add_argument("--target-folder", default="stage1_qwen3_8b")
    parser.add_argument("--apply", action="store_true", help="Execute remote move. Default is dry-run.")
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Copy to target but do not delete source files in runs/<run>.",
    )
    parser.add_argument(
        "--manifest-path",
        default="runs/_hf_cache/stage1_reorg_manifest.json",
        help="Local manifest output path.",
    )
    return parser


def collect_stage1_run_files(
    *,
    repo_files: list[str],
    runs_remote_dir: str,
    run_marker: str,
    target_folder: str,
) -> dict[str, list[str]]:
    source_prefix = f"{runs_remote_dir.rstrip('/')}/"
    target_prefix = f"{source_prefix}{target_folder}/"
    grouped: dict[str, list[str]] = {}

    for file_path in repo_files:
        if not file_path.startswith(source_prefix):
            continue
        if file_path.startswith(target_prefix):
            continue

        relative = file_path[len(source_prefix) :]
        if "/" not in relative:
            continue
        run_name, _rest = relative.split("/", 1)
        if run_marker.lower() not in run_name.lower():
            continue
        grouped.setdefault(run_name, []).append(file_path)

    for run_name in grouped:
        grouped[run_name].sort()
    return grouped


def build_move_plan(
    grouped_files: dict[str, list[str]],
    runs_remote_dir: str,
    target_folder: str,
    keep_source: bool,
) -> list[MovePlan]:
    source_prefix = f"{runs_remote_dir.rstrip('/')}/"
    target_prefix = f"{source_prefix}{target_folder}/"

    plans: list[MovePlan] = []
    for run_name in sorted(grouped_files):
        source_files = grouped_files[run_name]
        op_count = len(source_files) * (1 if keep_source else 2)
        plans.append(
            MovePlan(
                run_name=run_name,
                source_files=source_files,
                operations_count=op_count,
            )
        )

    return plans


def apply_move_plan(
    *,
    api: HfApi,
    repo_id: str,
    repo_type: str,
    runs_remote_dir: str,
    target_folder: str,
    plans: list[MovePlan],
    keep_source: bool,
) -> None:
    source_prefix = f"{runs_remote_dir.rstrip('/')}/"
    target_prefix = f"{source_prefix}{target_folder}/"

    for plan in plans:
        run_root = f"{source_prefix}{plan.run_name}/"
        operations = []
        for src in plan.source_files:
            relative = src[len(run_root) :]
            dst = f"{target_prefix}{plan.run_name}/{relative}"
            operations.append(CommitOperationCopy(src_path_in_repo=src, path_in_repo=dst))
        if not keep_source:
            for src in plan.source_files:
                operations.append(CommitOperationDelete(path_in_repo=src))

        api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=operations,
            commit_message=f"Move stage1 run {plan.run_name} to {target_folder}",
        )


def main() -> None:
    args = build_parser().parse_args()
    hf_config = load_hf_config(args.hf_config)
    repo_type = hf_config.get("repo_type", "model")
    runs_remote_dir = hf_config["runs_remote_dir"]

    token = get_hf_token(hf_config)
    repo_id = resolve_hf_repo_id(hf_config)
    api = HfApi(token=token)

    repo_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    grouped = collect_stage1_run_files(
        repo_files=repo_files,
        runs_remote_dir=runs_remote_dir,
        run_marker=args.run_marker,
        target_folder=args.target_folder,
    )
    plans = build_move_plan(
        grouped_files=grouped,
        runs_remote_dir=runs_remote_dir,
        target_folder=args.target_folder,
        keep_source=args.keep_source,
    )

    manifest = {
        "timestamp": now_timestamp(),
        "repo_id": repo_id,
        "repo_type": repo_type,
        "runs_remote_dir": runs_remote_dir,
        "target_folder": args.target_folder,
        "run_marker": args.run_marker,
        "apply": bool(args.apply),
        "keep_source": bool(args.keep_source),
        "matched_runs": [plan.run_name for plan in plans],
        "matched_run_count": len(plans),
        "matched_file_count": sum(len(plan.source_files) for plan in plans),
        "operations_count": sum(plan.operations_count for plan in plans),
        "plan": [
            {
                "run_name": plan.run_name,
                "file_count": len(plan.source_files),
                "operations_count": plan.operations_count,
            }
            for plan in plans
        ],
    }

    manifest_path = Path(args.manifest_path)
    write_json(manifest_path, manifest)

    print(f"Repo: {repo_id}")
    print(f"Matched stage1 runs: {manifest['matched_run_count']}")
    print(f"Matched files: {manifest['matched_file_count']}")
    print(f"Planned operations: {manifest['operations_count']}")
    print(f"Manifest: {manifest_path}")

    if not args.apply:
        print("Dry-run only. Re-run with --apply to execute remote move.")
        return

    if not plans:
        print("No matching runs to move.")
        return

    apply_move_plan(
        api=api,
        repo_id=repo_id,
        repo_type=repo_type,
        runs_remote_dir=runs_remote_dir,
        target_folder=args.target_folder,
        plans=plans,
        keep_source=bool(args.keep_source),
    )
    print("Remote move completed.")


if __name__ == "__main__":
    main()
