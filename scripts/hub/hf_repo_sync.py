#!/usr/bin/env python3
from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import HfHubHTTPError

from scripts.common.runtime_utils import (
    OFFICIAL_HF_ENDPOINT,
    ensure_dir,
    get_hf_token,
    load_yaml,
    now_timestamp,
    resolve_hf_repo_id,
    resolve_path,
    run_output_dir,
    sanitize_name,
    write_json,
)


DEFAULT_HF_CONFIG = "configs/hf/default.yaml"
DEFAULT_DATASET_CONFIG = "configs/datasets/stage2_search_fixed_10k.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync datasets, models, and run artifacts with Hugging Face.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--hf-config", default=DEFAULT_HF_CONFIG)

    upload_datasets = subparsers.add_parser("upload-datasets", parents=[shared])
    upload_datasets.add_argument("--source-dir", default="datasets/processed")
    upload_datasets.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)

    fetch_datasets = subparsers.add_parser("fetch-datasets", parents=[shared])
    fetch_datasets.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    fetch_datasets.add_argument("--output-dir")

    fetch_base_model = subparsers.add_parser("fetch-base-model", parents=[shared])
    fetch_base_model.add_argument("--model-id")
    fetch_base_model.add_argument("--output-dir")

    upload_run = subparsers.add_parser("upload-run", parents=[shared])
    upload_run.add_argument("--run-name", required=True)
    upload_run.add_argument("--source-dir")

    fetch_run = subparsers.add_parser("fetch-run", parents=[shared])
    fetch_run.add_argument("--run-name", required=True)
    fetch_run.add_argument("--output-dir")

    return parser


def load_hf_config(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def load_dataset_config(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def build_api(hf_config: dict[str, Any]) -> tuple[HfApi, str, str]:
    token = get_hf_token(hf_config)
    repo_id = resolve_hf_repo_id(hf_config)
    api = HfApi(token=token)
    return api, token, repo_id


def build_upload_api(hf_config: dict[str, Any]) -> tuple[HfApi, str, str]:
    token = get_hf_token(hf_config)
    repo_id = resolve_hf_repo_id(hf_config)
    api = HfApi(endpoint=OFFICIAL_HF_ENDPOINT, token=token)
    return api, token, repo_id


def ensure_remote_repo(api: HfApi, hf_config: dict[str, Any], repo_id: str) -> None:
    api.create_repo(
        repo_id=repo_id,
        repo_type=hf_config.get("repo_type", "model"),
        private=hf_config.get("private", True),
        exist_ok=True,
    )


def _is_retryable_hf_upload_error(exc: Exception) -> bool:
    if isinstance(exc, HfHubHTTPError):
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        if status in {408, 409, 425, 429}:
            return True
        if status is not None and status >= 500:
            return True
        return False

    message = str(exc).lower()
    transient_markers = (
        "timeout",
        "temporarily unavailable",
        "connection reset",
        "connection aborted",
        "goaway",
        "internal server error",
    )
    return any(marker in message for marker in transient_markers)


def upload_folder_with_retry(
    api: HfApi,
    hf_config: dict[str, Any],
    *,
    repo_id: str,
    repo_type: str,
    folder_path: str,
    path_in_repo: str,
    allow_patterns: list[str] | None,
    commit_message: str,
    revision: str | None = None,
) -> None:
    max_attempts = max(1, int(hf_config.get("upload_max_attempts", 6)))
    initial_backoff = max(0.1, float(hf_config.get("upload_retry_initial_backoff_s", 2.0)))
    max_backoff = max(initial_backoff, float(hf_config.get("upload_retry_max_backoff_s", 30.0)))

    for attempt in range(1, max_attempts + 1):
        try:
            api.upload_folder(
                repo_id=repo_id,
                repo_type=repo_type,
                folder_path=folder_path,
                path_in_repo=path_in_repo,
                allow_patterns=allow_patterns,
                commit_message=commit_message,
                revision=revision,
            )
            return
        except Exception as exc:
            should_retry = _is_retryable_hf_upload_error(exc)
            if not should_retry or attempt >= max_attempts:
                raise

            wait_s = min(max_backoff, initial_backoff * (2 ** (attempt - 1)))
            print(
                "[hf_repo_sync] upload_folder transient failure "
                f"(attempt {attempt}/{max_attempts}) for {path_in_repo}: {exc}. "
                f"Retrying in {wait_s:.1f}s..."
            )
            time.sleep(wait_s)


def dataset_filenames_from_config(dataset_config: dict[str, Any]) -> list[str]:
    filenames = {entry["filename"] for entry in dataset_config["datasets"].values()}
    for diagnostic in dataset_config.get("diagnostic_sets", {}).values():
        if diagnostic.get("enabled"):
            filenames.add(diagnostic["filename"])
    return sorted(filenames)


def fetch_dataset_files(
    dataset_config: dict[str, Any],
    hf_config: dict[str, Any],
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    _, token, repo_id = build_api(hf_config)
    output_dir = ensure_dir(output_dir or hf_config["datasets_local_dir"])
    remote_dir = hf_config["dataset_remote_dir"].rstrip("/")
    remote_dir_candidates = [remote_dir]
    if remote_dir == "datasets":
        remote_dir_candidates.append("datasets/v4")
    copied_files: list[dict[str, Any]] = []

    for filename in dataset_filenames_from_config(dataset_config):
        cached_path = None
        remote_path = None
        last_error: Exception | None = None
        for remote_dir_candidate in remote_dir_candidates:
            candidate_path = f"{remote_dir_candidate}/{filename}"
            try:
                cached_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=candidate_path,
                    repo_type=hf_config.get("repo_type", "model"),
                    revision=hf_config.get("revision", "main"),
                    token=token,
                )
                remote_path = candidate_path
                break
            except Exception as exc:
                last_error = exc
        if cached_path is None or remote_path is None:
            raise last_error or FileNotFoundError(f"Unable to fetch dataset file: {filename}")
        destination = output_dir / filename
        shutil.copy2(cached_path, destination)
        copied_files.append(
            {
                "remote_path": remote_path,
                "local_path": str(destination),
                "size_bytes": destination.stat().st_size,
            }
        )

    manifest = {
        "timestamp": now_timestamp(),
        "repo_id": repo_id,
        "remote_dir": remote_dir,
        "remote_dir_candidates": remote_dir_candidates,
        "output_dir": str(output_dir),
        "files": copied_files,
    }
    write_json(output_dir / "hf_fetch_manifest.json", manifest)
    return manifest


def upload_dataset_folder(
    dataset_config: dict[str, Any],
    hf_config: dict[str, Any],
    source_dir: str | Path,
) -> dict[str, Any]:
    api, _, repo_id = build_upload_api(hf_config)
    ensure_remote_repo(api, hf_config, repo_id)
    source_dir = resolve_path(source_dir)
    remote_dir = hf_config["dataset_remote_dir"].rstrip("/")

    filenames = set(dataset_filenames_from_config(dataset_config))
    local_dataset_files = {
        path.name
        for path in source_dir.iterdir()
        if path.is_file()
        and path.suffix in {".json", ".jsonl"}
        and path.name not in {"hf_upload_manifest.json", "hf_fetch_manifest.json"}
    }
    extra_names = {"normalization_summary.json", "cleaning_summary.json"}
    extra_names.update(path.name for path in source_dir.glob("*_summary.json"))
    allow_patterns = [
        name
        for name in sorted(filenames | local_dataset_files | extra_names)
        if (source_dir / name).exists()
    ]

    upload_folder_with_retry(
        api,
        hf_config,
        repo_id=repo_id,
        repo_type=hf_config.get("repo_type", "model"),
        folder_path=str(source_dir),
        path_in_repo=remote_dir,
        allow_patterns=allow_patterns,
        commit_message=f"Upload processed datasets to {remote_dir}",
    )

    manifest = {
        "timestamp": now_timestamp(),
        "repo_id": repo_id,
        "source_dir": str(source_dir),
        "remote_dir": remote_dir,
        "uploaded_files": allow_patterns,
    }
    write_json(source_dir / "hf_upload_manifest.json", manifest)
    return manifest


def fetch_base_model_snapshot(hf_config: dict[str, Any], model_id: str | None, output_dir: str | Path | None) -> dict[str, Any]:
    try:
        _, token, _ = build_api(hf_config)
    except Exception:
        token = None
    model_id = model_id or hf_config["base_model_id"]
    output_dir = ensure_dir(output_dir or Path(hf_config["models_local_dir"]) / sanitize_name(model_id.split("/")[-1]))

    snapshot_download(
        repo_id=model_id,
        repo_type="model",
        local_dir=str(output_dir),
        token=token,
        resume_download=True,
    )

    manifest = {
        "timestamp": now_timestamp(),
        "model_id": model_id,
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "hf_model_fetch_manifest.json", manifest)
    return manifest


def stage_run_artifacts(source_dir: str | Path, hf_config: dict[str, Any]) -> Path:
    source_dir = resolve_path(source_dir)
    include_subdirs = []
    upload_cfg = hf_config.get("upload", {})
    if upload_cfg.get("include_best_checkpoint", True):
        include_subdirs.append("checkpoints/best")
    if upload_cfg.get("include_final_checkpoint", True):
        include_subdirs.append("checkpoints/final")
    if upload_cfg.get("include_logs", True):
        include_subdirs.append("logs")
    if upload_cfg.get("include_configs", True):
        include_subdirs.append("configs")
    if upload_cfg.get("include_meta", True):
        include_subdirs.append("meta")

    temp_root = Path(tempfile.mkdtemp(prefix="post-train-run-stage-"))
    staged_root = temp_root / source_dir.name
    ensure_dir(staged_root)

    for rel_path in include_subdirs:
        src = source_dir / rel_path
        if src.exists():
            dst = staged_root / rel_path
            ensure_dir(dst.parent)
            shutil.copytree(src, dst)

    return staged_root


def upload_run_artifacts(hf_config: dict[str, Any], run_name: str, source_dir: str | Path | None = None) -> dict[str, Any]:
    api, _, repo_id = build_upload_api(hf_config)
    ensure_remote_repo(api, hf_config, repo_id)
    source_dir = resolve_path(source_dir or run_output_dir(run_name))
    staged_dir = stage_run_artifacts(source_dir, hf_config)
    remote_dir = f"{hf_config['runs_remote_dir'].rstrip('/')}/{run_name}"

    upload_folder_with_retry(
        api,
        hf_config,
        repo_id=repo_id,
        repo_type=hf_config.get("repo_type", "model"),
        folder_path=str(staged_dir),
        path_in_repo=remote_dir,
        allow_patterns=None,
        commit_message=f"Upload run artifacts for {run_name}",
    )

    uploaded_files = sorted(str(path.relative_to(staged_dir)) for path in staged_dir.rglob("*") if path.is_file())
    manifest = {
        "timestamp": now_timestamp(),
        "repo_id": repo_id,
        "run_name": run_name,
        "source_dir": str(source_dir),
        "remote_dir": remote_dir,
        "uploaded_files": uploaded_files,
    }
    write_json(source_dir / "meta" / "hf_sync_manifest.json", manifest)
    shutil.rmtree(staged_dir.parent, ignore_errors=True)
    return manifest


def fetch_run_artifacts(hf_config: dict[str, Any], run_name: str, output_dir: str | Path | None = None) -> dict[str, Any]:
    _, token, repo_id = build_api(hf_config)
    remote_prefix = f"{hf_config['runs_remote_dir'].rstrip('/')}/{run_name}/"
    output_dir = ensure_dir(output_dir or run_output_dir(run_name))

    api = HfApi(token=token)
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type=hf_config.get("repo_type", "model"))
    matched = sorted(path for path in repo_files if path.startswith(remote_prefix))
    copied: list[dict[str, Any]] = []

    for remote_path in matched:
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            repo_type=hf_config.get("repo_type", "model"),
            revision=hf_config.get("revision", "main"),
            token=token,
        )
        relative = Path(remote_path).relative_to(remote_prefix.rstrip("/"))
        destination = output_dir / relative
        ensure_dir(destination.parent)
        shutil.copy2(cached_path, destination)
        copied.append({"remote_path": remote_path, "local_path": str(destination)})

    manifest = {
        "timestamp": now_timestamp(),
        "repo_id": repo_id,
        "run_name": run_name,
        "output_dir": str(output_dir),
        "files": copied,
    }
    write_json(output_dir / "meta" / "hf_fetch_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    hf_config = load_hf_config(args.hf_config)

    if args.command == "upload-datasets":
        dataset_config = load_dataset_config(args.dataset_config)
        upload_dataset_folder(dataset_config, hf_config, args.source_dir)
        return

    if args.command == "fetch-datasets":
        dataset_config = load_dataset_config(args.dataset_config)
        fetch_dataset_files(dataset_config, hf_config, args.output_dir)
        return

    if args.command == "fetch-base-model":
        fetch_base_model_snapshot(hf_config, args.model_id, args.output_dir)
        return

    if args.command == "upload-run":
        upload_run_artifacts(hf_config, args.run_name, args.source_dir)
        return

    if args.command == "fetch-run":
        fetch_run_artifacts(hf_config, args.run_name, args.output_dir)
        return

    raise ValueError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
