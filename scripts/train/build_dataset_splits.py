#!/usr/bin/env python3
from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import json
import random
from pathlib import Path
from typing import Any

from scripts.hub.hf_repo_sync import fetch_dataset_files, load_dataset_config, load_hf_config
from scripts.common.runtime_utils import (
    copy_resolved_configs,
    ensure_dir,
    generate_run_name,
    now_timestamp,
    read_json,
    resolve_path,
    run_output_dir,
    write_json,
    write_jsonl,
)


DEFAULT_DATASET_CONFIG = "configs/datasets/stage2_search_fixed_10k.yaml"
DEFAULT_TRAIN_CONFIG = "configs/train/stage2_search_lr1e4_r16_e1_ms500.yaml"
DEFAULT_HF_CONFIG = "configs/hf/default.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build reproducible train/val splits from processed datasets.")
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--train-config", default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--hf-config", default=DEFAULT_HF_CONFIG)
    parser.add_argument("--group")
    parser.add_argument("--phase")
    parser.add_argument("--run-name")
    parser.add_argument("--output-dir")
    parser.add_argument("--force-local", action="store_true")
    return parser


def stable_rng(seed: int, *parts: str) -> random.Random:
    return random.Random(f"{seed}::" + "::".join(parts))


def resolve_dataset_source_dir(
    dataset_config: dict[str, Any],
    hf_config: dict[str, Any],
    force_local: bool = False,
) -> Path:
    if force_local or dataset_config.get("source_mode") == "local":
        return resolve_path(dataset_config["local_fallback_dir"])

    candidate = resolve_path(hf_config["datasets_local_dir"])
    required = [candidate / entry["filename"] for entry in dataset_config["datasets"].values()]
    if any(not path.exists() for path in required):
        fetch_dataset_files(dataset_config, hf_config, candidate)
    return candidate


def load_named_datasets(source_dir: Path, dataset_config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    datasets: dict[str, list[dict[str, Any]]] = {}
    for name, entry in dataset_config["datasets"].items():
        datasets[name] = read_json(source_dir / entry["filename"])
    return datasets


def sample_component_rows(
    rows: list[dict[str, Any]],
    requested_count: int,
    seed: int,
    group: str,
    component_name: str,
) -> tuple[list[dict[str, Any]], list[int]]:
    if requested_count > len(rows):
        raise ValueError(
            f"Requested {requested_count} rows from {component_name}, but only {len(rows)} rows are available."
        )

    indices = list(range(len(rows)))
    stable_rng(seed, group, component_name, "sample").shuffle(indices)
    chosen_indices = indices[:requested_count]

    sampled_rows: list[dict[str, Any]] = []
    for rank, index in enumerate(chosen_indices):
        row = dict(rows[index])
        row["mixture_component"] = component_name
        row["mixture_group"] = group
        row["mixture_sample_rank"] = rank
        sampled_rows.append(row)
    return sampled_rows, chosen_indices


def split_component_rows(
    rows: list[dict[str, Any]],
    train_ratio: float,
    seed: int,
    group: str,
    component_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int], list[int]]:
    indices = list(range(len(rows)))
    stable_rng(seed, group, component_name, "split").shuffle(indices)
    train_count = int(len(rows) * train_ratio)
    train_idx = indices[:train_count]
    val_idx = indices[train_count:]

    train_rows = [dict(rows[idx], split="train") for idx in train_idx]
    val_rows = [dict(rows[idx], split="val") for idx in val_idx]
    return train_rows, val_rows, train_idx, val_idx


def build_splits(
    dataset_config: dict[str, Any],
    datasets: dict[str, list[dict[str, Any]]],
    group: str,
) -> dict[str, Any]:
    recipe = dataset_config["recipes"][group]
    train_ratio = float(dataset_config["split"]["train_ratio"])
    seed = int(dataset_config.get("seed", 42))

    train_rows_all: list[dict[str, Any]] = []
    val_rows_all: list[dict[str, Any]] = []
    component_summary: dict[str, Any] = {}

    for component_name, requested_count in recipe["components"].items():
        sampled_rows, sampled_source_indices = sample_component_rows(
            rows=datasets[component_name],
            requested_count=requested_count,
            seed=seed,
            group=group,
            component_name=component_name,
        )
        train_rows, val_rows, train_indices, val_indices = split_component_rows(
            rows=sampled_rows,
            train_ratio=train_ratio,
            seed=seed,
            group=group,
            component_name=component_name,
        )
        train_rows_all.extend(train_rows)
        val_rows_all.extend(val_rows)
        component_summary[component_name] = {
            "requested_count": requested_count,
            "available_count": len(datasets[component_name]),
            "sampled_source_indices": sampled_source_indices,
            "train_sample_positions": train_indices,
            "val_sample_positions": val_indices,
            "train_count": len(train_rows),
            "val_count": len(val_rows),
        }

    stable_rng(seed, group, "train-final").shuffle(train_rows_all)
    stable_rng(seed, group, "val-final").shuffle(val_rows_all)

    return {
        "recipe": recipe,
        "train_rows": train_rows_all,
        "val_rows": val_rows_all,
        "component_summary": component_summary,
    }


def maybe_export_diagnostic_sets(
    output_dir: Path,
    source_dir: Path,
    dataset_config: dict[str, Any],
) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for diagnostic_name, diagnostic_cfg in dataset_config.get("diagnostic_sets", {}).items():
        if not diagnostic_cfg.get("enabled"):
            continue
        rows = read_json(source_dir / diagnostic_cfg["filename"])
        destination = output_dir / "dataset" / f"{diagnostic_name}.jsonl"
        write_jsonl(destination, rows)
        exported.append(
            {
                "name": diagnostic_name,
                "filename": diagnostic_cfg["filename"],
                "output_path": str(destination),
                "row_count": len(rows),
            }
        )
    return exported


def main() -> None:
    args = build_parser().parse_args()
    dataset_config = load_dataset_config(args.dataset_config)
    hf_config = load_hf_config(args.hf_config)
    train_config = load_dataset_config(args.train_config)

    group = args.group or dataset_config.get("default_group", "C")
    phase = args.phase or train_config.get("phase") or dataset_config.get("default_phase", "stage1")
    run_name = args.run_name or generate_run_name(phase, group, train_config, dataset_config)
    output_dir = resolve_path(args.output_dir) if args.output_dir else run_output_dir(run_name)
    ensure_dir(output_dir / "dataset")
    ensure_dir(output_dir / "meta")
    ensure_dir(output_dir / "configs")

    source_dir = resolve_dataset_source_dir(dataset_config, hf_config, force_local=args.force_local)
    named_datasets = load_named_datasets(source_dir, dataset_config)
    split_payload = build_splits(dataset_config, named_datasets, group)

    train_path = output_dir / "dataset" / "train.jsonl"
    val_path = output_dir / "dataset" / "val.jsonl"
    write_jsonl(train_path, split_payload["train_rows"])
    write_jsonl(val_path, split_payload["val_rows"])

    diagnostics = maybe_export_diagnostic_sets(output_dir, source_dir, dataset_config)

    manifest = {
        "timestamp": now_timestamp(),
        "run_name": run_name,
        "phase": phase,
        "group": group,
        "recipe": split_payload["recipe"],
        "source_dir": str(source_dir),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_count": len(split_payload["train_rows"]),
        "val_count": len(split_payload["val_rows"]),
        "diagnostic_sets": diagnostics,
    }
    split_summary = {
        "component_summary": split_payload["component_summary"],
        "train_count": len(split_payload["train_rows"]),
        "val_count": len(split_payload["val_rows"]),
    }
    sample_indices = {
        component: {
            "sampled_source_indices": summary["sampled_source_indices"],
            "train_sample_positions": summary["train_sample_positions"],
            "val_sample_positions": summary["val_sample_positions"],
        }
        for component, summary in split_payload["component_summary"].items()
    }

    write_json(output_dir / "meta" / "dataset_manifest.json", manifest)
    write_json(output_dir / "meta" / "split_summary.json", split_summary)
    write_json(output_dir / "meta" / "sample_indices.json", sample_indices)
    copy_resolved_configs(output_dir / "configs", dataset_config, train_config, hf_config)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
