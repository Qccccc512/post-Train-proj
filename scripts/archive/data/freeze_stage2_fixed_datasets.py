#!/usr/bin/env python3
from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
from pathlib import Path
from typing import Any

from scripts.train.build_dataset_splits import (
    load_dataset_config,
    resolve_dataset_source_dir,
    sample_component_rows,
    stable_rng,
)
from scripts.hub.hf_repo_sync import load_hf_config
from scripts.common.runtime_utils import ensure_dir, now_timestamp, read_json, write_json


DEFAULT_DATASET_CONFIG = "configs/datasets/stage2_default.yaml"
DEFAULT_HF_CONFIG = "configs/hf/default.yaml"
DEFAULT_OUTPUT_DIR = "datasets/processed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze Stage 2 fixed mixture datasets for search(10k) and final(60k)."
    )
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--hf-config", default=DEFAULT_HF_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force-local", action="store_true")
    return parser


def build_frozen_mixture(
    *,
    group_tag: str,
    named_datasets: dict[str, list[dict[str, Any]]],
    seed: int,
    components: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mixed_rows: list[dict[str, Any]] = []
    component_summary: dict[str, Any] = {}

    for component_name, requested_count in components.items():
        sampled_rows, sampled_source_indices = sample_component_rows(
            rows=named_datasets[component_name],
            requested_count=int(requested_count),
            seed=seed,
            group=group_tag,
            component_name=component_name,
        )
        mixed_rows.extend(sampled_rows)
        component_summary[component_name] = {
            "requested_count": int(requested_count),
            "available_count": len(named_datasets[component_name]),
            "sampled_source_indices": sampled_source_indices,
        }

    stable_rng(seed, group_tag, "frozen-final").shuffle(mixed_rows)

    return mixed_rows, component_summary


def main() -> None:
    args = build_parser().parse_args()

    dataset_config = load_dataset_config(args.dataset_config)
    hf_config = load_hf_config(args.hf_config)
    seed = int(dataset_config.get("seed", 42))

    source_dir = resolve_dataset_source_dir(dataset_config, hf_config, force_local=args.force_local)
    named_datasets = {
        name: read_json(source_dir / entry["filename"])
        for name, entry in dataset_config["datasets"].items()
    }

    output_dir = ensure_dir(args.output_dir)

    specs = [
        {
            "name": "stage2_search_10k",
            "filename": "stage2_search_10k_messages.json",
            "description": "Frozen Stage2 search mixture: Hermes 7000 + Step tool-call 2000 + Step general 1000",
            "components": {
                "hermes": 7000,
                "step_toolcall": 2000,
                "step20k": 1000,
            },
        },
        {
            "name": "stage2_final_60k",
            "filename": "stage2_final_60k_messages.json",
            "description": "Frozen Stage2 final mixture: Hermes 42000 + Step tool-call 12000 + Step general 6000",
            "components": {
                "hermes": 42000,
                "step_toolcall": 12000,
                "step20k": 6000,
            },
        },
    ]

    summary_payload = {
        "timestamp": now_timestamp(),
        "seed": seed,
        "source_dataset_config": args.dataset_config,
        "source_dir": str(source_dir),
        "outputs": [],
    }

    for spec in specs:
        rows, component_summary = build_frozen_mixture(
            group_tag=spec["name"],
            named_datasets=named_datasets,
            seed=seed,
            components=spec["components"],
        )

        output_path = output_dir / spec["filename"]
        write_json(output_path, rows)

        summary_payload["outputs"].append(
            {
                "name": spec["name"],
                "description": spec["description"],
                "filename": spec["filename"],
                "output_path": str(output_path),
                "row_count": len(rows),
                "components": spec["components"],
                "component_summary": component_summary,
            }
        )

        print(f"Wrote {spec['name']} -> {output_path} ({len(rows)} rows)")

    summary_path = output_dir / "stage2_fixed_datasets_summary.json"
    write_json(summary_path, summary_payload)
    print(f"Wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
