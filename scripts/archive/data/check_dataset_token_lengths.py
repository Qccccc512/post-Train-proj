#!/usr/bin/env python3
from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from transformers import AutoProcessor, AutoTokenizer

from scripts.train.build_dataset_splits import load_dataset_config, load_named_datasets, resolve_dataset_source_dir
from scripts.hub.hf_repo_sync import load_hf_config
from scripts.common.runtime_utils import repo_root, sanitize_name, write_json
from scripts.train.training_data_utils import render_chat_text


DEFAULT_DATASET_CONFIG = "configs/datasets/stage2_search_fixed_10k.yaml"
DEFAULT_TRAIN_CONFIG = "configs/train/stage2_search_lr1e4_r16_e1_ms500.yaml"
DEFAULT_HF_CONFIG = "configs/hf/default.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure rendered token lengths for each training dataset and estimate 4096-token truncation risk."
    )
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--train-config", default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--hf-config", default=DEFAULT_HF_CONFIG)
    parser.add_argument("--threshold", type=int, default=4096)
    parser.add_argument(
        "--output",
        default="runs/dataset_length_reports/stage1_token_lengths_le4096.json",
    )
    parser.add_argument("--model-source", default=None)
    parser.add_argument("--force-local-datasets", action="store_true")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print a progress line every N measured rows within each dataset.",
    )
    parser.add_argument(
        "--top-k-longest",
        type=int,
        default=5,
        help="Store the K longest rows for quick manual inspection.",
    )
    return parser


def resolve_model_source(
    train_config: dict[str, Any],
    hf_config: dict[str, Any],
    explicit_model_source: str | None,
) -> str:
    if explicit_model_source:
        return explicit_model_source
    model_name = train_config["model_name"]
    local_dir = Path(hf_config["models_local_dir"]) / sanitize_name(model_name.split("/")[-1])
    local_dir = (repo_root() / local_dir).resolve()
    if local_dir.exists():
        return str(local_dir)
    return model_name


def load_chat_tokenizer(model_source: str) -> Any:
    failures: list[str] = []
    for loader_name, loader in (("AutoProcessor", AutoProcessor), ("AutoTokenizer", AutoTokenizer)):
        try:
            loaded = loader.from_pretrained(model_source, trust_remote_code=True)
            tokenizer = getattr(loaded, "tokenizer", loaded)
            if hasattr(tokenizer, "apply_chat_template"):
                return tokenizer
            failures.append(f"{loader_name} loaded object without apply_chat_template")
        except Exception as exc:  # pragma: no cover - exercised in real envs
            failures.append(f"{loader_name}: {exc}")
    raise RuntimeError("Unable to load a chat-template-capable tokenizer: " + " | ".join(failures))


def percentile(sorted_values: list[int], ratio: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * ratio
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(sorted_values[low])
    weight = rank - low
    return float(sorted_values[low] * (1 - weight) + sorted_values[high] * weight)


def summarize_lengths(
    lengths: list[int],
    threshold: int,
    longest_rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    if not lengths:
        return {
            "measured_rows": 0,
            "error_rows": len(errors),
            "ratio_le_threshold": None,
            "ratio_gt_threshold": None,
            "longest_rows": longest_rows,
            "errors": errors,
        }

    sorted_lengths = sorted(lengths)
    le_count = sum(length <= threshold for length in sorted_lengths)
    gt_count = len(sorted_lengths) - le_count
    return {
        "measured_rows": len(sorted_lengths),
        "error_rows": len(errors),
        "rows_le_threshold": le_count,
        "rows_gt_threshold": gt_count,
        "ratio_le_threshold": round(le_count / len(sorted_lengths), 4),
        "ratio_gt_threshold": round(gt_count / len(sorted_lengths), 4),
        "avg_tokens": round(sum(sorted_lengths) / len(sorted_lengths), 2),
        "p50_tokens": round(percentile(sorted_lengths, 0.50), 2),
        "p90_tokens": round(percentile(sorted_lengths, 0.90), 2),
        "p95_tokens": round(percentile(sorted_lengths, 0.95), 2),
        "p99_tokens": round(percentile(sorted_lengths, 0.99), 2),
        "max_tokens": sorted_lengths[-1],
        "longest_rows": longest_rows,
        "errors": errors,
    }


def maybe_record_longest(
    longest_rows: list[dict[str, Any]],
    row_info: dict[str, Any],
    top_k: int,
) -> None:
    longest_rows.append(row_info)
    longest_rows.sort(key=lambda item: item["token_length"], reverse=True)
    del longest_rows[top_k:]


def measure_dataset_lengths(
    dataset_name: str,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    threshold: int,
    progress_every: int,
    top_k_longest: int,
) -> dict[str, Any]:
    lengths: list[int] = []
    errors: list[dict[str, Any]] = []
    longest_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        try:
            rendered = render_chat_text(tokenizer, row["messages"], row.get("tools"))
            token_length = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
            lengths.append(token_length)
            maybe_record_longest(
                longest_rows,
                {
                    "row_index": index,
                    "token_length": token_length,
                    "source_dataset": row.get("source_dataset"),
                    "source_index": row.get("source_index"),
                },
                top_k_longest,
            )
        except Exception as exc:
            if len(errors) < 20:
                errors.append(
                    {
                        "row_index": index,
                        "source_dataset": row.get("source_dataset"),
                        "source_index": row.get("source_index"),
                        "error": str(exc),
                    }
                )
        if progress_every > 0 and (index + 1) % progress_every == 0:
            print(
                f"[{dataset_name}] measured {index + 1}/{len(rows)} rows",
                flush=True,
            )

    summary = summarize_lengths(lengths, threshold, longest_rows, errors)
    summary["total_rows"] = len(rows)
    return summary


def estimate_recipe_threshold_risk(
    dataset_stats: dict[str, dict[str, Any]],
    dataset_config: dict[str, Any],
    threshold: int,
) -> dict[str, Any]:
    train_ratio = float(dataset_config["split"]["train_ratio"])
    recipe_estimates: dict[str, Any] = {}
    for group, recipe in dataset_config["recipes"].items():
        total_samples = int(recipe["total_samples"])
        expected_rows_le_threshold = 0.0
        missing_components: list[str] = []
        for component_name, requested_count in recipe["components"].items():
            component_stats = dataset_stats.get(component_name)
            ratio_le_threshold = None if component_stats is None else component_stats.get("ratio_le_threshold")
            if ratio_le_threshold is None:
                missing_components.append(component_name)
                continue
            expected_rows_le_threshold += requested_count * float(ratio_le_threshold)
        if missing_components:
            recipe_estimates[group] = {
                "description": recipe["description"],
                "missing_components": missing_components,
            }
            continue
        expected_ratio_le_threshold = expected_rows_le_threshold / max(1, total_samples)
        recipe_estimates[group] = {
            "description": recipe["description"],
            "total_samples": total_samples,
            "threshold": threshold,
            "expected_rows_le_threshold": round(expected_rows_le_threshold, 2),
            "expected_rows_gt_threshold": round(total_samples - expected_rows_le_threshold, 2),
            "expected_ratio_le_threshold": round(expected_ratio_le_threshold, 4),
            "expected_ratio_gt_threshold": round(1.0 - expected_ratio_le_threshold, 4),
            "expected_train_rows_le_threshold": round(expected_rows_le_threshold * train_ratio, 2),
            "expected_val_rows_le_threshold": round(expected_rows_le_threshold * (1.0 - train_ratio), 2),
        }
    return recipe_estimates


def main() -> None:
    args = build_parser().parse_args()
    dataset_config = load_dataset_config(args.dataset_config)
    train_config = load_dataset_config(args.train_config)
    hf_config = load_hf_config(args.hf_config)

    source_dir = resolve_dataset_source_dir(
        dataset_config,
        hf_config,
        force_local=args.force_local_datasets,
    )
    model_source = resolve_model_source(train_config, hf_config, args.model_source)
    tokenizer = load_chat_tokenizer(model_source)
    named_datasets = load_named_datasets(source_dir, dataset_config)

    dataset_stats: dict[str, dict[str, Any]] = {}
    for dataset_name, rows in named_datasets.items():
        print(f"[{dataset_name}] start: rows={len(rows)} threshold={args.threshold}", flush=True)
        summary = measure_dataset_lengths(
            dataset_name=dataset_name,
            rows=rows,
            tokenizer=tokenizer,
            threshold=args.threshold,
            progress_every=args.progress_every,
            top_k_longest=args.top_k_longest,
        )
        summary["filename"] = dataset_config["datasets"][dataset_name]["filename"]
        dataset_stats[dataset_name] = summary
        print(
            f"[{dataset_name}] <= {args.threshold}: {summary['rows_le_threshold']}/{summary['measured_rows']} "
            f"({summary['ratio_le_threshold']}) | p95={summary['p95_tokens']} | max={summary['max_tokens']}",
            flush=True,
        )

    payload = {
        "threshold": args.threshold,
        "model_source": model_source,
        "source_dir": str(source_dir),
        "dataset_config": args.dataset_config,
        "train_config": args.train_config,
        "datasets": dataset_stats,
        "recipe_estimates": estimate_recipe_threshold_risk(dataset_stats, dataset_config, args.threshold),
    }
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
