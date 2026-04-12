#!/usr/bin/env python3
from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from scripts.train.build_dataset_splits import build_splits, load_dataset_config, load_named_datasets
from scripts.common.runtime_utils import resolve_path
from scripts.train.training_data_utils import (
    build_tokenized_supervised_example,
    find_think_content_spans,
    render_chat_text,
    token_indices_overlapping_span,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check Qwen3 format compatibility and think-content masking.")
    parser.add_argument("--tokenizer-path", default="runs/_hf_cache/models/Qwen3-0.6B")
    parser.add_argument("--data-dir", default="datasets/processed")
    parser.add_argument("--dataset-config", default="configs/datasets/stage2_search_fixed_10k.yaml")
    parser.add_argument("--smoke-config", default="configs/datasets/smoke_stage1.yaml")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--samples-per-dataset", type=int, default=20)
    parser.add_argument("--samples-per-group", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260402)
    parser.add_argument("--output", default="runs/final_mask_format_check.json")
    return parser


def has_user_prefix(messages: list[dict[str, Any]], upto_idx: int) -> bool:
    return any(message.get("role") == "user" for message in messages[: upto_idx + 1])


def check_row(tokenizer: Any, row: dict[str, Any], max_length: int) -> dict[str, Any]:
    rendered = render_chat_text(tokenizer, row["messages"], row.get("tools"))
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    built = build_tokenized_supervised_example(tokenizer, row, max_length=max_length)
    completion_mask = built["completion_mask"]
    messages = row["messages"]

    prev_full_len = 0
    think_checks: list[dict[str, Any]] = []
    any_answer_supervised = False

    for index, message in enumerate(messages):
        prefix = messages[: index + 1]
        if not has_user_prefix(messages, index):
            continue

        current_full = len(
            tokenizer(
                render_chat_text(tokenizer, prefix, row.get("tools")),
                add_special_tokens=False,
            )["input_ids"]
        )

        if message.get("role") != "assistant":
            prev_full_len = current_full
            continue

        start_tok = prev_full_len
        end_tok = min(current_full, len(completion_mask))
        if start_tok >= len(completion_mask):
            prev_full_len = current_full
            continue

        if any(mask == 1 for mask in completion_mask[start_tok:end_tok]):
            any_answer_supervised = True

        non_empty = [(start, end) for start, end in offsets[start_tok:end_tok] if end > start]
        if not non_empty:
            prev_full_len = current_full
            continue

        assistant_char_start, assistant_char_end = non_empty[0][0], non_empty[-1][1]
        assistant_text = rendered[assistant_char_start:assistant_char_end]
        for think_start, think_end in find_think_content_spans(assistant_text):
            abs_start = assistant_char_start + think_start
            abs_end = assistant_char_start + think_end
            token_indices = token_indices_overlapping_span(offsets, abs_start, abs_end)
            if token_indices:
                think_checks.append(
                    {
                        "token_count": len(token_indices),
                        "all_masked": all(completion_mask[token_index] == 0 for token_index in token_indices),
                    }
                )

        prev_full_len = current_full

    return {
        "seq_len": built["seq_len"],
        "supervised_tokens": built["supervised_tokens"],
        "has_nonempty_supervision": built["supervised_tokens"] > 0,
        "has_any_think_spans": bool(think_checks),
        "all_think_spans_masked": all(item["all_masked"] for item in think_checks) if think_checks else None,
        "assistant_has_supervised_tokens": any_answer_supervised,
    }


def summarize_checks(checks: list[dict[str, Any]]) -> dict[str, Any]:
    think_rows = [check for check in checks if check["has_any_think_spans"]]
    return {
        "sample_count": len(checks),
        "nonempty_supervision_count": sum(check["has_nonempty_supervision"] for check in checks),
        "assistant_supervision_count": sum(check["assistant_has_supervised_tokens"] for check in checks),
        "rows_with_think": len(think_rows),
        "rows_with_all_think_spans_masked": sum(
            check["all_think_spans_masked"] is True for check in think_rows
        ),
        "min_supervised_tokens": min(check["supervised_tokens"] for check in checks),
        "max_supervised_tokens": max(check["supervised_tokens"] for check in checks),
        "avg_supervised_tokens": round(
            sum(check["supervised_tokens"] for check in checks) / len(checks),
            2,
        ),
    }


def main() -> None:
    args = build_parser().parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    data_dir = resolve_path(args.data_dir)

    dataset_cfg = load_dataset_config(args.dataset_config)
    all_rows = load_named_datasets(data_dir, dataset_cfg)

    summary: dict[str, Any] = {"datasets": {}, "groups": {}}
    for dataset_name, rows in all_rows.items():
        rng = random.Random(f"dataset::{dataset_name}::{args.seed}")
        sample_indices = sorted(rng.sample(range(len(rows)), min(args.samples_per_dataset, len(rows))))
        checks = [check_row(tokenizer, rows[index], args.max_length) for index in sample_indices]
        summary["datasets"][dataset_name] = summarize_checks(checks)

    smoke_cfg = load_dataset_config(args.smoke_config)
    smoke_rows = load_named_datasets(data_dir, smoke_cfg)
    for group in sorted(smoke_cfg["recipes"].keys()):
        payload = build_splits(smoke_cfg, smoke_rows, group)
        train_rows = payload["train_rows"]
        rng = random.Random(f"group::{group}::{args.seed}")
        sample_indices = sorted(rng.sample(range(len(train_rows)), min(args.samples_per_group, len(train_rows))))
        checks = [check_row(tokenizer, train_rows[index], args.max_length) for index in sample_indices]
        group_summary = summarize_checks(checks)
        group_summary["recipe_components"] = payload["recipe"]["components"]
        summary["groups"][group] = group_summary

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
