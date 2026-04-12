#!/usr/bin/env python3
from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
import json
import random
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from scripts.train.training_data_utils import (
    THINK_CLOSE,
    THINK_OPEN,
    build_tokenized_supervised_example,
    find_think_content_spans,
    render_chat_text,
    token_indices_overlapping_span,
)
from scripts.common.runtime_utils import load_yaml


DEFAULT_TOKENIZER = "auto"
DEFAULT_DATA_DIR = "datasets/processed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run targeted training-readiness checks.")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--samples-per-dataset", type=int, default=10)
    parser.add_argument("--output", default="datasets/processed/training_readiness_summary.json")
    parser.add_argument("--hf-config", default="configs/hf/default.yaml")
    return parser


def load_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_tokenizer_source(tokenizer_arg: str, hf_config_path: str) -> str:
    if tokenizer_arg != "auto":
        return tokenizer_arg
    local_snapshot = Path("datasets/raw/step35_partial/tokenizers/qwen3")
    if local_snapshot.exists():
        return str(local_snapshot)
    hf_config = load_yaml(hf_config_path)
    return hf_config.get("base_model_id", "Qwen/Qwen3-8B")


def token_len(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _char_span_for_token_slice(
    offsets: list[tuple[int, int]],
    start_index: int,
    end_index: int,
) -> tuple[int, int] | None:
    non_empty = [(start, end) for start, end in offsets[start_index:end_index] if end > start]
    if not non_empty:
        return None
    return non_empty[0][0], non_empty[-1][1]


def _has_user_prefix(messages: list[dict[str, Any]], upto_idx: int) -> bool:
    return any(message.get("role") == "user" for message in messages[: upto_idx + 1])


def assert_synthetic_think_mask(tokenizer: Any) -> dict[str, Any]:
    example = {
        "messages": [
            {"role": "user", "content": "Please help."},
            {"role": "assistant", "content": "<think>\nsecret planning\n</think>\n\nVisible answer."},
        ],
        "tools": None,
    }
    rendered = render_chat_text(tokenizer, example["messages"], example["tools"])
    built = build_tokenized_supervised_example(tokenizer, example, max_length=4096)
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        truncation=True,
        max_length=4096,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    completion_mask = built["completion_mask"]

    previous_full_len = 0
    think_token_indices: list[int] = []
    think_tag_token_indices: list[int] = []
    answer_token_indices: list[int] = []

    for index, message in enumerate(example["messages"]):
        if not _has_user_prefix(example["messages"], index):
            continue
        current_full_len = token_len(
            tokenizer,
            render_chat_text(tokenizer, example["messages"][: index + 1], example["tools"]),
        )
        if message.get("role") != "assistant":
            previous_full_len = current_full_len
            continue

        assistant_start = previous_full_len
        assistant_end = min(current_full_len, len(completion_mask))
        assistant_char_span = _char_span_for_token_slice(offsets, assistant_start, assistant_end)
        if assistant_char_span is None:
            previous_full_len = current_full_len
            continue

        assistant_char_start, assistant_char_end = assistant_char_span
        assistant_text = rendered[assistant_char_start:assistant_char_end]

        for think_start, think_end in find_think_content_spans(assistant_text):
            abs_start = assistant_char_start + think_start
            abs_end = assistant_char_start + think_end
            think_token_indices.extend(token_indices_overlapping_span(offsets, abs_start, abs_end))

        open_tag_start = assistant_text.find(THINK_OPEN)
        if open_tag_start != -1:
            think_tag_token_indices.extend(
                token_indices_overlapping_span(
                    offsets,
                    assistant_char_start + open_tag_start,
                    assistant_char_start + open_tag_start + len(THINK_OPEN),
                )
            )

        close_tag_start = assistant_text.find(THINK_CLOSE)
        if close_tag_start != -1:
            think_tag_token_indices.extend(
                token_indices_overlapping_span(
                    offsets,
                    assistant_char_start + close_tag_start,
                    assistant_char_start + close_tag_start + len(THINK_CLOSE),
                )
            )
            answer_token_indices.extend(
                token_indices_overlapping_span(
                    offsets,
                    assistant_char_start + close_tag_start + len(THINK_CLOSE),
                    assistant_char_end,
                )
            )
        else:
            answer_token_indices.extend(
                token_indices_overlapping_span(offsets, assistant_char_start, assistant_char_end)
            )

        previous_full_len = current_full_len

    think_content_all_masked = bool(think_token_indices) and all(
        completion_mask[index] == 0 for index in think_token_indices
    )
    think_tag_supervised = any(completion_mask[index] == 1 for index in think_tag_token_indices)
    answer_supervised = any(completion_mask[index] == 1 for index in answer_token_indices)
    return {
        "rendered_preview": rendered[:400],
        "think_content_all_masked": think_content_all_masked,
        "think_tag_supervised": think_tag_supervised,
        "answer_supervised": answer_supervised,
        "seq_len": built["seq_len"],
        "supervised_tokens": built["supervised_tokens"],
        "probe_method": "offset_span_on_rendered_chat",
    }


def review_dataset_samples(
    tokenizer: Any,
    path: Path,
    sample_count: int,
) -> dict[str, Any]:
    rows = load_rows(path)
    rng = random.Random(f"training-readiness::{path.name}")
    sample_indices = rng.sample(range(len(rows)), min(sample_count, len(rows)))
    checked = []
    failures = []

    for index in sample_indices:
        row = rows[index]
        try:
            built = build_tokenized_supervised_example(tokenizer, row, max_length=8192)
            checked.append(
                {
                    "index": index,
                    "terminal_role": row["messages"][-1]["role"],
                    "assistant_turns": sum(message["role"] == "assistant" for message in row["messages"]),
                    "seq_len": built["seq_len"],
                    "supervised_tokens": built["supervised_tokens"],
                    "has_tool_turn": any(message["role"] == "tool" for message in row["messages"]),
                }
            )
        except Exception as exc:
            failures.append({"index": index, "error": repr(exc)})

    return {
        "row_count": len(rows),
        "sample_count": len(sample_indices),
        "checked": checked,
        "failures": failures,
        "all_passed": not failures,
    }


def main() -> None:
    args = build_parser().parse_args()
    tokenizer_source = resolve_tokenizer_source(args.tokenizer, args.hf_config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token

    data_dir = Path(args.data_dir)
    dataset_files = {
        "hermes": data_dir / "hermes_reasoning_tool_use_messages.json",
        "xlam": data_dir / "xlam_function_calling_60k_messages.json",
        "step20k": data_dir / "step35_clean_general_qwen3lenle8192_candidate_20000_messages.json",
        "qwen35_train": data_dir / "qwen3_5_toolcalling_v2_train_messages.json",
        "glaive": data_dir / "glaive_function_calling_v2_messages.json",
    }

    summary = {
        "tokenizer_source": tokenizer_source,
        "synthetic_think_mask": assert_synthetic_think_mask(tokenizer),
        "datasets": {},
    }

    for name, path in dataset_files.items():
        if path.exists():
            summary["datasets"][name] = review_dataset_samples(tokenizer, path, args.samples_per_dataset)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
