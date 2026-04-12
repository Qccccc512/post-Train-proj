#!/usr/bin/env python3
from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import ast
import json
import random
import re
import shutil
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Iterator

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

from scripts.common.runtime_utils import DEFAULT_EMPTY_THINK_PREFIX, ensure_dir, load_yaml, resolve_path, write_json
from scripts.train.training_data_utils import render_chat_text

try:  # pragma: no cover - optional acceleration
    import ijson
except Exception:  # pragma: no cover
    ijson = None

try:  # pragma: no cover - optional acceleration
    import orjson
except Exception:  # pragma: no cover
    orjson = None


RAW_ROOT_DEFAULT = "datasets/raw"
PROCESSED_ROOT_DEFAULT = "datasets/processed"
STEP_REPO_ID_DEFAULT = "stepfun-ai/Step-3.5-Flash-SFT"
STEP_RAW_SUBDIR = "step35_partial"
STEP_TOKENIZER_REL = "tokenizers/qwen3"
STEP_TOKENIZER_FALLBACK_MODEL = "Qwen/Qwen3-8B"
STEP20K_CANDIDATE_FILENAME = "step35_clean_general_qwen3lenle8192_candidate_20000.jsonl"
STEP20K_MANIFEST_FILENAME = "step35_length_filtered_candidates_manifest_le8192.jsonl"
STEP20K_SUMMARY_FILENAME = "step35_sampling_summary.json"
STEP_TOOLCALL_ALL_MESSAGES_FILENAME = "step35_toolcall_qwen3lenle8192_all_messages.json"
STEP_TOOLCALL_ALL_SUMMARY_FILENAME = "step35_toolcall_qwen3lenle8192_all_summary.json"
STEP_SAMPLE_DEFAULT_SHARDS = [
    "json/general/chunk_0.json",
    "json/general/chunk_4.json",
    "json/general/chunk_47.json",
    "json/general/chunk_31.json",
    "json/general/chunk_63.json",
    "json/general/chunk_5.json",
    "json/general/chunk_29.json",
    "json/general/chunk_18.json",
    "json/general/chunk_79.json",
    "json/general/chunk_56.json",
]

TURN_PATTERN = re.compile(r"(USER|ASSISTANT|FUNCTION RESPONSE):")
SYSTEM_PREFIX = "SYSTEM:"
END_OF_TEXT = "<|endoftext|>"
STEP_CHUNK_RE = re.compile(r"chunk_(\d+)\.json$")

STANDARD_DATASET_CHOICES = ["hermes", "xlam", "qwen35_train", "qwen35_test", "step20k", "glaive"]
STANDARD_OUTPUT_FILES = {
    "hermes": "hermes_reasoning_tool_use_messages.json",
    "xlam": "xlam_function_calling_60k_messages.json",
    "qwen35_train": "qwen3_5_toolcalling_v2_train_messages.json",
    "qwen35_test": "qwen3_5_toolcalling_v2_test_messages.json",
    "step20k": "step35_clean_general_qwen3lenle8192_candidate_20000_messages.json",
    "glaive": "glaive_function_calling_v2_messages.json",
}
TRANSFORM_CHOICES = [
    "drop_think_mismatch",
    "drop_nonassistant_terminal",
    "normalize_assistant_think_prefix",
]
STANDARD_TRANSFORMS = [
    "drop_think_mismatch",
    "drop_nonassistant_terminal",
    "normalize_assistant_think_prefix",
]

_TOKENIZER_TLS = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified dataset preparation entrypoint for standard processed datasets and StepFun tool-call extraction."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_standard = subparsers.add_parser(
        "build-standard",
        help="Build the standard processed training datasets from raw sources with configurable row transforms.",
    )
    build_standard.add_argument("--raw-root", default=RAW_ROOT_DEFAULT)
    build_standard.add_argument("--output-dir", default=PROCESSED_ROOT_DEFAULT)
    build_standard.add_argument(
        "--datasets",
        nargs="*",
        default=STANDARD_DATASET_CHOICES,
        choices=STANDARD_DATASET_CHOICES,
    )
    build_standard.add_argument(
        "--transforms",
        nargs="*",
        default=STANDARD_TRANSFORMS,
        choices=TRANSFORM_CHOICES,
        help="Transforms are applied in the listed order.",
    )
    build_standard.add_argument(
        "--summary-name",
        default="normalization_summary.json",
        help="Summary JSON filename written under --output-dir.",
    )

    download_step = subparsers.add_parser(
        "download-step-shards",
        help="Download additional Step-3.5-Flash-SFT general shards into datasets/raw.",
    )
    download_step.add_argument("--repo-id", default=STEP_REPO_ID_DEFAULT)
    download_step.add_argument("--raw-root", default=RAW_ROOT_DEFAULT)
    download_step.add_argument("--max-shards", type=int, default=None)
    download_step.add_argument("--include-tokenizer", action="store_true")
    download_step.add_argument("--max-workers", type=int, default=4)

    sample_step = subparsers.add_parser(
        "sample-step-clean-general",
        help="Build the clean-general Step sample used for the 20k no-tool subset under a Qwen3 length filter.",
    )
    sample_step.add_argument("--raw-root", default=RAW_ROOT_DEFAULT)
    sample_step.add_argument("--output-dir", default=PROCESSED_ROOT_DEFAULT)
    sample_step.add_argument(
        "--cache-dir",
        default=".hf_cache",
        help="Retained for compatibility; local JSON scanning does not require a datasets cache.",
    )
    sample_step.add_argument(
        "--tokenizer-source",
        default=None,
        help="Defaults to datasets/raw/step35_partial/tokenizers/qwen3 if present, otherwise Qwen/Qwen3-8B.",
    )
    sample_step.add_argument("--threshold", type=int, default=8192)
    sample_step.add_argument("--fallback-threshold", type=int, default=16384)
    sample_step.add_argument("--sample-size", type=int, default=20000)
    sample_step.add_argument("--max-workers", type=int, default=12)
    sample_step.add_argument("--sample-seed", type=int, default=20260401)
    sample_step.add_argument("--only-shards", nargs="*", default=None)
    sample_step.add_argument(
        "--all-local-shards",
        action="store_true",
        help="Use all locally downloaded general shards instead of the historical 10-shard baseline.",
    )
    sample_step.add_argument("--stats-only", action="store_true")
    sample_step.add_argument("--summary-output", default=None)
    sample_step.add_argument("--combined-output", default=None)
    sample_step.add_argument("--manifest-output", default=None)

    extract_step = subparsers.add_parser(
        "extract-step-toolcall",
        help="Extract StepFun tool-call rows into project-ready messages+tools format.",
    )
    extract_step.add_argument("--repo-id", default=STEP_REPO_ID_DEFAULT)
    extract_step.add_argument("--raw-root", default=RAW_ROOT_DEFAULT)
    extract_step.add_argument(
        "--output",
        default=f"{PROCESSED_ROOT_DEFAULT}/{STEP_TOOLCALL_ALL_MESSAGES_FILENAME}",
    )
    extract_step.add_argument(
        "--summary-output",
        default=f"{PROCESSED_ROOT_DEFAULT}/{STEP_TOOLCALL_ALL_SUMMARY_FILENAME}",
    )
    extract_step.add_argument("--max-token-length", type=int, default=8192)
    extract_step.add_argument(
        "--target-rows",
        type=int,
        default=0,
        help="Maximum number of accepted rows to keep. Use 0 to keep all accepted rows.",
    )
    extract_step.add_argument("--max-workers", type=int, default=12)
    extract_step.add_argument("--auto-download", action="store_true", default=True)
    extract_step.add_argument("--no-auto-download", dest="auto_download", action="store_false")
    extract_step.add_argument("--max-shards", type=int, default=None)
    extract_step.add_argument(
        "--mode",
        choices=["any", "complete"],
        default="any",
        help="any = any tool-related signal; complete = tools + assistant tool_calls + tool outputs.",
    )
    extract_step.add_argument(
        "--tokenizer-source",
        default=None,
        help="Defaults to datasets/raw/step35_partial/tokenizers/qwen3 if present, otherwise Qwen/Qwen3-8B.",
    )

    return parser.parse_args()


def fast_load_json(path: str | Path) -> Any:
    path_obj = resolve_path(path)
    data = path_obj.read_bytes()
    if orjson is not None:  # pragma: no branch
        return orjson.loads(data)
    return json.loads(data.decode("utf-8"))


def write_json_array(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path_obj = resolve_path(path)
    ensure_dir(path_obj.parent)
    count = 0
    with path_obj.open("wb") as handle:
        handle.write(b"[\n")
        first = True
        for row in rows:
            if not first:
                handle.write(b",\n")
            payload = (
                orjson.dumps(row)
                if orjson is not None
                else json.dumps(row, ensure_ascii=False).encode("utf-8")
            )
            handle.write(payload)
            first = False
            count += 1
        handle.write(b"\n]\n")
    return count


def iter_json_array(path: str | Path) -> Iterator[dict[str, Any]]:
    path_obj = resolve_path(path)
    if ijson is not None:
        with path_obj.open("rb") as handle:
            yield from ijson.items(handle, "item")
        return
    for row in fast_load_json(path_obj):
        yield row


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path_obj = resolve_path(path)
    with path_obj.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def chunk_sort_key(path_like: str | Path) -> tuple[int, str]:
    name = Path(path_like).name
    match = STEP_CHUNK_RE.search(name)
    if match:
        return int(match.group(1)), name
    return 10**9, name


def step_raw_root(raw_root: str | Path) -> Path:
    return resolve_path(raw_root) / STEP_RAW_SUBDIR


def step_tokenizer_default(raw_root: str | Path) -> Path:
    return step_raw_root(raw_root) / STEP_TOKENIZER_REL


def processed_root(path_like: str | Path = PROCESSED_ROOT_DEFAULT) -> Path:
    return resolve_path(path_like)


def step20k_candidate_default_path() -> Path:
    return processed_root() / STEP20K_CANDIDATE_FILENAME


def step20k_manifest_default_path() -> Path:
    return processed_root() / STEP20K_MANIFEST_FILENAME


def step20k_summary_default_path() -> Path:
    return processed_root() / STEP20K_SUMMARY_FILENAME


def resolve_step_tokenizer_source(raw_root: str | Path, tokenizer_source: str | None) -> str:
    if tokenizer_source:
        return tokenizer_source
    local_snapshot = step_tokenizer_default(raw_root)
    if local_snapshot.exists():
        return str(local_snapshot)
    return STEP_TOKENIZER_FALLBACK_MODEL


def normalize_role(role: str | None) -> str:
    role_map = {
        "human": "user",
        "gpt": "assistant",
        "answer": "assistant",
        "reasoning": "assistant",
        "tool_call": "assistant",
        "tool_output": "tool",
    }
    if role is None:
        return "user"
    return role_map.get(role, role)


def normalize_message(role: str | None, content: object) -> dict[str, str]:
    return {
        "role": normalize_role(role),
        "content": "" if content is None else str(content),
    }


def merge_reasoning_content(content: object, reasoning_content: object) -> str:
    answer = "" if content is None else str(content)
    reasoning = "" if reasoning_content is None else str(reasoning_content)
    if reasoning and answer:
        return f"<think>\n{reasoning}\n</think>\n\n{answer}"
    if reasoning:
        return f"<think>\n{reasoning}\n</think>"
    return answer


def compact_json(value: object) -> str:
    value = to_json_compatible(value)
    if orjson is not None:
        return orjson.dumps(value).decode("utf-8")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def safe_json_loads(value: object) -> object:
    if value is None or not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def to_json_compatible(value: Any) -> Any:
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    if isinstance(value, dict):
        return {str(key): to_json_compatible(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [to_json_compatible(item) for item in value]
    return value


def render_tool_calls(tool_calls: Iterable[dict[str, Any]]) -> str:
    rendered = []
    for call in tool_calls:
        payload = {
            "name": call.get("name"),
            "arguments": call.get("arguments", {}),
        }
        rendered.append(f"<tool_call>\n{compact_json(payload)}\n</tool_call>")
    return "\n".join(rendered)


def row_has_assistant_think_mismatch(row: dict[str, Any]) -> bool:
    for message in row.get("messages", []):
        if message.get("role") != "assistant":
            continue
        content = message.get("content") or ""
        if content.count("<think>") != content.count("</think>"):
            return True
    return False


def terminal_role(row: dict[str, Any]) -> str:
    if not row.get("messages"):
        return "__empty__"
    return row["messages"][-1]["role"]


def normalize_assistant_content(content: str) -> tuple[str, bool]:
    stripped = content.lstrip()
    if stripped.startswith("<think>"):
        if stripped == content:
            return content, False
        return stripped, True
    return DEFAULT_EMPTY_THINK_PREFIX + stripped, True


def apply_row_transforms(
    row: dict[str, Any],
    transforms: list[str],
    counters: dict[str, int],
) -> dict[str, Any] | None:
    transformed = row
    for transform in transforms:
        if transform == "drop_think_mismatch":
            if row_has_assistant_think_mismatch(transformed):
                counters["dropped_think_mismatch"] += 1
                return None
        elif transform == "drop_nonassistant_terminal":
            if terminal_role(transformed) != "assistant":
                counters["dropped_nonassistant_terminal"] += 1
                return None
        elif transform == "normalize_assistant_think_prefix":
            row_changed = False
            for message in transformed.get("messages", []):
                if message.get("role") != "assistant":
                    continue
                counters["assistant_messages_seen"] += 1
                normalized, changed = normalize_assistant_content(message.get("content") or "")
                if changed:
                    message["content"] = normalized
                    counters["assistant_messages_normalized"] += 1
                    row_changed = True
            if row_changed:
                counters["rows_touched_by_normalize_assistant_think_prefix"] += 1
        else:  # pragma: no cover
            raise ValueError(f"Unsupported transform: {transform}")
    return transformed


def iter_hermes(raw_root: Path) -> Iterator[dict[str, Any]]:
    dataset = load_dataset(
        "parquet",
        data_files=str(raw_root / "hermes_reasoning_tool_use" / "data" / "train-00000-of-00001.parquet"),
        split="train",
    )
    for row in dataset:
        yield {
            "messages": [
                normalize_message(message.get("from"), message.get("value"))
                for message in row["conversations"]
            ],
            "tools": None,
            "source_dataset": "hermes_reasoning_tool_use",
            "task": row.get("task"),
            "category": row.get("category"),
            "source": row.get("source"),
            "scenario_category": row.get("scenario_category"),
        }


def iter_xlam(raw_root: Path) -> Iterator[dict[str, Any]]:
    xlam_path = raw_root / "xlam-function-calling-60k" / "xlam_function_calling_60k.json"
    rows = fast_load_json(xlam_path)
    for row in rows:
        tools = safe_json_loads(row.get("tools")) or []
        answers = safe_json_loads(row.get("answers")) or []
        yield {
            "messages": [
                {"role": "user", "content": row.get("query", "")},
                {"role": "assistant", "content": render_tool_calls(answers)},
            ],
            "tools": tools or None,
            "source_dataset": "xlam-function-calling-60k",
            "source_id": row.get("id"),
            "tool_calls": answers,
        }


def iter_qwen35(raw_root: Path, split: str) -> Iterator[dict[str, Any]]:
    split_files = {
        "train": [
            str(raw_root / "qwen3.5-toolcalling-v2" / "data" / "train-00000-of-00002.parquet"),
            str(raw_root / "qwen3.5-toolcalling-v2" / "data" / "train-00001-of-00002.parquet"),
        ],
        "test": [
            str(raw_root / "qwen3.5-toolcalling-v2" / "data" / "test-00000-of-00001.parquet"),
        ],
    }
    dataset = load_dataset(
        "parquet",
        data_files={split: split_files[split]},
        split=split,
    )
    for row in dataset:
        yield {
            "messages": [
                normalize_message(message.get("role"), message.get("content"))
                for message in row["messages"]
            ],
            "tools": None,
            "source_dataset": "qwen3.5-toolcalling-v2",
            "split": split,
        }


def iter_step20k(raw_root: Path) -> Iterator[dict[str, Any]]:
    step_path_candidates = [
        step20k_candidate_default_path(),
        raw_root / STEP_RAW_SUBDIR / STEP20K_CANDIDATE_FILENAME,
    ]
    step_path = next((path for path in step_path_candidates if path.exists()), None)
    if step_path is None:
        raise FileNotFoundError(
            "Step clean-general candidate file not found. "
            f"Checked: {', '.join(str(path) for path in step_path_candidates)}"
        )
    for row in iter_jsonl(step_path):
        yield {
            "messages": [
                {
                    "role": normalize_role(message.get("role")),
                    "content": merge_reasoning_content(
                        message.get("content"),
                        message.get("reasoning_content"),
                    ),
                }
                for message in row["conversations"]
            ],
            "tools": None,
            "source_dataset": "step35_partial_general",
            "source_shard": row.get("_source_shard"),
            "source_index": row.get("_source_index"),
            "sample_seed": row.get("_sample_seed"),
            "length_threshold": row.get("_length_threshold"),
            "qwen3_token_len": row.get("_qwen3_token_len"),
        }


def clean_text(value: str) -> str:
    return value.replace(END_OF_TEXT, "").strip()


def parse_turn_blocks(chat: str) -> list[tuple[str, str]]:
    matches = list(TURN_PATTERN.finditer(chat))
    blocks: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        role = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(chat)
        content = clean_text(chat[start:end])
        if content:
            blocks.append((role, content))
    return blocks


def parse_functioncall_payload(raw_content: str) -> tuple[str, dict[str, Any] | None]:
    if "<functioncall>" not in raw_content:
        return raw_content, None
    before, after = raw_content.split("<functioncall>", 1)
    preamble = before.strip()
    payload_text = after.strip()
    parsed: dict[str, Any] | None = None
    try:
        candidate = ast.literal_eval(payload_text)
        if isinstance(candidate, dict):
            parsed = dict(candidate)
    except Exception:
        parsed = None
    if parsed and "arguments" in parsed and isinstance(parsed["arguments"], str):
        try:
            parsed["arguments"] = json.loads(parsed["arguments"])
        except Exception:
            pass
    return preamble, parsed


def convert_assistant_content_for_glaive(raw_content: str) -> tuple[str, bool, bool]:
    cleaned = clean_text(raw_content)
    preamble, parsed_call = parse_functioncall_payload(cleaned)
    if parsed_call is None:
        if cleaned.lstrip().startswith("<think>"):
            return cleaned.strip(), False, False
        return f"{DEFAULT_EMPTY_THINK_PREFIX}{cleaned}".strip(), False, False
    call_payload = compact_json(parsed_call)
    parts = [DEFAULT_EMPTY_THINK_PREFIX.rstrip()]
    if preamble:
        parts.append(preamble)
    parts.append(f"<tool_call>\n{call_payload}\n</tool_call>")
    return "\n\n".join(parts).strip(), True, bool(preamble)


def iter_glaive(raw_root: Path) -> Iterator[dict[str, Any]]:
    input_path = raw_root / "glaive-function-calling-v2" / "glaive-function-calling-v2.json"
    raw_rows = fast_load_json(input_path)
    for source_index, row in enumerate(raw_rows):
        system_content = row["system"].strip()
        if system_content.startswith(SYSTEM_PREFIX):
            system_content = system_content[len(SYSTEM_PREFIX) :].strip()
        messages = [{"role": "system", "content": system_content}]
        for role, content in parse_turn_blocks(row["chat"]):
            if role == "USER":
                messages.append({"role": "user", "content": content})
            elif role == "FUNCTION RESPONSE":
                messages.append({"role": "tool", "content": content})
            elif role == "ASSISTANT":
                converted, _has_tool_call, _has_preamble = convert_assistant_content_for_glaive(content)
                messages.append({"role": "assistant", "content": converted})
            else:  # pragma: no cover
                raise ValueError(f"Unexpected role: {role}")
        yield {
            "messages": messages,
            "tools": None,
            "source_dataset": "glaive-function-calling-v2",
            "source_index": source_index,
        }


def iter_standard_dataset(dataset_name: str, raw_root: Path) -> Iterator[dict[str, Any]]:
    if dataset_name == "hermes":
        yield from iter_hermes(raw_root)
    elif dataset_name == "xlam":
        yield from iter_xlam(raw_root)
    elif dataset_name == "qwen35_train":
        yield from iter_qwen35(raw_root, "train")
    elif dataset_name == "qwen35_test":
        yield from iter_qwen35(raw_root, "test")
    elif dataset_name == "step20k":
        yield from iter_step20k(raw_root)
    elif dataset_name == "glaive":
        yield from iter_glaive(raw_root)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_standard_datasets(args: argparse.Namespace) -> None:
    raw_root = resolve_path(args.raw_root)
    output_dir = ensure_dir(args.output_dir)
    summary: dict[str, Any] = {
        "raw_root": str(raw_root),
        "output_dir": str(output_dir),
        "datasets": {},
        "transforms": args.transforms,
    }

    for dataset_name in args.datasets:
        output_path = output_dir / STANDARD_OUTPUT_FILES[dataset_name]
        counters = {
            "rows_before": 0,
            "rows_after": 0,
            "dropped_think_mismatch": 0,
            "dropped_nonassistant_terminal": 0,
            "rows_touched_by_normalize_assistant_think_prefix": 0,
            "assistant_messages_seen": 0,
            "assistant_messages_normalized": 0,
        }

        def row_iter() -> Iterator[dict[str, Any]]:
            for row in iter_standard_dataset(dataset_name, raw_root):
                counters["rows_before"] += 1
                transformed = apply_row_transforms(row, args.transforms, counters)
                if transformed is None:
                    continue
                counters["rows_after"] += 1
                yield transformed

        written_rows = write_json_array(output_path, row_iter())
        counters["rows_after"] = written_rows
        summary["datasets"][dataset_name] = {
            **counters,
            "output_path": str(output_path),
        }
        print(
            f"[build-standard] {dataset_name}: before={counters['rows_before']} after={counters['rows_after']}",
            flush=True,
        )

    summary_path = output_dir / args.summary_name
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def list_remote_step_shards(repo_id: str) -> list[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset")
    return sorted(
        [path for path in files if path.startswith("json/general/chunk_") and path.endswith(".json")],
        key=chunk_sort_key,
    )


def download_step_shard(repo_id: str, raw_root: Path, remote_path: str) -> Path:
    cached = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=remote_path)
    destination = step_raw_root(raw_root) / remote_path
    ensure_dir(destination.parent)
    if not destination.exists():
        shutil.copy2(cached, destination)
    return destination


def download_step_assets(args: argparse.Namespace) -> None:
    raw_root = resolve_path(args.raw_root)
    step_root = step_raw_root(raw_root)
    remote_shards = list_remote_step_shards(args.repo_id)
    local_rel = {
        str(path.relative_to(step_root))
        for path in (step_root / "json/general").glob("chunk_*.json")
    }
    to_download = [rel for rel in remote_shards if rel not in local_rel]
    if args.max_shards is not None:
        to_download = to_download[: args.max_shards]

    downloaded: list[str] = []
    if to_download:
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            futures = {
                executor.submit(download_step_shard, args.repo_id, raw_root, rel): rel
                for rel in to_download
            }
            for future in as_completed(futures):
                rel = futures[future]
                future.result()
                downloaded.append(rel)
                print(f"[download-step-shards] {rel}", flush=True)

    if args.include_tokenizer:
        tokenizer_files = [
            "README.md",
            f"{STEP_TOKENIZER_REL}/config.json",
            f"{STEP_TOKENIZER_REL}/merges.txt",
            f"{STEP_TOKENIZER_REL}/tokenizer.json",
            f"{STEP_TOKENIZER_REL}/tokenizer_config.json",
            f"{STEP_TOKENIZER_REL}/vocab.json",
        ]
        for rel in tokenizer_files:
            download_step_shard(args.repo_id, raw_root, rel)

    payload = {
        "repo_id": args.repo_id,
        "downloaded_count": len(downloaded),
        "downloaded": downloaded,
        "raw_root": str(raw_root),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def is_step_clean_general(conversation: list[dict[str, Any]]) -> bool:
    for message in conversation:
        if message.get("role") == "tool":
            return False
        if message.get("tool_calls"):
            return False
        if message.get("tool_call_id"):
            return False
        if message.get("tools"):
            return False
    return True


def step_conv_to_messages(conversation: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages = []
    for message in conversation:
        role = normalize_role(message.get("role"))
        content = merge_reasoning_content(
            message.get("content"),
            message.get("reasoning_content") if role == "assistant" else None,
        )
        messages.append({"role": role, "content": content})
    return messages


def process_clean_general_candidate(
    row: dict[str, Any],
    tokenizer_source: str,
    threshold: int,
    fallback_threshold: int,
) -> dict[str, Any]:
    tokenizer = get_thread_tokenizer(tokenizer_source)
    messages = step_conv_to_messages(row["conversations"])
    rendered = render_chat_text(tokenizer, messages, tools=None)
    token_length = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
    return {
        "token_len": token_length,
        "turns": len(messages),
        "le_threshold": token_length <= threshold,
        "le_fallback_threshold": token_length <= fallback_threshold,
    }


def default_step_clean_general_outputs(
    output_dir: str | Path,
    threshold: int,
    sample_size: int,
) -> tuple[Path, Path, Path]:
    base_dir = processed_root(output_dir)
    combined = base_dir / f"step35_clean_general_qwen3lenle{threshold}_candidate_{sample_size}.jsonl"
    summary = base_dir / STEP20K_SUMMARY_FILENAME
    manifest = base_dir / f"step35_length_filtered_candidates_manifest_le{threshold}.jsonl"
    return combined, summary, manifest


def iter_selected_step_rows(local_path: Path, selected_items: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    if not selected_items:
        return
    selected_by_index = {item["index"]: item for item in selected_items}
    for source_index, row in enumerate(iter_json_array(local_path)):
        selected = selected_by_index.get(source_index)
        if selected is None:
            continue
        record = to_json_compatible(row)
        record["_source_shard"] = selected["shard"]
        record["_source_index"] = source_index
        record["_sample_seed"] = selected["sample_seed"]
        record["_length_threshold"] = selected["length_threshold"]
        record["_qwen3_token_len"] = selected["token_len"]
        yield record


def pop_ready_candidates_in_source_order(
    candidate_order: deque[int],
    completed_results: dict[int, dict[str, Any] | None],
) -> list[tuple[int, dict[str, Any] | None]]:
    ready: list[tuple[int, dict[str, Any] | None]] = []
    while candidate_order and candidate_order[0] in completed_results:
        source_index = candidate_order.popleft()
        ready.append((source_index, completed_results.pop(source_index)))
    return ready


def sample_step_clean_general(args: argparse.Namespace) -> None:
    raw_root = resolve_path(args.raw_root)
    step_root = step_raw_root(raw_root)
    tokenizer_source = resolve_step_tokenizer_source(raw_root, args.tokenizer_source)

    if args.only_shards:
        shard_rel_paths = list(args.only_shards)
        shard_selection_strategy = "explicit_only_shards"
    elif args.all_local_shards:
        shard_rel_paths = [
            str(path.relative_to(step_root))
            for path in available_local_step_shards(raw_root)
        ]
        shard_selection_strategy = "all_local_shards"
    else:
        shard_rel_paths = list(STEP_SAMPLE_DEFAULT_SHARDS)
        shard_selection_strategy = "fixed_historical_baseline"

    combined_default, summary_default, manifest_default = default_step_clean_general_outputs(
        args.output_dir,
        args.threshold,
        args.sample_size,
    )
    combined_output = resolve_path(args.combined_output) if args.combined_output else combined_default
    summary_output = resolve_path(args.summary_output) if args.summary_output else summary_default
    manifest_output = resolve_path(args.manifest_output) if args.manifest_output else manifest_default
    ensure_dir(combined_output.parent)
    ensure_dir(summary_output.parent)
    ensure_dir(manifest_output.parent)

    all_candidates: list[dict[str, Any]] = []
    per_shard: list[dict[str, Any]] = []

    for shard_rel_path in shard_rel_paths:
        local_path = step_root / shard_rel_path
        if not local_path.exists():
            raise FileNotFoundError(f"Step shard not found: {local_path}")

        stats = {
            "shard": shard_rel_path,
            "path": str(local_path),
            "num_rows": 0,
            "clean_general_candidates_any_len": 0,
            "eligible_len_le_threshold": 0,
            "eligible_len_le_fallback_threshold": 0,
            "max_token_len_seen": 0,
            "max_turns_seen": 0,
        }
        shard_candidates: list[dict[str, Any]] = []
        pending: dict[Any, int] = {}

        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            for source_index, row in enumerate(iter_json_array(local_path)):
                stats["num_rows"] += 1
                conversation = row["conversations"]
                if not is_step_clean_general(conversation):
                    continue
                stats["clean_general_candidates_any_len"] += 1
                future = executor.submit(
                    process_clean_general_candidate,
                    row,
                    tokenizer_source,
                    args.threshold,
                    args.fallback_threshold,
                )
                pending[future] = source_index

                if len(pending) >= args.max_workers * 4:
                    done_future = next(as_completed(pending))
                    source_idx = pending.pop(done_future)
                    result = done_future.result()
                    stats["max_token_len_seen"] = max(stats["max_token_len_seen"], result["token_len"])
                    stats["max_turns_seen"] = max(stats["max_turns_seen"], result["turns"])
                    if result["le_fallback_threshold"]:
                        stats["eligible_len_le_fallback_threshold"] += 1
                    if result["le_threshold"]:
                        stats["eligible_len_le_threshold"] += 1
                        shard_candidates.append(
                            {
                                "shard": shard_rel_path,
                                "index": source_idx,
                                "token_len": result["token_len"],
                                "turns": result["turns"],
                            }
                        )

            for future in as_completed(list(pending.keys())):
                source_idx = pending[future]
                result = future.result()
                stats["max_token_len_seen"] = max(stats["max_token_len_seen"], result["token_len"])
                stats["max_turns_seen"] = max(stats["max_turns_seen"], result["turns"])
                if result["le_fallback_threshold"]:
                    stats["eligible_len_le_fallback_threshold"] += 1
                if result["le_threshold"]:
                    stats["eligible_len_le_threshold"] += 1
                    shard_candidates.append(
                        {
                            "shard": shard_rel_path,
                            "index": source_idx,
                            "token_len": result["token_len"],
                            "turns": result["turns"],
                        }
                    )

        shard_candidates.sort(key=lambda item: item["index"])
        all_candidates.extend(shard_candidates)
        per_shard.append(stats)
        print(
            f"[sample-step-clean-general] {local_path.name}: rows={stats['num_rows']} "
            f"clean_any={stats['clean_general_candidates_any_len']} "
            f"le{args.threshold}={stats['eligible_len_le_threshold']} "
            f"le{args.fallback_threshold}={stats['eligible_len_le_fallback_threshold']}",
            flush=True,
        )

    summary = {
        "sampling_recipe": "clean_general_no_tool_global_sample_with_qwen3_length_filter",
        "repo_id": STEP_REPO_ID_DEFAULT,
        "raw_root": str(raw_root),
        "tokenizer_source": tokenizer_source,
        "tokenizer_template_source": tokenizer_source,
        "shard_selection_strategy": shard_selection_strategy,
        "historical_baseline_shards": list(STEP_SAMPLE_DEFAULT_SHARDS),
        "global_sample_seed": args.sample_seed,
        "length_threshold_tokens": args.threshold,
        "fallback_length_threshold_tokens": args.fallback_threshold,
        "sample_size": args.sample_size,
        "max_workers": args.max_workers,
        "filters": {
            "no_tool_role": True,
            "no_tool_calls": True,
            "no_tool_call_id": True,
            "no_tools_field": True,
            "no_turn_limit": True,
        },
        "processed_shards": shard_rel_paths,
        "total_eligible_len_le_threshold": len(all_candidates),
        "stats_only": bool(args.stats_only),
        "shards": per_shard,
    }

    if args.stats_only:
        write_json(summary_output, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if len(all_candidates) < args.sample_size:
        raise SystemExit(
            f"Not enough <= {args.threshold} candidates: {len(all_candidates)} < {args.sample_size}. "
            f"Consider --all-local-shards or a higher threshold."
        )

    rng = random.Random(args.sample_seed)
    selected = rng.sample(all_candidates, args.sample_size)
    selected_lookup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        chosen = dict(item)
        chosen["sample_seed"] = args.sample_seed
        chosen["length_threshold"] = args.threshold
        selected_lookup[chosen["shard"]].append(chosen)
    for shard in selected_lookup:
        selected_lookup[shard].sort(key=lambda item: item["index"])

    with combined_output.open("w", encoding="utf-8") as handle:
        for shard_rel_path in shard_rel_paths:
            local_path = step_root / shard_rel_path
            for record in iter_selected_step_rows(local_path, selected_lookup.get(shard_rel_path, [])):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with manifest_output.open("w", encoding="utf-8") as handle:
        for item in all_candidates:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary["combined_output"] = str(combined_output)
    summary["candidates_manifest"] = str(manifest_output)
    summary["sampled_count"] = args.sample_size
    summary["sample_distribution"] = {
        shard: len(items) for shard, items in selected_lookup.items()
    }
    write_json(summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def step_signal_flags(conversation: list[dict[str, Any]]) -> dict[str, Any]:
    has_schema = False
    has_assistant_calls = False
    has_tool_role = False
    for message in conversation:
        if message.get("tools"):
            has_schema = True
        if message.get("tool_calls"):
            has_assistant_calls = True
        if message.get("role") == "tool" or message.get("tool_call_id"):
            has_tool_role = True
    return {
        "has_schema": has_schema,
        "has_assistant_calls": has_assistant_calls,
        "has_tool_role": has_tool_role,
    }


def should_keep_step_row(flags: dict[str, Any], mode: str) -> bool:
    if mode == "complete":
        return flags["has_schema"] and flags["has_assistant_calls"] and flags["has_tool_role"]
    return flags["has_schema"] or flags["has_assistant_calls"] or flags["has_tool_role"]


def extract_step_tools(conversation: list[dict[str, Any]]) -> Any:
    for message in conversation:
        if message.get("tools"):
            return to_json_compatible(message["tools"])
    return None


def convert_step_tool_row(row: dict[str, Any], shard_name: str, source_index: int) -> dict[str, Any]:
    conversation = row["conversations"]
    messages = []
    for message in conversation:
        role = normalize_role(message.get("role"))
        converted = {
            "role": role,
            "content": merge_reasoning_content(
                message.get("content"),
                message.get("reasoning_content") if role == "assistant" else None,
            ),
        }
        if message.get("name"):
            converted["name"] = to_json_compatible(message["name"])
        if message.get("tool_calls"):
            converted["tool_calls"] = to_json_compatible(message["tool_calls"])
        if message.get("tool_call_id"):
            converted["tool_call_id"] = to_json_compatible(message["tool_call_id"])
        messages.append(converted)
    return {
        "messages": messages,
        "tools": extract_step_tools(conversation),
        "source_dataset": "stepfun-ai/Step-3.5-Flash-SFT",
        "source_shard": shard_name,
        "source_index": source_index,
        "turn_count": len(messages),
    }


def get_thread_tokenizer(tokenizer_source: str) -> Any:
    tokenizer = getattr(_TOKENIZER_TLS, "tokenizer", None)
    tokenizer_path = getattr(_TOKENIZER_TLS, "tokenizer_source", None)
    if tokenizer is None or tokenizer_path != tokenizer_source:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        tokenizer.model_max_length = max(getattr(tokenizer, "model_max_length", 0) or 0, 10**9)
        _TOKENIZER_TLS.tokenizer = tokenizer
        _TOKENIZER_TLS.tokenizer_source = tokenizer_source
    return tokenizer


def process_step_candidate(
    row: dict[str, Any],
    shard_name: str,
    source_index: int,
    tokenizer_source: str,
    max_token_length: int,
) -> dict[str, Any] | None:
    converted = convert_step_tool_row(row, shard_name, source_index)
    tokenizer = get_thread_tokenizer(tokenizer_source)
    rendered = render_chat_text(tokenizer, converted["messages"], converted.get("tools"))
    token_length = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
    if token_length > max_token_length:
        return None
    converted["qwen3_token_len"] = token_length
    return converted


def available_local_step_shards(raw_root: Path) -> list[Path]:
    return sorted((step_raw_root(raw_root) / "json/general").glob("chunk_*.json"), key=chunk_sort_key)


def extract_step_toolcall_dataset(args: argparse.Namespace) -> None:
    raw_root = resolve_path(args.raw_root)
    step_root = step_raw_root(raw_root)
    tokenizer_source = resolve_step_tokenizer_source(raw_root, args.tokenizer_source)
    ensure_dir(step_root / "json/general")
    target_rows = args.target_rows if args.target_rows and args.target_rows > 0 else None

    if args.auto_download:
        remote_rel_paths = list_remote_step_shards(args.repo_id)
        if args.max_shards is not None:
            remote_rel_paths = remote_rel_paths[: args.max_shards]
    else:
        remote_rel_paths = [
            str(path.relative_to(step_root))
            for path in available_local_step_shards(raw_root)
        ]
        if args.max_shards is not None:
            remote_rel_paths = remote_rel_paths[: args.max_shards]

    output_path = resolve_path(args.output)
    summary_path = resolve_path(args.summary_output)
    summary = {
        "repo_id": args.repo_id,
        "mode": args.mode,
        "max_token_length": args.max_token_length,
        "target_rows": "all" if target_rows is None else target_rows,
        "processed_shards": [],
        "rows_scanned": 0,
        "rows_with_tool_signal": 0,
        "rows_accepted": 0,
        "local_step_root": str(step_root),
        "tokenizer_source": tokenizer_source,
        "per_shard_output_order": "source_index_ascending",
    }

    def row_iter() -> Iterator[dict[str, Any]]:
        stop_early = False
        for remote_rel_path in remote_rel_paths:
            if target_rows is not None and summary["rows_accepted"] >= target_rows:
                break

            local_path = step_root / remote_rel_path
            if not local_path.exists():
                if not args.auto_download:
                    continue
                print(f"[extract-step-toolcall] downloading {remote_rel_path}", flush=True)
                local_path = download_step_shard(args.repo_id, raw_root, remote_rel_path)

            shard_scanned = 0
            shard_tool_signal = 0
            shard_accepted = 0
            pending: dict[Any, int] = {}
            candidate_order: deque[int] = deque()
            completed_results: dict[int, dict[str, Any] | None] = {}

            with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
                for source_index, row in enumerate(iter_json_array(local_path)):
                    shard_scanned += 1
                    summary["rows_scanned"] += 1
                    flags = step_signal_flags(row["conversations"])
                    if not should_keep_step_row(flags, args.mode):
                        continue
                    shard_tool_signal += 1
                    summary["rows_with_tool_signal"] += 1
                    future = executor.submit(
                        process_step_candidate,
                        row,
                        local_path.name,
                        source_index,
                        tokenizer_source,
                        args.max_token_length,
                    )
                    pending[future] = source_index
                    candidate_order.append(source_index)

                    if len(pending) >= args.max_workers * 8:
                        done_future = next(as_completed(pending))
                        source_index_done = pending.pop(done_future)
                        completed_results[source_index_done] = done_future.result()
                        for _source_index, accepted in pop_ready_candidates_in_source_order(
                            candidate_order,
                            completed_results,
                        ):
                            if accepted is not None:
                                summary["rows_accepted"] += 1
                                shard_accepted += 1
                                yield accepted
                                if target_rows is not None and summary["rows_accepted"] >= target_rows:
                                    stop_early = True
                                    break
                        if stop_early:
                            break

                if not stop_early:
                    for future in as_completed(list(pending.keys())):
                        source_index_done = pending.pop(future)
                        completed_results[source_index_done] = future.result()
                        for _source_index, accepted in pop_ready_candidates_in_source_order(
                            candidate_order,
                            completed_results,
                        ):
                            if accepted is not None:
                                summary["rows_accepted"] += 1
                                shard_accepted += 1
                                yield accepted
                            if target_rows is not None and summary["rows_accepted"] >= target_rows:
                                stop_early = True
                                break
                        if target_rows is not None and summary["rows_accepted"] >= target_rows:
                            stop_early = True
                            break

            summary["processed_shards"].append(
                {
                    "shard": local_path.name,
                    "rows_scanned": shard_scanned,
                    "rows_with_tool_signal": shard_tool_signal,
                    "rows_accepted": shard_accepted,
                }
            )
            print(
                f"[extract-step-toolcall] {local_path.name}: scanned={shard_scanned} tool_signal={shard_tool_signal} accepted={shard_accepted} total={summary['rows_accepted']}",
                flush=True,
            )
            if stop_early:
                break

    written_rows = write_json_array(output_path, row_iter())
    summary["rows_accepted"] = written_rows
    summary["output_path"] = str(output_path)
    summary["summary_output"] = str(summary_path)
    summary["output_file_rows"] = written_rows
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "build-standard":
        build_standard_datasets(args)
    elif args.command == "download-step-shards":
        download_step_assets(args)
    elif args.command == "sample-step-clean-general":
        sample_step_clean_general(args)
    elif args.command == "extract-step-toolcall":
        extract_step_toolcall_dataset(args)
    else:  # pragma: no cover
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
