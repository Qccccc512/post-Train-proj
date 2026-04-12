#!/usr/bin/env python3
from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from unsloth import FastLanguageModel, is_bf16_supported
from peft import set_peft_model_state_dict
from peft.utils.save_and_load import load_peft_weights
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from transformers.trainer_utils import get_last_checkpoint

from scripts.train.build_dataset_splits import (
    build_splits,
    load_dataset_config,
    maybe_export_diagnostic_sets,
    resolve_dataset_source_dir,
)
from scripts.hub.hf_repo_sync import fetch_base_model_snapshot, load_hf_config, upload_run_artifacts
from scripts.common.runtime_utils import (
    collect_environment_info,
    copy_resolved_configs,
    ensure_dir,
    generate_run_name,
    now_timestamp,
    read_json,
    read_jsonl,
    resolve_path,
    run_output_dir,
    sanitize_name,
    write_json,
    write_jsonl,
)
from scripts.train.training_data_utils import build_tokenized_supervised_example


DEFAULT_DATASET_CONFIG = "configs/datasets/stage2_search_fixed_10k.yaml"
DEFAULT_TRAIN_CONFIG = "configs/train/stage2_search_lr1e4_r16_e1_ms500.yaml"
DEFAULT_HF_CONFIG = "configs/hf/default.yaml"
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Qwen/Qwen3 LoRA SFT runs on Colab or local.")
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--train-config", default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--hf-config", default=DEFAULT_HF_CONFIG)
    parser.add_argument("--group")
    parser.add_argument("--phase")
    parser.add_argument("--run-name")
    parser.add_argument("--output-dir")
    parser.add_argument("--force-local-datasets", action="store_true")
    parser.add_argument("--skip-auto-upload", action="store_true")
    parser.add_argument("--skip-auto-build", action="store_true")
    return parser


def append_console(log_path: Path, message: str) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_timestamp()}] {message}\n")


class _HideKnownTrainingLogNoise(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.msg if isinstance(record.msg, str) else ""
        if "The attention mask API under `transformers.modeling_attn_mask_utils`" in message:
            return False
        return True


def silence_known_training_log_noise() -> None:
    noise_filter = _HideKnownTrainingLogNoise()
    logging.getLogger("unsloth_zoo.log").addFilter(noise_filter)
    logging.getLogger("transformers.modeling_attn_mask_utils").addFilter(noise_filter)


def save_adapter_snapshot_from_checkpoint(
    *,
    snapshot_dir: Path,
    tokenizer_artifact: Any,
    trainer_model: Any,
    checkpoint_path: str | None,
    canonical_base_model_id: str,
    snapshot_label: str,
    console_log: Path,
    allow_fallback_to_current_model: bool,
) -> None:
    ensure_dir(snapshot_dir)
    checkpoint_dir = Path(checkpoint_path) if checkpoint_path else None
    label = snapshot_label.strip().lower() or "adapter"
    label_title = label.capitalize()

    if checkpoint_dir and checkpoint_dir.exists():
        peft_state = load_peft_weights(str(checkpoint_dir), device="cpu")
        set_peft_model_state_dict(trainer_model, peft_state)
        trainer_model.save_pretrained(snapshot_dir)
        tokenizer_artifact.save_pretrained(snapshot_dir)
        normalize_adapter_base_model_reference(snapshot_dir, canonical_base_model_id, console_log)
        append_console(
            console_log,
            f"Loaded {label} adapter weights from checkpoint and exported snapshot: {checkpoint_dir}.",
        )
        return

    if checkpoint_dir:
        append_console(console_log, f"{label_title} checkpoint path was unavailable for export: {checkpoint_dir}.")
    else:
        append_console(console_log, f"{label_title} checkpoint path was not recorded for export.")

    if not allow_fallback_to_current_model:
        raise FileNotFoundError(
            f"Unable to export {label} adapter snapshot because checkpoint_path={checkpoint_path!r} was unavailable."
        )

    try:
        trainer_model.save_pretrained(snapshot_dir)
        tokenizer_artifact.save_pretrained(snapshot_dir)
        normalize_adapter_base_model_reference(snapshot_dir, canonical_base_model_id, console_log)
        append_console(console_log, f"{label_title} snapshot export fell back to current in-memory model weights.")
    except Exception as exc:
        append_console(console_log, f"{label_title} snapshot export failed during fallback save: {exc}")
        raise


def normalize_adapter_base_model_reference(snapshot_dir: Path, canonical_base_model_id: str, console_log: Path) -> None:
    adapter_config_path = snapshot_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        append_console(console_log, f"Skipped adapter base model normalization because file is missing: {adapter_config_path}")
        return

    payload = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    current_value = payload.get("base_model_name_or_path")
    if current_value == canonical_base_model_id:
        return

    payload["base_model_name_or_path"] = canonical_base_model_id
    adapter_config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    append_console(
        console_log,
        "Normalized adapter_config base_model_name_or_path "
        f"from {current_value!r} to {canonical_base_model_id!r}.",
    )


def save_best_adapter_snapshot(
    *,
    best_dir: Path,
    tokenizer_artifact: Any,
    trainer_model: Any,
    best_checkpoint: str | None,
    canonical_base_model_id: str,
    console_log: Path,
) -> None:
    save_adapter_snapshot_from_checkpoint(
        snapshot_dir=best_dir,
        tokenizer_artifact=tokenizer_artifact,
        trainer_model=trainer_model,
        checkpoint_path=best_checkpoint,
        canonical_base_model_id=canonical_base_model_id,
        snapshot_label="best",
        console_log=console_log,
        allow_fallback_to_current_model=True,
    )


def save_final_adapter_snapshot(
    *,
    final_dir: Path,
    tokenizer_artifact: Any,
    trainer_model: Any,
    last_checkpoint: str | None,
    canonical_base_model_id: str,
    console_log: Path,
) -> None:
    save_adapter_snapshot_from_checkpoint(
        snapshot_dir=final_dir,
        tokenizer_artifact=tokenizer_artifact,
        trainer_model=trainer_model,
        checkpoint_path=last_checkpoint,
        canonical_base_model_id=canonical_base_model_id,
        snapshot_label="final",
        console_log=console_log,
        allow_fallback_to_current_model=False,
    )


def ensure_triton_allocator(console_log: Path) -> None:
    try:
        import triton
    except Exception as exc:
        append_console(console_log, f"Skipped Triton allocator init because triton import failed: {exc}")
        return

    if getattr(triton, "_post_train_allocator_set", False):
        append_console(console_log, "Triton allocator already initialized.")
        return

    if not hasattr(triton, "set_allocator"):
        append_console(console_log, "Skipped Triton allocator init because triton.set_allocator is unavailable.")
        return

    if not torch.cuda.is_available():
        append_console(console_log, "Skipped Triton allocator init because CUDA is unavailable.")
        return

    device = torch.device("cuda", torch.cuda.current_device())
    state: dict[str, torch.Tensor | None] = {"buffer": None}

    def persistent_alloc_fn(size: int, alignment: int, stream: Any) -> torch.Tensor:
        rounded_size = max(1, ((int(size) + 127) // 128) * 128)
        buffer = state["buffer"]
        if buffer is None or buffer.numel() * buffer.element_size() < rounded_size:
            buffer = torch.empty(int(rounded_size * 1.1), device=device, dtype=torch.uint8)
            state["buffer"] = buffer
        return buffer

    triton.set_allocator(persistent_alloc_fn)
    triton._post_train_allocator_set = True
    append_console(console_log, f"Initialized Triton allocator on device={device}.")


def ensure_dataset_splits(
    run_dir: Path,
    dataset_config: dict[str, Any],
    train_config: dict[str, Any],
    hf_config: dict[str, Any],
    group: str,
    phase: str,
    force_local_datasets: bool,
) -> tuple[Path, Path]:
    train_path = run_dir / "dataset" / "train.jsonl"
    val_path = run_dir / "dataset" / "val.jsonl"
    if train_path.exists() and val_path.exists():
        return train_path, val_path

    source_dir = resolve_dataset_source_dir(dataset_config, hf_config, force_local=force_local_datasets)
    named_datasets = {
        name: read_json(source_dir / entry["filename"])
        for name, entry in dataset_config["datasets"].items()
    }
    split_payload = build_splits(dataset_config, named_datasets, group)

    ensure_dir(train_path.parent)
    write_jsonl(train_path, split_payload["train_rows"])
    write_jsonl(val_path, split_payload["val_rows"])
    diagnostics = maybe_export_diagnostic_sets(run_dir, source_dir, dataset_config)
    manifest = {
        "timestamp": now_timestamp(),
        "run_name": run_dir.name,
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
    write_json(run_dir / "meta" / "dataset_manifest.json", manifest)
    write_json(run_dir / "meta" / "split_summary.json", split_summary)
    write_json(run_dir / "meta" / "sample_indices.json", sample_indices)
    copy_resolved_configs(run_dir / "configs", dataset_config, train_config, hf_config)
    return train_path, val_path


def resolve_model_source(train_config: dict[str, Any], hf_config: dict[str, Any]) -> str:
    model_name = train_config["model_name"]
    local_dir = resolve_path(hf_config["models_local_dir"]) / sanitize_name(model_name.split("/")[-1])
    if has_local_model_weights(local_dir):
        return str(local_dir)
    return model_name


def has_local_model_weights(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    patterns = (
        "*.safetensors",
        "*.bin",
        "*.pt",
    )
    return any(any(model_dir.glob(pattern)) for pattern in patterns)


def resolve_weight_loading_mode(train_config: dict[str, Any]) -> tuple[bool, bool]:
    load_in_4bit = bool(train_config.get("load_in_4bit", False))
    load_in_16bit = bool(train_config.get("load_in_16bit", True))
    if load_in_4bit and load_in_16bit:
        raise ValueError("Choose exactly one weight loading mode: load_in_4bit or load_in_16bit.")
    if not load_in_4bit and not load_in_16bit:
        raise ValueError("Enable one weight loading mode: set load_in_4bit=true or load_in_16bit=true.")
    return load_in_4bit, load_in_16bit


def load_unsloth_model_and_tokenizer(
    model_source: str,
    train_config: dict[str, Any],
    dataset_config: dict[str, Any],
) -> tuple[Any, Any]:
    load_in_4bit, load_in_16bit = resolve_weight_loading_mode(train_config)
    gradient_checkpointing = "unsloth" if train_config.get("gradient_checkpointing", False) else False
    lora_cfg = train_config["lora"]
    target_modules = lora_cfg.get("target_modules", DEFAULT_LORA_TARGET_MODULES)
    common_lora_kwargs = {
        "r": int(lora_cfg["r"]),
        "lora_alpha": int(lora_cfg["alpha"]),
        "lora_dropout": float(lora_cfg["dropout"]),
        "bias": lora_cfg["bias"],
        "random_state": int(train_config.get("seed", dataset_config.get("seed", 42))),
        "use_rslora": bool(lora_cfg.get("use_rslora", False)),
        "loftq_config": lora_cfg.get("loftq_config"),
        "target_modules": target_modules,
        "modules_to_save": lora_cfg.get("modules_to_save"),
        "ensure_weight_tying": bool(lora_cfg.get("ensure_weight_tying", False)),
        "max_seq_length": int(train_config["max_seq_length"]),
    }
    dtype = "bfloat16" if bool(train_config.get("bf16", False)) else None
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_source,
        max_seq_length=int(train_config["max_seq_length"]),
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        load_in_16bit=load_in_16bit,
        full_finetuning=bool(train_config.get("full_finetuning", False)),
        trust_remote_code=train_config.get("trust_remote_code", True),
        use_gradient_checkpointing=gradient_checkpointing,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        use_gradient_checkpointing=gradient_checkpointing,
        **common_lora_kwargs,
    )
    if hasattr(FastLanguageModel, "for_training"):
        maybe_model = FastLanguageModel.for_training(
            model,
            use_gradient_checkpointing=bool(train_config.get("gradient_checkpointing", False)),
        )
        if maybe_model is not None:
            model = maybe_model

    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "pad_token_id", None) is None:
        raise RuntimeError("Tokenizer must define a pad_token_id before training.")
    tokenizer.padding_side = "right"
    return model, tokenizer


def preprocess_rows(raw_rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> tuple[Dataset, dict[str, Any]]:
    processed_rows: list[dict[str, Any]] = []
    skipped_examples: list[dict[str, Any]] = []
    for example in raw_rows:
        try:
            processed = build_tokenized_supervised_example(tokenizer, example, max_length=max_length)
        except Exception as exc:
            if len(skipped_examples) < 20:
                skipped_examples.append(
                    {
                        "source_dataset": example.get("source_dataset"),
                        "source_index": example.get("source_index"),
                        "error": str(exc),
                    }
                )
            continue
        if processed["supervised_tokens"] <= 0:
            if len(skipped_examples) < 20:
                skipped_examples.append(
                    {
                        "source_dataset": example.get("source_dataset"),
                        "source_index": example.get("source_index"),
                        "error": "No supervised tokens remained after masking/truncation.",
                    }
                )
            continue
        processed_rows.append(
            {
                "input_ids": processed["input_ids"],
                "attention_mask": processed["attention_mask"],
                "completion_mask": processed["completion_mask"],
                "seq_len": processed["seq_len"],
                "supervised_tokens": processed["supervised_tokens"],
            }
        )
    if not processed_rows:
        raise RuntimeError("All rows were filtered during preprocessing.")
    diagnostics = {
        "raw_rows": len(raw_rows),
        "kept_rows": len(processed_rows),
        "skipped_rows": len(raw_rows) - len(processed_rows),
        "skipped_examples": skipped_examples,
    }
    return Dataset.from_list(processed_rows), diagnostics


def dataset_stats(dataset_split) -> dict[str, Any]:
    seq_lengths = dataset_split["seq_len"]
    supervised = dataset_split["supervised_tokens"]
    return {
        "rows": len(dataset_split),
        "avg_seq_len": round(sum(seq_lengths) / max(1, len(seq_lengths)), 2),
        "max_seq_len": max(seq_lengths) if seq_lengths else 0,
        "avg_supervised_tokens": round(sum(supervised) / max(1, len(supervised)), 2),
    }


def estimate_total_optimizer_steps(train_rows: int, train_config: dict[str, Any]) -> int:
    explicit_max_steps = int(train_config.get("max_steps", -1))
    if explicit_max_steps and explicit_max_steps > 0:
        return explicit_max_steps

    per_device_batch = int(train_config["per_device_train_batch_size"])
    grad_accum = int(train_config["gradient_accumulation_steps"])
    effective_batch = max(1, per_device_batch * grad_accum)
    steps_per_epoch = max(1, math.ceil(train_rows / effective_batch))
    epochs = float(train_config["num_train_epochs"])
    return max(1, math.ceil(steps_per_epoch * epochs))


def resolve_warmup_steps(train_rows: int, train_config: dict[str, Any]) -> int:
    explicit_keys = (
        "warmup_steps",
        "warm_up_steps",
        "warmup steps",
        "warm_up steps",
    )
    explicit_values: dict[str, int] = {}
    for key in explicit_keys:
        if key not in train_config or train_config.get(key) is None:
            continue
        try:
            explicit_values[key] = int(train_config[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {key}: {train_config[key]!r}. It must be an integer.") from exc

    if explicit_values:
        unique_values = set(explicit_values.values())
        if len(unique_values) > 1:
            raise ValueError(
                "Conflicting warmup step settings found: "
                f"{explicit_values}. Please keep only one key or make them equal."
            )
        return max(0, unique_values.pop())

    if "warmup_ratio" in train_config:
        warmup_ratio = float(train_config.get("warmup_ratio", 0.0))
        if warmup_ratio <= 0:
            return 0
        total_steps = estimate_total_optimizer_steps(train_rows, train_config)
        resolved = max(1, int(math.ceil(total_steps * warmup_ratio)))
        logging.getLogger(__name__).warning(
            "warmup_ratio is deprecated and will be removed. "
            "Resolved warmup_ratio=%s to warmup_steps=%s. "
            "Please migrate config to warmup_steps or warm_up_steps.",
            warmup_ratio,
            resolved,
        )
        return resolved

    return 0


def resolve_lr_scheduler_type(train_config: dict[str, Any]) -> str:
    scheduler = train_config.get("lr_scheduler_type", "linear")
    if scheduler is None:
        return "linear"

    scheduler_str = str(scheduler).strip()
    if not scheduler_str:
        return "linear"

    # Accept values such as "SchedulerType.LINEAR" while keeping raw names like "cosine".
    if "." in scheduler_str:
        scheduler_str = scheduler_str.rsplit(".", 1)[-1]

    return scheduler_str.lower()


def pack_dataset_split(dataset_split, max_length: int) -> Dataset:
    packed_rows: list[dict[str, Any]] = []
    current_ids: list[int] = []
    current_mask: list[int] = []
    current_completion_mask: list[int] = []

    def flush() -> None:
        nonlocal current_ids, current_mask, current_completion_mask
        if not current_ids:
            return
        packed_rows.append(
            {
                "input_ids": current_ids,
                "attention_mask": current_mask,
                "completion_mask": current_completion_mask,
                "seq_len": len(current_ids),
                "supervised_tokens": sum(current_completion_mask),
            }
        )
        current_ids = []
        current_mask = []
        current_completion_mask = []

    for row in dataset_split:
        row_len = len(row["input_ids"])
        if current_ids and len(current_ids) + row_len > max_length:
            flush()
        current_ids.extend(row["input_ids"])
        current_mask.extend(row["attention_mask"])
        current_completion_mask.extend(row["completion_mask"])
        if len(current_ids) >= max_length:
            flush()
    flush()
    return Dataset.from_list(packed_rows)


def write_metric_logs(run_dir: Path, trainer: Any) -> None:
    logs_dir = ensure_dir(run_dir / "logs")
    train_metrics = []
    eval_metrics = []
    for row in trainer.state.log_history:
        if "eval_loss" in row:
            eval_metrics.append(row)
        elif "loss" in row:
            train_metrics.append(row)
    write_jsonl(logs_dir / "train_metrics.jsonl", train_metrics)
    write_jsonl(logs_dir / "eval_metrics.jsonl", eval_metrics)
    trainer.state.save_to_json(str(logs_dir / "trainer_state.json"))


def write_run_summary(
    run_dir: Path,
    run_name: str,
    group: str,
    phase: str,
    train_config: dict[str, Any],
    preprocess_summary: dict[str, Any],
    train_result_metrics: dict[str, Any],
    eval_metrics: dict[str, Any],
    best_global_step: int | None,
    best_checkpoint: str | None,
    last_checkpoint: str | None,
) -> None:
    summary = "\n".join(
        [
            f"# Run Summary: {run_name}",
            "",
            f"- Timestamp: {now_timestamp()}",
            f"- Phase: {phase}",
            f"- Group: {group}",
            f"- Model: {train_config['model_name']}",
            f"- Max seq length: {train_config['max_seq_length']}",
            f"- Packing: {train_config.get('packing', False)}",
            f"- Best global step: {best_global_step if best_global_step is not None else 'N/A'}",
            f"- Best checkpoint: {best_checkpoint or 'N/A'}",
            f"- Last saved checkpoint: {last_checkpoint or 'N/A'}",
            "",
            "## Dataset",
            "",
            f"- Train rows: {preprocess_summary['train']['rows']}",
            f"- Val rows: {preprocess_summary['validation']['rows']}",
            f"- Train avg seq len: {preprocess_summary['train']['avg_seq_len']}",
            f"- Val avg seq len: {preprocess_summary['validation']['avg_seq_len']}",
            "",
            "## Metrics",
            "",
            f"- Train result: `{json.dumps(train_result_metrics, ensure_ascii=False)}`",
            f"- Eval result: `{json.dumps(eval_metrics, ensure_ascii=False)}`",
        ]
    )
    (run_dir / "meta" / "run_summary.md").write_text(summary + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    silence_known_training_log_noise()
    dataset_config = load_dataset_config(args.dataset_config)
    train_config = load_dataset_config(args.train_config)
    hf_config = load_hf_config(args.hf_config)

    group = args.group or dataset_config.get("default_group", "C")
    phase = args.phase or train_config.get("phase") or dataset_config.get("default_phase", "stage1")
    run_name = args.run_name or generate_run_name(phase, group, train_config, dataset_config)
    run_dir = resolve_path(args.output_dir) if args.output_dir else run_output_dir(run_name)
    trainer_output_dir = ensure_dir(run_dir / "trainer_output")
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    best_dir = ensure_dir(checkpoints_dir / "best")
    final_dir = ensure_dir(checkpoints_dir / "final")
    logs_dir = ensure_dir(run_dir / "logs")
    meta_dir = ensure_dir(run_dir / "meta")
    configs_dir = ensure_dir(run_dir / "configs")
    console_log = logs_dir / "console.log"

    append_console(console_log, f"Starting run {run_name}.")
    copy_resolved_configs(configs_dir, dataset_config, train_config, hf_config)

    if not args.skip_auto_build:
        train_path, val_path = ensure_dataset_splits(
            run_dir=run_dir,
            dataset_config=dataset_config,
            train_config=train_config,
            hf_config=hf_config,
            group=group,
            phase=phase,
            force_local_datasets=args.force_local_datasets,
        )
    else:
        train_path = run_dir / "dataset" / "train.jsonl"
        val_path = run_dir / "dataset" / "val.jsonl"
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError("Dataset splits are missing and --skip-auto-build was passed.")

    append_console(console_log, f"Using train split {train_path} and val split {val_path}.")

    local_model_dir = resolve_path(hf_config["models_local_dir"]) / sanitize_name(train_config["model_name"].split("/")[-1])
    if not has_local_model_weights(local_model_dir):
        append_console(console_log, "Base model snapshot missing or incomplete locally. Fetching from HF first.")
        fetch_base_model_snapshot(hf_config, train_config["model_name"], local_model_dir)

    model_source = resolve_model_source(train_config, hf_config)
    model, tokenizer = load_unsloth_model_and_tokenizer(
        model_source,
        train_config,
        dataset_config,
    )
    ensure_triton_allocator(console_log)
    append_console(
        console_log,
        f"Loaded model from {model_source} and attached LoRA adapters.",
    )

    raw_dataset = {
        "train": read_jsonl(train_path),
        "validation": read_jsonl(val_path),
    }
    tokenized_dataset = {}
    preprocess_diagnostics = {}
    for split, split_rows in raw_dataset.items():
        processed_dataset, diagnostics = preprocess_rows(split_rows, tokenizer, train_config["max_seq_length"])
        tokenized_dataset[split] = processed_dataset
        preprocess_diagnostics[split] = diagnostics
    if train_config.get("packing", False):
        tokenized_dataset["train"] = pack_dataset_split(tokenized_dataset["train"], train_config["max_seq_length"])
        append_console(console_log, "Enabled explicit packing for the training split.")
    preprocess_summary = {}
    for split, dataset in tokenized_dataset.items():
        split_stats = dataset_stats(dataset)
        split_stats["raw_rows"] = preprocess_diagnostics[split]["raw_rows"]
        split_stats["kept_rows"] = preprocess_diagnostics[split]["kept_rows"]
        split_stats["skipped_rows"] = preprocess_diagnostics[split]["skipped_rows"]
        preprocess_summary[split] = split_stats
    write_json(meta_dir / "preprocess_summary.json", preprocess_summary)
    write_json(meta_dir / "preprocess_diagnostics.json", preprocess_diagnostics)
    append_console(console_log, f"Tokenized dataset summary: {json.dumps(preprocess_summary, ensure_ascii=False)}")

    trainer_ready_dataset = {}
    for split, dataset in tokenized_dataset.items():
        removable = [name for name in ("seq_len", "supervised_tokens") if name in dataset.column_names]
        trainer_ready_dataset[split] = dataset.remove_columns(removable) if removable else dataset

    data_collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id,
        completion_only_loss=True,
    )
    bf16_enabled = bool(train_config.get("bf16", False) and is_bf16_supported())
    fp16_enabled = bool(not bf16_enabled)
    warmup_steps = resolve_warmup_steps(len(tokenized_dataset["train"]), train_config)
    lr_scheduler_type = resolve_lr_scheduler_type(train_config)
    append_console(console_log, f"Resolved warmup_steps={warmup_steps}.")
    append_console(console_log, f"Resolved lr_scheduler_type={lr_scheduler_type}.")
    training_args = SFTConfig(
        output_dir=str(trainer_output_dir),
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        per_device_eval_batch_size=train_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        learning_rate=float(train_config["learning_rate"]),
        num_train_epochs=float(train_config["num_train_epochs"]),
        max_steps=int(train_config.get("max_steps", -1)),
        warmup_steps=warmup_steps,
        lr_scheduler_type=lr_scheduler_type,
        weight_decay=float(train_config["weight_decay"]),
        max_grad_norm=float(train_config["max_grad_norm"]),
        logging_strategy="steps",
        logging_steps=int(train_config["logging_steps"]),
        eval_strategy=train_config["eval_strategy"],
        eval_steps=int(train_config["eval_steps"]),
        save_strategy=train_config["save_strategy"],
        save_steps=int(train_config["save_steps"]),
        save_total_limit=int(train_config["save_total_limit"]),
        load_best_model_at_end=bool(train_config["load_best_model_at_end"]),
        metric_for_best_model=train_config["metric_for_best_model"],
        greater_is_better=bool(train_config["greater_is_better"]),
        bf16=bf16_enabled,
        fp16=fp16_enabled,
        dataloader_num_workers=int(train_config.get("dataloader_num_workers", 2)),
        gradient_checkpointing=bool(train_config.get("gradient_checkpointing", False)),
        optim=train_config.get("optim", "adamw_8bit"),
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        seed=int(train_config.get("seed", dataset_config.get("seed", 42))),
        do_train=True,
        do_eval=True,
        max_length=int(train_config["max_seq_length"]),
        packing=False,
        dataset_num_proc=int(train_config.get("dataset_num_proc", 1)),
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=trainer_ready_dataset["train"],
        eval_dataset=trainer_ready_dataset["validation"],
        data_collator=data_collator,
    )
    trainer.data_collator = data_collator
    append_console(
        console_log,
        (
            "Trainer data_collator="
            f"{trainer.data_collator.__class__.__module__}.{trainer.data_collator.__class__.__name__}"
        ),
    )

    try:
        first_train_batch = next(iter(trainer.get_train_dataloader()))
        batch_shapes = {
            key: list(value.shape) if hasattr(value, "shape") else None
            for key, value in first_train_batch.items()
        }
        append_console(console_log, f"Prefetched first train batch successfully: {json.dumps(batch_shapes)}")
        del first_train_batch
    except Exception as exc:
        append_console(console_log, f"Failed while prefetching first train batch: {exc}")
        raise

    train_result = trainer.train()
    eval_metrics = trainer.evaluate(trainer_ready_dataset["validation"])
    write_metric_logs(run_dir, trainer)
    write_json(logs_dir / "train_result_metrics.json", train_result.metrics)
    write_json(logs_dir / "eval_metrics.json", eval_metrics)

    best_checkpoint = trainer.state.best_model_checkpoint
    best_global_step = getattr(trainer.state, "best_global_step", None)
    save_best_adapter_snapshot(
        best_dir=best_dir,
        tokenizer_artifact=tokenizer,
        trainer_model=trainer.model,
        best_checkpoint=best_checkpoint,
        canonical_base_model_id=train_config["model_name"],
        console_log=console_log,
    )

    last_checkpoint = get_last_checkpoint(str(trainer_output_dir))
    save_final_adapter_snapshot(
        final_dir=final_dir,
        tokenizer_artifact=tokenizer,
        trainer_model=trainer.model,
        last_checkpoint=last_checkpoint,
        canonical_base_model_id=train_config["model_name"],
        console_log=console_log,
    )

    env_info = collect_environment_info()
    env_info["run_name"] = run_name
    env_info["model_source"] = model_source
    env_info["best_global_step"] = best_global_step
    env_info["best_checkpoint"] = best_checkpoint
    env_info["last_checkpoint"] = last_checkpoint
    write_json(meta_dir / "env_info.json", env_info)

    write_run_summary(
        run_dir=run_dir,
        run_name=run_name,
        group=group,
        phase=phase,
        train_config=train_config,
        preprocess_summary=preprocess_summary,
        train_result_metrics=train_result.metrics,
        eval_metrics=eval_metrics,
        best_global_step=best_global_step,
        best_checkpoint=best_checkpoint,
        last_checkpoint=last_checkpoint,
    )

    if not args.skip_auto_upload:
        append_console(console_log, "Uploading run artifacts to HF.")
        upload_run_artifacts(hf_config, run_name, run_dir)
    else:
        append_console(console_log, "Skipping HF upload because --skip-auto-upload was provided.")

    append_console(console_log, f"Run {run_name} finished successfully.")


if __name__ == "__main__":
    main()
