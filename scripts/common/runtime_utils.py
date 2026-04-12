#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml
from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_EMPTY_THINK_PREFIX = "<think>\n</think>\n\n"
OFFICIAL_HF_ENDPOINT = "https://huggingface.co"


def repo_root() -> Path:
    return REPO_ROOT


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return repo_root() / path_obj


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(resolve_path(path).read_text(encoding="utf-8"))


def save_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    path_obj = resolve_path(path)
    ensure_dir(path_obj.parent)
    path_obj.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def write_json(path: str | Path, payload: Any) -> None:
    path_obj = resolve_path(path)
    ensure_dir(path_obj.parent)
    path_obj.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_json(path: str | Path) -> Any:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path_obj = resolve_path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path_obj = resolve_path(path)
    rows: list[dict[str, Any]] = []
    with path_obj.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_keys_json() -> dict[str, Any]:
    path = repo_root() / "keys.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_hf_token(hf_config: dict[str, Any] | None = None) -> str:
    hf_config = hf_config or {}
    token_env = hf_config.get("token_env", "HF_TOKEN")
    token = os.environ.get(token_env)
    if token:
        return token
    if hf_config.get("allow_keys_json_fallback", True):
        keys_payload = load_keys_json()
        token = keys_payload.get("hf_token")
        if token:
            return token
    raise RuntimeError(
        f"Hugging Face token not found. Set {token_env} or provide keys.json with hf_token."
    )


def resolve_hf_repo_id(hf_config: dict[str, Any]) -> str:
    configured = os.environ.get("HF_REPO_ID") or hf_config["repo_id"]
    if "/" in configured:
        return configured
    token = get_hf_token(hf_config)
    api = HfApi(endpoint=OFFICIAL_HF_ENDPOINT, token=token)
    who = api.whoami(token=token)
    username = who.get("name") or who.get("id")
    if not username:
        raise RuntimeError("Unable to resolve Hugging Face username from token.")
    return f"{username}/{configured}"


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1].lower().replace(".", "").replace("_", "-")


def sanitize_name(value: str) -> str:
    value = value.strip().replace(" ", "-")
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-")


def generate_run_name(
    phase: str,
    group: str,
    train_config: dict[str, Any],
    dataset_config: dict[str, Any],
) -> str:
    date_part = datetime.now().strftime("%Y-%m-%d")
    model_part = model_slug(train_config["model_name"])
    lr = format(float(train_config["learning_rate"]), ".0e").replace("+0", "").replace("+", "")
    rank = train_config["lora"]["r"]
    epochs = train_config["num_train_epochs"]
    seq = train_config["max_seq_length"]
    pack = 1 if train_config.get("packing", False) else 0
    seed = train_config.get("seed", dataset_config.get("seed", 42))
    max_steps = int(train_config.get("max_steps", -1))
    max_steps_part = f"_ms{max_steps}" if max_steps > 0 else ""
    return sanitize_name(
        f"{date_part}_{phase}_{group}_{model_part}_lr{lr}_r{rank}_e{epochs}{max_steps_part}_seq{seq}_pack{pack}_seed{seed}"
    )


def load_locked_requirements(path: str | Path) -> list[tuple[str, str]]:
    requirements: list[tuple[str, str]] = []
    for raw in resolve_path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise ValueError(f"Unsupported requirement line: {line}")
        name, version = line.split("==", 1)
        requirements.append((name.strip(), version.strip()))
    return requirements


def flatten_dict(prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        composed = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(composed, value))
        else:
            flat[composed] = value
    return flat


def now_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_output_dir(run_name: str) -> Path:
    return ensure_dir(repo_root() / "runs" / run_name)


def stage_copytree(
    source_dir: str | Path,
    destination_dir: str | Path,
    include_subdirs: Iterable[str] | None = None,
) -> Path:
    source_dir = resolve_path(source_dir)
    destination_dir = resolve_path(destination_dir)
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    ensure_dir(destination_dir)
    include_set = set(include_subdirs or [])
    if include_set:
        for subdir in include_set:
            src = source_dir / subdir
            if src.exists():
                shutil.copytree(src, destination_dir / subdir)
    else:
        shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)
    return destination_dir


def collect_environment_info() -> dict[str, Any]:
    import platform
    import sys

    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }
    try:
        import torch

        payload["torch"] = torch.__version__
        payload["cuda_available"] = torch.cuda.is_available()
        payload["cuda_device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            payload["cuda_device_name"] = torch.cuda.get_device_name(0)
            payload["bf16_supported"] = torch.cuda.is_bf16_supported()
    except Exception as exc:  # pragma: no cover
        payload["torch_probe_error"] = repr(exc)
    for module_name in ("unsloth", "transformers", "trl", "peft", "datasets", "accelerate", "bitsandbytes", "xformers"):
        try:
            module = __import__(module_name)
            payload[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover
            payload[f"{module_name}_probe_error"] = repr(exc)
    for module_name in ("fla", "causal_conv1d"):
        try:
            module = __import__(module_name)
            payload[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover
            payload[f"{module_name}_probe_error"] = repr(exc)
    return payload


def copy_resolved_configs(
    output_dir: str | Path,
    dataset_config: dict[str, Any],
    train_config: dict[str, Any],
    hf_config: dict[str, Any],
) -> None:
    output_dir = ensure_dir(output_dir)
    write_json(output_dir / "dataset_config_resolved.json", dataset_config)
    write_json(output_dir / "train_config_resolved.json", train_config)
    write_json(output_dir / "hf_config_resolved.json", hf_config)
    save_yaml(output_dir / "dataset_config_resolved.yaml", dataset_config)
    save_yaml(output_dir / "train_config_resolved.yaml", train_config)
    save_yaml(output_dir / "hf_config_resolved.yaml", hf_config)
