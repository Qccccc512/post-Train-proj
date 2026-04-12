#!/usr/bin/env python3
from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import gc
import hashlib
import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi, hf_hub_download
from unsloth import FastLanguageModel
from unsloth_zoo.saving_utils import find_skipped_quantized_modules
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from scripts.hub.hf_repo_sync import ensure_remote_repo, load_hf_config, upload_folder_with_retry
from scripts.common.runtime_utils import (
    OFFICIAL_HF_ENDPOINT,
    ensure_dir,
    get_hf_token,
    model_slug,
    now_timestamp,
    read_json,
    resolve_path,
    resolve_hf_repo_id,
    sanitize_name,
    write_json,
)

COMPATIBILITY_ASSET_FILENAMES = (
    "generation_config.json",
    "LICENSE",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a LoRA checkpoint into merged bf16 and merged 4bit full-model variants, "
            "then optionally upload both model folders to separate Hugging Face repos. "
            "The 4bit variant is derived from the merged bf16 snapshot."
        )
    )
    parser.add_argument("--adapter-dir", required=True, help="Local LoRA adapter directory, e.g. runs/<run>/checkpoints/best")
    parser.add_argument("--run-name")
    parser.add_argument("--checkpoint-kind", default="best")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--bf16-output-dir")
    parser.add_argument("--fourbit-output-dir")
    parser.add_argument("--hf-config", default="configs/hf/default.yaml")
    parser.add_argument("--base-model-id")
    parser.add_argument("--bf16-repo-id")
    parser.add_argument("--fourbit-repo-id")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--upload-bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--upload-fourbit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-order", choices=["bf16-first", "fourbit-first"], default="bf16-first")
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--bf16-dtype", choices=["auto", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--fourbit-compute-dtype", choices=["auto", "bfloat16", "float16"], default="auto")
    parser.add_argument("--maximum-memory-usage", type=float, default=0.75)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def is_hf_repo_id(value: str | None) -> bool:
    if not value:
        return False
    return "/" in value and not value.startswith("/") and not value.startswith("http")


def detect_base_model_id(adapter_dir: Path, explicit: str | None) -> str:
    if explicit:
        return explicit

    adapter_config_path = adapter_dir / "adapter_config.json"
    if adapter_config_path.exists():
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        candidate = adapter_config.get("base_model_name_or_path")
        if is_hf_repo_id(candidate):
            return candidate

    run_dir = adapter_dir.parent.parent
    train_config_path = run_dir / "configs" / "train_config_resolved.json"
    if train_config_path.exists():
        candidate = read_json(train_config_path).get("model_name")
        if is_hf_repo_id(candidate):
            return candidate

    raise RuntimeError(
        "Unable to determine a valid Hugging Face base model id. "
        "Pass --base-model-id explicitly."
    )


def detect_max_seq_length(adapter_dir: Path, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    train_config_path = adapter_dir.parent.parent / "configs" / "train_config_resolved.json"
    if train_config_path.exists():
        candidate = read_json(train_config_path).get("max_seq_length")
        if candidate:
            return int(candidate)
    return 32768


def has_local_model_weights(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    patterns = (
        "*.safetensors",
        "*.bin",
        "*.pt",
    )
    return any(any(model_dir.glob(pattern)) for pattern in patterns)


def resolve_base_model_source(
    adapter_dir: Path,
    base_model_id: str,
    hf_config: dict[str, Any],
) -> str:
    adapter_config_path = adapter_dir / "adapter_config.json"
    if adapter_config_path.exists():
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        candidate = adapter_config.get("base_model_name_or_path")
        if candidate:
            candidate_path = Path(candidate).expanduser()
            if candidate_path.is_absolute() and has_local_model_weights(candidate_path):
                return str(candidate_path)

    base_model_path = Path(base_model_id).expanduser()
    if base_model_path.is_absolute() and has_local_model_weights(base_model_path):
        return str(base_model_path)

    models_local_dir = hf_config.get("models_local_dir")
    if models_local_dir and is_hf_repo_id(base_model_id):
        cached_dir = resolve_path(models_local_dir) / sanitize_name(base_model_id.split("/")[-1])
        if has_local_model_weights(cached_dir):
            return str(cached_dir)

    return base_model_id


@contextmanager
def patched_adapter_dir(adapter_dir: Path, base_model_source: str):
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        yield adapter_dir
        return

    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    current_base_model = adapter_config.get("base_model_name_or_path")
    if current_base_model == base_model_source:
        yield adapter_dir
        return

    temp_root = Path(tempfile.mkdtemp(prefix="patched_adapter_"))
    patched_dir = temp_root / adapter_dir.name
    shutil.copytree(adapter_dir, patched_dir)
    patched_config_path = patched_dir / "adapter_config.json"
    adapter_config["base_model_name_or_path"] = base_model_source
    patched_config_path.write_text(json.dumps(adapter_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        yield patched_dir
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def compact_slug(value: str, max_length: int = 60) -> str:
    clean = value.strip().replace(" ", "-")
    clean = "".join(ch if ch.isalnum() or ch in "-._" else "-" for ch in clean)
    while "--" in clean:
        clean = clean.replace("--", "-")
    clean = clean.strip("-")
    if len(clean) <= max_length:
        return clean
    digest = hashlib.sha1(clean.encode("utf-8")).hexdigest()[:8]
    head = clean[: max_length - len(digest) - 1].rstrip("-")
    return f"{head}-{digest}"


def derive_repo_id(
    explicit_repo_id: str | None,
    *,
    hf_config: dict[str, Any],
    run_name: str,
    checkpoint_kind: str,
    variant_suffix: str,
) -> str:
    if explicit_repo_id:
        if "/" in explicit_repo_id:
            return explicit_repo_id
        token = get_hf_token(hf_config)
        api = HfApi(endpoint=OFFICIAL_HF_ENDPOINT, token=token)
        who = api.whoami(token=token)
        username = who.get("name") or who.get("id")
        if not username:
            raise RuntimeError("Unable to resolve Hugging Face username from token.")
        return f"{username}/{explicit_repo_id}"

    resolved_main_repo_id = resolve_hf_repo_id(hf_config)
    namespace, repo_name = resolved_main_repo_id.split("/", 1)
    repo_stub = compact_slug(
        f"{repo_name}-stage2-{run_name}-{checkpoint_kind}-{variant_suffix}",
        max_length=90,
    )
    return f"{namespace}/{repo_stub}"


def resolve_torch_dtype(name: str) -> torch.dtype | None:
    if name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return getattr(torch, name)


def clear_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_model_card(
    output_dir: Path,
    *,
    variant_name: str,
    base_model_id: str,
    run_name: str,
    checkpoint_kind: str,
    adapter_dir: Path,
) -> None:
    readme = "\n".join(
        [
            "---",
            f"base_model: {base_model_id}",
            "library_name: transformers",
            "pipeline_tag: text-generation",
            "tags:",
            "- unsloth",
            "- merged-lora",
            f"- {variant_name}",
            "---",
            "",
            f"# {variant_name}",
            "",
            "- Export source:",
            f"  - run_name: `{run_name}`",
            f"  - checkpoint_kind: `{checkpoint_kind}`",
            f"  - adapter_dir: `{adapter_dir}`",
            f"  - base_model: `{base_model_id}`",
            "",
            "This repo stores a merged full-model snapshot exported from the Stage 2 final-training LoRA checkpoint.",
            "",
        ]
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def resolve_base_model_compatibility_asset(
    *,
    filename: str,
    base_model_source: str,
    base_model_id: str,
    hf_config: dict[str, Any],
) -> Path | None:
    candidate_path = Path(base_model_source).expanduser()
    if candidate_path.is_absolute():
        local_file = candidate_path / filename
        if local_file.exists():
            return local_file

    if is_hf_repo_id(base_model_id):
        try:
            token = get_hf_token(hf_config)
        except Exception:
            token = None
        try:
            downloaded = hf_hub_download(
                repo_id=base_model_id,
                filename=filename,
                repo_type="model",
                revision=hf_config.get("revision", "main"),
                token=token,
            )
            return Path(downloaded)
        except Exception:
            return None

    return None


def enrich_export_with_base_model_assets(
    *,
    local_dir: Path,
    base_model_source: str,
    base_model_id: str,
    hf_config: dict[str, Any],
) -> dict[str, Any]:
    copied: list[str] = []
    already_present: list[str] = []
    missing_upstream: list[str] = []

    for filename in COMPATIBILITY_ASSET_FILENAMES:
        destination = local_dir / filename
        if destination.exists():
            already_present.append(filename)
            continue
        source = resolve_base_model_compatibility_asset(
            filename=filename,
            base_model_source=base_model_source,
            base_model_id=base_model_id,
            hf_config=hf_config,
        )
        if source is None:
            missing_upstream.append(filename)
            continue
        shutil.copy2(source, destination)
        copied.append(filename)

    if copied or missing_upstream:
        print(
            "[export_merged_model_variants] Compatibility asset sync: "
            f"copied={copied or '[]'}, already_present={already_present or '[]'}, "
            f"missing_upstream={missing_upstream or '[]'}",
            flush=True,
        )

    return {
        "copied": copied,
        "already_present": already_present,
        "missing_upstream": missing_upstream,
    }


def summarize_folder_for_upload(local_dir: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    total_bytes = 0
    largest_file: dict[str, Any] | None = None
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        size_bytes = path.stat().st_size
        rel_path = path.relative_to(local_dir).as_posix()
        entry = {"path": rel_path, "size_bytes": size_bytes}
        files.append(entry)
        total_bytes += size_bytes
        if largest_file is None or size_bytes > largest_file["size_bytes"]:
            largest_file = entry
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "largest_file": largest_file,
        "files": files,
    }


def upload_exported_model(
    *,
    local_dir: Path,
    repo_id: str,
    hf_config: dict[str, Any],
    revision: str,
    private: bool,
    commit_message: str,
) -> None:
    token = get_hf_token(hf_config)
    api = HfApi(endpoint=OFFICIAL_HF_ENDPOINT, token=token)
    ensure_remote_repo(
        api,
        {
            **hf_config,
            "private": private,
            "revision": revision,
            "repo_type": "model",
        },
        repo_id,
    )
    upload_summary = summarize_folder_for_upload(local_dir)
    largest_file = upload_summary["largest_file"]
    total_gib = upload_summary["total_bytes"] / (1024 ** 3)
    largest_file_mib = (largest_file["size_bytes"] / (1024 ** 2)) if largest_file else 0.0
    print(
        "[export_merged_model_variants] "
        f"Preparing upload for {repo_id}: files={upload_summary['file_count']}, "
        f"total_size={total_gib:.2f} GiB, "
        f"largest_file={largest_file['path'] if largest_file else 'n/a'} "
        f"({largest_file_mib:.1f} MiB)",
        flush=True,
    )

    use_large_folder_upload = (
        upload_summary["total_bytes"] >= 512 * 1024 * 1024
        or (largest_file is not None and largest_file["size_bytes"] >= 200 * 1024 * 1024)
    )

    if use_large_folder_upload:
        num_workers = max(1, int(hf_config.get("large_upload_num_workers", 1)))
        report_every = max(5, int(hf_config.get("large_upload_report_every_s", 15)))
        print(
            "[export_merged_model_variants] "
            f"Using upload_large_folder for {repo_id} "
            f"(num_workers={num_workers}, print_report_every={report_every}s).",
            flush=True,
        )
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(local_dir),
            revision=revision,
            private=private,
            num_workers=num_workers,
            print_report=True,
            print_report_every=report_every,
        )
    else:
        print(
            "[export_merged_model_variants] "
            f"Using upload_folder for {repo_id}.",
            flush=True,
        )
        upload_folder_with_retry(
            api,
            hf_config,
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(local_dir),
            path_in_repo="",
            allow_patterns=None,
            commit_message=commit_message,
            revision=revision,
        )


def should_export(
    local_dir: Path,
    manifest_path: Path,
    force: bool,
    *,
    expected_save_method: str,
    expected_compute_dtype_name: str,
    expected_source_model_dir: Path | None = None,
) -> bool:
    if force:
        return True
    if not local_dir.exists():
        return True
    if not (local_dir / "config.json").exists():
        return True
    has_weights = any(
        (local_dir / filename).exists()
        for filename in ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json")
    )
    if not has_weights:
        return True
    if manifest_path.exists():
        payload = read_json(manifest_path)
        if payload.get("save_method") != expected_save_method:
            return True
        if payload.get("compute_dtype") != expected_compute_dtype_name:
            return True
        manifest_source_dir = payload.get("source_model_dir")
        expected_source = str(expected_source_model_dir) if expected_source_model_dir else None
        if manifest_source_dir != expected_source:
            return True
        return False
    return False


def build_variant_manifest(
    *,
    variant_name: str,
    save_method: str,
    load_mode: str,
    compute_dtype_name: str,
    local_dir: Path,
    repo_id: str | None,
    base_model_id: str,
    base_model_source: str,
    adapter_dir: Path,
    run_name: str,
    checkpoint_kind: str,
    max_seq_length: int,
    upload_completed: bool,
    source_model_dir: str | None = None,
    compatibility_assets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": now_timestamp(),
        "variant": variant_name,
        "save_method": save_method,
        "load_mode": load_mode,
        "compute_dtype": compute_dtype_name,
        "local_dir": str(local_dir),
        "repo_id": repo_id,
        "base_model_id": base_model_id,
        "base_model_source": base_model_source,
        "adapter_dir": str(adapter_dir),
        "source_model_dir": source_model_dir,
        "run_name": run_name,
        "checkpoint_kind": checkpoint_kind,
        "max_seq_length": max_seq_length,
        "upload_completed": upload_completed,
        "compatibility_assets": compatibility_assets or {},
    }


def save_quantized_4bit_model(
    *,
    source_model_dir: Path,
    local_dir: Path,
    compute_dtype_name: str,
    trust_remote_code: bool,
) -> None:
    if not source_model_dir.exists():
        raise FileNotFoundError(f"Source merged bf16 model is missing: {source_model_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("Exporting the merged 4bit model requires CUDA.")

    compute_dtype = resolve_torch_dtype(compute_dtype_name)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    print(
        f"[export_merged_model_variants] Quantizing merged bf16 model from {source_model_dir} into 4bit",
        flush=True,
    )
    merged_model = AutoModelForCausalLM.from_pretrained(
        str(source_model_dir),
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(source_model_dir), trust_remote_code=trust_remote_code)
    merged_model.eval()

    try:
        skipped_modules, _ = find_skipped_quantized_modules(merged_model)
    except Exception:
        skipped_modules = []

    if skipped_modules:
        print(
            f"[export_merged_model_variants] Found skipped quantized modules: {skipped_modules}",
            flush=True,
        )
        quantization_config = getattr(merged_model.config, "quantization_config", None)
        if quantization_config is None:
            quantization_config = {}
            merged_model.config.quantization_config = quantization_config
        if isinstance(quantization_config, dict):
            quantization_config["llm_int8_skip_modules"] = skipped_modules
        else:
            setattr(quantization_config, "llm_int8_skip_modules", skipped_modules)

    print(f"[export_merged_model_variants] Saving merged 4bit model to {local_dir}", flush=True)
    try:
        merged_model.save_pretrained(str(local_dir))
    except NotImplementedError:
        print(
            "[export_merged_model_variants] transformers weight-conversion reversal is unavailable for this "
            "merged 4bit model; retrying save with cleared conversion metadata.",
            flush=True,
        )
        merged_model._weight_conversions = []
        merged_model.save_pretrained(str(local_dir))

    tokenizer.save_pretrained(str(local_dir))


def export_variant(
    *,
    variant_name: str,
    save_method: str,
    load_in_16bit: bool,
    load_in_4bit: bool,
    compute_dtype_name: str,
    local_dir: Path,
    repo_id: str | None,
    base_model_id: str,
    base_model_source: str,
    adapter_dir: Path,
    source_model_dir: Path | None,
    run_name: str,
    checkpoint_kind: str,
    max_seq_length: int,
    maximum_memory_usage: float,
    trust_remote_code: bool,
    force: bool,
    skip_upload: bool,
    hf_config: dict[str, Any],
    revision: str,
    private: bool,
) -> dict[str, Any]:
    manifest_path = local_dir / "hf_merged_export_manifest.json"
    load_mode = "4bit-from-bf16" if save_method == "merged_4bit_from_bf16" else ("4bit" if load_in_4bit else "16bit")
    should_save = should_export(
        local_dir,
        manifest_path,
        force,
        expected_save_method=save_method,
        expected_compute_dtype_name=compute_dtype_name,
        expected_source_model_dir=source_model_dir,
    )

    if should_save:
        if local_dir.exists():
            shutil.rmtree(local_dir)
        ensure_dir(local_dir.parent)
        compute_dtype = resolve_torch_dtype(compute_dtype_name)
        print(f"[export_merged_model_variants] Exporting {variant_name} to {local_dir}", flush=True)
        model = None
        tokenizer = None
        try:
            if save_method == "merged_4bit_from_bf16":
                if source_model_dir is None:
                    raise RuntimeError("source_model_dir is required for merged_4bit_from_bf16 export")
                save_quantized_4bit_model(
                    source_model_dir=source_model_dir,
                    local_dir=local_dir,
                    compute_dtype_name=compute_dtype_name,
                    trust_remote_code=trust_remote_code,
                )
            else:
                with patched_adapter_dir(adapter_dir, base_model_source) as load_adapter_dir:
                    if load_adapter_dir != adapter_dir:
                        print(
                            "[export_merged_model_variants] "
                            f"Patched adapter base_model_name_or_path for export: {base_model_source}",
                            flush=True,
                        )
                    print(
                        "[export_merged_model_variants] "
                        f"Loading base model + adapter with FastLanguageModel.from_pretrained: "
                        f"model_name={load_adapter_dir}, load_in_16bit={load_in_16bit}, "
                        f"load_in_4bit={load_in_4bit}, max_seq_length={max_seq_length}",
                        flush=True,
                    )
                    model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name=str(load_adapter_dir),
                        max_seq_length=max_seq_length,
                        dtype=compute_dtype,
                        load_in_4bit=load_in_4bit,
                        load_in_16bit=load_in_16bit,
                        trust_remote_code=trust_remote_code,
                    )
                    print(
                        "[export_merged_model_variants] "
                        "Model + adapter loaded successfully; starting save_pretrained_merged.",
                        flush=True,
                    )
                    model.eval()
                    print(
                        "[export_merged_model_variants] "
                        f"Saving merged weights to {local_dir} with save_method={save_method}.",
                        flush=True,
                    )
                    model.save_pretrained_merged(
                        str(local_dir),
                        tokenizer=tokenizer,
                        save_method=save_method,
                        maximum_memory_usage=maximum_memory_usage,
                    )
                    print(
                        "[export_merged_model_variants] "
                        f"Finished save_pretrained_merged for {variant_name}.",
                        flush=True,
                    )
        finally:
            model = None
            tokenizer = None
            clear_gpu_memory()
    else:
        print(
            f"[export_merged_model_variants] Reusing existing local export for {variant_name}: {local_dir}",
            flush=True,
        )

    write_model_card(
        local_dir,
        variant_name=variant_name,
        base_model_id=base_model_id,
        run_name=run_name,
        checkpoint_kind=checkpoint_kind,
        adapter_dir=adapter_dir,
    )
    compatibility_assets = enrich_export_with_base_model_assets(
        local_dir=local_dir,
        base_model_source=base_model_source,
        base_model_id=base_model_id,
        hf_config=hf_config,
    )

    upload_completed = False
    if not skip_upload:
        if repo_id is None:
            raise RuntimeError(f"repo_id is required for upload of {variant_name}")
        print(f"[export_merged_model_variants] Uploading {variant_name} to {repo_id}", flush=True)
        upload_exported_model(
            local_dir=local_dir,
            repo_id=repo_id,
            hf_config=hf_config,
            revision=revision,
            private=private,
            commit_message=f"Upload {variant_name} for {run_name} ({checkpoint_kind})",
        )
        upload_completed = True

    manifest = build_variant_manifest(
        variant_name=variant_name,
        save_method=save_method,
        load_mode=load_mode,
        compute_dtype_name=compute_dtype_name,
        local_dir=local_dir,
        repo_id=repo_id,
        base_model_id=base_model_id,
        base_model_source=base_model_source,
        adapter_dir=adapter_dir,
        run_name=run_name,
        checkpoint_kind=checkpoint_kind,
        max_seq_length=max_seq_length,
        upload_completed=upload_completed,
        source_model_dir=str(source_model_dir) if source_model_dir else None,
        compatibility_assets=compatibility_assets,
    )
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    adapter_dir = Path(args.adapter_dir).expanduser().resolve()
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")

    run_name = args.run_name or adapter_dir.parent.parent.name
    checkpoint_kind = args.checkpoint_kind
    output_root = Path(args.output_root).expanduser().resolve()
    bf16_output_dir = Path(args.bf16_output_dir).expanduser().resolve() if args.bf16_output_dir else output_root / "merged_bf16"
    fourbit_output_dir = (
        Path(args.fourbit_output_dir).expanduser().resolve() if args.fourbit_output_dir else output_root / "merged_4bit"
    )

    hf_config = load_hf_config(args.hf_config)
    private = hf_config.get("private", True) if args.private is None else args.private
    base_model_id = detect_base_model_id(adapter_dir, args.base_model_id)
    base_model_source = resolve_base_model_source(adapter_dir, base_model_id, hf_config)
    max_seq_length = detect_max_seq_length(adapter_dir, args.max_seq_length)

    upload_bf16 = (not args.skip_upload) and bool(args.upload_bf16)
    upload_fourbit = (not args.skip_upload) and bool(args.upload_fourbit)

    bf16_repo_id = None if not upload_bf16 else derive_repo_id(
        args.bf16_repo_id,
        hf_config=hf_config,
        run_name=run_name,
        checkpoint_kind=checkpoint_kind,
        variant_suffix="merged-bf16",
    )
    fourbit_repo_id = None if not upload_fourbit else derive_repo_id(
        args.fourbit_repo_id,
        hf_config=hf_config,
        run_name=run_name,
        checkpoint_kind=checkpoint_kind,
        variant_suffix="merged-4bit",
    )

    exports_dir = ensure_dir(output_root)
    manifest_payload = {
        "timestamp": now_timestamp(),
        "run_name": run_name,
        "checkpoint_kind": checkpoint_kind,
        "adapter_dir": str(adapter_dir),
        "output_root": str(exports_dir),
        "base_model_id": base_model_id,
        "base_model_source": base_model_source,
        "max_seq_length": max_seq_length,
        "upload_bf16": upload_bf16,
        "upload_fourbit": upload_fourbit,
        "export_order": args.export_order,
        "exports": {},
    }

    model_name_tag = model_slug(base_model_id)

    variant_specs = {
        "merged_bf16": {
            "variant_name": f"{model_name_tag}_merged_bf16",
            "save_method": "merged_16bit",
            "load_in_16bit": True,
            "load_in_4bit": False,
            "compute_dtype_name": args.bf16_dtype,
            "local_dir": bf16_output_dir,
            "repo_id": bf16_repo_id,
        },
        "merged_4bit": {
            "variant_name": f"{model_name_tag}_merged_4bit",
            "save_method": "merged_4bit_from_bf16",
            "load_in_16bit": False,
            "load_in_4bit": False,
            "compute_dtype_name": args.fourbit_compute_dtype,
            "local_dir": fourbit_output_dir,
            "repo_id": fourbit_repo_id,
        },
    }
    export_order = ["merged_bf16", "merged_4bit"]
    manifest_payload["effective_export_order"] = export_order

    for export_key in export_order:
        spec = variant_specs[export_key]
        manifest_payload["exports"][export_key] = export_variant(
            variant_name=spec["variant_name"],
            save_method=spec["save_method"],
            load_in_16bit=spec["load_in_16bit"],
            load_in_4bit=spec["load_in_4bit"],
            compute_dtype_name=spec["compute_dtype_name"],
            local_dir=spec["local_dir"],
            repo_id=spec["repo_id"],
            base_model_id=base_model_id,
            base_model_source=base_model_source,
            adapter_dir=adapter_dir,
            source_model_dir=bf16_output_dir if export_key == "merged_4bit" else None,
            run_name=run_name,
            checkpoint_kind=checkpoint_kind,
            max_seq_length=max_seq_length,
            maximum_memory_usage=args.maximum_memory_usage,
            trust_remote_code=args.trust_remote_code,
            force=args.force,
            skip_upload=args.skip_upload or spec["repo_id"] is None,
            hf_config=hf_config,
            revision=args.revision,
            private=private,
        )

    write_json(exports_dir / "merged_model_exports.json", manifest_payload)
    print(json.dumps(manifest_payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
