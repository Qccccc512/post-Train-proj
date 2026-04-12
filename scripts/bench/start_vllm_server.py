#!/usr/bin/env python3
from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import subprocess

from scripts.bench.benchmark_utils import (
    build_vllm_serve_command,
    build_vllm_process_env,
    load_benchmark_config,
    pick_port,
    resolve_model_spec,
    runtime_dir,
)


def build_parser() -> argparse.ArgumentParser:
    defaults = load_benchmark_config()
    server_defaults = defaults["server"]
    parser = argparse.ArgumentParser(description="Start the pinned benchmark vLLM OpenAI-compatible server.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", help="Local adapter path or HF repo id.")
    parser.add_argument("--tokenizer")
    parser.add_argument("--host", default=server_defaults["host"])
    parser.add_argument("--port", type=int, default=server_defaults["port"])
    parser.add_argument("--served-model-name")
    parser.add_argument("--vllm-env-dir", default=server_defaults.get("vllm_env_dir") or "")
    parser.add_argument("--tensor-parallel-size", type=int, default=server_defaults["tensor_parallel_size"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=server_defaults["gpu_memory_utilization"])
    parser.add_argument("--max-model-len", type=int, default=server_defaults["max_model_len"])
    parser.add_argument("--max-num-seqs", type=int, default=server_defaults["max_num_seqs"])
    parser.add_argument("--dtype", default=server_defaults["dtype"])
    parser.add_argument("--max-lora-rank", type=int, default=server_defaults["max_lora_rank"])
    parser.add_argument("--repetition-penalty", type=float, default=server_defaults.get("repetition_penalty", 1.0))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_spec = resolve_model_spec(
        base_model=args.base_model,
        adapter=args.adapter,
        tokenizer=args.tokenizer,
        served_model_name=args.served_model_name or args.base_model,
        cache_dir=runtime_dir() / "model_cache",
    )
    port = pick_port(args.port)
    command = build_vllm_serve_command(
        vllm_env_dir_or_name=args.vllm_env_dir,
        model_spec=model_spec,
        host=args.host,
        port=port,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        dtype=args.dtype,
        max_lora_rank=args.max_lora_rank,
        repetition_penalty=args.repetition_penalty,
    )
    print(f"Starting vLLM server at http://{args.host}:{port}/v1")
    subprocess.run(command, check=True, env=build_vllm_process_env())


if __name__ == "__main__":
    main()
