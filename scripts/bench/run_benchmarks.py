#!/usr/bin/env python3
from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import json
import os
from pathlib import Path

from scripts.bench.benchmark_utils import (
    BenchmarkError,
    benchmark_output_dir,
    benchmark_output_dir as suite_output_dir,
    build_bfcl_v4_run_ids,
    build_lm_eval_model_args,
    framework_workdir,
    load_benchmark_config,
    make_run_slug,
    managed_vllm_server,
    parse_host_port,
    pick_port,
    resolve_concurrency,
    resolve_model_spec,
    run_logged_subprocess,
    runtime_python,
    runtime_dir,
    write_bfcl_project_env,
)
from scripts.common.runtime_utils import ensure_dir, now_timestamp, write_json


def build_parser() -> argparse.ArgumentParser:
    defaults = load_benchmark_config()
    server_defaults = defaults["server"]
    parser = argparse.ArgumentParser(description="Run CEval, IFEval, and BFCL v4 against a shared OpenAI-compatible endpoint.")
    parser.add_argument("--suite", choices=["all", "ceval", "ifeval", "bfcl_v4"], default="all")
    parser.add_argument("--base-model", default=defaults.get("base_model", "Qwen/Qwen3-8B"))
    parser.add_argument("--adapter", help="Local adapter path or HF repo id.")
    parser.add_argument("--tokenizer")
    parser.add_argument("--label")
    parser.add_argument("--output-base-dir", default=defaults["output_base_dir"])
    parser.add_argument("--max-samples", type=int, default=defaults["max_samples"])
    parser.add_argument("--concurrent", type=int, default=defaults.get("concurrent", 2), help="Number of concurrent requests to the vLLM server.")
    parser.add_argument("--server-base-url", help="Reuse an existing OpenAI-compatible endpoint instead of starting vLLM.")
    parser.add_argument("--skip-server-start", action="store_true")
    parser.add_argument("--host", default=server_defaults["host"])
    parser.add_argument("--port", type=int, default=server_defaults["port"])
    parser.add_argument("--served-model-name")
    parser.add_argument(
        "--vllm-env-dir",
        default=server_defaults.get("vllm_env_dir") or "",
        help="Conda env name or path to benchmark environment.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=server_defaults["tensor_parallel_size"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=server_defaults["gpu_memory_utilization"])
    parser.add_argument("--max-model-len", type=int, default=server_defaults["max_model_len"])
    parser.add_argument("--max-num-seqs", type=int, default=server_defaults["max_num_seqs"])
    parser.add_argument("--dtype", default=server_defaults["dtype"])
    parser.add_argument("--max-lora-rank", type=int, default=server_defaults["max_lora_rank"])
    parser.add_argument("--bfcl-model-name", default=defaults["bfcl"]["model_name"])
    parser.add_argument("--bfcl-max-output-tokens", type=int, default=defaults["bfcl"]["max_output_tokens"])
    parser.add_argument("--bfcl-max-samples", type=int, default=defaults["bfcl"]["max_samples"], help="BFCL-specific sample count (overrides --max-samples for BFCL).")
    parser.add_argument("--bfcl-test-categories", help="Comma-separated BFCL v4 categories to sample from.")
    parser.add_argument("--repetition-penalty", type=float, default=server_defaults.get("repetition_penalty", 1.0), help="Repetition penalty for vLLM (important for Qwen3 thinking model).")
    return parser


def selected_suites(raw: str) -> list[str]:
    if raw == "all":
        return ["ceval", "ifeval", "bfcl_v4"]
    return [raw]


def run_lm_eval_task(
    *,
    suite_name: str,
    task_name: str,
    endpoint,
    tokenizer: str,
    concurrency: int,
    limit: int,
    run_slug: str,
    output_base_dir: str,
    benchmark_env_dir: str,
    max_gen_toks: int | None = None,
    repetition_penalty: float = 1.0,
) -> Path:
    output_dir = suite_output_dir(output_base_dir, run_slug, suite_name)
    log_path = output_dir / f"{suite_name}.log"
    lm_eval_root = framework_workdir("lm_eval")
    client_python = runtime_python(benchmark_env_dir)
    env = {
        "PYTHONPATH": str(lm_eval_root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    model_args = build_lm_eval_model_args(
        endpoint=endpoint,
        tokenizer=tokenizer,
        num_concurrent=concurrency,
    )
    command = [
        str(client_python),
        "-m",
        "lm_eval",
        "--model",
        "local-completions",
        "--model_args",
        model_args,
        "--tasks",
        task_name,
        "--output_path",
        str(output_dir),
        "--log_samples",
    ]
    if limit is not None and limit < 999999:
        command.insert(9, "--limit")
        command.insert(10, str(limit))
    # Build gen_kwargs with max_gen_toks and repetition_penalty
    gen_kwargs_parts = []
    if max_gen_toks is not None:
        gen_kwargs_parts.append(f"max_gen_toks={max_gen_toks}")
    if repetition_penalty > 1.0:
        gen_kwargs_parts.append(f"repetition_penalty={repetition_penalty}")
    if gen_kwargs_parts:
        command.extend(["--gen_kwargs", ",".join(gen_kwargs_parts)])

    run_logged_subprocess(command, cwd=lm_eval_root, env_overrides=env, log_path=log_path)
    return output_dir


def run_bfcl(
    *,
    endpoint,
    model_name: str,
    max_samples: int,
    max_output_tokens: int,
    concurrency: int,
    categories: list[str] | None,
    run_slug: str,
    output_base_dir: str,
    benchmark_env_dir: str,
    repetition_penalty: float = 1.0,
) -> Path:
    output_dir = benchmark_output_dir(output_base_dir, run_slug, "bfcl_v4")
    project_root = ensure_dir(output_dir / "project_root")
    bfcl_root = framework_workdir("bfcl")
    client_python = runtime_python(benchmark_env_dir)
    env = {
        "PYTHONPATH": str(bfcl_root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "BFCL_PROJECT_ROOT": str(project_root),
    }
    write_bfcl_project_env(project_root, host=endpoint.host, port=endpoint.port)
    run_ids = build_bfcl_v4_run_ids(
        bfcl_root=bfcl_root,
        max_samples=max_samples,
        selected_categories=categories,
    )
    write_json(project_root / "test_case_ids_to_generate.json", run_ids)
    # 生成阶段
    generate_cmd = [
        str(client_python),
        str(Path(__file__).with_name("run_bfcl_entry.py")),
        "generate",
        "--model",
        model_name,
        "--run-ids",
        "--skip-server-setup",
        "--num-threads",
        str(concurrency),
        "--max-output-tokens",
        str(max_output_tokens),
    ]
    if repetition_penalty > 1.0:
        generate_cmd.extend(["--repetition-penalty", str(repetition_penalty)])
    run_logged_subprocess(
        generate_cmd,
        cwd=bfcl_root,
        env_overrides=env,
        log_path=output_dir / "bfcl_generate.log",
    )
    # 评测阶段
    run_logged_subprocess(
        [
            str(client_python),
            str(Path(__file__).with_name("run_bfcl_entry.py")),
            "evaluate",
            "--model",
            model_name,
            "--partial-eval",
        ],
        cwd=bfcl_root,
        env_overrides=env,
        log_path=output_dir / "bfcl_evaluate.log",
    )
    return output_dir


def write_summary(
    *,
    run_slug: str,
    suites: list[str],
    base_model: str,
    adapter: str | None,
    output_base_dir: str,
    server_base_url: str,
) -> Path:
    summary_dir = benchmark_output_dir(output_base_dir, run_slug, "summary")
    payload = {
        "timestamp": now_timestamp(),
        "run_slug": run_slug,
        "suites": suites,
        "base_model": base_model,
        "adapter": adapter,
        "server_base_url": server_base_url,
        "artifacts_root": str(Path(output_base_dir) / "benchmarks" / run_slug),
    }
    summary_json = summary_dir / "benchmark_run_summary.json"
    write_json(summary_json, payload)
    summary_md = summary_dir / "benchmark_run_summary.md"
    summary_md.write_text(
        "\n".join(
            [
                f"# Benchmark Run: {run_slug}",
                "",
                f"- Timestamp: {payload['timestamp']}",
                f"- Suites: {', '.join(suites)}",
                f"- Base model: `{base_model}`",
                f"- Adapter: `{adapter or 'none'}`",
                f"- Endpoint: `{server_base_url}`",
                f"- Artifacts root: `{payload['artifacts_root']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary_json


def main() -> None:
    args = build_parser().parse_args()
    defaults = load_benchmark_config()
    suites = selected_suites(args.suite)
    concurrency = resolve_concurrency(args.concurrent)
    run_slug = make_run_slug(args.base_model, args.adapter, args.label)
    model_cache_dir = runtime_dir() / "model_cache"
    model_spec = resolve_model_spec(
        base_model=args.base_model,
        adapter=args.adapter,
        tokenizer=args.tokenizer,
        served_model_name=args.served_model_name or args.base_model,
        cache_dir=model_cache_dir,
    )

    server_port = pick_port(args.port)
    categories = [item.strip() for item in args.bfcl_test_categories.split(",")] if args.bfcl_test_categories else None

    if args.server_base_url:
        host, port = parse_host_port(args.server_base_url)
        endpoint = type("Endpoint", (), {"host": host, "port": port, "model_name": model_spec.served_model_name, "root_url": args.server_base_url.rstrip("/"), "completions_url": args.server_base_url.rstrip("/") + "/completions"})()
        server_context = None
    elif args.skip_server_start:
        endpoint = type("Endpoint", (), {"host": args.host, "port": server_port, "model_name": model_spec.served_model_name, "root_url": f"http://{args.host}:{server_port}/v1", "completions_url": f"http://{args.host}:{server_port}/v1/completions"})()
        server_context = None
    else:
        server_context = managed_vllm_server(
            vllm_env_dir_or_name=args.vllm_env_dir,
            model_spec=model_spec,
            host=args.host,
            port=server_port,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            dtype=args.dtype,
            max_lora_rank=args.max_lora_rank,
            log_path=runtime_dir() / f"{run_slug}_vllm_server.log",
            repetition_penalty=args.repetition_penalty,
        )
        endpoint = None

    lm_eval_suites = defaults["lm_eval"]["suites"]

    if server_context is None:
        active_endpoint = endpoint
        if active_endpoint is None:
            raise BenchmarkError("No active endpoint was configured.")
        if "ceval" in suites:
            run_lm_eval_task(
                suite_name="ceval",
                task_name=lm_eval_suites["ceval"]["task"],
                endpoint=active_endpoint,
                tokenizer=model_spec.tokenizer,
                concurrency=concurrency,
                limit=min(args.max_samples, int(lm_eval_suites["ceval"]["limit"])),
                run_slug=run_slug,
                output_base_dir=args.output_base_dir,
                benchmark_env_dir=args.vllm_env_dir,
                max_gen_toks=lm_eval_suites.get("ceval", {}).get("max_gen_toks"),
                repetition_penalty=args.repetition_penalty,
            )
        if "ifeval" in suites:
            run_lm_eval_task(
                suite_name="ifeval",
                task_name=lm_eval_suites["ifeval"]["task"],
                endpoint=active_endpoint,
                tokenizer=model_spec.tokenizer,
                concurrency=concurrency,
                limit=min(args.max_samples, int(lm_eval_suites["ifeval"]["limit"])),
                run_slug=run_slug,
                output_base_dir=args.output_base_dir,
                benchmark_env_dir=args.vllm_env_dir,
                max_gen_toks=lm_eval_suites.get("ifeval", {}).get("max_gen_toks"),
                repetition_penalty=args.repetition_penalty,
            )
        if "bfcl_v4" in suites:
            run_bfcl(
                endpoint=active_endpoint,
                model_name=args.bfcl_model_name,
                max_samples=args.bfcl_max_samples,
                max_output_tokens=args.bfcl_max_output_tokens,
                concurrency=concurrency,
                categories=categories,
                run_slug=run_slug,
                output_base_dir=args.output_base_dir,
                benchmark_env_dir=args.vllm_env_dir,
                repetition_penalty=args.repetition_penalty,
            )
        write_summary(
            run_slug=run_slug,
            suites=suites,
            base_model=args.base_model,
            adapter=args.adapter,
            output_base_dir=args.output_base_dir,
            server_base_url=active_endpoint.root_url,
        )
    else:
        with server_context as active_endpoint:
            if active_endpoint is None:
                raise BenchmarkError("No active endpoint was configured.")
            if "ceval" in suites:
                run_lm_eval_task(
                    suite_name="ceval",
                    task_name=lm_eval_suites["ceval"]["task"],
                    endpoint=active_endpoint,
                    tokenizer=model_spec.tokenizer,
                    concurrency=concurrency,
                    limit=min(args.max_samples, int(lm_eval_suites["ceval"]["limit"])),
                    run_slug=run_slug,
                    output_base_dir=args.output_base_dir,
                    benchmark_env_dir=args.vllm_env_dir,
                    max_gen_toks=lm_eval_suites.get("ceval", {}).get("max_gen_toks"),
                    repetition_penalty=args.repetition_penalty,
                )
            if "ifeval" in suites:
                run_lm_eval_task(
                    suite_name="ifeval",
                    task_name=lm_eval_suites["ifeval"]["task"],
                    endpoint=active_endpoint,
                    tokenizer=model_spec.tokenizer,
                    concurrency=concurrency,
                    limit=min(args.max_samples, int(lm_eval_suites["ifeval"]["limit"])),
                    run_slug=run_slug,
                    output_base_dir=args.output_base_dir,
                    benchmark_env_dir=args.vllm_env_dir,
                    max_gen_toks=lm_eval_suites.get("ifeval", {}).get("max_gen_toks"),
                    repetition_penalty=args.repetition_penalty,
                )
            if "bfcl_v4" in suites:
                run_bfcl(
                    endpoint=active_endpoint,
                    model_name=args.bfcl_model_name,
                    max_samples=args.bfcl_max_samples,
                    max_output_tokens=args.bfcl_max_output_tokens,
                    concurrency=concurrency,
                    categories=categories,
                    run_slug=run_slug,
                    output_base_dir=args.output_base_dir,
                    benchmark_env_dir=args.vllm_env_dir,
                    repetition_penalty=args.repetition_penalty,
                )
            write_summary(
                run_slug=run_slug,
                suites=suites,
                base_model=args.base_model,
                adapter=args.adapter,
                output_base_dir=args.output_base_dir,
                server_base_url=active_endpoint.root_url,
            )

if __name__ == "__main__":
    main()
