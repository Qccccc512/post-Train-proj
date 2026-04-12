#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = [
    "model_label",
    "model_id",
    "model_dtype",
    "model_quantization",
    "profile_label",
    "input_len",
    "output_len",
    "num_prompts",
    "max_concurrency",
    "request_rate",
    "completed",
    "failed",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "p95_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
    "max_output_tokens_per_s",
    "max_concurrent_requests",
    "duration",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize vllm bench serve result files.")
    parser.add_argument("--run-dir", required=True, help="Root directory produced by scripts/run_vllm_serve_bench.sh")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def collect_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    results_root = run_dir / "results"
    for result_path in sorted(results_root.glob("*/profiles/*/result.json")):
        model_dir = result_path.parents[2]
        profile_dir = result_path.parent
        model_manifest = read_json(model_dir / "model_manifest.json")
        profile_manifest = read_json(profile_dir / "profile_manifest.json")
        result_payload = read_json(result_path)

        row = {
            "model_label": model_manifest.get("model_label"),
            "model_id": model_manifest.get("model_id"),
            "model_dtype": model_manifest.get("model_dtype"),
            "model_quantization": model_manifest.get("model_quantization"),
            "profile_label": profile_manifest.get("profile_label"),
            "input_len": profile_manifest.get("input_len"),
            "output_len": profile_manifest.get("output_len"),
            "num_prompts": profile_manifest.get("num_prompts"),
            "max_concurrency": profile_manifest.get("max_concurrency"),
            "request_rate": profile_manifest.get("request_rate"),
            "completed": result_payload.get("completed"),
            "failed": result_payload.get("failed"),
            "request_throughput": result_payload.get("request_throughput"),
            "output_throughput": result_payload.get("output_throughput"),
            "total_token_throughput": result_payload.get("total_token_throughput"),
            "mean_ttft_ms": result_payload.get("mean_ttft_ms"),
            "p95_ttft_ms": result_payload.get("p95_ttft_ms"),
            "p99_ttft_ms": result_payload.get("p99_ttft_ms"),
            "mean_tpot_ms": result_payload.get("mean_tpot_ms"),
            "p95_tpot_ms": result_payload.get("p95_tpot_ms"),
            "p99_tpot_ms": result_payload.get("p99_tpot_ms"),
            "mean_itl_ms": result_payload.get("mean_itl_ms"),
            "p95_itl_ms": result_payload.get("p95_itl_ms"),
            "p99_itl_ms": result_payload.get("p99_itl_ms"),
            "mean_e2el_ms": result_payload.get("mean_e2el_ms"),
            "p95_e2el_ms": result_payload.get("p95_e2el_ms"),
            "p99_e2el_ms": result_payload.get("p99_e2el_ms"),
            "max_output_tokens_per_s": result_payload.get("max_output_tokens_per_s"),
            "max_concurrent_requests": result_payload.get("max_concurrent_requests"),
            "duration": result_payload.get("duration"),
            "result_json": str(result_path),
            "bench_log": str(profile_dir / "bench.log"),
            "server_log": str(model_dir / "server" / "server.log"),
        }
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS + ["result_json", "bench_log", "server_log"])
        writer.writeheader()
        writer.writerows(rows)


def write_json_summary(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "row_count": len(rows),
        "rows": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# vLLM Serve Benchmark Summary",
        "",
        f"- Rows: {len(rows)}",
        "",
        "| Model | Profile | In | Out | Prompts | Concurrency | Req/s | Req TP | Out TP | TTFT P99 | TPOT P99 | E2EL P99 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {model_label} | {profile_label} | {input_len} | {output_len} | {num_prompts} | {max_concurrency} | {request_rate} | {request_throughput} | {output_throughput} | {p99_ttft_ms} | {p99_tpot_ms} | {p99_e2el_ms} |".format(
                model_label=fmt_value(row.get("model_label")),
                profile_label=fmt_value(row.get("profile_label")),
                input_len=fmt_value(row.get("input_len")),
                output_len=fmt_value(row.get("output_len")),
                num_prompts=fmt_value(row.get("num_prompts")),
                max_concurrency=fmt_value(row.get("max_concurrency")),
                request_rate=fmt_value(row.get("request_rate")),
                request_throughput=fmt_value(row.get("request_throughput")),
                output_throughput=fmt_value(row.get("output_throughput")),
                p99_ttft_ms=fmt_value(row.get("p99_ttft_ms")),
                p99_tpot_ms=fmt_value(row.get("p99_tpot_ms")),
                p99_e2el_ms=fmt_value(row.get("p99_e2el_ms")),
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- Raw JSON, logs, and manifests remain under `results/<model>/profiles/<profile>/`.",
            "- Server logs remain under `results/<model>/server/`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    rows = collect_rows(run_dir)
    write_json_summary(rows, Path(args.output_json))
    write_csv(rows, Path(args.output_csv))
    write_markdown(rows, Path(args.output_md))


if __name__ == "__main__":
    main()
