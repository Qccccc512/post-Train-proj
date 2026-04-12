#!/usr/bin/env python3
from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from scripts.common.runtime_utils import ensure_dir, now_timestamp, resolve_path, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize CEval / IFEval / BFCL benchmark outputs across multiple candidate runs."
    )
    parser.add_argument(
        "run_dirs",
        nargs="*",
        help="Benchmark run directories, or run names under --benchmark-root.",
    )
    parser.add_argument(
        "--benchmark-root",
        help="Root directory that contains multiple benchmark run folders (for example analysis/stage2_best_benchmarks/benchmarks).",
    )
    parser.add_argument(
        "--reference",
        help="Reference run name or directory used for delta columns. If omitted, a single base-like run is auto-detected when possible.",
    )
    parser.add_argument("--output-md", help="Optional markdown output path.")
    parser.add_argument("--output-json", help="Optional json output path.")
    return parser


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().upper() == "N/A":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_percent_string(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.upper() == "N/A":
            return None
        if text.endswith("%"):
            text = text[:-1]
        parsed = maybe_float(text)
        return parsed
    return maybe_float(value)


def fraction_to_percent(value: Any) -> float | None:
    parsed = maybe_float(value)
    if parsed is None:
        return None
    return parsed * 100.0


def latest_result_json(suite_dir: Path) -> Path | None:
    candidates = sorted(suite_dir.glob("*/results_*.json"))
    if not candidates:
        return None
    return candidates[-1]


def format_percent(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}%"


def format_delta(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.{digits}f}pt"


def resolve_run_dir(raw: str, benchmark_root: Path | None) -> Path:
    direct = resolve_path(raw)
    if direct.exists():
        return direct
    if benchmark_root is not None:
        candidate = benchmark_root / raw
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to resolve benchmark run directory: {raw}")


def discover_run_dirs(benchmark_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for child in sorted(benchmark_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "summary" / "benchmark_run_summary.json").exists():
            candidates.append(child)
    return candidates


def load_ceval_metrics(run_dir: Path) -> dict[str, Any]:
    result_path = latest_result_json(run_dir / "ceval")
    if result_path is None:
        return {}
    payload = read_json_if_exists(result_path) or {}
    result = (payload.get("results") or {}).get("ceval-valid") or {}
    sample_info = (payload.get("n-samples") or {}).get("ceval-valid") or {}
    return {
        "ceval_acc": fraction_to_percent(result.get("acc_norm,none")),
        "ceval_stderr": fraction_to_percent(result.get("acc_norm_stderr,none")),
        "ceval_effective_samples": sample_info.get("effective"),
        "ceval_original_samples": sample_info.get("original"),
        "ceval_result_path": str(result_path),
    }


def load_ifeval_metrics(run_dir: Path) -> dict[str, Any]:
    result_path = latest_result_json(run_dir / "ifeval")
    if result_path is None:
        return {}
    payload = read_json_if_exists(result_path) or {}
    result = (payload.get("results") or {}).get("ifeval") or {}
    sample_info = (payload.get("n-samples") or {}).get("ifeval") or {}
    return {
        "ifeval_strict": fraction_to_percent(result.get("prompt_level_strict_acc,none")),
        "ifeval_strict_stderr": fraction_to_percent(result.get("prompt_level_strict_acc_stderr,none")),
        "ifeval_loose": fraction_to_percent(result.get("prompt_level_loose_acc,none")),
        "ifeval_loose_stderr": fraction_to_percent(result.get("prompt_level_loose_acc_stderr,none")),
        "ifeval_effective_samples": sample_info.get("effective"),
        "ifeval_original_samples": sample_info.get("original"),
        "ifeval_result_path": str(result_path),
    }


def load_bfcl_metrics(run_dir: Path) -> dict[str, Any]:
    csv_path = run_dir / "bfcl_v4" / "project_root" / "score" / "data_overall.csv"
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        first_row = next(reader, None) or {}
    return {
        "bfcl_overall": parse_percent_string(first_row.get("Overall Acc")),
        "bfcl_non_live": parse_percent_string(first_row.get("Non-Live AST Acc")),
        "bfcl_live": parse_percent_string(first_row.get("Live Acc")),
        "bfcl_multi_turn": parse_percent_string(first_row.get("Multi Turn Acc")),
        "bfcl_web_search": parse_percent_string(first_row.get("Web Search Acc")),
        "bfcl_memory": parse_percent_string(first_row.get("Memory Acc")),
        "bfcl_relevance": parse_percent_string(first_row.get("Relevance Detection")),
        "bfcl_irrelevance": parse_percent_string(first_row.get("Irrelevance Detection")),
        "bfcl_overall_csv_path": str(csv_path),
    }


def infer_candidate_name(run_dir: Path, summary_payload: dict[str, Any] | None) -> str:
    if summary_payload:
        run_slug = summary_payload.get("run_slug")
        if isinstance(run_slug, str) and run_slug.strip():
            return run_slug.strip()
    return run_dir.name


def load_record(run_dir: Path) -> dict[str, Any]:
    summary_payload = read_json_if_exists(run_dir / "summary" / "benchmark_run_summary.json")
    adapter = summary_payload.get("adapter") if isinstance(summary_payload, dict) else None
    record: dict[str, Any] = {
        "run_dir": str(run_dir),
        "run_name": infer_candidate_name(run_dir, summary_payload),
        "base_model": summary_payload.get("base_model") if isinstance(summary_payload, dict) else None,
        "adapter": adapter,
        "is_base": not adapter,
    }
    record.update(load_ceval_metrics(run_dir))
    record.update(load_ifeval_metrics(run_dir))
    record.update(load_bfcl_metrics(run_dir))
    return record


def resolve_reference(records: list[dict[str, Any]], raw_reference: str | None) -> dict[str, Any] | None:
    if raw_reference:
        reference_path = resolve_path(raw_reference)
        for record in records:
            if record["run_name"] == raw_reference:
                return record
            if Path(record["run_dir"]) == reference_path:
                return record
            if Path(record["run_dir"]).name == raw_reference:
                return record
        raise ValueError(f"Unable to match reference run: {raw_reference}")

    base_like = [record for record in records if record.get("is_base")]
    if len(base_like) == 1:
        return base_like[0]
    return None


def add_deltas(records: list[dict[str, Any]], reference: dict[str, Any] | None) -> None:
    if reference is None:
        return
    delta_fields = [
        ("ceval_acc", "delta_ceval_acc"),
        ("ifeval_strict", "delta_ifeval_strict"),
        ("ifeval_loose", "delta_ifeval_loose"),
        ("bfcl_overall", "delta_bfcl_overall"),
        ("bfcl_non_live", "delta_bfcl_non_live"),
        ("bfcl_live", "delta_bfcl_live"),
        ("bfcl_multi_turn", "delta_bfcl_multi_turn"),
        ("bfcl_web_search", "delta_bfcl_web_search"),
        ("bfcl_memory", "delta_bfcl_memory"),
    ]
    for record in records:
        for source_key, delta_key in delta_fields:
            base_value = reference.get(source_key)
            current_value = record.get(source_key)
            if base_value is None or current_value is None:
                record[delta_key] = None
            else:
                record[delta_key] = float(current_value) - float(base_value)


def rank_records(records: list[dict[str, Any]], reference: dict[str, Any] | None) -> list[dict[str, Any]]:
    def sort_key(record: dict[str, Any]) -> tuple[float, float, float, float]:
        if reference and record["run_name"] == reference["run_name"]:
            return (-1e9, -1e9, -1e9, -1e9)
        return (
            record.get("bfcl_overall") or -1e9,
            record.get("bfcl_live") or -1e9,
            record.get("bfcl_multi_turn") or -1e9,
            record.get("ceval_acc") or -1e9,
        )

    return sorted(records, key=sort_key, reverse=True)


def to_markdown(records: list[dict[str, Any]], reference: dict[str, Any] | None) -> str:
    lines = [
        "# Benchmark Candidate Summary",
        "",
        f"- generated_at: {now_timestamp()}",
        f"- total_runs: {len(records)}",
    ]
    if reference:
        lines.append(f"- reference_run: {reference['run_name']}")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| run_name | kind | C-Eval | IFEval Strict | IFEval Loose | BFCL Overall | BFCL Non-Live | BFCL Live | BFCL Multi-Turn | Web Search | Memory |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in records:
        lines.append(
            "| {run_name} | {kind} | {ceval} | {strict} | {loose} | {overall} | {non_live} | {live} | {multi_turn} | {web} | {memory} |".format(
                run_name=record["run_name"],
                kind="base" if record.get("is_base") else "adapter",
                ceval=format_percent(record.get("ceval_acc")),
                strict=format_percent(record.get("ifeval_strict")),
                loose=format_percent(record.get("ifeval_loose")),
                overall=format_percent(record.get("bfcl_overall")),
                non_live=format_percent(record.get("bfcl_non_live")),
                live=format_percent(record.get("bfcl_live")),
                multi_turn=format_percent(record.get("bfcl_multi_turn")),
                web=format_percent(record.get("bfcl_web_search")),
                memory=format_percent(record.get("bfcl_memory")),
            )
        )

    if reference:
        lines.extend(
            [
                "",
                "## Deltas vs Reference",
                "",
                "| run_name | dC-Eval | dIFEval Strict | dIFEval Loose | dBFCL Overall | dBFCL Live | dBFCL Multi-Turn | dWeb Search | dMemory |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for record in records:
            lines.append(
                "| {run_name} | {ceval} | {strict} | {loose} | {overall} | {live} | {multi_turn} | {web} | {memory} |".format(
                    run_name=record["run_name"],
                    ceval=format_delta(record.get("delta_ceval_acc")),
                    strict=format_delta(record.get("delta_ifeval_strict")),
                    loose=format_delta(record.get("delta_ifeval_loose")),
                    overall=format_delta(record.get("delta_bfcl_overall")),
                    live=format_delta(record.get("delta_bfcl_live")),
                    multi_turn=format_delta(record.get("delta_bfcl_multi_turn")),
                    web=format_delta(record.get("delta_bfcl_web_search")),
                    memory=format_delta(record.get("delta_bfcl_memory")),
                )
            )

    ranked = rank_records(records, reference)
    lines.extend(
        [
            "",
            "## Selection Lens",
            "",
            "- 优先看 `BFCL Live` 与 `BFCL Multi-Turn` 是否相对 base 有稳定提升，这两项最贴近当前项目的 tool-calling 目标。",
            "- `C-Eval` 与 `IFEval` 更适合作为护栏指标：如果 adapter 带来明显回退，需要谨慎。",
            "- 如果两个 adapter 的全量 benchmark 几乎持平，再回到 Stage 2 搜索结果，用更低的 `eval_loss` 作为 tie-breaker。",
            "",
            "## BFCL-Oriented Ranking",
            "",
        ]
    )
    for idx, record in enumerate(ranked, start=1):
        lines.append(
            f"{idx}. {record['run_name']}: BFCL Overall={format_percent(record.get('bfcl_overall'))}, "
            f"Live={format_percent(record.get('bfcl_live'))}, "
            f"Multi-Turn={format_percent(record.get('bfcl_multi_turn'))}, "
            f"C-Eval={format_percent(record.get('ceval_acc'))}"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    benchmark_root = resolve_path(args.benchmark_root) if args.benchmark_root else None

    if args.run_dirs:
        run_dirs = [resolve_run_dir(raw, benchmark_root) for raw in args.run_dirs]
    elif benchmark_root is not None:
        run_dirs = discover_run_dirs(benchmark_root)
    else:
        raise SystemExit("Provide at least one run_dir, or use --benchmark-root for auto-discovery.")

    if not run_dirs:
        raise SystemExit("No benchmark run directories found.")

    records = [load_record(run_dir) for run_dir in run_dirs]
    reference = resolve_reference(records, args.reference)
    add_deltas(records, reference)

    markdown = to_markdown(records, reference)
    sys.stdout.write(markdown)

    payload = {
        "generated_at": now_timestamp(),
        "reference_run": reference["run_name"] if reference else None,
        "records": records,
    }

    if args.output_json:
        output_json = resolve_path(args.output_json)
        ensure_dir(output_json.parent)
        write_json(output_json, payload)
    if args.output_md:
        output_md = resolve_path(args.output_md)
        ensure_dir(output_md.parent)
        output_md.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
