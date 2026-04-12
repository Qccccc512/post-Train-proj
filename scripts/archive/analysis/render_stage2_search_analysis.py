#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_ANALYSIS_DIR = "analysis/stage2/2026-04-07_remote_stage2_search_qwen3_8b"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render Stage2 search analysis from fetched remote logs/meta/configs.")
    parser.add_argument("--analysis-dir", default=DEFAULT_ANALYSIS_DIR)
    return parser


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def slug_from_run_name(run_name: str) -> str:
    if "stage2_search_" in run_name:
        return run_name.split("stage2_search_", 1)[1]
    return run_name


def extract_run_record(run_dir: Path) -> dict[str, Any]:
    run_name = run_dir.name
    train_cfg = read_json(run_dir / "configs" / "train_config_resolved.json") or {}
    trainer_state = read_json(run_dir / "logs" / "trainer_state.json") or {}
    eval_metrics = read_json(run_dir / "logs" / "eval_metrics.json") or {}
    train_result = read_json(run_dir / "logs" / "train_result_metrics.json") or {}
    preprocess_summary = read_json(run_dir / "meta" / "preprocess_summary.json") or {}

    lora_cfg = train_cfg.get("lora") if isinstance(train_cfg.get("lora"), dict) else {}

    best_eval = maybe_float(trainer_state.get("best_metric"))
    if best_eval is None:
        eval_losses = [
            maybe_float(item.get("eval_loss"))
            for item in trainer_state.get("log_history", [])
            if isinstance(item, dict) and item.get("eval_loss") is not None
        ]
        eval_losses = [x for x in eval_losses if x is not None]
        if eval_losses:
            best_eval = min(eval_losses)

    record = {
        "run_name": run_name,
        "run_slug": slug_from_run_name(run_name),
        "learning_rate": maybe_float(train_cfg.get("learning_rate")),
        "lora_r": lora_cfg.get("r"),
        "max_steps": train_cfg.get("max_steps"),
        "warmup_steps": train_cfg.get("warmup_steps", train_cfg.get("warm_up_steps")),
        "best_eval_loss": best_eval,
        "final_eval_loss": maybe_float(eval_metrics.get("eval_loss")),
        "train_loss": maybe_float(train_result.get("train_loss")),
        "train_runtime": maybe_float(train_result.get("train_runtime")),
        "train_steps_per_second": maybe_float(train_result.get("train_steps_per_second")),
        "best_global_step": trainer_state.get("best_global_step"),
        "global_step": trainer_state.get("global_step"),
        "train_rows": ((preprocess_summary.get("train") or {}).get("rows")),
        "val_rows": ((preprocess_summary.get("validation") or {}).get("rows")),
        "status": "ok" if trainer_state else "missing_trainer_state",
    }
    return record


def render_eval_loss_bar(plots_dir: Path, records: list[dict[str, Any]], key: str, title: str, filename: str) -> None:
    filtered = [row for row in records if row.get(key) is not None]
    filtered.sort(key=lambda row: float(row[key]))
    labels = [row["run_slug"] for row in filtered]
    values = [float(row[key]) for row in filtered]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    bars = ax.bar(range(len(labels)), values, color="#2F6F8F")
    ax.set_title(title)
    ax.set_ylabel(key)
    ax.set_xlabel("run")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    for idx, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{values[idx]:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(plots_dir / filename, dpi=180)
    plt.close(fig)


def render_loss_curves(plots_dir: Path, raw_dir: Path, run_dirs: list[Path], metric_key: str, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    cmap = plt.get_cmap("tab10")

    for idx, run_dir in enumerate(sorted(run_dirs)):
        run_slug = slug_from_run_name(run_dir.name)
        rows = read_jsonl(run_dir / "logs" / ("train_metrics.jsonl" if metric_key == "loss" else "eval_metrics.jsonl"))
        points = [(row.get("step"), maybe_float(row.get(metric_key))) for row in rows]
        points = [(step, val) for step, val in points if step is not None and val is not None]
        if not points:
            continue
        steps = [step for step, _ in points]
        vals = [val for _, val in points]
        ax.plot(steps, vals, marker="o", markersize=3, linewidth=1.7, color=cmap(idx % 10), label=run_slug)

    ax.set_title(title)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(metric_key)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / filename, dpi=180)
    plt.close(fig)


def write_summary_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_name",
        "run_slug",
        "learning_rate",
        "lora_r",
        "max_steps",
        "warmup_steps",
        "best_eval_loss",
        "final_eval_loss",
        "train_loss",
        "train_runtime",
        "train_steps_per_second",
        "best_global_step",
        "global_step",
        "train_rows",
        "val_rows",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def write_summary_md(path: Path, records: list[dict[str, Any]], plots_dir: Path, summary_csv: Path) -> None:
    ranked = [row for row in records if row.get("best_eval_loss") is not None]
    ranked.sort(key=lambda row: float(row["best_eval_loss"]))

    lines = [
        "# Stage2 Search Remote Analysis",
        "",
        f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- total_runs: {len(records)}",
        f"- summary_csv: {summary_csv}",
        f"- plot_best_eval_loss: {plots_dir / 'best_eval_loss_by_run.png'}",
        f"- plot_final_eval_loss: {plots_dir / 'final_eval_loss_by_run.png'}",
        f"- plot_train_loss_curve: {plots_dir / 'train_loss_over_steps_by_run.png'}",
        f"- plot_eval_loss_curve: {plots_dir / 'eval_loss_over_steps_by_run.png'}",
        "",
        "## Ranking (by best_eval_loss)",
        "",
    ]

    if not ranked:
        lines.append("- No valid best_eval_loss found.")
    else:
        for idx, row in enumerate(ranked, start=1):
            lines.append(
                f"{idx}. {row['run_name']} | best_eval_loss={row['best_eval_loss']:.6f} | "
                f"final_eval_loss={row['final_eval_loss'] if row['final_eval_loss'] is not None else 'N/A'} | "
                f"lr={row['learning_rate']} | r={row['lora_r']}"
            )

    lines.extend(
        [
            "",
            "## Quick Notes",
            "",
            "- 仅基于训练日志指标（eval_loss/train_loss）排序，不包含外部 benchmark。",
            "- 该分析目录只包含 logs/meta/configs，不包含 checkpoints/adapters。",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    analysis_dir = Path(args.analysis_dir)
    raw_dir = analysis_dir / "raw"
    plots_dir = analysis_dir / "plots"
    summary_dir = analysis_dir / "summary"

    plots_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(path for path in raw_dir.iterdir() if path.is_dir())
    records = [extract_run_record(run_dir) for run_dir in run_dirs]

    summary_csv = summary_dir / "stage2_search_runs_comparison.csv"
    summary_md = summary_dir / "analysis_summary.md"

    write_summary_csv(summary_csv, records)
    render_eval_loss_bar(plots_dir, records, "best_eval_loss", "Best Eval Loss by Run", "best_eval_loss_by_run.png")
    render_eval_loss_bar(plots_dir, records, "final_eval_loss", "Final Eval Loss by Run", "final_eval_loss_by_run.png")
    render_loss_curves(plots_dir, raw_dir, run_dirs, "loss", "Train Loss Over Steps by Run", "train_loss_over_steps_by_run.png")
    render_loss_curves(plots_dir, raw_dir, run_dirs, "eval_loss", "Eval Loss Over Steps by Run", "eval_loss_over_steps_by_run.png")
    write_summary_md(summary_md, records, plots_dir, summary_csv)

    print(f"analysis_dir={analysis_dir}")
    print(f"runs={len(records)}")
    print(f"summary_csv={summary_csv}")
    print(f"summary_md={summary_md}")


if __name__ == "__main__":
    main()
