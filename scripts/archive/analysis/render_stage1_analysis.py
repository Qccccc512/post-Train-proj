#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GROUP_ORDER = list("ABCDEFGHI")
OBSOLETE_PLOTS = (
    "adapter_size_best_vs_final.png",
    "keep_ratio_by_group.png",
    "eval_loss_by_group.png",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render stage1 analysis plots from fetched remote runs.")
    parser.add_argument(
        "--analysis-dir",
        default="analysis/stage1/2026-04-04_remote_stage1_qwen3_8b",
        help="Stage1 analysis directory containing raw/, plots/, and summary/.",
    )
    return parser


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_summary_rows(summary_csv: Path) -> list[dict]:
    numeric_fields = {
        "train_rows",
        "train_skipped_rows",
        "val_rows",
        "val_skipped_rows",
        "train_keep_ratio",
        "val_keep_ratio",
        "train_loss",
        "eval_loss",
        "train_runtime_s",
        "train_steps_per_second",
        "train_samples_per_second",
        "eval_runtime_s",
        "eval_steps_per_second",
        "eval_samples_per_second",
        "best_metric",
        "global_step",
        "max_steps",
        "best_adapter_bytes",
        "final_adapter_bytes",
        "adapter_size_ratio_final_to_best",
        "train_avg_seq_len",
        "val_avg_seq_len",
    }
    int_fields = {
        "train_rows",
        "train_skipped_rows",
        "val_rows",
        "val_skipped_rows",
        "global_step",
        "max_steps",
        "best_adapter_bytes",
        "final_adapter_bytes",
    }
    rows: list[dict] = []
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = dict(row)
            for field in numeric_fields:
                value = parsed.get(field)
                if value in (None, "", "N/A"):
                    parsed[field] = None
                elif field in int_fields:
                    parsed[field] = int(float(value))
                else:
                    parsed[field] = float(value)
            rows.append(parsed)
    rows.sort(key=lambda row: GROUP_ORDER.index(row["group"]))
    return rows


def load_loss_histories(raw_dir: Path, summary_rows: list[dict]) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    run_to_group = {row["run_name"]: row["group"] for row in summary_rows}
    train_histories: dict[str, list[dict]] = {}
    eval_histories: dict[str, list[dict]] = {}
    for run_dir in sorted(raw_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        group = run_to_group.get(run_dir.name)
        if not group:
            continue
        train_rows = read_jsonl(run_dir / "logs" / "train_metrics.jsonl")
        eval_rows = read_jsonl(run_dir / "logs" / "eval_metrics.jsonl")
        train_rows.sort(key=lambda row: row.get("step", 0))
        eval_rows.sort(key=lambda row: row.get("step", 0))
        train_histories[group] = train_rows
        eval_histories[group] = eval_rows
    return train_histories, eval_histories


def write_history_csvs(summary_dir: Path, train_histories: dict[str, list[dict]], eval_histories: dict[str, list[dict]]) -> None:
    train_path = summary_dir / "train_loss_history.csv"
    eval_path = summary_dir / "eval_loss_history.csv"
    with train_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "step", "epoch", "loss", "grad_norm", "learning_rate"])
        writer.writeheader()
        for group in GROUP_ORDER:
            for row in train_histories.get(group, []):
                writer.writerow(
                    {
                        "group": group,
                        "step": row.get("step"),
                        "epoch": row.get("epoch"),
                        "loss": row.get("loss"),
                        "grad_norm": row.get("grad_norm"),
                        "learning_rate": row.get("learning_rate"),
                    }
                )
    with eval_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "step", "epoch", "eval_loss"])
        writer.writeheader()
        for group in GROUP_ORDER:
            for row in eval_histories.get(group, []):
                writer.writerow(
                    {
                        "group": group,
                        "step": row.get("step"),
                        "epoch": row.get("epoch"),
                        "eval_loss": row.get("eval_loss"),
                    }
                )


def _plot_common(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)


def render_train_loss_plot(plots_dir: Path, train_histories: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    cmap = plt.get_cmap("tab10")
    for idx, group in enumerate(GROUP_ORDER):
        rows = train_histories.get(group, [])
        if not rows:
            continue
        steps = [row["step"] for row in rows]
        losses = [row["loss"] for row in rows]
        ax.plot(steps, losses, marker="o", markersize=3, linewidth=1.8, color=cmap(idx), label=group)
    _plot_common(ax, "Train Loss Over Steps by Group", "Optimizer Step", "Train Loss")
    ax.legend(title="Group", ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(plots_dir / "train_loss_over_steps_by_group.png", dpi=180)
    plt.close(fig)


def render_eval_loss_line_plot(plots_dir: Path, eval_histories: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    cmap = plt.get_cmap("tab10")
    for idx, group in enumerate(GROUP_ORDER):
        rows = eval_histories.get(group, [])
        if not rows:
            continue
        steps = [row["step"] for row in rows]
        losses = [row["eval_loss"] for row in rows]
        ax.plot(steps, losses, marker="o", markersize=5, linewidth=1.8, color=cmap(idx), label=group)
    _plot_common(ax, "Eval Loss Over Steps by Group", "Optimizer Step", "Eval Loss")
    ax.legend(title="Group", ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(plots_dir / "eval_loss_over_steps_by_group.png", dpi=180)
    plt.close(fig)


def render_final_eval_bar_plot(plots_dir: Path, summary_rows: list[dict]) -> None:
    labels = [row["group"] for row in summary_rows]
    values = [row["eval_loss"] for row in summary_rows]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.bar(labels, values, color=plt.get_cmap("tab10").colors[: len(labels)])
    _plot_common(ax, "Final Eval Loss by Group", "Group", "Eval Loss")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "final_eval_loss_by_group.png", dpi=180)
    plt.close(fig)


def render_throughput_plot(plots_dir: Path, summary_rows: list[dict]) -> None:
    labels = [row["group"] for row in summary_rows]
    values = [row["train_steps_per_second"] for row in summary_rows]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.bar(labels, values, color=plt.get_cmap("tab10").colors[: len(labels)])
    _plot_common(ax, "Train Throughput by Group", "Group", "Train Steps / Second")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "train_steps_per_second_by_group.png", dpi=180)
    plt.close(fig)


def render_seq_len_vs_throughput_plot(plots_dir: Path, summary_rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))
    cmap = plt.get_cmap("tab10")
    for idx, row in enumerate(summary_rows):
        ax.scatter(row["train_avg_seq_len"], row["train_steps_per_second"], color=cmap(idx), s=70)
        ax.annotate(row["group"], (row["train_avg_seq_len"], row["train_steps_per_second"]), xytext=(6, 5), textcoords="offset points")
    _plot_common(ax, "Average Train Seq Len vs Throughput", "Average Train Seq Len", "Train Steps / Second")
    fig.tight_layout()
    fig.savefig(plots_dir / "seq_len_vs_throughput.png", dpi=180)
    plt.close(fig)


def write_summary_markdown(analysis_dir: Path, summary_rows: list[dict]) -> None:
    summary_dir = analysis_dir / "summary"
    best_eval = min(summary_rows, key=lambda row: row["eval_loss"])
    fastest = max(summary_rows, key=lambda row: row["train_steps_per_second"])
    slowest = min(summary_rows, key=lambda row: row["train_steps_per_second"])
    lines = [
        "# Stage1 Remote Analysis",
        "",
        f"- Runs analyzed: `{len(summary_rows)}`",
        f"- Best eval loss: group `{best_eval['group']}` = `{best_eval['eval_loss']:.4f}`",
        f"- Fastest train throughput: group `{fastest['group']}` = `{fastest['train_steps_per_second']:.4f}` steps/s",
        f"- Slowest train throughput: group `{slowest['group']}` = `{slowest['train_steps_per_second']:.4f}` steps/s",
        "",
        "## Files",
        "",
        f"- Summary CSV: `{analysis_dir / 'summary' / 'stage1_runs_comparison.csv'}`",
        f"- Train loss trajectory plot: `{analysis_dir / 'plots' / 'train_loss_over_steps_by_group.png'}`",
        f"- Eval loss trajectory plot: `{analysis_dir / 'plots' / 'eval_loss_over_steps_by_group.png'}`",
        f"- Final eval loss bar plot: `{analysis_dir / 'plots' / 'final_eval_loss_by_group.png'}`",
        f"- Throughput plot: `{analysis_dir / 'plots' / 'train_steps_per_second_by_group.png'}`",
        f"- Seq len vs throughput plot: `{analysis_dir / 'plots' / 'seq_len_vs_throughput.png'}`",
        f"- Train loss history CSV: `{analysis_dir / 'summary' / 'train_loss_history.csv'}`",
        f"- Eval loss history CSV: `{analysis_dir / 'summary' / 'eval_loss_history.csv'}`",
    ]
    (summary_dir / "analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def remove_obsolete_plots(plots_dir: Path) -> None:
    for filename in OBSOLETE_PLOTS:
        path = plots_dir / filename
        if path.exists():
            path.unlink()


def main() -> None:
    args = build_parser().parse_args()
    analysis_dir = Path(args.analysis_dir)
    plots_dir = analysis_dir / "plots"
    summary_dir = analysis_dir / "summary"
    raw_dir = analysis_dir / "raw"
    plots_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = load_summary_rows(summary_dir / "stage1_runs_comparison.csv")
    train_histories, eval_histories = load_loss_histories(raw_dir, summary_rows)
    write_history_csvs(summary_dir, train_histories, eval_histories)
    render_train_loss_plot(plots_dir, train_histories)
    render_eval_loss_line_plot(plots_dir, eval_histories)
    render_final_eval_bar_plot(plots_dir, summary_rows)
    render_throughput_plot(plots_dir, summary_rows)
    render_seq_len_vs_throughput_plot(plots_dir, summary_rows)
    remove_obsolete_plots(plots_dir)
    write_summary_markdown(analysis_dir, summary_rows)


if __name__ == "__main__":
    main()
