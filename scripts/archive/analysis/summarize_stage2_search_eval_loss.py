#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize eval loss from Stage 2 search runs.")
    parser.add_argument("run_names", nargs="+", help="Run names under runs/ to summarize.")
    parser.add_argument("--runs-dir", default="runs", help="Root directory containing run folders.")
    parser.add_argument("--output-md", required=True, help="Path to write markdown summary.")
    parser.add_argument("--output-json", required=True, help="Path to write json summary.")
    return parser


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_record(runs_dir: Path, run_name: str) -> dict[str, Any]:
    run_dir = runs_dir / run_name
    record: dict[str, Any] = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "status": "ok",
        "learning_rate": None,
        "lora_r": None,
        "max_steps": None,
        "warmup_steps": None,
        "best_eval_loss": None,
        "final_eval_loss": None,
        "best_global_step": None,
        "global_step": None,
        "best_model_checkpoint": None,
    }

    if not run_dir.exists():
        record["status"] = "missing_run_dir"
        return record

    train_cfg = read_json(run_dir / "configs" / "train_config_resolved.json") or {}
    record["learning_rate"] = train_cfg.get("learning_rate")
    lora_cfg = train_cfg.get("lora") if isinstance(train_cfg.get("lora"), dict) else {}
    record["lora_r"] = lora_cfg.get("r")
    record["max_steps"] = train_cfg.get("max_steps")
    record["warmup_steps"] = train_cfg.get("warmup_steps", train_cfg.get("warm_up_steps"))

    trainer_state = read_json(run_dir / "logs" / "trainer_state.json") or {}
    eval_metrics = read_json(run_dir / "logs" / "eval_metrics.json") or {}

    log_history = trainer_state.get("log_history") if isinstance(trainer_state.get("log_history"), list) else []
    eval_losses = [
        maybe_float(entry.get("eval_loss"))
        for entry in log_history
        if isinstance(entry, dict) and entry.get("eval_loss") is not None
    ]
    eval_losses = [value for value in eval_losses if value is not None]

    best_eval_loss = maybe_float(trainer_state.get("best_metric"))
    if best_eval_loss is None and eval_losses:
        best_eval_loss = min(eval_losses)
    record["best_eval_loss"] = best_eval_loss

    record["final_eval_loss"] = maybe_float(eval_metrics.get("eval_loss"))
    record["best_global_step"] = trainer_state.get("best_global_step")
    record["global_step"] = trainer_state.get("global_step")
    record["best_model_checkpoint"] = trainer_state.get("best_model_checkpoint")

    if not trainer_state:
        record["status"] = "missing_trainer_state"

    return record


def format_number(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def to_markdown(records: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage 2 Search Eval Loss Summary",
        "",
        f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- total_runs: {len(records)}",
        "",
        "| run_name | lr | lora_r | warmup_steps | best_eval_loss | final_eval_loss | best_global_step | status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in records:
        lines.append(
            "| {run_name} | {lr} | {r} | {warmup} | {best} | {final} | {best_step} | {status} |".format(
                run_name=row["run_name"],
                lr=format_number(row["learning_rate"]),
                r=format_number(row["lora_r"], digits=0),
                warmup=format_number(row["warmup_steps"], digits=0),
                best=format_number(row["best_eval_loss"]),
                final=format_number(row["final_eval_loss"]),
                best_step=format_number(row["best_global_step"], digits=0),
                status=row["status"],
            )
        )

    ranked = [row for row in records if row.get("best_eval_loss") is not None]
    ranked.sort(key=lambda item: float(item["best_eval_loss"]))
    if ranked:
        lines.extend(
            [
                "",
                "## Ranking by best_eval_loss",
                "",
            ]
        )
        for idx, row in enumerate(ranked, start=1):
            lines.append(
                f"{idx}. {row['run_name']}: best_eval_loss={format_number(row['best_eval_loss'])}, "
                f"lr={format_number(row['learning_rate'])}, r={format_number(row['lora_r'], digits=0)}"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    runs_dir = Path(args.runs_dir)
    output_md = Path(args.output_md)
    output_json = Path(args.output_json)

    records = [extract_record(runs_dir, run_name) for run_name in args.run_names]

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    output_md.write_text(to_markdown(records), encoding="utf-8")


if __name__ == "__main__":
    main()
