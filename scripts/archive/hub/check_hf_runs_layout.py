#!/usr/bin/env python3
from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
from pathlib import Path

from huggingface_hub import HfApi

from scripts.hub.hf_repo_sync import load_hf_config
from scripts.common.runtime_utils import get_hf_token, now_timestamp, resolve_hf_repo_id, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check Hugging Face runs/ directory layout hygiene.")
    parser.add_argument("--hf-config", default="configs/hf/default.yaml")
    parser.add_argument("--run-marker", default="stage1", help="Marker used to detect stage1 runs.")
    parser.add_argument("--stage1-folder", default="stage1_qwen3_8b", help="Expected nested folder under runs/.")
    parser.add_argument(
        "--json-output",
        default="runs/_hf_cache/hf_runs_layout_check.json",
        help="Write inspection summary to this local json file.",
    )
    parser.add_argument(
        "--strict-stage1",
        action="store_true",
        help="Exit with non-zero code if stage1 runs are still found directly under runs/.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    hf_config = load_hf_config(args.hf_config)
    repo_type = hf_config.get("repo_type", "model")

    token = get_hf_token(hf_config)
    repo_id = resolve_hf_repo_id(hf_config)
    api = HfApi(token=token)

    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)

    direct_stage1 = set()
    nested_stage1 = set()
    top_level_runs = set()

    stage1_prefix = f"runs/{args.stage1_folder}/"

    for path in files:
        if not path.startswith("runs/"):
            continue
        rel = path[len("runs/") :]
        if "/" not in rel:
            continue

        top = rel.split("/", 1)[0]
        top_level_runs.add(top)

        if rel.startswith(f"{args.stage1_folder}/"):
            sub = rel[len(f"{args.stage1_folder}/") :]
            if "/" in sub:
                run_name = sub.split("/", 1)[0]
                if args.run_marker.lower() in run_name.lower():
                    nested_stage1.add(run_name)
            continue

        run_name = top
        if args.run_marker.lower() in run_name.lower():
            direct_stage1.add(run_name)

    payload = {
        "timestamp": now_timestamp(),
        "repo_id": repo_id,
        "repo_type": repo_type,
        "stage1_folder": args.stage1_folder,
        "run_marker": args.run_marker,
        "direct_stage1_count": len(direct_stage1),
        "direct_stage1_dirs": sorted(direct_stage1),
        "nested_stage1_count": len(nested_stage1),
        "nested_stage1_dirs": sorted(nested_stage1),
        "top_level_runs_dir_count": len(top_level_runs),
        "top_level_runs_dirs": sorted(top_level_runs),
        "stage1_prefix": stage1_prefix,
    }

    output_path = Path(args.json_output)
    write_json(output_path, payload)

    print(f"repo={repo_id}")
    print(f"direct_stage1_count={payload['direct_stage1_count']}")
    print("direct_stage1_dirs=" + (", ".join(payload["direct_stage1_dirs"]) if payload["direct_stage1_dirs"] else "[]"))
    print(f"nested_stage1_count={payload['nested_stage1_count']}")
    print("nested_stage1_dirs=" + (", ".join(payload["nested_stage1_dirs"]) if payload["nested_stage1_dirs"] else "[]"))
    print(f"report={output_path}")

    if args.strict_stage1 and payload["direct_stage1_count"] > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
