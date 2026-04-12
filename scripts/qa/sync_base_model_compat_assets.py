#!/usr/bin/env python3
from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse

from scripts.common.model_artifacts import (
    DEFAULT_BASE_MODEL_ID,
    default_smoke_output_dir,
    resolve_model_dir,
    resolve_output_dir,
    sync_base_model_compat_assets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fill missing full-model compatibility assets from the base model repo."
    )
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--base-model-id", default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--revision", default="main")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_dir = resolve_model_dir(args.model_dir)
    output_dir = resolve_output_dir(args.output_dir, default_dir=default_smoke_output_dir())
    manifest = sync_base_model_compat_assets(
        model_dir=model_dir,
        base_model_id=args.base_model_id,
        output_manifest_path=output_dir / "compat_assets_manifest.json",
        revision=args.revision,
    )
    print(
        "[sync_base_model_compat_assets] "
        f"model_dir={manifest['model_dir']} copied={manifest['copied']} "
        f"already_present={manifest['already_present']} "
        f"missing_upstream={manifest['missing_upstream']}"
    )


if __name__ == "__main__":
    main()
