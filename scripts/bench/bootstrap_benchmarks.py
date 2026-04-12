#!/usr/bin/env python3
"""Bootstrap benchmark frameworks by cloning from git.

This script is designed to run independently without requiring benchmark
dependencies installed. It only uses Python standard library and git.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def repo_root() -> Path:
    """Get the repository root directory."""
    return Path(__file__).resolve().parent.parent.parent


def load_framework_specs() -> dict:
    """Load framework specs from the lock file."""
    lock_path = repo_root() / "benchmark" / "framework_versions.lock.json"
    with open(lock_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run(command: list[str], cwd: Path | None = None) -> None:
    """Run a command with error checking."""
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def ensure_checkout(force: bool = False) -> None:
    """Clone and checkout benchmark frameworks to pinned versions."""
    specs = load_framework_specs()
    root = repo_root()
    
    for name, spec in specs.items():
        local_dir = root / spec["local_dir"]
        
        if force and local_dir.exists():
            print(f"Removing existing {name} directory: {local_dir}")
            shutil.rmtree(local_dir)
        
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        
        if not local_dir.exists():
            print(f"Cloning {name} from {spec['repo_url']}...")
            run(["git", "clone", spec["repo_url"], str(local_dir)])
        
        print(f"Checking out {name} to {spec['ref']}...")
        run(["git", "fetch", "--tags", "--all"], cwd=local_dir)
        run(["git", "checkout", spec["ref"]], cwd=local_dir)
        
        current_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(local_dir),
            text=True,
        ).strip()
        print(f"{name}: {current_commit} -> {local_dir}")
        
        # Validate BFCL workdir
        if name == "bfcl":
            workdir_subpath = spec.get("workdir_subpath")
            if workdir_subpath:
                workdir = local_dir / workdir_subpath
                if not workdir.exists():
                    raise RuntimeError(f"Expected BFCL workdir missing after checkout: {workdir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clone pinned benchmark framework sources into benchmark/."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and re-clone existing framework directories.",
    )
    args = parser.parse_args()
    ensure_checkout(force=args.force)


if __name__ == "__main__":
    main()
