#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from scripts.common.runtime_utils import read_json


def is_hf_repo_id(value: str | None) -> bool:
    if not value:
        return False
    return "/" in value and not value.startswith("/") and not value.startswith("http")


def detect_base_model_id(source_dir: Path, explicit: str | None) -> str:
    if explicit:
        return explicit

    adapter_config_path = source_dir / "adapter_config.json"
    if adapter_config_path.exists():
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        candidate = adapter_config.get("base_model_name_or_path")
        if is_hf_repo_id(candidate):
            return candidate

    run_dir = source_dir.parent.parent
    train_config_path = run_dir / "configs" / "train_config_resolved.json"
    if train_config_path.exists():
        candidate = read_json(train_config_path).get("model_name")
        if is_hf_repo_id(candidate):
            return candidate

    raise RuntimeError(
        "Unable to determine a valid Hugging Face base model id. "
        "Pass --base-model-id explicitly."
    )


def rewrite_model_card(readme_path: Path, base_model_id: str) -> None:
    text = readme_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    rewritten: list[str] = []
    in_front_matter = False
    front_matter_done = False
    for index, line in enumerate(lines):
        if index == 0 and line.strip() == "---":
            in_front_matter = True
            rewritten.append(line)
            continue
        if in_front_matter and line.strip() == "---":
            front_matter_done = True
            in_front_matter = False
            rewritten.append(line)
            continue
        if in_front_matter and line.startswith("base_model: "):
            rewritten.append(f"base_model: {base_model_id}")
            continue
        if in_front_matter and line.strip().startswith("- base_model:adapter:"):
            rewritten.append(f"- base_model:adapter:{base_model_id}")
            continue
        rewritten.append(line)
    if not front_matter_done:
        raise RuntimeError(f"README front matter is missing or malformed: {readme_path}")
    readme_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def patch_adapter_config(adapter_config_path: Path, canonical_base_model: str) -> dict[str, Any]:
    payload = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    old_value = payload.get("base_model_name_or_path")
    changed = old_value != canonical_base_model
    if changed:
        payload["base_model_name_or_path"] = canonical_base_model
        adapter_config_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return {
        "path": str(adapter_config_path),
        "old_base_model_name_or_path": old_value,
        "new_base_model_name_or_path": canonical_base_model,
        "changed": changed,
    }


def patch_readme_base_model(readme_path: Path, canonical_base_model: str) -> dict[str, Any]:
    before = readme_path.read_text(encoding="utf-8")
    rewrite_model_card(readme_path, canonical_base_model)
    after = readme_path.read_text(encoding="utf-8")
    return {
        "path": str(readme_path),
        "changed": before != after,
    }


def patch_local_adapter_metadata(adapter_dir: Path, base_model_id: str) -> dict[str, Any]:
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Adapter config is missing from source adapter repo: {adapter_config_path}")

    adapter_patch = patch_adapter_config(adapter_config_path, base_model_id)

    readme_path = adapter_dir / "README.md"
    readme_changed = False
    if readme_path.exists():
        readme_patch = patch_readme_base_model(readme_path, base_model_id)
        readme_changed = bool(readme_patch["changed"])

    return {
        "adapter_dir": str(adapter_dir),
        "old_base_model_name_or_path": adapter_patch["old_base_model_name_or_path"],
        "new_base_model_name_or_path": adapter_patch["new_base_model_name_or_path"],
        "changed_adapter_config": adapter_patch["changed"],
        "changed_readme": readme_changed,
    }


def stage_adapter_for_upload(source_dir: Path, base_model_id: str) -> Path:
    temp_root = Path(tempfile.mkdtemp(prefix="adapter_publish_"))
    staging_dir = temp_root / source_dir.name
    shutil.copytree(source_dir, staging_dir)

    patch_adapter_config(staging_dir / "adapter_config.json", base_model_id)

    readme_path = staging_dir / "README.md"
    if readme_path.exists():
        patch_readme_base_model(readme_path, base_model_id)

    return staging_dir
