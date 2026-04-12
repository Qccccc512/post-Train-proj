#!/usr/bin/env python3
from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.common.model_artifacts import (
    DEFAULT_BASE_MODEL_ID,
    FINAL_MODEL_SLUG,
    default_gguf_output_dir,
    default_llama_cpp_dir,
    default_smoke_output_dir,
    resolve_model_dir,
    resolve_output_dir,
    sync_base_model_compat_assets,
)
from scripts.common.runtime_utils import now_timestamp, repo_root, resolve_path, write_json
from scripts.qa.smoke_test_gguf import run_gguf_smoke_tests
from scripts.qa.smoke_test_merged_model import run_smoke_test


LLAMA_CPP_REPO_URL = "https://github.com/ggml-org/llama.cpp.git"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a merged HF model directory and export GGUF variants via official llama.cpp.")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--llama-cpp-dir", default=None)
    parser.add_argument("--base-model-id", default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--quantizations", default="f16,q4_k_m")
    parser.add_argument("--skip-smoke-test", action="store_true")
    parser.add_argument("--skip-gguf-smoke-test", action="store_true")
    parser.add_argument("--force-rebuild-llama-cpp", action="store_true")
    return parser


def parse_quantizations(raw: str) -> list[str]:
    parsed = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not parsed:
        raise ValueError("quantizations must not be empty")
    allowed = {"f16", "q4_k_m"}
    unknown = sorted(set(parsed) - allowed)
    if unknown:
        raise ValueError(f"Unsupported quantizations: {unknown}")
    if "q4_k_m" in parsed and "f16" not in parsed:
        parsed.insert(0, "f16")
    return parsed


def run_command(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print(
        "[export_gguf_from_merged_model] "
        f"Running: {' '.join(command)}",
        flush=True,
    )
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def ensure_llama_cpp_checkout(llama_cpp_dir: Path) -> dict[str, Any]:
    if llama_cpp_dir.exists():
        print(
            "[export_gguf_from_merged_model] "
            f"Reusing existing llama.cpp checkout at {llama_cpp_dir}",
            flush=True,
        )
    else:
        print(
            "[export_gguf_from_merged_model] "
            f"Cloning llama.cpp into {llama_cpp_dir}",
            flush=True,
        )
        run_command(
            ["git", "clone", "--depth", "1", LLAMA_CPP_REPO_URL, str(llama_cpp_dir)],
            cwd=repo_root(),
        )

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(llama_cpp_dir),
        text=True,
    ).strip()
    return {
        "repo_url": LLAMA_CPP_REPO_URL,
        "checkout_dir": str(llama_cpp_dir),
        "commit": commit,
    }


def find_binary(llama_cpp_dir: Path, names: list[str]) -> Path:
    for name in names:
        for candidate in (
            llama_cpp_dir / "build" / "bin" / name,
            llama_cpp_dir / "build" / "bin" / f"{name}.exe",
        ):
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Unable to locate any of {names} under {llama_cpp_dir / 'build' / 'bin'}")


def build_llama_cpp(
    *,
    llama_cpp_dir: Path,
    force_rebuild: bool,
) -> dict[str, Any]:
    build_dir = llama_cpp_dir / "build"
    if force_rebuild and build_dir.exists():
        shutil.rmtree(build_dir)

    fallback_reason: str | None = None
    backend = "existing"
    configure_commands: list[list[str]] = []

    cli_exists = (build_dir / "bin" / "llama-cli").exists() or (build_dir / "bin" / "llama-cli.exe").exists()
    quantize_exists = (
        (build_dir / "bin" / "llama-quantize").exists()
        or (build_dir / "bin" / "llama-quantize.exe").exists()
        or (build_dir / "bin" / "quantize").exists()
        or (build_dir / "bin" / "quantize.exe").exists()
    )
    if not (cli_exists and quantize_exists):
        try:
            configure_commands.append(["cmake", "-B", "build", "-DGGML_CUDA=ON", "-DLLAMA_BUILD_TESTS=OFF"])
            run_command(configure_commands[-1], cwd=llama_cpp_dir)
            run_command(["cmake", "--build", "build", "-j"], cwd=llama_cpp_dir)
            backend = "cuda"
        except subprocess.CalledProcessError as exc:
            fallback_reason = str(exc)
            if build_dir.exists():
                shutil.rmtree(build_dir)
            configure_commands.append(["cmake", "-B", "build", "-DGGML_CUDA=OFF", "-DLLAMA_BUILD_TESTS=OFF"])
            run_command(configure_commands[-1], cwd=llama_cpp_dir)
            run_command(["cmake", "--build", "build", "-j"], cwd=llama_cpp_dir)
            backend = "cpu"

    llama_cli = find_binary(llama_cpp_dir, ["llama-cli"])
    llama_quantize = find_binary(llama_cpp_dir, ["llama-quantize", "quantize"])
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found in {llama_cpp_dir}")

    return {
        "build_dir": str(build_dir),
        "backend": backend,
        "fallback_reason": fallback_reason,
        "configure_commands": configure_commands,
        "llama_cli_path": str(llama_cli),
        "llama_quantize_path": str(llama_quantize),
        "convert_script_path": str(convert_script),
    }


def convert_to_f16_gguf(
    *,
    model_dir: Path,
    output_path: Path,
    convert_script: Path,
) -> dict[str, Any]:
    if output_path.exists():
        print(
            "[export_gguf_from_merged_model] "
            f"Reusing existing FP16 GGUF at {output_path}",
            flush=True,
        )
        return {
            "path": str(output_path),
            "reused": True,
            "command": None,
            "size_bytes": output_path.stat().st_size,
        }

    command = [
        sys.executable,
        str(convert_script),
        str(model_dir),
        "--outfile",
        str(output_path),
        "--outtype",
        "f16",
    ]
    run_command(command, cwd=convert_script.parent)
    return {
        "path": str(output_path),
        "reused": False,
        "command": command,
        "size_bytes": output_path.stat().st_size,
    }


def quantize_gguf(
    *,
    source_path: Path,
    output_path: Path,
    quantize_binary: Path,
    quantization: str,
) -> dict[str, Any]:
    if output_path.exists():
        print(
            "[export_gguf_from_merged_model] "
            f"Reusing existing {quantization} GGUF at {output_path}",
            flush=True,
        )
        return {
            "path": str(output_path),
            "reused": True,
            "command": None,
            "size_bytes": output_path.stat().st_size,
            "quantization": quantization,
        }

    command = [
        str(quantize_binary),
        str(source_path),
        str(output_path),
        quantization.upper(),
    ]
    run_command(command, cwd=quantize_binary.parent)
    return {
        "path": str(output_path),
        "reused": False,
        "command": command,
        "size_bytes": output_path.stat().st_size,
        "quantization": quantization,
    }


def run_export(
    *,
    model_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    llama_cpp_dir: str | Path | None = None,
    base_model_id: str = DEFAULT_BASE_MODEL_ID,
    quantizations: str = "f16,q4_k_m",
    skip_smoke_test: bool = False,
    skip_gguf_smoke_test: bool = False,
    force_rebuild_llama_cpp: bool = False,
) -> dict[str, Any]:
    resolved_model_dir = resolve_model_dir(model_dir)
    resolved_output_dir = resolve_output_dir(output_dir, default_dir=default_gguf_output_dir())
    smoke_output_dir = resolve_output_dir(None, default_dir=default_smoke_output_dir())
    resolved_llama_cpp_dir = resolve_path(llama_cpp_dir) if llama_cpp_dir is not None else default_llama_cpp_dir()
    requested_quantizations = parse_quantizations(quantizations)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    compat_manifest = sync_base_model_compat_assets(
        model_dir=resolved_model_dir,
        base_model_id=base_model_id,
        output_manifest_path=smoke_output_dir / "compat_assets_manifest.json",
    )

    smoke_manifest = None
    if not skip_smoke_test:
        smoke_manifest = run_smoke_test(
            model_dir=resolved_model_dir,
            output_dir=smoke_output_dir,
        )

    llama_cpp_checkout = ensure_llama_cpp_checkout(resolved_llama_cpp_dir)
    llama_cpp_build = build_llama_cpp(
        llama_cpp_dir=resolved_llama_cpp_dir,
        force_rebuild=force_rebuild_llama_cpp,
    )

    f16_path = resolved_output_dir / f"{FINAL_MODEL_SLUG}-f16.gguf"
    q4_path = resolved_output_dir / f"{FINAL_MODEL_SLUG}-q4_k_m.gguf"

    conversions: dict[str, Any] = {}
    if "f16" in requested_quantizations:
        conversions["f16"] = convert_to_f16_gguf(
            model_dir=resolved_model_dir,
            output_path=f16_path,
            convert_script=Path(llama_cpp_build["convert_script_path"]),
        )

    if "q4_k_m" in requested_quantizations:
        conversions["q4_k_m"] = quantize_gguf(
            source_path=f16_path,
            output_path=q4_path,
            quantize_binary=Path(llama_cpp_build["llama_quantize_path"]),
            quantization="q4_k_m",
        )

    gguf_smoke_summary = None
    if not skip_gguf_smoke_test:
        gguf_smoke_summary = run_gguf_smoke_tests(
            llama_cpp_dir=resolved_llama_cpp_dir,
            f16_gguf=f16_path,
            q4_gguf=q4_path,
            output_dir=smoke_output_dir,
        )

    manifest = {
        "timestamp": now_timestamp(),
        "model_dir": str(resolved_model_dir),
        "output_dir": str(resolved_output_dir),
        "base_model_id": base_model_id,
        "requested_quantizations": requested_quantizations,
        "compatibility_assets": compat_manifest,
        "smoke_test": smoke_manifest,
        "llama_cpp": {
            **llama_cpp_checkout,
            **llama_cpp_build,
        },
        "conversions": conversions,
        "gguf_smoke": gguf_smoke_summary,
    }
    write_json(resolved_output_dir / "conversion_manifest.json", manifest)
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    manifest = run_export(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        llama_cpp_dir=args.llama_cpp_dir,
        base_model_id=args.base_model_id,
        quantizations=args.quantizations,
        skip_smoke_test=args.skip_smoke_test,
        skip_gguf_smoke_test=args.skip_gguf_smoke_test,
        force_rebuild_llama_cpp=args.force_rebuild_llama_cpp,
    )
    print(
        "[export_gguf_from_merged_model] "
        f"completed output_dir={manifest['output_dir']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
