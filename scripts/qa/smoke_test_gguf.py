#!/usr/bin/env python3
from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from scripts.common.model_artifacts import (
    FINAL_MODEL_SLUG,
    default_gguf_output_dir,
    default_llama_cpp_dir,
    default_smoke_output_dir,
    resolve_output_dir,
)
from scripts.common.runtime_utils import now_timestamp, resolve_path, write_json


DEFAULT_PROMPTS: list[dict[str, str]] = [
    {
        "id": "zh_lora_one_sentence",
        "prompt": "用一句话解释 LoRA 是什么。",
    },
    {
        "id": "json_bf16",
        "prompt": 'Return valid JSON with keys answer and confidence for: what is BF16?',
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run minimal llama.cpp smoke tests against GGUF files.")
    parser.add_argument("--llama-cpp-dir", default=None)
    parser.add_argument("--f16-gguf", default=None)
    parser.add_argument("--q4-gguf", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ngl", type=int, default=999)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    return parser


def extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[{[]", text):
        try:
            payload, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def find_binary(llama_cpp_dir: Path, names: list[str]) -> Path:
    for name in names:
        for candidate in (
            llama_cpp_dir / "build" / "bin" / name,
            llama_cpp_dir / "build" / "bin" / f"{name}.exe",
        ):
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Unable to locate any of {names} under {llama_cpp_dir / 'build' / 'bin'}")


def default_f16_gguf_path() -> Path:
    return default_gguf_output_dir() / f"{FINAL_MODEL_SLUG}-f16.gguf"


def default_q4_gguf_path() -> Path:
    return default_gguf_output_dir() / f"{FINAL_MODEL_SLUG}-q4_k_m.gguf"


def run_prompt(
    *,
    llama_cli_path: Path,
    model_path: Path,
    prompt: str,
    ngl: int,
    max_new_tokens: int,
    grammar_file: Path | None,
) -> dict[str, Any]:
    command = [
        str(llama_cli_path),
        "-m",
        str(model_path),
        "-no-cnv",
        "-ngl",
        str(ngl),
        "-n",
        str(max_new_tokens),
        "--temp",
        "0",
        "--top-p",
        "1.0",
        "--repeat-penalty",
        "1.05",
        "-p",
        prompt,
    ]
    if grammar_file is not None:
        command.extend(["--grammar-file", str(grammar_file)])

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    combined_text = (completed.stdout or "").strip()
    if not combined_text:
        combined_text = (completed.stderr or "").strip()
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "combined_text": combined_text,
    }


def run_model_smoke(
    *,
    label: str,
    model_path: Path,
    llama_cli_path: Path,
    ngl_candidates: list[int],
    max_new_tokens: int,
    grammar_file: Path | None,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for ngl in ngl_candidates:
        per_prompt: list[dict[str, Any]] = []
        all_passed = True
        for prompt_spec in DEFAULT_PROMPTS:
            prompt_grammar = grammar_file if prompt_spec["id"] == "json_bf16" else None
            result = run_prompt(
                llama_cli_path=llama_cli_path,
                model_path=model_path,
                prompt=prompt_spec["prompt"],
                ngl=ngl,
                max_new_tokens=max_new_tokens,
                grammar_file=prompt_grammar,
            )
            json_payload = extract_json_object(result["combined_text"]) if prompt_spec["id"] == "json_bf16" else None
            prompt_passed = result["returncode"] == 0 and bool(result["combined_text"].strip())
            if prompt_spec["id"] == "json_bf16":
                prompt_passed = prompt_passed and json_payload is not None
            enriched = {
                "id": prompt_spec["id"],
                "prompt": prompt_spec["prompt"],
                "ngl": ngl,
                "returncode": result["returncode"],
                "output": result["combined_text"],
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "command": result["command"],
                "json_payload": json_payload,
                "passed": prompt_passed,
            }
            per_prompt.append(enriched)
            all_passed = all_passed and prompt_passed
            if not prompt_passed:
                break

        attempt = {
            "model_label": label,
            "model_path": str(model_path),
            "ngl": ngl,
            "prompts": per_prompt,
            "passed": all_passed,
        }
        attempts.append(attempt)
        if all_passed:
            return {
                "model_label": label,
                "model_path": str(model_path),
                "selected_ngl": ngl,
                "attempts": attempts,
                "passed": True,
                "prompts": per_prompt,
            }

    return {
        "model_label": label,
        "model_path": str(model_path),
        "selected_ngl": None,
        "attempts": attempts,
        "passed": False,
        "prompts": attempts[-1]["prompts"] if attempts else [],
    }


def run_gguf_smoke_tests(
    *,
    llama_cpp_dir: str | Path | None = None,
    f16_gguf: str | Path | None = None,
    q4_gguf: str | Path | None = None,
    output_dir: str | Path | None = None,
    ngl: int = 999,
    max_new_tokens: int = 192,
) -> dict[str, Any]:
    resolved_output_dir = resolve_output_dir(output_dir, default_dir=default_smoke_output_dir())
    resolved_llama_cpp_dir = resolve_path(llama_cpp_dir) if llama_cpp_dir is not None else default_llama_cpp_dir()
    resolved_f16 = resolve_path(f16_gguf) if f16_gguf is not None else default_f16_gguf_path()
    resolved_q4 = resolve_path(q4_gguf) if q4_gguf is not None else default_q4_gguf_path()

    llama_cli_path = find_binary(resolved_llama_cpp_dir, ["llama-cli"])
    grammar_file = resolved_llama_cpp_dir / "grammars" / "json.gbnf"
    if not grammar_file.exists():
        grammar_file = None

    ngl_candidates = [candidate for candidate in [ngl, 128, 64, 32, 16, 0] if candidate <= ngl or candidate == 0]
    seen: set[int] = set()
    ngl_candidates = [candidate for candidate in ngl_candidates if not (candidate in seen or seen.add(candidate))]

    f16_result = run_model_smoke(
        label="f16",
        model_path=resolved_f16,
        llama_cli_path=llama_cli_path,
        ngl_candidates=ngl_candidates,
        max_new_tokens=max_new_tokens,
        grammar_file=grammar_file,
    )
    q4_result = run_model_smoke(
        label="q4_k_m",
        model_path=resolved_q4,
        llama_cli_path=llama_cli_path,
        ngl_candidates=ngl_candidates,
        max_new_tokens=max_new_tokens,
        grammar_file=grammar_file,
    )

    write_json(resolved_output_dir / "gguf_f16_outputs.json", f16_result)
    write_json(resolved_output_dir / "gguf_q4_k_m_outputs.json", q4_result)

    summary = {
        "timestamp": now_timestamp(),
        "llama_cpp_dir": str(resolved_llama_cpp_dir),
        "llama_cli_path": str(llama_cli_path),
        "f16_gguf": str(resolved_f16),
        "q4_gguf": str(resolved_q4),
        "passed": bool(f16_result["passed"] and q4_result["passed"]),
        "models": {
            "f16": {
                "selected_ngl": f16_result["selected_ngl"],
                "passed": f16_result["passed"],
            },
            "q4_k_m": {
                "selected_ngl": q4_result["selected_ngl"],
                "passed": q4_result["passed"],
            },
        },
    }

    lines = [
        f"# GGUF Smoke Summary for {FINAL_MODEL_SLUG}",
        "",
        f"- timestamp: {summary['timestamp']}",
        f"- passed: `{summary['passed']}`",
        f"- llama_cpp_dir: `{resolved_llama_cpp_dir}`",
        "",
        f"## f16",
        f"- selected_ngl: `{f16_result['selected_ngl']}`",
        f"- passed: `{f16_result['passed']}`",
        "",
        f"## q4_k_m",
        f"- selected_ngl: `{q4_result['selected_ngl']}`",
        f"- passed: `{q4_result['passed']}`",
        "",
    ]
    (resolved_output_dir / "gguf_smoke_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run_gguf_smoke_tests(
        llama_cpp_dir=args.llama_cpp_dir,
        f16_gguf=args.f16_gguf,
        q4_gguf=args.q4_gguf,
        output_dir=args.output_dir,
        ngl=args.ngl,
        max_new_tokens=args.max_new_tokens,
    )
    print(
        "[smoke_test_gguf] "
        f"passed={summary['passed']} f16_ngl={summary['models']['f16']['selected_ngl']} "
        f"q4_ngl={summary['models']['q4_k_m']['selected_ngl']}",
        flush=True,
    )
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
