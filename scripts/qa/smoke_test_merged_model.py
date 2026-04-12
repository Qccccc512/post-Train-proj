#!/usr/bin/env python3
from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import json
import re
from pathlib import Path
from typing import Any

from scripts.common.model_artifacts import (
    FINAL_MODEL_SLUG,
    default_smoke_output_dir,
    resolve_model_dir,
    resolve_output_dir,
)
from scripts.common.runtime_utils import now_timestamp, resolve_path, write_json


DEFAULT_QUESTIONS: list[dict[str, str]] = [
    {
        "id": "zh_lora_explanation",
        "prompt": "请用两句话解释 LoRA 的作用，并补充一个训练时需要注意的风险。",
    },
    {
        "id": "en_bf16_bullets",
        "prompt": "Answer in exactly 3 bullet points: what is BF16 and why is it used?",
    },
    {
        "id": "json_overfit_reason",
        "prompt": 'Return valid JSON with keys "answer" and "confidence" explaining why long training can overfit.',
    },
    {
        "id": "math_17x23",
        "prompt": "Compute 17 * 23 and briefly explain the calculation in one sentence.",
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a deterministic smoke test against a merged HF model directory.")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--questions-file")
    return parser


def load_questions(path: str | Path | None) -> list[dict[str, str]]:
    if path is None:
        return DEFAULT_QUESTIONS

    payload = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    questions: list[dict[str, str]] = []
    for index, item in enumerate(payload):
        if isinstance(item, str):
            questions.append({"id": f"q{index + 1}", "prompt": item})
            continue
        if isinstance(item, dict) and "prompt" in item:
            questions.append({"id": str(item.get("id") or f"q{index + 1}"), "prompt": str(item["prompt"])})
            continue
        raise ValueError("questions-file must contain a JSON list of strings or objects with a prompt field")
    return questions


def extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    candidates = [match.start() for match in re.finditer(r"[{[]", text)]
    for start in candidates:
        try:
            payload, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def looks_repetitive(text: str) -> bool:
    tokens = re.findall(r"\S+", text.lower())
    if len(tokens) < 24:
        return False
    unique_ratio = len(set(tokens)) / max(1, len(tokens))
    return unique_ratio < 0.35


def resolve_torch_dtype(name: str):
    import torch

    if name == "auto":
        return "auto"
    return getattr(torch, name)


def resolve_device_map(device: str) -> Any:
    if device == "auto":
        return "auto"
    return {"": device}


def resolve_input_device(model: Any):
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    hf_device_map = getattr(model, "hf_device_map", {})
    for value in hf_device_map.values():
        if isinstance(value, str) and value not in {"disk", "meta"}:
            return value
    raise RuntimeError("Unable to resolve a concrete device for model inputs")


def format_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def run_smoke_test(
    *,
    model_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    device: str = "auto",
    dtype: str = "auto",
    max_new_tokens: int = 256,
    questions_file: str | Path | None = None,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_model_dir = resolve_model_dir(model_dir)
    resolved_output_dir = resolve_output_dir(output_dir, default_dir=default_smoke_output_dir())
    if not resolved_model_dir.exists():
        raise FileNotFoundError(f"Merged model directory does not exist: {resolved_model_dir}")
    questions = load_questions(questions_file)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.05,
    }

    print(
        "[smoke_test_merged_model] "
        f"Loading tokenizer from {resolved_model_dir}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(resolved_model_dir), trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(
        "[smoke_test_merged_model] "
        f"Loading model from {resolved_model_dir} with device={device}, dtype={dtype}",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(resolved_model_dir),
        trust_remote_code=False,
        torch_dtype=resolve_torch_dtype(dtype),
        device_map=resolve_device_map(device),
        low_cpu_mem_usage=True,
    )
    model.eval()

    param_device = resolve_input_device(model)
    outputs: list[dict[str, Any]] = []
    overall_passed = True

    for question in questions:
        prompt = question["prompt"]
        formatted_prompt = format_prompt(tokenizer, prompt)
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {key: value.to(param_device) for key, value in inputs.items()}

        print(
            "[smoke_test_merged_model] "
            f"Generating answer for {question['id']}",
            flush=True,
        )
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                pad_token_id=tokenizer.pad_token_id,
                **generation_kwargs,
            )

        generated_tokens = generated[0][inputs["input_ids"].shape[1] :]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        json_payload = extract_json_object(answer)
        question_result = {
            "id": question["id"],
            "prompt": prompt,
            "answer": answer,
            "non_empty": bool(answer),
            "looks_repetitive": looks_repetitive(answer),
            "json_parsed": json_payload is not None if question["id"] == "json_overfit_reason" else None,
            "json_payload": json_payload if question["id"] == "json_overfit_reason" else None,
            "math_contains_391": "391" in answer if question["id"] == "math_17x23" else None,
        }
        question_result["passed"] = (
            question_result["non_empty"]
            and not question_result["looks_repetitive"]
            and (question_result["json_parsed"] is not False)
            and (question_result["math_contains_391"] is not False)
        )
        outputs.append(question_result)
        overall_passed = overall_passed and bool(question_result["passed"])

    smoke_prompts_path = resolved_output_dir / "smoke_prompts.json"
    smoke_outputs_path = resolved_output_dir / "smoke_outputs.json"
    smoke_manifest_path = resolved_output_dir / "smoke_manifest.json"
    smoke_markdown_path = resolved_output_dir / "smoke_outputs.md"

    write_json(smoke_prompts_path, questions)
    write_json(smoke_outputs_path, outputs)

    manifest = {
        "timestamp": now_timestamp(),
        "model_dir": str(resolved_model_dir),
        "output_dir": str(resolved_output_dir),
        "device": device,
        "dtype": dtype,
        "max_new_tokens": max_new_tokens,
        "question_count": len(questions),
        "passed": overall_passed,
        "checks": {
            "all_non_empty": all(item["non_empty"] for item in outputs),
            "json_prompt_valid": next((item["json_parsed"] for item in outputs if item["id"] == "json_overfit_reason"), False),
            "math_prompt_has_391": next((item["math_contains_391"] for item in outputs if item["id"] == "math_17x23"), False),
            "no_obvious_repetition": not any(item["looks_repetitive"] for item in outputs),
        },
    }
    write_json(smoke_manifest_path, manifest)

    lines = [
        f"# Smoke Test Summary for {FINAL_MODEL_SLUG}",
        "",
        f"- timestamp: {manifest['timestamp']}",
        f"- model_dir: `{resolved_model_dir}`",
        f"- passed: `{manifest['passed']}`",
        f"- device: `{device}`",
        f"- dtype: `{dtype}`",
        "",
    ]
    for item in outputs:
        lines.extend(
            [
                f"## {item['id']}",
                "",
                f"Prompt: `{item['prompt']}`",
                "",
                f"Passed: `{item['passed']}`",
                "",
                item["answer"] or "<empty>",
                "",
            ]
        )
    smoke_markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    manifest = run_smoke_test(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        questions_file=args.questions_file,
    )
    print(
        "[smoke_test_merged_model] "
        f"passed={manifest['passed']} output_dir={manifest['output_dir']}",
        flush=True,
    )
    if not manifest["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
