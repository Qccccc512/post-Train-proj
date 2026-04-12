#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time


def _patch_bfcl_local_generation(max_output_tokens: int, repetition_penalty: float = 1.0) -> None:
    from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler

    def patched_query_prompting(self, inference_data: dict):
        function: list[dict] = inference_data["function"]
        message: list[dict] = inference_data["message"]

        formatted_prompt: str = self._format_prompt(message, function)
        inference_data["inference_input_log"] = {"formatted_prompt": formatted_prompt}

        input_token_count = len(self.tokenizer.tokenize(formatted_prompt))
        # 增加 200 token 的安全缓冲区，避免边界情况导致的上下文长度超限
        # 这个缓冲区考虑了：tokenizer 估算误差、特殊 token、系统 overhead
        safety_buffer = 200
        if self.max_context_length < input_token_count + safety_buffer:
            # 输入已经接近或超过上下文限制，只分配较少的输出 token
            leftover_tokens_count = min(max_output_tokens, 512)
        else:
            leftover_tokens_count = min(
                max_output_tokens,
                self.max_context_length - input_token_count - safety_buffer,
            )
        leftover_tokens_count = max(1, leftover_tokens_count)

        extra_body = {}
        if hasattr(self, "stop_token_ids"):
            extra_body["stop_token_ids"] = self.stop_token_ids
        if hasattr(self, "skip_special_tokens"):
            extra_body["skip_special_tokens"] = self.skip_special_tokens
        # Add repetition_penalty for Qwen3 thinking model
        if repetition_penalty > 1.0:
            extra_body["repetition_penalty"] = repetition_penalty

        start_time = time.time()
        if extra_body:
            api_response = self.client.completions.create(
                model=self.model_path_or_id,
                temperature=self.temperature,
                prompt=formatted_prompt,
                max_tokens=leftover_tokens_count,
                extra_body=extra_body,
                timeout=72000,
            )
        else:
            api_response = self.client.completions.create(
                model=self.model_path_or_id,
                temperature=self.temperature,
                prompt=formatted_prompt,
                max_tokens=leftover_tokens_count,
                timeout=72000,
            )
        end_time = time.time()
        return api_response, end_time - start_time

    OSSHandler._query_prompting = patched_query_prompting


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command", nargs="?")
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    return parser.parse_known_args(argv)


def main() -> None:
    known, passthrough = _parse_args(sys.argv[1:])
    if known.command == "generate" and known.max_output_tokens is not None:
        _patch_bfcl_local_generation(known.max_output_tokens, known.repetition_penalty)

    argv = [known.command] if known.command else []
    argv.extend(passthrough)
    sys.argv = [sys.argv[0], *argv]

    from bfcl_eval.__main__ import cli

    cli()


if __name__ == "__main__":
    main()
