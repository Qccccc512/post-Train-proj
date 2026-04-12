#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def serialize_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def render_chat_text(tokenizer: Any, messages: list[dict[str, Any]], tools: Any = None) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": False}
    if tools:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(messages, **kwargs)


def find_think_content_spans(content: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        open_index = content.find(THINK_OPEN, cursor)
        if open_index == -1:
            break
        content_start = open_index + len(THINK_OPEN)
        close_index = content.find(THINK_CLOSE, content_start)
        if close_index == -1:
            spans.append((content_start, len(content)))
            break
        spans.append((content_start, close_index))
        cursor = close_index + len(THINK_CLOSE)
    return spans


def _token_count_for_prefix(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _conversation_token_count(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: Any,
    cache: dict[str, int],
) -> int:
    rendered = render_chat_text(tokenizer, messages, tools)
    if rendered not in cache:
        cache[rendered] = _token_count_for_prefix(tokenizer, rendered)
    return cache[rendered]


def char_span_for_token_slice(
    offsets: list[tuple[int, int]],
    start_index: int,
    end_index: int,
) -> tuple[int, int] | None:
    non_empty = [(start, end) for start, end in offsets[start_index:end_index] if end > start]
    if not non_empty:
        return None
    return non_empty[0][0], non_empty[-1][1]


def token_indices_overlapping_span(
    offsets: list[tuple[int, int]],
    span_start: int,
    span_end: int,
) -> list[int]:
    indices: list[int] = []
    for index, (token_start, token_end) in enumerate(offsets):
        if token_end <= span_start:
            continue
        if token_start >= span_end:
            break
        if token_start < span_end and token_end > span_start:
            indices.append(index)
    return indices


def build_tokenized_supervised_example(
    tokenizer: Any,
    example: dict[str, Any],
    max_length: int,
) -> dict[str, Any]:
    messages = example["messages"]
    tools = example.get("tools")
    rendered_text = render_chat_text(tokenizer, messages, tools)
    encoded = tokenizer(
        rendered_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    offsets = encoded["offset_mapping"]
    completion_mask = [0] * len(input_ids)

    token_count_cache: dict[str, int] = {rendered_text: _token_count_for_prefix(tokenizer, rendered_text)}
    previous_full_prefix_len = 0
    for message_index, message in enumerate(messages):
        prefix_messages = messages[: message_index + 1]
        if not any(prefix_message.get("role") == "user" for prefix_message in prefix_messages):
            continue
        current_full_prefix_len = _conversation_token_count(
            tokenizer,
            prefix_messages,
            tools,
            token_count_cache,
        )
        if message.get("role") != "assistant":
            previous_full_prefix_len = current_full_prefix_len
            continue

        assistant_start_token = previous_full_prefix_len
        assistant_end_token = min(current_full_prefix_len, len(input_ids))
        if assistant_start_token >= len(input_ids):
            previous_full_prefix_len = current_full_prefix_len
            continue
        completion_mask[assistant_start_token:assistant_end_token] = [1] * (
            assistant_end_token - assistant_start_token
        )

        assistant_char_span = char_span_for_token_slice(offsets, assistant_start_token, assistant_end_token)
        if assistant_char_span is None:
            previous_full_prefix_len = current_full_prefix_len
            continue
        assistant_char_start, assistant_char_end = assistant_char_span
        assistant_text = rendered_text[assistant_char_start:assistant_char_end]
        if not assistant_text:
            previous_full_prefix_len = current_full_prefix_len
            continue

        for think_start, think_end in find_think_content_spans(assistant_text):
            think_char_start = assistant_char_start + think_start
            think_char_end = assistant_char_start + think_end
            for token_index in token_indices_overlapping_span(offsets, think_char_start, think_char_end):
                completion_mask[token_index] = 0

        previous_full_prefix_len = current_full_prefix_len

    supervised_tokens = sum(completion_mask)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
        "text": rendered_text,
        "supervised_tokens": supervised_tokens,
        "seq_len": len(input_ids),
    }
