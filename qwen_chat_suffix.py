"""Small tokenizer-only helpers for incremental Qwen chat turns.

Kept dependency-free so the exact chat-template return-shape handling is
unit-testable without loading torch or transformers.
"""

from __future__ import annotations

from collections.abc import Mapping


def flat_token_ids(value) -> list[int]:
    """Normalize tokenizer/chat-template output to one flat list of ids.

    Transformers versions can return a plain list, a one-row nested list,
    a tensor-like object, or a BatchEncoding/mapping containing input_ids.
    """

    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError("chat template mapping has no input_ids")
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, tuple):
        value = list(value)

    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        value = list(value[0])

    if not isinstance(value, list):
        raise TypeError(
            "chat template output must be a token sequence or contain input_ids"
        )

    try:
        return [int(x) for x in value]
    except (TypeError, ValueError) as exc:
        raise TypeError("input_ids must be a flat integer token sequence") from exc


def assistant_end_id(tokenizer) -> int:
    tok = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if tok is None or tok == tokenizer.unk_token_id:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Qwen assistant-end token is unavailable")
        return int(tokenizer.eos_token_id)
    return int(tok)


def user_turn_suffix_ids(tokenizer, user_text: str) -> list[int]:
    """Return only the new user turn + assistant generation-prompt tokens."""

    anchor = "__AOC_PREVIOUS_ASSISTANT__"
    anchor_messages = [
        {"role": "system", "content": "__AOC_TEMPLATE_SYSTEM__"},
        {"role": "user", "content": "__AOC_TEMPLATE_USER__"},
        {"role": "assistant", "content": anchor},
    ]
    full_messages = anchor_messages + [
        {"role": "user", "content": user_text},
    ]

    try:
        anchor_raw = tokenizer.apply_chat_template(
            anchor_messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        full_raw = tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        anchor_raw = tokenizer.apply_chat_template(
            anchor_messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        full_raw = tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=True,
        )

    anchor_ids = flat_token_ids(anchor_raw)
    full_ids = flat_token_ids(full_raw)

    end_id = assistant_end_id(tokenizer)
    positions = [i for i, tok in enumerate(anchor_ids) if tok == end_id]
    if not positions:
        raise RuntimeError(
            "chat template anchor contained no assistant end token"
        )
    cut = positions[-1] + 1
    if full_ids[:cut] != anchor_ids[:cut]:
        raise RuntimeError(
            "Qwen chat template is not prefix-stable at assistant boundary"
        )
    return full_ids[cut:]
