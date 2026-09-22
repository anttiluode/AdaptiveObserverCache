"""Masked-spacer positional-distance control for the live Qwen AOC.

The earlier visible-filler sweep showed that observer control survives at +280
tokens, but the neutral A-vs-B basin itself flipped. That did not isolate cache
age from intervening conversational content.

This control advances DynamicCache length with masked spacer tokens. Qwen3
derives new RoPE positions from past_key_values.get_seq_length(), while the
2-D attention mask keeps those spacer rows unavailable as keys to later
queries. The source therefore becomes positionally older without adding
readable filler content.

Default:
    python3.13 qwen_observer_masked_distance.py

Receipt:
    results/qwen_observer_masked_distance.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

from qwen_observer import QwenObserverController
from qwen_observer_chat import (
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_QUESTION,
    input_device,
    load_model,
)
from qwen_observer_distance_sweep import (
    DEFAULT_CANDIDATE_A,
    DEFAULT_CANDIDATE_B,
    cache_fork_mode,
    fork_cache,
    parse_csv_floats,
    parse_csv_ints,
    tokenize_answer,
    write_receipt,
)
from qwen_observer_live_cache import (
    append_tokens,
    cache_length,
    greedy_continue,
    initial_prompt,
    load_plan,
    new_cache,
)
from qwen_chat_suffix import user_turn_suffix_ids


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Measure AOC trust control against near-pure positional distance "
            "using unreadable masked spacer K/V rows."
        )
    )
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--source-a", default=DEFAULT_A)
    p.add_argument("--source-b", default=DEFAULT_B)
    p.add_argument("--question", default=DEFAULT_QUESTION)
    p.add_argument("--candidate-a", default=DEFAULT_CANDIDATE_A)
    p.add_argument("--candidate-b", default=DEFAULT_CANDIDATE_B)
    p.add_argument(
        "--plan-receipt",
        default="results/qwen_observer_causal_compare.json",
    )
    p.add_argument("--target-margin", type=float, default=3.0)
    p.add_argument("--max-ratio", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--gpu-memory", default="6GiB")
    p.add_argument("--cpu-memory", default="6GiB")
    p.add_argument("--offload-dir", default=".offload_qwen_observer")
    p.add_argument("--distances", default="0,256")
    p.add_argument("--trust-grid", default="-1,0,1")
    p.add_argument(
        "--spacer-text",
        default="x",
        help="regular token used for masked spacer rows; never readable later",
    )
    p.add_argument(
        "--spacer-chunk",
        type=int,
        default=64,
        help="masked spacer tokens appended per forward pass",
    )
    p.add_argument(
        "--receipt",
        default="results/qwen_observer_masked_distance.json",
    )
    return p.parse_args()


@torch.inference_mode()
def append_tokens_with_mask(
    model,
    cache,
    history_ids: List[int],
    history_mask: List[int],
    token_ids: List[int],
    token_mask: List[int],
):
    """Append tokens while preserving a persistent key-visibility mask."""

    if not token_ids:
        raise ValueError("cannot append an empty token segment")
    if len(token_ids) != len(token_mask):
        raise ValueError("token ids and token mask lengths differ")
    if len(history_ids) != len(history_mask):
        raise RuntimeError("history ids and history mask lengths differ")

    device = input_device(model)
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    full_mask = torch.tensor(
        [history_mask + token_mask],
        dtype=torch.long,
        device=device,
    )
    out = model(
        input_ids=ids,
        attention_mask=full_mask,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    cache = out.past_key_values
    history_ids.extend(int(x) for x in token_ids)
    history_mask.extend(int(x) for x in token_mask)

    observed = cache_length(cache)
    if observed != len(history_ids):
        raise RuntimeError(
            f"KV cache/history mismatch: cache={observed}, history={len(history_ids)}"
        )
    if len(history_ids) != len(history_mask):
        raise RuntimeError("history mask lost alignment after append")
    return cache, out.logits[:, -1, :]


def choose_spacer_id(tokenizer, text: str) -> int:
    ids = [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
    if not ids:
        raise RuntimeError(f"spacer text tokenized empty: {text!r}")
    return ids[0]


@torch.inference_mode()
def grow_masked_to_distance(
    model,
    controller,
    cache,
    history_ids: List[int],
    history_mask: List[int],
    *,
    anchor_len: int,
    target_distance: int,
    spacer_id: int,
    spacer_chunk: int,
):
    """Advance exact cache position while new spacer rows remain unreadable."""

    if spacer_chunk <= 0:
        raise ValueError("spacer chunk must be positive")

    while cache_length(cache) - anchor_len < target_distance:
        remaining = target_distance - (cache_length(cache) - anchor_len)
        count = min(spacer_chunk, remaining)
        controller.set_trust(0.0)
        controller.begin_generation(preserve_source_baseline=True)
        cache, _logits = append_tokens_with_mask(
            model,
            cache,
            history_ids,
            history_mask,
            [spacer_id] * count,
            [0] * count,
        )
        summary = controller.summary()
        if not summary.get("cache_integrity_ok", False):
            raise RuntimeError(
                "source K rows changed while growing masked positional spacer"
            )

    actual = cache_length(cache) - anchor_len
    if actual != target_distance:
        raise RuntimeError(
            f"masked distance overshot target: target={target_distance}, actual={actual}"
        )
    return cache


@torch.inference_mode()
def score_candidate(
    model,
    tokenizer,
    controller,
    canonical_cache,
    canonical_history: List[int],
    canonical_mask: List[int],
    question_suffix: list[int],
    candidate: str,
    trust: float,
):
    """Teacher-force a candidate while masked spacer rows stay masked."""

    canonical_len = cache_length(canonical_cache)
    canonical_history_len = len(canonical_history)
    canonical_mask_snapshot = tuple(canonical_mask)

    branch = fork_cache(canonical_cache)
    branch_history = list(canonical_history)
    branch_mask = list(canonical_mask)

    controller.set_trust(trust)
    controller.begin_generation(preserve_source_baseline=True)
    branch, logits = append_tokens_with_mask(
        model,
        branch,
        branch_history,
        branch_mask,
        question_suffix,
        [1] * len(question_suffix),
    )

    candidate_ids = tokenize_answer(tokenizer, candidate)
    logprobs: list[float] = []
    for index, token_id in enumerate(candidate_ids):
        lp = torch.log_softmax(logits.float(), dim=-1)[0, token_id]
        logprobs.append(float(lp.item()))
        if index + 1 < len(candidate_ids):
            branch, logits = append_tokens_with_mask(
                model,
                branch,
                branch_history,
                branch_mask,
                [token_id],
                [1],
            )

    summary = controller.summary()

    if cache_length(canonical_cache) != canonical_len:
        raise RuntimeError("probe mutated canonical KV-cache length")
    if len(canonical_history) != canonical_history_len:
        raise RuntimeError("probe mutated canonical token history")
    if tuple(canonical_mask) != canonical_mask_snapshot:
        raise RuntimeError("probe mutated canonical visibility mask")
    if not summary.get("cache_integrity_ok", False):
        raise RuntimeError(
            "historical source K rows failed integrity check during masked score"
        )

    return {
        "text": candidate,
        "tokens": len(candidate_ids),
        "sum_logprob": sum(logprobs),
        "mean_logprob": sum(logprobs) / len(logprobs),
        "observer": summary,
    }


def checkpoint_row(
    model,
    tokenizer,
    controller,
    cache,
    history_ids,
    history_mask,
    *,
    anchor_len: int,
    target_distance: int,
    trust_grid: list[float],
    question_suffix: list[int],
    candidate_a: str,
    candidate_b: str,
):
    actual_distance = cache_length(cache) - anchor_len
    row = {
        "target_distance": target_distance,
        "actual_added_tokens": actual_distance,
        "canonical_cache_tokens": cache_length(cache),
        "masked_history_tokens": sum(1 for x in history_mask if x == 0),
        "dose_response": [],
    }

    for trust in trust_grid:
        a = score_candidate(
            model,
            tokenizer,
            controller,
            cache,
            history_ids,
            history_mask,
            question_suffix,
            candidate_a,
            trust,
        )
        b = score_candidate(
            model,
            tokenizer,
            controller,
            cache,
            history_ids,
            history_mask,
            question_suffix,
            candidate_b,
            trust,
        )
        margin = a["mean_logprob"] - b["mean_logprob"]
        row["dose_response"].append(
            {
                "trust": float(trust),
                "A": a,
                "B": b,
                "mean_logprob_margin_A_minus_B": margin,
            }
        )
        print(
            f"  distance={actual_distance:5d}  m={trust:+.2f}  "
            f"mean dlogp(A-B)={margin:+.5f}",
            flush=True,
        )
    return row


def summarize(payload):
    rows = payload["checkpoints"]
    if not rows:
        return {}

    summary_rows = []
    for row in rows:
        points = {
            x["trust"]: x["mean_logprob_margin_A_minus_B"]
            for x in row["dose_response"]
        }
        item = {
            "target_distance": row["target_distance"],
            "actual_added_tokens": row["actual_added_tokens"],
            "masked_history_tokens": row["masked_history_tokens"],
            "margin_at_minus1": points.get(-1.0),
            "margin_at_zero": points.get(0.0),
            "margin_at_plus1": points.get(1.0),
        }
        if -1.0 in points and 0.0 in points:
            item["B_control_delta_from_neutral"] = points[-1.0] - points[0.0]
        if 1.0 in points and 0.0 in points:
            item["A_control_delta_from_neutral"] = points[1.0] - points[0.0]
        if -1.0 in points and 1.0 in points:
            item["endpoint_swing"] = points[1.0] - points[-1.0]
        summary_rows.append(item)

    baseline = next(
        (x for x in summary_rows if x["target_distance"] == 0),
        summary_rows[0],
    )
    base_neutral = baseline.get("margin_at_zero")
    base_swing = baseline.get("endpoint_swing")

    for item in summary_rows:
        neutral = item.get("margin_at_zero")
        swing = item.get("endpoint_swing")
        item["neutral_shift_from_baseline"] = (
            None if base_neutral is None or neutral is None else neutral - base_neutral
        )
        item["swing_fraction_of_baseline"] = (
            None
            if base_swing in (None, 0.0) or swing is None
            else swing / base_swing
        )

    return {
        "rows": summary_rows,
        "decision": (
            "If masked spacing preserves the neutral basin while observer "
            "dose-response remains, the visible-filler basin flip was trajectory "
            "content rather than positional aging. If the neutral basin or "
            "observer swing changes under masked spacing, cache position/RoPE "
            "distance itself is load-bearing."
        ),
    }


def main():
    args = parse_args()
    distances = parse_csv_ints(args.distances)
    trust_grid = parse_csv_floats(args.trust_grid)
    plan = load_plan(Path(args.plan_receipt))
    receipt_path = Path(args.receipt)

    if 0 not in distances:
        raise ValueError("masked-distance control must include distance 0")

    print("Distance targets:", distances)
    print("Trust grid:", trust_grid)
    print("Growth mode: masked positional spacer")
    print("Loading one frozen Qwen model and one canonical growing cache...")

    import transformers

    print("Transformers version:", transformers.__version__)
    model, tokenizer = load_model(args)
    _messages, _rendered, ids, _mask, spans = initial_prompt(tokenizer, args)
    initial_ids = [int(x) for x in ids[0].tolist()]
    question_suffix = user_turn_suffix_ids(tokenizer, args.question)
    spacer_id = choose_spacer_id(tokenizer, args.spacer_text)

    controller = QwenObserverController(
        model,
        plan,
        trust=0.0,
        target_margin=args.target_margin,
        max_ratio=args.max_ratio,
    )
    controller.set_spans(spans)
    controller.install()

    cache = new_cache()
    history_ids: List[int] = []
    history_mask: List[int] = []

    payload = {
        "scope": (
            "Positional-distance control on one growing Qwen KV cache. Spacer "
            "tokens advance DynamicCache/RoPE position but remain masked from "
            "all later reads."
        ),
        "model": args.model,
        "transformers_version": transformers.__version__,
        "growth_mode": "masked-positional-spacer",
        "source_spans": {
            "A": list(spans.source_a),
            "B": list(spans.source_b),
        },
        "frozen_heads": [
            {
                "layer": h.layer,
                "query_head": h.query_head,
                "kv_head": h.kv_head,
            }
            for h in plan.heads
        ],
        "target_margin": args.target_margin,
        "max_ratio": args.max_ratio,
        "distance_targets": distances,
        "trust_grid": trust_grid,
        "candidate_a": args.candidate_a,
        "candidate_b": args.candidate_b,
        "spacer": {
            "text": args.spacer_text,
            "token_id": spacer_id,
            "decoded_token": tokenizer.decode([spacer_id]),
            "attention_visible_to_later_queries": False,
            "chunk_size": args.spacer_chunk,
        },
        "checkpoints": [],
        "complete": False,
    }

    try:
        controller.set_trust(0.0)
        controller.begin_generation(preserve_source_baseline=True)
        cache, logits = append_tokens(model, cache, history_ids, initial_ids)
        cache, logits, initial_answer, _emitted = greedy_continue(
            model,
            tokenizer,
            cache,
            history_ids,
            logits,
            max_new_tokens=args.max_new_tokens,
        )
        history_mask[:] = [1] * len(history_ids)

        if not controller.summary().get("cache_integrity_ok", False):
            raise RuntimeError("source K rows changed during initial answer")

        anchor_len = cache_length(cache)
        payload["anchor"] = {
            "initial_answer": initial_answer,
            "cache_tokens": anchor_len,
            "cache_fork_mode": cache_fork_mode(cache),
            "source_cache_integrity_ok": True,
        }
        write_receipt(receipt_path, payload)

        print("Cache fork mode:", payload["anchor"]["cache_fork_mode"])
        print(f"Anchor cache: {anchor_len} tokens")
        print("Initial answer:", initial_answer)

        for target_distance in distances:
            cache = grow_masked_to_distance(
                model,
                controller,
                cache,
                history_ids,
                history_mask,
                anchor_len=anchor_len,
                target_distance=target_distance,
                spacer_id=spacer_id,
                spacer_chunk=args.spacer_chunk,
            )
            print(
                f"\n=== masked checkpoint target={target_distance}, "
                f"actual={cache_length(cache) - anchor_len} ===",
                flush=True,
            )
            row = checkpoint_row(
                model,
                tokenizer,
                controller,
                cache,
                history_ids,
                history_mask,
                anchor_len=anchor_len,
                target_distance=target_distance,
                trust_grid=trust_grid,
                question_suffix=question_suffix,
                candidate_a=args.candidate_a,
                candidate_b=args.candidate_b,
            )
            payload["checkpoints"].append(row)
            payload["summary"] = summarize(payload)
            payload["canonical_cache_tokens"] = cache_length(cache)
            payload["source_cache_integrity_ok"] = bool(
                controller.cache_integrity_ok
            )
            write_receipt(receipt_path, payload)

        payload["complete"] = True
        payload["summary"] = summarize(payload)
        payload["canonical_cache_tokens"] = cache_length(cache)
        payload["source_cache_integrity_ok"] = bool(controller.cache_integrity_ok)
        write_receipt(receipt_path, payload)

    finally:
        controller.uninstall()
        payload["summary"] = summarize(payload)
        payload["canonical_cache_tokens"] = len(history_ids)
        payload["source_cache_integrity_ok"] = bool(controller.cache_integrity_ok)
        write_receipt(receipt_path, payload)
        print(f"\nreceipt: {receipt_path}")


if __name__ == "__main__":
    main()
