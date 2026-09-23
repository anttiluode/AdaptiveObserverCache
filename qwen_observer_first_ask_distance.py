"""First-ask masked-distance discriminator for Qwen AdaptiveObserverCache.

This fixes the re-ask confound in the earlier distance harness. The canonical
cache contains only the closed system/source message. Masked spacer rows may
then advance cache/RoPE position. The user question is appended for the first
and only time inside each isolated diagnostic branch.

Primary gate: summed sequence likelihood must switch winner with observer sign:
    m=-1 -> B wins (sum margin A-B < 0)
    m=+1 -> A wins (sum margin A-B > 0)

A matched first-divergent-token gap is reported as a length-neutral secondary
diagnostic because the two full answer candidates have different token counts.

Default small run:
    python3.13 qwen_observer_first_ask_distance.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

from qwen_chat_suffix import assistant_end_id
from qwen_first_ask_metrics import (
    first_divergence,
    sequence_control_summary,
    split_first_ask_ids,
    winner_flip_gate,
)
from qwen_observer import QwenObserverController, locate_source_spans
from qwen_observer_chat import (
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_QUESTION,
    encode_rendered,
    load_model,
    render_chat,
    system_message,
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
    load_plan,
    new_cache,
)
from qwen_observer_masked_distance import (
    append_tokens_with_mask,
    choose_spacer_id,
    grow_masked_to_distance,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Test whether AOC controls a first-ask decision after source records "
            "recede behind unreadable masked cache positions."
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
    p.add_argument("--gpu-memory", default="6GiB")
    p.add_argument("--cpu-memory", default="6GiB")
    p.add_argument("--offload-dir", default=".offload_qwen_observer")
    p.add_argument("--distances", default="0,256")
    p.add_argument("--trust-grid", default="-1,0,1")
    p.add_argument("--saturation-threshold", type=float, default=0.99)
    p.add_argument("--spacer-text", default="x")
    p.add_argument("--spacer-chunk", type=int, default=64)
    p.add_argument(
        "--continue-without-baseline-flip",
        action="store_true",
        help=(
            "continue to nonzero distance even if the summed sequence winner "
            "does not flip at distance zero; off by default because distance "
            "is uninterpretable without baseline control"
        ),
    )
    p.add_argument(
        "--receipt",
        default="results/qwen_observer_first_ask_distance.json",
    )
    return p.parse_args()


def first_ask_parts(tokenizer, args):
    """Tokenize exact first-ask prompt, then withhold question until checkpoint."""

    messages = [
        {
            "role": "system",
            "content": system_message(args.source_a, args.source_b),
        },
        {"role": "user", "content": args.question},
    ]
    rendered = render_chat(tokenizer, messages)
    ids, _mask, offsets = encode_rendered(tokenizer, rendered)
    full_ids = [int(x) for x in ids[0].tolist()]
    source_ids, question_suffix = split_first_ask_ids(
        full_ids,
        assistant_end_id=assistant_end_id(tokenizer),
    )
    spans = locate_source_spans(
        rendered,
        offsets,
        args.source_a,
        args.source_b,
    )
    source_end = max(spans.source_a[1], spans.source_b[1])
    if source_end > len(source_ids):
        raise RuntimeError(
            "source span extends beyond source-only cache boundary: "
            f"source_end={source_end}, source_cache_tokens={len(source_ids)}"
        )
    return source_ids, question_suffix, spans


@torch.inference_mode()
def score_candidate_first_ask(
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
    """Teacher-force one full candidate on an isolated first-ask branch."""

    canonical_len = cache_length(canonical_cache)
    history_len = len(canonical_history)
    mask_snapshot = tuple(canonical_mask)

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
    _assert_canonical_untouched(
        canonical_cache,
        canonical_history,
        canonical_mask,
        canonical_len,
        history_len,
        mask_snapshot,
    )
    if not summary.get("cache_integrity_ok", False):
        raise RuntimeError("source K rows failed integrity check during score")

    return {
        "text": candidate,
        "tokens": len(candidate_ids),
        "sum_logprob": sum(logprobs),
        "mean_logprob": sum(logprobs) / len(logprobs),
        "observer": summary,
    }


def _assert_canonical_untouched(
    cache,
    history_ids,
    history_mask,
    expected_cache_len,
    expected_history_len,
    expected_mask,
):
    if cache_length(cache) != expected_cache_len:
        raise RuntimeError("diagnostic branch mutated canonical KV-cache length")
    if len(history_ids) != expected_history_len:
        raise RuntimeError("diagnostic branch mutated canonical token history")
    if tuple(history_mask) != expected_mask:
        raise RuntimeError("diagnostic branch mutated canonical visibility mask")


@torch.inference_mode()
def score_decision_token(
    model,
    tokenizer,
    controller,
    canonical_cache,
    canonical_history: List[int],
    canonical_mask: List[int],
    question_suffix: list[int],
    candidate_a: str,
    candidate_b: str,
    trust: float,
):
    """Score the first equal-depth A/B token after their common answer prefix."""

    ids_a = tokenize_answer(tokenizer, candidate_a)
    ids_b = tokenize_answer(tokenizer, candidate_b)
    divergence = first_divergence(ids_a, ids_b)

    canonical_len = cache_length(canonical_cache)
    history_len = len(canonical_history)
    mask_snapshot = tuple(canonical_mask)

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
    for token_id in divergence["common_prefix"]:
        branch, logits = append_tokens_with_mask(
            model,
            branch,
            branch_history,
            branch_mask,
            [token_id],
            [1],
        )

    logp = torch.log_softmax(logits.float(), dim=-1)[0]
    logp_a = float(logp[divergence["token_a"]].item())
    logp_b = float(logp[divergence["token_b"]].item())
    summary = controller.summary()

    _assert_canonical_untouched(
        canonical_cache,
        canonical_history,
        canonical_mask,
        canonical_len,
        history_len,
        mask_snapshot,
    )
    if not summary.get("cache_integrity_ok", False):
        raise RuntimeError(
            "source K rows failed integrity check during decision-token score"
        )

    return {
        "common_prefix_tokens": len(divergence["common_prefix"]),
        "divergence_index": divergence["index"],
        "token_a": divergence["token_a"],
        "token_b": divergence["token_b"],
        "token_a_text": tokenizer.decode([divergence["token_a"]]),
        "token_b_text": tokenizer.decode([divergence["token_b"]]),
        "logprob_a": logp_a,
        "logprob_b": logp_b,
        "gap_A_minus_B": logp_a - logp_b,
        "observer": summary,
    }


def _score_trust(
    model,
    tokenizer,
    controller,
    cache,
    history_ids,
    history_mask,
    *,
    question_suffix,
    candidate_a,
    candidate_b,
    trust,
    saturation_threshold,
):
    a = score_candidate_first_ask(
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
    b = score_candidate_first_ask(
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
    sequence = sequence_control_summary(
        sum_a=a["sum_logprob"],
        sum_b=b["sum_logprob"],
        saturation_threshold=saturation_threshold,
    )
    decision = score_decision_token(
        model,
        tokenizer,
        controller,
        cache,
        history_ids,
        history_mask,
        question_suffix,
        candidate_a,
        candidate_b,
        trust,
    )
    return {
        "trust": float(trust),
        "A": a,
        "B": b,
        "sequence_decision": sequence,
        "matched_decision_token": decision,
    }


def checkpoint_row(
    model,
    tokenizer,
    controller,
    cache,
    history_ids,
    history_mask,
    *,
    anchor_len,
    target_distance,
    trust_grid,
    question_suffix,
    candidate_a,
    candidate_b,
    saturation_threshold,
):
    actual = cache_length(cache) - anchor_len
    row = {
        "target_distance": target_distance,
        "actual_added_tokens": actual,
        "canonical_cache_tokens": cache_length(cache),
        "masked_history_tokens": sum(1 for x in history_mask if x == 0),
        "first_ask": True,
        "dose_response": [],
    }

    # Neutral first: report saturation before interpreting observer dose-response.
    ordered = [0.0] + [x for x in trust_grid if x != 0.0]
    results = {}
    for trust in ordered:
        result = _score_trust(
            model,
            tokenizer,
            controller,
            cache,
            history_ids,
            history_mask,
            question_suffix=question_suffix,
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            trust=trust,
            saturation_threshold=saturation_threshold,
        )
        results[trust] = result
        seq = result["sequence_decision"]
        token = result["matched_decision_token"]
        if trust == 0.0:
            sat = "SATURATED" if seq["saturated"] else "not saturated"
            print(
                f"  neutral first-ask: winner={seq['winner']} "
                f"pair_p={seq['winner_pair_probability']:.6f} {sat}",
                flush=True,
            )
        print(
            f"  distance={actual:5d}  m={trust:+.2f}  "
            f"sumA={seq['sum_logprob_A']:+.5f}  "
            f"sumB={seq['sum_logprob_B']:+.5f}  "
            f"margin={seq['sum_margin_A_minus_B']:+.5f}  "
            f"winner={seq['winner']}  "
            f"token_gap={token['gap_A_minus_B']:+.5f}",
            flush=True,
        )

    row["dose_response"] = [results[x] for x in trust_grid]
    sequence_margins = {
        x: results[x]["sequence_decision"]["sum_margin_A_minus_B"]
        for x in trust_grid
    }
    token_margins = {
        x: results[x]["matched_decision_token"]["gap_A_minus_B"]
        for x in trust_grid
    }
    row["sequence_winner_flip_control"] = winner_flip_gate(sequence_margins)
    row["matched_token_winner_flip_control"] = winner_flip_gate(token_margins)
    row["neutral_saturated"] = results[0.0]["sequence_decision"]["saturated"]
    row["neutral_winner"] = results[0.0]["sequence_decision"]["winner"]

    print(
        "  gate: sequence winner flip="
        + ("PASS" if row["sequence_winner_flip_control"] else "FAIL")
        + "; matched-token flip="
        + ("PASS" if row["matched_token_winner_flip_control"] else "FAIL"),
        flush=True,
    )
    return row


def summarize(payload):
    rows = payload["checkpoints"]
    if not rows:
        return {}
    return {
        "rows": [
            {
                "target_distance": row["target_distance"],
                "actual_added_tokens": row["actual_added_tokens"],
                "neutral_winner": row["neutral_winner"],
                "neutral_saturated": row["neutral_saturated"],
                "sequence_winner_flip_control": row[
                    "sequence_winner_flip_control"
                ],
                "matched_token_winner_flip_control": row[
                    "matched_token_winner_flip_control"
                ],
            }
            for row in rows
        ],
        "distance_claim_rule": (
            "Distance is interpretable only if sequence_winner_flip_control "
            "passes at distance 0. Survival at a later checkpoint requires the "
            "same B-at-minus1/A-at-plus1 summed-sequence winner flip there."
        ),
    }


def main():
    args = parse_args()
    distances = parse_csv_ints(args.distances)
    trust_grid = parse_csv_floats(args.trust_grid)
    if not distances or distances[0] != 0:
        raise ValueError("first-ask distance sweep must begin at distance 0")
    if 0.0 not in trust_grid or -1.0 not in trust_grid or 1.0 not in trust_grid:
        raise ValueError("trust grid must contain -1, 0, and +1")

    plan = load_plan(Path(args.plan_receipt))
    receipt_path = Path(args.receipt)

    print("Distance targets:", distances)
    print("Trust grid:", trust_grid)
    print("Regime: FIRST ASK (no prior question or assistant answer in cache)")
    print("Growth mode: masked positional spacer")
    print("Loading one frozen Qwen model and one canonical growing cache...")

    import transformers

    print("Transformers version:", transformers.__version__)
    model, tokenizer = load_model(args)
    source_ids, question_suffix, spans = first_ask_parts(tokenizer, args)
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
            "First-ask AOC control versus positional distance. Canonical cache "
            "contains system/sources only; question is withheld until each "
            "isolated diagnostic branch. Masked spacer rows advance cache/RoPE "
            "position but remain unreadable to later queries."
        ),
        "model": args.model,
        "transformers_version": transformers.__version__,
        "regime": "first-ask",
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
        "distance_targets": distances,
        "trust_grid": trust_grid,
        "candidate_a": args.candidate_a,
        "candidate_b": args.candidate_b,
        "saturation_threshold": args.saturation_threshold,
        "question_was_in_anchor": False,
        "prior_assistant_answer_in_anchor": False,
        "spacer": {
            "text": args.spacer_text,
            "token_id": spacer_id,
            "decoded_token": tokenizer.decode([spacer_id]),
            "attention_visible_to_later_queries": False,
            "chunk_size": args.spacer_chunk,
        },
        "checkpoints": [],
        "complete": False,
        "run_status": "running",
    }

    try:
        controller.set_trust(0.0)
        controller.begin_generation(preserve_source_baseline=True)
        cache, _logits = append_tokens(model, cache, history_ids, source_ids)
        history_mask[:] = [1] * len(history_ids)
        if not controller.summary().get("cache_integrity_ok", False):
            raise RuntimeError("source K rows changed during source-only prefill")

        anchor_len = cache_length(cache)
        payload["anchor"] = {
            "cache_tokens": anchor_len,
            "question_suffix_tokens": len(question_suffix),
            "cache_fork_mode": cache_fork_mode(cache),
            "source_cache_integrity_ok": True,
            "question_has_been_asked": False,
            "assistant_answer_exists": False,
        }
        write_receipt(receipt_path, payload)
        print("Cache fork mode:", payload["anchor"]["cache_fork_mode"])
        print(f"Source-only anchor cache: {anchor_len} tokens")
        print(f"Withheld first-question suffix: {len(question_suffix)} tokens")

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
                f"\n=== FIRST-ASK checkpoint target={target_distance}, "
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
                saturation_threshold=args.saturation_threshold,
            )
            payload["checkpoints"].append(row)
            payload["summary"] = summarize(payload)
            payload["canonical_cache_tokens"] = cache_length(cache)
            payload["source_cache_integrity_ok"] = bool(
                controller.cache_integrity_ok
            )
            write_receipt(receipt_path, payload)

            if (
                target_distance == 0
                and not row["sequence_winner_flip_control"]
                and not args.continue_without_baseline_flip
            ):
                payload["run_status"] = "stopped_no_baseline_winner_flip"
                payload["complete"] = False
                payload["summary"] = summarize(payload)
                write_receipt(receipt_path, payload)
                print(
                    "\nSTOP: first-ask summed-sequence control did not flip "
                    "the winner at distance 0. Distance is not interpretable; "
                    "debug scoring/calibration before spending the +distance run.",
                    flush=True,
                )
                return

        payload["complete"] = True
        payload["run_status"] = "completed"
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
