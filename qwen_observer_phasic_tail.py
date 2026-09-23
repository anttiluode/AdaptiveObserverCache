"""Phasic-vs-tonic observer tail test for Qwen AdaptiveObserverCache.

Motivation (results/qwen_observer_first_ask_distance.json, 2026-09-23): at
+256 masked positions and m=-1 the observer made the valve/sensor decision
token favour "sensor" MORE strongly than at distance 0, yet the complete
sensor sentence lost because its tail ("K drifted.") got ~1.8 nats more
expensive. Two plain explanations:

  1. the observer, still active on tail tokens, disrupts copying;
  2. distance alone makes copying source-B details harder.

Same first-ask anchor, same masked spacers, same frozen heads. At each
distance both candidates are teacher-forced under three observer schedules:

  tonic    observer at trust m for every candidate token (the old harness)
  phasic   observer at trust m up to and including the decision token,
           then trust 0 for the tail
  neutral  trust 0 throughout

The pre-registered reading lives in qwen_tail_schedule.classify.

Default:
    python3.13 qwen_observer_phasic_tail.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from qwen_first_ask_metrics import first_divergence
from qwen_observer import QwenObserverController
from qwen_observer_chat import DEFAULT_A, DEFAULT_B, DEFAULT_QUESTION, load_model
from qwen_observer_distance_sweep import (
    DEFAULT_CANDIDATE_A,
    DEFAULT_CANDIDATE_B,
    cache_fork_mode,
    fork_cache,
    parse_csv_ints,
    tokenize_answer,
    write_receipt,
)
from qwen_observer_first_ask_distance import (
    _assert_canonical_untouched,
    first_ask_parts,
)
from qwen_observer_live_cache import append_tokens, cache_length, load_plan, new_cache
from qwen_observer_masked_distance import (
    append_tokens_with_mask,
    choose_spacer_id,
    grow_masked_to_distance,
)
from qwen_chat_suffix import assistant_end_id
from qwen_tail_schedule import (
    ARMS,
    classify,
    generation_choice,
    generation_verdict,
    split_logprobs,
    trust_for_index,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--source-a", default=DEFAULT_A)
    p.add_argument("--source-b", default=DEFAULT_B)
    p.add_argument("--question", default=DEFAULT_QUESTION)
    p.add_argument("--candidate-a", default=DEFAULT_CANDIDATE_A)
    p.add_argument("--candidate-b", default=DEFAULT_CANDIDATE_B)
    p.add_argument("--plan-receipt", default="results/qwen_observer_causal_compare.json")
    p.add_argument("--target-margin", type=float, default=3.0)
    p.add_argument("--max-ratio", type=float, default=1.0)
    p.add_argument("--gpu-memory", default="6GiB")
    p.add_argument("--cpu-memory", default="6GiB")
    p.add_argument("--offload-dir", default=".offload_qwen_observer")
    p.add_argument("--distances", default="0,256")
    p.add_argument(
        "--trust",
        type=float,
        default=-1.0,
        help="observer trust for tonic/phasic arms; must be negative (B-favouring)",
    )
    p.add_argument("--spacer-text", default="x")
    p.add_argument("--spacer-chunk", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--no-generate", action="store_true",
                   help="skip the greedy generation branches")
    p.add_argument("--receipt", default="results/qwen_observer_phasic_tail_gen.json")
    return p.parse_args()


def top_tokens(tokenizer, logits, k):
    logp = torch.log_softmax(logits.float(), dim=-1)[0]
    vals, ids = torch.topk(logp, k)
    return [
        {"token": tokenizer.decode([int(i)]), "id": int(i), "logprob": float(v)}
        for v, i in zip(vals.tolist(), ids.tolist())
    ]


@torch.inference_mode()
def generate_scheduled(
    model,
    tokenizer,
    controller,
    canonical_cache,
    canonical_history: List[int],
    canonical_mask: List[int],
    question_suffix: List[int],
    *,
    arm: str,
    trust: float,
    decision_index: int,
    max_new_tokens: int,
    top_k: int,
):
    """Greedy generation on an isolated branch that keeps spacer rows masked.

    The trust schedule uses the same index rule as scoring: while producing
    emitted token i the observer runs at trust_for_index(arm, i, ...).
    """

    canonical_len = cache_length(canonical_cache)
    history_len = len(canonical_history)
    mask_snapshot = tuple(canonical_mask)

    branch = fork_cache(canonical_cache)
    branch_history = list(canonical_history)
    branch_mask = list(canonical_mask)

    end_id = assistant_end_id(tokenizer)
    eos_id = tokenizer.eos_token_id
    stop_ids = {int(end_id)} | ({int(eos_id)} if eos_id is not None else set())

    controller.begin_generation(preserve_source_baseline=True)
    controller.set_trust(trust_for_index(arm, 0, decision_index, trust))
    branch, logits = append_tokens_with_mask(
        model, branch, branch_history, branch_mask,
        question_suffix, [1] * len(question_suffix),
    )

    emitted: List[int] = []
    steps = []
    stopped = False
    for i in range(max_new_tokens):
        steps.append({
            "index": i,
            "trust": trust_for_index(arm, i, decision_index, trust),
            "top": top_tokens(tokenizer, logits, top_k),
        })
        next_id = int(torch.argmax(logits.float(), dim=-1)[0])
        if next_id in stop_ids:
            stopped = True
            break
        emitted.append(next_id)
        controller.set_trust(trust_for_index(arm, i + 1, decision_index, trust))
        branch, logits = append_tokens_with_mask(
            model, branch, branch_history, branch_mask, [next_id], [1],
        )

    controller.set_trust(0.0)
    summary = controller.summary()
    _assert_canonical_untouched(
        canonical_cache, canonical_history, canonical_mask,
        canonical_len, history_len, mask_snapshot,
    )
    if not summary.get("cache_integrity_ok", False):
        raise RuntimeError("source K rows failed integrity check during generation")

    return {
        "text": tokenizer.decode(emitted, skip_special_tokens=True).strip(),
        "tokens": [tokenizer.decode([t]) for t in emitted],
        "stopped_on_end_token": stopped,
        "steps": steps,
        "observer": summary,
    }


@torch.inference_mode()
def score_scheduled(
    model,
    tokenizer,
    controller,
    canonical_cache,
    canonical_history: List[int],
    canonical_mask: List[int],
    question_suffix: List[int],
    candidate: str,
    *,
    arm: str,
    trust: float,
    decision_index: int,
    top_k: int = 5,
):
    """Teacher-force one candidate on an isolated branch with a per-token trust schedule."""

    canonical_len = cache_length(canonical_cache)
    history_len = len(canonical_history)
    mask_snapshot = tuple(canonical_mask)

    branch = fork_cache(canonical_cache)
    branch_history = list(canonical_history)
    branch_mask = list(canonical_mask)

    candidate_ids = tokenize_answer(tokenizer, candidate)
    schedule = [
        trust_for_index(arm, i, decision_index, trust)
        for i in range(len(candidate_ids))
    ]

    controller.begin_generation(preserve_source_baseline=True)
    controller.set_trust(schedule[0])
    branch, logits = append_tokens_with_mask(
        model, branch, branch_history, branch_mask,
        question_suffix, [1] * len(question_suffix),
    )

    logprobs: List[float] = []
    tops = []
    for index, token_id in enumerate(candidate_ids):
        lp = torch.log_softmax(logits.float(), dim=-1)[0, token_id]
        logprobs.append(float(lp.item()))
        tops.append(top_tokens(tokenizer, logits, top_k))
        if index + 1 < len(candidate_ids):
            controller.set_trust(schedule[index + 1])
            branch, logits = append_tokens_with_mask(
                model, branch, branch_history, branch_mask, [token_id], [1],
            )

    controller.set_trust(0.0)
    summary = controller.summary()
    _assert_canonical_untouched(
        canonical_cache, canonical_history, canonical_mask,
        canonical_len, history_len, mask_snapshot,
    )
    if not summary.get("cache_integrity_ok", False):
        raise RuntimeError("source K rows failed integrity check during scheduled score")

    split = split_logprobs(logprobs, decision_index)
    return {
        "text": candidate,
        "token_texts": [tokenizer.decode([t]) for t in candidate_ids],
        "trust_schedule": schedule,
        "logprobs": logprobs,
        "top": tops,
        **split,
        "observer": summary,
    }


def checkpoint(model, tokenizer, controller, cache, history_ids, history_mask, *,
               args, question_suffix, decision_index, anchor_len, word_a, word_b):
    actual = cache_length(cache) - anchor_len
    print(f"\n=== checkpoint +{actual} masked positions, trust={args.trust:+.2f} ===",
          flush=True)
    arms = {}
    for arm in ARMS:
        arms[arm] = {}
        for label, text in (("A", args.candidate_a), ("B", args.candidate_b)):
            arms[arm][label] = score_scheduled(
                model, tokenizer, controller, cache, history_ids, history_mask,
                question_suffix, text,
                arm=arm, trust=args.trust, decision_index=decision_index,
                top_k=args.top_k,
            )
        a, b = arms[arm]["A"], arms[arm]["B"]
        margin = a["total"] - b["total"]
        print(
            f"  {arm:8s} sumA={a['total']:+9.4f} sumB={b['total']:+9.4f} "
            f"margin={margin:+8.4f} winner={'A' if margin > 0 else 'B'}  "
            f"| decision A={a['decision']:+7.3f} B={b['decision']:+7.3f}  "
            f"| tail A={a['tail']:+8.4f} B={b['tail']:+8.4f}",
            flush=True,
        )
    print("  what the model wanted instead of B's final token:")
    for arm in ARMS:
        top = arms[arm]["B"]["top"][-1]
        shown = "  ".join(f"{t['token']!r}:{t['logprob']:+.2f}" for t in top)
        print(f"    {arm:8s} {shown}")

    generations = {}
    if not args.no_generate:
        print("  greedy generation:")
        for arm in ARMS:
            gen = generate_scheduled(
                model, tokenizer, controller, cache, history_ids, history_mask,
                question_suffix, arm=arm, trust=args.trust,
                decision_index=decision_index,
                max_new_tokens=args.max_new_tokens, top_k=args.top_k,
            )
            gen["choice"] = generation_choice(gen["text"], word_a, word_b)
            generations[arm] = gen
            print(f"    {arm:8s} [{gen['choice']:7s}] {gen['text']!r}", flush=True)
    return {"added_masked_positions": actual, "arms": arms, "generation": generations}


def main():
    args = parse_args()
    if not args.trust < 0.0:
        raise ValueError("--trust must be negative: the test asks whether B can be kept")
    distances = parse_csv_ints(args.distances)
    if not distances or distances[0] != 0 or len(distances) < 2:
        raise ValueError("--distances must start at 0 and include a far checkpoint")

    plan = load_plan(Path(args.plan_receipt))
    receipt_path = Path(args.receipt)

    import transformers

    print("Transformers version:", transformers.__version__)
    model, tokenizer = load_model(args)
    source_ids, question_suffix, spans = first_ask_parts(tokenizer, args)
    spacer_id = choose_spacer_id(tokenizer, args.spacer_text)

    divergence = first_divergence(
        tokenize_answer(tokenizer, args.candidate_a),
        tokenize_answer(tokenizer, args.candidate_b),
    )
    decision_index = divergence["index"]
    word_a = tokenizer.decode([divergence["token_a"]])
    word_b = tokenizer.decode([divergence["token_b"]])
    print(
        f"Decision token index {decision_index}: "
        f"A={tokenizer.decode([divergence['token_a']])!r} "
        f"B={tokenizer.decode([divergence['token_b']])!r}"
    )

    controller = QwenObserverController(
        model, plan, trust=0.0,
        target_margin=args.target_margin, max_ratio=args.max_ratio,
    )
    controller.set_spans(spans)
    controller.install()

    cache = new_cache()
    history_ids: List[int] = []
    history_mask: List[int] = []
    payload = {
        "scope": (
            "Phasic vs tonic vs neutral observer schedules on the first-ask, "
            "masked-distance Qwen cache. Separates observer-caused tail damage "
            "from distance-caused tail damage."
        ),
        "model": args.model,
        "transformers_version": transformers.__version__,
        "trust": args.trust,
        "distances": distances,
        "decision_index": decision_index,
        "candidate_a": args.candidate_a,
        "candidate_b": args.candidate_b,
        "arms": list(ARMS),
        "checkpoints": [],
        "complete": False,
    }

    try:
        controller.set_trust(0.0)
        controller.begin_generation(preserve_source_baseline=True)
        cache, _ = append_tokens(model, cache, history_ids, source_ids)
        history_mask[:] = [1] * len(history_ids)
        anchor_len = cache_length(cache)
        payload["anchor"] = {
            "cache_tokens": anchor_len,
            "cache_fork_mode": cache_fork_mode(cache),
        }
        print(f"Source-only anchor cache: {anchor_len} tokens")

        for target in distances:
            cache = grow_masked_to_distance(
                model, controller, cache, history_ids, history_mask,
                anchor_len=anchor_len, target_distance=target,
                spacer_id=spacer_id, spacer_chunk=args.spacer_chunk,
            )
            row = checkpoint(
                model, tokenizer, controller, cache, history_ids, history_mask,
                args=args, question_suffix=question_suffix,
                decision_index=decision_index, anchor_len=anchor_len,
                word_a=word_a, word_b=word_b,
            )
            payload["checkpoints"].append(row)
            write_receipt(receipt_path, payload)

        near = payload["checkpoints"][0]["arms"]
        far = payload["checkpoints"][-1]["arms"]
        readout = classify(near, far)
        if not args.no_generate:
            near_c = {a: g["choice"] for a, g in payload["checkpoints"][0]["generation"].items()}
            far_c = {a: g["choice"] for a, g in payload["checkpoints"][-1]["generation"].items()}
            readout["generation_near"] = near_c
            readout["generation_far"] = far_c
            readout["generation_verdict"] = generation_verdict(near_c, far_c)
        payload["readout"] = readout
        payload["complete"] = True
        write_receipt(receipt_path, payload)

        print("\n=== readout ===")
        print(f"  near winners: {readout['near_winner']}")
        print(f"  far winners:  {readout['far_winner']}")
        print(f"  observer tail damage (phasic B tail - tonic B tail): "
              f"{readout['observer_tail_damage_nats']:+.3f} nats")
        print(f"  distance tail cost (neutral B tail, near - far):     "
              f"{readout['distance_tail_cost_nats']:+.3f} nats")
        print(f"  decision-token mismatch tonic vs phasic (should be ~0): "
              f"{readout['far_decision_logprob_mismatch_tonic_vs_phasic']:.4f}")
        print(f"  likelihood VERDICT: {readout['verdict']}")
        if "generation_verdict" in readout:
            print(f"  generation near: {readout['generation_near']}")
            print(f"  generation far:  {readout['generation_far']}")
            print(f"  generation VERDICT: {readout['generation_verdict']}")
    finally:
        controller.uninstall()
        write_receipt(receipt_path, payload)
        print(f"\nreceipt: {receipt_path}")


if __name__ == "__main__":
    main()
