"""Distance x trust discriminator on one growing Qwen KV cache.

The canonical cache is filled exactly once, receives a neutral initial answer,
and then grows only through deterministic neutral filler turns. At each distance
checkpoint every observer-trust probe runs on a fork of that exact cache, so a
probe cannot contaminate the history seen by later trust values.

Default run is intentionally small:
    python3.13 qwen_observer_distance_sweep.py

Full preregistered-style grid:
    python3.13 qwen_observer_distance_sweep.py \
      --distances 0,128,512,1024,2048 \
      --trust-grid=-1,-0.75,-0.5,0,0.5,0.75,1 \
      --generate-endpoints \
      --receipt results/qwen_observer_distance_full.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

from qwen_chat_suffix import assistant_end_id, user_turn_suffix_ids
from qwen_observer import QwenObserverController
from qwen_observer_chat import (
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_QUESTION,
    load_model,
)
from qwen_observer_live_cache import (
    append_tokens,
    cache_length,
    greedy_continue,
    initial_prompt,
    load_plan,
    new_cache,
)


DEFAULT_CANDIDATE_A = "The device failed because valve C was obstructed."
DEFAULT_CANDIDATE_B = "The device failed because sensor K drifted."


def parse_csv_ints(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("distance list cannot be empty")
    if any(x < 0 for x in values):
        raise ValueError("distances must be non-negative")
    if values != sorted(set(values)):
        raise ValueError("distances must be unique and ascending")
    return values


def parse_csv_floats(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("trust grid cannot be empty")
    if any(x < -1.0 or x > 1.0 for x in values):
        raise ValueError("trust values must lie in [-1, 1]")
    if len(values) != len(set(values)):
        raise ValueError("trust values must be unique")
    return values


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Measure observer dose-response versus temporal distance while one "
            "canonical Qwen KV cache grows."
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
    p.add_argument(
        "--distances",
        default="0,256",
        help=(
            "target added-token distances after the initial closed answer; "
            "neutral filler turns may overshoot a target slightly"
        ),
    )
    p.add_argument(
        "--trust-grid",
        default="-1,0,1",
        help=(
            "comma-separated observer values. When the first value is negative, "
            "use --trust-grid=-1,0,1 form so argparse does not treat it as a flag"
        ),
    )
    p.add_argument(
        "--generate-endpoints",
        action="store_true",
        help="also greedily generate from branch caches at m=-1,0,+1",
    )
    p.add_argument(
        "--receipt",
        default="results/qwen_observer_distance_sweep.json",
    )
    return p.parse_args()


def _layer_tensor_pairs(cache):
    """Read K/V tensors from the newer DynamicCache layer API.

    The tensors are intentionally not cloned here. DynamicLayer.update grows
    caches by assigning torch.cat(...) results, so the branch gets a new tensor
    when tokens are appended while the canonical historical tensor remains
    untouched. The caller still verifies canonical length and source digests.
    """

    layers = getattr(cache, "layers", None)
    if layers is None:
        return None

    pairs = []
    for index, layer in enumerate(layers):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None:
            raise RuntimeError(
                f"DynamicCache layer {index} exposes no keys/values tensors"
            )
        pairs.append((keys, values))
    return pairs


def cache_fork_mode(cache) -> str:
    """Describe the API path used to fork this installed Transformers cache."""

    from transformers import DynamicCache

    to_legacy = getattr(cache, "to_legacy_cache", None)
    from_legacy = getattr(DynamicCache, "from_legacy_cache", None)
    if callable(to_legacy) and callable(from_legacy):
        return "legacy-conversion"

    if _layer_tensor_pairs(cache) is not None:
        return "layer-tensor-constructor"

    return "unsupported"


def fork_cache(cache):
    """Branch a DynamicCache across both legacy and newer Transformers APIs.

    Older Transformers releases expose to_legacy_cache()/from_legacy_cache().
    Newer releases removed those helpers and expose cache.layers[i].keys/values;
    they reconstruct DynamicCache from existing K/V through ddp_cache_data.

    Historical tensors are shared read-only at fork time. Appending to the
    branch must not advance the canonical cache; every caller checks that
    invariant immediately after the probe.
    """

    if cache is None:
        raise RuntimeError("distance sweep requires a DynamicCache")

    from transformers import DynamicCache, __version__ as transformers_version

    canonical_len = cache_length(cache)
    to_legacy = getattr(cache, "to_legacy_cache", None)
    from_legacy = getattr(DynamicCache, "from_legacy_cache", None)

    if callable(to_legacy) and callable(from_legacy):
        branch = from_legacy(to_legacy())
    else:
        pairs = _layer_tensor_pairs(cache)
        if pairs is None:
            raise RuntimeError(
                "unsupported DynamicCache API in transformers "
                f"{transformers_version}: neither legacy conversion nor "
                "cache.layers K/V tensors are available"
            )

        # Transformers >=4.56 / 5.x accepts existing K/V through
        # ddp_cache_data. Keep a positional fallback for intermediate API
        # variants that accepted the same iterable as the first argument.
        try:
            branch = DynamicCache(ddp_cache_data=pairs)
        except TypeError as keyword_error:
            try:
                branch = DynamicCache(pairs)
            except Exception as positional_error:
                raise RuntimeError(
                    "could not reconstruct DynamicCache branch under "
                    f"transformers {transformers_version}; "
                    f"keyword error={keyword_error!r}; "
                    f"positional error={positional_error!r}"
                ) from positional_error

    if cache_length(branch) != canonical_len:
        raise RuntimeError(
            "forked cache length differs from canonical cache: "
            f"branch={cache_length(branch)} canonical={canonical_len}"
        )
    return branch


def tokenize_answer(tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids = [int(x) for x in ids]
    if not ids:
        raise ValueError("candidate answer tokenized to an empty sequence")
    return ids


@torch.inference_mode()
def score_candidate(
    model,
    tokenizer,
    controller,
    canonical_cache,
    canonical_history: List[int],
    question_suffix: list[int],
    candidate: str,
    trust: float,
):
    """Teacher-force one candidate on an isolated cache branch."""

    canonical_len = cache_length(canonical_cache)
    branch = fork_cache(canonical_cache)
    branch_history = list(canonical_history)

    controller.set_trust(trust)
    controller.begin_generation(preserve_source_baseline=True)
    branch, logits = append_tokens(
        model, branch, branch_history, question_suffix
    )

    candidate_ids = tokenize_answer(tokenizer, candidate)
    logprobs: list[float] = []
    for index, token_id in enumerate(candidate_ids):
        lp = torch.log_softmax(logits.float(), dim=-1)[0, token_id]
        logprobs.append(float(lp.item()))
        if index + 1 < len(candidate_ids):
            branch, logits = append_tokens(
                model, branch, branch_history, [token_id]
            )

    summary = controller.summary()
    if cache_length(canonical_cache) != canonical_len:
        raise RuntimeError(
            "read-only branch mutated canonical KV-cache length"
        )
    if len(canonical_history) != canonical_len:
        raise RuntimeError(
            "read-only branch mutated canonical token history"
        )
    if not summary.get("cache_integrity_ok", False):
        raise RuntimeError(
            "historical source K rows failed integrity check during branch score"
        )

    return {
        "text": candidate,
        "tokens": len(candidate_ids),
        "sum_logprob": sum(logprobs),
        "mean_logprob": sum(logprobs) / len(logprobs),
        "observer": summary,
    }


@torch.inference_mode()
def generate_branch(
    model,
    tokenizer,
    controller,
    canonical_cache,
    canonical_history: List[int],
    question_suffix: list[int],
    trust: float,
    max_new_tokens: int,
):
    canonical_len = cache_length(canonical_cache)
    branch = fork_cache(canonical_cache)
    branch_history = list(canonical_history)

    controller.set_trust(trust)
    controller.begin_generation(preserve_source_baseline=True)
    branch, logits = append_tokens(
        model, branch, branch_history, question_suffix
    )
    branch, logits, text, emitted = greedy_continue(
        model,
        tokenizer,
        branch,
        branch_history,
        logits,
        max_new_tokens=max_new_tokens,
    )
    summary = controller.summary()

    if cache_length(canonical_cache) != canonical_len:
        raise RuntimeError(
            "generation branch mutated canonical KV-cache length"
        )
    if len(canonical_history) != canonical_len:
        raise RuntimeError(
            "generation branch mutated canonical token history"
        )
    if not summary.get("cache_integrity_ok", False):
        raise RuntimeError(
            "historical source K rows failed integrity check during branch generation"
        )

    return {
        "trust": float(trust),
        "answer": text,
        "emitted_tokens": len(emitted),
        "observer": summary,
    }


def filler_turn_ids(tokenizer, serial: int) -> list[int]:
    """One deterministic semantically neutral, syntactically closed chat turn."""

    user = (
        f"Calibration filler turn {serial}. "
        "No new evidence about the earlier device failure is provided."
    )
    suffix = user_turn_suffix_ids(tokenizer, user)
    assistant = tokenizer.encode(
        "Acknowledged.", add_special_tokens=False
    )
    return (
        [int(x) for x in suffix]
        + [int(x) for x in assistant]
        + [assistant_end_id(tokenizer)]
    )


@torch.inference_mode()
def grow_to_distance(
    model,
    tokenizer,
    controller,
    cache,
    history_ids: List[int],
    *,
    anchor_len: int,
    target_distance: int,
    filler_serial: int,
):
    """Advance only the canonical cache with neutral deterministic turns."""

    while cache_length(cache) - anchor_len < target_distance:
        filler_serial += 1
        tokens = filler_turn_ids(tokenizer, filler_serial)
        controller.set_trust(0.0)
        controller.begin_generation(preserve_source_baseline=True)
        cache, _logits = append_tokens(
            model, cache, history_ids, tokens
        )
        summary = controller.summary()
        if not summary.get("cache_integrity_ok", False):
            raise RuntimeError(
                "source K rows changed while growing neutral canonical cache"
            )
    return cache, filler_serial


def write_receipt(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def checkpoint_row(
    model,
    tokenizer,
    controller,
    cache,
    history_ids,
    *,
    anchor_len: int,
    target_distance: int,
    trust_grid: list[float],
    question_suffix: list[int],
    candidate_a: str,
    candidate_b: str,
    generate_endpoints: bool,
    max_new_tokens: int,
):
    actual_distance = cache_length(cache) - anchor_len
    row = {
        "target_distance": target_distance,
        "actual_added_tokens": actual_distance,
        "canonical_cache_tokens": cache_length(cache),
        "dose_response": [],
        "generation": [],
    }

    for trust in trust_grid:
        score_a = score_candidate(
            model,
            tokenizer,
            controller,
            cache,
            history_ids,
            question_suffix,
            candidate_a,
            trust,
        )
        score_b = score_candidate(
            model,
            tokenizer,
            controller,
            cache,
            history_ids,
            question_suffix,
            candidate_b,
            trust,
        )
        margin = score_a["mean_logprob"] - score_b["mean_logprob"]
        dose = {
            "trust": float(trust),
            "A": score_a,
            "B": score_b,
            "mean_logprob_margin_A_minus_B": margin,
        }
        row["dose_response"].append(dose)
        print(
            f"  distance={actual_distance:5d}  m={trust:+.2f}  "
            f"mean dlogp(A-B)={margin:+.5f}",
            flush=True,
        )

    if generate_endpoints:
        available = set(trust_grid)
        for trust in (-1.0, 0.0, 1.0):
            if trust not in available:
                continue
            run = generate_branch(
                model,
                tokenizer,
                controller,
                cache,
                history_ids,
                question_suffix,
                trust,
                max_new_tokens,
            )
            row["generation"].append(run)
            print(
                f"    generate m={trust:+.2f}: {run['answer']}",
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
            "margin_at_minus1": points.get(-1.0),
            "margin_at_zero": points.get(0.0),
            "margin_at_plus1": points.get(1.0),
        }
        if -1.0 in points and 1.0 in points:
            item["endpoint_swing"] = points[1.0] - points[-1.0]
        summary_rows.append(item)

    baseline = next(
        (x for x in summary_rows if x["target_distance"] == 0),
        summary_rows[0],
    )
    base_swing = baseline.get("endpoint_swing")
    for item in summary_rows:
        swing = item.get("endpoint_swing")
        item["swing_fraction_of_baseline"] = (
            None
            if base_swing in (None, 0.0) or swing is None
            else swing / base_swing
        )
    return {
        "rows": summary_rows,
        "interpretation_boundary": (
            "Do not infer success from attention mass alone. Compare the "
            "A-minus-B likelihood dose-response and its endpoint swing across "
            "distance while canonical source integrity remains true."
        ),
    }


def main():
    args = parse_args()
    distances = parse_csv_ints(args.distances)
    trust_grid = parse_csv_floats(args.trust_grid)
    plan = load_plan(Path(args.plan_receipt))
    receipt_path = Path(args.receipt)

    print("Distance targets:", distances)
    print("Trust grid:", trust_grid)
    print("Loading one frozen Qwen model and one canonical growing cache...")

    import transformers

    print("Transformers version:", transformers.__version__)
    model, tokenizer = load_model(args)
    _messages, _rendered, ids, _mask, spans = initial_prompt(tokenizer, args)
    initial_ids = [int(x) for x in ids[0].tolist()]
    question_suffix = user_turn_suffix_ids(tokenizer, args.question)

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
    payload = {
        "scope": (
            "Matched observer trust x temporal-distance sweep on one canonical "
            "growing Qwen KV cache. All diagnostic probes use isolated cache "
            "branches and must not advance the canonical history."
        ),
        "model": args.model,
        "transformers_version": transformers.__version__,
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
        "checkpoint_definition": (
            "added canonical KV tokens after the initial closed assistant answer"
        ),
        "checkpoints": [],
        "complete": False,
    }

    try:
        controller.set_trust(0.0)
        controller.begin_generation(preserve_source_baseline=True)
        cache, logits = append_tokens(
            model, cache, history_ids, initial_ids
        )
        cache, logits, initial_answer, _emitted = greedy_continue(
            model,
            tokenizer,
            cache,
            history_ids,
            logits,
            max_new_tokens=args.max_new_tokens,
        )
        if not controller.summary().get("cache_integrity_ok", False):
            raise RuntimeError("source K rows changed during initial answer")

        anchor_len = cache_length(cache)
        payload["anchor"] = {
            "initial_answer": initial_answer,
            "cache_tokens": anchor_len,
            "cache_fork_mode": cache_fork_mode(cache),
            "source_cache_integrity_ok": True,
        }
        print("Cache fork mode:", payload["anchor"]["cache_fork_mode"])
        write_receipt(receipt_path, payload)
        print(f"Anchor cache: {anchor_len} tokens")
        print("Initial answer:", initial_answer)

        filler_serial = 0
        for target_distance in distances:
            cache, filler_serial = grow_to_distance(
                model,
                tokenizer,
                controller,
                cache,
                history_ids,
                anchor_len=anchor_len,
                target_distance=target_distance,
                filler_serial=filler_serial,
            )
            actual = cache_length(cache) - anchor_len
            print(
                f"\n=== checkpoint target={target_distance}, actual={actual} ===",
                flush=True,
            )
            row = checkpoint_row(
                model,
                tokenizer,
                controller,
                cache,
                history_ids,
                anchor_len=anchor_len,
                target_distance=target_distance,
                trust_grid=trust_grid,
                question_suffix=question_suffix,
                candidate_a=args.candidate_a,
                candidate_b=args.candidate_b,
                generate_endpoints=args.generate_endpoints,
                max_new_tokens=args.max_new_tokens,
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
        payload["source_cache_integrity_ok"] = bool(
            controller.cache_integrity_ok
        )
        write_receipt(receipt_path, payload)

    finally:
        controller.uninstall()
        payload["summary"] = summarize(payload)
        payload["canonical_cache_tokens"] = len(history_ids)
        payload["source_cache_integrity_ok"] = bool(
            controller.cache_integrity_ok
        )
        write_receipt(receipt_path, payload)
        print(f"\nreceipt: {receipt_path}")


if __name__ == "__main__":
    main()
