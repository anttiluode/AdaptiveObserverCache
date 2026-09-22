"""Order-reversal attacker for the Qwen3-8B AdaptiveObserverCache observer.

WHAT THIS ISOLATES
------------------
Exactly one variable: the physical order of the two source blocks in the
prompt.  Content, wording, entities, labels and the observer head plan are all
held fixed.  Everything else about the run is identical to the canonical arm.

WHY IT IS ITS OWN ARM
---------------------
Position is the specific attacker that already killed this mechanism once, at
DistilGPT2 Gate 3, where the frozen coordinate turned out to be slot-1 / slot-2
rather than trust-A / trust-B and the order-swap attacker scored 0/2 with the
cosine flipping sign.  The Qwen neutral arm currently leans ~2.25:1 toward A,
which is consistent with a primacy prior.  If a held-out run changes entities,
facts, wording AND order together and fails, the result cannot distinguish
"task-specific heads" from "slot-reading heads".  Those are different
diagnoses with different next steps, so order gets measured alone.

The head identities are FROZEN and passed in on the command line.  This script
deliberately does not re-run any selection scan.  Re-selecting would answer a
different question.

MEASUREMENT
-----------
Full candidate-sequence log-likelihood under the actual steered decode process,
not the argmax token.  The candidate is scored one token at a time through the
KV cache so that the controller steers each position exactly as it does during
generation (the controller edits query_states[..., -1, :], i.e. only the
current position, so a single teacher-forced forward pass would under-apply
the intervention by a factor of the sequence length).

The absolute value of delta = LL(valve) - LL(sensor) is not interpretable: the
two candidates have different token counts, which contributes a constant
offset.  What is interpretable is how delta MOVES with trust, and whether that
movement survives order reversal.

PRE-REGISTERED READING (fill in before you look at the numbers)
---------------------------------------------------------------
Let slope = d(delta)/d(m) estimated over the sweep, per arm.

  IDENTITY COORDINATE   sign(slope_ab) == sign(slope_ba)
                        and 0.5 <= |slope_ba / slope_ab| <= 2.0
                        and the generated sentence crosses in both arms.

  SLOT-READING          sign(slope_ba) != sign(slope_ab)
                        or |slope_ba / slope_ab| < 0.25

  INCONCLUSIVE          anything else.  Report as inconclusive; do not narrate
                        a story around it.

Secondary, and independent of the above: compare delta at m = 0 across the two
arms.  If neutral delta tracks physical slot rather than content, the 2.25:1
lean is a primacy prior and should be reported as one.

Usage
-----
    python attack_order.py \
        --heads 24:29,24:23,30:11,18:28 \
        --receipt results/attack_order.json

Add --generate-at -1,0,1 to control which sweep points also produce a decoded
sentence (generation is the slow part under disk offload; likelihoods are
computed at every sweep point regardless).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_observer import (
    QwenObserverController,
    SourceSpans,
    locate_source_spans,
)


DEFAULT_A = (
    "The device failed because valve C was obstructed. "
    "The pressure log supports this diagnosis."
)
DEFAULT_B = (
    "The device failed because sensor K drifted. "
    "The calibration log supports this diagnosis."
)
DEFAULT_QUESTION = "Why did the device fail? Give the cause in one short sentence."

CAND_A = "The device failed because valve C was obstructed."
CAND_B = "The device failed because sensor K drifted."


# --------------------------------------------------------------------------
# Frozen head plan.
#
# Duck-typed against QwenObserverController.install(), which only ever calls
# plan.by_layer().  Using a local type here instead of importing HeadSelection
# means this script keeps working if the selection dataclass is refactored,
# and it makes it structurally impossible to accidentally carry a selection
# statistic into an attack that is supposed to be selection-free.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenHead:
    layer: int
    query_head: int


@dataclass(frozen=True)
class FrozenPlan:
    heads: Tuple[FrozenHead, ...]

    def by_layer(self) -> Dict[int, Tuple[int, ...]]:
        grouped: Dict[int, List[int]] = {}
        for item in self.heads:
            grouped.setdefault(item.layer, []).append(item.query_head)
        return {layer: tuple(hs) for layer, hs in grouped.items()}


def parse_heads(spec: str) -> FrozenPlan:
    heads = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        layer, qhead = chunk.split(":")
        heads.append(FrozenHead(layer=int(layer), query_head=int(qhead)))
    if not heads:
        raise ValueError("no heads parsed from --heads")
    return FrozenPlan(heads=tuple(heads))


def parse_floats(spec: str) -> List[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(
        description="Order-reversal attacker with a frozen observer head plan."
    )
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--source-a", default=DEFAULT_A)
    p.add_argument("--source-b", default=DEFAULT_B)
    p.add_argument("--candidate-a", default=CAND_A)
    p.add_argument("--candidate-b", default=CAND_B)
    p.add_argument("--question", default=DEFAULT_QUESTION)
    p.add_argument(
        "--heads",
        default="24:29,24:23,30:11,18:28",
        help="frozen layer:query_head list from the causal scan; NOT re-selected here",
    )
    p.add_argument(
        "--trust-sweep",
        default="-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1",
    )
    p.add_argument(
        "--generate-at",
        default="-1,0,1",
        help="sweep points that also get a decoded sentence; '' for none",
    )
    p.add_argument("--target-margin", type=float, default=3.0)
    p.add_argument("--max-ratio", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--gpu-memory", default="6GiB")
    p.add_argument("--cpu-memory", default="6GiB")
    p.add_argument("--offload-dir", default=".offload_qwen_observer")
    p.add_argument("--receipt", default="results/attack_order.json")
    return p.parse_args()


# --------------------------------------------------------------------------
# Model plumbing (same profile as qwen_observer_chat.py)
# --------------------------------------------------------------------------


def input_device(model) -> torch.device:
    emb = model.get_input_embeddings()
    hook = getattr(emb, "_hf_hook", None)
    execution_device = getattr(hook, "execution_device", None)
    if execution_device is not None:
        return torch.device(execution_device)
    for parameter in emb.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs = {"dtype": dtype, "device_map": "auto", "low_cpu_mem_usage": True}
    if torch.cuda.is_available():
        kwargs["max_memory"] = {0: args.gpu_memory, "cpu": args.cpu_memory}
        kwargs["offload_folder"] = args.offload_dir
        kwargs["offload_state_dict"] = True

    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def system_message(source_a: str, source_b: str, order: str) -> str:
    """Labels travel with their content.  Only physical order changes."""
    block_a = "[BEGIN SOURCE A]\n" + source_a + "\n[END SOURCE A]"
    block_b = "[BEGIN SOURCE B]\n" + source_b + "\n[END SOURCE B]"
    blocks = (block_a, block_b) if order == "ab" else (block_b, block_a)
    return (
        "You answer questions using the two reference records below. "
        "They may conflict. Give a concise answer grounded in the records. "
        "Do not invent a trust policy and do not discuss hidden model controls."
        "\n\n" + blocks[0] + "\n\n" + blocks[1]
    )


def render_chat(tokenizer, messages) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_prompt(tokenizer, args, order: str, device):
    messages = [
        {"role": "system", "content": system_message(args.source_a, args.source_b, order)},
        {"role": "user", "content": args.question},
    ]
    rendered = render_chat(tokenizer, messages)
    encoded = tokenizer(rendered, return_offsets_mapping=True, add_special_tokens=False)
    ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    offsets = [tuple(x) for x in encoded["offset_mapping"]]
    spans = locate_source_spans(rendered, offsets, args.source_a, args.source_b)
    return ids, mask, spans


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _new_cache():
    try:
        from transformers import DynamicCache

        return DynamicCache()
    except Exception:  # very old transformers
        return None


@torch.inference_mode()
def score_sequence(
    model,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    cand_ids: Sequence[int],
    *,
    controller=None,
    spans: SourceSpans | None = None,
    trust: float = 0.0,
) -> Tuple[float, List[float]]:
    """Log P(candidate | prompt) under the ACTUAL steered decode process.

    Stepped one token at a time through the KV cache so that the controller
    sees each candidate position as the current position, exactly as it does
    in generate().  A single teacher-forced forward pass would steer only the
    final position and badly under-report the intervention.
    """
    if controller is not None:
        controller.set_trust(trust)
        controller.set_spans(spans)
        controller.begin_generation()

    device = prompt_ids.device
    cache = _new_cache()
    out = model(
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        past_key_values=cache,
        use_cache=True,
    )
    cache = out.past_key_values
    logits = out.logits[:, -1, :]
    mask = prompt_mask

    total = 0.0
    per_token: List[float] = []
    for i, tok in enumerate(cand_ids):
        logp = float(torch.log_softmax(logits.float(), dim=-1)[0, int(tok)])
        total += logp
        per_token.append(logp)
        if i == len(cand_ids) - 1:
            break
        step = torch.tensor([[int(tok)]], dtype=torch.long, device=device)
        mask = torch.cat(
            [mask, torch.ones((1, 1), dtype=mask.dtype, device=device)], dim=1
        )
        out = model(
            input_ids=step,
            attention_mask=mask,
            past_key_values=cache,
            use_cache=True,
        )
        cache = out.past_key_values
        logits = out.logits[:, -1, :]
    return total, per_token


@torch.inference_mode()
def generate_once(model, tokenizer, controller, prompt_ids, prompt_mask, spans, trust, max_new):
    controller.set_trust(trust)
    controller.set_spans(spans)
    controller.begin_generation()
    outputs = model.generate(
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=max_new,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(outputs[0, prompt_ids.shape[1]:], skip_special_tokens=True)
    return text.strip(), controller.summary()


def slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Least-squares slope of y on x."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 1e-12 else float("nan")


def zero_crossing(xs: Sequence[float], ys: Sequence[float]):
    """First m where delta changes sign, linearly interpolated.  None if never."""
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if (y0 > 0) != (y1 > 0):
            if abs(y1 - y0) < 1e-12:
                return xs[i]
            return xs[i] + (xs[i + 1] - xs[i]) * (0.0 - y0) / (y1 - y0)
    return None


def main():
    args = parse_args()
    plan = parse_heads(args.heads)
    sweep = parse_floats(args.trust_sweep)
    gen_at = set(parse_floats(args.generate_at)) if args.generate_at.strip() else set()

    model, tokenizer = load_model(args)
    device = input_device(model)

    n_layers = len(model.model.layers)
    n_qheads = model.config.num_attention_heads
    for h in plan.heads:
        if not (0 <= h.layer < n_layers):
            raise ValueError(f"layer {h.layer} outside 0..{n_layers - 1}")
        if not (0 <= h.query_head < n_qheads):
            raise ValueError(f"query head {h.query_head} outside 0..{n_qheads - 1}")

    cand_a_ids = tokenizer(args.candidate_a, add_special_tokens=False)["input_ids"]
    cand_b_ids = tokenizer(args.candidate_b, add_special_tokens=False)["input_ids"]

    print("FROZEN head plan (not re-selected here):")
    for h in plan.heads:
        print(f"  L{h.layer:02d} Q{h.query_head:02d}")
    print(f"candidate A: {len(cand_a_ids)} tokens   candidate B: {len(cand_b_ids)} tokens")
    print("absolute delta carries a length offset; read the MOVEMENT of delta\n")

    receipt = {
        "model": args.model,
        "frozen_heads": [{"layer": h.layer, "query_head": h.query_head} for h in plan.heads],
        "question": args.question,
        "candidate_a": args.candidate_a,
        "candidate_b": args.candidate_b,
        "candidate_a_tokens": len(cand_a_ids),
        "candidate_b_tokens": len(cand_b_ids),
        "trust_sweep": sweep,
        "arms": {},
        "controls": {},
    }

    prompts = {}
    for order in ("ab", "ba"):
        ids, mask, spans = build_prompt(tokenizer, args, order, device)
        prompts[order] = (ids, mask, spans)
        print(f"[{order}] prompt {ids.shape[1]} tokens, "
              f"span A {spans.source_a}, span B {spans.source_b}")

    # ---- control: unmodified model, no controller installed -------------
    # The wrapper replaces SDPA with eager attention on the patched layers,
    # so trust=0 with the controller installed is NOT the unmodified model.
    # Any difference here is a constant offset sitting under every delta.
    print("\nControl: unmodified model (no controller installed)")
    for order in ("ab", "ba"):
        ids, mask, spans = prompts[order]
        ll_a, _ = score_sequence(model, ids, mask, cand_a_ids)
        ll_b, _ = score_sequence(model, ids, mask, cand_b_ids)
        receipt["controls"][f"unpatched_{order}"] = {
            "ll_a": ll_a, "ll_b": ll_b, "delta": ll_a - ll_b,
        }
        print(f"  [{order}] LL(A)={ll_a:+.4f}  LL(B)={ll_b:+.4f}  delta={ll_a - ll_b:+.4f}")

    controller = QwenObserverController(
        model, plan, trust=0.0,
        target_margin=args.target_margin, max_ratio=args.max_ratio,
    )
    controller.install()

    try:
        for order in ("ab", "ba"):
            ids, mask, spans = prompts[order]
            first_block = "A" if order == "ab" else "B"
            print(f"\n=== arm {order}  (physically first block: SOURCE {first_block}) ===")
            print(f"{'m':>6} {'LL(A)':>11} {'LL(B)':>11} {'delta':>10}  generation")
            rows = []
            for m in sweep:
                ll_a, _ = score_sequence(
                    model, ids, mask, cand_a_ids,
                    controller=controller, spans=spans, trust=m,
                )
                ll_b, _ = score_sequence(
                    model, ids, mask, cand_b_ids,
                    controller=controller, spans=spans, trust=m,
                )
                row = {
                    "trust": m,
                    "ll_a": ll_a,
                    "ll_b": ll_b,
                    "delta": ll_a - ll_b,
                }
                text = ""
                if m in gen_at:
                    text, summary = generate_once(
                        model, tokenizer, controller, ids, mask, spans,
                        m, args.max_new_tokens,
                    )
                    row["generation"] = text
                    row["observer"] = summary
                rows.append(row)
                print(f"{m:>6.2f} {ll_a:>+11.4f} {ll_b:>+11.4f} "
                      f"{ll_a - ll_b:>+10.4f}  {text}")

            xs = [r["trust"] for r in rows]
            ys = [r["delta"] for r in rows]
            receipt["arms"][order] = {
                "first_block": first_block,
                "rows": rows,
                "slope": slope(xs, ys),
                "delta_at_neutral": next(
                    (r["delta"] for r in rows if abs(r["trust"]) < 1e-9), None
                ),
                "zero_crossing": zero_crossing(xs, ys),
            }
    finally:
        controller.uninstall()

    # ---- verdict against the pre-registered rule ------------------------
    s_ab = receipt["arms"]["ab"]["slope"]
    s_ba = receipt["arms"]["ba"]["slope"]
    ratio = abs(s_ba / s_ab) if abs(s_ab) > 1e-12 else float("inf")
    same_sign = (s_ab > 0) == (s_ba > 0)

    if same_sign and 0.5 <= ratio <= 2.0:
        verdict = "IDENTITY_COORDINATE"
    elif (not same_sign) or ratio < 0.25:
        verdict = "SLOT_READING"
    else:
        verdict = "INCONCLUSIVE"

    n_ab = receipt["arms"]["ab"]["delta_at_neutral"]
    n_ba = receipt["arms"]["ba"]["delta_at_neutral"]
    receipt["verdict"] = {
        "rule": verdict,
        "slope_ab": s_ab,
        "slope_ba": s_ba,
        "slope_ratio": ratio,
        "neutral_delta_ab": n_ab,
        "neutral_delta_ba": n_ba,
        "neutral_swing": (n_ab - n_ba) if (n_ab is not None and n_ba is not None) else None,
    }

    print(f"\nslope ab {s_ab:+.4f}   slope ba {s_ba:+.4f}   ratio {ratio:.3f}")
    print(f"neutral delta  ab {n_ab:+.4f}   ba {n_ba:+.4f}")
    print("  (a large neutral swing with content fixed is a primacy prior, "
          "not an observer effect)")
    print(f"VERDICT (pre-registered rule): {verdict}")

    out = Path(args.receipt)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2))
    print(f"\nreceipt: {out}")


if __name__ == "__main__":
    main()
