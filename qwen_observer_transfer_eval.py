"""Held-out transfer, dose-response and ablation evaluation for Qwen AOC.

This script deliberately does NOT re-run causal head selection. It freezes the
head identities written by the prior Qwen causal receipt and asks whether that
observer transfers to unrelated source conflicts.

Primary questions:
1. does the frozen head set transfer beyond valve-vs-sensor semantics?
2. can source order change without changing the meaning of observer A/B?
3. does trust behave like a graded control, not only a +/-1 switch?
4. is the result carried by one indispensable head?

Teacher-forced candidate scoring uses the complete candidate token sequence.
The controller steers only answer-side query positions during that scoring;
historical source K/V rows remain immutable and are checked byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qwen_observer import (
    HeadSelection,
    ObserverPlan,
    QwenObserverController,
    locate_source_spans,
)
from qwen_observer_chat import (
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_QUESTION,
    encode_rendered,
    input_device,
    load_model,
    render_chat,
)


HELD_OUT_CASES = (
    {
        "name": "telescope",
        "order": "AB",
        "source_a": (
            "The telescope image blur was caused by a warped secondary mirror. "
            "The star-test rings support this diagnosis."
        ),
        "source_b": (
            "The telescope image blur was caused by tracking-motor jitter. "
            "The encoder residuals support this diagnosis."
        ),
        "question": "Why was the telescope image blurred? Give the cause in one short sentence.",
        "candidate_a": "The telescope image was blurred by a warped secondary mirror.",
        "candidate_b": "The telescope image was blurred by tracking-motor jitter.",
        "anchor_a": "mirror",
        "anchor_b": "jitter",
    },
    {
        "name": "database",
        "order": "BA",
        "source_a": (
            "The service outage was caused by corruption in the primary index file. "
            "The storage checksum report supports this diagnosis."
        ),
        "source_b": (
            "The service outage was caused by an expired authentication certificate. "
            "The handshake log supports this diagnosis."
        ),
        "question": "Why did the database service go offline? Give the cause in one short sentence.",
        "candidate_a": "The service went offline because the primary index file was corrupted.",
        "candidate_b": "The service went offline because the authentication certificate expired.",
        "anchor_a": "index",
        "anchor_b": "certificate",
    },
    {
        "name": "reactor",
        "order": "AB",
        "source_a": (
            "The reaction stopped because the catalyst bed was poisoned. "
            "The spectroscopy report supports this diagnosis."
        ),
        "source_b": (
            "The reaction stopped because the feed pump was cavitating. "
            "The inlet-pressure trace supports this diagnosis."
        ),
        "question": "Why did the chemical reaction stop? Give the cause in one short sentence.",
        "candidate_a": "The reaction stopped because the catalyst bed was poisoned.",
        "candidate_b": "The reaction stopped because the feed pump was cavitating.",
        "anchor_a": "catalyst",
        "anchor_b": "pump",
    },
    {
        "name": "compiler",
        "order": "BA",
        "source_a": (
            "The build failed because a parser regression rejected a valid token sequence. "
            "The parse trace supports this diagnosis."
        ),
        "source_b": (
            "The build failed because the generated schema was stale. "
            "The dependency manifest supports this diagnosis."
        ),
        "question": "Why did the software build fail? Give the cause in one short sentence.",
        "candidate_a": "The build failed because of a parser regression.",
        "candidate_b": "The build failed because the generated schema was stale.",
        "anchor_a": "parser",
        "anchor_b": "schema",
    },
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate the frozen Qwen AOC head set on held-out conflicts."
    )
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument(
        "--plan-receipt",
        default="results/qwen_observer_causal_compare.json",
    )
    p.add_argument(
        "--trust-grid",
        default="-1,-0.5,0,0.5,1",
        help="comma-separated observer values for the dose-response sweep",
    )
    p.add_argument("--target-margin", type=float, default=3.0)
    p.add_argument("--max-ratio", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--gpu-memory", default="6GiB")
    p.add_argument("--cpu-memory", default="6GiB")
    p.add_argument("--offload-dir", default=".offload_qwen_observer")
    p.add_argument(
        "--receipt",
        default="results/qwen_observer_transfer_eval.json",
    )
    p.add_argument(
        "--skip-ablation",
        action="store_true",
        help="skip the leave-one-head-out diagnostic",
    )
    return p.parse_args()


def load_frozen_plan(path: Path) -> ObserverPlan:
    payload = json.loads(path.read_text())
    rows = payload.get("selected_heads") or []
    if not rows:
        raise RuntimeError(
            f"{path} contains no selected_heads; run the causal comparison first"
        )
    heads = tuple(
        HeadSelection(
            layer=int(row["layer"]),
            query_head=int(row["query_head"]),
            kv_head=int(row["kv_head"]),
            symmetric_source_mass=float(row["symmetric_source_mass"]),
            mass_when_a=float(row["mass_when_a"]),
            mass_when_b=float(row["mass_when_b"]),
            ratio_when_a=float(row["ratio_when_a"]),
            ratio_when_b=float(row["ratio_when_b"]),
        )
        for row in rows
    )
    return ObserverPlan(heads=heads)


def ordered_system(case: dict) -> str:
    intro = (
        "You answer questions using the two reference records below. "
        "They may conflict. Give a concise answer grounded in the records. "
        "Do not invent a trust policy and do not discuss hidden model controls.\n\n"
    )
    blocks = {
        "A": "[BEGIN SOURCE A]\n" + case["source_a"] + "\n[END SOURCE A]",
        "B": "[BEGIN SOURCE B]\n" + case["source_b"] + "\n[END SOURCE B]",
    }
    return intro + "\n\n".join(blocks[x] for x in case["order"])


def case_messages(case: dict):
    return [
        {"role": "system", "content": ordered_system(case)},
        {"role": "user", "content": case["question"]},
    ]


def encode_case(tokenizer, case: dict):
    rendered = render_chat(tokenizer, case_messages(case))
    ids, mask, offsets = encode_rendered(tokenizer, rendered)
    spans = locate_source_spans(
        rendered,
        offsets,
        case["source_a"],
        case["source_b"],
    )
    return rendered, ids, mask, spans


def classify_answer(text: str, case: dict) -> str:
    low = text.lower()
    has_a = case["anchor_a"].lower() in low
    has_b = case["anchor_b"].lower() in low
    if has_a and not has_b:
        return "A"
    if has_b and not has_a:
        return "B"
    if has_a and has_b:
        return "both"
    return "other"


@torch.inference_mode()
def generate_case(model, tokenizer, controller, case: dict, args, trust: float):
    _rendered, ids, mask, spans = encode_case(tokenizer, case)
    controller.set_query_start(None)
    controller.set_trust(trust)
    controller.set_spans(spans)
    controller.begin_generation()

    device = input_device(model)
    ids = ids.to(device)
    mask = mask.to(device)
    prompt_len = ids.shape[1]

    outputs = model.generate(
        input_ids=ids,
        attention_mask=mask,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(
        outputs[0, prompt_len:], skip_special_tokens=True
    ).strip()
    return {
        "trust": float(trust),
        "answer": text,
        "class": classify_answer(text, case),
        "observer": controller.summary(),
    }


@torch.inference_mode()
def score_candidate(
    model,
    tokenizer,
    controller,
    case: dict,
    candidate: str,
    trust: float,
):
    _rendered, prompt_ids, prompt_mask, spans = encode_case(tokenizer, case)
    candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
    if not candidate_ids:
        raise RuntimeError(f"empty candidate tokenization: {candidate!r}")

    device = input_device(model)
    prompt_ids = prompt_ids.to(device)
    prompt_mask = prompt_mask.to(device)
    cand = torch.tensor([candidate_ids], dtype=prompt_ids.dtype, device=device)
    full_ids = torch.cat([prompt_ids, cand], dim=1)
    full_mask = torch.cat(
        [
            prompt_mask,
            torch.ones(
                (1, len(candidate_ids)),
                dtype=prompt_mask.dtype,
                device=device,
            ),
        ],
        dim=1,
    )

    prompt_len = prompt_ids.shape[1]
    controller.set_trust(trust)
    controller.set_spans(spans)
    controller.set_query_start(prompt_len - 1)
    controller.begin_generation()
    try:
        out = model(
            input_ids=full_ids,
            attention_mask=full_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = out.logits[
            0,
            prompt_len - 1 : prompt_len - 1 + len(candidate_ids),
            :,
        ].float()
        targets = torch.tensor(candidate_ids, device=logits.device)
        logp = torch.log_softmax(logits, dim=-1)
        token_logps = logp.gather(1, targets[:, None]).squeeze(1)
        values = [float(x) for x in token_logps.detach().cpu()]
        return {
            "text": candidate,
            "token_ids": candidate_ids,
            "token_count": len(candidate_ids),
            "sum_logprob": float(token_logps.sum()),
            "mean_logprob": float(token_logps.mean()),
            "token_logprobs": values,
            "observer": controller.summary(),
        }
    finally:
        controller.set_query_start(None)


def sequence_margin(model, tokenizer, controller, case: dict, trust: float):
    a = score_candidate(
        model, tokenizer, controller, case, case["candidate_a"], trust
    )
    b = score_candidate(
        model, tokenizer, controller, case, case["candidate_b"], trust
    )
    return {
        "trust": float(trust),
        "A": a,
        "B": b,
        "mean_logprob_margin_A_minus_B": (
            a["mean_logprob"] - b["mean_logprob"]
        ),
        "sum_logprob_margin_A_minus_B": (
            a["sum_logprob"] - b["sum_logprob"]
        ),
    }


def plan_dict(plan: ObserverPlan):
    return [
        {
            "layer": h.layer,
            "query_head": h.query_head,
            "kv_head": h.kv_head,
            "symmetric_source_mass": h.symmetric_source_mass,
            "mass_when_a": h.mass_when_a,
            "mass_when_b": h.mass_when_b,
            "ratio_when_a": h.ratio_when_a,
            "ratio_when_b": h.ratio_when_b,
        }
        for h in plan.heads
    ]


def main():
    args = parse_args()
    trust_grid = [
        float(x.strip())
        for x in args.trust_grid.split(",")
        if x.strip()
    ]
    if -1.0 not in trust_grid or 0.0 not in trust_grid or 1.0 not in trust_grid:
        raise ValueError("trust grid must contain -1, 0 and +1")

    plan_path = Path(args.plan_receipt)
    frozen_plan = load_frozen_plan(plan_path)
    print("Frozen observer plan (NO reselection):")
    for h in frozen_plan.heads:
        print(
            f"  L{h.layer:02d} Q{h.query_head:02d}/KV{h.kv_head} "
            f"sym_mass={h.symmetric_source_mass:.3f}"
        )

    model, tokenizer = load_model(args)
    controller = QwenObserverController(
        model,
        frozen_plan,
        target_margin=args.target_margin,
        max_ratio=args.max_ratio,
    )
    controller.install()

    receipt = {
        "scope": (
            "Frozen-head held-out transfer evaluation after the first Qwen "
            "language-level AOC switch."
        ),
        "model": args.model,
        "plan_receipt": str(plan_path),
        "frozen_heads": plan_dict(frozen_plan),
        "target_margin": args.target_margin,
        "max_ratio": args.max_ratio,
        "trust_grid": trust_grid,
        "held_out": [],
        "ablation": [],
    }

    try:
        print("\n=== Held-out semantic transfer: generation ===")
        for case in HELD_OUT_CASES:
            row = {
                "name": case["name"],
                "order": case["order"],
                "generation": [],
                "dose_response": [],
            }
            print(f"\n[{case['name']}] source order={case['order']}")
            for label, trust in (("A", +1.0), ("neutral", 0.0), ("B", -1.0)):
                run = generate_case(
                    model, tokenizer, controller, case, args, trust
                )
                row["generation"].append({"mode": label, **run})
                print(
                    f"  {label:7s} -> {run['class']:5s} | "
                    f"{run['answer']}"
                )

            print("  dose response:", flush=True)
            for trust in trust_grid:
                score = sequence_margin(
                    model, tokenizer, controller, case, trust
                )
                row["dose_response"].append(score)
                print(
                    f"    m={trust:+.2f}  "
                    f"mean dlogp(A-B)="
                    f"{score['mean_logprob_margin_A_minus_B']:+.4f}"
                )
            receipt["held_out"].append(row)

        if not args.skip_ablation:
            print("\n=== Leave-one-head-out diagnostic on original conflict ===")
            original = {
                "name": "original_valve_sensor",
                "order": "AB",
                "source_a": DEFAULT_A,
                "source_b": DEFAULT_B,
                "question": DEFAULT_QUESTION,
                "candidate_a": (
                    "The device failed because valve C was obstructed."
                ),
                "candidate_b": (
                    "The device failed because sensor K drifted."
                ),
                "anchor_a": "valve",
                "anchor_b": "sensor",
            }
            variants = [("full", frozen_plan)]
            for removed in frozen_plan.heads:
                kept = tuple(h for h in frozen_plan.heads if h != removed)
                variants.append(
                    (
                        f"without_L{removed.layer}_Q{removed.query_head}",
                        ObserverPlan(heads=kept),
                    )
                )

            controller.uninstall()
            for name, plan in variants:
                local = QwenObserverController(
                    model,
                    plan,
                    target_margin=args.target_margin,
                    max_ratio=args.max_ratio,
                )
                local.install()
                try:
                    a = sequence_margin(
                        model, tokenizer, local, original, +1.0
                    )
                    b = sequence_margin(
                        model, tokenizer, local, original, -1.0
                    )
                    row = {
                        "variant": name,
                        "heads": plan_dict(plan),
                        "A_trust": a,
                        "B_trust": b,
                    }
                    receipt["ablation"].append(row)
                    print(
                        f"  {name:24s} "
                        f"Atrust={a['mean_logprob_margin_A_minus_B']:+.4f} "
                        f"Btrust={b['mean_logprob_margin_A_minus_B']:+.4f}"
                    )
                finally:
                    local.uninstall()
            controller.install()

    finally:
        controller.uninstall()

    dual = 0
    reversed_dual = 0
    endpoint_ordered = 0
    integrity_ok = True
    no_caps = True

    for row in receipt["held_out"]:
        by_mode = {x["mode"]: x for x in row["generation"]}
        is_dual = (
            by_mode["A"]["class"] == "A"
            and by_mode["B"]["class"] == "B"
        )
        dual += int(is_dual)
        if row["order"] == "BA":
            reversed_dual += int(is_dual)

        margins = {
            x["trust"]: x["mean_logprob_margin_A_minus_B"]
            for x in row["dose_response"]
        }
        endpoint_ordered += int(
            margins[+1.0] > margins[0.0] > margins[-1.0]
        )

        for gen in row["generation"]:
            s = gen["observer"]
            integrity_ok &= bool(s.get("cache_integrity_ok", False))
            no_caps &= s.get("capped_fraction", 0.0) == 0.0
        for score in row["dose_response"]:
            for side in ("A", "B"):
                s = score[side]["observer"]
                integrity_ok &= bool(s.get("cache_integrity_ok", False))
                no_caps &= s.get("capped_fraction", 0.0) == 0.0

    strong_transfer = dual >= 3
    order_pass = reversed_dual == 2
    graded_pass = endpoint_ordered >= 3
    overall = (
        strong_transfer
        and order_pass
        and graded_pass
        and integrity_ok
        and no_caps
    )

    receipt["summary"] = {
        "held_out_cases": len(receipt["held_out"]),
        "dual_switch_cases": dual,
        "reversed_order_dual_switch_cases": reversed_dual,
        "endpoint_ordered_dose_response_cases": endpoint_ordered,
        "cache_integrity_all": integrity_ok,
        "no_capped_updates": no_caps,
        "strong_transfer_pass": strong_transfer,
        "reversed_order_pass": order_pass,
        "graded_control_pass": graded_pass,
        "overall_pass": overall,
        "frozen_before_run_contract": {
            "strong_transfer": ">=3/4 held-out cases dual-switch",
            "reversed_order": "2/2 BA-order cases dual-switch",
            "graded_control": (
                "margin(+1) > margin(0) > margin(-1) in >=3/4 cases"
            ),
            "integrity": "all source-cache checks true",
            "boundedness": "no query update hits the norm cap",
        },
    }

    out = Path(args.receipt)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2))
    print("\n=== Frozen contract summary ===")
    print(json.dumps(receipt["summary"], indent=2))
    print(f"\nreceipt: {out}")


if __name__ == "__main__":
    main()
