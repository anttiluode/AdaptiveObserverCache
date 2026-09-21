"""Gate 1: frozen transformer attention, adaptive reader.

Same projected K/V. Same present state. Frozen parameters.
Only one scalar observer state changes the query geometry.
"""

from __future__ import annotations

import json

from frozen_attention_observer import (
    cache_digest,
    gate1_fixture,
    parameter_digest,
)


SCHEDULE = tuple("AAAA" "BBBB" "AAAA" "BBBB")
FIXED_OBSERVERS = (-1.0, 0.0, 1.0)


def truth_for(regime: str) -> int:
    return 1 if regime == "A" else -1


def run_fixed(model, cache, present, observer: float):
    rows = []
    correct = 0
    for step, regime in enumerate(SCHEDULE):
        result = model.read(present, cache, observer)
        truth = truth_for(regime)
        ok = result.prediction == truth
        correct += int(ok)
        rows.append(
            {
                "step": step,
                "regime": regime,
                "observer": observer,
                "selected_provenance": result.selected_provenance,
                "prediction": result.prediction,
                "correct": ok,
            }
        )
    return {
        "observer": observer,
        "accuracy": correct / len(SCHEDULE),
        "rows": rows,
    }


def run_adaptive(model, cache, present):
    observer = 1.0
    rows = []
    correct = 0
    errors_on_switch_steps = 0
    previous_regime = SCHEDULE[0]

    for step, regime in enumerate(SCHEDULE):
        truth = truth_for(regime)

        # The read happens before truth is supplied to the updater.
        result = model.read(present, cache, observer)
        ok = result.prediction == truth
        correct += int(ok)

        if step > 0 and regime != previous_regime and not ok:
            errors_on_switch_steps += 1

        next_observer = model.update_observer(observer, result, truth)
        rows.append(
            {
                "step": step,
                "regime": regime,
                "truth": truth,
                "observer_before": observer,
                "query": [float(x) for x in result.query],
                "mass_A": result.mass_a,
                "mass_B": result.mass_b,
                "selected_provenance": result.selected_provenance,
                "prediction": result.prediction,
                "correct": ok,
                "observer_after": next_observer,
            }
        )
        observer = next_observer
        previous_regime = regime

    return {
        "accuracy": correct / len(SCHEDULE),
        "errors_on_switch_steps": errors_on_switch_steps,
        "rows": rows,
    }


def build_receipt():
    model, cache, present = gate1_fixture()

    cache_before = cache_digest(cache)
    parameters_before = parameter_digest(model)

    paired_a = model.read(present, cache, +1.0)
    paired_b = model.read(present, cache, -1.0)

    fixed = [
        run_fixed(model, cache, present, observer)
        for observer in FIXED_OBSERVERS
    ]
    best_fixed = max(item["accuracy"] for item in fixed)
    adaptive = run_adaptive(model, cache, present)

    cache_after = cache_digest(cache)
    parameters_after = parameter_digest(model)

    paired_pass = (
        paired_a.selected_provenance == "A"
        and paired_a.prediction == 1
        and paired_b.selected_provenance == "B"
        and paired_b.prediction == -1
    )
    integrity_pass = (
        cache_before == cache_after
        and parameters_before == parameters_after
        and all(not p.requires_grad for p in model.parameters())
    )
    adaptive_advantage = adaptive["accuracy"] - best_fixed

    gate_pass = (
        paired_pass
        and integrity_pass
        and adaptive["accuracy"] >= 0.75
        and adaptive_advantage >= 0.20
        and adaptive["errors_on_switch_steps"] <= 3
    )

    return {
        "gate": "1",
        "claim": (
            "a persistent scalar observer can modulate only the query of a "
            "frozen attention layer, changing reads from identical projected "
            "K/V and identical present state; feedback changes the next query "
            "without cache or parameter mutation"
        ),
        "architecture": {
            "equation": "q' = W_Q h + strength * m * u",
            "observer_rank": 1,
            "observer_strength": model.observer_strength,
            "read_budget_per_step": 1,
            "schedule": "".join(SCHEDULE),
            "fixed_observer_grid": list(FIXED_OBSERVERS),
        },
        "paired_same_cache_same_present": {
            "observer_plus_1": {
                "selected_provenance": paired_a.selected_provenance,
                "prediction": paired_a.prediction,
                "mass_A": paired_a.mass_a,
                "mass_B": paired_a.mass_b,
            },
            "observer_minus_1": {
                "selected_provenance": paired_b.selected_provenance,
                "prediction": paired_b.prediction,
                "mass_A": paired_b.mass_a,
                "mass_B": paired_b.mass_b,
            },
            "pass": paired_pass,
        },
        "fixed_readers": fixed,
        "best_fixed_accuracy": best_fixed,
        "adaptive": adaptive,
        "adaptive_advantage": adaptive_advantage,
        "integrity": {
            "cache_digest_before": cache_before,
            "cache_digest_after": cache_after,
            "parameter_digest_before": parameters_before,
            "parameter_digest_after": parameters_after,
            "all_parameters_frozen": all(
                not p.requires_grad for p in model.parameters()
            ),
            "pass": integrity_pass,
        },
        "pass": gate_pass,
    }


def main():
    receipt = build_receipt()
    print(json.dumps(receipt, indent=2))
    if not receipt["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
