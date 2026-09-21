"""Deterministic Gate-0 receipt for AdaptiveObserverCache."""

from __future__ import annotations

import json

from adaptive_observer_cache import ObserverState, gate0_cache


PRESENT = (1.0, 1.0)
A_STATE = ObserverState(+2.0, -2.0)
B_STATE = ObserverState(-2.0, +2.0)
SCHEDULE = tuple("AAAA" "BBBB" "AAAA" "BBBB")


def paired_observer_receipt():
    cache = gate0_cache()
    a = cache.read(PRESENT, A_STATE)
    b = cache.read(PRESENT, B_STATE)
    return {
        "same_cache": True,
        "same_present": list(PRESENT),
        "observer_A": {
            "query": list(a.query),
            "prediction": a.prediction,
            "selected_provenance": a.selected_provenance,
            "mass_A": a.mass_a,
            "mass_B": a.mass_b,
        },
        "observer_B": {
            "query": list(b.query),
            "prediction": b.prediction,
            "selected_provenance": b.selected_provenance,
            "mass_A": b.mass_a,
            "mass_B": b.mass_b,
        },
        "pass": (
            a.selected_provenance == "A"
            and a.prediction == 1
            and b.selected_provenance == "B"
            and b.prediction == -1
        ),
    }


def switching_world_receipt():
    cache = gate0_cache()
    before = cache.checksum()
    adaptive = A_STATE
    static = A_STATE
    rows = []
    adaptive_correct = 0
    static_correct = 0
    switch_errors = 0

    previous_regime = SCHEDULE[0]

    for step, regime in enumerate(SCHEDULE):
        truth = 1 if regime == "A" else -1
        adaptive_read = cache.read(PRESENT, adaptive)
        static_read = cache.read(PRESENT, static)

        adaptive_ok = adaptive_read.prediction == truth
        static_ok = static_read.prediction == truth
        adaptive_correct += int(adaptive_ok)
        static_correct += int(static_ok)

        if step > 0 and regime != previous_regime and not adaptive_ok:
            switch_errors += 1

        next_adaptive = cache.update(adaptive, adaptive_read, truth)

        rows.append(
            {
                "step": step,
                "trusted_provenance": regime,
                "truth": truth,
                "observer_before": [
                    adaptive.log_gain_a,
                    adaptive.log_gain_b,
                ],
                "selected_provenance": adaptive_read.selected_provenance,
                "prediction": adaptive_read.prediction,
                "correct": adaptive_ok,
                "mass_A": adaptive_read.mass_a,
                "mass_B": adaptive_read.mass_b,
                "observer_after": [
                    next_adaptive.log_gain_a,
                    next_adaptive.log_gain_b,
                ],
                "static_prediction": static_read.prediction,
                "static_correct": static_ok,
            }
        )

        adaptive = next_adaptive
        previous_regime = regime

    after = cache.checksum()
    adaptive_accuracy = adaptive_correct / len(SCHEDULE)
    static_accuracy = static_correct / len(SCHEDULE)

    return {
        "schedule": "".join(SCHEDULE),
        "steps": len(SCHEDULE),
        "read_budget_per_step": 1,
        "adaptive_accuracy": adaptive_accuracy,
        "static_accuracy": static_accuracy,
        "adaptive_advantage": adaptive_accuracy - static_accuracy,
        "switches": 3,
        "switches_with_one_initial_contradiction": switch_errors,
        "cache_unchanged": before == after,
        "rows": rows,
        "pass": (
            adaptive_accuracy >= 0.75
            and adaptive_accuracy - static_accuracy >= 0.20
            and switch_errors <= 3
            and before == after
        ),
    }


def main():
    paired = paired_observer_receipt()
    switching = switching_world_receipt()
    receipt = {
        "gate": "0",
        "claim": (
            "same fixed memory can reveal different information under different "
            "persistent observer states, and contradictory evidence can change "
            "future reads without rewriting memory"
        ),
        "paired_observer": paired,
        "switching_world": switching,
        "pass": paired["pass"] and switching["pass"],
    }
    print(json.dumps(receipt, indent=2))
    if not receipt["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
