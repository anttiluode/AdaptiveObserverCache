"""Gate 8: history chooses which measurement is worth buying."""

from __future__ import annotations

import json

from gate6_relational_alias import canonical_key, read_slot
from gate8_active_probing import (
    active_bind,
    build_active_probe_suite,
    current_probe_field,
    exact_random_order_expected_cost,
    shuffled_pair_history,
)
from pretrained_observer import cache_digest


TARGET_MASS = 0.20
TARGET_SHARE = 0.80


def read_success(result) -> bool:
    return (
        result.prediction == result.target_canonical
        and result.target_mass >= TARGET_MASS
        and result.target_share >= TARGET_SHARE
    )


def evaluate_case(base, tokenizer, projection, history):
    before = cache_digest(projection.cache)
    family = projection.case.family
    pair_keys = (
        canonical_key(tokenizer, family.canonical0),
        canonical_key(tokenizer, family.canonical1),
    )
    field = current_probe_field(projection)
    shuffled = shuffled_pair_history(pair_keys, history)

    rows = {}
    active_correct = 0
    shuffled_correct = 0
    active_costs = []
    random_costs = []

    for canonical, persistent_key in zip(
        (family.canonical0, family.canonical1),
        pair_keys,
    ):
        trace = active_bind(
            persistent_key,
            pair_keys,
            history,
            field,
            projection,
        )
        result = read_slot(
            base, projection, canonical, trace.chosen_slot
        )
        ok = read_success(result)
        active_correct += int(ok)
        active_costs.append(trace.cost)

        random_expected = exact_random_order_expected_cost(
            persistent_key,
            history,
            field,
        )
        random_costs.append(random_expected)

        shuffled_trace = active_bind(
            persistent_key,
            pair_keys,
            shuffled,
            field,
            projection,
        )
        shuffled_result = read_slot(
            base,
            projection,
            canonical,
            shuffled_trace.chosen_slot,
        )
        shuffled_ok = read_success(shuffled_result)
        shuffled_correct += int(shuffled_ok)

        rows[canonical] = {
            "persistent_key_tokens": list(persistent_key),
            "corrupted_probe": field.corrupted_probe,
            "active_probes": list(trace.probes),
            "active_cost": trace.cost,
            "active_observations": [
                {
                    "probe": probe,
                    "slot0": response0,
                    "slot1": response1,
                }
                for probe, response0, response1 in trace.observations
            ],
            "chosen_slot": trace.chosen_slot,
            "target_mass": result.target_mass,
            "target_share": result.target_share,
            "prediction": result.prediction,
            "active_pass": ok,
            "random_order_expected_cost": random_expected,
            "shuffled_history_pass": shuffled_ok,
        }

    after = cache_digest(projection.cache)
    return {
        "name": projection.case.name,
        "order": projection.case.order,
        "canonical_pair": [family.canonical0, family.canonical1],
        "current_alias_pair": [family.alias0, family.alias1],
        "corrupted_probe": field.corrupted_probe,
        "reads": rows,
        "dual_success": active_correct == 2,
        "active_accuracy": active_correct / 2,
        "shuffled_history_accuracy": shuffled_correct / 2,
        "mean_active_probe_cost": sum(active_costs) / len(active_costs),
        "max_active_probe_cost": max(active_costs),
        "mean_random_order_expected_cost": sum(random_costs) / len(random_costs),
        "cache_unchanged": before == after,
        "cache_digest": before,
    }


def build_receipt():
    base, tokenizer, projections, history = build_active_probe_suite()
    rows = [
        evaluate_case(base, tokenizer, projection, history)
        for projection in projections
    ]

    values = [
        value
        for row in rows
        for value in row["reads"].values()
    ]
    total_reads = len(values)

    active_accuracy = sum(
        int(value["active_pass"]) for value in values
    ) / total_reads
    shuffled_accuracy = sum(
        int(value["shuffled_history_pass"]) for value in values
    ) / total_reads

    dual_rate = sum(int(row["dual_success"]) for row in rows) / len(rows)
    reversed_rows = [row for row in rows if row["order"] == "10"]
    reversed_dual_rate = sum(
        int(row["dual_success"]) for row in reversed_rows
    ) / len(reversed_rows)

    mean_mass = sum(value["target_mass"] for value in values) / total_reads
    mean_active_cost = sum(
        value["active_cost"] for value in values
    ) / total_reads
    max_active_cost = max(value["active_cost"] for value in values)
    mean_random_cost = sum(
        value["random_order_expected_cost"] for value in values
    ) / total_reads
    cost_ratio = mean_active_cost / mean_random_cost

    integrity_pass = all(row["cache_unchanged"] for row in rows)
    active_pass = (
        active_accuracy >= 0.95
        and dual_rate >= 0.95
        and reversed_dual_rate >= 0.95
        and mean_mass >= 0.80
        and mean_active_cost <= 1.50
        and max_active_cost <= 2
        and mean_random_cost >= 2.80
        and cost_ratio <= 0.55
        and shuffled_accuracy <= 0.10
    )

    return {
        "gate": "8",
        "claim": (
            "use persistent behavioral history to choose informative identity "
            "probes and recover the current trusted source with fewer paid "
            "measurements than matched random-order sensing"
        ),
        "frozen_reader": {
            "model": base.model_id,
            "revision": base.revision,
            "layer": base.layer,
            "head": base.head,
        },
        "probe_economy": {
            "available_probes": 8,
            "informative_historical_probes_per_family": 3,
            "corrupted_informative_probes_per_case": 1,
            "cost_per_probe": 1,
            "active_policy": (
                "query only coordinates where the two historical candidate "
                "fingerprints differ; stop at first non-ambiguous binding"
            ),
            "random_baseline": (
                "same history and stopping rule; exact expectation over all "
                "8! uniformly random probe orders"
            ),
        },
        "results": rows,
        "summary": {
            "families": 6,
            "cases": len(rows),
            "identity_reads": total_reads,
            "active_accuracy": active_accuracy,
            "dual_success_rate": dual_rate,
            "reversed_order_dual_success_rate": reversed_dual_rate,
            "mean_target_mass": mean_mass,
            "mean_active_probe_cost": mean_active_cost,
            "max_active_probe_cost": max_active_cost,
            "mean_random_order_expected_cost": mean_random_cost,
            "active_to_random_cost_ratio": cost_ratio,
            "shuffled_history_accuracy": shuffled_accuracy,
        },
        "checks": {
            "integrity_pass": integrity_pass,
            "active_probe_pass": active_pass,
        },
        "pass": integrity_pass and active_pass,
    }


def main():
    receipt = build_receipt()
    print(json.dumps(receipt, indent=2))
    if not receipt["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
