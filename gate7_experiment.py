"""Gate 7: infer provenance continuity from behavioral history."""

from __future__ import annotations

import json

from gate6_relational_alias import canonical_key, read_slot
from gate7_history_continuity import (
    HistoryMemory,
    bind_from_history,
    build_history_suite,
    incomplete_relation_bind,
    observe_current_behavior,
    static_tie_slot,
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


def swapped_history_for_pair(history, tokenizer, projection) -> HistoryMemory:
    family = projection.case.family
    k0 = canonical_key(tokenizer, family.canonical0)
    k1 = canonical_key(tokenizer, family.canonical1)
    copied = dict(history.fingerprints)
    copied[k0], copied[k1] = copied[k1], copied[k0]
    return HistoryMemory(fingerprints=copied)


def evaluate_case(base, tokenizer, projection, history):
    before = cache_digest(projection.cache)
    family = projection.case.family
    observations = observe_current_behavior(projection)
    shuffled_history = swapped_history_for_pair(
        history, tokenizer, projection
    )

    rows = {}
    history_correct = 0
    static_correct = 0
    one_probe_correct = 0
    reset_correct = 0
    shuffled_correct = 0
    incomplete_graph_correct = 0
    margins = []

    for canonical in (family.canonical0, family.canonical1):
        state_key = canonical_key(tokenizer, canonical)

        binding = bind_from_history(
            state_key, history, observations, probe_count=5
        )
        bound = read_slot(
            base, projection, canonical, binding.chosen_slot
        )
        bound_ok = read_success(bound)
        history_correct += int(bound_ok)
        margins.append(binding.margin)

        # Static position assumes canonical0->slot0/canonical1->slot1.
        static_slot = 0 if canonical == family.canonical0 else 1
        static = read_slot(base, projection, canonical, static_slot)
        static_ok = read_success(static)
        static_correct += int(static_ok)

        # One probe is deliberately uninformative. Ambiguity falls back to slot0.
        try:
            one_binding = bind_from_history(
                state_key, history, observations, probe_count=1
            )
            one_slot = one_binding.chosen_slot
        except RuntimeError:
            one_slot = static_tie_slot(observations)
        one = read_slot(base, projection, canonical, one_slot)
        one_ok = read_success(one)
        one_probe_correct += int(one_ok)

        # Reset-history control sees current behavior but forgot the prior source
        # fingerprint, so it has no relation between trust and current slots.
        reset_slot = static_tie_slot(observations)
        reset = read_slot(base, projection, canonical, reset_slot)
        reset_ok = read_success(reset)
        reset_correct += int(reset_ok)

        # Same current probe evidence, but prior source histories are swapped.
        shuffled_binding = bind_from_history(
            state_key,
            shuffled_history,
            observations,
            probe_count=5,
        )
        shuffled = read_slot(
            base,
            projection,
            canonical,
            shuffled_binding.chosen_slot,
        )
        shuffled_ok = read_success(shuffled)
        shuffled_correct += int(shuffled_ok)

        graph_slot = incomplete_relation_bind(
            tokenizer, projection, state_key
        )
        graph_ok = False
        if graph_slot is not None:
            graph_ok = read_success(
                read_slot(base, projection, canonical, graph_slot)
            )
        incomplete_graph_correct += int(graph_ok)

        rows[canonical] = {
            "persistent_key_tokens": list(state_key),
            "historical_fingerprint": list(
                history.fingerprints[state_key]
            ),
            "current_slot_observations": [
                {
                    "slot": obs.slot,
                    "fingerprint": list(obs.fingerprint),
                }
                for obs in observations
            ],
            "chosen_slot": binding.chosen_slot,
            "best_distance": binding.best_distance,
            "second_distance": binding.second_distance,
            "distance_margin": binding.margin,
            "target_mass": bound.target_mass,
            "target_share": bound.target_share,
            "prediction": bound.prediction,
            "history_bound_pass": bound_ok,
            "static_slot_pass": static_ok,
            "one_probe_pass": one_ok,
            "reset_history_pass": reset_ok,
            "shuffled_history_pass": shuffled_ok,
            "incomplete_graph_slot": graph_slot,
            "incomplete_graph_pass": graph_ok,
        }

    after = cache_digest(projection.cache)
    return {
        "name": projection.case.name,
        "order": projection.case.order,
        "canonical_pair": [family.canonical0, family.canonical1],
        "current_alias_pair": [family.alias0, family.alias1],
        "reads": rows,
        "dual_success": history_correct == 2,
        "history_accuracy": history_correct / 2,
        "static_accuracy": static_correct / 2,
        "one_probe_accuracy": one_probe_correct / 2,
        "reset_history_accuracy": reset_correct / 2,
        "shuffled_history_accuracy": shuffled_correct / 2,
        "incomplete_graph_accuracy": incomplete_graph_correct / 2,
        "minimum_distance_margin": min(margins),
        "cache_unchanged": before == after,
        "cache_digest": before,
    }


def build_receipt():
    base, tokenizer, projections, history = build_history_suite()
    rows = [
        evaluate_case(base, tokenizer, projection, history)
        for projection in projections
    ]

    total_reads = 2 * len(rows)
    values = [
        value
        for row in rows
        for value in row["reads"].values()
    ]

    def accuracy(field):
        return sum(int(value[field]) for value in values) / total_reads

    history_accuracy = accuracy("history_bound_pass")
    static_accuracy = accuracy("static_slot_pass")
    one_probe_accuracy = accuracy("one_probe_pass")
    reset_accuracy = accuracy("reset_history_pass")
    shuffled_accuracy = accuracy("shuffled_history_pass")
    graph_accuracy = accuracy("incomplete_graph_pass")

    dual_rate = sum(int(row["dual_success"]) for row in rows) / len(rows)
    reversed_rows = [row for row in rows if row["order"] == "10"]
    reversed_dual_rate = sum(
        int(row["dual_success"]) for row in reversed_rows
    ) / len(reversed_rows)

    mean_target_mass = sum(
        value["target_mass"] for value in values
    ) / total_reads
    minimum_margin = min(
        row["minimum_distance_margin"] for row in rows
    )

    integrity_pass = all(row["cache_unchanged"] for row in rows)
    history_pass = (
        len(history.fingerprints) >= 12
        and history_accuracy >= 0.95
        and dual_rate >= 0.95
        and reversed_dual_rate >= 0.95
        and mean_target_mass >= 0.80
        and minimum_margin >= 1
        and static_accuracy <= 0.50
        and one_probe_accuracy <= 0.50
        and reset_accuracy <= 0.50
        and shuffled_accuracy <= 0.10
        and graph_accuracy <= 0.10
    )

    return {
        "gate": "7",
        "claim": (
            "infer which current anonymous source continues a persistent "
            "trusted provenance identity from prior behavioral observations "
            "when the supplied alias graph no longer reaches current records"
        ),
        "frozen_reader": {
            "model": base.model_id,
            "revision": base.revision,
            "layer": base.layer,
            "head": base.head,
            "first_slot_mode": base.mode_a.state,
            "second_slot_mode": base.mode_b.state,
        },
        "continuity_memory": {
            "canonical_sources": len(history.fingerprints),
            "historical_probe_count": 5,
            "current_probe_count": 5,
            "current_probe_corruptions_per_source": 1,
            "matching_rule": "minimum Hamming distance",
            "trust_used_to_generate_probe_evidence": False,
        },
        "thresholds": {
            "target_absolute_mass": TARGET_MASS,
            "target_source_share": TARGET_SHARE,
        },
        "results": rows,
        "summary": {
            "families": 6,
            "cases": len(rows),
            "identity_reads": total_reads,
            "historical_sources": len(history.fingerprints),
            "history_bound_accuracy": history_accuracy,
            "dual_success_rate": dual_rate,
            "reversed_order_dual_success_rate": reversed_dual_rate,
            "mean_target_mass": mean_target_mass,
            "minimum_history_match_margin": minimum_margin,
            "static_slot_accuracy": static_accuracy,
            "one_probe_accuracy": one_probe_accuracy,
            "reset_history_accuracy": reset_accuracy,
            "shuffled_history_accuracy": shuffled_accuracy,
            "incomplete_relation_graph_accuracy": graph_accuracy,
        },
        "checks": {
            "integrity_pass": integrity_pass,
            "history_continuity_pass": history_pass,
        },
        "pass": integrity_pass and history_pass,
    }


def main():
    receipt = build_receipt()
    print(json.dumps(receipt, indent=2))
    if not receipt["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
