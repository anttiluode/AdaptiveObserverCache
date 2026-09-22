"""Gate 6: persistent provenance survives explicit alias/representation drift."""

from __future__ import annotations

import argparse
import json

from gate5_generic_binding import capture_persistent_key
from gate6_alias_binding import (
    bind_alias_key,
    build_alias_registry,
    build_alias_suite,
    exact_key_slot,
    read_slot,
)
from pretrained_observer import cache_digest


TARGET_MASS = 0.20
TARGET_SHARE = 0.80


def read_success(result) -> bool:
    return (
        result.prediction == result.target_identity
        and result.target_mass >= TARGET_MASS
        and result.target_share >= TARGET_SHARE
    )


def evaluate_case(base, tokenizer, projection):
    before = cache_digest(projection.cache)
    family = projection.case.family
    generation = projection.case.generation

    correct_registry = build_alias_registry(
        tokenizer, family, generation, wrong=False
    )
    wrong_registry = build_alias_registry(
        tokenizer, family, generation, wrong=True
    )
    one_hop_registry = build_alias_registry(
        tokenizer,
        family,
        generation,
        wrong=False,
        truncate_to_one_hop=True,
    )

    rows = {}
    alias_correct = 0
    static_correct = 0
    wrong_correct = 0
    exact_correct = 0
    one_hop_correct = 0

    for target_index, target in enumerate((family.key0, family.key1)):
        persistent_key = capture_persistent_key(tokenizer, target)

        slot = bind_alias_key(
            persistent_key, projection, correct_registry
        )
        bound = read_slot(base, projection, target, slot)

        static = read_slot(
            base, projection, target, target_index
        )

        wrong_slot = bind_alias_key(
            persistent_key, projection, wrong_registry
        )
        wrong = read_slot(base, projection, target, wrong_slot)

        exact_slot = exact_key_slot(persistent_key, projection)
        exact_ok = False
        if exact_slot is not None:
            exact_ok = read_success(
                read_slot(base, projection, target, exact_slot)
            )

        one_hop_ok = False
        try:
            one_hop_slot = bind_alias_key(
                persistent_key, projection, one_hop_registry
            )
            one_hop_ok = read_success(
                read_slot(base, projection, target, one_hop_slot)
            )
        except RuntimeError:
            one_hop_slot = None

        bound_ok = read_success(bound)
        static_ok = read_success(static)
        wrong_ok = read_success(wrong)

        alias_correct += int(bound_ok)
        static_correct += int(static_ok)
        wrong_correct += int(wrong_ok)
        exact_correct += int(exact_ok)
        one_hop_correct += int(one_hop_ok)

        rows[target] = {
            "persistent_original_key": list(persistent_key),
            "current_record_key": list(
                projection.record_keys[slot]
            ),
            "resolved_slot": slot,
            "observer": bound.observer,
            "target_mass": bound.target_mass,
            "target_share": bound.target_share,
            "prediction": bound.prediction,
            "alias_bound_pass": bound_ok,
            "static_slot_pass": static_ok,
            "wrong_alias_map_pass": wrong_ok,
            "exact_key_match_available": exact_slot is not None,
            "exact_key_pass": exact_ok,
            "one_hop_only_slot": one_hop_slot,
            "one_hop_only_pass": one_hop_ok,
        }

    after = cache_digest(projection.cache)
    return {
        "name": projection.case.name,
        "generation": generation,
        "order": projection.case.order,
        "provenance_pair": [family.key0, family.key1],
        "prompt": projection.prompt,
        "reads": rows,
        "dual_success": alias_correct == 2,
        "alias_accuracy": alias_correct / 2,
        "static_accuracy": static_correct / 2,
        "wrong_alias_accuracy": wrong_correct / 2,
        "exact_key_accuracy": exact_correct / 2,
        "one_hop_accuracy": one_hop_correct / 2,
        "cache_unchanged": before == after,
        "cache_digest": before,
    }


def build_receipt():
    base, tokenizer, projections = build_alias_suite()
    rows = [evaluate_case(base, tokenizer, p) for p in projections]

    total_reads = 2 * len(rows)
    all_values = [
        value for row in rows for value in row["reads"].values()
    ]

    alias_correct = sum(int(v["alias_bound_pass"]) for v in all_values)
    static_correct = sum(int(v["static_slot_pass"]) for v in all_values)
    wrong_correct = sum(int(v["wrong_alias_map_pass"]) for v in all_values)
    exact_correct = sum(int(v["exact_key_pass"]) for v in all_values)

    gen2_values = [
        value
        for row in rows
        if row["generation"] == 2
        for value in row["reads"].values()
    ]
    gen2_one_hop_correct = sum(
        int(v["one_hop_only_pass"]) for v in gen2_values
    )

    reversed_rows = [row for row in rows if row["order"] == "10"]
    gen2_rows = [row for row in rows if row["generation"] == 2]

    accuracy = alias_correct / total_reads
    static_accuracy = static_correct / total_reads
    wrong_accuracy = wrong_correct / total_reads
    exact_accuracy = exact_correct / total_reads

    dual_rate = sum(int(row["dual_success"]) for row in rows) / len(rows)
    reversed_dual_rate = sum(
        int(row["dual_success"]) for row in reversed_rows
    ) / len(reversed_rows)
    gen2_dual_rate = sum(
        int(row["dual_success"]) for row in gen2_rows
    ) / len(gen2_rows)

    gen2_one_hop_accuracy = (
        gen2_one_hop_correct / len(gen2_values)
    )

    mean_target_mass = sum(v["target_mass"] for v in all_values) / total_reads

    exact_match_count = sum(
        int(v["exact_key_match_available"]) for v in all_values
    )

    persistent_keys = {
        tuple(v["persistent_original_key"]) for v in all_values
    }
    current_keys = {
        tuple(v["current_record_key"]) for v in all_values
    }

    integrity_pass = all(row["cache_unchanged"] for row in rows)
    alias_pass = (
        len(persistent_keys) >= 12
        and len(current_keys) >= 24
        and exact_match_count == 0
        and accuracy >= 0.95
        and dual_rate >= 0.95
        and reversed_dual_rate >= 0.95
        and gen2_dual_rate >= 0.95
        and mean_target_mass >= 0.80
        and static_accuracy <= 0.50
        and exact_accuracy <= 0.10
        and wrong_accuracy <= 0.10
        and gen2_one_hop_accuracy <= 0.10
    )

    return {
        "gate": "6",
        "claim": (
            "translate a persistent provenance key across explicit one-hop "
            "and transitive alias drift before binding it to the current slot"
        ),
        "frozen_reader": {
            "model": base.model_id,
            "revision": base.revision,
            "layer": base.layer,
            "head": base.head,
            "first_slot_mode": base.mode_a.state,
            "second_slot_mode": base.mode_b.state,
        },
        "persistent_state": {
            "type": "original arbitrary provenance token tuple",
            "rewritten_when_alias_changes": False,
        },
        "representation_update": {
            "type": "generic undirected alias-equivalence edges",
            "maps_both_sources_symmetrically": True,
            "contains_trust_label": False,
            "supports_transitive_closure": True,
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
            "persistent_original_keys": len(persistent_keys),
            "distinct_current_alias_keys": len(current_keys),
            "exact_current_key_matches": exact_match_count,
            "alias_bound_accuracy": accuracy,
            "dual_success_rate": dual_rate,
            "reversed_order_dual_success_rate": reversed_dual_rate,
            "transitive_generation_dual_success_rate": gen2_dual_rate,
            "mean_target_mass": mean_target_mass,
            "static_slot_accuracy": static_accuracy,
            "gate5_exact_key_accuracy": exact_accuracy,
            "wrong_alias_map_accuracy": wrong_accuracy,
            "generation2_one_hop_only_accuracy": gen2_one_hop_accuracy,
        },
        "checks": {
            "integrity_pass": integrity_pass,
            "alias_translation_pass": alias_pass,
        },
        "pass": integrity_pass and alias_pass,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when the frozen scientific Gate-6 contract fails",
    )
    args = parser.parse_args()

    receipt = build_receipt()
    print(json.dumps(receipt, indent=2))
    if args.strict and not receipt["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
