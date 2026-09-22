"""Gate 4: persistent identity choice + current provenance-to-slot binding."""

from __future__ import annotations

import json

from gate4_binding import bind_slots, build_binding_suite, read_slot
from pretrained_observer import cache_digest


TARGET_MASS = 0.20
TARGET_SHARE = 0.80


def read_success(result) -> bool:
    return (
        result.prediction == result.target_identity
        and result.target_mass >= TARGET_MASS
        and result.target_share >= TARGET_SHARE
    )


def evaluate_case(base, binder, projection):
    before = cache_digest(projection.cache)
    mapping = bind_slots(binder, projection)

    identity_rows = {}
    binder_successes = 0
    static_successes = 0
    inverted_successes = 0

    for identity in ("Alice", "Bob"):
        actual_slot = mapping[identity]
        bound = read_slot(base, projection, identity, actual_slot)

        static_slot = 0 if identity == "Alice" else 1
        static = read_slot(base, projection, identity, static_slot)

        inverted = read_slot(
            base, projection, identity, 1 - actual_slot
        )

        bound_ok = read_success(bound)
        static_ok = read_success(static)
        inverted_ok = read_success(inverted)

        binder_successes += int(bound_ok)
        static_successes += int(static_ok)
        inverted_successes += int(inverted_ok)

        identity_rows[identity] = {
            "actual_slot": actual_slot,
            "bound_observer": bound.observer,
            "target_mass": bound.target_mass,
            "target_share": bound.target_share,
            "prediction": bound.prediction,
            "bound_pass": bound_ok,
            "static_binding_slot": static_slot,
            "static_binding_pass": static_ok,
            "inverted_binding_pass": inverted_ok,
        }

    after = cache_digest(projection.cache)

    return {
        "name": projection.case.name,
        "order": projection.case.order,
        "prompt": projection.prompt,
        "binding": mapping,
        "identity_reads": identity_rows,
        "dual_identity_success": binder_successes == 2,
        "binder_accuracy": binder_successes / 2,
        "static_binding_accuracy": static_successes / 2,
        "inverted_binding_accuracy": inverted_successes / 2,
        "cache_unchanged": before == after,
        "cache_digest": before,
    }


def build_receipt():
    base, binder, projections = build_binding_suite()
    rows = [evaluate_case(base, binder, p) for p in projections]

    total_identity_reads = 2 * len(rows)
    binder_correct = sum(
        int(v["bound_pass"])
        for row in rows
        for v in row["identity_reads"].values()
    )
    static_correct = sum(
        int(v["static_binding_pass"])
        for row in rows
        for v in row["identity_reads"].values()
    )
    inverted_correct = sum(
        int(v["inverted_binding_pass"])
        for row in rows
        for v in row["identity_reads"].values()
    )

    dual_success_rate = sum(
        int(row["dual_identity_success"]) for row in rows
    ) / len(rows)
    swapped_rows = [row for row in rows if row["order"] == "BA"]
    swapped_dual_rate = sum(
        int(row["dual_identity_success"]) for row in swapped_rows
    ) / len(swapped_rows)

    mean_target_mass = sum(
        v["target_mass"]
        for row in rows
        for v in row["identity_reads"].values()
    ) / total_identity_reads

    alice_sequence_accuracy = sum(
        int(row["identity_reads"]["Alice"]["bound_pass"])
        for row in rows
    ) / len(rows)
    bob_sequence_accuracy = sum(
        int(row["identity_reads"]["Bob"]["bound_pass"])
        for row in rows
    ) / len(rows)

    binder_accuracy = binder_correct / total_identity_reads
    static_accuracy = static_correct / total_identity_reads
    inverted_accuracy = inverted_correct / total_identity_reads

    integrity_pass = all(row["cache_unchanged"] for row in rows)
    binding_pass = (
        binder_accuracy >= 0.95
        and dual_success_rate >= 0.95
        and swapped_dual_rate >= 0.95
        and alice_sequence_accuracy >= 0.95
        and bob_sequence_accuracy >= 0.95
        and mean_target_mass >= 0.80
        and static_accuracy <= 0.50
        and inverted_accuracy <= 0.10
    )

    return {
        "gate": "4",
        "claim": (
            "factor persistent trusted identity from current source-slot "
            "binding, then reuse the frozen Gate-2/3 positional read modes"
        ),
        "frozen_reader": {
            "model": base.model_id,
            "revision": base.revision,
            "layer": base.layer,
            "head": base.head,
            "first_slot_mode": base.mode_a.state,
            "second_slot_mode": base.mode_b.state,
            "observer_direction_recomputed_on_test": False,
        },
        "persistent_state": {
            "values": ["Alice", "Bob"],
            "updated_during_test_sequence": False,
        },
        "binder": {
            "input": (
                "explicit source-record boundary + first provenance token ID"
            ),
            "identity_token_ids": binder.identity_token_ids,
            "uses_attention_KV_geometry": False,
            "uses_trust_label": False,
        },
        "thresholds": {
            "target_absolute_mass": TARGET_MASS,
            "target_source_share": TARGET_SHARE,
        },
        "results": rows,
        "summary": {
            "cases": len(rows),
            "identity_reads": total_identity_reads,
            "binder_accuracy": binder_accuracy,
            "dual_identity_success_rate": dual_success_rate,
            "swapped_dual_identity_success_rate": swapped_dual_rate,
            "persistent_Alice_sequence_accuracy": (
                alice_sequence_accuracy
            ),
            "persistent_Bob_sequence_accuracy": bob_sequence_accuracy,
            "mean_target_mass": mean_target_mass,
            "static_identity_to_slot_accuracy": static_accuracy,
            "inverted_binder_accuracy": inverted_accuracy,
        },
        "checks": {
            "integrity_pass": integrity_pass,
            "binding_pass": binding_pass,
        },
        "pass": integrity_pass and binding_pass,
    }


def main():
    receipt = build_receipt()
    print(json.dumps(receipt, indent=2))
    if not receipt["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
