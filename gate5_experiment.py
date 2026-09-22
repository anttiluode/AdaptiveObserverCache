"""Gate 5: arbitrary provenance key -> current slot -> frozen slot reader."""

from __future__ import annotations

import json

from gate5_generic_binding import (
    bind_persistent_key,
    build_generic_suite,
    capture_persistent_key,
    read_slot,
)
from pretrained_observer import cache_digest


TARGET_MASS = 0.20
TARGET_SHARE = 0.80


def read_success(result) -> bool:
    return (
        result.prediction == result.target_key
        and result.target_mass >= TARGET_MASS
        and result.target_share >= TARGET_SHARE
    )


def evaluate_case(base, tokenizer, projection):
    before = cache_digest(projection.cache)
    family = projection.case.family
    rows = {}
    bound_correct = 0
    static_correct = 0
    wrong_key_correct = 0

    for target_index, target_key in enumerate(
        (family.key0, family.key1)
    ):
        # Calibration copies the selected record's arbitrary provenance field
        # into slow state. There is no global identity table.
        state_key = capture_persistent_key(tokenizer, target_key)
        actual_slot = bind_persistent_key(state_key, projection)
        bound = read_slot(base, projection, target_key, actual_slot)

        # Attacker: assumes calibration order is permanently key0->slot0,
        # key1->slot1 even after records move.
        static_slot = target_index
        static = read_slot(base, projection, target_key, static_slot)

        other_key = (
            family.key1 if target_key == family.key0 else family.key0
        )
        wrong_state = capture_persistent_key(tokenizer, other_key)
        wrong_slot = bind_persistent_key(wrong_state, projection)
        wrong = read_slot(base, projection, target_key, wrong_slot)

        bound_ok = read_success(bound)
        static_ok = read_success(static)
        wrong_ok = read_success(wrong)

        bound_correct += int(bound_ok)
        static_correct += int(static_ok)
        wrong_key_correct += int(wrong_ok)

        rows[target_key] = {
            "persistent_key_tokens": list(state_key),
            "actual_slot": actual_slot,
            "observer": bound.observer,
            "target_mass": bound.target_mass,
            "target_share": bound.target_share,
            "prediction": bound.prediction,
            "bound_pass": bound_ok,
            "static_slot_pass": static_ok,
            "wrong_key_pass": wrong_ok,
        }

    after = cache_digest(projection.cache)

    return {
        "name": projection.case.name,
        "order": projection.case.order,
        "provenance_pair": [family.key0, family.key1],
        "prompt": projection.prompt,
        "reads": rows,
        "dual_success": bound_correct == 2,
        "bound_accuracy": bound_correct / 2,
        "static_accuracy": static_correct / 2,
        "wrong_key_accuracy": wrong_key_correct / 2,
        "cache_unchanged": before == after,
        "cache_digest": before,
    }


def build_receipt():
    base, tokenizer, projections = build_generic_suite()
    rows = [evaluate_case(base, tokenizer, p) for p in projections]

    total_reads = 2 * len(rows)
    bound_correct = sum(
        int(v["bound_pass"])
        for row in rows
        for v in row["reads"].values()
    )
    static_correct = sum(
        int(v["static_slot_pass"])
        for row in rows
        for v in row["reads"].values()
    )
    wrong_correct = sum(
        int(v["wrong_key_pass"])
        for row in rows
        for v in row["reads"].values()
    )

    reversed_rows = [row for row in rows if row["order"] == "10"]
    dual_rate = sum(int(row["dual_success"]) for row in rows) / len(rows)
    reversed_dual_rate = sum(
        int(row["dual_success"]) for row in reversed_rows
    ) / len(reversed_rows)

    mean_target_mass = sum(
        v["target_mass"]
        for row in rows
        for v in row["reads"].values()
    ) / total_reads

    bound_accuracy = bound_correct / total_reads
    static_accuracy = static_correct / total_reads
    wrong_accuracy = wrong_correct / total_reads

    unique_keys = {
        tuple(v["persistent_key_tokens"])
        for row in rows
        for v in row["reads"].values()
    }

    integrity_pass = all(row["cache_unchanged"] for row in rows)
    generic_pass = (
        len(unique_keys) >= 12
        and bound_accuracy >= 0.95
        and dual_rate >= 0.95
        and reversed_dual_rate >= 0.95
        and mean_target_mass >= 0.80
        and static_accuracy <= 0.50
        and wrong_accuracy <= 0.10
    )

    return {
        "gate": "5",
        "claim": (
            "replace the hard-coded Alice/Bob identity table with an "
            "arbitrary provenance key captured into persistent state"
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
            "type": "arbitrary provenance token tuple",
            "captured_from_calibration_record": True,
            "hard_coded_identity_table": False,
        },
        "binder": {
            "operation": "exact equality of persistent key to current record key",
            "uses_trust_during_test": False,
            "uses_attention_KV_geometry": False,
        },
        "thresholds": {
            "target_absolute_mass": TARGET_MASS,
            "target_source_share": TARGET_SHARE,
        },
        "results": rows,
        "summary": {
            "families": 6,
            "cases": len(rows),
            "unique_persistent_keys": len(unique_keys),
            "identity_reads": total_reads,
            "generic_bound_accuracy": bound_accuracy,
            "dual_success_rate": dual_rate,
            "reversed_order_dual_success_rate": reversed_dual_rate,
            "mean_target_mass": mean_target_mass,
            "static_calibration_slot_accuracy": static_accuracy,
            "wrong_persistent_key_accuracy": wrong_accuracy,
        },
        "checks": {
            "integrity_pass": integrity_pass,
            "generic_binding_pass": generic_pass,
        },
        "pass": integrity_pass and generic_pass,
    }


def main():
    receipt = build_receipt()
    print(json.dumps(receipt, indent=2))
    if not receipt["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
