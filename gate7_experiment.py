"""Gate 7: actively recover identity when alias relations are missing."""

from __future__ import annotations

import json

from gate6_relational_alias import read_slot
from gate7_active_identity import (
    PROBES,
    audit_current_aliases,
    build_active_identity_suite,
    infer_mapping,
    select_active_probe,
    static_slot_mapping,
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


def evaluate_case(
    base,
    history,
    fixed_probe: str,
    projection,
):
    before = cache_digest(projection.cache)
    family = projection.case.family
    c0, c1 = family.canonical0, family.canonical1

    active_probe = select_active_probe(history, c0, c1)
    active_observation = audit_current_aliases(projection, active_probe)
    active_mapping = infer_mapping(
        history,
        c0,
        c1,
        active_observation,
    )
    if active_mapping is None:
        raise RuntimeError("active audit failed to identify current binding")

    fixed_observation = audit_current_aliases(projection, fixed_probe)
    fixed_mapping = infer_mapping(
        history,
        c0,
        c1,
        fixed_observation,
    )
    fixed_resolved = fixed_mapping is not None
    if fixed_mapping is None:
        fixed_mapping = static_slot_mapping(c0, c1)

    passive_mapping = static_slot_mapping(c0, c1)

    shuffled_history_mapping = infer_mapping(
        history,
        c0,
        c1,
        active_observation,
        swap_history_labels=True,
    )
    if shuffled_history_mapping is None:
        raise RuntimeError("shuffled historical labels unexpectedly unresolved")

    rows = {}
    active_correct = 0
    fixed_correct = 0
    passive_correct = 0
    shuffled_correct = 0

    for canonical in (c0, c1):
        active = read_slot(
            base,
            projection,
            canonical,
            active_mapping[canonical],
        )
        fixed = read_slot(
            base,
            projection,
            canonical,
            fixed_mapping[canonical],
        )
        passive = read_slot(
            base,
            projection,
            canonical,
            passive_mapping[canonical],
        )
        shuffled = read_slot(
            base,
            projection,
            canonical,
            shuffled_history_mapping[canonical],
        )

        active_ok = read_success(active)
        fixed_ok = read_success(fixed)
        passive_ok = read_success(passive)
        shuffled_ok = read_success(shuffled)

        active_correct += int(active_ok)
        fixed_correct += int(fixed_ok)
        passive_correct += int(passive_ok)
        shuffled_correct += int(shuffled_ok)

        rows[canonical] = {
            "active_slot": active_mapping[canonical],
            "active_observer": active.observer,
            "target_mass": active.target_mass,
            "target_share": active.target_share,
            "active_pass": active_ok,
            "fixed_global_probe_pass": fixed_ok,
            "passive_static_slot_pass": passive_ok,
            "shuffled_history_pass": shuffled_ok,
        }

    after = cache_digest(projection.cache)

    return {
        "name": projection.case.name,
        "order": projection.case.order,
        "canonical_pair": [c0, c1],
        "current_alias_pair": [family.alias0, family.alias1],
        "active_probe": active_probe,
        "active_responses_by_slot": list(
            active_observation.responses_by_slot
        ),
        "fixed_global_probe": fixed_probe,
        "fixed_probe_resolved_pair": fixed_resolved,
        "reads": rows,
        "active_dual_success": active_correct == 2,
        "active_accuracy": active_correct / 2,
        "fixed_probe_accuracy": fixed_correct / 2,
        "passive_accuracy": passive_correct / 2,
        "shuffled_history_accuracy": shuffled_correct / 2,
        "cache_unchanged": before == after,
        "cache_digest": before,
    }


def build_receipt():
    (
        base,
        _tokenizer,
        projections,
        history,
        fixed_probe,
        fixed_separable_pairs,
    ) = build_active_identity_suite()

    rows = [
        evaluate_case(base, history, fixed_probe, projection)
        for projection in projections
    ]

    total_reads = 2 * len(rows)
    active_correct = sum(
        int(v["active_pass"])
        for row in rows
        for v in row["reads"].values()
    )
    fixed_correct = sum(
        int(v["fixed_global_probe_pass"])
        for row in rows
        for v in row["reads"].values()
    )
    passive_correct = sum(
        int(v["passive_static_slot_pass"])
        for row in rows
        for v in row["reads"].values()
    )
    shuffled_correct = sum(
        int(v["shuffled_history_pass"])
        for row in rows
        for v in row["reads"].values()
    )

    active_accuracy = active_correct / total_reads
    fixed_accuracy = fixed_correct / total_reads
    passive_accuracy = passive_correct / total_reads
    shuffled_accuracy = shuffled_correct / total_reads

    reversed_rows = [row for row in rows if row["order"] == "10"]
    dual_rate = sum(
        int(row["active_dual_success"]) for row in rows
    ) / len(rows)
    reversed_dual_rate = sum(
        int(row["active_dual_success"]) for row in reversed_rows
    ) / len(reversed_rows)

    mean_target_mass = sum(
        v["target_mass"]
        for row in rows
        for v in row["reads"].values()
    ) / total_reads

    active_probes_used = sorted({row["active_probe"] for row in rows})
    all_active_pairs_separated = all(
        row["active_responses_by_slot"][0]
        != row["active_responses_by_slot"][1]
        for row in rows
    )

    integrity_pass = all(row["cache_unchanged"] for row in rows)
    active_pass = (
        active_accuracy >= 0.95
        and dual_rate >= 0.95
        and reversed_dual_rate >= 0.95
        and mean_target_mass >= 0.80
        and all_active_pairs_separated
        and len(active_probes_used) >= 3
        and active_accuracy - fixed_accuracy >= 0.10
        and passive_accuracy <= 0.50
        and shuffled_accuracy <= 0.10
    )

    return {
        "gate": "7",
        "claim": (
            "when current alias relations are missing, choose one audit "
            "challenge from historical source profiles to infer the live "
            "identity-to-slot permutation before reading frozen K/V"
        ),
        "frozen_reader": {
            "model": base.model_id,
            "revision": base.revision,
            "layer": base.layer,
            "head": base.head,
            "first_slot_mode": base.mode_a.state,
            "second_slot_mode": base.mode_b.state,
        },
        "identity_recovery": {
            "candidate_probes": list(PROBES),
            "challenge_rounds_per_case": 1,
            "responses_per_round": 2,
            "historical_profile_source": "prior audit receipts",
            "current_alias_relation_edge_supplied": False,
            "active_selector": (
                "choose a challenge whose historical responses differ for "
                "the two candidate canonical identities"
            ),
            "best_fixed_global_probe": fixed_probe,
            "best_fixed_separable_pairs_of_6": fixed_separable_pairs,
        },
        "thresholds": {
            "target_absolute_mass": TARGET_MASS,
            "target_source_share": TARGET_SHARE,
        },
        "results": rows,
        "summary": {
            "cases": len(rows),
            "identity_reads": total_reads,
            "active_binding_accuracy": active_accuracy,
            "active_dual_success_rate": dual_rate,
            "reversed_order_dual_success_rate": reversed_dual_rate,
            "mean_target_mass": mean_target_mass,
            "distinct_active_probes_used": len(active_probes_used),
            "active_probes_used": active_probes_used,
            "best_fixed_probe_accuracy": fixed_accuracy,
            "active_advantage_over_best_fixed": (
                active_accuracy - fixed_accuracy
            ),
            "passive_static_slot_accuracy": passive_accuracy,
            "shuffled_history_accuracy": shuffled_accuracy,
        },
        "checks": {
            "integrity_pass": integrity_pass,
            "active_identity_recovery_pass": active_pass,
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
