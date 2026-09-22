"""Gate 6: canonical trust survives multi-hop provenance alias drift."""

from __future__ import annotations

import json

from gate6_relational_alias import (
    bind_exact_key,
    bind_one_hop,
    bind_via_relation_graph,
    build_alias_suite,
    canonical_key,
    read_slot,
    relation_edges,
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


def evaluate_case(base, tokenizer, projection):
    before = cache_digest(projection.cache)
    family = projection.case.family
    correct_edges = relation_edges(tokenizer, family, shuffled=False)
    shuffled_edges = relation_edges(tokenizer, family, shuffled=True)

    rows = {}
    graph_correct = 0
    exact_correct = 0
    one_hop_correct = 0
    shuffled_correct = 0
    path_lengths = []

    for canonical in (family.canonical0, family.canonical1):
        state_key = canonical_key(tokenizer, canonical)

        slot, path_length = bind_via_relation_graph(
            state_key,
            projection,
            correct_edges,
        )
        bound = read_slot(base, projection, canonical, slot)
        bound_ok = read_success(bound)
        graph_correct += int(bound_ok)
        path_lengths.append(path_length)

        exact_slot = bind_exact_key(state_key, projection)
        exact_ok = False
        if exact_slot is not None:
            exact_ok = read_success(
                read_slot(base, projection, canonical, exact_slot)
            )
        exact_correct += int(exact_ok)

        one_hop_slot = bind_one_hop(
            state_key,
            projection,
            correct_edges,
        )
        one_hop_ok = False
        if one_hop_slot is not None:
            one_hop_ok = read_success(
                read_slot(base, projection, canonical, one_hop_slot)
            )
        one_hop_correct += int(one_hop_ok)

        shuffled_slot, shuffled_path = bind_via_relation_graph(
            state_key,
            projection,
            shuffled_edges,
        )
        shuffled = read_slot(
            base,
            projection,
            canonical,
            shuffled_slot,
        )
        shuffled_ok = read_success(shuffled)
        shuffled_correct += int(shuffled_ok)

        rows[canonical] = {
            "persistent_canonical_key_tokens": list(state_key),
            "current_alias_slot": slot,
            "relation_path_length": path_length,
            "observer": bound.observer,
            "target_mass": bound.target_mass,
            "target_share": bound.target_share,
            "prediction": bound.prediction,
            "graph_bound_pass": bound_ok,
            "exact_key_pass": exact_ok,
            "one_hop_pass": one_hop_ok,
            "shuffled_graph_slot": shuffled_slot,
            "shuffled_graph_path_length": shuffled_path,
            "shuffled_graph_pass": shuffled_ok,
        }

    after = cache_digest(projection.cache)

    return {
        "name": projection.case.name,
        "order": projection.case.order,
        "canonical_pair": [family.canonical0, family.canonical1],
        "current_alias_pair": [family.alias0, family.alias1],
        "prompt": projection.prompt,
        "reads": rows,
        "dual_success": graph_correct == 2,
        "graph_accuracy": graph_correct / 2,
        "exact_key_accuracy": exact_correct / 2,
        "one_hop_accuracy": one_hop_correct / 2,
        "shuffled_graph_accuracy": shuffled_correct / 2,
        "minimum_relation_hops": min(path_lengths),
        "cache_unchanged": before == after,
        "cache_digest": before,
    }


def build_receipt():
    base, tokenizer, projections = build_alias_suite()
    rows = [evaluate_case(base, tokenizer, p) for p in projections]

    total_reads = 2 * len(rows)
    graph_correct = sum(
        int(v["graph_bound_pass"])
        for row in rows
        for v in row["reads"].values()
    )
    exact_correct = sum(
        int(v["exact_key_pass"])
        for row in rows
        for v in row["reads"].values()
    )
    one_hop_correct = sum(
        int(v["one_hop_pass"])
        for row in rows
        for v in row["reads"].values()
    )
    shuffled_correct = sum(
        int(v["shuffled_graph_pass"])
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

    unique_canonical_keys = {
        tuple(v["persistent_canonical_key_tokens"])
        for row in rows
        for v in row["reads"].values()
    }

    graph_accuracy = graph_correct / total_reads
    exact_accuracy = exact_correct / total_reads
    one_hop_accuracy = one_hop_correct / total_reads
    shuffled_accuracy = shuffled_correct / total_reads
    minimum_relation_hops = min(
        row["minimum_relation_hops"] for row in rows
    )

    integrity_pass = all(row["cache_unchanged"] for row in rows)
    relational_pass = (
        len(unique_canonical_keys) >= 12
        and graph_accuracy >= 0.95
        and dual_rate >= 0.95
        and reversed_dual_rate >= 0.95
        and mean_target_mass >= 0.80
        and minimum_relation_hops >= 2
        and exact_accuracy <= 0.10
        and one_hop_accuracy <= 0.10
        and shuffled_accuracy <= 0.10
    )

    return {
        "gate": "6",
        "claim": (
            "carry a canonical trusted provenance key while current records "
            "change surface aliases, and bind through an independent "
            "multi-hop identity relation graph"
        ),
        "frozen_reader": {
            "model": base.model_id,
            "revision": base.revision,
            "layer": base.layer,
            "head": base.head,
            "first_slot_mode": base.mode_a.state,
            "second_slot_mode": base.mode_b.state,
        },
        "state_and_binding": {
            "persistent_state": "canonical provenance token tuple",
            "current_record_identity": "different terminal alias token tuple",
            "binder": "generic undirected graph reachability",
            "minimum_required_relation_hops": 2,
            "relation_graph_uses_trust": False,
            "relation_graph_uses_attention_KV_geometry": False,
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
            "unique_canonical_keys": len(unique_canonical_keys),
            "relational_bound_accuracy": graph_accuracy,
            "dual_success_rate": dual_rate,
            "reversed_order_dual_success_rate": reversed_dual_rate,
            "mean_target_mass": mean_target_mass,
            "minimum_relation_hops": minimum_relation_hops,
            "exact_key_accuracy": exact_accuracy,
            "one_hop_alias_accuracy": one_hop_accuracy,
            "shuffled_graph_accuracy": shuffled_accuracy,
        },
        "checks": {
            "integrity_pass": integrity_pass,
            "relational_binding_pass": relational_pass,
        },
        "pass": integrity_pass and relational_pass,
    }


def main():
    receipt = build_receipt()
    print(json.dumps(receipt, indent=2))
    if not receipt["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
