"""Gate 3: frozen observer transfer to unseen DistilGPT2 caches."""

from __future__ import annotations

import json

from gate3_transfer import build_transfer_suite, read_transfer
from pretrained_observer import cache_digest


TARGET_MASS = 0.20
TARGET_SHARE = 0.80


def evaluate_projection(base, projection):
    before = cache_digest(projection.cache)

    a = read_transfer(base, projection, base.mode_a.state)
    b = read_transfer(base, projection, base.mode_b.state)
    neutral = read_transfer(base, projection, 0.0)

    after = cache_digest(projection.cache)

    a_ok = a.mass_a >= TARGET_MASS and a.source_share_a >= TARGET_SHARE
    b_ok = (
        b.mass_b >= TARGET_MASS
        and b.source_share_a <= (1.0 - TARGET_SHARE)
    )

    return {
        "name": projection.case.name,
        "category": projection.case.category,
        "prompt": projection.prompt,
        "local_direction_cosine_to_frozen_u": (
            projection.local_direction_cosine
        ),
        "A_mode": {
            "observer": base.mode_a.state,
            "mass_A": a.mass_a,
            "mass_B": a.mass_b,
            "source_share_A": a.source_share_a,
            "prediction": a.prediction,
            "pass": a_ok,
        },
        "B_mode": {
            "observer": base.mode_b.state,
            "mass_A": b.mass_a,
            "mass_B": b.mass_b,
            "source_share_A": b.source_share_a,
            "prediction": b.prediction,
            "pass": b_ok,
        },
        "neutral": {
            "mass_A": neutral.mass_a,
            "mass_B": neutral.mass_b,
            "source_share_A": neutral.source_share_a,
            "prediction": neutral.prediction,
        },
        "dual_success": a_ok and b_ok,
        "cache_unchanged": before == after,
        "cache_digest": before,
    }


def build_receipt():
    base, projections = build_transfer_suite()
    rows = [evaluate_projection(base, p) for p in projections]

    transfer_rows = [
        row for row in rows if row["category"] != "order_swap"
    ]
    same_layout = [
        row for row in rows if row["category"] == "same_layout"
    ]
    paraphrase = [
        row for row in rows if row["category"] == "paraphrase"
    ]
    order_swap = [
        row for row in rows if row["category"] == "order_swap"
    ]

    def rate(group):
        return (
            sum(int(row["dual_success"]) for row in group) / len(group)
            if group
            else 0.0
        )

    transfer_rate = rate(transfer_rows)
    same_layout_rate = rate(same_layout)
    paraphrase_rate = rate(paraphrase)
    order_swap_rate = rate(order_swap)

    mean_target_mass = sum(
        min(row["A_mode"]["mass_A"], row["B_mode"]["mass_B"])
        for row in transfer_rows
    ) / len(transfer_rows)

    mean_direction_cosine = sum(
        row["local_direction_cosine_to_frozen_u"]
        for row in transfer_rows
    ) / len(transfer_rows)

    integrity_pass = all(row["cache_unchanged"] for row in rows)
    transfer_pass = (
        transfer_rate >= 5 / 7
        and same_layout_rate >= 4 / 5
        and paraphrase_rate >= 1 / 2
        and mean_target_mass >= 0.20
    )

    return {
        "gate": "3",
        "claim": (
            "freeze Gate-2 layer/head, observer direction, and scalar modes; "
            "apply them to unseen pretrained caches without rebuilding"
        ),
        "frozen_from_gate2": {
            "model": base.model_id,
            "revision": base.revision,
            "layer": base.layer,
            "head": base.head,
            "observer_direction_recomputed_on_test": False,
            "mode_A": base.mode_a.state,
            "mode_B": base.mode_b.state,
        },
        "thresholds": {
            "target_absolute_mass": TARGET_MASS,
            "target_source_share": TARGET_SHARE,
            "transfer_dual_success_required": ">=5/7",
            "same_layout_required": ">=4/5",
            "paraphrase_required": ">=1/2",
            "order_swap": "diagnostic attacker; not part of Gate-3 pass",
        },
        "results": rows,
        "summary": {
            "transfer_dual_success_rate": transfer_rate,
            "same_layout_dual_success_rate": same_layout_rate,
            "paraphrase_dual_success_rate": paraphrase_rate,
            "order_swap_dual_success_rate": order_swap_rate,
            "mean_worst_target_mass_transfer": mean_target_mass,
            "mean_local_direction_cosine_to_frozen_u": (
                mean_direction_cosine
            ),
        },
        "checks": {
            "integrity_pass": integrity_pass,
            "transfer_pass": transfer_pass,
        },
        "pass": integrity_pass and transfer_pass,
    }


def main():
    receipt = build_receipt()
    print(json.dumps(receipt, indent=2))
    if not receipt["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
