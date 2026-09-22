"""Gate 2: real pretrained K/V, persistent scalar query observer.

One calibration read per block may update the observer.  The next three reads
are blind and must reuse the exact same prompt and projected K/V.
"""

from __future__ import annotations

import json

import torch

from pretrained_observer import (
    MAX_PERTURBATION_RATIO,
    OBSERVER_GRID_STEPS,
    build_fixture,
    cache_digest,
    read,
)


BLOCKS = ("A", "B", "A", "B")
READS_PER_BLOCK = 4


def _observer_for(fixture, source: str) -> float:
    return (
        fixture.mode_a.state
        if source == "A"
        else fixture.mode_b.state
    )


def run_fixed(fixture, observer: float):
    test_correct = 0
    test_total = 0
    all_correct = 0

    for trusted in BLOCKS:
        for offset in range(READS_PER_BLOCK):
            result = read(fixture, observer)
            correct = result.prediction == trusted
            all_correct += int(correct)
            if offset != 0:
                test_total += 1
                test_correct += int(correct)

    return {
        "observer": observer,
        "test_accuracy": test_correct / test_total,
        "all_read_accuracy": all_correct / (
            len(BLOCKS) * READS_PER_BLOCK
        ),
    }


def run_adaptive(fixture):
    observer = 0.0
    test_correct = 0
    test_total = 0
    all_correct = 0
    rows = []

    for block_index, trusted in enumerate(BLOCKS):
        for offset in range(READS_PER_BLOCK):
            result = read(fixture, observer)
            correct = result.prediction == trusted
            all_correct += int(correct)

            is_calibration = offset == 0
            observer_after = observer

            if is_calibration:
                # Receipt arrives only after this read.
                observer_after = _observer_for(fixture, trusted)
            else:
                test_total += 1
                test_correct += int(correct)

            rows.append(
                {
                    "block": block_index,
                    "offset": offset,
                    "trusted": trusted,
                    "calibration": is_calibration,
                    "feedback_visible_after_read": (
                        trusted if is_calibration else None
                    ),
                    "observer_before": observer,
                    "observer_after": observer_after,
                    "prediction": result.prediction,
                    "mass_A": result.mass_a,
                    "mass_B": result.mass_b,
                    "source_share_A": result.source_share_a,
                    "correct": correct,
                }
            )
            observer = observer_after

    return {
        "test_accuracy": test_correct / test_total,
        "all_read_accuracy": all_correct / (
            len(BLOCKS) * READS_PER_BLOCK
        ),
        "test_reads": test_total,
        "rows": rows,
    }


def run_reset_control(fixture):
    test_correct = 0
    test_total = 0

    for trusted in BLOCKS:
        read(fixture, 0.0)  # calibration read, then erase state
        for _ in range(READS_PER_BLOCK - 1):
            result = read(fixture, 0.0)
            test_total += 1
            test_correct += int(result.prediction == trusted)

    return test_correct / test_total


def build_receipt():
    fixture = build_fixture()
    cache_before = cache_digest(fixture.cache)

    mode_a = read(fixture, fixture.mode_a.state)
    neutral = read(fixture, 0.0)
    mode_b = read(fixture, fixture.mode_b.state)

    fixed_states = (
        fixture.mode_b.state,
        0.0,
        fixture.mode_a.state,
    )
    fixed = [
        run_fixed(fixture, observer) for observer in fixed_states
    ]
    best_fixed = max(item["test_accuracy"] for item in fixed)
    adaptive = run_adaptive(fixture)
    reset_accuracy = run_reset_control(fixture)

    cache_after = cache_digest(fixture.cache)

    output_distance = float(
        torch.linalg.vector_norm(mode_a.output - mode_b.output)
    )
    fixed_advantage = adaptive["test_accuracy"] - best_fixed

    geometry_pass = (
        mode_a.mass_a >= 0.20
        and mode_b.mass_b >= 0.20
        and mode_a.source_share_a >= 0.80
        and mode_b.source_share_a <= 0.20
        and output_distance > 1e-6
        and abs(fixture.mode_a.state) <= MAX_PERTURBATION_RATIO
        and abs(fixture.mode_b.state) <= MAX_PERTURBATION_RATIO
    )
    integrity_pass = cache_before == cache_after
    persistence_pass = (
        adaptive["test_accuracy"] >= 0.90
        and fixed_advantage >= 0.30
        and reset_accuracy <= 0.60
    )

    gate_pass = geometry_pass and integrity_pass and persistence_pass

    return {
        "gate": "2",
        "model": {
            "id": fixture.model_id,
            "revision": fixture.revision,
            "layer": fixture.layer,
            "head": fixture.head,
            "head_dim": fixture.head_dim,
        },
        "prompt": fixture.prompt,
        "source_spans": {
            "A": list(fixture.cache.source_a),
            "B": list(fixture.cache.source_b),
        },
        "tokens": list(fixture.tokens),
        "observer": {
            "equation": "q' = q_pretrained + m * ||q|| * u",
            "rank": 1,
            "max_abs_state": MAX_PERTURBATION_RATIO,
            "grid_steps": OBSERVER_GRID_STEPS,
            "mode_A": {
                "state": fixture.mode_a.state,
                "target_mass_from_search": fixture.mode_a.target_mass,
                "source_share_A_from_search": (
                    fixture.mode_a.source_share_a
                ),
            },
            "mode_B": {
                "state": fixture.mode_b.state,
                "target_mass_from_search": fixture.mode_b.target_mass,
                "source_share_A_from_search": (
                    fixture.mode_b.source_share_a
                ),
            },
            "symmetric_target_capture": fixture.symmetric_capture,
            "head_selection": (
                "for each pretrained head, derive u from the natural "
                "source-key mean difference; search the fixed scalar grid "
                "for the A-mass-maximizing and B-mass-maximizing states; "
                "choose the head maximizing the worse absolute target mass"
            ),
        },
        "paired_same_prompt_same_KV": {
            "mode_A": {
                "observer": fixture.mode_a.state,
                "prediction": mode_a.prediction,
                "mass_A": mode_a.mass_a,
                "mass_B": mode_a.mass_b,
                "source_share_A": mode_a.source_share_a,
            },
            "neutral": {
                "observer": 0.0,
                "prediction": neutral.prediction,
                "mass_A": neutral.mass_a,
                "mass_B": neutral.mass_b,
                "source_share_A": neutral.source_share_a,
            },
            "mode_B": {
                "observer": fixture.mode_b.state,
                "prediction": mode_b.prediction,
                "mass_A": mode_b.mass_a,
                "mass_B": mode_b.mass_b,
                "source_share_A": mode_b.source_share_a,
            },
            "attention_output_l2_between_modes": output_distance,
        },
        "protocol": {
            "blocks": list(BLOCKS),
            "reads_per_block": READS_PER_BLOCK,
            "calibration_reads_per_block": 1,
            "blind_test_reads_per_block": READS_PER_BLOCK - 1,
            "truth_visible_on_test_reads": False,
            "read_budget_per_trial": 1,
        },
        "fixed_readers": fixed,
        "best_fixed_test_accuracy": best_fixed,
        "adaptive": adaptive,
        "adaptive_advantage": fixed_advantage,
        "reset_observer_test_accuracy": reset_accuracy,
        "integrity": {
            "cache_digest_before": cache_before,
            "cache_digest_after": cache_after,
            "target_attention_parameter_digest": fixture.parameter_digest,
            "cache_unchanged": cache_before == cache_after,
        },
        "checks": {
            "absolute_source_engagement_pass": (
                mode_a.mass_a >= 0.20 and mode_b.mass_b >= 0.20
            ),
            "geometry_pass": geometry_pass,
            "integrity_pass": integrity_pass,
            "persistence_pass": persistence_pass,
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
