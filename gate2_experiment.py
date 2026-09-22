"""Gate 2: real pretrained hidden states/KV, persistent query observer.

Each block has one calibration read followed by three blind test reads.
Only the calibration read reveals which provenance is trusted.  The prompt,
pretrained K/V, and query base are identical on every trial.
"""

from __future__ import annotations

import json

import torch

from pretrained_observer import (
    build_fixture,
    cache_digest,
    read,
)


BLOCKS = ("A", "B", "A", "B")
READS_PER_BLOCK = 4
FIXED_OBSERVERS = (-1.0, 0.0, 1.0)


def _observer_for(source: str) -> float:
    return 1.0 if source == "A" else -1.0


def run_fixed(fixture, observer: float):
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
            if not is_calibration:
                test_total += 1
                test_correct += int(correct)

            rows.append(
                {
                    "block": block_index,
                    "offset": offset,
                    "trusted": trusted,
                    "calibration": is_calibration,
                    "observer": observer,
                    "prediction": result.prediction,
                    "source_share_A": result.source_share_a,
                    "correct": correct,
                }
            )

    return {
        "observer": observer,
        "test_accuracy": test_correct / test_total,
        "all_read_accuracy": all_correct / (len(BLOCKS) * READS_PER_BLOCK),
        "rows": rows,
    }


def run_adaptive(fixture):
    # Neutral at the very beginning.  Only a calibration receipt can set the
    # persistent observer for subsequent blind reads.
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
                # Truth arrives strictly after this read and is then carried
                # forward.  No later read in the block receives truth.
                observer_after = _observer_for(trusted)
            else:
                test_total += 1
                test_correct += int(correct)

            rows.append(
                {
                    "block": block_index,
                    "offset": offset,
                    "trusted": trusted,
                    "calibration": is_calibration,
                    "feedback_visible_after_read": trusted
                    if is_calibration
                    else None,
                    "observer_before": observer,
                    "observer_after": observer_after,
                    "prediction": result.prediction,
                    "source_share_A": result.source_share_a,
                    "correct": correct,
                }
            )
            observer = observer_after

    return {
        "test_accuracy": test_correct / test_total,
        "all_read_accuracy": all_correct / (len(BLOCKS) * READS_PER_BLOCK),
        "test_reads": test_total,
        "rows": rows,
    }


def run_reset_control(fixture):
    """Calibration exists, but state is erased before every blind test read."""

    test_correct = 0
    test_total = 0
    for trusted in BLOCKS:
        # Calibration read; receipt is intentionally discarded.
        read(fixture, 0.0)
        for _ in range(READS_PER_BLOCK - 1):
            result = read(fixture, 0.0)
            test_total += 1
            test_correct += int(result.prediction == trusted)

    return test_correct / test_total


def build_receipt():
    fixture = build_fixture()
    cache_before = cache_digest(fixture.cache)

    plus = read(fixture, +1.0)
    neutral = read(fixture, 0.0)
    minus = read(fixture, -1.0)

    fixed = [
        run_fixed(fixture, observer) for observer in FIXED_OBSERVERS
    ]
    best_fixed = max(item["test_accuracy"] for item in fixed)
    adaptive = run_adaptive(fixture)
    reset_accuracy = run_reset_control(fixture)

    cache_after = cache_digest(fixture.cache)

    output_distance = float(
        torch.linalg.vector_norm(plus.output - minus.output)
    )
    fixed_advantage = adaptive["test_accuracy"] - best_fixed

    geometry_pass = (
        plus.source_share_a >= 0.80
        and minus.source_share_a <= 0.20
        and output_distance > 1e-6
        and fixture.perturbation_ratio <= 4.0
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
            "equation": "q' = q_pretrained + m * strength * u",
            "rank": 1,
            "target_mean_logit_margin": 4.0,
            "strength": fixture.observer_strength,
            "perturbation_to_base_query_norm": fixture.perturbation_ratio,
            "base_source_logit_delta": fixture.base_source_logit_delta,
            "unit_source_logit_delta": fixture.unit_source_logit_delta,
            "head_selection": (
                "choose the pretrained layer/head requiring the smallest "
                "observer perturbation, relative to its natural query norm, "
                "to create +/-4 mean source-logit separation"
            ),
        },
        "paired_same_prompt_same_KV": {
            "m_plus_1": {
                "prediction": plus.prediction,
                "mass_A": plus.mass_a,
                "mass_B": plus.mass_b,
                "source_share_A": plus.source_share_a,
            },
            "m_0": {
                "prediction": neutral.prediction,
                "mass_A": neutral.mass_a,
                "mass_B": neutral.mass_b,
                "source_share_A": neutral.source_share_a,
            },
            "m_minus_1": {
                "prediction": minus.prediction,
                "mass_A": minus.mass_a,
                "mass_B": minus.mass_b,
                "source_share_A": minus.source_share_a,
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
