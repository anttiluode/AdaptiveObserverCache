"""Dependency-free helpers for the first-ask Qwen distance discriminator."""

from __future__ import annotations

import math


def split_first_ask_ids(full_ids, *, assistant_end_id: int):
    """Split system-only cache prefix from first user-question suffix.

    ``full_ids`` must be the exact tokenization of system + first user question
    with an assistant generation prompt. The first message terminator closes
    the system message; everything after it is withheld until the checkpoint.
    """

    ids = [int(x) for x in full_ids]
    try:
        boundary = ids.index(int(assistant_end_id))
    except ValueError as exc:
        raise ValueError("first-ask prompt has no system-message terminator") from exc

    source_ids = ids[: boundary + 1]
    question_suffix = ids[boundary + 1 :]
    if not source_ids or not question_suffix:
        raise ValueError("first-ask split produced an empty prefix or suffix")
    return source_ids, question_suffix


def first_divergence(ids_a, ids_b):
    """Return the first equal-depth token decision after the common prefix."""

    a = [int(x) for x in ids_a]
    b = [int(x) for x in ids_b]
    limit = min(len(a), len(b))
    index = 0
    while index < limit and a[index] == b[index]:
        index += 1

    if index >= limit:
        raise ValueError(
            "candidate tokenizations do not contain an equal-depth divergence"
        )

    return {
        "index": index,
        "common_prefix": a[:index],
        "token_a": a[index],
        "token_b": b[index],
    }


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def sequence_control_summary(
    *,
    sum_a: float,
    sum_b: float,
    saturation_threshold: float = 0.99,
):
    """Summarize the pairwise decision implied by two sequence log scores."""

    if not 0.5 < saturation_threshold < 1.0:
        raise ValueError("saturation_threshold must lie in (0.5, 1.0)")

    margin = float(sum_a) - float(sum_b)
    pair_a = _sigmoid(margin)
    if margin > 0.0:
        winner = "A"
        winner_probability = pair_a
    elif margin < 0.0:
        winner = "B"
        winner_probability = 1.0 - pair_a
    else:
        winner = "tie"
        winner_probability = 0.5

    return {
        "sum_logprob_A": float(sum_a),
        "sum_logprob_B": float(sum_b),
        "sum_margin_A_minus_B": margin,
        "winner": winner,
        "pair_probability_A": pair_a,
        "pair_probability_B": 1.0 - pair_a,
        "winner_pair_probability": winner_probability,
        "saturated": winner_probability >= saturation_threshold,
        "saturation_threshold": float(saturation_threshold),
    }


def winner_flip_gate(margins_by_trust) -> bool:
    """True only when B wins at m=-1 and A wins at m=+1."""

    try:
        b_trust_margin = float(margins_by_trust[-1.0])
        a_trust_margin = float(margins_by_trust[1.0])
    except KeyError as exc:
        raise ValueError("winner-flip gate requires trust endpoints -1 and +1") from exc
    return b_trust_margin < 0.0 and a_trust_margin > 0.0
