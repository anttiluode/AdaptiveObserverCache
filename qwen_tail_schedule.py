"""Dependency-free helpers for the phasic-vs-tonic observer tail test.

Index convention: ``index`` is the candidate-token position whose logits are
being produced. The forward pass over the question suffix produces logits for
index 0; appending candidate token ``i-1`` produces logits for index ``i``.
The trust that is active during that forward pass is the trust "for" index i.
"""

from __future__ import annotations

ARMS = ("tonic", "phasic", "neutral")

# Frozen before the run. bf16 scores move in quarter-nat steps, so 0.5 nats
# is two quanta: the smallest difference this instrument can call real.
TAIL_EFFECT_THRESHOLD = 0.5


def trust_for_index(arm: str, index: int, decision_index: int, trust: float) -> float:
    """Observer trust active while producing logits for candidate token ``index``."""

    if arm == "neutral":
        return 0.0
    if arm == "tonic":
        return float(trust)
    if arm == "phasic":
        return float(trust) if index <= decision_index else 0.0
    raise ValueError(f"unknown arm: {arm!r}")


def split_logprobs(logprobs, decision_index: int):
    """Split one candidate's per-token log-probs around the decision token."""

    lps = [float(x) for x in logprobs]
    if not 0 <= decision_index < len(lps):
        raise ValueError("decision_index outside candidate")
    return {
        "prefix": sum(lps[:decision_index]),
        "decision": lps[decision_index],
        "tail": sum(lps[decision_index + 1 :]),
        "total": sum(lps),
    }


def winner(sum_a: float, sum_b: float) -> str:
    margin = float(sum_a) - float(sum_b)
    if margin > 0.0:
        return "A"
    if margin < 0.0:
        return "B"
    return "tie"


def classify(near: dict, far: dict, *, threshold: float = TAIL_EFFECT_THRESHOLD):
    """Pre-registered reading of the tail test.

    ``near`` / ``far`` map arm -> {"A": split, "B": split} for distance 0 and
    the far checkpoint, all scored at the same nonzero observer trust.

    Verdicts (first match wins):
      NO_BASELINE_CONTROL      tonic does not make B win at distance 0, so the
                               far checkpoint is uninterpretable.
      PHASIC_RESCUES           far: tonic -> A, phasic -> B. Releasing the
                               observer after the decision restores the answer.
      OBSERVER_DAMAGES_TAIL    far: phasic B tail is >= threshold better than
                               tonic B tail, but not enough to flip the winner.
      DISTANCE_DAMAGES_TAIL    far: observer on/off makes no real tail
                               difference, and the neutral B tail itself got
                               >= threshold worse from distance 0 to far.
      UNRESOLVED               none of the above.
    """

    def w(block):
        return winner(block["A"]["total"], block["B"]["total"])

    observer_tail_damage = far["phasic"]["B"]["tail"] - far["tonic"]["B"]["tail"]
    distance_tail_cost = near["neutral"]["B"]["tail"] - far["neutral"]["B"]["tail"]
    decision_match = abs(
        far["phasic"]["B"]["decision"] - far["tonic"]["B"]["decision"]
    )

    readout = {
        "near_winner": {arm: w(near[arm]) for arm in ARMS},
        "far_winner": {arm: w(far[arm]) for arm in ARMS},
        "observer_tail_damage_nats": observer_tail_damage,
        "distance_tail_cost_nats": distance_tail_cost,
        "far_decision_logprob_mismatch_tonic_vs_phasic": decision_match,
        "threshold_nats": threshold,
    }

    if readout["near_winner"]["tonic"] != "B":
        verdict = "NO_BASELINE_CONTROL"
    elif readout["far_winner"]["tonic"] == "A" and readout["far_winner"]["phasic"] == "B":
        verdict = "PHASIC_RESCUES"
    elif observer_tail_damage >= threshold:
        verdict = "OBSERVER_DAMAGES_TAIL"
    elif distance_tail_cost >= threshold:
        verdict = "DISTANCE_DAMAGES_TAIL"
    else:
        verdict = "UNRESOLVED"
    readout["verdict"] = verdict
    return readout
