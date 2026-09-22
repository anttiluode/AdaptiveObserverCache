"""Gate 7: recover missing identity binding with one active audit round.

Gate 6 assumed an alias-relation graph. Gate 7 removes the terminal identity
relation. Two historical canonical identities and two current aliases remain,
but their permutation is unknown.

Historical audit receipts contain each canonical source's binary response to a
small family of challenge types. Under a strict one-round budget, the active
binder chooses a challenge that separates the two candidate identities, sends
that same challenge to both current aliases, and matches the returned bits to
history.

The pretrained attention reader remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Dict, Optional, Tuple

from gate6_relational_alias import AliasProjection, build_alias_suite


PROBES = tuple(f"audit-{index}" for index in range(8))


@dataclass(frozen=True)
class AuditHistory:
    profiles: Dict[str, Tuple[int, ...]]


@dataclass(frozen=True)
class ProbeObservation:
    probe: str
    responses_by_slot: Tuple[int, int]


def source_behavior(canonical: str, probe: str) -> int:
    """Hidden environment behavior, stable across aliases.

    The deterministic hash avoids hand-tuning pair-specific answers. The
    binder never receives the canonical identity of a current alias; it sees
    only the returned response bit.
    """

    payload = (canonical + "|" + probe).encode("utf-8")
    return sha256(payload).digest()[0] & 1


def build_history(projections: Tuple[AliasProjection, ...]) -> AuditHistory:
    canonicals = {
        canonical
        for projection in projections
        for canonical in (
            projection.case.family.canonical0,
            projection.case.family.canonical1,
        )
    }
    profiles = {
        canonical: tuple(
            source_behavior(canonical, probe) for probe in PROBES
        )
        for canonical in sorted(canonicals)
    }
    return AuditHistory(profiles=profiles)


def profile_bit(history: AuditHistory, canonical: str, probe: str) -> int:
    return history.profiles[canonical][PROBES.index(probe)]


def probe_separates(
    history: AuditHistory,
    canonical0: str,
    canonical1: str,
    probe: str,
) -> bool:
    return (
        profile_bit(history, canonical0, probe)
        != profile_bit(history, canonical1, probe)
    )


def select_active_probe(
    history: AuditHistory,
    canonical0: str,
    canonical1: str,
) -> str:
    """Choose the first maximally informative one-bit challenge.

    With two candidates, a challenge has one bit of identification value iff
    their historical responses differ, otherwise zero.
    """

    scored = [
        (
            int(probe_separates(history, canonical0, canonical1, probe)),
            -index,
            probe,
        )
        for index, probe in enumerate(PROBES)
    ]
    score, _negative_index, probe = max(scored)
    if score <= 0:
        raise RuntimeError(
            f"no one-round audit separates {canonical0} from {canonical1}"
        )
    return probe


def best_fixed_probe(
    history: AuditHistory,
    canonical_pairs: Tuple[Tuple[str, str], ...],
) -> Tuple[str, int]:
    """Strongest global fixed challenge under the same one-round budget."""

    scored = []
    for index, probe in enumerate(PROBES):
        separable = sum(
            int(probe_separates(history, left, right, probe))
            for left, right in canonical_pairs
        )
        scored.append((separable, -index, probe))
    separable, _negative_index, probe = max(scored)
    return probe, separable


def audit_current_aliases(
    projection: AliasProjection,
    probe: str,
) -> ProbeObservation:
    """Environment action: one challenge round, one bit from each source."""

    responses = tuple(
        source_behavior(canonical, probe)
        for canonical, _alias, _text in projection.case.records
    )
    if len(responses) != 2:
        raise RuntimeError("Gate 7 requires exactly two current sources")
    return ProbeObservation(
        probe=probe,
        responses_by_slot=(int(responses[0]), int(responses[1])),
    )


def infer_mapping(
    history: AuditHistory,
    canonical0: str,
    canonical1: str,
    observation: ProbeObservation,
    swap_history_labels: bool = False,
) -> Optional[Dict[str, int]]:
    """Infer canonical -> current slot from one challenge observation."""

    expected0 = profile_bit(history, canonical0, observation.probe)
    expected1 = profile_bit(history, canonical1, observation.probe)

    if swap_history_labels:
        expected0, expected1 = expected1, expected0

    if expected0 == expected1:
        return None

    slot_for_0 = [
        slot
        for slot, response in enumerate(observation.responses_by_slot)
        if response == expected0
    ]
    slot_for_1 = [
        slot
        for slot, response in enumerate(observation.responses_by_slot)
        if response == expected1
    ]

    if len(slot_for_0) != 1 or len(slot_for_1) != 1:
        return None
    if slot_for_0[0] == slot_for_1[0]:
        return None

    return {
        canonical0: slot_for_0[0],
        canonical1: slot_for_1[0],
    }


def static_slot_mapping(canonical0: str, canonical1: str) -> Dict[str, int]:
    return {canonical0: 0, canonical1: 1}


def build_active_identity_suite():
    base, tokenizer, projections = build_alias_suite()
    history = build_history(projections)
    pairs = tuple(
        (p.case.family.canonical0, p.case.family.canonical1)
        for p in projections[::2]
    )
    fixed_probe, fixed_separable_pairs = best_fixed_probe(history, pairs)
    return (
        base,
        tokenizer,
        projections,
        history,
        fixed_probe,
        fixed_separable_pairs,
    )
