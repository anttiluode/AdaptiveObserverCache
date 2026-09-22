"""Gate 7: infer current provenance binding from historical behavior.

Gate 6 was given a complete alias relation graph. Gate 7 removes the terminal
identity relation. The slow trusted state still carries an old canonical
provenance key, but current records expose unrelated aliases.

A separate historical memory contains a small behavioral fingerprint observed
for every canonical source in an earlier epoch. At the current epoch the two
anonymous source slots can be probed with the same fixed probe battery. A
generic nearest-history matcher binds the persistent trusted source to the
current slot.

The final retrieval still uses the frozen Gate-2/3 positional attention reader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from gate5_generic_binding import TokenKey
from gate6_relational_alias import (
    AliasProjection,
    bind_via_relation_graph,
    build_alias_suite,
    canonical_key,
    read_slot,
    relation_edges,
)


Fingerprint = Tuple[int, ...]


@dataclass(frozen=True)
class HistoryMemory:
    fingerprints: Dict[TokenKey, Fingerprint]


@dataclass(frozen=True)
class ProbeObservation:
    slot: int
    fingerprint: Fingerprint


@dataclass(frozen=True)
class HistoryBinding:
    chosen_slot: int
    best_distance: int
    second_distance: int
    margin: int


def _family_index(projection: AliasProjection) -> int:
    names = ("vault", "garden", "station", "weather", "package", "river")
    return names.index(projection.case.family.name)


def canonical_fingerprint(
    projection: AliasProjection,
    canonical: str,
) -> Fingerprint:
    """Prior observed behavior, independent of current alias and record order.

    Both sources in a family deliberately share probe 0. The remaining four
    probes form complementary codewords, so one current-probe corruption is
    recoverable but one observation alone is not.
    """

    family = projection.case.family
    idx = _family_index(projection)
    common = idx % 3

    # Vary the code across families while preserving Hamming distance 4
    # between the two sources after the shared first coordinate.
    base = (
        (idx >> 0) & 1,
        (idx >> 1) & 1,
        (idx >> 2) & 1,
        ((idx + 1) >> 1) & 1,
    )
    if canonical == family.canonical0:
        tail = base
    elif canonical == family.canonical1:
        tail = tuple(1 - bit for bit in base)
    else:
        raise ValueError(f"unknown canonical source {canonical}")
    return (common, *tail)


def build_history_memory(
    tokenizer,
    projections: Tuple[AliasProjection, ...],
) -> HistoryMemory:
    """Store earlier source behavior under arbitrary canonical token keys."""

    fingerprints: Dict[TokenKey, Fingerprint] = {}
    # One projection per family is sufficient; behavior is source-persistent.
    seen_families = set()
    for projection in projections:
        family = projection.case.family
        if family.name in seen_families:
            continue
        seen_families.add(family.name)
        for canonical in (family.canonical0, family.canonical1):
            key = canonical_key(tokenizer, canonical)
            fingerprints[key] = canonical_fingerprint(projection, canonical)
    return HistoryMemory(fingerprints=fingerprints)


def _corrupt_one_probe(
    fingerprint: Fingerprint,
    *,
    family_index: int,
    slot: int,
    order: str,
) -> Fingerprint:
    """Current behavior drifts by exactly one non-prefix probe.

    The corruption location depends on current slot/order, not source identity.
    """

    values = list(fingerprint)
    offset = 1 + ((family_index + slot + (1 if order == "10" else 0)) % 4)
    values[offset] = 1 - values[offset]
    return tuple(values)


def observe_current_behavior(
    projection: AliasProjection,
) -> Tuple[ProbeObservation, ...]:
    """Environment-facing fixed five-probe battery for current anonymous slots."""

    idx = _family_index(projection)
    observations = []
    for slot, (canonical, _alias, _text) in enumerate(projection.case.records):
        history = canonical_fingerprint(projection, canonical)
        current = _corrupt_one_probe(
            history,
            family_index=idx,
            slot=slot,
            order=projection.case.order,
        )
        observations.append(
            ProbeObservation(slot=slot, fingerprint=current)
        )
    return tuple(observations)


def hamming(left: Fingerprint, right: Fingerprint) -> int:
    if len(left) != len(right):
        raise ValueError("fingerprints must have equal length")
    return sum(int(a != b) for a, b in zip(left, right))


def bind_from_history(
    persistent_key: TokenKey,
    history: HistoryMemory,
    observations: Tuple[ProbeObservation, ...],
    *,
    probe_count: int = 5,
) -> HistoryBinding:
    """Bind trusted historical behavior to one current source slot."""

    if persistent_key not in history.fingerprints:
        raise RuntimeError("persistent source has no historical fingerprint")
    if probe_count < 1 or probe_count > 5:
        raise ValueError("probe_count must be in [1,5]")

    target = history.fingerprints[persistent_key][:probe_count]
    scored = sorted(
        (
            hamming(target, obs.fingerprint[:probe_count]),
            obs.slot,
        )
        for obs in observations
    )
    if len(scored) != 2:
        raise RuntimeError("Gate 7 expects exactly two current source slots")

    best_distance, best_slot = scored[0]
    second_distance, _ = scored[1]
    if best_distance == second_distance:
        raise RuntimeError("behavioral evidence is ambiguous")

    return HistoryBinding(
        chosen_slot=best_slot,
        best_distance=best_distance,
        second_distance=second_distance,
        margin=second_distance - best_distance,
    )


def static_tie_slot(
    observations: Tuple[ProbeObservation, ...],
) -> int:
    """Deterministic fallback used by controls with no usable continuity state."""

    return min(obs.slot for obs in observations)


def incomplete_relation_bind(
    tokenizer,
    projection: AliasProjection,
    persistent_key: TokenKey,
) -> int | None:
    """Gate-6 graph attacker with terminal alias edges removed."""

    full = relation_edges(tokenizer, projection.case.family, shuffled=False)
    # Keep only canonical -> middle edges; terminal aliases become disconnected.
    incomplete = (full[0], full[2])
    try:
        slot, _distance = bind_via_relation_graph(
            persistent_key,
            projection,
            incomplete,
        )
        return slot
    except RuntimeError:
        return None


def build_history_suite():
    base, tokenizer, projections = build_alias_suite()
    history = build_history_memory(tokenizer, projections)
    return base, tokenizer, projections, history
