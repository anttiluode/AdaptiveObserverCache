"""Gate 8: history-guided active probe selection.

Gate 7 bought all current behavioral probes. Gate 8 makes each probe cost one
unit. Historical source fingerprints tell the observer which measurements can
possibly distinguish the two candidate identities.

The final memory read remains the frozen Gate-2/3 positional attention reader.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Dict, Iterable, Tuple

from gate5_generic_binding import TokenKey
from gate6_relational_alias import (
    AliasProjection,
    build_alias_suite,
    canonical_key,
    read_slot,
)


Signature = Tuple[int, ...]
PROBE_COUNT = 8


@dataclass(frozen=True)
class ProbeHistory:
    signatures: Dict[TokenKey, Signature]


@dataclass(frozen=True)
class CurrentProbeField:
    # slot -> full latent response signature; the observer may reveal entries
    # only by purchasing individual probe indices.
    slot_signatures: Tuple[Signature, Signature]
    corrupted_probe: int


@dataclass(frozen=True)
class ProbeTrace:
    chosen_slot: int
    probes: Tuple[int, ...]
    observations: Tuple[Tuple[int, int, int], ...]
    cost: int


def family_index(projection: AliasProjection) -> int:
    names = ("vault", "garden", "station", "weather", "package", "river")
    return names.index(projection.case.family.name)


def informative_probes(projection: AliasProjection) -> Tuple[int, ...]:
    """Three family-specific probes distinguish the two historical sources."""

    idx = family_index(projection)
    # Three unique positions whose location changes by family, preventing one
    # globally privileged probe from solving every case.
    return tuple(sorted((idx % 8, (idx + 3) % 8, (idx + 6) % 8)))


def historical_signature(
    projection: AliasProjection,
    canonical: str,
) -> Signature:
    family = projection.case.family
    idx = family_index(projection)

    # Shared background behavior.
    base = tuple((idx + 2 * probe + (probe // 2)) % 2 for probe in range(8))
    informative = set(informative_probes(projection))

    if canonical == family.canonical0:
        return base
    if canonical == family.canonical1:
        return tuple(
            (1 - value) if probe in informative else value
            for probe, value in enumerate(base)
        )
    raise ValueError(f"unknown canonical source {canonical}")


def build_probe_history(tokenizer, projections) -> ProbeHistory:
    signatures: Dict[TokenKey, Signature] = {}
    seen = set()
    for projection in projections:
        family = projection.case.family
        if family.name in seen:
            continue
        seen.add(family.name)
        for canonical in (family.canonical0, family.canonical1):
            signatures[canonical_key(tokenizer, canonical)] = historical_signature(
                projection, canonical
            )
    return ProbeHistory(signatures=signatures)


def current_probe_field(projection: AliasProjection) -> CurrentProbeField:
    """Current anonymous behavior with one slot-based corrupted probe.

    The corruption contains no trust information. At one informative probe,
    slot 1 is forced to emit slot 0's response, making that measurement
    ambiguous. Other informative probes preserve identity continuity.
    """

    slot_signatures = [
        list(historical_signature(projection, canonical))
        for canonical, _alias, _text in projection.case.records
    ]
    informative = informative_probes(projection)
    idx = family_index(projection)
    order_bit = 1 if projection.case.order == "10" else 0
    corrupted = informative[(idx + order_bit) % len(informative)]

    slot_signatures[1][corrupted] = slot_signatures[0][corrupted]
    return CurrentProbeField(
        slot_signatures=(
            tuple(slot_signatures[0]),
            tuple(slot_signatures[1]),
        ),
        corrupted_probe=corrupted,
    )


def reveal_probe(field: CurrentProbeField, probe: int) -> Tuple[int, int]:
    if probe < 0 or probe >= PROBE_COUNT:
        raise ValueError("probe index out of range")
    return (
        field.slot_signatures[0][probe],
        field.slot_signatures[1][probe],
    )


def _slot_from_observation(
    target_signature: Signature,
    probe: int,
    response0: int,
    response1: int,
) -> int | None:
    if response0 == response1:
        return None

    expected = target_signature[probe]
    matches = [
        slot
        for slot, value in enumerate((response0, response1))
        if value == expected
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def active_bind(
    persistent_key: TokenKey,
    pair_keys: Tuple[TokenKey, TokenKey],
    history: ProbeHistory,
    field: CurrentProbeField,
    projection: AliasProjection,
) -> ProbeTrace:
    """Buy only probes historical data predicts can separate the identities."""

    target = history.signatures[persistent_key]
    left = history.signatures[pair_keys[0]]
    right = history.signatures[pair_keys[1]]

    candidates = tuple(
        probe
        for probe in range(PROBE_COUNT)
        if left[probe] != right[probe]
    )
    if not candidates:
        raise RuntimeError("history predicts no informative probes")

    observations = []
    for probe in candidates:
        response0, response1 = reveal_probe(field, probe)
        observations.append((probe, response0, response1))
        slot = _slot_from_observation(
            target, probe, response0, response1
        )
        if slot is not None:
            return ProbeTrace(
                chosen_slot=slot,
                probes=tuple(x[0] for x in observations),
                observations=tuple(observations),
                cost=len(observations),
            )
    raise RuntimeError("active probe budget exhausted without identification")


def bind_with_order(
    persistent_key: TokenKey,
    history: ProbeHistory,
    field: CurrentProbeField,
    order: Iterable[int],
) -> ProbeTrace:
    """Matched baseline: same history and stopping rule, different probe order."""

    target = history.signatures[persistent_key]
    observations = []
    for probe in order:
        response0, response1 = reveal_probe(field, probe)
        observations.append((probe, response0, response1))
        slot = _slot_from_observation(
            target, probe, response0, response1
        )
        if slot is not None:
            return ProbeTrace(
                chosen_slot=slot,
                probes=tuple(x[0] for x in observations),
                observations=tuple(observations),
                cost=len(observations),
            )
    raise RuntimeError("probe order exhausted without identification")


def exact_random_order_expected_cost(
    persistent_key: TokenKey,
    history: ProbeHistory,
    field: CurrentProbeField,
) -> float:
    """Exact expectation over all 8! uniformly random probe orders."""

    total_cost = 0
    count = 0
    for order in permutations(range(PROBE_COUNT)):
        trace = bind_with_order(
            persistent_key, history, field, order
        )
        total_cost += trace.cost
        count += 1
    return total_cost / count


def shuffled_pair_history(
    pair_keys: Tuple[TokenKey, TokenKey],
    history: ProbeHistory,
) -> ProbeHistory:
    copied = dict(history.signatures)
    left, right = pair_keys
    copied[left], copied[right] = copied[right], copied[left]
    return ProbeHistory(signatures=copied)


def build_active_probe_suite():
    base, tokenizer, projections = build_alias_suite()
    history = build_probe_history(tokenizer, projections)
    return base, tokenizer, projections, history
