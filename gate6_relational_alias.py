"""Gate 6: persistent canonical provenance through multi-hop alias drift.

The trusted slow state stores a canonical provenance key learned during an
earlier calibration. Test records no longer expose that key. They expose a new
surface alias. A separate, trust-independent relation graph connects canonical
keys to current aliases through multi-hop provenance relations.

The attention reader remains the frozen Gate-2/3 source-slot operator.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import sqrt
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from gate3_transfer import _split_heads, load_model
from gate5_generic_binding import TokenKey, tokenize_key
from pretrained_observer import NaturalCache, build_fixture


Edge = Tuple[TokenKey, TokenKey]


@dataclass(frozen=True)
class AliasFamily:
    name: str
    canonical0: str
    canonical1: str
    middle0: str
    middle1: str
    alias0: str
    alias1: str
    text0: str
    text1: str
    query: str


@dataclass(frozen=True)
class AliasCase:
    name: str
    order: str
    family: AliasFamily
    records: Tuple[Tuple[str, str, str], ...]


@dataclass(frozen=True)
class AliasProjection:
    case: AliasCase
    prompt: str
    query_base: Tensor
    query_norm: float
    cache: NaturalCache
    record_spans: Tuple[Tuple[int, int], ...]
    record_alias_keys: Tuple[TokenKey, ...]
    canonical_spans: Dict[str, Tuple[int, int]]


@dataclass(frozen=True)
class AliasRead:
    target_canonical: str
    chosen_slot: int
    observer: float
    target_mass: float
    target_share: float
    prediction: str
    output: Tensor


def families() -> Tuple[AliasFamily, ...]:
    return (
        AliasFamily(
            "vault",
            "Carol", "Dave",
            "CSeven", "DThree",
            "Orion", "Delta",
            " Orion reports the vault color red.",
            " Delta reports the vault color blue.",
            " Which report should be trusted for the vault color?",
        ),
        AliasFamily(
            "garden",
            "Eve", "Frank",
            "EFour", "FNine",
            "Juniper", "Quartz",
            " Juniper reports the garden animal fox.",
            " Quartz reports the garden animal owl.",
            " Which report should be trusted for the garden animal?",
        ),
        AliasFamily(
            "station",
            "Grace", "Henry",
            "GTwo", "HEight",
            "Cedar", "Ember",
            " Cedar reports the station number seven.",
            " Ember reports the station number nine.",
            " Which report should be trusted for the station number?",
        ),
        AliasFamily(
            "weather",
            "Iris", "Jack",
            "IFive", "JSix",
            "Lyra", "Vega",
            " Lyra reports the weather today sunny.",
            " Vega reports the weather today cloudy.",
            " Which report should be trusted for the weather today?",
        ),
        AliasFamily(
            "package",
            "Kira", "Liam",
            "KFour", "LTwo",
            "Nova", "Atlas",
            " Nova says the package is heavy.",
            " Atlas says the package is light.",
            " Which report should be trusted about the package?",
        ),
        AliasFamily(
            "river",
            "Mona", "Nate",
            "MThree", "NSeven",
            "Solace", "Harbor",
            " Solace records river level high.",
            " Harbor records river level low.",
            " Which report should be trusted for river level?",
        ),
    )


def cases() -> Tuple[AliasCase, ...]:
    out = []
    for family in families():
        rec0 = (family.canonical0, family.alias0, family.text0)
        rec1 = (family.canonical1, family.alias1, family.text1)
        out.append(
            AliasCase(
                name=f"{family.name}-01",
                order="01",
                family=family,
                records=(rec0, rec1),
            )
        )
        out.append(
            AliasCase(
                name=f"{family.name}-10",
                order="10",
                family=family,
                records=(rec1, rec0),
            )
        )
    return tuple(out)


def canonical_key(tokenizer, canonical: str) -> TokenKey:
    """Calibration state: a canonical provenance key copied earlier."""

    return tokenize_key(tokenizer, canonical)


def relation_edges(
    tokenizer,
    family: AliasFamily,
    shuffled: bool = False,
) -> Tuple[Edge, ...]:
    """Trust-independent identity relation metadata for the current epoch."""

    c0 = tokenize_key(tokenizer, family.canonical0)
    c1 = tokenize_key(tokenizer, family.canonical1)
    m0 = tokenize_key(tokenizer, family.middle0)
    m1 = tokenize_key(tokenizer, family.middle1)
    a0 = tokenize_key(tokenizer, family.alias0)
    a1 = tokenize_key(tokenizer, family.alias1)

    # No direct canonical->current-alias edges exist. A one-hop binder cannot
    # solve this gate. The shuffled attacker preserves graph shape while
    # crossing the terminal aliases between source components.
    if shuffled:
        return (
            (c0, m0),
            (m0, a1),
            (c1, m1),
            (m1, a0),
        )
    return (
        (c0, m0),
        (m0, a0),
        (c1, m1),
        (m1, a1),
    )


def _adjacency(edges: Tuple[Edge, ...]):
    graph = {}
    for left, right in edges:
        graph.setdefault(left, set()).add(right)
        graph.setdefault(right, set()).add(left)
    return graph


def shortest_path_length(
    start: TokenKey,
    goal: TokenKey,
    edges: Tuple[Edge, ...],
) -> Optional[int]:
    if start == goal:
        return 0

    graph = _adjacency(edges)
    queue = deque([(start, 0)])
    seen = {start}

    while queue:
        node, distance = queue.popleft()
        for nxt in graph.get(node, ()):
            if nxt == goal:
                return distance + 1
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, distance + 1))
    return None


def bind_via_relation_graph(
    state_key: TokenKey,
    projection: AliasProjection,
    edges: Tuple[Edge, ...],
) -> Tuple[int, int]:
    """Bind canonical slow state to one current record alias by connectivity."""

    reachable = []
    for slot, alias_key in enumerate(projection.record_alias_keys):
        distance = shortest_path_length(state_key, alias_key, edges)
        if distance is not None:
            reachable.append((slot, distance))

    if len(reachable) != 1:
        raise RuntimeError(
            f"relation binder reached {len(reachable)} current records"
        )
    return reachable[0]


def bind_exact_key(
    state_key: TokenKey,
    projection: AliasProjection,
) -> Optional[int]:
    matches = [
        slot
        for slot, alias_key in enumerate(projection.record_alias_keys)
        if alias_key == state_key
    ]
    return matches[0] if len(matches) == 1 else None


def bind_one_hop(
    state_key: TokenKey,
    projection: AliasProjection,
    edges: Tuple[Edge, ...],
) -> Optional[int]:
    graph = _adjacency(edges)
    neighbors = graph.get(state_key, set())
    matches = [
        slot
        for slot, alias_key in enumerate(projection.record_alias_keys)
        if alias_key in neighbors
    ]
    return matches[0] if len(matches) == 1 else None


def slot_mode(base, slot: int) -> float:
    if slot == 0:
        return float(base.mode_a.state)
    if slot == 1:
        return float(base.mode_b.state)
    raise ValueError("Gate 6 supports exactly two source slots")


@torch.no_grad()
def project_case(model, tokenizer, base, case: AliasCase) -> AliasProjection:
    ids = []
    record_spans = []
    alias_keys = []
    canonical_spans = {}

    for canonical, alias, text in case.records:
        piece = tokenizer.encode(text, add_special_tokens=False)
        alias_key = tokenize_key(tokenizer, alias)

        if tuple(int(x) for x in piece[: len(alias_key)]) != alias_key:
            raise RuntimeError(
                f"record for alias {alias} does not begin with its alias key"
            )

        start = len(ids)
        ids.extend(piece)
        span = (start, len(ids))
        record_spans.append(span)
        alias_keys.append(alias_key)
        canonical_spans[canonical] = span

    ids.extend(tokenizer.encode(case.family.query, add_special_tokens=False))
    input_ids = torch.tensor([ids], dtype=torch.long)

    outputs = model(
        input_ids=input_ids,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    block = model.transformer.h[base.layer]
    hidden_in = outputs.hidden_states[base.layer][0]
    attn_in = block.ln_1(hidden_in)
    qkv = block.attn.c_attn(attn_in)

    n_embd = int(model.config.n_embd)
    n_head = int(model.config.n_head)
    q_all, k_all, v_all = qkv.split(n_embd, dim=-1)
    q_heads = _split_heads(q_all, n_head)
    k_heads = _split_heads(k_all, n_head)
    v_heads = _split_heads(v_all, n_head)

    query_base = q_heads[-1, base.head, :].detach().clone()
    keys = k_heads[:, base.head, :].detach().clone()
    values = v_heads[:, base.head, :].detach().clone()

    cache = NaturalCache(
        keys=keys,
        values=values,
        source_a=canonical_spans[case.family.canonical0],
        source_b=canonical_spans[case.family.canonical1],
        input_ids=tuple(int(x) for x in ids),
    )

    return AliasProjection(
        case=case,
        prompt="".join(text for _, _, text in case.records)
        + case.family.query,
        query_base=query_base,
        query_norm=float(torch.linalg.vector_norm(query_base)),
        cache=cache,
        record_spans=tuple(record_spans),
        record_alias_keys=tuple(alias_keys),
        canonical_spans=canonical_spans,
    )


@torch.no_grad()
def read_slot(
    base,
    projection: AliasProjection,
    target_canonical: str,
    slot: int,
) -> AliasRead:
    observer = slot_mode(base, slot)
    query = (
        projection.query_base
        + observer
        * projection.query_norm
        * base.observer_direction
    )
    scores = projection.cache.keys @ query / sqrt(base.head_dim)
    weights = torch.softmax(scores, dim=0)
    output = weights @ projection.cache.values

    family = projection.case.family
    s0 = projection.canonical_spans[family.canonical0]
    s1 = projection.canonical_spans[family.canonical1]
    mass0 = float(weights[s0[0] : s0[1]].sum())
    mass1 = float(weights[s1[0] : s1[1]].sum())
    total = mass0 + mass1
    share0 = mass0 / total if total > 0.0 else 0.5

    if target_canonical == family.canonical0:
        target_mass = mass0
        target_share = share0
    elif target_canonical == family.canonical1:
        target_mass = mass1
        target_share = 1.0 - share0
    else:
        raise ValueError(f"unknown canonical source {target_canonical}")

    prediction = family.canonical0 if share0 >= 0.5 else family.canonical1

    return AliasRead(
        target_canonical=target_canonical,
        chosen_slot=slot,
        observer=observer,
        target_mass=target_mass,
        target_share=target_share,
        prediction=prediction,
        output=output.detach().clone(),
    )


def build_alias_suite():
    torch.manual_seed(0)
    torch.set_num_threads(2)

    base = build_fixture()
    model, tokenizer = load_model()
    projections = tuple(
        project_case(model, tokenizer, base, case) for case in cases()
    )
    return base, tokenizer, projections
