"""Gate 6: representation translation for provenance alias drift.

Gate 5 carried an arbitrary persistent provenance key, but exact equality
failed as soon as the current record used a different surface key. Gate 6
adds a generic alias/equivalence receipt layer:

persistent original key
        -> alias equivalence graph
        -> current record key
        -> current slot
        -> frozen positional read mode

The alias receipts describe both sources symmetrically and never contain
trust labels.  The slow trusted-key state itself remains the original
calibration key.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, Iterable, Tuple

import torch
from torch import Tensor

from gate3_transfer import _split_heads, load_model
from gate5_generic_binding import (
    KeyedFamily,
    families as gate5_families,
    slot_mode,
    tokenize_key,
)
from pretrained_observer import NaturalCache, build_fixture


TokenKey = Tuple[int, ...]
AliasEdge = Tuple[TokenKey, TokenKey]


@dataclass(frozen=True)
class AliasPath:
    original: str
    alias1: str
    alias2: str


@dataclass(frozen=True)
class AliasCase:
    name: str
    order: str
    generation: int
    family: KeyedFamily
    records: Tuple[Tuple[str, str, str], ...]
    # Each record is (original identity, current alias, rendered text).


@dataclass(frozen=True)
class AliasProjection:
    case: AliasCase
    prompt: str
    query_base: Tensor
    query_norm: float
    cache: NaturalCache
    record_spans: Tuple[Tuple[int, int], ...]
    record_keys: Tuple[TokenKey, ...]
    identity_spans: Dict[str, Tuple[int, int]]


@dataclass(frozen=True)
class AliasRegistry:
    edges: Tuple[AliasEdge, ...]


@dataclass(frozen=True)
class AliasRead:
    target_identity: str
    chosen_slot: int
    observer: float
    target_mass: float
    target_share: float
    prediction: str
    output: Tensor


_ALIAS_PATHS = (
    AliasPath("Carol", "Clara", "Cora"),
    AliasPath("Dave", "Dean", "Drew"),
    AliasPath("Eve", "Eva", "Erin"),
    AliasPath("Frank", "Fred", "Felix"),
    AliasPath("Grace", "Greta", "Gwen"),
    AliasPath("Henry", "Harry", "Hugo"),
    AliasPath("Iris", "Ivy", "Ida"),
    AliasPath("Jack", "Jake", "Joel"),
    AliasPath("Kira", "Kara", "Kim"),
    AliasPath("Liam", "Leo", "Luke"),
    AliasPath("Mona", "Mina", "Maya"),
    AliasPath("Nate", "Noah", "Neil"),
)


def alias_paths() -> Dict[str, AliasPath]:
    return {path.original: path for path in _ALIAS_PATHS}


def _replace_record_prefix(text: str, original: str, alias: str) -> str:
    prefix = " " + original
    if not text.startswith(prefix):
        raise RuntimeError(
            f"record does not begin with expected provenance {original}: {text!r}"
        )
    return " " + alias + text[len(prefix) :]


def _alias_for(original: str, generation: int) -> str:
    path = alias_paths()[original]
    if generation == 1:
        return path.alias1
    if generation == 2:
        return path.alias2
    raise ValueError("generation must be 1 or 2")


def cases() -> Tuple[AliasCase, ...]:
    out = []
    for family in gate5_families():
        for generation in (1, 2):
            a0 = _alias_for(family.key0, generation)
            a1 = _alias_for(family.key1, generation)
            r0 = (
                family.key0,
                a0,
                _replace_record_prefix(family.text0, family.key0, a0),
            )
            r1 = (
                family.key1,
                a1,
                _replace_record_prefix(family.text1, family.key1, a1),
            )
            out.append(
                AliasCase(
                    name=f"{family.name}-g{generation}-01",
                    order="01",
                    generation=generation,
                    family=family,
                    records=(r0, r1),
                )
            )
            out.append(
                AliasCase(
                    name=f"{family.name}-g{generation}-10",
                    order="10",
                    generation=generation,
                    family=family,
                    records=(r1, r0),
                )
            )
    return tuple(out)


def _normal_alias_edges(
    tokenizer,
    family: KeyedFamily,
    generation: int,
    *,
    truncate_to_one_hop: bool = False,
) -> Tuple[AliasEdge, ...]:
    edges = []
    for original in (family.key0, family.key1):
        path = alias_paths()[original]
        k0 = tokenize_key(tokenizer, original)
        k1 = tokenize_key(tokenizer, path.alias1)
        edges.append((k0, k1))
        if generation == 2 and not truncate_to_one_hop:
            k2 = tokenize_key(tokenizer, path.alias2)
            edges.append((k1, k2))
    return tuple(edges)


def build_alias_registry(
    tokenizer,
    family: KeyedFamily,
    generation: int,
    *,
    wrong: bool = False,
    truncate_to_one_hop: bool = False,
) -> AliasRegistry:
    """Build generic representation-update receipts.

    Correct receipts map both sources symmetrically. Wrong receipts cross the
    two alias paths, preserving the same amount of metadata while binding each
    original key to the other source's current representation.
    """

    if not wrong:
        return AliasRegistry(
            edges=_normal_alias_edges(
                tokenizer,
                family,
                generation,
                truncate_to_one_hop=truncate_to_one_hop,
            )
        )

    p0 = alias_paths()[family.key0]
    p1 = alias_paths()[family.key1]
    o0 = tokenize_key(tokenizer, family.key0)
    o1 = tokenize_key(tokenizer, family.key1)
    a01 = tokenize_key(tokenizer, p0.alias1)
    a11 = tokenize_key(tokenizer, p1.alias1)

    edges = [(o0, a11), (o1, a01)]
    if generation == 2 and not truncate_to_one_hop:
        a02 = tokenize_key(tokenizer, p0.alias2)
        a12 = tokenize_key(tokenizer, p1.alias2)
        # Continue the crossed paths so the wrong registry still resolves
        # cleanly; it simply resolves to the wrong current source.
        edges.extend(((a11, a12), (a01, a02)))
    return AliasRegistry(edges=tuple(edges))


def _component(seed: TokenKey, edges: Iterable[AliasEdge]) -> set[TokenKey]:
    adjacency: Dict[TokenKey, set[TokenKey]] = {}
    for left, right in edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)

    seen = {seed}
    stack = [seed]
    while stack:
        current = stack.pop()
        for nxt in adjacency.get(current, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def bind_alias_key(
    persistent_key: TokenKey,
    projection: AliasProjection,
    registry: AliasRegistry,
) -> int:
    """Resolve an old persistent key to the unique current record slot."""

    equivalent = _component(persistent_key, registry.edges)
    matches = [
        slot
        for slot, current_key in enumerate(projection.record_keys)
        if current_key in equivalent
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"alias key matched {len(matches)} current records; expected one"
        )
    return matches[0]


def exact_key_slot(
    persistent_key: TokenKey,
    projection: AliasProjection,
) -> int | None:
    matches = [
        slot
        for slot, current_key in enumerate(projection.record_keys)
        if current_key == persistent_key
    ]
    if len(matches) == 1:
        return matches[0]
    return None


@torch.no_grad()
def project_case(model, tokenizer, base, case: AliasCase) -> AliasProjection:
    ids = []
    record_spans = []
    record_keys = []
    identity_spans = {}

    for original, alias, text in case.records:
        piece = tokenizer.encode(text, add_special_tokens=False)
        alias_tokens = tokenize_key(tokenizer, alias)
        if tuple(int(x) for x in piece[: len(alias_tokens)]) != alias_tokens:
            raise RuntimeError(
                f"record for alias {alias} does not begin with its provenance key"
            )

        start = len(ids)
        ids.extend(piece)
        span = (start, len(ids))
        record_spans.append(span)
        record_keys.append(alias_tokens)
        identity_spans[original] = span

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
        source_a=identity_spans[case.family.key0],
        source_b=identity_spans[case.family.key1],
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
        record_keys=tuple(record_keys),
        identity_spans=identity_spans,
    )


@torch.no_grad()
def read_slot(
    base,
    projection: AliasProjection,
    target_identity: str,
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
    s0 = projection.identity_spans[family.key0]
    s1 = projection.identity_spans[family.key1]
    mass0 = float(weights[s0[0] : s0[1]].sum())
    mass1 = float(weights[s1[0] : s1[1]].sum())
    total = mass0 + mass1
    share0 = mass0 / total if total > 0.0 else 0.5

    if target_identity == family.key0:
        target_mass = mass0
        target_share = share0
    elif target_identity == family.key1:
        target_mass = mass1
        target_share = 1.0 - share0
    else:
        raise ValueError("target identity is not in this family")

    prediction = family.key0 if share0 >= 0.5 else family.key1
    return AliasRead(
        target_identity=target_identity,
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
