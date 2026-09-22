"""Gate 5: generic provenance-key binding with no hard-coded identity table.

The slow state carries an arbitrary provenance key captured during calibration.
At test time a generic equality binder maps that key to whichever current
source record carries it, then reuses the frozen Gate-2/3 slot reader.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, Tuple

import torch
from torch import Tensor

from gate3_transfer import _split_heads, load_model
from pretrained_observer import NaturalCache, build_fixture


TokenKey = Tuple[int, ...]


@dataclass(frozen=True)
class KeyedFamily:
    name: str
    key0: str
    key1: str
    text0: str
    text1: str
    query: str


@dataclass(frozen=True)
class KeyedCase:
    name: str
    order: str
    family: KeyedFamily
    records: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class GenericProjection:
    case: KeyedCase
    prompt: str
    query_base: Tensor
    query_norm: float
    cache: NaturalCache
    record_spans: Tuple[Tuple[int, int], ...]
    record_keys: Tuple[TokenKey, ...]
    key_spans: Dict[str, Tuple[int, int]]


@dataclass(frozen=True)
class GenericRead:
    target_key: str
    chosen_slot: int
    observer: float
    target_mass: float
    target_share: float
    prediction: str
    output: Tensor


def families() -> Tuple[KeyedFamily, ...]:
    return (
        KeyedFamily(
            "vault",
            "Carol",
            "Dave",
            " Carol reports the vault color red.",
            " Dave reports the vault color blue.",
            " Which report should be trusted for the vault color?",
        ),
        KeyedFamily(
            "garden",
            "Eve",
            "Frank",
            " Eve reports the garden animal fox.",
            " Frank reports the garden animal owl.",
            " Which report should be trusted for the garden animal?",
        ),
        KeyedFamily(
            "station",
            "Grace",
            "Henry",
            " Grace reports the station number seven.",
            " Henry reports the station number nine.",
            " Which report should be trusted for the station number?",
        ),
        KeyedFamily(
            "weather",
            "Iris",
            "Jack",
            " Iris reports the weather today sunny.",
            " Jack reports the weather today cloudy.",
            " Which report should be trusted for the weather today?",
        ),
        KeyedFamily(
            "package",
            "Kira",
            "Liam",
            " Kira says the package is heavy.",
            " Liam says the package is light.",
            " Which report should be trusted about the package?",
        ),
        KeyedFamily(
            "river",
            "Mona",
            "Nate",
            " Mona records river level high.",
            " Nate records river level low.",
            " Which report should be trusted for river level?",
        ),
    )


def cases() -> Tuple[KeyedCase, ...]:
    out = []
    for family in families():
        rec0 = (family.key0, family.text0)
        rec1 = (family.key1, family.text1)
        out.append(
            KeyedCase(
                name=f"{family.name}-01",
                order="01",
                family=family,
                records=(rec0, rec1),
            )
        )
        out.append(
            KeyedCase(
                name=f"{family.name}-10",
                order="10",
                family=family,
                records=(rec1, rec0),
            )
        )
    return tuple(out)


def tokenize_key(tokenizer, key: str) -> TokenKey:
    ids = tokenizer.encode(" " + key, add_special_tokens=False)
    if not ids:
        raise RuntimeError(f"empty provenance key for {key}")
    return tuple(int(x) for x in ids)


def capture_persistent_key(tokenizer, provenance: str) -> TokenKey:
    """Calibration action: copy an arbitrary source provenance key into state."""

    return tokenize_key(tokenizer, provenance)


def bind_persistent_key(
    state_key: TokenKey,
    projection: GenericProjection,
) -> int:
    """Generic equality binder. No identity table and no trust/KV inspection."""

    matches = [
        slot
        for slot, record_key in enumerate(projection.record_keys)
        if record_key == state_key
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"persistent key matched {len(matches)} current source records"
        )
    return matches[0]


def slot_mode(base, slot: int) -> float:
    if slot == 0:
        return float(base.mode_a.state)
    if slot == 1:
        return float(base.mode_b.state)
    raise ValueError("Gate 5 supports exactly two source slots")


@torch.no_grad()
def project_case(model, tokenizer, base, case: KeyedCase) -> GenericProjection:
    ids = []
    record_spans = []
    record_keys = []
    key_spans = {}

    for key, text in case.records:
        piece = tokenizer.encode(text, add_special_tokens=False)
        key_tokens = tokenize_key(tokenizer, key)

        # The provenance field must actually be the record prefix; the generic
        # binder is metadata-equivalent, not a hidden side channel.
        if tuple(int(x) for x in piece[: len(key_tokens)]) != key_tokens:
            raise RuntimeError(
                f"record for {key} does not begin with its provenance key"
            )

        start = len(ids)
        ids.extend(piece)
        span = (start, len(ids))
        record_spans.append(span)
        record_keys.append(key_tokens)
        key_spans[key] = span

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
        source_a=key_spans[case.family.key0],
        source_b=key_spans[case.family.key1],
        input_ids=tuple(int(x) for x in ids),
    )

    return GenericProjection(
        case=case,
        prompt="".join(text for _, text in case.records)
        + case.family.query,
        query_base=query_base,
        query_norm=float(torch.linalg.vector_norm(query_base)),
        cache=cache,
        record_spans=tuple(record_spans),
        record_keys=tuple(record_keys),
        key_spans=key_spans,
    )


@torch.no_grad()
def read_slot(
    base,
    projection: GenericProjection,
    target_key: str,
    slot: int,
) -> GenericRead:
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
    s0 = projection.key_spans[family.key0]
    s1 = projection.key_spans[family.key1]
    mass0 = float(weights[s0[0] : s0[1]].sum())
    mass1 = float(weights[s1[0] : s1[1]].sum())
    total = mass0 + mass1
    share0 = mass0 / total if total > 0.0 else 0.5

    if target_key == family.key0:
        target_mass = mass0
        target_share = share0
    elif target_key == family.key1:
        target_mass = mass1
        target_share = 1.0 - share0
    else:
        raise ValueError(f"target {target_key} not in family")

    prediction = family.key0 if share0 >= 0.5 else family.key1

    return GenericRead(
        target_key=target_key,
        chosen_slot=slot,
        observer=observer,
        target_mass=target_mass,
        target_share=target_share,
        prediction=prediction,
        output=output.detach().clone(),
    )


def build_generic_suite():
    torch.manual_seed(0)
    torch.set_num_threads(2)

    base = build_fixture()
    model, tokenizer = load_model()
    projections = tuple(
        project_case(model, tokenizer, base, case) for case in cases()
    )
    return base, tokenizer, projections
