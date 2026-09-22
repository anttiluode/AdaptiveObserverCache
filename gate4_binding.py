"""Gate 4: bind persistent provenance identity to a reusable source-slot reader.

Gate 3 established a strong first-source <-> second-source coordinate, but it
failed when Alice/Bob order was swapped. Gate 4 factors the mechanism:

persistent trusted identity
        -> current identity-to-slot binding
        -> frozen Gate-2/3 slot read mode

The binder is deliberately transparent. Each source record begins with an
explicit provenance identity token (Alice or Bob). It uses only those frozen
token IDs and record boundaries. It never sees trust labels or K/V geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, Tuple

import torch
from torch import Tensor

from gate3_transfer import _split_heads, load_model
from pretrained_observer import NaturalCache, build_fixture


@dataclass(frozen=True)
class BindingCase:
    name: str
    order: str
    records: Tuple[Tuple[str, str], ...]
    query: str


@dataclass(frozen=True)
class IdentityProjection:
    case: BindingCase
    prompt: str
    query_base: Tensor
    query_norm: float
    cache: NaturalCache
    record_spans: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class IdentityBinder:
    identity_token_ids: Dict[str, int]


@dataclass(frozen=True)
class BoundRead:
    target_identity: str
    chosen_slot: int
    observer: float
    mass_alice: float
    mass_bob: float
    target_mass: float
    target_share: float
    prediction: str
    output: Tensor


def cases() -> Tuple[BindingCase, ...]:
    families = (
        (
            "vault",
            " Alice reports the vault color red.",
            " Bob reports the vault color blue.",
            " Which report should be trusted for the vault color?",
        ),
        (
            "garden",
            " Alice reports the garden animal fox.",
            " Bob reports the garden animal owl.",
            " Which report should be trusted for the garden animal?",
        ),
        (
            "station",
            " Alice reports the station number seven.",
            " Bob reports the station number nine.",
            " Which report should be trusted for the station number?",
        ),
        (
            "weather",
            " Alice reports the weather today sunny.",
            " Bob reports the weather today cloudy.",
            " Which report should be trusted for the weather today?",
        ),
        (
            "package",
            " Alice says the package is heavy.",
            " Bob says the package is light.",
            " Which report should be trusted about the package?",
        ),
        (
            "river",
            " Alice records river level high.",
            " Bob records river level low.",
            " Which report should be trusted for river level?",
        ),
    )

    out = []
    for name, alice, bob, query in families:
        out.append(
            BindingCase(
                name=f"{name}-AB",
                order="AB",
                records=(("Alice", alice), ("Bob", bob)),
                query=query,
            )
        )
        out.append(
            BindingCase(
                name=f"{name}-BA",
                order="BA",
                records=(("Bob", bob), ("Alice", alice)),
                query=query,
            )
        )
    return tuple(out)


def build_binder(tokenizer) -> IdentityBinder:
    identity_token_ids = {}
    for identity in ("Alice", "Bob"):
        ids = tokenizer.encode(" " + identity, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"expected {identity} to be one provenance token, got {ids}"
            )
        identity_token_ids[identity] = int(ids[0])
    return IdentityBinder(identity_token_ids=identity_token_ids)


@torch.no_grad()
def project_case(model, tokenizer, base, case: BindingCase) -> IdentityProjection:
    ids = []
    record_spans = []
    identity_spans = {}

    for identity, text in case.records:
        piece = tokenizer.encode(text, add_special_tokens=False)
        start = len(ids)
        ids.extend(piece)
        span = (start, len(ids))
        record_spans.append(span)
        identity_spans[identity] = span

    ids.extend(tokenizer.encode(case.query, add_special_tokens=False))
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
        source_a=identity_spans["Alice"],
        source_b=identity_spans["Bob"],
        input_ids=tuple(int(x) for x in ids),
    )

    return IdentityProjection(
        case=case,
        prompt="".join(text for _, text in case.records) + case.query,
        query_base=query_base,
        query_norm=float(torch.linalg.vector_norm(query_base)),
        cache=cache,
        record_spans=tuple(record_spans),
    )


def bind_slots(
    binder: IdentityBinder,
    projection: IdentityProjection,
) -> Dict[str, int]:
    """Return identity -> current record slot without reading trust or K/V."""

    mapping = {}
    reverse = {
        token_id: identity
        for identity, token_id in binder.identity_token_ids.items()
    }

    for slot, (start, _stop) in enumerate(projection.record_spans):
        token_id = int(projection.cache.input_ids[start])
        identity = reverse.get(token_id)
        if identity is None:
            raise RuntimeError(
                f"record slot {slot} begins with unknown provenance token "
                f"{token_id}"
            )
        if identity in mapping:
            raise RuntimeError(f"duplicate provenance identity {identity}")
        mapping[identity] = slot

    if set(mapping) != {"Alice", "Bob"}:
        raise RuntimeError(f"incomplete provenance binding: {mapping}")
    return mapping


def slot_mode(base, slot: int) -> float:
    if slot == 0:
        return float(base.mode_a.state)
    if slot == 1:
        return float(base.mode_b.state)
    raise ValueError("Gate 4 supports exactly two source slots")


@torch.no_grad()
def read_slot(
    base,
    projection: IdentityProjection,
    target_identity: str,
    slot: int,
) -> BoundRead:
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

    a0, a1 = projection.cache.source_a
    b0, b1 = projection.cache.source_b
    mass_alice = float(weights[a0:a1].sum())
    mass_bob = float(weights[b0:b1].sum())
    source_total = mass_alice + mass_bob
    share_alice = (
        mass_alice / source_total if source_total > 0.0 else 0.5
    )

    if target_identity == "Alice":
        target_mass = mass_alice
        target_share = share_alice
    elif target_identity == "Bob":
        target_mass = mass_bob
        target_share = 1.0 - share_alice
    else:
        raise ValueError("target identity must be Alice or Bob")

    prediction = "Alice" if share_alice >= 0.5 else "Bob"

    return BoundRead(
        target_identity=target_identity,
        chosen_slot=slot,
        observer=observer,
        mass_alice=mass_alice,
        mass_bob=mass_bob,
        target_mass=target_mass,
        target_share=target_share,
        prediction=prediction,
        output=output.detach().clone(),
    )


def build_binding_suite():
    torch.manual_seed(0)
    torch.set_num_threads(2)

    base = build_fixture()
    model, tokenizer = load_model()
    binder = build_binder(tokenizer)
    projections = tuple(
        project_case(model, tokenizer, base, case) for case in cases()
    )
    return base, binder, projections
