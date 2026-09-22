"""Gate 3: apply the Gate-2 observer to unseen pretrained caches.

Nothing in the observer is rebuilt on test prompts:
- layer/head are frozen from Gate 2
- the 64-D observer direction is frozen from Gate 2
- scalar A/B states are frozen from Gate 2

Only each new prompt's ordinary pretrained q/K/V are recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Tuple

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from pretrained_observer import (
    MODEL_ID,
    MODEL_REVISION,
    NaturalCache,
    build_fixture,
    cache_digest,
)


@dataclass(frozen=True)
class PromptCase:
    name: str
    category: str
    segments: Tuple[Tuple[str, str], ...]
    query: str


@dataclass(frozen=True)
class TransferProjection:
    case: PromptCase
    prompt: str
    query_base: Tensor
    query_norm: float
    cache: NaturalCache
    local_direction_cosine: float


@dataclass(frozen=True)
class TransferRead:
    observer: float
    mass_a: float
    mass_b: float
    source_share_a: float
    prediction: str
    output: Tensor


def cases() -> Tuple[PromptCase, ...]:
    return (
        PromptCase(
            "garden-animal",
            "same_layout",
            (
                ("A", " Alice reports the garden animal fox."),
                ("B", " Bob reports the garden animal owl."),
            ),
            " Which report should be trusted for the garden animal?",
        ),
        PromptCase(
            "station-number",
            "same_layout",
            (
                ("A", " Alice reports the station number seven."),
                ("B", " Bob reports the station number nine."),
            ),
            " Which report should be trusted for the station number?",
        ),
        PromptCase(
            "weather",
            "same_layout",
            (
                ("A", " Alice reports the weather today sunny."),
                ("B", " Bob reports the weather today cloudy."),
            ),
            " Which report should be trusted for the weather today?",
        ),
        PromptCase(
            "door-code",
            "same_layout",
            (
                ("A", " Alice reports the door code alpha."),
                ("B", " Bob reports the door code omega."),
            ),
            " Which report should be trusted for the door code?",
        ),
        PromptCase(
            "meeting-room",
            "same_layout",
            (
                ("A", " Alice reports the meeting room east."),
                ("B", " Bob reports the meeting room west."),
            ),
            " Which report should be trusted for the meeting room?",
        ),
        PromptCase(
            "package-paraphrase",
            "paraphrase",
            (
                ("A", " Alice says the package is heavy."),
                ("B", " Bob says the package is light."),
            ),
            " Which report should be trusted about the package?",
        ),
        PromptCase(
            "river-paraphrase",
            "paraphrase",
            (
                ("A", " Alice records river level high."),
                ("B", " Bob records river level low."),
            ),
            " Which report should be trusted for river level?",
        ),
        PromptCase(
            "vault-order-swap",
            "order_swap",
            (
                ("B", " Bob reports the vault color blue."),
                ("A", " Alice reports the vault color red."),
            ),
            " Which report should be trusted for the vault color?",
        ),
        PromptCase(
            "garden-order-swap",
            "order_swap",
            (
                ("B", " Bob reports the garden animal owl."),
                ("A", " Alice reports the garden animal fox."),
            ),
            " Which report should be trusted for the garden animal?",
        ),
    )


def _split_heads(x: Tensor, n_head: int) -> Tensor:
    seq, width = x.shape
    return x.reshape(seq, n_head, width // n_head)


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        use_safetensors=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


@torch.no_grad()
def project_case(model, tokenizer, base, case: PromptCase):
    ids = []
    spans = {}

    for label, text in case.segments:
        piece = tokenizer.encode(text, add_special_tokens=False)
        start = len(ids)
        ids.extend(piece)
        spans[label] = (start, len(ids))

    query_ids = tokenizer.encode(case.query, add_special_tokens=False)
    ids.extend(query_ids)
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
        source_a=spans["A"],
        source_b=spans["B"],
        input_ids=tuple(int(x) for x in ids),
    )

    # Diagnostic only.  This local direction is never used by the reader.
    a0, a1 = spans["A"]
    b0, b1 = spans["B"]
    local_gap = keys[a0:a1].mean(0) - keys[b0:b1].mean(0)
    local_norm = float(torch.linalg.vector_norm(local_gap))
    if local_norm <= 1e-8:
        cosine = 0.0
    else:
        local_u = local_gap / local_norm
        cosine = float(torch.dot(local_u, base.observer_direction))

    return TransferProjection(
        case=case,
        prompt="".join(text for _, text in case.segments) + case.query,
        query_base=query_base,
        query_norm=float(torch.linalg.vector_norm(query_base)),
        cache=cache,
        local_direction_cosine=cosine,
    )


@torch.no_grad()
def read_transfer(base, projection: TransferProjection, observer: float):
    query = (
        projection.query_base
        + float(observer)
        * projection.query_norm
        * base.observer_direction
    )
    scores = projection.cache.keys @ query / sqrt(base.head_dim)
    weights = torch.softmax(scores, dim=0)
    output = weights @ projection.cache.values

    a0, a1 = projection.cache.source_a
    b0, b1 = projection.cache.source_b
    mass_a = float(weights[a0:a1].sum())
    mass_b = float(weights[b0:b1].sum())
    source_total = mass_a + mass_b
    share_a = mass_a / source_total if source_total > 0.0 else 0.5

    return TransferRead(
        observer=float(observer),
        mass_a=mass_a,
        mass_b=mass_b,
        source_share_a=share_a,
        prediction="A" if share_a >= 0.5 else "B",
        output=output.detach().clone(),
    )


def build_transfer_suite():
    torch.manual_seed(0)
    torch.set_num_threads(2)

    base = build_fixture()
    model, tokenizer = load_model()
    projections = tuple(
        project_case(model, tokenizer, base, case) for case in cases()
    )
    return base, projections
