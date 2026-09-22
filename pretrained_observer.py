"""Gate 2: observer-modulated query over real pretrained DistilGPT2 K/V.

The pretrained model produces the hidden states and projected Q/K/V.  After the
cache is materialized, every experimental read reuses exactly the same K/V.
Only a scalar persistent observer changes the final-token query of one frozen
attention head.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from math import sqrt
from typing import Tuple

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "distilbert/distilgpt2"
MODEL_REVISION = "2290a62"
TARGET_MEAN_LOGIT_MARGIN = 4.0


@dataclass(frozen=True)
class NaturalCache:
    keys: Tensor
    values: Tensor
    source_a: Tuple[int, int]
    source_b: Tuple[int, int]
    input_ids: Tuple[int, ...]


@dataclass(frozen=True)
class PretrainedObserverFixture:
    model_id: str
    revision: str
    prompt: str
    layer: int
    head: int
    head_dim: int
    query_base: Tensor
    observer_direction: Tensor
    observer_strength: float
    perturbation_ratio: float
    base_source_logit_delta: float
    unit_source_logit_delta: float
    cache: NaturalCache
    parameter_digest: str
    tokens: Tuple[str, ...]


@dataclass(frozen=True)
class PretrainedRead:
    observer: float
    query: Tensor
    output: Tensor
    weights: Tensor
    mass_a: float
    mass_b: float
    source_share_a: float
    prediction: str


def _tensor_bytes(tensor: Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def tensor_digest(tensor: Tensor) -> str:
    return sha256(_tensor_bytes(tensor)).hexdigest()


def cache_digest(cache: NaturalCache) -> str:
    h = sha256()
    h.update(_tensor_bytes(cache.keys))
    h.update(_tensor_bytes(cache.values))
    h.update(str(cache.source_a).encode("ascii"))
    h.update(str(cache.source_b).encode("ascii"))
    h.update(",".join(str(x) for x in cache.input_ids).encode("ascii"))
    return h.hexdigest()


def module_parameter_digest(module) -> str:
    h = sha256()
    for name, parameter in sorted(module.named_parameters()):
        h.update(name.encode("utf-8"))
        h.update(_tensor_bytes(parameter))
    return h.hexdigest()


def _split_heads(x: Tensor, n_head: int) -> Tensor:
    if x.ndim != 2:
        raise ValueError("expected [sequence, hidden] projection")
    seq, width = x.shape
    if width % n_head:
        raise ValueError("hidden width must divide evenly across heads")
    return x.reshape(seq, n_head, width // n_head)


def _source_mean(keys: Tensor, span: Tuple[int, int]) -> Tensor:
    start, stop = span
    return keys[start:stop].mean(dim=0)


def _source_delta(query: Tensor, keys: Tensor, a, b) -> float:
    mean_a = _source_mean(keys, a)
    mean_b = _source_mean(keys, b)
    return float(torch.dot(query, mean_a - mean_b) / sqrt(query.numel()))


@lru_cache(maxsize=1)
def build_fixture() -> PretrainedObserverFixture:
    torch.manual_seed(0)
    torch.set_num_threads(2)

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

    # Pieces are tokenized independently so the two provenance spans are exact.
    source_a_text = " Alice reports the vault color red."
    source_b_text = " Bob reports the vault color blue."
    query_text = " Which report should be trusted for the vault color?"

    a_ids = tokenizer.encode(source_a_text, add_special_tokens=False)
    b_ids = tokenizer.encode(source_b_text, add_special_tokens=False)
    q_ids = tokenizer.encode(query_text, add_special_tokens=False)

    input_ids_list = a_ids + b_ids + q_ids
    source_a = (0, len(a_ids))
    source_b = (len(a_ids), len(a_ids) + len(b_ids))
    input_ids = torch.tensor([input_ids_list], dtype=torch.long)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    config = model.config
    n_head = int(config.n_head)
    n_embd = int(config.n_embd)
    head_dim = n_embd // n_head

    # Head selection is geometry-only.  For every real pretrained head, derive
    # the source-separation direction from its natural K vectors, compute how
    # large a query perturbation would be needed to guarantee a +/-4 difference
    # between the *mean* source logits, and choose the head requiring the
    # smallest perturbation relative to its natural query norm.
    candidates = []
    cached_projections = {}

    with torch.no_grad():
        for layer_index, block in enumerate(model.transformer.h):
            hidden_in = outputs.hidden_states[layer_index][0]
            attn_in = block.ln_1(hidden_in)
            qkv = block.attn.c_attn(attn_in)
            q_all, k_all, v_all = qkv.split(n_embd, dim=-1)
            q_heads = _split_heads(q_all, n_head)
            k_heads = _split_heads(k_all, n_head)
            v_heads = _split_heads(v_all, n_head)

            cached_projections[layer_index] = (q_heads, k_heads, v_heads)

            for head_index in range(n_head):
                keys = k_heads[:, head_index, :]
                query = q_heads[-1, head_index, :]
                mean_gap = _source_mean(keys, source_a) - _source_mean(
                    keys, source_b
                )
                gap_norm = float(torch.linalg.vector_norm(mean_gap))
                if gap_norm <= 1e-8:
                    continue

                direction = mean_gap / gap_norm
                base_delta = _source_delta(
                    query, keys, source_a, source_b
                )
                unit_delta = float(
                    torch.dot(direction, mean_gap) / sqrt(head_dim)
                )
                strength = (
                    abs(base_delta) + TARGET_MEAN_LOGIT_MARGIN
                ) / unit_delta
                q_norm = float(torch.linalg.vector_norm(query))
                ratio = strength / max(q_norm, 1e-8)
                candidates.append(
                    (
                        ratio,
                        layer_index,
                        head_index,
                        strength,
                        base_delta,
                        unit_delta,
                        direction.detach().clone(),
                    )
                )

    if not candidates:
        raise RuntimeError("no pretrained attention head separated the sources")

    (
        ratio,
        layer_index,
        head_index,
        strength,
        base_delta,
        unit_delta,
        direction,
    ) = min(candidates, key=lambda item: item[0])

    q_heads, k_heads, v_heads = cached_projections[layer_index]
    query_base = q_heads[-1, head_index, :].detach().clone()
    keys = k_heads[:, head_index, :].detach().clone()
    values = v_heads[:, head_index, :].detach().clone()

    cache = NaturalCache(
        keys=keys,
        values=values,
        source_a=source_a,
        source_b=source_b,
        input_ids=tuple(int(x) for x in input_ids_list),
    )

    target_attention = model.transformer.h[layer_index].attn
    param_digest = module_parameter_digest(target_attention)

    tokens = tuple(
        tokenizer.convert_ids_to_tokens(list(cache.input_ids))
    )

    return PretrainedObserverFixture(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        prompt=source_a_text + source_b_text + query_text,
        layer=layer_index,
        head=head_index,
        head_dim=head_dim,
        query_base=query_base,
        observer_direction=direction,
        observer_strength=float(strength),
        perturbation_ratio=float(ratio),
        base_source_logit_delta=float(base_delta),
        unit_source_logit_delta=float(unit_delta),
        cache=cache,
        parameter_digest=param_digest,
        tokens=tokens,
    )


@torch.no_grad()
def read(
    fixture: PretrainedObserverFixture,
    observer: float,
) -> PretrainedRead:
    cache = fixture.cache
    query = (
        fixture.query_base
        + float(observer)
        * fixture.observer_strength
        * fixture.observer_direction
    )

    scores = cache.keys @ query / sqrt(fixture.head_dim)
    weights = torch.softmax(scores, dim=0)
    output = weights @ cache.values

    a0, a1 = cache.source_a
    b0, b1 = cache.source_b
    mass_a = float(weights[a0:a1].sum())
    mass_b = float(weights[b0:b1].sum())
    source_total = mass_a + mass_b
    if source_total <= 0.0:
        raise RuntimeError("source spans received zero attention mass")

    source_share_a = mass_a / source_total
    prediction = "A" if source_share_a >= 0.5 else "B"

    return PretrainedRead(
        observer=float(observer),
        query=query.detach().clone(),
        output=output.detach().clone(),
        weights=weights.detach().clone(),
        mass_a=mass_a,
        mass_b=mass_b,
        source_share_a=source_share_a,
        prediction=prediction,
    )
