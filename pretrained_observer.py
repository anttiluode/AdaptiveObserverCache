"""Gate 2: persistent 1-D observer over real pretrained DistilGPT2 K/V.

DistilGPT2 produces every hidden state and Q/K/V tensor.  The experiment then
freezes one head's projected K/V.  A scalar observer moves the final-token
query only along a source-address direction extracted from that frozen cache.

Gate 2 is deliberately cache-specific.  Gate 3 must test transfer.
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
MAX_PERTURBATION_RATIO = 4.0
OBSERVER_GRID_STEPS = 161


@dataclass(frozen=True)
class NaturalCache:
    keys: Tensor
    values: Tensor
    source_a: Tuple[int, int]
    source_b: Tuple[int, int]
    input_ids: Tuple[int, ...]


@dataclass(frozen=True)
class ObserverMode:
    state: float
    target_mass: float
    source_share_a: float


@dataclass(frozen=True)
class PretrainedObserverFixture:
    model_id: str
    revision: str
    prompt: str
    layer: int
    head: int
    head_dim: int
    query_base: Tensor
    query_norm: float
    observer_direction: Tensor
    mode_a: ObserverMode
    mode_b: ObserverMode
    symmetric_capture: float
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
    seq, width = x.shape
    if width % n_head:
        raise ValueError("hidden width must divide evenly across heads")
    return x.reshape(seq, n_head, width // n_head)


def _source_mean(keys: Tensor, span: Tuple[int, int]) -> Tensor:
    start, stop = span
    return keys[start:stop].mean(dim=0)


@torch.no_grad()
def _read_raw(
    query_base: Tensor,
    query_norm: float,
    direction: Tensor,
    keys: Tensor,
    values: Tensor,
    source_a: Tuple[int, int],
    source_b: Tuple[int, int],
    observer: float,
):
    query = query_base + float(observer) * query_norm * direction
    scores = keys @ query / sqrt(query.numel())
    weights = torch.softmax(scores, dim=0)
    output = weights @ values

    a0, a1 = source_a
    b0, b1 = source_b
    mass_a = float(weights[a0:a1].sum())
    mass_b = float(weights[b0:b1].sum())
    source_total = mass_a + mass_b
    share_a = mass_a / source_total if source_total > 0.0 else 0.5
    return query, weights, output, mass_a, mass_b, share_a


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

    grid = torch.linspace(
        -MAX_PERTURBATION_RATIO,
        MAX_PERTURBATION_RATIO,
        OBSERVER_GRID_STEPS,
    )

    # Search is blind to the future trusted-source schedule.  It knows only
    # which token spans are provenance A and B.  For each real pretrained head,
    # the address direction is the normalized K-mean difference.  Along that
    # single line, find the scalar state that maximizes absolute attention mass
    # on A and the state that maximizes absolute mass on B.  Choose the head
    # whose worse mode captures the most target attention.
    candidates = []

    with torch.no_grad():
        for layer_index, block in enumerate(model.transformer.h):
            hidden_in = outputs.hidden_states[layer_index][0]
            attn_in = block.ln_1(hidden_in)
            qkv = block.attn.c_attn(attn_in)
            q_all, k_all, v_all = qkv.split(n_embd, dim=-1)
            q_heads = _split_heads(q_all, n_head)
            k_heads = _split_heads(k_all, n_head)
            v_heads = _split_heads(v_all, n_head)

            for head_index in range(n_head):
                keys = k_heads[:, head_index, :].detach().clone()
                values = v_heads[:, head_index, :].detach().clone()
                query_base = q_heads[-1, head_index, :].detach().clone()
                query_norm = float(torch.linalg.vector_norm(query_base))
                if query_norm <= 1e-8:
                    continue

                mean_gap = _source_mean(keys, source_a) - _source_mean(
                    keys, source_b
                )
                gap_norm = float(torch.linalg.vector_norm(mean_gap))
                if gap_norm <= 1e-8:
                    continue
                direction = mean_gap / gap_norm

                best_a = None
                best_b = None

                for state_tensor in grid:
                    state = float(state_tensor)
                    (
                        _query,
                        _weights,
                        _output,
                        mass_a,
                        mass_b,
                        share_a,
                    ) = _read_raw(
                        query_base,
                        query_norm,
                        direction,
                        keys,
                        values,
                        source_a,
                        source_b,
                        state,
                    )

                    if best_a is None or mass_a > best_a[0]:
                        best_a = (mass_a, state, share_a)
                    if best_b is None or mass_b > best_b[0]:
                        best_b = (mass_b, state, share_a)

                assert best_a is not None and best_b is not None
                symmetric_capture = min(best_a[0], best_b[0])

                candidates.append(
                    (
                        symmetric_capture,
                        -max(abs(best_a[1]), abs(best_b[1])),
                        layer_index,
                        head_index,
                        query_base,
                        query_norm,
                        direction.detach().clone(),
                        keys,
                        values,
                        best_a,
                        best_b,
                    )
                )

    if not candidates:
        raise RuntimeError("no usable pretrained attention heads")

    (
        symmetric_capture,
        _negative_max_state,
        layer_index,
        head_index,
        query_base,
        query_norm,
        direction,
        keys,
        values,
        best_a,
        best_b,
    ) = max(candidates, key=lambda item: (item[0], item[1]))

    cache = NaturalCache(
        keys=keys,
        values=values,
        source_a=source_a,
        source_b=source_b,
        input_ids=tuple(int(x) for x in input_ids_list),
    )

    target_attention = model.transformer.h[layer_index].attn
    param_digest = module_parameter_digest(target_attention)
    tokens = tuple(tokenizer.convert_ids_to_tokens(list(cache.input_ids)))

    mode_a = ObserverMode(
        state=float(best_a[1]),
        target_mass=float(best_a[0]),
        source_share_a=float(best_a[2]),
    )
    mode_b = ObserverMode(
        state=float(best_b[1]),
        target_mass=float(best_b[0]),
        source_share_a=float(best_b[2]),
    )

    return PretrainedObserverFixture(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        prompt=source_a_text + source_b_text + query_text,
        layer=layer_index,
        head=head_index,
        head_dim=head_dim,
        query_base=query_base,
        query_norm=query_norm,
        observer_direction=direction,
        mode_a=mode_a,
        mode_b=mode_b,
        symmetric_capture=float(symmetric_capture),
        cache=cache,
        parameter_digest=param_digest,
        tokens=tokens,
    )


@torch.no_grad()
def read(
    fixture: PretrainedObserverFixture,
    observer: float,
) -> PretrainedRead:
    (
        query,
        weights,
        output,
        mass_a,
        mass_b,
        source_share_a,
    ) = _read_raw(
        fixture.query_base,
        fixture.query_norm,
        fixture.observer_direction,
        fixture.cache.keys,
        fixture.cache.values,
        fixture.cache.source_a,
        fixture.cache.source_b,
        observer,
    )

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
