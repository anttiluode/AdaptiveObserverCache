"""Qwen3 bridge for AdaptiveObserverCache.

The durable object is not a steering vector.  It is a small observer state
(trust/provenance).  At each read we reconstruct the minimum query correction
from the *current post-RoPE key geometry* of the source spans.

This module keeps all Qwen-specific heavy imports lazy so its pure geometry can
be unit-tested without loading an 8B model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import sqrt
from types import MethodType
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch import Tensor


Span = Tuple[int, int]


@dataclass(frozen=True)
class SourceSpans:
    source_a: Span
    source_b: Span


@dataclass(frozen=True)
class QueryUpdate:
    trust: float
    natural_gap: float
    desired_gap: float
    achieved_gap: float
    delta_norm: float
    query_norm: float
    ratio: float
    capped: bool


@dataclass(frozen=True)
class CapturedGeometry:
    layer: int
    query_states: Tensor  # [num_query_heads, head_dim], CPU float32
    key_states: Tensor    # [num_kv_heads, sequence, head_dim], CPU float32
    scaling: float
    num_key_value_groups: int


@dataclass(frozen=True)
class HeadSelection:
    layer: int
    query_head: int
    kv_head: int
    symmetric_source_mass: float
    mass_when_a: float
    mass_when_b: float
    ratio_when_a: float
    ratio_when_b: float


@dataclass(frozen=True)
class ObserverPlan:
    heads: Tuple[HeadSelection, ...]

    def by_layer(self) -> Dict[int, Tuple[int, ...]]:
        grouped: Dict[int, List[int]] = {}
        for item in self.heads:
            grouped.setdefault(item.layer, []).append(item.query_head)
        return {layer: tuple(heads) for layer, heads in grouped.items()}


@dataclass
class RuntimeRead:
    layer: int
    query_head: int
    kv_head: int
    trust: float
    target_mass: float
    other_mass: float
    update_ratio: float
    natural_gap: float
    achieved_gap: float
    capped: bool


def rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    """Apply Qwen/Llama RoPE for [B,H,T,D] q/k and [B,T,D] cos/sin."""

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def minimum_norm_query_update(
    query: Tensor,
    direction: Tensor,
    *,
    trust: float,
    target_margin: float,
    scaling: float,
    max_ratio: float,
) -> Tuple[Tensor, QueryUpdate]:
    """Minimum-norm local correction satisfying a signed A-vs-B logit margin.

    Positive trust favors source A, negative trust favors source B.  A zero
    observer leaves the query untouched.  If the natural query already
    satisfies the requested inequality, no correction is applied.
    """

    trust = max(-1.0, min(1.0, float(trust)))
    target_margin = max(0.0, float(target_margin))
    max_ratio = max(0.0, float(max_ratio))

    q = query
    d = direction.to(device=q.device, dtype=q.dtype)
    q_norm = float(torch.linalg.vector_norm(q.float()))
    d2 = float(torch.dot(d.float(), d.float()))
    natural = float(scaling * torch.dot(q.float(), d.float()))

    if abs(trust) < 1e-12 or d2 <= 1e-12 or q_norm <= 1e-12:
        info = QueryUpdate(
            trust=trust,
            natural_gap=natural,
            desired_gap=0.0,
            achieved_gap=natural,
            delta_norm=0.0,
            query_norm=q_norm,
            ratio=0.0,
            capped=False,
        )
        return q, info

    desired = trust * target_margin
    satisfied = (trust > 0.0 and natural >= desired) or (
        trust < 0.0 and natural <= desired
    )
    if satisfied:
        info = QueryUpdate(
            trust=trust,
            natural_gap=natural,
            desired_gap=desired,
            achieved_gap=natural,
            delta_norm=0.0,
            query_norm=q_norm,
            ratio=0.0,
            capped=False,
        )
        return q, info

    # scaling * (q + c d).d = desired
    coeff = (desired - natural) / max(1e-12, scaling * d2)
    delta = coeff * d
    delta_norm = float(torch.linalg.vector_norm(delta.float()))
    ratio = delta_norm / max(1e-12, q_norm)
    capped = False

    if ratio > max_ratio and delta_norm > 0.0:
        delta = delta * (max_ratio * q_norm / delta_norm)
        delta_norm = float(torch.linalg.vector_norm(delta.float()))
        ratio = delta_norm / max(1e-12, q_norm)
        capped = True

    corrected = q + delta
    achieved = float(scaling * torch.dot(corrected.float(), d.float()))
    info = QueryUpdate(
        trust=trust,
        natural_gap=natural,
        desired_gap=desired,
        achieved_gap=achieved,
        delta_norm=delta_norm,
        query_norm=q_norm,
        ratio=ratio,
        capped=capped,
    )
    return corrected, info


def attention_masses(
    query: Tensor,
    keys: Tensor,
    spans: SourceSpans,
    *,
    scaling: float,
) -> Tuple[float, float]:
    logits = torch.mv(keys.float(), query.float()) * float(scaling)
    weights = torch.softmax(logits, dim=0)
    a0, a1 = spans.source_a
    b0, b1 = spans.source_b
    return float(weights[a0:a1].sum()), float(weights[b0:b1].sum())


def locate_token_span(
    offsets: Sequence[Tuple[int, int]],
    char_start: int,
    char_stop: int,
) -> Span:
    indices = [
        i
        for i, (start, stop) in enumerate(offsets)
        if stop > start and stop > char_start and start < char_stop
    ]
    if not indices:
        raise RuntimeError(
            f"no tokens overlap character range {char_start}:{char_stop}"
        )
    return min(indices), max(indices) + 1


def locate_source_spans(
    rendered: str,
    offsets: Sequence[Tuple[int, int]],
    source_a: str,
    source_b: str,
) -> SourceSpans:
    try:
        a_char = rendered.index(source_a)
        b_char = rendered.index(source_b)
    except ValueError as exc:
        raise RuntimeError("source text was not preserved in rendered chat") from exc

    return SourceSpans(
        source_a=locate_token_span(offsets, a_char, a_char + len(source_a)),
        source_b=locate_token_span(offsets, b_char, b_char + len(source_b)),
    )


def _candidate(
    capture: CapturedGeometry,
    query_head: int,
    spans: SourceSpans,
    *,
    target_margin: float,
    max_ratio: float,
) -> HeadSelection:
    kv_head = query_head // capture.num_key_value_groups
    query = capture.query_states[query_head]
    keys = capture.key_states[kv_head]

    a0, a1 = spans.source_a
    b0, b1 = spans.source_b
    direction = keys[a0:a1].mean(dim=0) - keys[b0:b1].mean(dim=0)

    q_a, info_a = minimum_norm_query_update(
        query,
        direction,
        trust=+1.0,
        target_margin=target_margin,
        scaling=capture.scaling,
        max_ratio=max_ratio,
    )
    q_b, info_b = minimum_norm_query_update(
        query,
        direction,
        trust=-1.0,
        target_margin=target_margin,
        scaling=capture.scaling,
        max_ratio=max_ratio,
    )
    mass_a, _ = attention_masses(q_a, keys, spans, scaling=capture.scaling)
    _, mass_b = attention_masses(q_b, keys, spans, scaling=capture.scaling)

    return HeadSelection(
        layer=capture.layer,
        query_head=query_head,
        kv_head=kv_head,
        symmetric_source_mass=min(mass_a, mass_b),
        mass_when_a=mass_a,
        mass_when_b=mass_b,
        ratio_when_a=info_a.ratio,
        ratio_when_b=info_b.ratio,
    )


def choose_observer_plan(
    captures: Mapping[int, CapturedGeometry],
    spans: SourceSpans,
    *,
    target_margin: float,
    max_ratio: float,
    num_heads: int,
) -> ObserverPlan:
    candidates: List[HeadSelection] = []
    for capture in captures.values():
        for head in range(capture.query_states.shape[0]):
            candidates.append(
                _candidate(
                    capture,
                    head,
                    spans,
                    target_margin=target_margin,
                    max_ratio=max_ratio,
                )
            )

    candidates.sort(
        key=lambda item: (
            item.symmetric_source_mass,
            -max(item.ratio_when_a, item.ratio_when_b),
        ),
        reverse=True,
    )
    return ObserverPlan(heads=tuple(candidates[: max(1, int(num_heads))]))


def tensor_digest(tensor: Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().numpy().tobytes()
    return sha256(raw).hexdigest()


def _forward_storage(module) -> str:
    """Return the forward slot that preserves Accelerate's device/offload hook.

    Accelerate wraps module.forward and stores the real implementation in
    _old_forward. Replacing forward directly would bypass its pre/post device
    hooks on disk/CPU-offloaded models.
    """

    if hasattr(module, "_hf_hook") and hasattr(module, "_old_forward"):
        return "_old_forward"
    return "forward"


@torch.inference_mode()
def capture_qwen_geometry(
    model,
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    layers: Iterable[int],
) -> Dict[int, CapturedGeometry]:
    """Capture current post-RoPE q/k geometry without changing model execution.

    Capture wrappers are inserted *inside* any Accelerate offload wrapper, so
    hidden states and layer weights are already on the execution device.
    """

    captures: Dict[int, CapturedGeometry] = {}
    originals = []

    for layer_index in sorted(set(int(x) for x in layers)):
        attn = model.model.layers[layer_index].self_attn
        storage = _forward_storage(attn)
        original = getattr(attn, storage)
        originals.append((attn, storage, original))

        def wrapped_capture(
            module,
            *args,
            _layer_index=layer_index,
            _original=original,
            **kwargs,
        ):
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[0]

            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None and len(args) > 1:
                position_embeddings = args[1]

            if hidden_states is None or position_embeddings is None:
                raise RuntimeError(
                    "Qwen attention capture did not receive "
                    "hidden_states/position_embeddings"
                )

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)
            q = module.q_norm(
                module.q_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            k = module.k_norm(
                module.k_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            cos, sin = position_embeddings
            q, k = apply_rope(q, k, cos, sin)

            captures[_layer_index] = CapturedGeometry(
                layer=_layer_index,
                query_states=q[0, :, -1, :].detach().float().cpu(),
                key_states=k[0].detach().float().cpu(),
                scaling=float(module.scaling),
                num_key_value_groups=int(module.num_key_value_groups),
            )
            return _original(*args, **kwargs)

        setattr(attn, storage, MethodType(wrapped_capture, attn))

    try:
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    finally:
        for attn, storage, original in originals:
            setattr(attn, storage, original)

    missing = sorted(set(int(x) for x in layers) - set(captures))
    if missing:
        raise RuntimeError(f"failed to capture Qwen geometry at layers {missing}")
    return captures


class QwenObserverController:
    """Temporarily replaces selected Qwen attention forwards.

    The selected layer/head identities are frozen after calibration.  The
    actual read direction is *not*: at every generation step it is recomputed
    from the current post-RoPE source keys.
    """

    def __init__(
        self,
        model,
        plan: ObserverPlan,
        *,
        trust: float = 0.0,
        target_margin: float = 3.0,
        max_ratio: float = 1.0,
    ):
        self.model = model
        self.plan = plan
        self.trust = float(trust)
        self.target_margin = float(target_margin)
        self.max_ratio = float(max_ratio)
        self.spans: SourceSpans | None = None
        self.runtime_reads: List[RuntimeRead] = []
        self.cache_integrity_ok = True
        self._generation_snapshots: Dict[Tuple[int, int], Tuple[str, str]] = {}
        self._original_forward: Dict[int, Tuple[str, object]] = {}
        self._installed = False

    def set_trust(self, trust: float) -> None:
        self.trust = max(-1.0, min(1.0, float(trust)))

    def set_spans(self, spans: SourceSpans) -> None:
        self.spans = spans

    def begin_generation(self) -> None:
        self.runtime_reads.clear()
        self.cache_integrity_ok = True
        self._generation_snapshots.clear()

    def _check_source_cache(
        self, layer: int, kv_head: int, keys: Tensor, spans: SourceSpans
    ) -> None:
        a0, a1 = spans.source_a
        b0, b1 = spans.source_b
        key = (layer, kv_head)
        digests = (
            tensor_digest(keys[a0:a1]),
            tensor_digest(keys[b0:b1]),
        )
        if key not in self._generation_snapshots:
            self._generation_snapshots[key] = digests
        elif self._generation_snapshots[key] != digests:
            self.cache_integrity_ok = False

    def install(self) -> None:
        if self._installed:
            return

        grouped = self.plan.by_layer()
        controller = self

        for layer_index, selected_heads in grouped.items():
            attn = self.model.model.layers[layer_index].self_attn
            storage = _forward_storage(attn)
            self._original_forward[layer_index] = (
                storage,
                getattr(attn, storage),
            )

            def wrapped(
                module,
                hidden_states,
                position_embeddings,
                attention_mask=None,
                past_key_value=None,
                cache_position=None,
                past_key_values=None,
                _layer_index=layer_index,
                _selected_heads=selected_heads,
                **kwargs,
            ):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, module.head_dim)

                query_states = module.q_norm(
                    module.q_proj(hidden_states).view(hidden_shape)
                ).transpose(1, 2)
                key_states = module.k_norm(
                    module.k_proj(hidden_states).view(hidden_shape)
                ).transpose(1, 2)
                value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                query_states, key_states = apply_rope(
                    query_states, key_states, cos, sin
                )

                cache = past_key_values if past_key_values is not None else past_key_value
                if cache is not None:
                    cache_kwargs = {
                        "sin": sin,
                        "cos": cos,
                        "cache_position": cache_position,
                    }
                    try:
                        key_states, value_states = cache.update(
                            key_states,
                            value_states,
                            module.layer_idx,
                            cache_kwargs,
                        )
                    except TypeError:
                        key_states, value_states = cache.update(
                            key_states, value_states, module.layer_idx
                        )

                spans = controller.spans
                if spans is None:
                    raise RuntimeError("observer source spans were not set")
                if hidden_states.shape[0] != 1:
                    raise RuntimeError("Qwen observer chat currently supports batch size 1")

                a0, a1 = spans.source_a
                b0, b1 = spans.source_b
                if max(a1, b1) > key_states.shape[-2]:
                    raise RuntimeError(
                        "source span lies beyond current Qwen cache; prompt geometry changed"
                    )

                query_states = query_states.clone()
                for q_head in _selected_heads:
                    kv_head = q_head // module.num_key_value_groups
                    keys = key_states[0, kv_head]
                    controller._check_source_cache(
                        _layer_index, kv_head, keys, spans
                    )
                    direction = (
                        keys[a0:a1].mean(dim=0)
                        - keys[b0:b1].mean(dim=0)
                    )

                    current = query_states[0, q_head, -1, :]
                    corrected, info = minimum_norm_query_update(
                        current,
                        direction,
                        trust=controller.trust,
                        target_margin=controller.target_margin,
                        scaling=float(module.scaling),
                        max_ratio=controller.max_ratio,
                    )
                    query_states[0, q_head, -1, :] = corrected
                    mass_a, mass_b = attention_masses(
                        corrected,
                        keys,
                        spans,
                        scaling=float(module.scaling),
                    )
                    target = mass_a if controller.trust >= 0 else mass_b
                    other = mass_b if controller.trust >= 0 else mass_a
                    controller.runtime_reads.append(
                        RuntimeRead(
                            layer=_layer_index,
                            query_head=q_head,
                            kv_head=kv_head,
                            trust=controller.trust,
                            target_mass=target,
                            other_mass=other,
                            update_ratio=info.ratio,
                            natural_gap=info.natural_gap,
                            achieved_gap=info.achieved_gap,
                            capped=info.capped,
                        )
                    )

                # Eager attention for the patched layer only.  Qwen3-8B uses no
                # sliding window, and this path keeps the modification local and
                # auditable even when the rest of the model uses SDPA.
                groups = int(module.num_key_value_groups)
                key_rep = key_states.repeat_interleave(groups, dim=1)
                value_rep = value_states.repeat_interleave(groups, dim=1)
                scores = torch.matmul(
                    query_states, key_rep.transpose(2, 3)
                ) * float(module.scaling)

                q_len = query_states.shape[-2]
                k_len = key_rep.shape[-2]
                if attention_mask is not None:
                    mask = attention_mask
                    if mask.ndim == 4:
                        mask = mask[..., :q_len, :k_len]
                        if mask.dtype == torch.bool:
                            scores = scores.masked_fill(
                                ~mask,
                                torch.finfo(scores.dtype).min,
                            )
                        else:
                            scores = scores + mask
                    elif mask.ndim == 2:
                        allowed = mask[:, None, None, :k_len].to(torch.bool)
                        scores = scores.masked_fill(
                            ~allowed,
                            torch.finfo(scores.dtype).min,
                        )
                    else:
                        raise RuntimeError(
                            f"unsupported attention mask rank {mask.ndim}"
                        )
                else:
                    past_len = k_len - q_len
                    q_pos = past_len + torch.arange(
                        q_len, device=scores.device
                    )
                    k_pos = torch.arange(k_len, device=scores.device)
                    causal = k_pos[None, :] <= q_pos[:, None]
                    scores = scores.masked_fill(
                        ~causal[None, None, :, :],
                        torch.finfo(scores.dtype).min,
                    )

                weights = torch.softmax(
                    scores, dim=-1, dtype=torch.float32
                ).to(query_states.dtype)
                attn_output = torch.matmul(weights, value_rep)
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(*input_shape, -1)
                attn_output = module.o_proj(attn_output)
                return attn_output, None

            setattr(attn, storage, MethodType(wrapped, attn))

        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        for layer_index, (storage, original) in self._original_forward.items():
            attn = self.model.model.layers[layer_index].self_attn
            setattr(attn, storage, original)
        self._original_forward.clear()
        self._installed = False

    def summary(self) -> dict:
        if not self.runtime_reads:
            return {
                "reads": 0,
                "cache_integrity_ok": self.cache_integrity_ok,
            }
        return {
            "reads": len(self.runtime_reads),
            "cache_integrity_ok": self.cache_integrity_ok,
            "mean_target_mass": sum(x.target_mass for x in self.runtime_reads)
            / len(self.runtime_reads),
            "mean_other_mass": sum(x.other_mass for x in self.runtime_reads)
            / len(self.runtime_reads),
            "mean_query_update_ratio": sum(
                x.update_ratio for x in self.runtime_reads
            )
            / len(self.runtime_reads),
            "max_query_update_ratio": max(
                x.update_ratio for x in self.runtime_reads
            ),
            "capped_fraction": sum(1 for x in self.runtime_reads if x.capped)
            / len(self.runtime_reads),
        }
