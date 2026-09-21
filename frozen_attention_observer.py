"""Frozen attention with a persistent rank-1 observer on the query only.

Gate 1 intentionally leaves projected K/V and all model parameters untouched.
The only evolving state is one scalar observer coordinate m_t.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import sqrt
from typing import Tuple

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ProjectedCache:
    keys: Tensor
    values: Tensor
    provenance: Tuple[str, ...]


@dataclass(frozen=True)
class AttentionRead:
    observer: float
    query: Tensor
    output: Tensor
    weights: Tensor
    prediction: int
    selected_provenance: str
    mass_a: float
    mass_b: float


def tensor_digest(tensor: Tensor) -> str:
    raw = bytes(
        tensor.detach()
        .cpu()
        .contiguous()
        .view(torch.uint8)
        .reshape(-1)
        .tolist()
    )
    return sha256(raw).hexdigest()


def cache_digest(cache: ProjectedCache) -> str:
    h = sha256()
    h.update(tensor_digest(cache.keys).encode("ascii"))
    h.update(tensor_digest(cache.values).encode("ascii"))
    h.update("|".join(cache.provenance).encode("utf-8"))
    return h.hexdigest()


def parameter_digest(module: nn.Module) -> str:
    h = sha256()
    for name, parameter in sorted(module.named_parameters()):
        h.update(name.encode("utf-8"))
        h.update(tensor_digest(parameter).encode("ascii"))
    return h.hexdigest()


class FrozenObserverAttention(nn.Module):
    """One frozen attention head with q' = W_Q h + strength * m * u."""

    def __init__(self, observer_strength: float = 2.5):
        super().__init__()
        self.d_model = 4
        self.d_head = 2
        self.observer_strength = float(observer_strength)

        self.q_proj = nn.Linear(
            self.d_model, self.d_head, bias=False, dtype=torch.float64
        )
        self.k_proj = nn.Linear(
            self.d_model, self.d_head, bias=False, dtype=torch.float64
        )
        self.v_proj = nn.Linear(
            self.d_model, self.d_head, bias=False, dtype=torch.float64
        )

        with torch.no_grad():
            # Present coordinates -> query; address coordinates -> keys.
            qk = torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=torch.float64,
            )
            self.q_proj.weight.copy_(qk)
            self.k_proj.weight.copy_(qk)

            # Evidence coordinates -> values.
            self.v_proj.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=torch.float64,
                )
            )

        for parameter in self.parameters():
            parameter.requires_grad_(False)

        # +m turns toward A addresses; -m turns toward B addresses.
        self.register_buffer(
            "observer_direction",
            torch.tensor([1.0, -1.0], dtype=torch.float64),
            persistent=True,
        )

    @torch.no_grad()
    def project_cache(
        self, memory_hidden: Tensor, provenance: Tuple[str, ...]
    ) -> ProjectedCache:
        if memory_hidden.ndim != 2 or memory_hidden.shape[1] != self.d_model:
            raise ValueError("memory_hidden must have shape [n, d_model]")
        if len(provenance) != memory_hidden.shape[0]:
            raise ValueError("one provenance label is required per memory row")

        # Detach+clone makes the Gate-1 cache an explicit frozen object.
        keys = self.k_proj(memory_hidden).detach().clone()
        values = self.v_proj(memory_hidden).detach().clone()
        return ProjectedCache(keys=keys, values=values, provenance=tuple(provenance))

    @torch.no_grad()
    def read(
        self, present_hidden: Tensor, cache: ProjectedCache, observer: float
    ) -> AttentionRead:
        if present_hidden.shape != (self.d_model,):
            raise ValueError("present_hidden must have shape [d_model]")

        q_base = self.q_proj(present_hidden)
        query = (
            q_base
            + self.observer_strength
            * float(observer)
            * self.observer_direction
        )
        scores = cache.keys @ query / sqrt(self.d_head)
        weights = torch.softmax(scores, dim=0)
        output = weights @ cache.values

        mass_a = sum(
            float(weights[i])
            for i, label in enumerate(cache.provenance)
            if label == "A"
        )
        mass_b = sum(
            float(weights[i])
            for i, label in enumerate(cache.provenance)
            if label == "B"
        )
        selected = "A" if mass_a >= mass_b else "B"
        prediction = 1 if float(output[0]) >= 0.0 else -1

        return AttentionRead(
            observer=float(observer),
            query=query.detach().clone(),
            output=output.detach().clone(),
            weights=weights.detach().clone(),
            prediction=prediction,
            selected_provenance=selected,
            mass_a=mass_a,
            mass_b=mass_b,
        )

    @staticmethod
    def update_observer(
        observer: float, result: AttentionRead, truth: int
    ) -> float:
        """Use feedback only after the read; no target information enters q_t."""

        if truth not in (-1, 1):
            raise ValueError("truth must be -1 or +1")
        if result.prediction == truth:
            return float(observer)

        # A contradiction swaps the one-dimensional read mode.
        # +1 = trust A-addressed evidence; -1 = trust B-addressed evidence.
        return float(truth)


def gate1_fixture():
    model = FrozenObserverAttention(observer_strength=2.5)

    # Address dimensions are first two coordinates.
    # Evidence/value dimensions are last two.
    memory_hidden = torch.tensor(
        [
            [1.00, 0.15, +1.00, +0.20],
            [0.90, -0.10, +0.80, +0.10],
            [0.15, 1.00, -1.00, -0.20],
            [-0.10, 0.90, -0.80, -0.10],
        ],
        dtype=torch.float64,
    )
    provenance = ("A", "A", "B", "B")
    present_hidden = torch.tensor(
        [1.0, 1.0, 0.0, 0.0], dtype=torch.float64
    )
    cache = model.project_cache(memory_hidden, provenance)
    return model, cache, present_hidden
