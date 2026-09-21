"""Minimal observer-in-the-loop cache primitive.

The cache is immutable. Persistent observer state changes query geometry.
Contradictory feedback updates the observer, not the stored memories.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from typing import Iterable, Tuple


Vec2 = Tuple[float, float]


@dataclass(frozen=True)
class CacheEntry:
    key: Vec2
    value: float
    provenance: str


@dataclass(frozen=True)
class ObserverState:
    log_gain_a: float
    log_gain_b: float


@dataclass(frozen=True)
class ReadResult:
    query: Vec2
    value: float
    prediction: int
    selected_provenance: str
    mass_a: float
    mass_b: float
    confidence: float


def _dot(a: Vec2, b: Vec2) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _unit(v: Vec2) -> Vec2:
    norm = sqrt(_dot(v, v))
    if norm == 0.0:
        raise ValueError("zero query")
    return (v[0] / norm, v[1] / norm)


def _softmax(values: Iterable[float]) -> Tuple[float, ...]:
    values = tuple(values)
    peak = max(values)
    weights = tuple(exp(v - peak) for v in values)
    total = sum(weights)
    return tuple(w / total for w in weights)


class AdaptiveObserverCache:
    def __init__(self, entries: Iterable[CacheEntry], temperature: float = 0.2):
        self._entries = tuple(entries)
        if not self._entries:
            raise ValueError("cache must not be empty")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    @property
    def entries(self) -> Tuple[CacheEntry, ...]:
        return self._entries

    def checksum(self) -> Tuple[Tuple[Vec2, float, str], ...]:
        return tuple((entry.key, entry.value, entry.provenance) for entry in self._entries)

    def query_vector(self, present: Vec2, observer: ObserverState) -> Vec2:
        # Observer state changes the geometry of the probe, not the cache.
        return _unit(
            (
                present[0] * exp(observer.log_gain_a),
                present[1] * exp(observer.log_gain_b),
            )
        )

    def read(self, present: Vec2, observer: ObserverState) -> ReadResult:
        query = self.query_vector(present, observer)
        scores = tuple(
            _dot(query, entry.key) / self.temperature for entry in self._entries
        )
        weights = _softmax(scores)
        value = sum(w * entry.value for w, entry in zip(weights, self._entries))

        mass_a = sum(
            w for w, entry in zip(weights, self._entries) if entry.provenance == "A"
        )
        mass_b = sum(
            w for w, entry in zip(weights, self._entries) if entry.provenance == "B"
        )
        selected = "A" if mass_a >= mass_b else "B"
        prediction = 1 if value >= 0.0 else -1
        confidence = abs(mass_a - mass_b)

        return ReadResult(
            query=query,
            value=value,
            prediction=prediction,
            selected_provenance=selected,
            mass_a=mass_a,
            mass_b=mass_b,
            confidence=confidence,
        )

    @staticmethod
    def update(
        observer: ObserverState,
        result: ReadResult,
        truth: int,
        surprise_step: float = 3.0,
    ) -> ObserverState:
        if truth not in (-1, 1):
            raise ValueError("truth must be -1 or +1")
        if result.prediction == truth:
            return observer

        a = observer.log_gain_a
        b = observer.log_gain_b

        # A contradiction demotes the provenance that dominated the read and
        # promotes the alternative. The memory itself is untouched.
        if result.selected_provenance == "A":
            a -= surprise_step
            b += surprise_step
        else:
            b -= surprise_step
            a += surprise_step

        return ObserverState(
            log_gain_a=max(-6.0, min(6.0, a)),
            log_gain_b=max(-6.0, min(6.0, b)),
        )


def gate0_cache() -> AdaptiveObserverCache:
    return AdaptiveObserverCache(
        (
            CacheEntry((1.00, 0.15), +1.00, "A"),
            CacheEntry((0.90, -0.10), +0.80, "A"),
            CacheEntry((0.15, 1.00), -1.00, "B"),
            CacheEntry((-0.10, 0.90), -0.80, "B"),
        ),
        temperature=0.2,
    )
