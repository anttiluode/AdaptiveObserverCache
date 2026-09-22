"""Persistent observer evidence state.

This module intentionally has no torch/transformers dependency.  It separates
externally anchored evidence from self/model prediction so a steered read
cannot certify its own trust state.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import List


ANCHOR_KINDS = {
    "sensor",
    "independent_model",
    "tool",
    "user_verification",
}
SELF_KINDS = {
    "self_prediction",
    "model_output",
    "self",
}


@dataclass
class EvidenceReceipt:
    source: str
    weight: float
    kind: str
    note: str
    anchored: bool
    score_delta: float
    score_after: float
    trust_after: float


class EvidenceLedger:
    """Slow observer state with explicit evidence provenance.

    score is an additive signed evidence accumulator. Effective anchored trust
    is tanh(score), so evidence composes smoothly but remains bounded. A manual
    override is allowed for experiments, but it is kept separate from evidence.
    """

    def __init__(
        self,
        score: float = 0.0,
        receipts: List[EvidenceReceipt] | None = None,
        manual_override: float | None = None,
    ):
        self.score = float(score)
        self.receipts = list(receipts or [])
        self.manual_override = (
            None
            if manual_override is None
            else max(-1.0, min(1.0, float(manual_override)))
        )

    @property
    def anchored_trust(self) -> float:
        return math.tanh(self.score)

    @property
    def effective_trust(self) -> float:
        if self.manual_override is not None:
            return self.manual_override
        return self.anchored_trust

    def set_manual(self, trust: float | None) -> None:
        self.manual_override = (
            None
            if trust is None
            else max(-1.0, min(1.0, float(trust)))
        )

    def apply(
        self,
        source: str,
        weight: float,
        kind: str,
        note: str = "",
    ) -> EvidenceReceipt:
        source = source.strip().upper()
        kind = kind.strip().lower()
        if source not in {"A", "B"}:
            raise ValueError("source must be A or B")
        weight = float(weight)
        if not (0.0 <= weight <= 1.0):
            raise ValueError("evidence weight must lie in [0, 1]")
        if kind not in ANCHOR_KINDS | SELF_KINDS:
            allowed = ", ".join(sorted(ANCHOR_KINDS | SELF_KINDS))
            raise ValueError(
                f"unknown evidence kind {kind!r}; use one of: {allowed}"
            )

        anchored = kind in ANCHOR_KINDS
        sign = +1.0 if source == "A" else -1.0
        delta = sign * weight if anchored else 0.0
        self.score = max(-6.0, min(6.0, self.score + delta))
        receipt = EvidenceReceipt(
            source=source,
            weight=weight,
            kind=kind,
            note=note,
            anchored=anchored,
            score_delta=delta,
            score_after=self.score,
            trust_after=self.anchored_trust,
        )
        self.receipts.append(receipt)
        return receipt

    def to_json(self) -> dict:
        return {
            "score": self.score,
            "anchored_trust": self.anchored_trust,
            "manual_override": self.manual_override,
            "effective_trust": self.effective_trust,
            "receipts": [asdict(x) for x in self.receipts],
        }

    @classmethod
    def from_json(cls, payload: dict) -> "EvidenceLedger":
        rows = [
            EvidenceReceipt(**row)
            for row in payload.get("receipts", [])
        ]
        return cls(
            score=float(payload.get("score", 0.0)),
            receipts=rows,
            manual_override=payload.get("manual_override"),
        )
