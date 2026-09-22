# Gate 3 contract — freeze the observer, move the cache

## Question

Gate 2 calibrated its address direction and scalar modes on the same cache it
later read.

Gate 3 asks whether that observer coordinate is reusable.

## Frozen from Gate 2

The following are frozen **before any Gate-3 test prompt is projected**:

- DistilGPT2 revision `2290a62`;
- selected layer `3`;
- selected head `2`;
- the 64-D observer direction `u`;
- scalar A mode `m_A ~= +1.15`;
- scalar B mode `m_B = -4.0`.

On a Gate-3 prompt:

```text
q' = q_new + m * ||q_new|| * u_gate2
```

No test-cache key vectors may be used to modify `u`, choose another head, or
retune either scalar state.

Source spans are known **only to the evaluator** so it can measure where
attention landed.

## Unseen caches

Seven transfer prompts change content and/or wording while keeping Alice as
provenance A and Bob as provenance B:

- five same-layout content changes;
- two wording/paraphrase changes.

Two additional prompts reverse source order. These are a diagnostic attacker,
not part of the Gate-3 pass boundary. They test whether the frozen direction
looks more like source identity or source position.

## Per-prompt success

Both modes must succeed on a prompt.

A mode:

- >=20% of total attention on A;
- >=80% of A-vs-B source share on A.

B mode:

- >=20% of total attention on B;
- >=80% of A-vs-B source share on B.

Projected K/V must remain byte-identical during reads.

## Pass boundary

Gate 3 passes only if:

- >=5/7 non-order-swapped transfer prompts succeed in both modes;
- >=4/5 same-layout prompts succeed;
- >=1/2 paraphrase prompts succeed;
- mean of the worse target mass per transfer prompt >=0.20;
- every test cache remains unchanged.

## Diagnostic: local direction cosine

For analysis only, after projection the evaluator may calculate:

```text
cos(u_gate2, normalize(mean(K_A_new)-mean(K_B_new)))
```

This local direction is **never used to read**. It only tells us whether
success/failure tracks geometric drift.

## Interpretation boundary

A Gate-3 pass would establish limited cross-cache transfer with fixed source
identities. It would not establish order invariance, new-source generality, or
language-model output improvement.

The order-swap attacker determines the next fork:

- if it survives, attack new source identities;
- if it fails, the current observer is likely entangled with position/layout
  and the next mechanism must factor provenance identity from position.
