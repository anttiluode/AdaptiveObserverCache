# Qwen first-ask distance × observer-control discriminator

Date: 2026-09-23

## The question we actually need to ask

Can the frozen Qwen observer still change **which answer wins on the first ask**
after the source records have receded inside one growing KV cache?

The corrected execution order is:

```text
system + Source A + Source B
        |
        +-- cache grows by masked spacer positions
        |
        +-- first and only user question
        |
        +-- observer m=-1 / 0 / +1
        |
        +-- candidate decision
```

No prior assistant answer exists in the canonical cache.

## Why the 2026-09-22 distance runs do not answer this

`qwen_observer_distance_sweep.py` and `qwen_observer_masked_distance.py` first
built a cache containing:

```text
system + sources + question + Qwen's closed valve answer
```

and then appended the **same question again** before candidate scoring.

The visible-filler receipt therefore measured a saturated re-ask regime, not
first-ask control. At distance 0 the sensor sentence already had probability
approximately one for every observer value; at +280 the valve sentence had
probability approximately one for every observer value. The observer changed
the losing candidate's score without changing the winning candidate.

Those runs remain useful as re-ask/saturation diagnostics and cache-plumbing
checks, but they do not establish or refute distance-dependent decision control.

## Corrected harness

Run:

```bash
python3.13 qwen_observer_first_ask_distance.py
```

Default checkpoints are exact masked distances `0,256` with trust values
`m=-1,0,+1`.

The canonical cache contains only the closed system/source message. The question
suffix is withheld. Masked spacer rows then advance DynamicCache/RoPE position
while remaining invisible to later attention. At each checkpoint every trust
condition forks the same canonical cache and appends the question for the first
and only time.

The receipt is:

```text
results/qwen_observer_first_ask_distance.json
```

## Primary gate: did the winner change?

For each candidate record the **summed** complete-sequence log probability:

```text
sumA = log P(valve sentence)
sumB = log P(sensor sentence)
M    = sumA - sumB
```

The primary control gate is:

```text
m=-1 : M < 0   -> B wins
m=+1 : M > 0   -> A wins
```

or equivalently:

```math
M_d(-1) < 0 < M_d(+1)
```

A likelihood swing that never changes the winner is not counted as decision
control.

The harness prints the two raw sums, the margin, and the winner for every trust
value.

## Saturation warning

For the pair of complete candidates, the harness reports

```math
p_A^{pair} = sigmoid(sumA - sumB)
```

and flags the neutral decision as `SATURATED` when the pairwise winning
probability is at least 0.99.

This warning matters because a perturbation that only moves a deeply losing
candidate can produce a large-looking dose-response while having essentially no
chance to change the decision.

## Length-neutral secondary diagnostic

The full candidates have different token counts, so summed sequence scores have
a length effect. The harness therefore also finds the first token where the two
candidate tokenizations diverge after their shared answer prefix and reports the
single-step log-probability gap there.

For the default answers this should correspond to the local `valve` versus
`sensor` decision. It is a secondary diagnostic, not a replacement for the
full-sequence winner gate.

## Fail fast at distance zero

Distance is meaningless if the observer cannot control the first-ask decision
at the anchor.

Therefore the default harness stops after distance 0 if the summed-sequence
winner does not flip between `m=-1` and `m=+1`. It will not spend the +256 GPU
work unless baseline control is established.

An override exists only for debugging:

```bash
python3.13 qwen_observer_first_ask_distance.py --continue-without-baseline-flip
```

Do not use an overridden run to make a distance-control claim.

## Interpretation

### Flip at 0 and flip at +256

The current frozen observer has controlled the same first-ask A/B decision even
when the source records are 256 unreadable cache positions older.

### Flip at 0, fail at +256

The distance question is finally earned: positional/cache age is a failure mode
of the current actuator or its locally reconstructed tangent.

### No flip at 0

Do not interpret distance. Debug the first-ask scorer/calibration against
`qwen_observer_causal_compare.py`, which previously produced a language-level
A/B switch on a fresh first question.

### Full-sequence and matched-token gates disagree

Inspect tokenization/length and downstream continuation effects. Do not collapse
the disagreement into a single success/failure story.

## Boundary

This still does not test PCA/PAC, entorhinal transverse sweeps, GAx, or a
residual-stream/Jacobian actuator. Those become relevant after the basic
first-ask distance control question has a valid result.
