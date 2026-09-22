# Gate 3 result — transferable coordinate, positional meaning

Gate 3 passed its preregistered transfer boundary and the diagnostic attacker identified what the coordinate encodes.

## Frozen before test prompts

From the Gate-2 calibration cache:

```text
model revision: distilbert/distilgpt2 @ 2290a62
layer / head: 3 / 2
observer direction: one frozen 64-D vector
A mode: m ~= +1.15
B mode: m ~= -4.0  (current CI: -3.95)
```

No Gate-3 test cache may change the head, direction, or scalar modes.

## Transfer result

| group | dual-success |
|---|---:|
| same-layout content changes | 5 / 5 |
| paraphrase / wording changes | 2 / 2 |
| all non-order-swapped transfer | 7 / 7 |
| order-swapped diagnostics | 0 / 2 |

Mean worse intended-source total-attention mass over the seven transfer prompts:

```text
0.9999947974
```

Mean cosine between the frozen Gate-2 direction and each unseen cache's locally measured A→B key direction:

```text
0.9441036327
```

The local direction is diagnostic only and is never used by the reader.

Every projected test cache remained byte-identical during reads.

## What transferred

Across new objects, values and two wording changes, the fixed coordinate behaved almost identically:

```text
m_A -> source slot 1
m_B -> source slot 2
```

The absolute source engagement remained near one, not merely a normalized-share effect.

## What did not transfer

When source order was reversed, both diagnostic cases failed.

```text
vault order swap:
    local direction cosine = -0.8748066

garden order swap:
    local direction cosine = -0.8470161
```

The modes reversed identity: A-mode read the first-position Bob source and B-mode read the second-position Alice source.

## Interpretation

Gate 3 therefore narrows the claim.

The observer coordinate is **reusable across cache content and wording**, but it is presently a positional/source-slot coordinate rather than an order-invariant source-identity coordinate.

This suggests the next mechanism should factor two objects that Gate 2/3 had conflated:

```text
persistent choice of provenance/identity
                +
current binding of that provenance to an address/slot
                +
reusable positional read coordinate
```

The next gate should test whether identity can remain persistent while source order changes, with only the binding updated.
