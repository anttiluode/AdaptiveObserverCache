# Gate 1 result — frozen attention, adaptive observer

Gate 1 passed its pre-registered bridge contract.

## Setup

A single frozen PyTorch attention head uses:

```text
q' = W_Q h + 2.5 * m * u
```

with fixed `W_Q/W_K/W_V`, identical present hidden state, projected K/V frozen once, one read per step, and one scalar persistent observer state `m`.

Feedback is supplied only after the current read.

The trusted provenance schedule is:

```text
AAAA BBBB AAAA BBBB
```

The static attacker searches `m=-1,0,+1` and takes the best fixed result.

## Receipt

| measurement | result |
|---|---:|
| A mass at `m=+1` | 0.962236 |
| B mass at `m=+1` | 0.037764 |
| A mass at `m=-1` | 0.037764 |
| B mass at `m=-1` | 0.962236 |
| best fixed accuracy | 0.5000 |
| adaptive accuracy | 0.8125 |
| adaptive advantage | +0.3125 |
| switch-step contradictions | 3 / 3 |
| cache changed | no |
| parameters changed | no |
| trainable parameters | none |

Cache digest before/after:

```text
c98fa957ef01a9ba7ae49bb943e145f389d10864553efa3ed9796ae5ee5fd19c
```

Parameter digest before/after:

```text
d882c794aecd821a164dfa01057bf19bebc593abf1698b3ef19d0ff43f144958
```

CI passed the Gate-1 receipt and four Gate-1 unit tests on CPU PyTorch.

## Narrow interpretation

This proves the mechanism, not the application.

The result shows that a one-dimensional persistent observer can change retrieval from identical fixed K/V by changing only query geometry, and that post-read feedback can alter the next retrieval without cache or weight mutation.

The memory fixture, observer direction and feedback rule are constructed. No claim is made yet that a pretrained language model naturally exposes the same useful axis.

That is Gate 2.
