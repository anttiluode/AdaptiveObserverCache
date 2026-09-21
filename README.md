# AdaptiveObserverCache

**Can the same fixed memory reveal different useful information because the reader itself has persistent state — and can new evidence change that reader without rewriting the memory?**

This repo isolates one mechanism from the recent `AInsteinInsideTransformerResidualStream` / Genealogy "observer in the loop" line:

```text
fixed cache C
fixed present h
persistent observer state m
        |
        v
query geometry Q(h, m)
        |
        v
read fixed cache C
        |
        v
observation + receipt
        |
        v
update m
        |
        v
next query is different
```

The intended transformer bridge is simple:

```text
ordinary attention:
q_t = W_Q h_t

adaptive observer:
q_t = Q(h_t, m_t)

m_(t+1) = U(m_t, observation_t, receipt_t, goal_t)
```

The cache is historical material. The observer state is the changing measuring apparatus.

## Gate 0 — same cache, moving reader

Gate 0 is deliberately tiny and model-free. It uses a two-dimensional KV-like memory with conflicting evidence under two provenance families, A and B.

The contract freezes:

- the cache;
- the present input;
- the read budget: one cache read per step;
- all mechanism parameters;
- no learned weights;
- no hidden cache mutation.

Only the two-dimensional observer state may change.

Two tests matter.

### 0a. Observer state changes what the same memory reveals

With identical cache and identical present:

```text
m_A -> query points toward A-addressed evidence -> +1
m_B -> query points toward B-addressed evidence -> -1
```

The purpose is not the toy classification. It is to establish the primitive:

```text
memory content != memory currently observable
```

### 0b. Observation changes the future observer

The trusted provenance alternates in four blocks:

```text
AAAA BBBB AAAA BBBB
```

A static reader begins A-biased and never changes. The adaptive reader receives the same one-read budget, but after a contradicted read it updates only its observer state and re-queries the unchanged cache differently on the next step.

Pre-registered Gate-0 pass:

- same cache + same present + opposite observer states must select opposite provenance families;
- adaptive accuracy must be at least 0.75 on the switch schedule;
- adaptive must beat the static reader by at least 0.20 absolute accuracy;
- cache checksum must remain unchanged;
- recovery after a provenance switch must take no more than one contradicted read.

Run:

```bash
python gate0_experiment.py
python -m unittest discover -s tests -v
```

No external Python packages are required.

## Why this is not yet "a better KV cache"

Gate 0 does **not** claim that a production transformer should mutate its KV cache, that this is a new attention mechanism, or that persistent observer state is sufficient for useful reasoning.

It establishes a smaller object:

```text
fast present state h_t
slow observer state m_t
fixed historical field C_t
```

with

```text
read_t = A(Q(h_t, m_t), C_t)
m_(t+1) = U(m_t, read_t, receipt_t)
```

The next attack is to replace the synthetic two-address cache with a frozen transformer attention layer while keeping the same discipline: modify query/read geometry with a tiny low-rank observer state, do not alter model weights, and compare against a static reader under an equal retrieval/compute budget.

## Relationship to the recent repos

- **ReadWrite** — a state may be invisible until the right intervention/query is applied.
- **WhatToLookAt** — memory changes which later measurement is worth buying.
- **PredictiveHKT** — a changing representation can masquerade as a changing world.
- **AuditedEpistemicCache** — reusable evidence needs a receipt for the observer/representation that produced it.
- **OperatorTime** — the reader's resident state participates in the effective operator.
- **AInstein** — provenance matters; the right ingredients at the wrong addresses are not equivalent.
- **AInsteinInsideTransformerResidualStream** — temporary latent computation and compute foveation live in the fast stream; this repo isolates the slower reader state that could decide how history is interrogated.

The working slogan is:

> **Do not only remember the past. Let experience change the apparatus that reads the past.**
