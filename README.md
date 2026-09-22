# AdaptiveObserverCache

**Can the same fixed memory reveal different useful information because the reader itself has persistent state — and can new evidence change that reader without rewriting the memory?**

The working object is:

```text
fixed historical field C
fixed present h
persistent observer state m
        |
        v
query geometry Q(h, m)
        |
        v
read fixed memory C
        |
        v
receipt arrives after the read
        |
        v
update m
        |
        v
next read changes without rewriting memory
```

The cache is historical material. The observer state is the changing measuring apparatus.

## Gate 0 — synthetic primitive

Gate 0 established the smallest model-free version: identical cache and identical present can reveal different provenance depending on persistent reader state. On the alternating `AAAA BBBB AAAA BBBB` schedule, the static reader gets 50% while the adaptive reader gets 81.25%.

Run:

```bash
python gate0_experiment.py
```

## Gate 1 — frozen PyTorch attention

Gate 1 moved the primitive into an actual frozen attention calculation:

```text
q_base = W_Q h
q'     = q_base + 2.5 * m * u
scores = K q' / sqrt(d_head)
read   = softmax(scores) V
```

`W_Q/W_K/W_V`, projected K/V and the present hidden state are frozen. Only one scalar observer state changes the query.

Receipt:

```text
best fixed observer accuracy = 0.5000
adaptive observer accuracy   = 0.8125
adaptive advantage           = +0.3125

same K/V + m=+1 -> 96.22% A mass
same K/V + m=-1 -> 96.22% B mass
```

Projected-cache and parameter digests are byte-identical before/after.

See [GATE1_CONTRACT.md](GATE1_CONTRACT.md) and [RESULTS_GATE1.md](RESULTS_GATE1.md).

## Gate 2 — real pretrained DistilGPT2 K/V

Gate 2 removes the hand-built Q/K/V geometry.

Pinned model:

```text
distilbert/distilgpt2
revision 2290a62
```

The prompt is tokenized normally, DistilGPT2 produces hidden states and Q/K/V, and one real attention head is selected by a preregistered cache-geometry rule. The model weights are never trained or changed.

The observer is still one-dimensional:

```text
u  = normalize(mean(K_A) - mean(K_B))
q' = q_pretrained + m * ||q_pretrained|| * u
```

with `|m| <= 4`.

A fixed 161-point scalar grid is searched on every pretrained head. For each head, one scalar state maximizes **absolute total attention mass** on source A and one maximizes absolute total attention mass on source B. The chosen head maximizes the worse of those two target masses.

### Why Gate 2 was hardened before merge

The first Gate-2 receipt exposed a denominator loophole.

A B-mode could win almost all of the normalized A-vs-B source share while both source spans received essentially zero attention. That technically satisfied the first contract but was not a real retrieval.

So Gate 2 was strengthened before merge: each mode must put at least **20% of all attention** on its intended source, not merely beat the other source.

### Hardened Gate 2 receipt

CI selected:

```text
layer = 3
head  = 2
head dimension = 64
```

Modes:

```text
A mode: m = +1.15
    total attention on A = 1.000000
    A-vs-B share on A    = 0.999999983

B mode: m = -4.00
    total attention on B = 1.000000
    A-vs-B share on B    = 0.999999999984
```

Protocol:

```text
block A: 1 calibration read -> 3 blind reads
block B: 1 calibration read -> 3 blind reads
block A: 1 calibration read -> 3 blind reads
block B: 1 calibration read -> 3 blind reads
```

Only the first read of each block receives the trusted-source receipt, and it arrives **after** the read. The next three reads receive no truth signal.

Results:

```text
blind adaptive test accuracy = 12 / 12 = 1.000
best fixed reader            =  6 / 12 = 0.500
reset-observer control       =  6 / 12 = 0.500
adaptive advantage           = +0.500
```

The projected K/V digest was identical before/after:

```text
79a3fa9e0fbb26e50b3acd5e03b90a74c8fcf1b87c650a5ed6578f36a0543cde
```

Target-attention parameter digest:

```text
ac80d2a0f80207e5a62aa81ff34b386bd19fef9a255d4aabf33e132723cdb824
```

See [GATE2_CONTRACT.md](GATE2_CONTRACT.md) and [RESULTS_GATE2.md](RESULTS_GATE2.md).

Run:

```bash
pip install -r requirements-gate2.txt --extra-index-url https://download.pytorch.org/whl/cpu
python gate2_experiment.py
```

## Gate 3 — freeze the observer, move the cache

Gate 3 freezes the Gate-2 layer/head, the one-dimensional observer direction, and the two scalar read modes **before projecting any test prompt**. Seven new caches then receive only their ordinary pretrained q/K/V; no test-cache geometry is allowed to rebuild the observer.

The transfer result is unexpectedly clean:

```text
same-layout content changes     5 / 5 dual-success
paraphrase / wording changes    2 / 2 dual-success
all non-swapped transfer        7 / 7 dual-success

mean worse target mass          0.9999948
mean cos(local key axis, u_G2)  0.9441
all projected test caches       unchanged
```

So the Gate-2 coordinate is not merely a one-cache steering trick. Across these unseen prompts, the frozen A mode still puts essentially all attention on the first source span and the frozen B mode puts essentially all attention on the second.

### The order-swap attacker identifies the coordinate

Both order-swapped prompts fail **0 / 2**.

When Bob is moved to the first source slot and Alice to the second, the local A→B key axis flips relative to the frozen Gate-2 direction:

```text
vault order swap   cosine = -0.8748
garden order swap  cosine = -0.8470
```

and the observer modes reverse which identity they retrieve.

That changes the interpretation of the mechanism:

```text
not yet:  persistent "trust Alice / trust Bob" coordinate
closer to: persistent "read source slot 1 / read source slot 2" coordinate
```

This is still useful. A stable address axis across content and paraphrase is exactly the sort of thing a persistent observer can exploit. But the next mechanism must bind **who/what occupies an address** separately from the positional read coordinate.

See [GATE3_CONTRACT.md](GATE3_CONTRACT.md) and [RESULTS_GATE3.md](RESULTS_GATE3.md).

Run:

```bash
python gate3_experiment.py
```

## Current boundary

The chain now establishes:

```text
Gate 0  synthetic moving reader
Gate 1  frozen attention moving query
Gate 2  real pretrained K/V + persistent calibration state
Gate 3  frozen read coordinate transfers across new caches,
        but tracks source slot/order rather than source identity
```

The next gate should therefore **not** search another better steering vector. It should factor the problem:

```text
identity / provenance binding   +   reusable positional read coordinate
```

so that swapping source order changes the binding, not the meaning of the observer state.

## Relationship to recent repos

- **ReadWrite** — a state may be invisible until the right intervention/query is applied.
- **WhatToLookAt** — memory changes which later measurement is worth buying.
- **PredictiveHKT** — a changing representation can masquerade as a changing world.
- **AuditedEpistemicCache** — reusable evidence needs a receipt for the observer/representation that produced it.
- **OperatorTime** — the reader's resident state participates in the effective operator.
- **AInstein** — provenance matters; ingredients at the wrong addresses are not equivalent.
- **AInsteinInsideTransformerResidualStream** — temporary latent computation lives in the fast stream; this repo isolates a slower reader state that decides how history is interrogated.

> **Do not only remember the past. Let experience change the apparatus that reads the past.**
