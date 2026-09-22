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

## What Gate 2 does — and does not — establish

Gate 2 now has a real pretrained attention field:

```text
pretrained prompt -> natural hidden states -> natural projected K/V
                                              |
historical receipt -> scalar m -> query line -+
                                              |
                                              v
                                      different read
```

The blind-test result demonstrates that an earlier calibration can persist as one scalar and materially alter later retrieval from unchanged pretrained K/V.

But the mechanism is still **cache-specific**. The source-address direction, selected head, and two useful scalar modes were calibrated on the same cache later used for testing. The B mode also lands exactly on the allowed `m=-4` boundary. That asymmetry is evidence, not decoration.

Therefore the next gate is transfer:

> Freeze the selected head and observer mechanism on calibration caches, then move to different prompts/caches and forbid rebuilding the address mechanism there.

If it survives that, the object starts looking less like a clever local steering trick and more like a reusable observer state.

## Relationship to recent repos

- **ReadWrite** — a state may be invisible until the right intervention/query is applied.
- **WhatToLookAt** — memory changes which later measurement is worth buying.
- **PredictiveHKT** — a changing representation can masquerade as a changing world.
- **AuditedEpistemicCache** — reusable evidence needs a receipt for the observer/representation that produced it.
- **OperatorTime** — the reader's resident state participates in the effective operator.
- **AInstein** — provenance matters; ingredients at the wrong addresses are not equivalent.
- **AInsteinInsideTransformerResidualStream** — temporary latent computation lives in the fast stream; this repo isolates a slower reader state that decides how history is interrogated.

> **Do not only remember the past. Let experience change the apparatus that reads the past.**
