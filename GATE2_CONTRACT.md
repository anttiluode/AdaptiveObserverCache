# Gate 2 contract — pretrained K/V, persistent observer

## Question

Can a one-scalar persistent observer steer a **real frozen pretrained
attention reader** over unchanged naturally produced K/V, carrying an earlier
calibration into later blind retrieval?

Gate 2 may not manufacture Q/K/V tensors. DistilGPT2 produces all hidden
states and projected Q/K/V from the tokenized prompt.

Pinned model:

```text
distilbert/distilgpt2
revision 2290a62
```

## Prompt and cache

Every read uses the exact same prompt:

```text
 Alice reports the vault color red.
 Bob reports the vault color blue.
 Which report should be trusted for the vault color?
```

A and B source spans are known provenance spans. Projected K/V are cloned once
and then reused byte-for-byte for every experimental read.

## One-dimensional observer

For each pretrained head, derive only one address direction:

```text
u = normalize(mean(K_A) - mean(K_B))
```

The observer can move only along that line:

```text
q' = q_pretrained + m * ||q_pretrained|| * u
```

with

```text
-4 <= m <= +4
```

A fixed 161-point grid is searched for every real pretrained head.

For each head:

- `m_A` is the grid point maximizing **absolute total attention mass on A**;
- `m_B` is the grid point maximizing **absolute total attention mass on B**.

The selected head maximizes:

```text
min(attention_mass_A(m_A), attention_mass_B(m_B))
```

This search sees source addresses but never sees the future trusted-source
schedule.

## Why absolute mass is required

An earlier version of Gate 2 used only normalized A-vs-B share. It passed, but
inspection exposed a loophole: B could win almost all of the A-vs-B share
while both source spans received essentially zero total attention.

That is not retrieval.

The hardened gate therefore requires both modes to **engage the intended
source in the full attention distribution**.

## Calibration protocol

Trusted source blocks:

```text
AAAA BBBB AAAA BBBB
```

Each four-read block has:

1. one calibration read;
2. only after that read, reveal trusted source A or B;
3. store `m_A` or `m_B` in the scalar persistent observer;
4. three blind reads with no truth signal and no further update.

Thus all blind test reads have:

- identical prompt;
- identical pretrained K/V;
- one attention read;
- no current truth;
- only historical observer state differs.

## Attackers

1. best fixed reader among `m_B, 0, m_A`;
2. ordinary pretrained query `m=0`;
3. reset control: receives calibration but erases `m` before every blind read.

## Pass boundary

Gate 2 passes only if:

1. A mode puts >=20% of **all attention** on source A;
2. B mode puts >=20% of **all attention** on source B;
3. A mode gives >=80% of A-vs-B source share to A;
4. B mode gives >=80% of A-vs-B source share to B;
5. both mode states stay within `|m| <= 4`;
6. the two modes produce different attention outputs;
7. adaptive blind-test accuracy >=0.90;
8. adaptive beats best fixed by >=0.30 absolute;
9. reset control <=0.60;
10. projected K/V digest is identical before/after.

## Interpretation boundary

A pass does **not** mean DistilGPT2 learned this observer, and does not yet
show better language generation.

The head and two scalar states are calibrated on the same cache later used for
testing. Gate 2 establishes a pretrained-attention mechanism only.

A pass earns Gate 3: **freeze the observer address mechanism on calibration
prompts, then test it on different prompts/caches where it may not be rebuilt.**
