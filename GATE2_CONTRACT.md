# Gate 2 contract — pretrained K/V, persistent observer

## Question

Does the Gate-1 mechanism survive contact with a **real frozen pretrained
transformer layer**?

Gate 2 may not manufacture Q/K/V tensors.  DistilGPT2 must produce the hidden
states and projected Q/K/V from an actual tokenized prompt.  Once projected,
K/V are frozen and reused byte-for-byte.

The model is pinned to:

```text
distilbert/distilgpt2
revision 2290a62
```

DistilGPT2 is used only as a frozen source of pretrained attention geometry.

## Prompt

The same exact prompt is used on every read:

```text
 Alice reports the vault color red.
 Bob reports the vault color blue.
 Which report should be trusted for the vault color?
```

The experiment knows the token spans belonging to Alice/source A and
Bob/source B.  It does **not** rewrite those spans or the projected cache.

## Observer

For each pretrained attention head, Gate 2 looks at its naturally produced K
vectors and derives the one-dimensional source-separation direction

```text
u = normalize(mean(K_A) - mean(K_B))
```

This does not say which source is trusted.  It only gives the local address
axis already available in that cache.

For every head, the experiment computes the smallest perturbation that would
create a +/-4 difference between the *mean* source logits:

```text
q' = q_pretrained + m * strength * u
```

The chosen head is the one requiring the smallest perturbation relative to
its natural query norm.  This selection rule is frozen before the schedule is
run and never sees the trusted-source labels.

The observer state is one scalar:

```text
m = +1  -> A direction
m =  0  -> ordinary pretrained query
m = -1  -> B direction
```

Gate 2 fails if the selected perturbation is more than 4x the natural query
norm.

## Calibration protocol

The trusted source alternates by four-read block:

```text
AAAA BBBB AAAA BBBB
```

The first read of each block is a **calibration read**.

Only after that read is complete, the environment reveals whether A or B is
trusted.  The adaptive observer may store that receipt in `m`.

The next three reads are blind test reads:

- exact same prompt;
- exact same projected K/V;
- no truth signal;
- one read each;
- no observer update.

Therefore the test asks whether information from an earlier calibration can
change later retrieval without changing memory.

## Attackers

All attackers get the same prompt, pretrained cache, and one-read budget.

1. best fixed observer from `m = -1, 0, +1`;
2. ordinary pretrained query is explicitly `m = 0`;
3. reset control receives each calibration but erases observer state before
   every blind test read.

## Pass boundary

Gate 2 passes only if:

1. `m=+1` sends at least 80% of A-vs-B source attention share to A;
2. `m=-1` sends at least 80% to B;
3. the two modes produce different attention outputs;
4. observer perturbation norm <= 4x the natural query norm;
5. adaptive blind-test accuracy >= 0.90;
6. adaptive beats the best fixed observer by >= 0.30 absolute;
7. reset-observer control <= 0.60;
8. projected K/V digest is identical before/after.

## Interpretation boundary

A pass does **not** show that DistilGPT2 spontaneously learned an adaptive
observer, nor that this improves language-model generation.

The address direction is extracted from the current cache using known source
spans.  Gate 2 asks a narrower question:

> Can a one-scalar persistent state steer a real pretrained attention reader
> over unchanged natural K/V, carrying an earlier calibration into later
> retrieval?

A pass earns Gate 3: the observer/address mechanism must transfer across
different prompts or content rather than being rebuilt and tested on one
identical cache.
