# Gate 2 result — pretrained K/V, persistent scalar observer

Gate 2 passed after a post-receipt hardening step closed a false-positive loophole.

## Pretrained source

```text
model: distilbert/distilgpt2
revision: 2290a62
selected layer/head: 3 / 2
head dimension: 64
```

DistilGPT2 produces the hidden states and Q/K/V from the actual prompt. Projected K/V are cloned once and then held fixed.

## Observer

```text
u  = normalize(mean(K_A) - mean(K_B))
q' = q_pretrained + m * ||q_pretrained|| * u
```

A fixed 161-point grid spans `m in [-4,+4]`. Head selection maximizes the smaller of the best absolute A-source attention mass and best absolute B-source attention mass.

Selected modes:

| mode | scalar m | intended-source total attention | intended A-vs-B share |
|---|---:|---:|---:|
| A | +1.15 | 1.000000 | 0.999999983 on A |
| B | -4.00 | 1.000000 | 0.999999999984 on B |

The B mode reaching the boundary is retained as a limitation.

## Calibration / blind test

Each block has one calibration read followed by three blind reads. The trusted-source receipt is revealed only after the calibration read. Blind reads receive no truth signal.

| condition | blind accuracy |
|---|---:|
| persistent adaptive observer | 1.000 |
| best fixed observer | 0.500 |
| reset observer | 0.500 |

Adaptive advantage: **+0.500**.

## Integrity

Projected K/V digest before and after:

```text
79a3fa9e0fbb26e50b3acd5e03b90a74c8fcf1b87c650a5ed6578f36a0543cde
```

Target attention parameter digest:

```text
ac80d2a0f80207e5a62aa81ff34b386bd19fef9a255d4aabf33e132723cdb824
```

Three Gate-2 tests pass in CI.

## Loophole found and closed

The first receipt used normalized A-vs-B source share. It passed while the B source received only about `1.8e-6` total attention. That meant the head was mostly ignoring both sources.

The hardened contract requires at least 20% **absolute total attention** on the intended source. The final selected head gives 100% target-source attention in both modes.

## Interpretation boundary

This is still a cache-specific steering mechanism. The address direction and scalar modes are derived on the same prompt later tested.

Gate 3 must freeze those choices and test them on unseen prompts/caches without rebuilding the observer geometry.
