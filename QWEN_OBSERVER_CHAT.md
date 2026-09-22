# Qwen3-8B observer chat bridge

This is the first practical bridge from the scientific
`AdaptiveObserverCache` gates to a model that can produce ordinary language.

It is deliberately **not Gate 8**. Gate 8 remains the active-observation /
measurement-cost question exposed by Gate 7.

## What persists

The durable object is a scalar observer state:

```text
+1     strongly favor provenance/source A
 0     neutral
-1     strongly favor provenance/source B
```

The scalar is **not written into the prompt** and it is not a model parameter.

The selected layer/query-head identities are calibrated once at startup and
then frozen. What changes at every generation step is the local read tangent.

For each selected Q head, the controller uses that head's shared Qwen GQA KV
head and the current post-RoPE source keys:

```text
d_t = mean(K_A,t) - mean(K_B,t)
```

It then computes the minimum-norm query correction required to reach the
observer's requested signed A-vs-B attention-logit margin:

```text
q'_t = q_t + delta_t
```

subject to a fixed `||delta|| / ||q||` cap.

That means the system carries the **relation / trust state**, not one global
steering vector. RoPE and the changing conversation geometry are handled by
reconstructing the tangent from the coordinates Qwen is using now.

## Why this is different from the DistilGPT2 gates

Gates 2–3 showed that a fixed direction could work across several DistilGPT2
caches but initially encoded source slot/order rather than source identity.
Gates 4–7 progressively separated persistent provenance from address, alias
and behavioral continuity.

Qwen adds two complications deliberately:

- RoPE makes a fixed K/Q direction position dependent.
- Qwen3-8B uses grouped-query attention (32 Q heads / 8 KV heads).

The bridge therefore computes observer updates in **post-RoPE GQA geometry**.

## What stays frozen

- Qwen weights are frozen.
- The prompt does not contain the trust setting.
- Existing source K/V rows are never rewritten by the observer.
- During one generation, the controller hashes the source K slices after
  prefill and checks that the cached rows remain byte-identical on later decode
  steps.

The cache still grows normally with newly generated tokens.

## First run

This reuses the conservative placement profile that already worked for the
recent Qwen3-8B experiments:

```bash
pip install -r requirements-qwen.txt

python qwen_observer_chat.py --compare
```

Defaults:

```text
model            Qwen/Qwen3-8B
candidate layers 18,24,30
observer heads   4
GPU cap          6 GiB
CPU cap          6 GiB
overflow         .offload_qwen_observer
max new tokens   64
```

If Windows kills the process while loading shards, use the already-tested
more aggressive fallback:

```bash
python qwen_observer_chat.py --compare \
  --gpu-memory 4GiB --cpu-memory 4GiB
```

The comparison runs the **same prompt** three times:

```text
trust A
neutral
trust B
```

No trust instruction is added to the text.

## Interactive mode

```bash
python qwen_observer_chat.py
```

Commands:

```text
/a
/b
/neutral
/trust 0.35
/state
/compare Why did the device fail?
/quit
```

The same selected heads stay frozen for the session. The local query
correction is reconstructed from the current cache every turn.

## What counts as interesting

The first useful receipt is not merely "attention moved." We want all three:

1. A/B observer states produce a reproducible difference in generated
   source-grounded answers under identical textual instructions.
2. The required query correction stays bounded rather than saturating the
   `max_ratio` cap everywhere.
3. Source K rows remain unchanged during decoding.

A failure is also useful. If attention moves cleanly while language behavior
does not, then one/four local Q heads are not a sufficiently load-bearing
control surface and the next step is to widen the observer over a measured
head/layer set rather than simply increasing perturbation strength.

## Relationship to Anthropic J-space

The Jacobian lens identifies a privileged residual-stream control surface
inside a forward pass. This bridge is orthogonal: it adds a small recurrent
state outside the normal text stream that changes **how historical
representations are read**.

A later experiment can compose them:

```text
persistent observer
    -> choose / reweight memory read
    -> result enters workspace-like residual representation
    -> downstream reasoning
    -> evidence updates persistent observer
```

This bridge does not yet claim that composition.


## First Qwen3-8B receipt: attention moved, answer did not

The first real local run produced a useful dissociation.

With four geometry-selected heads at layer 24:

```text
A trust:
  mean intended-source mass  0.56876
  mean other-source mass     0.03690
  mean query update ratio    0.26343

neutral:
  mean A-source mass         0.35373
  mean B-source mass         0.11909

B trust:
  mean intended-source mass  0.51965
  mean other-source mass     0.03146
  mean query update ratio    0.47282

cache_integrity_ok = true in all three runs
capped_fraction    = 0.0 in all three runs
```

Yet all three generated:

```text
The device failed because valve C was obstructed.
```

So the bridge passed the **read-control** test but failed the first
**language-behavior** test.

The important correction is:

> geometry calibration found heads that are easy to aim at either source; it
> did not establish that those heads are load-bearing for the answer.

Do not respond by simply increasing `target_margin` or the perturbation cap.

## Causal head selector

`qwen_observer_causal_compare.py` adds a second calibration stage.

Stage 1 keeps a small pool of heads that can be symmetrically aimed at both
sources. To avoid selecting only one easy layer, the pool takes the best
geometry-qualified heads **per candidate layer**.

Stage 2 gives each candidate head a serial, identical-shape one-word task:

```text
Which component caused the failure?
Answer with exactly one word: valve or sensor.
```

For each head it measures

```text
g_A = log p(valve) - log p(sensor)   under A-trust
g_B = log p(valve) - log p(sensor)   under B-trust

causal_swing = g_A - g_B
```

Only positive-swing heads are eligible. Negative-swing heads are not silently
inverted. The A/B runs use the same prompt shape and the same patched attention
implementation; only the observer sign changes.

Stage 3 freezes the causally selected head identities and reruns the original
free-generation A / neutral / B comparison.

Run:

```bash
git pull
python3.13 qwen_observer_causal_compare.py
```

The default pool is 4 heads per layer across layers 18, 24 and 30 (12 serial
candidates, 24 causal probe forward passes). It writes:

```text
results/qwen_observer_causal_compare.json
```

This is deliberately closer to the J-space question: not merely *can this
internal coordinate be changed?*, but *does changing it causally alter a
downstream decision?*
