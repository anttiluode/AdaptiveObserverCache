# Qwen live-cache distance × trust discriminator

Date: 2026-09-22

## Question

Does the frozen Qwen observer keep causal leverage over an old source as one
real KV cache grows, or does the observer dose-response flatten with temporal
distance?

The first successful live second turn confounded two things:

- source distance increased;
- B trust was only `m=-0.664`, weaker than the earlier `m=-1` switch.

This experiment holds observer strength fixed across matched distance
checkpoints.

## Critical invariant

There is exactly one **canonical growing cache**.

Only neutral deterministic filler turns advance it. Every trust probe runs on a
temporary fork of that exact cache. A probe must not change the canonical cache
length or token history, and the existing source-row integrity check must remain
true.

That makes all trust values at one checkpoint measurements of the same history.

## Primary readout

For the original valve/sensor conflict, teacher-force both complete candidate
answers and record

```text
mean log p(A) - mean log p(B)
```

across observer trust.

Attention movement is diagnostic only. The main object is the downstream
candidate-likelihood dose-response.

For checkpoints that contain `m=-1` and `m=+1`, also record

```text
endpoint swing = margin(+1) - margin(-1)
```

and its fraction of the distance-zero swing.

## Run tonight

The default is intentionally small:

```bash
git pull
python3.13 qwen_observer_distance_sweep.py
```

This measures target distances `0,256` with `m=-1,0,+1`.

A slightly denser second run, if useful:

```bash
python3.13 qwen_observer_distance_sweep.py \
  --distances 0,256,512 \
  --trust-grid=-1,-0.5,0,0.5,1 \
  --receipt results/qwen_observer_distance_medium.json
```

## Full run

```bash
python3.13 qwen_observer_distance_sweep.py \
  --distances 0,128,512,1024,2048 \
  --trust-grid=-1,-0.75,-0.5,0,0.5,0.75,1 \
  --generate-endpoints \
  --receipt results/qwen_observer_distance_full.json
```

The script writes the receipt after every completed checkpoint so a later crash
does not erase earlier evidence.

Distances are targets measured in canonical KV tokens added **after the initial
closed assistant answer**. Because growth uses complete neutral chat turns, an
actual checkpoint can overshoot its target slightly; both target and actual
distance are recorded.

## Competing outcomes

### Stable control

If the likelihood curves and endpoint swing remain similar as distance grows,
while old source K rows remain intact, the persistent observer has earned a
long-context control claim. The earlier `m=-0.664` non-switch was mostly a
strength/threshold issue.

### Distance attenuation

If the curves flatten, shrink, or drift back toward the neutral/primacy basin
with distance at matched trust, temporal distance is a real failure mode of the
current local-tangent controller.

### Content/head specificity

If this calibration conflict stays stable with distance while held-out
conflicts remain inconsistent, the dominant problem is not memory age. The
frozen causal actuator itself needs to generalize better.

## Boundary

This experiment does **not** test PCA, PAC, entorhinal-style transverse sweeps,
or GAx. Those remain unearned side hypotheses until this distance discriminator
is resolved.


## First run exposed a trajectory confound

The first completed visible-filler receipt is now recorded in
\`RESULTS_QWEN_DISTANCE_SWEEP.md\`.

The observer did not fade at +280 tokens: the endpoint likelihood swing grew
from about 0.425 to 1.719. However the neutral A-vs-B margin itself moved from
about -0.950 to +2.313. The visible filler therefore changed the computational
basin strongly enough that it cannot serve as a pure distance manipulation.

The next discriminator is the masked positional control:

\`\`\`bash
python3.13 qwen_observer_masked_distance.py
\`\`\`

It creates cache rows that advance Qwen's cached position while remaining
masked from all later attention. This separates readable intervening trajectory
from positional age much more cleanly.
