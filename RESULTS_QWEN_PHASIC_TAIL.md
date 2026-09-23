# Final Qwen phasic-tail result

Date: 2026-09-23

This is the closeout for the Qwen3-8B distance / tail thread in AdaptiveObserverCache.

The important result is a disagreement between two preregistered readouts:

- exact fixed-candidate likelihood says the +256 B sentence loses;
- greedy generation still produces the B explanation under the observer.

Token-level inspection shows why. The long-distance penalty is not on the source-specific content `K drifted`; it is almost entirely on the final period of the short candidate. Qwen prefers to continue the same B explanation with a comma.

## Setup

Frozen ingredients:

- model: `Qwen/Qwen3-8B`
- Transformers: `5.15.0`
- source-only first-ask anchor: 93 cached tokens
- observer trust for B: `m=-1`
- same four causally selected observer heads
- same source spans and source-key integrity checks
- exactly 256 unreadable masked spacer rows for the far checkpoint
- decision token: candidate index 4, ` valve` versus ` sensor`

Three schedules were compared on isolated cache forks:

```text
tonic   m=-1 for every candidate token
phasic  m=-1 through the valve/sensor decision, then m=0
neutral m=0 throughout
```

The schedule and interpretation threshold were frozen before the run. The tail classifier calls an observer-tail effect real only at >=0.5 nats.

Receipt:

```text
results/qwen_observer_phasic_tail_gen.json
```

## Result at distance 0

```text
           sumA       sumB      A-B margin   decision gap      B tail     greedy
neutral   -0.0050   -11.4545   +11.4495      A strongly       -0.2014    A
                                                                        
tonic     -1.1621    -0.8751    -0.2870      B by 0.750       -0.4741    B
phasic    -1.1551    -0.7879    -0.3672      B by 0.750       -0.3869    B
```

So the corrected first-ask baseline is real: the observer changes both the matched decision and the generated answer from valve/A to sensor/B.

## Result after +256 masked positions

```text
           sumA       sumB      A-B margin   decision gap      B tail     greedy
neutral   -0.1388   -11.6879   +11.5491      A strongly       -1.3133    A

tonic     -2.6070    -3.1283    +0.5213      B by 1.750       -2.3502    B
phasic    -2.5767    -2.9050    +0.3283      B by 1.750       -2.1269    B
```

The fixed short B candidate therefore loses under summed likelihood, even though the local semantic choice moves *more strongly* toward `sensor` and free generation still chooses B.

Generated text at +256:

```text
tonic   -> The device failed because sensor K drifted, as supported by the calibration log.
phasic  -> The device failed because sensor K drifted, as supported by the calibration log.
neutral -> The device failed because valve C was obstructed.
```

## Where the apparent B-tail failure actually lives

For the B candidate `The device failed because sensor K drifted.` the token log-probabilities show:

```text
+0 tonic:
  sensor   -0.3986
  K         0.0000
  drifted  ~0.0000
  .        -0.4741

+256 tonic:
  sensor   -0.7548
  K         0.0000
  drifted  ~0.0000
  .        -2.3502

+256 phasic:
  sensor   -0.7548
  K         0.0000
  drifted  ~0.0000
  .        -2.1269
```

At +256, the model's preferred final token after `sensor K drifted` is a comma, not a period. Greedy generation follows that preference and continues with `, as supported by the calibration log.`

Therefore the distance penalty is **not evidence that the old source details `K` or `drifted` became inaccessible**. Those content tokens remain essentially certain once B has been selected. The penalty is a change in termination / continuation preference for this exact candidate string.

## Phasic hypothesis

The preregistered classifier reports:

```text
observer tail damage (phasic B tail - tonic B tail) = +0.223 nats
distance tail cost (neutral B tail, near - far)      = +1.112 nats
threshold                                              0.500 nats
likelihood verdict                                     DISTANCE_DAMAGES_TAIL
generation verdict                                     GENERATION_CONTROL_SURVIVES
```

The phasic improvement is below the frozen 0.5-nat threshold and tonic/phasic greedy generations are identical at both checkpoints.

So this experiment does **not** support the idea that leaving the observer active after the decision is the main source of tail damage. On this prompt, switching the observer off after `sensor` is not required to preserve the semantic answer.

The exact-string likelihood verdict remains part of the record; it should be read literally as a failure to prefer the *short period-terminated B sentence*, not as loss of B semantic control.

## Final AOC interpretation

What survives:

1. A frozen Qwen3-8B observer intervention can change the generated causal answer on the calibration conflict without changing historical source K rows.
2. With a corrected source-only first-ask anchor, the same observer controls the matched `valve`/`sensor` decision.
3. After exactly 256 unreadable masked cache positions, that local semantic decision still favors B under `m=-1` and greedy generation still produces the B explanation.
4. Source-cache integrity remains true and no observer update hits its norm cap.
5. The neutral first-ask A-vs-B margin is nearly unchanged between 0 and +256, so the dramatic neutral flip in the older visible-filler sweep came from readable conversational trajectory, not masked positional age alone.

What does not survive as a claim:

1. The exact fixed short B candidate does not remain the summed-likelihood winner at +256.
2. The phasic schedule does not materially rescue the tail under the frozen threshold.
3. The frozen observer is not a general answer-switch mechanism: the earlier held-out transfer gate achieved only 2/4 full generation switches, despite 4/4 directional likelihood movement.
4. This is one synthetic valve/sensor conflict, not a general long-context benchmark.

The most accurate closeout is therefore:

> **AOC demonstrates a persistent external observer state that can causally alter how a frozen transformer reads unchanged history and can preserve semantic answer control across at least 256 masked cache positions on its calibration conflict. Exact candidate-string likelihood is less stable because continuation style can move independently of the semantic decision. General transfer remains the main unsolved boundary.**

## Incomplete longer-distance run

`results/qwen_observer_phasic_tail_far.json` was started with distances `0,512,1024,2048`, but the committed receipt has `complete=false` and contains only the distance-0 checkpoint. Partial console output from +512 is therefore not promoted to evidence here.

No further GPU run is required for this closeout.
