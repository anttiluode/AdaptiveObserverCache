# Qwen distance sweep: re-ask saturation result

Date: 2026-09-22

Receipt: `results/qwen_observer_distance_sweep.json`

## 2026-09-23 correction

The original interpretation of this receipt was too strong.

The harness did **not** test observer control of a first-ask decision. Its
canonical cache already contained:

```text
system + sources + question + Qwen's closed valve answer
```

and the diagnostic branch then appended the same question again before scoring.
So this receipt is a **re-ask / saturation diagnostic**, not a valid temporal-
distance control experiment.

The corrected experiment is `qwen_observer_first_ask_distance.py` and is
documented in `QWEN_DISTANCE_SWEEP.md`.

## What the raw summed sequence scores show

The two candidate sentences have different token counts, so the original
per-token means hid how saturated the pairwise decision was.

```text
added   m     log P(valve sentence)   log P(sensor sentence)   winner
    0  -1             -10.2501                -0.0001          sensor
    0   0              -9.5001                -0.0001          sensor
    0  +1              -6.0025                -0.0025          sensor
  280  -1              -0.0006                -7.5006          valve
  280   0               0.0000               -18.5000          valve
  280  +1               0.0000               -21.2500          valve
```

At distance 0 the sensor sentence wins for every observer state. At +280 the
valve sentence wins for every observer state. The observer changes the losing
candidate's likelihood, but never changes which complete candidate wins.

Therefore the old endpoint-swing statistic cannot be interpreted as decision
control.

## What remains established by this receipt

- The one-cache / fork plumbing ran successfully under Transformers 5.15.0.
- Source spans remained A=[45,62), B=[72,87).
- Source-cache integrity remained true.
- No observer update hit the norm cap.
- The observer measurably changed downstream likelihoods in the saturated
  re-ask regime.

Those are useful mechanical facts, but they are weaker than the earlier claim
that long-cache decision control survived.

## Why the winner changed between checkpoints

Because the question had already been answered once, distance 0 means roughly:

```text
Q: Why did the device fail?
A: valve C was obstructed.
Q: Why did the device fail?     <- diagnostic re-ask
```

Qwen strongly preferred the sensor candidate on that re-ask. The later visible
filler repeatedly stated that no new evidence about the earlier failure had
arrived; after that trajectory, the same re-ask strongly preferred the valve
candidate.

Thus the old run changed both cache position and conversational history, while
also operating in a saturated decision regime.

## Verdict

**Valid re-ask/saturation control; invalid test of first-ask distance control.**

Do not use this receipt to claim either persistence or decay of AOC decision
control with source age.
