# Qwen live growing-cache result

Date: 2026-09-22

This receipt records the first successful second-turn run of
`qwen_observer_live_cache.py` on the development machine after hardening the
incremental chat-template path.

## Setup

- Model: `Qwen/Qwen3-8B`
- Frozen causally selected observer heads:
  - L24/Q29
  - L24/Q23
  - L30/Q11
  - L18/Q28
- Initial source spans:
  - A: tokens 45..61
  - B: tokens 72..86
- One real `DynamicCache`; the transcript is not rerendered between turns.
- Initial observer state: `m = 0`.
- External evidence receipt:
  - source: B
  - kind: sensor
  - weight: 0.8
  - evidence score: -0.8
  - resulting anchored trust: `m = tanh(-0.8) = -0.6640367703`

## Observed run

Initial answer:

```text
The device failed because valve C was obstructed.
```

Initial cache:

```text
cache_tokens   131
history_tokens 131
source A       [45, 62)
source B       [72, 87)
cache integrity true
```

After the B evidence receipt, the user asked the same causal question again.
Only the new turn was appended to the already-live cache:

```text
cache 131 -> 149 -> 160 tokens
```

The historical source addresses stayed fixed at A=[45,62), B=[72,87), while
`cache_tokens == history_tokens == 160` and the session-level source-cache
integrity check remained true.

Observer diagnostics on the second answer:

```text
trust m                    -0.6640367703
reads                       48
mean target (B) mass         0.2531621
mean other (A) mass          0.0362338
mean query-update ratio      0.2838921
max query-update ratio       0.6243267
capped fraction              0.0
cache integrity              true
```

The generated sentence nevertheless remained:

```text
The device failed because valve C was obstructed.
```

## Verdict

**The growing-cache mechanism passed its first mechanical persistence test.**
One Qwen KV cache survived a real second conversational turn, grew from 131 to
160 committed tokens, kept the original source rows at their old addresses,
and preserved byte-level source-cache integrity while an external observer
state changed outside the prompt.

**The language-level switch did not occur at m=-0.664.** This is not yet a
temporal-distance failure. The previous order sweep already showed that the
original AB conflict needs a strong B-side intervention before the candidate
likelihood crosses; intermediate negative trust values remained on the A side.
The live run changes both temporal distance and trust magnitude relative to the
earlier m=-1 switch, so those factors are still confounded.

The next discriminator is therefore a distance-by-trust grid on the *same live
cache*: hold the frozen heads and sources fixed, grow the cache to several
distances, and at each distance measure complete A-vs-B sequence likelihood at
matched trust values (especially m=-1, -0.75, -0.5, 0). That will tell us
whether the observer's control decays with temporal distance or whether this
run simply stayed below the known decoding threshold.
