# Qwen distance sweep: first visible-filler result

Date: 2026-09-22

Receipt: \`results/qwen_observer_distance_sweep.json\`

## What ran

- Model: \`Qwen/Qwen3-8B\`
- Transformers: \`5.15.0\`
- Frozen causal observer heads:
  - L24/Q29
  - L24/Q23
  - L30/Q11
  - L18/Q28
- Source spans stayed fixed at A=[45,62), B=[72,87).
- One canonical DynamicCache grew from 131 to 411 tokens.
- Source-cache integrity stayed true.
- No observer update hit the norm cap.

The first distance manipulation used deterministic visible chat filler. Its
target of +256 tokens landed at +280 because complete filler turns were used.

## Likelihood result

Primary readout is complete-answer mean log-probability margin:

\`\`\`text
margin = mean log p(A) - mean log p(B)
\`\`\`

At the anchor:

\`\`\`text
m=-1   -1.024999
m= 0   -0.949997
m=+1   -0.599937
endpoint swing = 0.425061
\`\`\`

After +280 visible filler tokens:

\`\`\`text
m=-1   +0.937514
m= 0   +2.312500
m=+1   +2.656251
endpoint swing = 1.718737
\`\`\`

The endpoint swing was about 4.04x the anchor swing.

## The important asymmetry

At distance zero:

\`\`\`text
B-control relative to neutral: -0.075001
A-control relative to neutral: +0.350060
\`\`\`

At +280:

\`\`\`text
B-control relative to neutral: -1.374986
A-control relative to neutral: +0.343751
\`\`\`

The A-side perturbation retained almost the same downstream effect. The B-side
perturbation became much stronger. Therefore there is no evidence here that
observer causal leverage simply faded with cache age.

But the neutral model itself moved from a B-favoring margin of about -0.95 to an
A-favoring margin of about +2.31. That is a shift of about +3.26 log-probability
units per candidate token.

So the visible-filler run changed two things at once:

1. source/query positional distance;
2. the intervening conversational trajectory.

The run therefore does **not** isolate temporal distance.

## Verdict

**Mechanical long-cache control survived. Positional-distance isolation failed.**

This is still a useful result. The observer continued to alter the downstream
candidate likelihood with intact historical K/V, and at +280 tokens its total
A/B endpoint swing was larger, not smaller.

The next experiment must determine whether the large neutral-basin flip came
from readable filler content or from positional aging itself.

## Next discriminator: masked positional spacers

\`qwen_observer_masked_distance.py\` advances the DynamicCache with tokens whose
attention-mask entries are zero.

For Qwen3, later position IDs are derived from the number of tokens already in
the cache. Thus these spacer rows move the later query to larger RoPE positions,
while the persistent attention mask prevents later queries from reading the
spacer K/V rows.

The small control is:

\`\`\`bash
python3.13 qwen_observer_masked_distance.py
\`\`\`

It repeats distance 0 and +256 with the same trust values -1, 0, +1.

Interpretation:

- if the neutral basin remains near the distance-zero value while observer
  control survives, the previous flip was caused mainly by readable filler
  trajectory;
- if the neutral basin or observer swing changes strongly even with unreadable
  spacers, cache position / RoPE distance itself is load-bearing.
