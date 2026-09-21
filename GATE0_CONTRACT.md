# Gate 0 contract — moving reader, fixed memory

## Claim under test

A persistent observer state can change which information is recovered from the same fixed cache, and contradictory evidence can update that observer so the next read changes, without modifying the cache.

## Frozen before the receipt

- Cache: four immutable entries, two provenance A and two provenance B.
- Present: `(1, 1)` for every read.
- Initial adaptive/static observer: `(+2, -2)` in log-gain coordinates.
- Opposite observer for the paired read: `(-2, +2)`.
- Attention temperature: `0.2`.
- Surprise update magnitude: `3.0`.
- Switch schedule: `AAAA BBBB AAAA BBBB`.
- Read budget: exactly one cache read per step.

## Primary outcomes

1. Paired-observer test:
   - A-biased state selects provenance A and predicts +1.
   - B-biased state selects provenance B and predicts -1.
2. Switching-world test:
   - adaptive accuracy >= 0.75;
   - adaptive - static accuracy >= 0.20;
   - after each regime switch, adaptation requires at most one contradicted read.
3. Integrity:
   - cache checksum before and after the run is identical.

## Kill conditions

The stronger claim fails if:

- changing observer state does not change the selected provenance under identical cache/present;
- the adaptive reader requires cache mutation;
- the static equal-budget reader performs equally well;
- recovery requires more than one contradiction under the frozen update rule;
- the cache checksum changes.

## Interpretation boundary

A pass establishes only a synthetic observer-in-the-loop primitive. It does not establish transformer usefulness, KV-cache editing, continual learning, or improved reasoning.
