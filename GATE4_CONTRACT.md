# Gate 4 contract — identity state, address binding, positional reader

## Motivation

Gate 3 transferred perfectly across seven unseen non-swapped caches, but both
order-swap attackers reversed the read modes.

That result says the reusable Gate-2/3 coordinate behaves like:

```text
read source slot 1  <->  read source slot 2
```

not:

```text
trust Alice  <->  trust Bob
```

Gate 4 therefore stops asking one vector to do two jobs.

## Factored mechanism

```text
persistent trusted identity z
          |
          v
current identity -> slot binding B
          |
          v
slot 1 / slot 2
          |
          v
frozen Gate-2/3 query mode
```

The attention cache is never rewritten.

## What is frozen

From Gate 2/3:

- DistilGPT2 revision `2290a62`;
- layer 3 / head 2;
- the one-dimensional slot-read direction;
- the two scalar slot-read modes.

No test prompt may retune those objects.

## Persistent state

The slow state is semantic provenance identity:

```text
z in {Alice, Bob}
```

For each evaluation sequence, `z` is assumed to have been set by an earlier
receipt. It remains unchanged across all content changes and source-order
changes in the Gate-4 test sequence.

No trust label is available during test reads.

## Binder

Gate 4 intentionally uses an explicit, auditable provenance channel rather
than pretending identity recognition is the hard part.

Each source record starts with a provenance token, Alice or Bob. The binder
is frozen to those two token IDs. On each cache it receives:

- the two source-record boundaries;
- the first token ID of each source record.

It returns:

```text
Alice -> current slot
Bob   -> current slot
```

It may not inspect trust, attention K/V geometry, or test outcomes.

This gate tests the **factorization**, not general provenance recognition.

## Test suite

Six content families are each tested in both orders:

```text
Alice first, Bob second
Bob first, Alice second
```

for 12 unseen-cache cases total and 24 identity-conditioned reads.

The content includes same-layout statements and paraphrased statement forms.

## Per-read success

The selected identity must receive:

- >=20% absolute total attention;
- >=80% of Alice-vs-Bob source attention share;
- the predicted source identity must match persistent state `z`.

## Attackers

1. **Static identity-to-slot mapping**
   - Alice always -> slot 1
   - Bob always -> slot 2
   - ignores current order.

2. **Inverted binder**
   - receives the same provenance metadata;
   - deliberately chooses the opposite slot.

These distinguish persistent semantic identity from a fixed positional mode.

## Pass boundary

Gate 4 passes only if:

- bound-reader identity accuracy >=95%;
- >=95% of cases succeed for both identities;
- >=95% of reversed-order cases succeed for both identities;
- persistent Alice sequence accuracy >=95%;
- persistent Bob sequence accuracy >=95%;
- mean target absolute attention mass >=80%;
- static identity-to-slot attacker <=50%;
- inverted binder attacker <=10%;
- every projected K/V cache remains unchanged.

## Interpretation boundary

A pass would show that the Gate-3 order failure can be repaired by separating:

```text
what should be trusted
```

from:

```text
where that source currently lives
```

It would **not** show that identity binding itself has been learned or
discovered. Gate 5 should attack the explicit binder by changing provenance
identities or removing the exact identity token channel.
