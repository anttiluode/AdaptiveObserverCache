# Gate 7 contract — infer identity continuity from history

## Motivation

Gate 6 lets a persistent trust state survive arbitrary surface-name changes,
but it is given a complete canonical-to-alias relation graph.

Gate 7 removes the terminal relation evidence. Current aliases are therefore
not reachable from canonical keys through the supplied graph.

The question becomes:

> Can prior observations identify which current anonymous source continues
> the trusted historical source?

## State decomposition

The test keeps three objects separate:

```text
persistent trusted canonical key
        +
historical behavioral memory for all sources
        +
current anonymous probe observations
        |
        v
inferred current slot
        |
        v
frozen Gate-2/3 positional reader
```

No model weights or projected K/V are changed.

## Historical fingerprints

Each canonical source has five previously observed probe outcomes.

Within every source pair:

- probe 0 is deliberately identical;
- probes 1–4 form complementary codewords.

At the current epoch each anonymous source is probed with the same five probes,
but exactly one of probes 1–4 is corrupted. The corruption depends on current
slot/order, not source identity or trust.

A generic nearest-history matcher uses Hamming distance. No identity table or
alias edge is used.

## Population

The six Gate-6 canonical source pairs and their unrelated terminal aliases are
reused.

Every family is tested in both source orders:

```text
6 families x 2 orders x 2 trusted identities = 24 reads
```

## Attackers

1. **Incomplete Gate-6 relation graph**
   - retains canonical->middle edges;
   - removes middle->current-alias edges;
   - cannot reach either current record.

2. **Static slot**
   - assumes canonical source 0 is always slot 0 and source 1 is always slot 1.

3. **One-probe observer**
   - receives only probe 0;
   - both sources deliberately have the same probe-0 result.

4. **Reset-history observer**
   - sees all current probe outcomes but has forgotten the historical source
     fingerprint that gives those outcomes identity meaning.

5. **Shuffled historical memory**
   - receives the same complete five-probe evidence;
   - swaps the two historical fingerprints inside each source family.

## Pass boundary

Gate 7 passes only if:

- at least 12 historical source fingerprints are exercised;
- history-bound reader accuracy >=95%;
- dual-source case success >=95%;
- reversed-order dual success >=95%;
- mean intended-source absolute attention mass >=80%;
- every winning historical match has a Hamming-distance margin >=1;
- static-slot attacker <=50%;
- one-probe attacker <=50%;
- reset-history attacker <=50%;
- shuffled-history attacker <=10%;
- incomplete-graph attacker <=10%;
- every projected K/V cache remains unchanged.

## Interpretation boundary

A pass establishes **behavioral continuity from stored observations**. It does
not establish general semantic identity recognition.

The probe battery and persistence assumption are supplied. A stronger next
gate would make probes costly and let the observer choose which measurement to
buy, connecting this mechanism to active observability / WhatToLookAt.
