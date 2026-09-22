# Gate 7 result — identity continuity from behavioral history

Gate 7 passed the preregistered history-continuity boundary.

## Removed dependency

Gate 6 was given a complete multi-hop identity graph:

```text
canonical key -> intermediate alias -> current alias
```

Gate 7 removes the terminal relation edge. The old canonical key can no longer reach either current source through the supplied graph.

Instead the observer has:

```text
persistent trusted canonical key
        +
historical behavioral memory
        +
current anonymous probe observations
```

and must infer the current slot before invoking the unchanged frozen attention reader.

## Probe construction

Each of twelve historical sources has a five-outcome fingerprint.

Within every source pair:

- probe 0 is identical and therefore uninformative by itself;
- probes 1–4 are complementary;
- each current source has exactly one of probes 1–4 corrupted.

A generic minimum-Hamming matcher compares the trusted source's historical fingerprint with the two current anonymous probe vectors.

## Receipt

| measurement | result |
|---|---:|
| families | 6 |
| caches | 12 |
| identity-conditioned reads | 24 |
| historical sources | 12 |
| history-bound accuracy | 1.000 |
| dual-success rate | 1.000 |
| reversed-order dual success | 1.000 |
| mean intended-source total attention | 0.9619481 |
| minimum winning Hamming margin | 2 |
| static-slot attacker | 0.500 |
| one-probe attacker | 0.500 |
| reset-history attacker | 0.500 |
| shuffled-history attacker | 0.000 |
| incomplete-relation-graph attacker | 0.000 |

All projected K/V caches remained unchanged.

## What the attackers establish

The current probe observations are not sufficient on their own: the reset-history condition has all current observations and still falls to 50%.

Historical data is not sufficient without the correct provenance relation: swapping the old fingerprints between the two canonical sources drives success to 0%.

A single current observation is deliberately ambiguous and remains at 50%.

The complete old Gate-6 graph is no longer available and scores 0% because its terminal identity edges have been deleted.

The successful computation therefore requires the combination:

```text
persistent trusted source
        +
old behavioral distinction
        +
new measurements
        -> current binding
```

## Interpretation boundary

This is still a designed probe space. It does not establish open-ended semantic re-identification.

Gate 8 should make measurements costly and allow the observer to choose probes adaptively. If historical state can reduce the number of measurements needed to resolve current identity, the architecture begins to buy something operational rather than merely representational.
