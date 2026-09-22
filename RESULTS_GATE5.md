# Gate 5 result — generic provenance-key binding

Gate 5 passed the preregistered no-identity-table test.

## Mechanism

Gate 4 used a frozen Alice/Bob token table. Gate 5 replaces that table with a generic persistent key:

```text
calibration receipt
      -> copy selected record provenance key
      -> carry token tuple through time
      -> equality-bind it to a current record
      -> use frozen source-slot query mode
```

The binder contains no list of source identities.

## Test population

Six new provenance pairs:

- Carol / Dave
- Eve / Frank
- Grace / Henry
- Iris / Jack
- Kira / Liam
- Mona / Nate

Each pair is tested in both source orders.

## Receipt

| measurement | result |
|---|---:|
| families | 6 |
| caches | 12 |
| distinct persistent keys | 12 |
| identity-conditioned reads | 24 |
| generic bound accuracy | 1.000 |
| dual-success rate | 1.000 |
| reversed-order dual success | 1.000 |
| mean intended-source total attention | 0.9729603 |
| static calibration-slot attacker | 0.500 |
| wrong persistent-key attacker | 0.000 |

All projected K/V caches remained unchanged.

## Interpretation

The slow observer state no longer needs a designed mode such as “Alice” or “Bob.” It can carry an arbitrary symbolic provenance key captured from an earlier trusted record.

The same generic operation then converts that persistent key into a current address before invoking the already-frozen positional reader.

This establishes:

```text
persistent source key != current source address
```

and shows the two can be joined later by a generic equality binding operation.

## Boundary

This is still exact symbolic identity. If a source is renamed, aliased, or represented by a different provenance surface form, exact key equality will fail.

That is the next useful gate.
