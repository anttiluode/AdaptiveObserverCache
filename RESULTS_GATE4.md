# Gate 4 result — persistent identity, current address binding

Gate 4 passed its preregistered factorization test.

## Mechanism

Gate 3 showed that the frozen pretrained observer coordinate is a strong source-slot reader, but not an order-invariant source-identity coordinate.

Gate 4 therefore composes three objects:

```text
persistent trusted identity
        -> current identity-to-slot binding
        -> frozen positional query mode
```

The slow identity state is either Alice or Bob and does not change during the test sequence.

The binder receives only:

- explicit source-record boundaries;
- the first provenance token ID in each record;
- frozen Alice/Bob token IDs.

It does not inspect trust, attention K/V geometry, or test outcomes.

## Receipt

| measurement | result |
|---|---:|
| test cases | 12 |
| identity-conditioned reads | 24 |
| bound-reader identity accuracy | 1.000 |
| dual-identity case success | 1.000 |
| reversed-order dual success | 1.000 |
| persistent Alice sequence | 1.000 |
| persistent Bob sequence | 1.000 |
| mean intended-source total attention | 0.9968126 |
| static Alice→slot1/Bob→slot2 attacker | 0.500 |
| inverted binder attacker | 0.000 |

All projected K/V caches remained unchanged.

## Causal interpretation

The static attacker succeeds on the six Alice-first/Bob-second cases and fails on the six reversed cases, giving exactly 50%.

The correct binder succeeds in both orders.

The inverted binder gets 0%, showing that the binding operation is not decorative: its output determines which of the two frozen positional modes is used.

## What Gate 4 establishes

The order-swap failure from Gate 3 can be repaired without retraining the transformer, changing K/V, or discovering a new attention direction.

The failure came from conflating:

```text
semantic provenance identity
```

with:

```text
current memory address / source slot
```

Separating them lets the semantic state persist while the address changes.

## What Gate 4 does not establish

The provenance binder is explicit. Alice and Bob are recognized by exact frozen token IDs.

Gate 5 should therefore attack the binder itself: use new identities, changed provenance markers, or a binding relation that cannot be solved by exact token lookup.
