# Gate 6 result — persistent provenance through alias drift

Gate 6 passed the preregistered multi-hop alias-binding test.

## Mechanism

The slow trust state stores an old canonical provenance key. Test records use unrelated current aliases, so exact key equality no longer works.

An independent identity relation graph provides continuity:

```text
canonical trusted key
      -> intermediate alias relation
      -> current terminal alias
      -> current record slot
      -> frozen Gate-2/3 positional reader
```

The relation graph is trust-independent and never inspects attention K/V.

## Population

Six canonical provenance pairs are tested in both source orders, for 12 caches and 24 identity-conditioned reads.

Each canonical source reaches its current record only through a path of at least two relation edges.

## Receipt

| measurement | result |
|---|---:|
| families | 6 |
| cases | 12 |
| identity-conditioned reads | 24 |
| distinct canonical keys | 12 |
| relational bound accuracy | 1.000 |
| dual-success rate | 1.000 |
| reversed-order dual success | 1.000 |
| mean intended-source total attention | 0.9606801 |
| minimum relation hops | 2 |
| exact-key attacker | 0.000 |
| one-hop alias attacker | 0.000 |
| shuffled-relation attacker | 0.000 |

All projected K/V caches remained unchanged.

## Interpretation

The persistent state no longer depends on the source retaining the same visible name.

The current system now factors four things:

```text
what was trusted
    !=
how that identity is currently named
    !=
where the current record is located
    !=
how the frozen attention head reads that location
```

That decomposition is the useful result.

## Boundary

The alias relation graph is supplied metadata. Gate 6 therefore does not establish relation discovery.

The next gate should delete or corrupt relation evidence and ask whether identity continuity can be recovered from prior observations, behavior, or active probes.
