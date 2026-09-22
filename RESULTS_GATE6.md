# Gate 6 result — alias translation works, frozen reader misses one absolute-engagement case

Gate 6 **fails** its preregistered scientific boundary. The failure is retained rather than tuned away.

## Mechanism

The persistent state remains the original arbitrary provenance key. Explicit alias-update receipts map both sources symmetrically and carry no trust label:

```text
persistent original key
        -> alias equivalence graph
        -> current alias
        -> current slot
        -> frozen Gate-2/3 positional reader
```

Generation 2 requires transitive closure:

```text
old key <-> alias 1 <-> alias 2
```

## Receipt

| measurement | result |
|---|---:|
| families | 6 |
| caches | 24 |
| identity-conditioned reads | 48 |
| persistent original keys | 12 |
| distinct current alias keys | 24 |
| exact old/new key matches | 0 |
| alias-bound read accuracy | 47 / 48 = 0.97917 |
| dual-success rate | 23 / 24 = 0.95833 |
| reversed-order dual success | 12 / 12 = 1.000 |
| generation-2 dual success | 11 / 12 = 0.91667 **FAIL** |
| mean intended-source mass | 0.95855 |
| static-slot attacker | 0.47917 |
| Gate-5 exact-key attacker | 0.000 |
| wrong-alias-map attacker | 0.000 |
| generation-2 one-hop-only attacker | 0.000 |

All projected K/V caches remained unchanged.

## The single miss

The only failed read is:

```text
family: package
generation: 2
order: original
persistent source: Liam
current alias: Luke
resolved slot: 1
frozen observer mode: -4.0
```

The binder is correct and the relative source preference is overwhelming:

```text
target A-vs-B share = 0.999999999988
```

but the absolute attention on the intended source is only:

```text
target mass = 0.174361974
```

below the frozen 0.20 boundary.

So this is **not an alias-resolution failure**. It is a reader-engagement failure: the frozen global slot-2 direction knows which source it prefers, but in this cache it does not allocate enough total attention to either source.

## Interpretation

Gate 6 therefore establishes most of the alias-translation mechanism but rejects the stronger claim that one frozen global positional read vector is sufficient after arbitrary representation drift.

That suggests a sharper next object:

```text
persistent provenance relation
        +
current alias/address binding
        +
locally recalibrated read direction
```

The persistent object should say **what relation/source to recover**. The concrete steering tangent may need to be reconstructed from the current cache.

This is the same observer-in-the-loop lesson in a stricter form: representation translation can succeed while the measuring geometry itself has become stale.
