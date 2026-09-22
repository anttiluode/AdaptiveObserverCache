# Gate 6 contract — provenance survives alias / representation drift

## Motivation

Gate 5 removed the hard-coded identity table, but its generic binder still
required exact token-tuple equality:

```text
persistent key == current record key
```

That fails when the same source is renamed or represented by a different
provenance surface form.

Arbitrary renaming without evidence is not identifiable. Gate 6 therefore
tests the useful, falsifiable case where a **representation-update receipt**
states how source keys changed, while trust itself remains private to the
persistent observer state.

## Mechanism

Calibration stores an arbitrary original provenance key:

```text
persistent state = original provenance token tuple
```

Later, alias-update receipts arrive for **both** sources:

```text
old key <-> new key
```

or transitively:

```text
old key <-> alias 1 <-> alias 2
```

The receipts contain no trust labels.

At test time:

```text
persistent original key
        |
        v
generic alias-equivalence graph
        |
        v
current surface key
        |
        v
current record slot
        |
        v
frozen Gate-2/3 positional read mode
```

The persistent trusted key is not rewritten.

## Population

The 12 Gate-5 source identities receive two new surface forms each, producing
24 distinct current alias keys.

Every family is tested at:

- alias generation 1;
- alias generation 2, requiring transitive closure;
- original record order;
- reversed record order.

That produces 24 caches and 48 identity-conditioned reads.

## Attackers

1. **Gate-5 exact equality**
   - has only the original persistent key;
   - receives the current aliased record keys;
   - should find no exact match.

2. **Static slot**
   - assumes the original source index is still the current slot.

3. **Wrong alias map**
   - receives the same amount of alias metadata but cross-binds the two source
     paths.

4. **One-hop-only resolver**
   - receives only original->alias1 edges;
   - is tested on generation-2 aliases and cannot use transitive closure.

## Pass boundary

Gate 6 passes only if:

- at least 12 persistent original keys are exercised;
- at least 24 distinct current alias keys are exercised;
- zero current keys exactly equal their persistent original keys;
- alias-bound accuracy >=95%;
- dual-source success >=95%;
- reversed-order dual success >=95%;
- generation-2/transitive dual success >=95%;
- mean intended-source absolute attention mass >=80%;
- static-slot attacker <=50%;
- Gate-5 exact-key attacker <=10%;
- wrong-alias-map attacker <=10%;
- generation-2 one-hop-only attacker <=10%;
- every projected K/V cache remains unchanged.

## Interpretation boundary

A pass establishes **representation translation from explicit alias
receipts**. It does not establish semantic alias discovery.

If no relation is supplied between an old name and a new arbitrary name, the
identity relation cannot be recovered from nothing.

A useful next gate would replace explicit alias edges with evidence from a
stable descriptor, signed credential, or semantic/relational context and ask
whether the binding can be inferred rather than handed over.
