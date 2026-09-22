# Gate 5 contract — arbitrary provenance key, no identity table

## Motivation

Gate 4 repaired source-order swaps by separating persistent semantic identity
from current source address. But its binder still contained a frozen
Alice/Bob token table.

Gate 5 removes that table.

## Mechanism

During calibration, the slow state copies the selected source record's
provenance key:

```text
state = provenance_key(selected_source)
```

At test time:

```text
persistent provenance key
          |
          v
generic equality binder
          |
          v
current source slot
          |
          v
frozen Gate-2/3 slot read mode
```

The binder has no list of valid identities.

## Provenance keys

Six identity pairs not used by Gate 4 are tested:

- Carol / Dave
- Eve / Frank
- Grace / Henry
- Iris / Jack
- Kira / Liam
- Mona / Nate

The implementation supports provenance keys that tokenize to one or multiple
tokens. The persistent key is the complete token tuple.

Each pair is tested in both record orders, producing 12 caches and 24
identity-conditioned reads.

## Calibration boundary

The environment may identify which source was trusted during calibration.
The mechanism may then copy that record's provenance key into persistent
state.

During test reads:

- no trust label is visible;
- the persistent key is unchanged;
- the generic binder compares it with current record provenance keys;
- no attention K/V geometry is used for binding;
- the pretrained K/V cache is never rewritten.

## Attackers

1. **Static calibration-slot attacker**
   - assumes key0 always remains slot 1 and key1 slot 2.

2. **Wrong persistent key attacker**
   - stores the other source's valid provenance key;
   - uses the same generic binder.

## Pass boundary

Gate 5 passes only if:

- at least 12 distinct persistent provenance token tuples are exercised;
- generic bound-reader accuracy >=95%;
- >=95% of all cases succeed for both keys;
- >=95% of reversed-order cases succeed for both keys;
- mean intended-source absolute attention mass >=80%;
- static calibration-slot attacker <=50%;
- wrong-key attacker <=10%;
- every projected K/V cache remains unchanged.

## Interpretation boundary

A pass would establish a generic **symbolic provenance-key binder**, not
semantic identity recognition.

The next attacker is surface-form drift: the same source may appear under a
different alias or representation, where exact key equality no longer works.
