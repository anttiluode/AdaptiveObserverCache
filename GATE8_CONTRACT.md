# Gate 8 contract — history buys sensing efficiency

## Motivation

Gate 7 receives all five current behavioral probes for free.

Gate 8 turns observation into a budget. Each probe costs one unit. The observer
must choose which measurement to buy and stop as soon as current source
identity is resolved.

The claim is not that active probing improves final accuracy over an exhaustive
reader. The claim is:

> persistent history can make **the same successful binding cheaper** by
> telling the observer which measurements can distinguish its hypotheses.

## Probe world

Every source family has eight possible behavioral probes.

Historical source fingerprints differ on exactly three family-specific probe
indices and are identical on the other five.

At the current epoch:

- the two anonymous source slots retain those historical behaviors;
- one of the three informative probes is corrupted by forcing slot 1 to emit
  slot 0's response;
- the corruption depends on current family/order, never on trust.

Therefore:

- five probes are useless by historical construction;
- one informative probe is currently ambiguous;
- two informative probes remain capable of resolving identity.

## Active observer

The observer has the historical fingerprints for the two candidate sources.

It ranks probes structurally:

```text
probe is worth buying iff historical candidate responses differ
```

It buys those probes one at a time and stops at the first current response that
uniquely binds the trusted historical source to a current slot.

The final retrieval uses the same frozen DistilGPT2 positional reader as Gates
2–7.

## Matched random-order baseline

The random baseline receives:

- the same historical fingerprints;
- the same current probe field;
- the same stopping rule;
- the same cost per probe.

Only probe order differs.

Its expected cost is computed exactly over all 8! possible probe orders, not
estimated from lucky random seeds.

## Attacker

Shuffled historical fingerprints preserve all current observations and the
same active policy but attach old behavior to the wrong canonical identities.

## Pass boundary

Gate 8 passes only if:

- active final read accuracy >=95%;
- dual-source case success >=95%;
- reversed-order dual success >=95%;
- mean intended-source absolute attention mass >=80%;
- mean active probe cost <=1.50;
- no active run costs more than 2 probes;
- matched random-order expected cost >=2.80;
- active/random mean cost ratio <=0.55;
- shuffled-history accuracy <=10%;
- every projected K/V cache remains unchanged.

## Interpretation boundary

This is still a designed finite probe menu. The observer is not inventing new
experiments.

A stronger next gate would attach unequal probe costs or information values
and choose by expected information gain per unit cost, connecting directly to
AnotherOddThing and the earlier computational-foveation budget.
