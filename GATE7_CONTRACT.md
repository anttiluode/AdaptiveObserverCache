# Gate 7 contract — active identity recovery with one audit round

## Motivation

Gate 6 can follow a trusted identity through arbitrary alias changes, but only
when an identity-relation graph supplies the continuity.

Gate 7 deletes the final alias relation.

The system knows which two historical canonical sources could correspond to
the two current aliases, but it does not know their current permutation.

## Historical identity evidence

Each canonical source has prior binary audit receipts for eight challenge
types.

The receipt bits are generated deterministically from source identity and
challenge type so the experiment does not hand-author a special answer table
for each pair.

The current alias preserves the underlying source's challenge behavior even
though its surface name and memory address have changed.

## Equal one-round budget

For each test cache both adaptive and fixed conditions receive exactly one
challenge round:

    choose one challenge type
       -> send it to both current sources
       -> receive one bit from each
       -> infer binding if possible
       -> perform the frozen attention read

The active condition may choose its challenge from the two candidate sources'
historical profiles.

The fixed attacker must use the single global challenge type that separates
the largest number of source pairs.

If the fixed challenge does not distinguish a pair, it falls back to the old
static slot assumption.

## Frozen attention reader

Gate 7 does not modify:

- DistilGPT2 revision 2290a62;
- layer 3 / head 2;
- Gate-2/3 slot-read direction;
- first-slot and second-slot scalar modes;
- projected K/V.

## Population

The six canonical pairs and twelve current alias caches from Gate 6 are reused
in both source orders.

No canonical-to-current terminal alias edge is supplied to the binder.

## Attackers

1. Best global fixed audit challenge under the same one-round budget.
2. Passive static source-slot mapping with no audit.
3. Shuffled historical profile labels while preserving the same active
   challenge and current responses.

## Pass boundary

Gate 7 passes only if:

- active identity-conditioned read accuracy >=95%;
- active dual-success rate >=95%;
- reversed-order dual-success rate >=95%;
- mean intended-source total attention >=80%;
- every active challenge actually separates the pair;
- at least three different challenge types are selected across the population;
- active accuracy beats the best fixed one-round probe by >=10 percentage
  points;
- passive static-slot accuracy <=50%;
- shuffled-history accuracy <=10%;
- every projected K/V cache remains unchanged.

## Interpretation boundary

A pass establishes active identity disambiguation from stable behavioral audit
history under a strict matched probe budget.

It does not establish that arbitrary real-world sources expose such a stable
challenge-response signature. The next gate should add noisy or drifting audit
behavior and test whether the binder can represent uncertainty instead of
making a hard one-shot identity decision.
