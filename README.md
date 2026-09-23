# AdaptiveObserverCache

**Can the same fixed memory reveal different useful information because the reader itself has persistent state — and can new evidence change that reader without rewriting the memory?**

The working object is:

```text
fixed historical field C
fixed present h
persistent observer state m
        |
        v
query geometry Q(h, m)
        |
        v
read fixed memory C
        |
        v
receipt arrives after the read
        |
        v
update m
        |
        v
next read changes without rewriting memory
```

The cache is historical material. The observer state is the changing measuring apparatus.

## Gate 0 — synthetic primitive

Gate 0 established the smallest model-free version: identical cache and identical present can reveal different provenance depending on persistent reader state. On the alternating `AAAA BBBB AAAA BBBB` schedule, the static reader gets 50% while the adaptive reader gets 81.25%.

Run:

```bash
python gate0_experiment.py
```

## Gate 1 — frozen PyTorch attention

Gate 1 moved the primitive into an actual frozen attention calculation:

```text
q_base = W_Q h
q'     = q_base + 2.5 * m * u
scores = K q' / sqrt(d_head)
read   = softmax(scores) V
```

`W_Q/W_K/W_V`, projected K/V and the present hidden state are frozen. Only one scalar observer state changes the query.

Receipt:

```text
best fixed observer accuracy = 0.5000
adaptive observer accuracy   = 0.8125
adaptive advantage           = +0.3125

same K/V + m=+1 -> 96.22% A mass
same K/V + m=-1 -> 96.22% B mass
```

Projected-cache and parameter digests are byte-identical before/after.

See [GATE1_CONTRACT.md](GATE1_CONTRACT.md) and [RESULTS_GATE1.md](RESULTS_GATE1.md).

## Gate 2 — real pretrained DistilGPT2 K/V

Gate 2 removes the hand-built Q/K/V geometry.

Pinned model:

```text
distilbert/distilgpt2
revision 2290a62
```

The prompt is tokenized normally, DistilGPT2 produces hidden states and Q/K/V, and one real attention head is selected by a preregistered cache-geometry rule. The model weights are never trained or changed.

The observer is still one-dimensional:

```text
u  = normalize(mean(K_A) - mean(K_B))
q' = q_pretrained + m * ||q_pretrained|| * u
```

with `|m| <= 4`.

A fixed 161-point scalar grid is searched on every pretrained head. For each head, one scalar state maximizes **absolute total attention mass** on source A and one maximizes absolute total attention mass on source B. The chosen head maximizes the worse of those two target masses.

### Why Gate 2 was hardened before merge

The first Gate-2 receipt exposed a denominator loophole.

A B-mode could win almost all of the normalized A-vs-B source share while both source spans received essentially zero attention. That technically satisfied the first contract but was not a real retrieval.

So Gate 2 was strengthened before merge: each mode must put at least **20% of all attention** on its intended source, not merely beat the other source.

### Hardened Gate 2 receipt

CI selected:

```text
layer = 3
head  = 2
head dimension = 64
```

Modes:

```text
A mode: m = +1.15
    total attention on A = 1.000000
    A-vs-B share on A    = 0.999999983

B mode: m = -4.00
    total attention on B = 1.000000
    A-vs-B share on B    = 0.999999999984
```

Protocol:

```text
block A: 1 calibration read -> 3 blind reads
block B: 1 calibration read -> 3 blind reads
block A: 1 calibration read -> 3 blind reads
block B: 1 calibration read -> 3 blind reads
```

Only the first read of each block receives the trusted-source receipt, and it arrives **after** the read. The next three reads receive no truth signal.

Results:

```text
blind adaptive test accuracy = 12 / 12 = 1.000
best fixed reader            =  6 / 12 = 0.500
reset-observer control       =  6 / 12 = 0.500
adaptive advantage           = +0.500
```

The projected K/V digest was identical before/after:

```text
79a3fa9e0fbb26e50b3acd5e03b90a74c8fcf1b87c650a5ed6578f36a0543cde
```

Target-attention parameter digest:

```text
ac80d2a0f80207e5a62aa81ff34b386bd19fef9a255d4aabf33e132723cdb824
```

See [GATE2_CONTRACT.md](GATE2_CONTRACT.md) and [RESULTS_GATE2.md](RESULTS_GATE2.md).

Run:

```bash
pip install -r requirements-gate2.txt --extra-index-url https://download.pytorch.org/whl/cpu
python gate2_experiment.py
```

## Gate 3 — freeze the observer, move the cache

Gate 3 freezes the Gate-2 layer/head, the one-dimensional observer direction, and the two scalar read modes **before projecting any test prompt**. Seven new caches then receive only their ordinary pretrained q/K/V; no test-cache geometry is allowed to rebuild the observer.

The transfer result is unexpectedly clean:

```text
same-layout content changes     5 / 5 dual-success
paraphrase / wording changes    2 / 2 dual-success
all non-swapped transfer        7 / 7 dual-success

mean worse target mass          0.9999948
mean cos(local key axis, u_G2)  0.9441
all projected test caches       unchanged
```

So the Gate-2 coordinate is not merely a one-cache steering trick. Across these unseen prompts, the frozen A mode still puts essentially all attention on the first source span and the frozen B mode puts essentially all attention on the second.

### The order-swap attacker identifies the coordinate

Both order-swapped prompts fail **0 / 2**.

When Bob is moved to the first source slot and Alice to the second, the local A→B key axis flips relative to the frozen Gate-2 direction:

```text
vault order swap   cosine = -0.8748
garden order swap  cosine = -0.8470
```

and the observer modes reverse which identity they retrieve.

That changes the interpretation of the mechanism:

```text
not yet:  persistent "trust Alice / trust Bob" coordinate
closer to: persistent "read source slot 1 / read source slot 2" coordinate
```

This is still useful. A stable address axis across content and paraphrase is exactly the sort of thing a persistent observer can exploit. But the next mechanism must bind **who/what occupies an address** separately from the positional read coordinate.

See [GATE3_CONTRACT.md](GATE3_CONTRACT.md) and [RESULTS_GATE3.md](RESULTS_GATE3.md).

Run:

```bash
python gate3_experiment.py
```

## Current boundary

The chain now establishes:

```text
Gate 0  synthetic moving reader
Gate 1  frozen attention moving query
Gate 2  real pretrained K/V + persistent calibration state
Gate 3  frozen read coordinate transfers across new caches,
        but tracks source slot/order rather than source identity
```

The next gate should therefore **not** search another better steering vector. It should factor the problem:

```text
identity / provenance binding   +   reusable positional read coordinate
```

so that swapping source order changes the binding, not the meaning of the observer state.

## Gate 4 — persistent identity + current address binding

Gate 3's order-swap failure was a representational conflation: one state was being asked to encode both **who should be trusted** and **where that source currently lives**.

Gate 4 factors those roles:

```text
persistent trusted identity
        |
        v
current identity -> source-slot binding
        |
        v
slot 1 / slot 2
        |
        v
frozen Gate-2/3 positional read mode
```

The binder is intentionally explicit and auditable. Each source record begins with a provenance token (Alice or Bob); frozen token IDs map that identity to its current slot. The binder never sees trust labels, K/V geometry, or test outcomes.

Across six content families in both source orders:

```text
cases                              12
identity-conditioned reads         24
bound identity accuracy          1.00
dual-identity case success       1.00
reversed-order dual success      1.00
persistent Alice sequence        1.00
persistent Bob sequence          1.00
mean intended-source mass      0.9968

static Alice->slot1/Bob->slot2   0.50
inverted binder                  0.00
```

Every projected K/V cache remains unchanged.

The result is deliberately narrower than "the model learned provenance identity." It establishes the **factorization**:

```text
slow semantic choice != current address
```

and shows that a persistent semantic choice can survive permutations when a separate binding operation converts identity into the reusable positional read coordinate.

See [GATE4_CONTRACT.md](GATE4_CONTRACT.md) and [RESULTS_GATE4.md](RESULTS_GATE4.md).

### Next boundary

Gate 4's binder is explicit metadata. That is useful in real systems—provenance often *is* metadata—but it is also the obvious next attacker.

Gate 5 should remove the exact Alice/Bob token lookup and ask whether a binding rule calibrated on some source identities can bind **new identities or changed provenance markers** without knowing their current slot in advance.

That is now the interesting problem:

```text
persistent relation / trust
        +
learned or inferred current binding
        +
reusable read coordinate
```

## Gate 5 — arbitrary provenance keys

Gate 4 still knew an Alice/Bob identity table. Gate 5 removes it.

The slow state is now simply the token tuple copied from whichever source provenance field was selected during calibration:

```text
persistent arbitrary key
        |
        v
generic key-equality binder
        |
        v
current slot
        |
        v
frozen positional read mode
```

The binder has no predefined identity vocabulary. Six new provenance pairs—Carol/Dave, Eve/Frank, Grace/Henry, Iris/Jack, Kira/Liam, and Mona/Nate—exercise twelve distinct persistent keys, each in both record orders.

Receipt:

```text
families                           6
cases                             12
unique persistent keys            12
identity-conditioned reads        24

generic bound accuracy          1.00
dual-success rate              1.00
reversed-order dual success    1.00
mean intended-source mass    0.97296

static calibration-slot         0.50
wrong persistent key            0.00
```

Every projected K/V cache remains unchanged.

This is a more general binding primitive than Gate 4, but it is still symbolic: the same source must present the same provenance key. The obvious next attacker is **alias drift**—the same underlying source represented by a different surface key.

See [GATE5_CONTRACT.md](GATE5_CONTRACT.md) and [RESULTS_GATE5.md](RESULTS_GATE5.md).

## Gate 6 — persistent identity through alias drift

Gate 5 still assumed the current record exposed exactly the same provenance key that had been stored earlier. Gate 6 lets the visible source identity change completely.

The slow state keeps the old canonical provenance key, while an independent relation graph tracks identity continuity:

```text
persistent canonical trust key
        +
changing identity relation graph
        |
        v
current unrelated surface alias
        |
        v
current source slot
        |
        v
frozen positional read operator
```

The current prompt no longer contains the canonical key. Example:

```text
persistent state: Carol

relation metadata:
Carol -> CSeven -> Orion

current source record:
Orion reports ...
```

Every valid identity path is at least two relation hops, so exact equality and one-hop alias lookup are incapable of solving the gate.

Across six provenance pairs, both source orders, and 24 identity-conditioned reads:

```text
relational bound accuracy         1.00
dual-success rate                 1.00
reversed-order dual success       1.00
mean intended-source mass       0.96068
minimum relation hops                2

exact-key attacker                0.00
one-hop alias attacker            0.00
shuffled relation graph           0.00
```

Every projected K/V cache remains unchanged.

This moves the object one step beyond a persistent key-value lookup: the slow state can survive a change in the source's surface name as long as identity continuity is represented somewhere independently.

See [GATE6_CONTRACT.md](GATE6_CONTRACT.md) and [RESULTS_GATE6.md](RESULTS_GATE6.md).

### Next boundary

Gate 6 is still given the identity-relation graph. That is now the obvious dependency to attack.

The next useful question is:

```text
if one alias edge disappears or becomes ambiguous,
can history recover which current source continues the trusted identity?
```

That would turn the binding layer from supplied metadata into an inferred, auditable hypothesis.

## Gate 7 — identity continuity from behavioral history

Gate 6 still received a complete identity relation graph. Gate 7 deletes the terminal alias relation and asks whether earlier observations can identify which current anonymous source is the continuation of a trusted historical source.

The mechanism separates:

```text
persistent trusted canonical key
        +
historical behavioral fingerprint
        +
current anonymous probe responses
        |
        v
inferred current source slot
        |
        v
frozen positional attention reader
```

Each of the twelve historical sources has five prior probe outcomes. Within a source pair, probe 0 is deliberately identical, while probes 1–4 form complementary codes. At the current epoch, every anonymous source receives the same five probes with exactly one non-prefix probe corrupted. A generic minimum-Hamming matcher performs the continuity inference.

CI receipt:

```text
families                              6
cases                                12
identity-conditioned reads           24

history-bound accuracy             1.00
dual-success rate                  1.00
reversed-order dual success        1.00
mean intended-source mass       0.96195
minimum history-match margin          2

static slot                        0.50
one ambiguous probe                0.50
reset / forgotten history          0.50
shuffled historical fingerprints   0.00
incomplete Gate-6 graph            0.00
```

Every projected K/V cache remains unchanged.

The key result is not that five synthetic probe bits can identify two sources. It is the dependency structure:

```text
current observations alone are insufficient
historical record alone is insufficient
trust key alone is insufficient

history + current measurement + persistent trust
        -> current address
        -> useful retrieval
```

See [GATE7_CONTRACT.md](GATE7_CONTRACT.md) and [RESULTS_GATE7.md](RESULTS_GATE7.md).

### Next boundary

Gate 7 gets all five current probes for free. That is now the obvious luxury to remove.

The next gate should give each probe a cost and require the observer to choose **which measurement to buy next** from its current uncertainty. That reconnects AdaptiveObserverCache directly to `WhatToLookAt` / active observability:

```text
history
  -> uncertainty over current identity
  -> choose informative probe
  -> update observer
  -> stop when identity is resolved
  -> read memory
```

The useful question is no longer just “can history recover identity?” but:

> **Can history make sensing cheaper by telling the observer what it needs to measure?**

## Relationship to recent repos

- **ReadWrite** — a state may be invisible until the right intervention/query is applied.
- **WhatToLookAt** — memory changes which later measurement is worth buying.
- **PredictiveHKT** — a changing representation can masquerade as a changing world.
- **AuditedEpistemicCache** — reusable evidence needs a receipt for the observer/representation that produced it.
- **OperatorTime** — the reader's resident state participates in the effective operator.
- **AInstein** — provenance matters; ingredients at the wrong addresses are not equivalent.
- **AInsteinInsideTransformerResidualStream** — temporary latent computation lives in the fast stream; this repo isolates a slower reader state that decides how history is interrogated.

> **Do not only remember the past. Let experience change the apparatus that reads the past.**

## Qwen3-8B practical observer bridge

The scientific gates now have a practical language-model harness in
[`qwen_observer_chat.py`](qwen_observer_chat.py). It keeps the Qwen weights and
historical K/V untouched while a small persistent observer state changes the
query geometry used to read those sources. The selected layer/head identities
are calibrated once and frozen; the actual read tangent is reconstructed from
the **current post-RoPE key geometry** on every decode step.

This is intentionally not Gate 8. Gate 8 remains the measurement-cost / active
probe question exposed by Gate 7.

Quick comparison on the same textual prompt:

```bash
pip install -r requirements-qwen.txt
python qwen_observer_chat.py --compare
```

The default placement profile reuses the recent Qwen3-8B setup that survives
on the development machine: 6 GiB GPU + 6 GiB CPU with disk overflow. If
Windows kills the load, use `--gpu-memory 4GiB --cpu-memory 4GiB`.

See [QWEN_OBSERVER_CHAT.md](QWEN_OBSERVER_CHAT.md) for the mechanism, controls,
interactive commands, and interpretation boundary.


### Qwen causal head selection

The first Qwen3-8B chat receipt established strong query-level source control
without a language-level answer change: A-trust drove selected-head attention
to A, B-trust drove it to B, K-cache integrity held, and no perturbation hit
the cap, but all three generations still chose the A explanation.

That negative boundary is now explicit. The follow-up
[`qwen_observer_causal_compare.py`](qwen_observer_causal_compare.py) selects
heads by **downstream causal swing**, not by source-attention engagement alone,
before rerunning the same conversational A / neutral / B comparison. See
[QWEN_OBSERVER_CHAT.md](QWEN_OBSERVER_CHAT.md).


### First Qwen language-level switch

The causal selector produced the first language-level AOC switch on frozen
Qwen3-8B.  With the same prompt, same model weights and unchanged historical
source K/V rows:

```text
A trust  -> The device failed because valve C was obstructed.
neutral  -> The device failed because valve C was obstructed.
B trust  -> The device failed because sensor K drifted.
```

The causally selected plan spans layers 18, 24 and 30.  In B mode the selected
heads put mean attention mass `0.4194` on B versus `0.0160` on A, with mean
query-update ratio `0.3372`, max ratio `0.6244`, no capped updates, and
`cache_integrity_ok=true`.

This is an existence result, not yet a transfer result.  The head set was
selected on the valve/sensor conflict itself.

One instrumentation caveat is also frozen into the record: the causal profiler
used the first tokenizer token of each label, so `valve` was represented by
its first token (`val`) while `sensor` was one token.  The free-generation
switch does not depend on that approximation, but causal-swing magnitudes
should not be treated as exact sequence-level likelihood effects.

### Frozen-head transfer evaluation

`qwen_observer_transfer_eval.py` freezes the selected head identities from
`results/qwen_observer_causal_compare.json`; it does **no head reselection**.

It evaluates four unrelated conflicts, including two cases with physical source
order reversed, and sweeps observer trust over `-1,-0.5,0,0.5,1`.  The
dose-response readout uses complete candidate-sequence log likelihood rather
than the earlier first-token proxy.  A leave-one-head-out diagnostic is run on
the original conflict.

The pre-run contract is:

```text
>= 3/4 held-out cases must dual-switch under A vs B trust
2/2 reversed-order cases must dual-switch
>= 3/4 cases must satisfy margin(+1) > margin(0) > margin(-1)
all source-cache integrity checks must remain true
no query update may hit the norm cap
```

Run:

```bash
python3.13 qwen_observer_transfer_eval.py
```

Receipt:

```text
results/qwen_observer_transfer_eval.json
```


### Frozen-head transfer result

The frozen-head transfer contract did **not** fully pass, and the failure is
kept rather than softened:

```text
held-out cases                         4
full A<->B generation switches        2/4
reversed-order generation switches    2/2
ordered dose-response endpoints       4/4
source-cache integrity                true
capped query updates                  0
overall preregistered pass            false
```

The useful residue is narrower: the same frozen observer heads moved complete
candidate-sequence likelihood in the A-relative direction on all four unrelated
conflicts, even when the generated sentence did not cross the final decoding
boundary.  The controller therefore transfers more reliably as a **graded
directional bias** than as a guaranteed binary answer switch.

Leave-one-head-out scoring on the original conflict shows L24/Q29 and L24/Q23
carry most of the B-side crossing, L30/Q11 is supportive, and L18/Q28 is small.

### Order-only attacker

`attack_order.py` holds content, wording, source labels and head identities
fixed and changes only physical source order.  The unmodified model itself has
a very large first-record prior:

```text
A physically first: delta(A-B) = +10.8754
B physically first: delta(A-B) = -10.7571
```

Against that prior, the AOC dose-response slope keeps the **same semantic
direction** when order is reversed:

```text
slope AB = +7.9095
slope BA = +8.1761
ratio    = 1.034
```

So the frozen observer effect is not explained by a first-slot coordinate.
However, the historical receipt printed `IDENTITY_COORDINATE` while the BA
arm did not flip the greedy sentence at m=+1.  The script's old verdict code
forgot the preregistered "generation crosses in both arms" clause.  That bug is
now fixed: likelihood-only identity behavior is reported separately as
`IDENTITY_LIKELIHOOD_ONLY`.

### Live growing-cache observer

`qwen_observer_live_cache.py` closes the practical gap left by the earlier
chat harness.

The old chat harness rerendered and refilled the complete transcript every
turn.  The live harness instead performs exactly one full prefill, then keeps a
single Qwen `DynamicCache` alive and appends only:

```text
new user-turn suffix -> generated token -> generated token -> ...
```

The source K rows are fingerprinted once and checked against the same baseline
for the entire session.  Cache length must increase exactly with committed token
history or the run aborts.

The slow observer state is now also persistent outside the text.  External
evidence receipts update an additive evidence score whose bounded control value
is:

```math
m = tanh(score)
```

but self/model-prediction receipts are explicitly non-anchoring:

```text
sensor / tool / independent_model / user_verification -> may change trust
self_prediction / model_output                         -> logged only
```

This prevents a steered read from certifying itself.

Run:

```bash
python3.13 qwen_observer_live_cache.py
```

Useful commands:

```text
/state
/cache
/a
/b
/neutral
/trust 0.35
/auto
/evidence a 0.8 sensor checked externally
/evidence b 1.0 independent_model second system agreed
/self a 1.0 model preferred A
/quit
```

The observer evidence state survives process restarts in
`results/qwen_observer_live_state.json`.  The session receipt is written to
`results/qwen_observer_live_cache.json`.

This is the first harness in the repo where both objects are genuinely
persistent at different timescales:

```text
slow observer state persists across turns / restarts
fast Qwen KV state grows token by token inside one live conversation
```

The first successful second-turn run now answers the mechanical half. The same
live cache grew from 131 to 160 tokens while the original source spans stayed at
A=[45,62) and B=[72,87), cache/history lengths remained equal, and the
session-level source-cache integrity check stayed true. An external B-sensor
receipt moved anchored trust to m=-0.664 and selected-head attention strongly
toward B (mean target mass 0.2532 versus 0.0362 for A), but the generated
sentence still remained on A. That is a useful boundary rather than a pass:
persistent cache + persistent observer is now real, while language-level
control across distance is not yet established.

See [RESULTS_QWEN_LIVE_CACHE.md](RESULTS_QWEN_LIVE_CACHE.md).

The next discriminator is a matched distance-by-trust sweep on one live cache:
hold sources and frozen heads fixed, grow temporal distance, and measure
complete A-vs-B sequence likelihood at the same trust values. That separates
distance decay from the already-observed decoding threshold.

## Final Qwen closeout — semantic control survives +256 masked positions

The corrected first-ask and phasic-tail experiments close the Qwen distance thread more cleanly than the older fixed-candidate score suggested.

At distance 0, `m=-1` flips both the matched `valve`/`sensor` decision and greedy generation to B. After exactly +256 unreadable masked positions, the same observer still makes `sensor` beat `valve` at the decision boundary and both tonic and phasic greedy generation still produce the B explanation:

```text
The device failed because sensor K drifted, as supported by the calibration log.
```

The exact short B candidate `The device failed because sensor K drifted.` nevertheless loses the summed-likelihood comparison at +256. Token-level inspection shows that this is **not** because `K` or `drifted` became hard to recover: both remain essentially probability 1 once B is selected. The penalty is concentrated on the final period, because Qwen now prefers a comma and a longer B continuation.

The preregistered tail classifier therefore keeps its literal likelihood verdict `DISTANCE_DAMAGES_TAIL`, while the greedy-generation verdict is `GENERATION_CONTROL_SURVIVES`. Those are not contradictory once semantic choice and exact termination style are separated.

The phasic hypothesis also does not earn a positive result here. Turning the observer off after `sensor` improves the +256 B tail by only `+0.223` nats, below the frozen `0.5`-nat threshold, and tonic/phasic generations are identical. On this prompt, continuing the observer is not the main reason the exact short candidate loses.

What AOC has earned is narrower and useful:

```text
persistent external observer state
    -> causally changes how frozen history is read
    -> changes the semantic answer on the calibration conflict
    -> still controls that semantic answer after +256 masked cache positions

but

exact candidate-string likelihood can drift with continuation style
and held-out full-generation transfer remains only 2/4.
```

So the remaining boundary is **general transfer**, not a demonstrated 256-token semantic-control failure.

See [RESULTS_QWEN_PHASIC_TAIL.md](RESULTS_QWEN_PHASIC_TAIL.md) and `results/qwen_observer_phasic_tail_gen.json` for the complete receipt. The attempted `0,512,1024,2048` receipt is incomplete (`complete=false`) and is not used as evidence in this closeout.
