# AdaptiveObserverCache

**Can the same fixed memory reveal different useful information because the reader itself has persistent state — and can new evidence change that reader without rewriting the memory?**

This repo isolates one mechanism from the recent `AInsteinInsideTransformerResidualStream` / Genealogy "observer in the loop" line:

```text
fixed cache C
fixed present h
persistent observer state m
        |
        v
query geometry Q(h, m)
        |
        v
read fixed cache C
        |
        v
observation + receipt
        |
        v
update m
        |
        v
next query is different
```

The cache is historical material. The observer state is the changing measuring apparatus.

## Gate 0 — same cache, moving reader

Gate 0 is deliberately tiny and model-free. It uses a two-dimensional KV-like memory with conflicting evidence under two provenance families, A and B.

The contract freezes the cache, present input, one-read budget, and mechanism parameters. Only observer state may change.

With identical cache and identical present:

```text
m_A -> query points toward A-addressed evidence -> +1
m_B -> query points toward B-addressed evidence -> -1
```

On the alternating trusted-provenance schedule

```text
AAAA BBBB AAAA BBBB
```

the static reader gets 50%, while the adaptive reader gets 81.25% and changes only its observer state after contradicted reads.

Run:

```bash
python gate0_experiment.py
```

Gate 0 is intentionally only a synthetic primitive.

## Gate 1 — frozen transformer attention, moving query

Gate 1 moves the primitive into a real PyTorch attention calculation with frozen `nn.Linear` projections:

```text
q_base = W_Q h
q'     = q_base + 2.5 * m * u

scores = K q' / sqrt(d_head)
read   = softmax(scores) V
```

Everything except the scalar observer `m` is frozen:

- identical present hidden state on every read;
- `W_Q/W_K/W_V` are frozen and never trained;
- memory is projected into `K,V` exactly once;
- projected `K,V` are never rewritten;
- one attention read per step;
- feedback arrives only **after** the current read and may affect only the next query.

The attacker is not one unlucky static baseline. Gate 1 tests fixed observers `m=-1,0,+1` and compares against the **best** of them.

### Gate 1 receipt

CI on Python 3.12 / CPU PyTorch produced:

```text
same K/V + same present + m=+1:
    A attention mass = 0.962236
    B attention mass = 0.037764
    prediction = +1

same K/V + same present + m=-1:
    A attention mass = 0.037764
    B attention mass = 0.962236
    prediction = -1

best fixed observer accuracy = 0.5000
adaptive observer accuracy   = 0.8125
adaptive advantage           = +0.3125
```

Every provenance switch costs exactly its first contradicted read; the observer then changes the next query.

The projected-cache digest is identical before and after:

```text
c98fa957ef01a9ba7ae49bb943e145f389d10864553efa3ed9796ae5ee5fd19c
```

The frozen-parameter digest is also identical before and after:

```text
d882c794aecd821a164dfa01057bf19bebc593abf1698b3ef19d0ff43f144958
```

All parameters remain `requires_grad=False`.

See [GATE1_CONTRACT.md](GATE1_CONTRACT.md) and [RESULTS_GATE1.md](RESULTS_GATE1.md).

Run:

```bash
pip install -r requirements-gate1.txt --index-url https://download.pytorch.org/whl/cpu
python gate1_experiment.py
```

## What Gate 1 does and does not establish

Gate 1 now establishes this executable object:

```text
fast present h_t
fixed historical field (K,V)
slow persistent observer m_t

q_t = W_Q h_t + u m_t
read_t = Attention(q_t, K, V)
m_(t+1) = U(m_t, read_t, feedback_t)
```

It does **not** establish useful adaptation inside a pretrained LLM. The A/B memory and observer direction are deliberately constructed so the mechanism is transparent.

The next hard gate is therefore not another synthetic schedule. It is to attach a tiny low-rank observer state to a **real frozen pretrained transformer layer**, use naturally produced hidden states/KV, and ask whether earlier calibration experience can improve later ambiguous retrieval under an equal-compute static-reader attacker.

The discipline stays the same:

```text
make the reader dynamic before making the memory dynamic
```

## Relationship to the recent repos

- **ReadWrite** — a state may be invisible until the right intervention/query is applied.
- **WhatToLookAt** — memory changes which later measurement is worth buying.
- **PredictiveHKT** — a changing representation can masquerade as a changing world.
- **AuditedEpistemicCache** — reusable evidence needs a receipt for the observer/representation that produced it.
- **OperatorTime** — the reader's resident state participates in the effective operator.
- **AInstein** — provenance matters; the right ingredients at the wrong addresses are not equivalent.
- **AInsteinInsideTransformerResidualStream** — temporary latent computation and compute foveation live in the fast stream; this repo isolates the slower reader state that could decide how history is interrogated.

The working slogan is:

> **Do not only remember the past. Let experience change the apparatus that reads the past.**
