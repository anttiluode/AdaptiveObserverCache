# Gate 1 contract — frozen transformer attention, moving query

## Question

Can a persistent observer state alter retrieval from **identical projected K/V**
inside a frozen attention mechanism, and can feedback alter the next query
without changing the cache or model parameters?

Gate 1 is a bridge from Gate 0 to a real pretrained transformer. It uses
PyTorch `nn.Linear` projections and scaled dot-product attention, but the
weights are hand-frozen and tiny so the mechanism remains auditable.

## Frozen before running

Architecture:

```text
q_base = W_Q h
q'     = q_base + 2.5 * m * u

scores = K q' / sqrt(d_head)
read   = softmax(scores) V
```

where:

- `h` is identical at every step;
- `K,V` are projected once and frozen;
- `u = [1,-1]` is fixed;
- `m` is one scalar persistent observer state;
- all `W_Q,W_K,W_V` parameters are frozen;
- the reader gets exactly one attention read per step.

The trusted provenance schedule is:

```text
AAAA BBBB AAAA BBBB
```

The adaptive reader starts at `m=+1`.

Feedback is supplied **after** each read. A contradiction may change `m`
for the next step; truth is never passed into the current query.

## Attackers

The adaptive reader must beat the **best fixed observer** among:

```text
m = -1
m =  0
m = +1
```

All receive the same present state, projected cache, attention mechanism, and
one-read budget.

## Pass boundary

Gate 1 passes only if all are true:

1. identical cache + identical present + `m=+1` reads A and predicts +1;
2. identical cache + identical present + `m=-1` reads B and predicts -1;
3. adaptive accuracy >= 0.75;
4. adaptive accuracy exceeds the best fixed observer by >= 0.20;
5. cache digest is byte-identical before/after;
6. parameter digest is byte-identical before/after;
7. every parameter remains `requires_grad=False`;
8. each provenance switch costs at most its first contradicted read.

## Kill boundary

Do not call this observer-in-the-loop if the effect requires:

- editing K/V;
- changing model weights;
- seeing the current truth before the read;
- extra cache reads for the adaptive condition;
- comparing only to a deliberately bad fixed observer.

## Interpretation boundary

A pass establishes a frozen-attention mechanism, **not** usefulness in a
pretrained LLM. Gate 2 must attach the same low-rank observer idea to a real
frozen transformer layer and use naturally produced hidden states/KV.
