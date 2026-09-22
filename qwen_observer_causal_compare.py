"""Causal head selection for the Qwen AdaptiveObserverCache bridge.

The first Qwen run showed a clean dissociation:

    source attention moved strongly
    generated answer did not

So this calibration does not ask "which heads are easiest to steer?" alone.
It first builds a small geometry-qualified pool, then serially tests each head
on a one-word A-vs-B answer probe under identical execution shape.

Only heads for which A-trust moves the answer logit gap toward A relative to
B-trust are eligible for the final conversational observer.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from qwen_observer import (
    CausalHeadScore,
    ObserverPlan,
    QwenObserverController,
    capture_qwen_geometry,
    observer_candidates,
    select_causal_plan,
)
from qwen_observer_chat import (
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_QUESTION,
    answer_once,
    encode_rendered,
    input_device,
    load_model,
    prepare_turn,
    render_chat,
    system_message,
)


DEFAULT_PROBE_QUESTION = (
    'Which component caused the failure? '
    'Answer with exactly one word: valve or sensor.'
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            'Find Qwen heads whose observer-controlled source read is '
            'causally load-bearing for an A-vs-B answer, then run the '
            'same conversational comparison.'
        )
    )
    p.add_argument('--model', default='Qwen/Qwen3-8B')
    p.add_argument('--source-a', default=DEFAULT_A)
    p.add_argument('--source-b', default=DEFAULT_B)
    p.add_argument('--question', default=DEFAULT_QUESTION)
    p.add_argument('--probe-question', default=DEFAULT_PROBE_QUESTION)
    p.add_argument('--probe-a', default='valve')
    p.add_argument('--probe-b', default='sensor')
    p.add_argument('--layers', default='18,24,30')
    p.add_argument('--candidates-per-layer', type=int, default=4)
    p.add_argument('--observer-heads', type=int, default=4)
    p.add_argument('--target-margin', type=float, default=3.0)
    p.add_argument('--max-ratio', type=float, default=1.0)
    p.add_argument('--min-causal-swing', type=float, default=0.0)
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--gpu-memory', default='6GiB')
    p.add_argument('--cpu-memory', default='6GiB')
    p.add_argument('--offload-dir', default='.offload_qwen_observer')
    p.add_argument(
        '--receipt',
        default='results/qwen_observer_causal_compare.json',
    )
    return p.parse_args()


def _first_probe_token(tokenizer, text: str):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise RuntimeError(f'probe text produced no token: {text!r}')
    return ids[0], ids


def _gap(logits, token_a: int, token_b: int) -> float:
    logp = torch.log_softmax(logits.float(), dim=-1)
    return float(logp[token_a] - logp[token_b])


@torch.inference_mode()
def score_head(
    model,
    candidate,
    probe_ids,
    probe_mask,
    probe_spans,
    token_a,
    token_b,
    args,
):
    plan = ObserverPlan(heads=(candidate,))
    controller = QwenObserverController(
        model,
        plan,
        target_margin=args.target_margin,
        max_ratio=args.max_ratio,
    )
    controller.install()
    try:
        rows = {}
        for label, trust in (('A', +1.0), ('B', -1.0)):
            controller.set_trust(trust)
            controller.set_spans(probe_spans)
            controller.begin_generation()
            out = model(
                input_ids=probe_ids,
                attention_mask=probe_mask,
                use_cache=False,
                return_dict=True,
            )
            gap = _gap(out.logits[0, -1], token_a, token_b)
            rows[label] = {
                'trust': trust,
                'gap_A_minus_B': gap,
                'observer': controller.summary(),
            }
        swing = rows['A']['gap_A_minus_B'] - rows['B']['gap_A_minus_B']
        mean_ratio = 0.5 * (
            rows['A']['observer'].get('mean_query_update_ratio', 0.0)
            + rows['B']['observer'].get('mean_query_update_ratio', 0.0)
        )
        capped = (
            rows['A']['observer'].get('capped_fraction', 0.0) > 0.0
            or rows['B']['observer'].get('capped_fraction', 0.0) > 0.0
        )
        score = CausalHeadScore(
            head=candidate,
            gap_when_a=rows['A']['gap_A_minus_B'],
            gap_when_b=rows['B']['gap_A_minus_B'],
            causal_swing=swing,
            mean_update_ratio=mean_ratio,
            capped=capped,
        )
        return score, rows
    finally:
        controller.uninstall()


def _head_dict(head):
    return {
        'layer': head.layer,
        'query_head': head.query_head,
        'kv_head': head.kv_head,
        'symmetric_source_mass': head.symmetric_source_mass,
        'mass_when_a': head.mass_when_a,
        'mass_when_b': head.mass_when_b,
        'ratio_when_a': head.ratio_when_a,
        'ratio_when_b': head.ratio_when_b,
    }


def main():
    args = parse_args()
    out_path = Path(args.receipt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args)
    system = system_message(args.source_a, args.source_b)

    target_messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': args.question},
    ]

    target_rendered, target_ids, target_offsets = None, None, None
    target_rendered = render_chat(tokenizer, target_messages)
    target_ids, target_mask, target_offsets = encode_rendered(
        tokenizer, target_rendered
    )
    from qwen_observer import locate_source_spans

    target_spans = locate_source_spans(
        target_rendered,
        target_offsets,
        args.source_a,
        args.source_b,
    )

    layers = [
        int(x.strip()) for x in args.layers.split(',') if x.strip()
    ]
    max_layer = len(model.model.layers) - 1
    for layer in layers:
        if layer < 0 or layer > max_layer:
            raise ValueError(f'layer {layer} outside 0..{max_layer}')

    device = input_device(model)
    target_ids = target_ids.to(device)
    target_mask = target_mask.to(device)

    print(
        'Stage 1/3: geometry pool at layers '
        + ','.join(str(x) for x in layers)
    )
    captures = capture_qwen_geometry(
        model,
        target_ids,
        target_mask,
        layers=layers,
    )
    all_candidates = observer_candidates(
        captures,
        target_spans,
        target_margin=args.target_margin,
        max_ratio=args.max_ratio,
    )

    pool = []
    for layer in layers:
        layer_rows = [x for x in all_candidates if x.layer == layer]
        pool.extend(layer_rows[: max(1, args.candidates_per_layer)])

    # Geometry capture tensors can be large. The head descriptors are tiny.
    del captures
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f'Geometry-qualified causal pool: {len(pool)} heads')
    for item in pool:
        print(
            f'  L{item.layer:02d} Q{item.query_head:02d}/KV{item.kv_head}: '
            f'sym_mass={item.symmetric_source_mass:.3f} '
            f'ratioA={item.ratio_when_a:.3f} '
            f'ratioB={item.ratio_when_b:.3f}'
        )

    probe_messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': args.probe_question},
    ]
    _probe_rendered, probe_ids, probe_mask, probe_spans = prepare_turn(
        tokenizer, probe_messages, args
    )
    probe_ids = probe_ids.to(device)
    probe_mask = probe_mask.to(device)

    token_a, token_seq_a = _first_probe_token(tokenizer, args.probe_a)
    token_b, token_seq_b = _first_probe_token(tokenizer, args.probe_b)
    if token_a == token_b:
        raise RuntimeError(
            'probe A/B share their first token; choose probe labels with '
            'different first tokenizer IDs'
        )

    print(
        '\nStage 2/3: serial causal scan. '
        'Same prompt shape; only observer sign changes.'
    )
    print(
        f'  probe A={args.probe_a!r} ids={token_seq_a} '
        f'first={token_a} ({tokenizer.decode([token_a])!r})'
    )
    print(
        f'  probe B={args.probe_b!r} ids={token_seq_b} '
        f'first={token_b} ({tokenizer.decode([token_b])!r})'
    )

    score_rows = []
    causal_scores = []
    for index, candidate in enumerate(pool, start=1):
        print(
            f'  [{index:02d}/{len(pool):02d}] '
            f'L{candidate.layer:02d} Q{candidate.query_head:02d}/'
            f'KV{candidate.kv_head}',
            flush=True,
        )
        score, rows = score_head(
            model,
            candidate,
            probe_ids,
            probe_mask,
            probe_spans,
            token_a,
            token_b,
            args,
        )
        causal_scores.append(score)
        score_rows.append(
            {
                'head': _head_dict(candidate),
                'gap_when_A_trust': score.gap_when_a,
                'gap_when_B_trust': score.gap_when_b,
                'causal_swing': score.causal_swing,
                'mean_update_ratio': score.mean_update_ratio,
                'capped': score.capped,
                'runs': rows,
            }
        )
        print(
            f'      gap A={score.gap_when_a:+.4f} '
            f'gap B={score.gap_when_b:+.4f} '
            f'swing={score.causal_swing:+.4f} '
            f'ratio={score.mean_update_ratio:.3f}'
            + (' CAPPED' if score.capped else '')
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_plan = select_causal_plan(
        causal_scores,
        num_heads=args.observer_heads,
        min_swing=args.min_causal_swing,
    )

    receipt = {
        'scope': (
            'Qwen causal head selection after the first attention-only '
            'observer run moved source attention but not generated language.'
        ),
        'model': args.model,
        'target_question': args.question,
        'probe_question': args.probe_question,
        'probe_a': {
            'text': args.probe_a,
            'token_ids': token_seq_a,
            'first_token_id': token_a,
        },
        'probe_b': {
            'text': args.probe_b,
            'token_ids': token_seq_b,
            'first_token_id': token_b,
        },
        'layers': layers,
        'candidates_per_layer': args.candidates_per_layer,
        'target_margin': args.target_margin,
        'max_ratio': args.max_ratio,
        'min_causal_swing': args.min_causal_swing,
        'causal_scan': score_rows,
        'selected_heads': [_head_dict(x) for x in final_plan.heads],
        'runs': [],
    }

    if not final_plan.heads:
        receipt['status'] = 'no_positive_causal_heads'
        out_path.write_text(json.dumps(receipt, indent=2))
        print(
            '\nNo geometry-qualified head had positive causal swing. '
            'Stopping without brute-force amplification.'
        )
        print(f'receipt: {out_path}')
        return

    selected_lookup = {
        (s.head.layer, s.head.query_head): s
        for s in causal_scores
    }
    print('\nSelected causally aligned observer heads:')
    for head in final_plan.heads:
        s = selected_lookup[(head.layer, head.query_head)]
        print(
            f'  L{head.layer:02d} Q{head.query_head:02d}/KV{head.kv_head}: '
            f'swing={s.causal_swing:+.4f} '
            f'sym_mass={head.symmetric_source_mass:.3f} '
            f'ratio={s.mean_update_ratio:.3f}'
        )

    print('\nStage 3/3: same conversational prompt, three observer states')
    controller = QwenObserverController(
        model,
        final_plan,
        target_margin=args.target_margin,
        max_ratio=args.max_ratio,
    )
    controller.install()
    try:
        for label, trust in (
            ('A', +1.0),
            ('neutral', 0.0),
            ('B', -1.0),
        ):
            text, summary = answer_once(
                model,
                tokenizer,
                controller,
                target_messages,
                args,
                trust=trust,
            )
            print(f'\n[{label} trust {trust:+.1f}]\n{text}')
            print('observer:', json.dumps(summary, indent=2))
            receipt['runs'].append(
                {
                    'mode': label,
                    'trust': trust,
                    'answer': text,
                    'observer': summary,
                }
            )
    finally:
        controller.uninstall()

    receipt['status'] = 'completed'
    out_path.write_text(json.dumps(receipt, indent=2))
    print(f'\nreceipt: {out_path}')


if __name__ == '__main__':
    main()
