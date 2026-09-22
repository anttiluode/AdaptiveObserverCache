"""Live Qwen3-8B AOC with one genuinely growing KV cache.

This is the first conversational harness in the repo that does NOT rerender and
refill the whole transcript on every turn.  The initial source records are
prefilled once, one DynamicCache grows monotonically, and all later user turns
and generated tokens are appended to that same cache.

The persistent AOC state is separate from both prompt text and model weights.
Evidence receipts may update that state, but self/model-prediction receipts are
logged without changing epistemic trust.  This prevents a steered read from
certifying itself.

Run:
    python3.13 qwen_observer_live_cache.py

Useful commands:
    /state
    /cache
    /a | /b | /neutral | /trust 0.35
    /auto
    /evidence a 0.8 sensor optional note
    /evidence b 1.0 independent_model optional note
    /self a 1.0 model thought (logged, trust unchanged)
    /quit
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import torch

from qwen_observer import (
    HeadSelection,
    ObserverPlan,
    QwenObserverController,
    locate_source_spans,
)
from qwen_observer_chat import (
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_QUESTION,
    encode_rendered,
    input_device,
    load_model,
    render_chat,
    system_message,
)


ANCHOR_KINDS = {
    "sensor",
    "independent_model",
    "tool",
    "user_verification",
}
SELF_KINDS = {
    "self_prediction",
    "model_output",
    "self",
}


@dataclass
class EvidenceReceipt:
    source: str
    weight: float
    kind: str
    note: str
    anchored: bool
    score_delta: float
    score_after: float
    trust_after: float


class EvidenceLedger:
    """Slow observer state with explicit evidence provenance.

    score is an additive signed evidence accumulator.  Effective anchored trust
    is tanh(score), so evidence composes smoothly but remains bounded.  A manual
    override is allowed for experiments, but it is kept separate from evidence.
    """

    def __init__(
        self,
        score: float = 0.0,
        receipts: List[EvidenceReceipt] | None = None,
        manual_override: float | None = None,
    ):
        self.score = float(score)
        self.receipts = list(receipts or [])
        self.manual_override = (
            None
            if manual_override is None
            else max(-1.0, min(1.0, float(manual_override)))
        )

    @property
    def anchored_trust(self) -> float:
        return math.tanh(self.score)

    @property
    def effective_trust(self) -> float:
        if self.manual_override is not None:
            return self.manual_override
        return self.anchored_trust

    def set_manual(self, trust: float | None) -> None:
        self.manual_override = (
            None
            if trust is None
            else max(-1.0, min(1.0, float(trust)))
        )

    def apply(
        self,
        source: str,
        weight: float,
        kind: str,
        note: str = "",
    ) -> EvidenceReceipt:
        source = source.strip().upper()
        kind = kind.strip().lower()
        if source not in {"A", "B"}:
            raise ValueError("source must be A or B")
        weight = float(weight)
        if not (0.0 <= weight <= 1.0):
            raise ValueError("evidence weight must lie in [0, 1]")
        if kind not in ANCHOR_KINDS | SELF_KINDS:
            allowed = ", ".join(sorted(ANCHOR_KINDS | SELF_KINDS))
            raise ValueError(f"unknown evidence kind {kind!r}; use one of: {allowed}")

        anchored = kind in ANCHOR_KINDS
        sign = +1.0 if source == "A" else -1.0
        delta = sign * weight if anchored else 0.0
        self.score = max(-6.0, min(6.0, self.score + delta))
        receipt = EvidenceReceipt(
            source=source,
            weight=weight,
            kind=kind,
            note=note,
            anchored=anchored,
            score_delta=delta,
            score_after=self.score,
            trust_after=self.anchored_trust,
        )
        self.receipts.append(receipt)
        return receipt

    def to_json(self) -> dict:
        return {
            "score": self.score,
            "anchored_trust": self.anchored_trust,
            "manual_override": self.manual_override,
            "effective_trust": self.effective_trust,
            "receipts": [asdict(x) for x in self.receipts],
        }

    @classmethod
    def from_json(cls, payload: dict) -> "EvidenceLedger":
        rows = [
            EvidenceReceipt(**row)
            for row in payload.get("receipts", [])
        ]
        return cls(
            score=float(payload.get("score", 0.0)),
            receipts=rows,
            manual_override=payload.get("manual_override"),
        )


def parse_args():
    p = argparse.ArgumentParser(
        description="Talk to frozen Qwen through AOC while one real KV cache grows."
    )
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--source-a", default=DEFAULT_A)
    p.add_argument("--source-b", default=DEFAULT_B)
    p.add_argument("--question", default=DEFAULT_QUESTION)
    p.add_argument(
        "--plan-receipt",
        default="results/qwen_observer_causal_compare.json",
    )
    p.add_argument("--target-margin", type=float, default=3.0)
    p.add_argument("--max-ratio", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--gpu-memory", default="6GiB")
    p.add_argument("--cpu-memory", default="6GiB")
    p.add_argument("--offload-dir", default=".offload_qwen_observer")
    p.add_argument(
        "--state-file",
        default="results/qwen_observer_live_state.json",
    )
    p.add_argument(
        "--receipt",
        default="results/qwen_observer_live_cache.json",
    )
    p.add_argument(
        "--fresh-state",
        action="store_true",
        help="ignore any persisted observer evidence state",
    )
    p.add_argument(
        "--trust",
        type=float,
        default=None,
        help="optional manual observer override for this session",
    )
    return p.parse_args()


def load_plan(path: Path) -> ObserverPlan:
    payload = json.loads(path.read_text())
    rows = payload.get("selected_heads") or []
    if not rows:
        raise RuntimeError(
            f"{path} has no selected_heads; run qwen_observer_causal_compare.py first"
        )
    return ObserverPlan(
        heads=tuple(
            HeadSelection(
                layer=int(row["layer"]),
                query_head=int(row["query_head"]),
                kv_head=int(row["kv_head"]),
                symmetric_source_mass=float(row["symmetric_source_mass"]),
                mass_when_a=float(row["mass_when_a"]),
                mass_when_b=float(row["mass_when_b"]),
                ratio_when_a=float(row["ratio_when_a"]),
                ratio_when_b=float(row["ratio_when_b"]),
            )
            for row in rows
        )
    )


def load_ledger(path: Path, fresh: bool) -> EvidenceLedger:
    if fresh or not path.exists():
        return EvidenceLedger()
    return EvidenceLedger.from_json(json.loads(path.read_text()))


def save_ledger(path: Path, ledger: EvidenceLedger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger.to_json(), indent=2))


def cache_length(cache) -> int:
    if cache is None:
        return 0
    getter = getattr(cache, "get_seq_length", None)
    if getter is not None:
        return int(getter())
    try:
        return int(cache[0][0].shape[-2])
    except Exception as exc:
        raise RuntimeError("cannot determine KV-cache length") from exc


def new_cache():
    try:
        from transformers import DynamicCache
        return DynamicCache()
    except Exception:
        return None


def assistant_end_id(tokenizer) -> int:
    tok = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if tok is None or tok == tokenizer.unk_token_id:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Qwen assistant-end token is unavailable")
        return int(tokenizer.eos_token_id)
    return int(tok)


def user_turn_suffix_ids(tokenizer, user_text: str) -> List[int]:
    """Render only the bytes/tokens after the previous assistant <|im_end|>.

    A synthetic assistant anchor makes this independent of how the previous
    generated answer happened to tokenize.  We cut the template immediately
    after that anchor's im_end token and keep the exact template-generated
    suffix for the new user turn plus assistant generation prompt.
    """

    anchor = "__AOC_PREVIOUS_ASSISTANT__"
    anchor_messages = [{"role": "assistant", "content": anchor}]
    full_messages = [
        {"role": "assistant", "content": anchor},
        {"role": "user", "content": user_text},
    ]

    try:
        anchor_ids = tokenizer.apply_chat_template(
            anchor_messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        full_ids = tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        anchor_ids = tokenizer.apply_chat_template(
            anchor_messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        full_ids = tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=True,
        )

    end_id = assistant_end_id(tokenizer)
    positions = [i for i, tok in enumerate(anchor_ids) if int(tok) == end_id]
    if not positions:
        raise RuntimeError("chat template anchor contained no assistant end token")
    cut = positions[-1] + 1
    prefix = [int(x) for x in anchor_ids[:cut]]
    if [int(x) for x in full_ids[:cut]] != prefix:
        raise RuntimeError("Qwen chat template is not prefix-stable at assistant boundary")
    return [int(x) for x in full_ids[cut:]]


def initial_prompt(tokenizer, args):
    messages = [
        {
            "role": "system",
            "content": system_message(args.source_a, args.source_b),
        },
        {"role": "user", "content": args.question},
    ]
    rendered = render_chat(tokenizer, messages)
    ids, mask, offsets = encode_rendered(tokenizer, rendered)
    spans = locate_source_spans(
        rendered,
        offsets,
        args.source_a,
        args.source_b,
    )
    return messages, rendered, ids, mask, spans


@torch.inference_mode()
def append_tokens(
    model,
    cache,
    history_ids: List[int],
    token_ids: List[int],
):
    if not token_ids:
        raise ValueError("cannot append an empty token segment")
    device = input_device(model)
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    total = len(history_ids) + len(token_ids)
    mask = torch.ones((1, total), dtype=torch.long, device=device)
    out = model(
        input_ids=ids,
        attention_mask=mask,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    cache = out.past_key_values
    history_ids.extend(int(x) for x in token_ids)
    observed = cache_length(cache)
    if observed != len(history_ids):
        raise RuntimeError(
            f"KV cache/history mismatch: cache={observed}, history={len(history_ids)}"
        )
    return cache, out.logits[:, -1, :]


@torch.inference_mode()
def greedy_continue(
    model,
    tokenizer,
    cache,
    history_ids: List[int],
    first_logits,
    *,
    max_new_tokens: int,
):
    end_id = assistant_end_id(tokenizer)
    eos_id = tokenizer.eos_token_id
    logits = first_logits
    emitted: List[int] = []
    closed = False

    for _ in range(max_new_tokens):
        next_id = int(torch.argmax(logits.float(), dim=-1)[0])
        if next_id == end_id or (eos_id is not None and next_id == int(eos_id)):
            # Feed the closing token so the persistent cache and token history
            # really contain the same closed assistant turn.
            cache, logits = append_tokens(
                model, cache, history_ids, [next_id]
            )
            if next_id != end_id:
                cache, logits = append_tokens(
                    model, cache, history_ids, [end_id]
                )
            closed = True
            break

        emitted.append(next_id)
        cache, logits = append_tokens(
            model, cache, history_ids, [next_id]
        )

    if not closed:
        # Max-token truncation must still leave a syntactically closed chat
        # turn before a new user turn is appended.
        cache, logits = append_tokens(
            model, cache, history_ids, [end_id]
        )

    text = tokenizer.decode(emitted, skip_special_tokens=True).strip()
    return cache, logits, text, emitted


def observer_state_line(ledger: EvidenceLedger) -> str:
    mode = (
        "manual"
        if ledger.manual_override is not None
        else "evidence"
    )
    return (
        f"observer m={ledger.effective_trust:+.3f} ({mode}); "
        f"anchored={ledger.anchored_trust:+.3f}; "
        f"evidence_score={ledger.score:+.3f}; "
        f"receipts={len(ledger.receipts)}"
    )


def apply_command(
    text: str,
    ledger: EvidenceLedger,
    state_path: Path,
):
    if text == "/a":
        ledger.set_manual(+1.0)
    elif text == "/b":
        ledger.set_manual(-1.0)
    elif text == "/neutral":
        ledger.set_manual(0.0)
    elif text == "/auto":
        ledger.set_manual(None)
    elif text.startswith("/trust "):
        ledger.set_manual(float(text.split(maxsplit=1)[1]))
    elif text.startswith("/evidence ") or text.startswith("/self "):
        parts = text.split(maxsplit=4)
        if len(parts) < 3:
            raise ValueError(
                "usage: /evidence <a|b> <weight> [kind] [note]"
            )
        command = parts[0]
        source = parts[1]
        weight = float(parts[2])
        if command == "/self":
            kind = "self_prediction"
            note = parts[3] if len(parts) >= 4 else ""
        else:
            kind = parts[3] if len(parts) >= 4 else "sensor"
            note = parts[4] if len(parts) >= 5 else ""
        receipt = ledger.apply(source, weight, kind, note)
        print(
            "evidence receipt:",
            json.dumps(asdict(receipt), indent=2),
        )
    else:
        return False

    save_ledger(state_path, ledger)
    print(observer_state_line(ledger))
    return True


def main():
    args = parse_args()
    plan = load_plan(Path(args.plan_receipt))
    state_path = Path(args.state_file)
    ledger = load_ledger(state_path, args.fresh_state)
    if args.trust is not None:
        ledger.set_manual(args.trust)

    model, tokenizer = load_model(args)
    messages, rendered, ids, _mask, spans = initial_prompt(tokenizer, args)
    initial_ids = [int(x) for x in ids[0].tolist()]

    controller = QwenObserverController(
        model,
        plan,
        trust=ledger.effective_trust,
        target_margin=args.target_margin,
        max_ratio=args.max_ratio,
    )
    controller.set_spans(spans)
    controller.install()

    cache = new_cache()
    history_ids: List[int] = []
    session = {
        "model": args.model,
        "scope": "one persistent growing KV cache plus persistent external observer state",
        "source_spans": {
            "A": list(spans.source_a),
            "B": list(spans.source_b),
        },
        "frozen_heads": [
            {"layer": h.layer, "query_head": h.query_head, "kv_head": h.kv_head}
            for h in plan.heads
        ],
        "turns": [],
    }

    try:
        # First and only full prefill.
        controller.set_trust(ledger.effective_trust)
        controller.begin_generation(preserve_source_baseline=True)
        cache, logits = append_tokens(
            model, cache, history_ids, initial_ids
        )
        prefill_len = cache_length(cache)
        cache, logits, text, _emitted = greedy_continue(
            model,
            tokenizer,
            cache,
            history_ids,
            logits,
            max_new_tokens=args.max_new_tokens,
        )
        summary = controller.summary()
        print(observer_state_line(ledger))
        print(f"KV cache after first answer: {cache_length(cache)} tokens")
        print("\nQwen:", text)
        print("observer:", json.dumps(summary, indent=2))
        messages.append({"role": "assistant", "content": text})
        session["turns"].append(
            {
                "kind": "initial",
                "question": args.question,
                "answer": text,
                "trust": ledger.effective_trust,
                "cache_len_before_answer": prefill_len,
                "cache_len_after_answer": cache_length(cache),
                "observer": summary,
            }
        )

        print(
            "\nCommands: /state, /cache, /a, /b, /neutral, /trust x, /auto, "
            "/evidence <a|b> <0..1> [sensor|independent_model|tool|user_verification] [note], "
            "/self <a|b> <0..1> [note], /quit"
        )

        while True:
            try:
                user = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user:
                continue
            if user == "/quit":
                break
            if user == "/state":
                print(observer_state_line(ledger))
                print(json.dumps(ledger.to_json(), indent=2))
                continue
            if user == "/cache":
                print(
                    json.dumps(
                        {
                            "cache_tokens": cache_length(cache),
                            "history_tokens": len(history_ids),
                            "source_A": list(spans.source_a),
                            "source_B": list(spans.source_b),
                            "source_cache_integrity_ok": controller.cache_integrity_ok,
                        },
                        indent=2,
                    )
                )
                continue
            try:
                if apply_command(user, ledger, state_path):
                    controller.set_trust(ledger.effective_trust)
                    continue
            except Exception as exc:
                print("command error:", exc)
                continue

            # A real new turn: only its exact chat-template suffix enters the
            # existing cache.  The source records and old conversation are not
            # rerun.
            suffix = user_turn_suffix_ids(tokenizer, user)
            before = cache_length(cache)
            controller.set_trust(ledger.effective_trust)
            controller.begin_generation(preserve_source_baseline=True)
            cache, logits = append_tokens(
                model, cache, history_ids, suffix
            )
            after_user = cache_length(cache)
            cache, logits, text, _emitted = greedy_continue(
                model,
                tokenizer,
                cache,
                history_ids,
                logits,
                max_new_tokens=args.max_new_tokens,
            )
            summary = controller.summary()
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": text})
            print("\nQwen:", text)
            print(
                "cache:",
                f"{before} -> {after_user} -> {cache_length(cache)} tokens",
            )
            print("observer:", json.dumps(summary, indent=2))
            session["turns"].append(
                {
                    "kind": "chat",
                    "question": user,
                    "answer": text,
                    "trust": ledger.effective_trust,
                    "cache_len_before_user": before,
                    "cache_len_after_user": after_user,
                    "cache_len_after_answer": cache_length(cache),
                    "appended_user_tokens": len(suffix),
                    "observer": summary,
                }
            )

            if not controller.cache_integrity_ok:
                raise RuntimeError(
                    "historical source K rows changed during the growing-cache session"
                )

    finally:
        controller.uninstall()
        save_ledger(state_path, ledger)
        session["final_cache_tokens"] = len(history_ids)
        session["source_cache_integrity_ok"] = controller.cache_integrity_ok
        session["observer_state"] = ledger.to_json()
        out = Path(args.receipt)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(session, indent=2))
        print(f"\nstate: {state_path}")
        print(f"receipt: {out}")


if __name__ == "__main__":
    main()
