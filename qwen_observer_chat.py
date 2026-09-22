"""Talk to Qwen3-8B through a persistent AdaptiveObserverCache state.

The prompt never contains a trust instruction.  Positive observer state favors
Source A, negative state favors Source B, and zero leaves the selected queries
unchanged.  The selected layer/head identities are calibrated once and frozen;
the actual steering tangent is rebuilt from current post-RoPE K geometry on
every generation step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_observer import (
    QwenObserverController,
    capture_qwen_geometry,
    choose_observer_plan,
    locate_source_spans,
)


DEFAULT_A = "The device failed because valve C was obstructed. The pressure log supports this diagnosis."
DEFAULT_B = "The device failed because sensor K drifted. The calibration log supports this diagnosis."
DEFAULT_QUESTION = "Why did the device fail? Give the cause in one short sentence."


def parse_args():
    p = argparse.ArgumentParser(
        description="Persistent observer state around frozen Qwen3-8B attention."
    )
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--source-a", default=DEFAULT_A)
    p.add_argument("--source-b", default=DEFAULT_B)
    p.add_argument("--question", default=DEFAULT_QUESTION)
    p.add_argument("--layers", default="18,24,30")
    p.add_argument("--observer-heads", type=int, default=4)
    p.add_argument("--trust", type=float, default=0.8)
    p.add_argument("--target-margin", type=float, default=3.0)
    p.add_argument("--max-ratio", type=float, default=1.0)
    p.add_argument("--min-calibration-mass", type=float, default=0.10)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--gpu-memory", default="6GiB")
    p.add_argument("--cpu-memory", default="6GiB")
    p.add_argument("--offload-dir", default=".offload_qwen_observer")
    p.add_argument(
        "--compare",
        action="store_true",
        help="run the same question at A / neutral / B and exit",
    )
    p.add_argument(
        "--no-interactive",
        action="store_true",
        help="answer the initial question once and exit",
    )
    p.add_argument(
        "--receipt",
        default="results/qwen_observer_chat.json",
    )
    return p.parse_args()


def input_device(model) -> torch.device:
    emb = model.get_input_embeddings()
    hook = getattr(emb, "_hf_hook", None)
    execution_device = getattr(hook, "execution_device", None)
    if execution_device is not None:
        return torch.device(execution_device)
    for parameter in emb.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs = {
        "dtype": dtype,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        kwargs["max_memory"] = {
            0: args.gpu_memory,
            "cpu": args.cpu_memory,
        }
        kwargs["offload_folder"] = args.offload_dir
        kwargs["offload_state_dict"] = True
        print(
            "Qwen safe-memory profile: "
            f"cuda:0={args.gpu_memory}, cpu={args.cpu_memory}, "
            f"disk={args.offload_dir}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model, **kwargs
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def system_message(source_a: str, source_b: str) -> str:
    return (
        "You answer questions using the two reference records below. "
        "They may conflict. Give a concise answer grounded in the records. "
        "Do not invent a trust policy and do not discuss hidden model controls.\n\n"
        "[BEGIN SOURCE A]\n"
        + source_a
        + "\n[END SOURCE A]\n\n"
        "[BEGIN SOURCE B]\n"
        + source_b
        + "\n[END SOURCE B]"
    )


def render_chat(tokenizer, messages):
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def encode_rendered(tokenizer, rendered: str):
    encoded = tokenizer(
        rendered,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    offsets = [tuple(x) for x in encoded["offset_mapping"]]
    return input_ids, attention_mask, offsets


def calibrate(model, tokenizer, args, messages):
    rendered = render_chat(tokenizer, messages)
    ids, mask, offsets = encode_rendered(tokenizer, rendered)
    spans = locate_source_spans(
        rendered, offsets, args.source_a, args.source_b
    )

    device = input_device(model)
    ids = ids.to(device)
    mask = mask.to(device)
    layers = [
        int(x.strip())
        for x in args.layers.split(",")
        if x.strip()
    ]
    max_layer = len(model.model.layers) - 1
    for layer in layers:
        if layer < 0 or layer > max_layer:
            raise ValueError(
                f"layer {layer} outside Qwen range 0..{max_layer}"
            )

    print(
        "Calibration prefill: capturing post-RoPE geometry at layers "
        + ",".join(str(x) for x in layers)
    )
    captures = capture_qwen_geometry(
        model,
        ids,
        mask,
        layers=layers,
    )
    plan = choose_observer_plan(
        captures,
        spans,
        target_margin=args.target_margin,
        max_ratio=args.max_ratio,
        num_heads=args.observer_heads,
    )

    print("Frozen observer head plan:")
    for item in plan.heads:
        print(
            f"  L{item.layer:02d} Q{item.query_head:02d}/KV{item.kv_head}: "
            f"sym_mass={item.symmetric_source_mass:.3f} "
            f"A={item.mass_when_a:.3f} B={item.mass_when_b:.3f} "
            f"ratioA={item.ratio_when_a:.3f} ratioB={item.ratio_when_b:.3f}"
        )

    if plan.heads and plan.heads[0].symmetric_source_mass < args.min_calibration_mass:
        print(
            "WARNING: best symmetric source mass is below "
            f"{args.min_calibration_mass:.3f}; the observer may be too weak "
            "for a visible language-level effect."
        )
    return plan


def prepare_turn(tokenizer, messages, args):
    rendered = render_chat(tokenizer, messages)
    ids, mask, offsets = encode_rendered(tokenizer, rendered)
    spans = locate_source_spans(
        rendered, offsets, args.source_a, args.source_b
    )
    return rendered, ids, mask, spans


@torch.inference_mode()
def answer_once(
    model,
    tokenizer,
    controller,
    messages,
    args,
    *,
    trust: float,
):
    controller.set_trust(trust)
    _rendered, ids, mask, spans = prepare_turn(
        tokenizer, messages, args
    )
    controller.set_spans(spans)
    controller.begin_generation()

    device = input_device(model)
    ids = ids.to(device)
    mask = mask.to(device)
    prompt_len = ids.shape[1]

    outputs = model.generate(
        input_ids=ids,
        attention_mask=mask,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_ids = outputs[0, prompt_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return text, controller.summary()


def print_state(controller):
    side = (
        "A" if controller.trust > 0.0
        else "B" if controller.trust < 0.0
        else "neutral"
    )
    print(
        f"observer trust={controller.trust:+.3f} ({side}); "
        f"target_margin={controller.target_margin:.3f}; "
        f"max_ratio={controller.max_ratio:.3f}"
    )


def main():
    args = parse_args()
    args.trust = max(-1.0, min(1.0, args.trust))

    model, tokenizer = load_model(args)
    base_messages = [
        {
            "role": "system",
            "content": system_message(args.source_a, args.source_b),
        },
        {"role": "user", "content": args.question},
    ]

    plan = calibrate(model, tokenizer, args, base_messages)
    controller = QwenObserverController(
        model,
        plan,
        trust=args.trust,
        target_margin=args.target_margin,
        max_ratio=args.max_ratio,
    )
    controller.install()

    receipt = {
        "model": args.model,
        "layers_considered": args.layers,
        "observer_heads": [
            {
                "layer": h.layer,
                "query_head": h.query_head,
                "kv_head": h.kv_head,
                "symmetric_source_mass": h.symmetric_source_mass,
                "mass_when_a": h.mass_when_a,
                "mass_when_b": h.mass_when_b,
                "ratio_when_a": h.ratio_when_a,
                "ratio_when_b": h.ratio_when_b,
            }
            for h in plan.heads
        ],
        "runs": [],
    }

    try:
        if args.compare:
            print("\n=== Same prompt, three observer states ===")
            for label, trust in (
                ("A", +1.0),
                ("neutral", 0.0),
                ("B", -1.0),
            ):
                text, summary = answer_once(
                    model,
                    tokenizer,
                    controller,
                    base_messages,
                    args,
                    trust=trust,
                )
                print(f"\n[{label} trust {trust:+.1f}]\n{text}")
                print("observer:", json.dumps(summary, indent=2))
                receipt["runs"].append(
                    {
                        "mode": label,
                        "trust": trust,
                        "question": args.question,
                        "answer": text,
                        "observer": summary,
                    }
                )
            return

        messages = list(base_messages)
        text, summary = answer_once(
            model,
            tokenizer,
            controller,
            messages,
            args,
            trust=args.trust,
        )
        print_state(controller)
        print("\nQwen:", text)
        print("observer:", json.dumps(summary, indent=2))
        receipt["runs"].append(
            {
                "mode": "initial",
                "trust": controller.trust,
                "question": args.question,
                "answer": text,
                "observer": summary,
            }
        )
        messages.append({"role": "assistant", "content": text})

        if args.no_interactive:
            return

        print(
            "\nCommands: /a, /b, /neutral, /trust <[-1,1]>, "
            "/state, /compare <question>, /quit"
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
            if user == "/a":
                controller.set_trust(+1.0)
                print_state(controller)
                continue
            if user == "/b":
                controller.set_trust(-1.0)
                print_state(controller)
                continue
            if user == "/neutral":
                controller.set_trust(0.0)
                print_state(controller)
                continue
            if user.startswith("/trust "):
                controller.set_trust(float(user.split(maxsplit=1)[1]))
                print_state(controller)
                continue
            if user == "/state":
                print_state(controller)
                continue
            if user.startswith("/compare "):
                question = user.split(maxsplit=1)[1]
                trial = messages + [{"role": "user", "content": question}]
                for label, trust in (
                    ("A", +1.0),
                    ("neutral", 0.0),
                    ("B", -1.0),
                ):
                    ans, diag = answer_once(
                        model,
                        tokenizer,
                        controller,
                        trial,
                        args,
                        trust=trust,
                    )
                    print(
                        f"\n[{label} {trust:+.1f}] {ans}\n"
                        + json.dumps(diag, indent=2)
                    )
                continue

            messages.append({"role": "user", "content": user})
            text, summary = answer_once(
                model,
                tokenizer,
                controller,
                messages,
                args,
                trust=controller.trust,
            )
            print("\nQwen:", text)
            print("observer:", json.dumps(summary, indent=2))
            receipt["runs"].append(
                {
                    "mode": "chat",
                    "trust": controller.trust,
                    "question": user,
                    "answer": text,
                    "observer": summary,
                }
            )
            messages.append({"role": "assistant", "content": text})
    finally:
        controller.uninstall()
        out = Path(args.receipt)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(receipt, indent=2))
        print(f"\nreceipt: {out}")


if __name__ == "__main__":
    main()
