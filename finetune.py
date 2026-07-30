#!/usr/bin/env python
"""Instruction-tune a pretrained checkpoint. Run this after train.py.

    python finetune.py --run-dir /kaggle/working/run --hours 0.5

Cheap: a single GPU for 20 to 40 minutes is plenty at this scale. It finds the
newest pretrained checkpoint by itself and writes sft_step*.pt next to it, which
the dashboard picks up and knows to chat with using the right format.

Two details worth knowing.

The vocab is padded to 50304 but GPT-2 BPE only uses 0 to 50256, so slots 50257
and 50258 are free embedding rows nobody has claimed. They become real
<|user|> and <|assistant|> turn tokens rather than a text delimiter the model
could confuse with ordinary prose.

Loss is masked over the prompt: only the response tokens contribute. The model
learns to answer, not to re-generate the question. That is what ignore_index=-1
in the model's cross entropy is for.

Expect it to learn the shape of answering and remain factually hopeless. A 57M
model does not have the capacity for real question answering, so treat this as
"watch the format click", not "build an assistant".
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from model import GPT, strip_prefixes
from config import GPTConfig
from runstate import RunDir, default_search_dirs, find_checkpoints

USER_TOKEN = 50257
ASSISTANT_TOKEN = 50258


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default="/kaggle/working/run")
    p.add_argument("--from-checkpoint", default="", help="default: newest one found")
    p.add_argument("--dataset", default="yahma/alpaca-cleaned",
                   help="short, simple instruction data suits a small model")
    p.add_argument("--max-examples", type=int, default=0, help="0 uses all of them")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--hours", type=float, default=0.5, help="hard stop, saves first")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def find_base(args) -> Path:
    if args.from_checkpoint:
        return Path(args.from_checkpoint)
    cands = [c for c in find_checkpoints([Path(args.run_dir), *default_search_dirs()])
             if c["kind"] != "sft"]
    if not cands:
        raise SystemExit(
            f"no pretrained checkpoint found under {args.run_dir} or /kaggle/input.\n"
            "Run train.py first, or pass --from-checkpoint explicitly.")
    return cands[0]["path"]


def build_dataset(args, enc) -> tuple[np.ndarray, np.ndarray]:
    from datasets import load_dataset

    print(f"loading {args.dataset}")
    rows = load_dataset(args.dataset, split="train")
    if args.max_examples:
        rows = rows.select(range(min(args.max_examples, len(rows))))

    L = args.max_len
    xs = np.full((len(rows), L - 1), enc.eot_token, dtype=np.int64)
    ys = np.full((len(rows), L - 1), -1, dtype=np.int64)
    kept = 0
    for row in rows:
        prompt = row["instruction"].strip()
        extra = (row.get("input") or "").strip()
        if extra:
            prompt += "\n\n" + extra
        p_ids = [USER_TOKEN] + enc.encode_ordinary(prompt) + [ASSISTANT_TOKEN]
        r_ids = enc.encode_ordinary(row["output"].strip()) + [enc.eot_token]
        if len(p_ids) >= L - 8:  # no room left to answer in
            continue
        seq = (p_ids + r_ids)[:L]
        n = len(seq) - 1
        xs[kept, :n] = seq[:-1]
        # Mask the prompt: position i predicts seq[i+1], and we only want a
        # gradient once that target is part of the response.
        for i in range(n):
            if i + 1 >= len(p_ids):
                ys[kept, i] = seq[i + 1]
        kept += 1
    print(f"{kept:,} examples usable out of {len(rows):,}, {L} tokens each")
    return xs[:kept], ys[:kept]


def main() -> None:
    args = parse_args()
    import tiktoken

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda"
    enc = tiktoken.get_encoding("gpt2")

    base = find_base(args)
    print(f"base checkpoint: {base}")
    ckpt = torch.load(base, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ckpt["config"]["model"])
    if cfg.vocab_size <= ASSISTANT_TOKEN:
        raise SystemExit(f"vocab_size {cfg.vocab_size} has no free slot for turn tokens")
    model = GPT(cfg)
    state = strip_prefixes(ckpt["model"])
    if cfg.tie_embeddings:
        state.pop("lm_head.weight", None)
    model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    model = model.to(device)
    base_step = int(ckpt.get("step", 0))
    print(model.param_report())
    print(f"pretrained for {base_step:,} steps\n")
    del ckpt

    xs, ys = build_dataset(args, enc)
    n = len(xs)
    tokens_per_step = args.batch_size * args.grad_accum * (args.max_len - 1)
    steps_per_epoch = max(1, n // (args.batch_size * args.grad_accum))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    print(f"{steps_per_epoch:,} steps/epoch, {total_steps:,} total, "
          f"{tokens_per_step:,} tokens/step")

    optimizer = model.configure_optimizer(args.lr, args.weight_decay, (0.9, 0.95), device.type)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    autocast = (torch.amp.autocast("cuda", dtype=torch.float16) if use_fp16
                else torch.autocast("cpu", enabled=False))

    rs = RunDir(args.run_dir)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)
    cursor = 0
    deadline = time.time() + args.hours * 3600
    loss_ema = None
    model.train()

    def next_batch():
        nonlocal cursor, order
        if cursor + args.batch_size > n:
            order = rng.permutation(n)
            cursor = 0
        idx = order[cursor : cursor + args.batch_size]
        cursor += args.batch_size
        return (torch.from_numpy(xs[idx]).to(device),
                torch.from_numpy(ys[idx]).to(device))

    print("finetuning\n")
    for step in range(total_steps):
        frac = step / max(1, total_steps)
        if step < args.warmup:
            lr = args.lr * (step + 1) / args.warmup
        else:
            lr = 0.1 * args.lr + 0.9 * args.lr * 0.5 * (1 + math.cos(math.pi * frac))
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), device=device)
        for _ in range(args.grad_accum):
            x, y = next_batch()
            with autocast:
                _, loss = model(x, y)
            total += loss.detach()
            scaler.scale(loss / args.grad_accum).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        lossf = (total / args.grad_accum).item()
        loss_ema = lossf if loss_ema is None else 0.9 * loss_ema + 0.1 * lossf
        if (step + 1) % args.log_every == 0:
            left = max(0.0, deadline - time.time())
            print(f"sft {step + 1:>6}/{total_steps} | loss {lossf:.4f} "
                  f"(ema {loss_ema:.4f}) | lr {lr:.2e} | {left / 60:.0f} min left", flush=True)
            rs.append_csv({"step": base_step + step + 1, "loss": f"{lossf:.5f}",
                           "loss_ema": f"{loss_ema:.5f}", "lr": f"{lr:.6e}", "phase": "sft"})
        if time.time() > deadline:
            print(f"hit the {args.hours}h limit at step {step + 1}")
            break

    out = Path(args.run_dir) / f"sft_step{base_step:07d}.pt"
    payload = {
        "step": base_step,
        "model": {k: v.half() if v.is_floating_point() else v
                  for k, v in model.state_dict().items()},
        "config": {"model": cfg.as_dict()},
        "sft": True,
        "user_token": USER_TOKEN,
        "assistant_token": ASSISTANT_TOKEN,
        "sft_dataset": args.dataset,
        "base_checkpoint": str(base),
    }
    tmp = out.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, out)
    print(f"\nsaved {out}  ({out.stat().st_size / 1e6:.0f} MB)")
    print("open the dashboard, pick this checkpoint, and use the chat tab")


if __name__ == "__main__":
    main()
