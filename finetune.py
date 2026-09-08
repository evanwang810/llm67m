#!/usr/bin/env python
"""Instruction-tune a pretrained checkpoint. Run this after train.py.

    python finetune.py --run-dir /kaggle/working/run --hours 0.5

Cheap: a single GPU for 20 to 40 minutes is plenty at this scale. It finds the
newest pretrained checkpoint by itself and writes sft_step*.pt next to it, which
the dashboard picks up and knows to chat with using the right format.

Three details worth knowing.

The data matters more here than anything else in the file. This stage does not
add knowledge, it decides behaviour, so the tuning set is what the answers end
up sounding like. The default is SmolTalk, built for SmolLM2 at roughly this
size, so its answers are short and directly written instead of being long
essays a 173M model can only imitate badly. --dataset takes a comma separated
mix of name[:config][:weight], or one of the MIXES shorthands.

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

# Shorthands for --dataset. SmolTalk is the one that matters: it was built for
# SmolLM2, which is the same size class as this model, so its answers are short
# and plainly written instead of being written for a 70B to imitate. It is also
# multi turn, which Alpaca is not, and a model tuned only on single turn data
# reads a second question as a pattern to repeat rather than as context.
MIXES = {
    "smol": "HuggingFaceTB/smoltalk:all",
    "chat": "HuggingFaceTB/smoltalk:everyday-conversations:2,"
            "HuggingFaceTB/smoltalk:smol-magpie-ultra:6,"
            "HuggingFaceTB/smoltalk:smol-summarize:1,"
            "HuggingFaceTB/smoltalk:smol-constraints:1",
    "alpaca": "yahma/alpaca-cleaned",
    "dolly": "databricks/databricks-dolly-15k",
}
DEFAULT_MIX = "chat"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default="/kaggle/working/run")
    p.add_argument("--from-checkpoint", default="", help="default: newest one found")
    p.add_argument("--dataset", default=DEFAULT_MIX,
                   help="comma separated name[:config][:weight], see MIXES for shorthands")
    p.add_argument("--max-examples", type=int, default=250_000,
                   help="cap on the assembled mix, 0 means no cap")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--epochs", type=float, default=1.0)
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


def parse_specs(spec: str) -> list[tuple[str, str | None, float]]:
    """Turn "name[:config][:weight],..." into (name, config, weight) triples.

    A dataset name always contains a slash and a weight is always numeric, so the
    three fields can be told apart positionally without needing a real syntax.
    """
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        chunk = MIXES.get(chunk, chunk)
        if "," in chunk:  # a shorthand that expanded into a mix of its own
            out += parse_specs(chunk)
            continue
        parts = chunk.split(":")
        weight = 1.0
        if len(parts) > 1 and _numeric(parts[-1]):
            weight = float(parts.pop())
        name = parts[0]
        config = parts[1] if len(parts) > 1 else None
        out.append((name, config, weight))
    return out


def _numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def row_to_turns(row: dict) -> list[tuple[str, str]]:
    """Normalise one row of any common SFT schema into [(role, content), ...].

    Every instruction set on the hub picks its own column names for the same
    three ideas, and the good multi-turn ones are all messages-shaped while the
    older single-turn ones are all instruction-shaped. Reading both here is what
    lets --dataset point anywhere without a per-dataset adapter.
    """
    msgs = row.get("messages") or row.get("conversations") or row.get("conversation")
    if isinstance(msgs, list) and msgs:
        turns = []
        for m in msgs:
            if not isinstance(m, dict):
                return []
            role = str(m.get("role") or m.get("from") or "").lower()
            content = str(m.get("content") or m.get("value") or "").strip()
            if not content:
                continue
            if role in ("assistant", "gpt", "model"):
                turns.append(("assistant", content))
            elif role in ("user", "human"):
                turns.append(("user", content))
            elif role == "system":
                turns.append(("system", content))
        return turns

    prompt = str(row.get("instruction") or row.get("prompt") or row.get("question") or "").strip()
    answer = str(row.get("output") or row.get("response") or row.get("answer")
                 or row.get("completion") or "").strip()
    extra = str(row.get("input") or row.get("context") or "").strip()
    if extra:
        prompt += "\n\n" + extra
    if not prompt or not answer:
        return []
    return [("user", prompt), ("assistant", answer)]


def build_dataset(args, enc) -> tuple[np.ndarray, np.ndarray]:
    from datasets import load_dataset

    specs = parse_specs(args.dataset)
    total_weight = sum(w for _, _, w in specs) or 1.0
    budget = args.max_examples or 0
    rng = np.random.default_rng(args.seed)

    # Collected first and encoded in one batch at the end: tiktoken releases the
    # GIL and threads the batch call, which is the difference between seconds
    # and a quarter of an hour once the mix is in the hundreds of thousands.
    convos: list[list[tuple[str, str]]] = []
    for name, config, weight in specs:
        share = int(budget * weight / total_weight) if budget else 0
        label = f"{name}" + (f":{config}" if config else "")
        print(f"loading {label}")
        rows = load_dataset(name, config, split="train") if config \
            else load_dataset(name, split="train")
        if share and len(rows) > share:
            # Sampled rather than truncated: many of these are grouped by source
            # subset, so the first N rows are not a sample of the dataset.
            rows = rows.select(rng.choice(len(rows), size=share, replace=False))
        take = [t for t in (row_to_turns(r) for r in rows) if t]
        print(f"  {len(take):,} conversations")
        convos += take

    if not convos:
        raise SystemExit(f"no usable rows in {args.dataset}; check the column names")
    rng.shuffle(convos)

    # A system prompt has no token of its own, so it rides on the first user
    # turn. The alternative is a third special token the pretrained model has
    # never seen, for a field most of the mix does not set.
    flat: list[str] = []
    for turns in convos:
        sys_txt = " ".join(c for r, c in turns if r == "system")
        first_user = True
        for role, content in turns:
            if role == "system":
                continue
            if role == "user" and first_user and sys_txt:
                content = sys_txt + "\n\n" + content
                first_user = False
            flat.append(content)
    print(f"encoding {len(flat):,} turns")
    encoded = enc.encode_ordinary_batch(flat, num_threads=8)

    L = args.max_len
    # int32 not int64: the mix is large enough that the difference is gigabytes,
    # and the widest value in either array is the 50258 assistant token.
    xs = np.full((len(convos), L - 1), enc.eot_token, dtype=np.int32)
    ys = np.full((len(convos), L - 1), -1, dtype=np.int32)
    cursor = 0
    kept = 0
    for turns in convos:
        roles = [r for r, _ in turns if r != "system"]
        ids = encoded[cursor : cursor + len(roles)]
        cursor += len(roles)

        seq: list[int] = []
        # True where the token is part of an answer, and so is worth a gradient.
        supervised: list[bool] = []
        for role, body in zip(roles, ids):
            if role == "user":
                seq += [USER_TOKEN] + body + [ASSISTANT_TOKEN]
                supervised += [False] * (len(body) + 2)
            else:
                # The turn-ending eot is supervised too, or nothing ever teaches
                # the model to stop.
                seq += body + [enc.eot_token]
                supervised += [True] * (len(body) + 1)
            if len(seq) > L:
                break
        seq, supervised = seq[:L], supervised[:L]
        n = len(seq) - 1
        if n < 8 or not any(supervised[1:]):  # nothing left to learn from
            continue
        xs[kept, :n] = seq[:-1]
        # Position i predicts seq[i+1], so the mask that matters is the one on
        # the target, not on the input.
        tgt = np.array(seq[1:], dtype=np.int32)
        keep = np.array(supervised[1:], dtype=bool)
        ys[kept, :n] = np.where(keep, tgt, -1)
        kept += 1

    turns_avg = sum(len(c) for c in convos) / len(convos)
    print(f"{kept:,} usable of {len(convos):,}, {L} tokens each, "
          f"{turns_avg:.1f} turns per conversation")
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
    hard_cap = total_steps
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
        # Stored int32 to halve the resident size of a large mix, but embedding
        # lookup and cross entropy both want int64.
        return (torch.from_numpy(xs[idx]).long().to(device),
                torch.from_numpy(ys[idx]).long().to(device))

    print("finetuning\n")
    started = time.time()
    for step in range(hard_cap):
        if step == 30:
            # Refit the schedule to the time actually available. Cutting a cosine
            # off at 20% leaves the model parked at a high learning rate, which
            # undoes a good part of what the tuning was for; better to decay
            # over the steps that will really happen.
            rate = (time.time() - started) / 30
            fits = int((deadline - started) / rate)
            if fits < total_steps:
                total_steps = max(args.warmup + 10, fits)
                print(f"schedule refit to {total_steps:,} steps to finish in "
                      f"{args.hours}h at {rate:.2f}s/step", flush=True)
        if step >= total_steps:
            break
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
