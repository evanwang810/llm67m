#!/usr/bin/env python
"""Instruction tuning on a Kaggle TPU VM v3-8, writing the same sft_*.pt.

    python finetune_tpu.py --run-dir /kaggle/working/run --hours 0.8

kaggle_run.sh picks this over finetune.py automatically when DEVICE=tpu. It
matters that it does: finetune.py chooses cuda-or-cpu, and on a TPU VM there is
no CUDA, so it would quietly fine-tune on the host CPU and either miss the
session deadline or produce nothing.

Fine tuning is a friendlier XLA target than pretraining. build_dataset already
pads every example to max_len and next_batch always yields a full batch, so the
step graph is static without anything having to change. What does have to change
is the same short list as train_tpu.py, for the same reasons: bf16 instead of a
GradScaler, xm.optimizer_step instead of scaler.step, scalars read on log
boundaries rather than every step, and the stop decision made collectively.

The one thing that is genuinely different here is the data. finetune.py walks a
single shuffled cursor, which is correct for one device and wrong for eight:
every replica would draw the identical batch, so the all-reduce would average
eight copies of the same gradient and the run would cost eight times the compute
for one device's progress. Each replica takes its own stripe of the shuffle
instead.
"""

from __future__ import annotations

import contextlib
import math
import os
import time
from pathlib import Path

for _var in ("TPU_PROCESS_ADDRESSES", "CLOUD_TPU_TASK_ID"):
    os.environ.pop(_var, None)
os.environ.setdefault("PJRT_DEVICE", "TPU")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import finetune as base_ft  # noqa: E402
from config import GPTConfig  # noqa: E402
from model import GPT, strip_prefixes  # noqa: E402
from runstate import RunDir  # noqa: E402

# Built once in the parent and inherited through fork, rather than tokenizing
# 52k examples separately in each of the replicas.
_DATA: tuple[np.ndarray, np.ndarray] | None = None
_BASE: Path | None = None


def autocast_bf16():
    try:
        return torch.autocast("xla", dtype=torch.bfloat16)
    except Exception:
        return contextlib.nullcontext()


def _mp_fn(index, args):  # noqa: ARG001
    import xla_compat as X

    dev = X.device()
    ordinal = X.ordinal()
    world = X.world_size()
    master = ordinal == 0

    def say(msg):
        if master:
            print(msg, flush=True)

    torch.manual_seed(args.seed)
    xs, ys = _DATA
    n = len(xs)

    ckpt = torch.load(_BASE, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ckpt["config"]["model"])
    if cfg.vocab_size <= base_ft.ASSISTANT_TOKEN:
        raise SystemExit(f"vocab_size {cfg.vocab_size} has no free slot for turn tokens")
    model = GPT(cfg)
    state = strip_prefixes(ckpt["model"])
    if cfg.tie_embeddings:
        state.pop("lm_head.weight", None)
    model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    base_step = int(ckpt.get("step", 0))
    del ckpt

    model = model.to(dev)
    # Same insurance as train_tpu.py: XLA has no equivalent of DDP's broadcast at
    # construction, and eight slightly different models still produce a falling
    # loss curve.
    with torch.no_grad():
        X.all_reduce_sum(list(model.parameters()), scale=1.0 / world)
    X.sync()

    optimizer = model.configure_optimizer(args.lr, args.weight_decay, (0.9, 0.95), "xla")

    per_step = args.batch_size * args.grad_accum * world
    steps_per_epoch = max(1, n // per_step)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    tokens_per_step = per_step * (args.max_len - 1)

    if master:
        print(model.param_report())
        print(f"pretrained for {base_step:,} steps")
        print(f"{world} replicas, {steps_per_epoch:,} steps/epoch, {total_steps:,} total, "
              f"{tokens_per_step:,} tokens/step", flush=True)

    rs = RunDir(args.run_dir)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)
    cursor = 0
    deadline = time.time() + args.hours * 3600
    loss_ema = None

    def next_batch():
        """This replica's stripe of the shuffle.

        Every replica draws the same permutation from the same seed and then
        takes the slice at its own ordinal, so the eight of them cover
        batch_size * world distinct examples per micro step with no overlap and
        no communication.
        """
        nonlocal cursor, order
        span = args.batch_size * world
        if cursor + span > n:
            order = rng.permutation(n)
            cursor = 0
        lo = cursor + ordinal * args.batch_size
        idx = order[lo : lo + args.batch_size]
        cursor += span
        return (torch.from_numpy(xs[idx]).to(dev),
                torch.from_numpy(ys[idx]).to(dev))

    model.train()
    say("finetuning\n")

    stopped_at = total_steps
    for step in range(total_steps):
        frac = step / max(1, total_steps)
        if step < args.warmup:
            lr = args.lr * (step + 1) / args.warmup
        else:
            lr = 0.1 * args.lr + 0.9 * args.lr * 0.5 * (1 + math.cos(math.pi * frac))
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), device=dev)
        for _ in range(args.grad_accum):
            x, y = next_batch()
            with autocast_bf16():
                _, loss = model(x, y)
            total = total + loss.detach().float()
            (loss / args.grad_accum).backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        X.optimizer_step(optimizer)

        # Reading the loss forces the queued graph to run and the host to wait,
        # so it happens on log boundaries only. The deadline goes through the
        # same collective: eight replicas comparing their own clocks would not
        # agree on which step to break at, and one leaving the loop early strands
        # the other seven at a collective that never completes.
        if (step + 1) % args.log_every == 0:
            avg = X.mean(total / args.grad_accum)
            lossf = float(avg.cpu())
            loss_ema = lossf if loss_ema is None else 0.9 * loss_ema + 0.1 * lossf
            left = max(0.0, deadline - time.time())
            if master:
                print(f"sft {step + 1:>6}/{total_steps} | loss {lossf:.4f} "
                      f"(ema {loss_ema:.4f}) | lr {lr:.2e} | {left / 60:.0f} min left",
                      flush=True)
                rs.append_csv({"step": base_step + step + 1, "loss": f"{lossf:.5f}",
                               "loss_ema": f"{loss_ema:.5f}", "lr": f"{lr:.6e}",
                               "phase": "sft"})
            out_of_time = X.mesh_reduce(
                "sft-stop", int(master and time.time() > deadline), max)
            if out_of_time:
                say(f"hit the {args.hours}h limit at step {step + 1}")
                stopped_at = step + 1
                break

    if master:
        X.sync()
        out = Path(args.run_dir) / f"sft_step{base_step:07d}.pt"
        payload = {
            "step": base_step,
            "model": {k: (v.detach().cpu().half() if v.is_floating_point()
                          else v.detach().cpu())
                      for k, v in model.state_dict().items()},
            "config": {"model": cfg.as_dict()},
            "sft": True,
            "user_token": base_ft.USER_TOKEN,
            "assistant_token": base_ft.ASSISTANT_TOKEN,
            "sft_dataset": args.dataset,
            "base_checkpoint": str(_BASE),
            "sft_steps": stopped_at,
        }
        tmp = out.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, out)
        print(f"\nsaved {out}  ({out.stat().st_size / 1e6:.0f} MB)", flush=True)
    X.rendezvous("sft-done")


def main() -> None:
    global _DATA, _BASE

    args = base_ft.parse_args()
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    _BASE = base_ft.find_base(args)
    print(f"base checkpoint: {_BASE}", flush=True)
    _DATA = base_ft.build_dataset(args, enc)

    import torch_xla.distributed.xla_multiprocessing as xmp

    xmp.spawn(_mp_fn, args=(args,), start_method="fork")


if __name__ == "__main__":
    main()
