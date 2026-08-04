#!/usr/bin/env python
"""Train the same GPT on a Kaggle TPU VM v3-8, in the same checkpoint format.

    python train_tpu.py --preset 67m --data-dir /kaggle/working/tokens

A separate file rather than a branch inside train.py. The two paths disagree on
almost everything structural: torchrun versus xmp.spawn, NCCL versus XLA
collectives, fp16 with a GradScaler versus native bf16, eager execution versus a
traced graph. Threading all of that through the CUDA trainer would put the
working path at risk to add an unproven one. Everything that is genuinely shared
is imported from train.py, including save_checkpoint, so the checkpoints this
writes are byte-identical in structure and chat.py, finetune.py and the
dashboard read them without knowing which trainer produced them.

Five things differ from the CUDA path, and each one is deliberate:

  * No GradScaler. TPU does bf16 natively, which has fp32's exponent range, so
    there is nothing to rescale. The checkpoint still carries a null scaler key
    so train.py can resume from a TPU checkpoint and vice versa.

  * No DDP. Gradients accumulate into .grad across the micro batches and
    xm.optimizer_step does the all-reduce once, which is what DDP's no_sync
    dance was emulating anyway.

  * Every replica initialises from the same seed and then all-reduces its
    weights once before the first step. DDP broadcasts rank 0's weights at
    construction; XLA has no equivalent, and eight replicas each training their
    own slightly different model would still produce a falling loss curve.

  * Control flags and loss values are read at most once every log_every steps.
    Pulling a scalar off the device forces the graph to execute and the host to
    wait; doing it per step is a large throughput loss for a number nobody
    reads.

  * Samples are generated on the CPU from the checkpoint snapshot. Incremental
    decoding grows the sequence by one token each pass, so on device it would
    compile a fresh graph per token. On the host it costs under a second and
    the log keeps its samples.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path

# Kaggle's TPU image presets these for its own launcher and torch_xla reads them
# at import, coming up half configured if they survive. They must go before the
# first torch_xla import anywhere in the process.
for _var in ("TPU_PROCESS_ADDRESSES", "CLOUD_TPU_TASK_ID"):
    os.environ.pop(_var, None)
os.environ.setdefault("PJRT_DEVICE", "TPU")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import train as cuda  # noqa: E402  (shared arg parsing, schedule, checkpoint IO)
from config import GPTConfig, non_embedding_params  # noqa: E402
from data import BatchSampler, Corpus, val_batches  # noqa: E402
from model import GPT  # noqa: E402
from runstate import RunDir  # noqa: E402


class NullScaler:
    """Stands in for GradScaler so the checkpoint payload keeps its shape."""

    def state_dict(self):
        return None

    def get_scale(self):
        return 1.0


_HOST_MODEL: GPT | None = None  # reused by _sample_on_host


def autocast_bf16():
    try:
        return torch.autocast("xla", dtype=torch.bfloat16)
    except Exception:
        return contextlib.nullcontext()


def _mp_fn(index, args):  # noqa: ARG001  (xmp.spawn passes the process index)
    import xla_compat as X

    dev = X.device()
    ordinal = X.ordinal()
    world = X.world_size()
    master = ordinal == 0

    def say(msg):
        if master:
            print(msg, flush=True)

    process_start = time.time()
    session_start = args.session_start or process_start
    deadline = session_start + args.deadline_hours * 3600 - args.reserve_minutes * 60

    # Identical on every replica: the sampler shards by ordinal arithmetic, not
    # by RNG, so there is nothing that wants a per-replica stream, and model init
    # must match exactly across replicas.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = Path(args.run_dir)
    rs = RunDir(run_dir)

    if args.smoke_test and master:
        cuda.make_synthetic_corpus(Path(args.data_dir))
    X.rendezvous("corpus-ready")

    train_corpus = Corpus(args.data_dir, args.block_size, "train")
    try:
        val_corpus = Corpus(args.data_dir, args.block_size, "val")
    except (ValueError, KeyError):
        val_corpus = None

    cfg = GPTConfig(
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        block_size=args.block_size,
        vocab_size=((train_corpus.vocab_size + 63) // 64) * 64,
        dropout=args.dropout, tie_embeddings=not args.no_tie,
    )

    # Build and resume on the host, then move once. Loading a state dict into
    # tensors that already live on device works but traces a copy per parameter
    # for no reason.
    model = GPT(cfg)
    fingerprint = train_corpus.fingerprint()
    step, tokens_seen = 0, 0
    decay_start: int | None = None
    best_val: float | None = None
    resumed_ema: float | None = None
    resume_optimizer = None

    info = cuda.pick_resume_checkpoint(args)
    if info is not None:
        # Every replica loads the same file, so they end up identical; only
        # master narrates it, otherwise the log gets eight copies of everything.
        quiet = contextlib.nullcontext() if master else open(os.devnull, "w")
        with (contextlib.nullcontext() if master else contextlib.redirect_stdout(quiet)):
            ckpt, has_optim = cuda.load_resume(info, model, args, fingerprint)
        if not master:
            quiet.close()
        step = int(ckpt["step"])
        tokens_seen = int(ckpt.get("tokens_seen", 0))
        decay_start = ckpt.get("decay_start")
        best_val = ckpt.get("best_val")
        resumed_ema = ckpt.get("loss_ema")
        resume_optimizer = ckpt["optimizer"] if has_optim else None
        del ckpt
    elif master:
        print("no checkpoint found, initializing fresh", flush=True)

    model = model.to(dev)

    # Insurance against silent divergence. These should already be identical, so
    # the mean is a no-op numerically, but it is one cheap collective against a
    # failure that looks exactly like normal training.
    with torch.no_grad():
        X.all_reduce_sum(list(model.parameters()), scale=1.0 / world)
    X.sync()

    # A cross-device move used to untie wte from lm_head, which cost an extra
    # embedding of weights, gradients and Adam moments per replica and, worse,
    # produced checkpoints whose output head is discarded on resume. GPT._apply
    # reties now; this refuses to run if that ever stops working, because the
    # symptom is a model that trains perfectly well and is quietly the wrong one.
    live = sum(p.numel() for p in model.parameters())
    want = non_embedding_params(cfg) + cfg.vocab_size * cfg.n_embd * (1 if cfg.tie_embeddings else 2)
    if live != want:
        raise SystemExit(
            f"model has {live:,} parameters on device, expected {want:,}. "
            f"Weight tying did not survive the move to {dev}."
        )

    optimizer = model.configure_optimizer(args.lr, args.weight_decay, (0.9, 0.95), "xla")
    if resume_optimizer is not None:
        optimizer.load_state_dict(resume_optimizer)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(dev)
        del resume_optimizer
    scaler = NullScaler()

    tokens_per_step = args.micro_batch * args.grad_accum * world * args.block_size
    if master:
        print("=" * 78)
        print(model.param_report())
        print(f"target check: non-embedding = {non_embedding_params(cfg) / 1e6:.2f}M")
        print(f"corpus: {train_corpus.total_tokens / 1e9:.3f}B tokens, "
              f"{train_corpus.n_blocks:,} blocks")
        print(f"batch: {args.micro_batch} x {args.grad_accum} accum x {world} core x "
              f"{args.block_size} = {tokens_per_step:,} tokens/step")
        print(f"device=TPU replicas={world} bf16=native")
        if info is not None:
            print(f"resumed at step {step}, tokens_seen {tokens_seen / 1e9:.3f}B, "
                  f"decay_start={decay_start}")
        print(f"deadline in {(deadline - time.time()) / 3600:.2f}h")
        print("=" * 78, flush=True)

    if args.decay and decay_start is None:
        decay_start = step

    sampler = BatchSampler(train_corpus, args.micro_batch, args.grad_accum,
                           world, ordinal, args.seed)
    if master:
        rs.init_csv()

    loss_ema = resumed_ema
    last_sample = ""
    last_save = time.time()
    last_milestone = time.time()
    saved_at_step = step
    last_log_t = time.time()
    last_log_step = step
    secs_per_step = 0.0
    lr = cuda.lr_at(step, args, decay_start)
    val_loss: float | None = None
    stop_reason = ""

    def evaluate():
        if val_corpus is None:
            return None
        model.eval()
        total = torch.zeros((), device=dev)
        n = 0
        with torch.no_grad():
            for x, y in val_batches(val_corpus, args.micro_batch, args.eval_batches, dev):
                with autocast_bf16():
                    _, loss = model(x, y)
                total = total + loss.detach().float()
                n += 1
        model.train()
        avg = X.mean(total / max(1, n))
        return float(avg.cpu())

    model.train()
    say("training")

    while True:
        # ---- control, agreed across replicas ----
        # Only on log boundaries: each of these is a host round trip and the
        # answer cannot change meaningfully in twenty steps.
        if step % args.log_every == 0:
            local = 0
            if master:
                hit = rs.poll_flags()
                local |= int("SAVE_NOW" in hit)
                local |= int("DECAY_NOW" in hit) << 1
                local |= int("STOP_NOW" in hit) << 2
                # Save timing comes off master's clock only. Every replica has
                # to enter the save rendezvous on the same step, and eight
                # independently compared wall clocks will not agree on where
                # the boundary falls; one replica waiting at a rendezvous that
                # the others never reach is a hang with no watchdog to end it.
                if (time.time() - last_save) / 60.0 >= args.save_every_min:
                    local |= 1 << 3
                if (args.milestone_every_min > 0 and
                        (time.time() - last_milestone) / 60.0 >= args.milestone_every_min):
                    local |= 1 << 4
            if cuda.STOP_REQUESTED or time.time() > deadline:
                local |= 1 << 2
                stop_reason = stop_reason or "deadline"
            if args.max_steps and step >= args.max_steps:
                local |= 1 << 2
                stop_reason = stop_reason or "max-steps"
            if args.auto_decay and decay_start is None and secs_per_step > 0:
                if deadline - time.time() <= args.decay_steps * secs_per_step * 1.05:
                    local |= 1 << 1
            flags = X.mesh_reduce("ctl", local, lambda vs: int(np.bitwise_or.reduce(vs)))
            force_save = bool(flags & 1) or bool(flags & 8)
            want_decay = bool(flags & 2)
            want_stop = bool(flags & 4)
            is_milestone = bool(flags & 16)
        else:
            force_save = want_decay = want_stop = is_milestone = False

        if want_decay and decay_start is None:
            decay_start = step
            say(f"[step {step}] entering decay phase over {args.decay_steps} steps")
        if decay_start is not None and step >= decay_start + args.decay_steps:
            want_stop = True
            stop_reason = stop_reason or "decay-complete"
        if want_stop:
            stop_reason = stop_reason or "requested"
            break

        # ---- one optimizer step ----
        lr = cuda.lr_at(step, args, decay_start)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=dev)
        for micro in range(args.grad_accum):
            x, y = sampler.batch(step, micro, dev)
            with autocast_bf16():
                _, loss = model(x, y)
            loss_sum = loss_sum + loss.detach().float()
            (loss / args.grad_accum).backward()

        grad_norm_t = torch.zeros((), device=dev)
        if args.grad_clip > 0:
            # Stays a device tensor. float() here would sync every step.
            grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        X.optimizer_step(optimizer)  # all-reduce + step + graph execution

        step += 1
        tokens_seen += tokens_per_step

        # ---- logging ----
        if step % args.log_every == 0 or force_save:
            lossf = X.mean(loss_sum / args.grad_accum)
            lossf = float(lossf.cpu())
            grad_norm = float(grad_norm_t.cpu()) if args.grad_clip > 0 else 0.0
            loss_ema = lossf if loss_ema is None else 0.9 * loss_ema + 0.1 * lossf
            now = time.time()
            dt = max(1e-6, now - last_log_t)
            steps_done = max(1, step - last_log_step)
            secs_per_step = dt / steps_done
            tok_per_s = steps_done * tokens_per_step / dt
            last_log_t, last_log_step = now, step

            if master:
                remaining = max(0.0, deadline - now)
                print(f"step {step:>7} | loss {lossf:.4f} (ema {loss_ema:.4f}) | lr {lr:.2e} | "
                      f"{tok_per_s / 1e3:.1f}k tok/s | gn {grad_norm:.2f} | "
                      f"{tokens_seen / 1e9:.3f}B tok | {remaining / 3600:.2f}h left", flush=True)
                rs.append_csv({
                    "step": step, "wall_s": round(now - session_start, 1), "tokens": tokens_seen,
                    "lr": f"{lr:.6e}", "loss": f"{lossf:.5f}", "loss_ema": f"{loss_ema:.5f}",
                    "val_loss": "" if val_loss is None else f"{val_loss:.5f}",
                    "grad_norm": f"{grad_norm:.3f}", "scale": 1,
                    "tok_per_s": round(tok_per_s, 1),
                    "phase": cuda.phase_of(step, args, decay_start),
                })
                rs.write_status({
                    "step": step, "tokens_seen": tokens_seen, "loss": lossf, "loss_ema": loss_ema,
                    "val_loss": val_loss, "best_val": best_val, "lr": lr,
                    "tok_per_s": tok_per_s, "secs_per_step": secs_per_step,
                    "phase": cuda.phase_of(step, args, decay_start), "decay_start": decay_start,
                    "decay_steps": args.decay_steps, "max_steps": args.max_steps or None,
                    "elapsed_s": now - session_start, "remaining_s": remaining,
                    "deadline_hours": args.deadline_hours, "epoch": sampler.epoch_at(step),
                    "world_size": world, "tokens_per_step": tokens_per_step,
                    "non_embedding_params": non_embedding_params(cfg),
                    "last_save_step": saved_at_step,
                    "last_save_ago_s": round(time.time() - last_save, 1),
                    "sample_prompt": args.sample_prompt, "sample": last_sample,
                    "device": "tpu",
                    "saving": False,
                    "run_dir": str(run_dir), "pid": os.getpid(), "alive": True,
                })

        # ---- eval ----
        if args.eval_every and step % args.eval_every == 0:
            val_loss = evaluate()
            if val_loss is not None:
                if best_val is None or val_loss < best_val:
                    best_val = val_loss
                say(f"step {step:>7} | val loss {val_loss:.4f} (best {best_val:.4f})")

        # ---- periodic save ----
        if force_save:
            if master:
                X.sync()
                p = cuda.save_checkpoint(run_dir, model, optimizer, scaler, step, decay_start,
                                         tokens_seen, args, fingerprint, best_val,
                                         not args.no_eval_copy, args.keep_checkpoints, loss_ema,
                                         args.keep_weights, is_milestone)
                print(f"saving {p.name} in the background"
                      + ("  [milestone kept]" if is_milestone else ""), flush=True)
                if args.sample_tokens > 0:
                    last_sample = _sample_on_host(model, cfg, args)
                    print(f'  sample @ {step}: "{last_sample}"', flush=True)
                    with open(run_dir / "samples.txt", "a", encoding="utf-8",
                              errors="replace") as f:
                        f.write(f"step {step}\tloss {loss_ema or float('nan'):.4f}\t"
                                f"{last_sample}\n")
            last_save = time.time()
            if is_milestone:
                last_milestone = last_save
            saved_at_step = step
            X.rendezvous("saved")

    # ---- final save ----
    if master:
        print(f"stopping: {stop_reason}. writing final checkpoint", flush=True)
        X.sync()
        p = cuda.save_checkpoint(run_dir, model, optimizer, scaler, step, decay_start,
                                 tokens_seen, args, fingerprint, best_val,
                                 not args.no_eval_copy, max(args.keep_checkpoints, 2),
                                 loss_ema, args.keep_weights, True, blocking=True)
        rs.write_status({
            "step": step, "tokens_seen": tokens_seen, "loss": loss_ema, "loss_ema": loss_ema,
            "val_loss": val_loss, "best_val": best_val,
            "phase": cuda.phase_of(step, args, decay_start),
            "decay_start": decay_start, "decay_steps": args.decay_steps,
            "max_steps": args.max_steps or None,
            "elapsed_s": time.time() - session_start, "remaining_s": 0,
            "deadline_hours": args.deadline_hours, "epoch": sampler.epoch_at(step),
            "tok_per_s": None, "secs_per_step": secs_per_step, "lr": lr,
            "alive": False, "stop_reason": stop_reason, "final_checkpoint": p.name,
            "run_dir": str(run_dir), "last_save_step": step, "device": "tpu",
            "tokens_per_step": tokens_per_step, "world_size": world,
            "non_embedding_params": non_embedding_params(cfg),
        })
        print(f"final: {p}  step={step}  tokens={tokens_seen / 1e9:.3f}B", flush=True)
    X.rendezvous("done")


def _sample_on_host(model, cfg, args) -> str:
    """Generate from a host copy of the weights.

    Incremental decoding changes the sequence length every pass, so on device
    each token would be a new graph and a new compilation. Copying ~100M
    parameters to the host costs a fraction of a second and this only runs at
    checkpoints.
    """
    global _HOST_MODEL
    try:
        # Built once and refilled. A fresh GPT per checkpoint means allocating
        # another 173M fp32 parameters, ~700MB, on a box that is already holding
        # eight replicas and a multi-gigabyte checkpoint snapshot.
        if _HOST_MODEL is None:
            _HOST_MODEL = GPT(cfg)
        host = _HOST_MODEL
        host.load_state_dict({k: v.detach().float().cpu()
                              for k, v in model.state_dict().items()}, strict=False)
        host.eval()
        return cuda.sample_text(host, torch.device("cpu"), args.sample_prompt,
                                args.sample_tokens)
    except Exception as exc:
        return f"<sampling failed: {type(exc).__name__}: {exc}>"


def main() -> None:
    args = cuda.parse_args()
    cuda.apply_preset(args)
    if args.smoke_test:
        cuda.apply_smoke_defaults(args)
    if not args.data_dir:
        raise SystemExit("--data-dir is required (or use --smoke-test)")

    import signal

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(ValueError, OSError):
            signal.signal(sig, cuda._handle_signal)

    import torch_xla.distributed.xla_multiprocessing as xmp

    # fork, not spawn: a notebook process cannot re-exec itself, and on v2/v3
    # torch_xla wants one process per chip with a thread per core, so nprocs is
    # left for it to decide.
    xmp.spawn(_mp_fn, args=(args,), start_method="fork")


if __name__ == "__main__":
    main()
