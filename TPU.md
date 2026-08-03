# Running on the TPU instead

A Kaggle TPU VM v3-8 is roughly ten to fifteen times the effective throughput of
the dual T4s, which turns the three-session plan for a 101M model into one
afternoon. It is also the less proven path in this repo: the CUDA trainer has
run for hours on real hardware, and the XLA one has been tested against a stubbed
`torch_xla` on CPU. Everything below exists because of that gap.

## The cell

One cell, same as the GPU path, with the accelerator set to **TPU VM v3-8** in
the notebook sidebar.

```python
!rm -rf /kaggle/working/code && git clone -q https://github.com/evanwang810/llm67m /kaggle/working/code
!cd /kaggle/working/code && DEVICE=tpu MONITOR=1 SFT_HOURS=0.8 bash kaggle_run.sh 8.5 tpu1session 3.8e9
```

`8.5`, not 11. **Kaggle caps TPU sessions at 9 hours**, where GPU gets 12. The
script warns if you ask for more, but it cannot extend the cap.

`tpu1session`, not `67m`. The whole point of the TPU is that it moves the
compute budget, and the best model for a budget scales with its square root. One
8.5 hour v3-8 session is about 3.9e18 FLOPs, where the optimum is near 173M total
parameters on 3.5B tokens, so the preset is 19 layers at GPT-2 small's width and
takes roughly 6.2 of the 8.5 hours. Running `67m` here would finish in about
three hours and leave half the session unspent on a model the T4s could have
reached anyway. Running the T4 preset on a TPU is the most common way to get a
TPU that is fast and pointless.

`DEVICE=tpu` is optional. With no setting the script picks GPU if it sees a CUDA
device and TPU otherwise, so on a TPU notebook it does the right thing on its
own. Set it explicitly when you want the run to fail loudly rather than silently
train on the wrong accelerator.

## The queue

Kaggle has far fewer TPUs than people who want them, and it has been that way
for a long time. Queues in the dozens and waits of one to two hours are normal,
not an outage. Committed runs queue longer than interactive ones because they
ask for a guaranteed multi-hour slot.

**Waiting costs nothing.** Quota is counted against execution time, not queue
time, so a commit that sits pending overnight has spent none of your 20 weekly
hours. Leave it queued; it will run when capacity frees.

Telling a queue apart from a hang takes one look at the log. Queued means
*zero* output, no banner at all. Once the container starts, the very first thing
printed is:

```
=== TPU preflight: proving the device works before spending the session ===
```

If you can see that line, you have a TPU and the code is running. If you cannot,
you are still waiting for hardware and nothing in this repo has executed yet.

Since checkpoints are interchangeable between the two trainers, the queue is
worth hedging against rather than waiting on: start a GPU commit as well, and
take whichever lands first.

## The preflight

Before it trains anything, the run spends about ten minutes proving the TPU
works. Five gates, cheapest first, and any failure exits without starting:

| gate | checks | why |
|---|---|---|
| 1 | `torch_xla` imports and reports a TPU | wrong accelerator setting |
| 2 | a matmul on device gives the right answer | backend wired up but broken |
| 3 | all replicas start and can `all_reduce` | the usual spawn failure |
| 4 | the real model trains at a real speed | **the one that matters** |
| 5 | a checkpoint survives a save and reload | untrustworthy output |

Gate 4 is the point of the whole exercise. A TPU that is present, passes every
correctness check and trains at 9k tokens/sec is the expensive failure: nothing
raises, the log looks healthy, and nine hours later there is a quarter of a
model. Two things cause it.

**Silent host fallback.** An operation with no XLA lowering runs on the CPU, and
every step pays a round trip off the device.

**Recompilation.** XLA compiles one program per distinct graph. Anything that
changes the shape, or bakes a changing Python scalar into the graph, makes it
compile again, and a compile costs far more than a step. The learning rate is
the most likely trigger, since it moves on every step during warmup and decay.
Gate 4 therefore varies the learning rate across its measured steps on purpose,
then reads XLA's `CompileTime` counter to see whether the count plateaued. Still
climbing means the run would spend its life compiling.

So gate 4 measures throughput and compares it to a floor, defaulting to 80k
tokens/sec against the T4 path's measured 42k. Below the floor it aborts:

```
PREFLIGHT FAILED at gate 4 (throughput)
  31k tok/s is below the 80k floor
  The TPU is running but something is falling back to the host.
```

Nothing has been spent at that point except the preflight.

To override: `MIN_TOK_S=50000` moves the floor, `PREFLIGHT=0` skips the check
entirely. Skipping is only sensible when you already know why it failed.

Nothing here can damage a TPU. The risk being managed is a wasted session.

## What differs from the CUDA path

`train_tpu.py` is a separate file rather than a branch inside `train.py`. The
two disagree on nearly everything structural — torchrun against `xmp.spawn`,
NCCL against XLA collectives, fp16 with a GradScaler against native bf16, eager
execution against a traced graph — and threading that through the working
trainer would put it at risk to add an unproven one. Everything genuinely shared
is imported, including `save_checkpoint`.

Five deliberate differences:

**No GradScaler.** bf16 has fp32's exponent range, so there is nothing to
rescale. The checkpoint still carries a null `scaler` key so the formats match.

**No DDP.** Gradients accumulate into `.grad` across micro batches and
`xm.optimizer_step` all-reduces once, which is what DDP's `no_sync` was
emulating anyway.

**An explicit weight sync at startup.** DDP broadcasts rank 0's weights at
construction; XLA has no equivalent. Eight replicas each training a slightly
different model would still produce a falling loss curve, so every replica seeds
identically and then all-reduces its parameters once before the first step.

**Control flags and loss values are read at most once every `log_every` steps.**
Pulling a scalar off the device forces the graph to execute and the host to
wait. Save timing comes off the master replica's clock alone and is distributed
through that same collective — eight wall clocks compared independently will not
agree on where a save boundary falls, and one replica waiting at a rendezvous
the others never reach is a hang with no watchdog to end it.

**Samples are generated on the host.** Incremental decoding grows the sequence
by one token per pass, so on device it would compile a fresh graph per token. On
the CPU it costs under a second and the log keeps its samples.

## Checkpoints are interchangeable

A checkpoint written by the TPU trainer resumes in the CUDA trainer and the
other way round, verified on a smoke run: loss continued at 6.54 across the
switch with no spike. So a TPU session that dies early can be finished on GPU,
and `chat.py`, `finetune.py` and the dashboard read either without knowing which
trainer produced it.

## If the preflight fails

| message | what it means |
|---|---|
| gate 1, `torch_xla` not importable | accelerator is not set to TPU. Do not pip install torch_xla; Kaggle's TPU image ships a matching build and installing over it breaks the runtime |
| gate 3, replicas failed to start | usually a stale TPU lock from an earlier cell in the same session. Restart the kernel and run this cell first |
| gate 4, recompilation | the step graph is not static. Report the number; it means something in the loop changes shape |
| gate 4, throughput | something is falling back to the host |
| gate 5, checkpoint mismatch | stop. The run would produce checkpoints you cannot trust |

Falling back costs one command: `DEVICE=gpu` on a GPU notebook, and the T4 path
runs exactly as before.
