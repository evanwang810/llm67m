# llm67m

A 68.8M non-embedding parameter GPT trained on FineWeb-Edu across many Kaggle
free-tier sessions. Built so you can stop and restart whenever without losing a
step, and watch it with a small CPU dashboard.

## Start here

**Just want one run?** [ONERUN.md](ONERUN.md). One notebook, two cells, one
button, tokenizing and training in the same session. Ten minutes of clicking and
you get a finished model the next day. Use `--preset 1session`.

**Want the multi-week version?** Keep reading, then
[QUICKSTART.md](QUICKSTART.md).

## The four things you actually do (multi-session)

1. Nothing. Every notebook cell clones this repo from GitHub, so updating the code
   is a `git push` here and a cell rerun there.
2. Tokenize FineWeb-Edu in a free **CPU** Kaggle notebook, about an hour, once.
   Its output attaches straight to the training notebook, so nothing large ever
   moves across your home connection.
3. New notebook, accelerator **GPU T4 x2**, two cells, **Save & Run All**, walk
   away for eleven hours.
4. Next session: download the `run` folder, upload it as a new version of a
   `llm67m-ckpt` dataset, attach it, Save & Run All again. It finds the
   checkpoint and continues at the exact step on its own.

Repeat step 4 until it sounds good. On the final session add `--decay`.

## Files

| file | what it does |
| --- | --- |
| `config.py` | model and training config, exact parameter accounting. `python config.py 67` prints configs near a target size |
| `model.py` | GPT with RoPE, RMSNorm, no biases, tied embeddings |
| `data.py` | O(1)-resumable deterministic loader over uint16 shards |
| `train.py` | DDP, fp16, WSD schedule, resume, deadline, atomic saves |
| `dashboard.py` | Gradio UI: live graphs, progress bar, save button, completions, chat |
| `monitor.py` | terminal version of the same thing, ASCII loss chart included |
| `finetune.py` | instruction tuning on top of a pretrained checkpoint |
| `kaggle_run.sh` | one command for a whole session, takes an hour budget |
| `runstate.py` | the CSV / status / flag-file protocol both sides speak |
| `tokenize_fineweb.py` | local pre-tokenization to shards |
| `check_resume.py` | proves a resume was clean |
| `kaggle_driver.py` | optional Kaggle API helper |
| `QUICKSTART.md` | click-by-click browser walkthrough, start here |
| `KAGGLE_CELLS.md` | just the notebook cells |

## Which size

Pick one before your first session and stay with it, because resuming into a
different architecture is refused on purpose.

```
--preset 1session  10L / 512d /  8H    31.47M non-emb,  57.22M total
--preset 67m       14L / 640d / 10H    68.83M non-emb, 101.03M total
--preset 125m      12L / 768d / 12H    84.95M non-emb, 123.59M total
--preset 65m       11L / 704d / 11H    65.44M non-emb, 100.85M total
```

**Doing a single session? Use `1session`.** One dual-T4 session is roughly 4e17
FLOPs, where the compute-optimal model is about 58M total parameters on 1.2B
tokens. A 125M model over the same eleven hours sees half as many tokens and
comes out worse. Bigger is only better once you are stacking sessions.

**Doing the multi-week version? I would take `125m`.** Two reasons. Your realistic budget is around 5e18 FLOPs
over a month, and Chinchilla-optimal for that is roughly 200M parameters, so
both sizes are undertrained and the larger one is closer to optimal rather than
further from it. More importantly 12L/768/12H is the GPT-2 small shape, so your
loss curve is directly comparable to numbers other people publish, which matters
a great deal when you have no independent way to tell whether training is
working. It costs about 20% throughput.

`67m` is the closest clean fit to a 67M non-embedding target with head_dim 64.
`python config.py 67` prints the nearby alternatives. All the widths here and the
padded 50304 vocab are multiples of 64, which is what the T4's fp16 tensor cores
want.

Note that the 50k GPT-2 vocab is a third of the total parameter count at this
scale. A 16k custom BPE would be more parameter efficient, but it is extra work
and the loss numbers stop being comparable to anyone else's, so this uses GPT-2
BPE.

Batch is 8 micro x 8 accum x 2 GPUs x 1024 tokens = **131,072 tokens per step**.
So 1B tokens is about 7,600 steps.

## Budget math

Expect roughly 15k to 20k tokens/sec across two T4s for this model. That is
600M to 800M tokens per 11.5 hour session, call it 5,000 steps. At about 2.5
sessions per week on a 30 GPU-hour quota, four weeks gets you 6B to 8B tokens.
For a 101M parameter model that is a 60x to 80x overtrain versus
Chinchilla-optimal, which is exactly what you want for autocomplete quality.

So tokenize about 4B tokens (8 GB of `.bin`) and you will see each token twice at
most. 2B is an acceptable minimum if your download speed is the problem.

## T4 specifics

- **No bf16.** Turing does not have it. This is fp16 with a GradScaler, and the
  loss and softmax are forced to fp32 inside autocast. That is why the scaler
  state has to be checkpointed too, see below.
- **No TF32 either.** That is an Ampere feature. Do not bother setting
  `allow_tf32`, it does nothing here.
- **No flash attention.** FlashAttention-2 needs sm80+. PyTorch SDPA silently
  falls back to the memory-efficient kernel on T4. Expected, not a bug.
- **No NVLink.** The two T4s talk over PCIe, so the gradient all-reduce is the
  scaling tax. Grad accumulation of 8 means one sync per 8 micro-batches, which
  is why the default is 8 and not 1.
- **If DDP hangs at startup**, set `NCCL_P2P_DISABLE=1`. This is a common Kaggle
  quirk and costs you very little given the accumulation.
- **torch.compile is off by default.** It usually works, costs a few minutes of
  warmup, and occasionally breaks on Kaggle's torch build. `--compile` if you
  feel lucky.
- **15 GB per GPU.** Default micro-batch 8 at 1024 context is comfortable. If
  you hit OOM, halve `--micro-batch` and double `--grad-accum` to keep the
  global batch identical.

## Instruction tuning

```bash
python finetune.py --run-dir run --hours 0.4
```

Finds the newest pretrained checkpoint on its own, trains on Alpaca-cleaned, and
writes `sft_step*.pt` beside it. Twenty to forty minutes on one GPU is plenty.

Two things make it work. The vocab is padded to 50304 while GPT-2 BPE only uses
0 to 50256, so slots 50257 and 50258 are unclaimed embedding rows that become
real `<|user|>` and `<|assistant|>` turn tokens rather than a text delimiter the
model could confuse with prose. And the loss is masked over the prompt, so only
response tokens produce a gradient: the model learns to answer rather than to
re-generate the question. The dashboard reads the `sft` flag in the checkpoint
and switches the chat tab to that format automatically.

Set expectations. A 57M model learns the *shape* of answering and stays
factually hopeless. It is a great "watch the format click" exercise and not an
assistant. If you want something that feels good to talk to, tune on one narrow
task instead of general question answering, because single-task behaviour is
learnable at this size in a way open-domain QA is not.

## Why not the TPU

Kaggle's TPU v3-8 is genuinely 5x to 15x a dual T4 on this workload, it has
native bf16 so the entire GradScaler problem disappears, and its ~20 hr/week
quota is separate from your GPU quota, so it would add to your budget rather than
replace it. The speedup is real.

The cost is that it is not a flag, it is a rewrite. `torch_xla` replaces DDP with
`xmp.spawn`, dataloading goes through `MpDeviceLoader`, saving needs `xm.save`,
and any shape variation triggers silent recompiles that present as hangs. Kaggle's
TPU stack also breaks periodically and version-pinning `torch_xla` is its own
afternoon. JAX would be the more reliable TPU path and is an even bigger rewrite.

Verdict: train on 2xT4 with code that works today. The TPU port earns its keep
only if you later want to push past roughly 350M parameters.

One thing worth checking though: if the accelerator dropdown offers anything
Ampere or newer (L4, A100), take it over the T4s. On sm80+ you get bf16 and real
flash attention, which means the loss scaler and half the caveats below stop
mattering.

## What bites you on resume

Everything below is already handled. It is listed so you know what to check if a
resume ever looks wrong.

- **Optimizer state.** Resuming without Adam's moments causes a loss spike that
  takes thousands of steps to recover. `train.py` refuses to resume from a
  weights-only file unless you pass `--allow-no-optimizer`.
- **The fp16 loss scale.** This one is subtle and specific to T4. A fresh
  GradScaler starts at 65536, overflows, halves, overflows again, and skips
  several real optimizer steps while it recalibrates. Looks exactly like a loss
  spike. The scaler state is saved and restored.
- **Adam moments on the wrong device.** After `load_state_dict` the moment
  tensors are on CPU. They get moved to the GPU explicitly.
- **Dataloader position.** Sample order is a Feistel permutation of block
  indices as a function of `(seed, step, rank)`. Resuming at step 41,237 does
  not restart the corpus and does not need a fast-forward loop. Same step means
  same batch, always.
- **Config drift.** If you change `--n-layer` and resume, it refuses to load and
  tells you exactly which fields disagree, instead of silently loading garbage.
- **Data fingerprint.** If you add shards to the dataset, the corpus size changes
  and so does the batch order. It warns rather than pretending nothing happened.
  Preferably finish a run on one dataset version.
- **The clock.** The deadline defaults to counting from when `train.py` starts,
  not when the Kaggle session started. If you spend 40 minutes poking at cells
  first, the script does not know. Pass `--session-start <unix ts>` from your
  first cell, or just leave a couple of hours of margin.
- **Half-written checkpoints.** Every save writes to a `.tmp` and then renames,
  so a session killed mid-write leaves the previous good checkpoint intact.
- **Disk.** A full checkpoint is fp32 weights plus two Adam moments, so about
  12 bytes per parameter: 1.2 GB for `125m`, 690 MB for `1session`. Three knobs
  control what survives:
  - `--keep-checkpoints 2` rolling full checkpoints, the resumable ones
  - `--keep-weights 4` rolling fp16 inference copies, a third of the size
  - `--milestone-every-min 90` permanent fp16 snapshots that are never pruned

  Milestones are the point: without them, pruning leaves you only the last hour
  of the run and you cannot compare early against late. Eight milestones of the
  `1session` model is under 1 GB.

## The WSD schedule

Warmup for 500 steps, then constant LR at 1e-3 for as long as you like, then a
`1 - sqrt(progress)` decay to 1e-5 over `--decay-steps` (default 3000).

You never have to commit to a token budget. Run as many constant-phase sessions
as you want. When you are ready to finish, either:

- launch with `--decay`, or
- press **start lr decay** in the dashboard mid-session, or
- pass `--auto-decay` so it starts decaying by itself, timed to land right
  before the deadline.

Training exits automatically once the decay phase completes.

## Dashboard

```bash
python dashboard.py --run-dir run          # local, on a downloaded run folder
python dashboard.py --run-dir /kaggle/working/run --share   # on Kaggle
```

No browser at all:

```bash
python monitor.py --run-dir run --watch      # bars, stats, ascii loss chart
python monitor.py --save                     # same three buttons, as flags
```

CPU only, so it never touches your GPU quota. On Kaggle `--share` is mandatory:
there is no port forwarding, so the only route to the UI is Gradio's public
`*.gradio.live` tunnel, which needs Internet On. A committed run cannot serve it
at all, since it executes headless. All four ways to actually see it are written
up in [ONERUN.md](ONERUN.md#seeing-the-frontend-on-kaggle).

- **live**: progress bars for session time, steps and decay, six stat cards,
  auto-refreshing loss and LR plot, and buttons for save now / start decay /
  stop and save. Buttons work across processes because they drop flag files in
  the run dir that the trainer polls.
- **completion**: pick any checkpoint by step, load it, generate with
  temperature / top-k / max-tokens, and get a per-token table with the chosen
  token's probability, the distribution's entropy, and the top 5 candidates.
  Low probability with low entropy means confidently wrong. Low probability with
  high entropy means it does not know yet.
- **chat**: same model, transcript-shaped prompt. It is a base model with no
  instruction tuning, so it will continue text rather than answer you.
- **loss log**: the full CSV, x axis in steps or tokens, optional log y.

Comparing step 20k against step 60k on the same prompt is just picking a
different entry in the checkpoint dropdown, as long as both files are somewhere
under `/kaggle/input` or the run dir.

## Pre-tokenization

**Do this on Kaggle, in a free CPU notebook**, not locally. Kaggle downloads
FineWeb-Edu much faster than a home connection, an accelerator-free notebook
spends no GPU quota, and the output mounts directly into the training notebook
via **Add Input -> Notebook Output**, so 8 GB never crosses your uplink.

```python
!python tokenize_fineweb.py --out-dir /kaggle/working/tokens --max-tokens 4e9
```

About an hour, mostly download. Needs Internet On and `pip install datasets
tiktoken`.

The same script runs locally if you prefer:

```bash
python tokenize_fineweb.py --out-dir data/fineweb-edu --max-tokens 4e9
```

Either way it is pure CPU. The Intel Arc iGPU cannot help with BPE, do not
install anything for it; tiktoken is already fast SIMD C. Safe to kill and
rerun, it records how many documents it consumed and skips them. Shards default
to 500M tokens, which is 1 GB each.

## Kaggle API, honestly

You asked whether the API is worth setting up. Partly.

**Not worth it for the training run.** The kernel metadata API has no field for
the "T4 x2" accelerator, so a pushed kernel gets one GPU and DDP never engages.
Pick the accelerator in the web editor.

**Worth it for the boring part**, which is versioning the checkpoint dataset
between sessions. That is a browser drag-and-drop of 2.8 GB otherwise, versus:

```bash
python kaggle_driver.py setup --user evanwang810
python kaggle_driver.py pull                              # grab the finished output
python kaggle_driver.py bump --message "step 61000"       # new dataset version
```

Setup is `pip install kaggle` plus dropping `kaggle.json` from your account page
into `%USERPROFILE%\.kaggle\`. Five minutes, and you can see status from the
terminal with `python kaggle_driver.py watch`.

## Smoke test

```bash
python train.py --smoke-test --run-dir scratch --max-steps 60
```

Tiny 4L/128d model, synthetic corpus generated on the fly, 5 minute deadline,
saves every 30 seconds, and it goes through the exact same code path as a real
run. Kill it, rerun it, then:

```bash
python check_resume.py --run-dir scratch
```

It locates every restart in `loss.csv` and checks the step gap, the loss jump
against the run's own noise level, and whether the loss scale reset.

Verified locally on CPU: two restarts, steps 30 to 35 and 60 to 65, loss deltas
of -0.34 and +0.09 against a typical step-to-step noise of 0.25. Clean.
