# One run, start to finish

Everything in the browser. One notebook, two cells, one button. Tokenizing and
training happen in the same session, so there is no dataset shuffling and no
second visit. You press Run, come back the next day, and download a finished
model.

Total hands-on time: about five minutes. Then eleven hours of waiting.

---

## 1. Verify your phone number

You just signed in. Do this first, because without it the accelerator and
internet toggles are locked and nothing below works.

1. Click your **avatar**, top right, then **Settings**.
2. Scroll to **Phone Verification**.
3. Enter your number, enter the code they text you.

If it already says verified, skip ahead.

---

## 2. Create the notebook

1. **Create** in the left sidebar, then **Notebook**.
2. Click the title at the top left (it says something like "notebook1a2b3c") and
   rename it `llm67m-run`.
3. Open the right sidebar if it is closed, the **>** arrow at the top right.
4. Set two things:
   - **Accelerator**: click it, pick **GPU T4 x2**. The `x2` matters, it is what
     makes both GPUs get used. If the list happens to offer an **L4** or **A100**,
     take that instead and tell me, because on those chips we can throw away the
     fp16 loss scaler and get flash attention.
   - **Internet**: toggle **On**. Needed to clone the code and download
     FineWeb-Edu.

No **Add Input** step. The code comes from GitHub and the data is downloaded in
cell 1.

---

## 3. Paste the two cells

**Cell 1**, the empty one already there:

```python
!pip install -q tiktoken datasets
!git clone -q https://github.com/evanwang810/llm67m /kaggle/working/code
%cd /kaggle/working/code
!python tokenize_fineweb.py --out-dir /kaggle/working/tokens --max-tokens 1.5e9
```

Click **+ Code** to add a second cell.

**Cell 2**:

```python
%cd /kaggle/working/code
!torchrun --nproc_per_node=2 train.py \
    --preset 1session \
    --data-dir /kaggle/working/tokens \
    --run-dir /kaggle/working/run \
    --deadline-hours 10.5 \
    --auto-decay --decay-steps 1200 \
    --keep-checkpoints 1 --keep-weights 2 --milestone-every-min 75
```

That last line is what gives you something to compare afterwards. Every 75
minutes it keeps a permanent 114 MB `milestone_step*.pt`, so you finish with
about eight snapshots spanning the whole run and can put the two-hour model and
the ten-hour model against the same prompt in the dashboard.

Do not run anything yet.

---

## 4. Optional but recommended: a 30 minute shakedown first

Eleven hours is a long time to find out cell 2 had a typo. Before committing,
spend half an hour of quota proving the whole thing works, with the dashboard
in front of you. See [the frontend section](#5-seeing-the-frontend-on-kaggle),
option 1. Then come back here.

---

## 5. Press the button

**Save Version**, top right. In the dialog:

- Version type: **Save & Run All (Commit)**
- **Save**

Close the tab. It now runs on Kaggle's machines whether or not your laptop is on.

What it does with the eleven hours: about 25 minutes downloading and tokenizing
1.5 billion tokens of FineWeb-Edu, then roughly ten hours of training, then it
notices the deadline coming, spends the last 1200 steps decaying the learning
rate, writes a final checkpoint, and exits cleanly.

To check on it: open the notebook, **Versions** tab, click the running version.
Kaggle streams the console log live, so you will see the loss lines arriving
every 20 steps. No graphs, but it answers "is it working".

---

## 6. Come back and get your model

1. Open the notebook, click the **Output** tab.
2. You want `run/`, containing:
   - `milestone_step*.pt`, about eight permanent 114 MB fp16 snapshots across the
     run. **These are the ones you want.**
   - `weights_step*.pt`, the last two of the same thing
   - `ckpt_step*.pt`, 690 MB each, weights plus optimizer state. Only needed if
     you later decide to continue training. Skip them otherwise.
   - `loss.csv`, the whole training history
3. Download the `milestone_*.pt` files and `loss.csv`, about 1 GB. Skip
   `tokens/`, that is 3 GB you do not need.

Or skip the download entirely and look at it on Kaggle, option 3 below.

---

## 7. What to actually expect

31.5M non-embedding params (57.2M total) trained on roughly 1.2B tokens. Final
loss somewhere around 3.6 to 3.9.

That means: fluent, grammatical English. Correct syntax, plausible word choice,
sentences that start well. It will also drift off topic within a paragraph, make
up facts confidently, and cannot follow instructions at all, because it is a base
model with no instruction tuning. Given `The mitochondrion is` you should get
something that reads like a real encyclopedia sentence. Given a question you will
get more questions.

That is what one T4 session buys, and it is a genuinely fun thing to poke at.

---

# Seeing the frontend on Kaggle

First the constraint, because it explains the options. Kaggle gives you no port
forwarding, so `localhost:7860` is unreachable either way. Kaggle also has no
equivalent of Colab's `serve_kernel_port_as_iframe`. The only route to an
interactive UI is Gradio's `share=True` tunnel, which prints a public
`*.gradio.live` link and needs **Internet On**. And a committed run (Save & Run
All) executes headless with no way to interact, so it cannot serve a UI at all
while it runs.

So you cannot watch a committed run live. Pick one of these instead.

## Option 0: the terminal monitor, no browser involved

Works everywhere, including inside a committed run. Progress bars, stat table,
and an ASCII loss chart printed straight to the log.

```bash
python monitor.py --run-dir /kaggle/working/run                 # one snapshot
python monitor.py --run-dir /kaggle/working/run --watch         # every 30s
python monitor.py --run-dir /kaggle/working/run --last 2000     # zoom the chart
```

```
  session  [######################------------------------] 5h12m40s / 10h30m00s

  step         4,180           phase        constant        train loss   3.8241
  perplexity   45.8            val loss     3.8902          best val     3.8902
  throughput   28.4k tok/s     sec/step     4.61            tokens seen  0.548B

  cross entropy (whole run) . raw   o ema   V val
  10.518 |o  o  o
         |   .     o  o
         |               o
   6.421 |                    V V   .     .   .
         +----------------------------------------
```

The dashboard's three buttons are flags here:

```bash
python monitor.py --save     # checkpoint now
python monitor.py --decay    # start the lr decay phase
python monitor.py --stop     # final checkpoint, then exit cleanly
```

To get those status blocks into a **committed** run's log, start training in the
background and put the monitor in the foreground:

```python
import os, subprocess, sys
os.makedirs("/kaggle/working/run", exist_ok=True)
log = open("/kaggle/working/run/train.log", "w")
subprocess.Popen(
    [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "train.py",
     "--preset", "1session", "--data-dir", "/kaggle/working/tokens",
     "--run-dir", "/kaggle/working/run", "--deadline-hours", "10.5",
     "--auto-decay", "--decay-steps", "1200",
     "--keep-checkpoints", "1", "--keep-weights", "2", "--milestone-every-min", "75"],
    cwd="/kaggle/working/code", stdout=log, stderr=subprocess.STDOUT,
    env=dict(os.environ, PYTHONUNBUFFERED="1"),
)
!python monitor.py --run-dir /kaggle/working/run --watch --interval 300 --until-done
```

`--until-done` makes the monitor exit as soon as training writes its final
checkpoint, or bail out and dump the last 40 log lines if the trainer dies. That
keeps the cell from burning hours on a dead run.

## Option 0b: the real UI, embedded in the notebook cell

Closest thing to what Colab does. Two flavours, both in an **interactive**
session.

A live view with no server and no tunnel at all, drawn directly into the cell
output and refreshed in place:

```python
import sys; sys.path.insert(0, "/kaggle/working/code")
import dashboard
dashboard.watch_inline("/kaggle/working/run", interval=15)
```

Or the whole clickable app embedded in the cell, which does need the share
tunnel because the iframe has to point somewhere your browser can reach:

```python
import sys; sys.path.insert(0, "/kaggle/working/code")
import dashboard
dashboard.launch("/kaggle/working/run", share=True, inline=True, blocking=False)
```

`blocking=False` is the useful part: the cell returns immediately and the server
keeps running in a background thread, so you can keep using other cells while
the UI stays up.

## Option 1: interactive shakedown, dashboard live (recommended)

Half an hour of quota to watch the whole pipeline work end to end. Do this once
before your real run.

Same notebook settings as above. Three cells.

**Cell 1**, note the much smaller token count:

```python
!pip install -q tiktoken datasets gradio
!git clone -q https://github.com/evanwang810/llm67m /kaggle/working/code
%cd /kaggle/working/code
!python tokenize_fineweb.py --out-dir /kaggle/working/tokens --max-tokens 2e8
```

**Cell 2**, training as a background process so the next cell can run:

```python
import os, subprocess, sys, time

os.makedirs("/kaggle/working/run", exist_ok=True)
log = open("/kaggle/working/run/train.log", "w")
proc = subprocess.Popen(
    [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "train.py",
     "--preset", "1session",
     "--data-dir", "/kaggle/working/tokens",
     "--run-dir", "/kaggle/working/run",
     "--deadline-hours", "0.5",
     "--log-every", "10", "--save-every-min", "5", "--milestone-every-min", "10"],
    cwd="/kaggle/working/code",
    stdout=log, stderr=subprocess.STDOUT,
    env=dict(os.environ, PYTHONUNBUFFERED="1"),
)
print("training pid", proc.pid)
time.sleep(120)
!tail -n 20 /kaggle/working/run/train.log
```

Wait for that to print actual loss lines before continuing. If it printed a
traceback instead, fix that first.

**Cell 3**, the dashboard:

```python
import sys; sys.path.insert(0, "/kaggle/working/code")
import dashboard
dashboard.launch("/kaggle/working/run", share=True)
```

It prints something like `Running on public URL: https://a1b2c3.gradio.live`.
Click it. That is your frontend, in a normal browser tab, live.

Things to know:

- **This cell blocks, on purpose.** The square stop button next to the cell kills
  the UI. Training keeps going, it is a separate process.
- The **live** tab now works for real: progress bars advance, the loss plot
  refreshes every 5 seconds, and the **save checkpoint now** and **start lr
  decay** buttons actually reach the trainer through flag files in the run dir.
- The **completion** tab works as soon as the first checkpoint exists, about five
  minutes in with these settings. At 200M tokens and half an hour the output will
  be near gibberish. That is expected. You are testing the plumbing, not the
  model.
- The share link is valid for 72 hours but dies with the session.

When you are satisfied, stop the session (**...** menu at the top, or just close
it), then go do the real committed run from step 5.

## Option 2: watch the committed run's log

No graphs, zero extra effort, fully reliable. During the eleven hours: open the
notebook, **Versions** tab, click the running version. Kaggle streams stdout.
You get a loss line every 20 steps and a `saved ckpt_...` line every 30 minutes.

This is what you will actually use most of the time.

## Option 3: dashboard in a second CPU notebook, after the run

Full frontend, in the browser, no local install, no GPU quota. Best option if you
would rather not download 1 GB or install torch on your laptop.

1. **Create -> Notebook**. **Accelerator: None**. **Internet: On**.
2. **+ Add Input**, the **Notebook Output** tab, pick your finished `llm67m-run`.
3. One cell:

```python
!pip install -q tiktoken gradio matplotlib
!git clone -q https://github.com/evanwang810/llm67m /kaggle/working/code
import sys; sys.path.insert(0, "/kaggle/working/code")
import dashboard
dashboard.launch("/kaggle/input/llm67m-run/run", share=True)
```

4. Run the cell, click the `gradio.live` link.

Check the path in the file browser on the right, the mount point sometimes has a
different name. The checkpoint dropdown also scans all of `/kaggle/input`
automatically, so if the path is slightly off you will probably still see your
checkpoints listed.

The **live** tab will show the final frozen status and its buttons do nothing,
since no trainer is listening. Everything else works: all your milestones in the
dropdown, completions, per-token probabilities, and the full loss curve.

## Option 4: your own laptop

```bash
git clone https://github.com/evanwang810/llm67m
cd llm67m
pip install torch numpy tiktoken gradio matplotlib
python dashboard.py --run-dir path\to\downloaded\run
```

Here `localhost:7860` does work, and generation is probably faster than Kaggle's
CPU.

---

## Why this size and not 67M or 125M

At one session of dual-T4 compute, roughly 4e17 FLOPs, the compute-optimal model
is about 58M total parameters trained on about 1.2B tokens. `--preset 1session`
is built to hit that. A 125M model on the same eleven hours would see half as
many tokens and come out worse, not better.

The bigger presets are for the multi-week version of this project, where you
stack six or ten sessions and the token budget grows to 6B or 8B. If you catch the
bug and want to do that, the resume machinery is all there:
[QUICKSTART.md](QUICKSTART.md).

## If it fails

| what you see | fix |
| --- | --- |
| accelerator/internet toggles greyed out | phone verification, step 1 |
| hangs right after the parameter report | put `NCCL_P2P_DISABLE=1` before `torchrun` in cell 2 |
| `CUDA out of memory` | add `--micro-batch 4 --grad-accum 16` to cell 2 |
| `no meta.json in /kaggle/working/tokens` | cell 1 did not finish. Check its output for a download error |
| `ModuleNotFoundError: tiktoken` | Internet was Off when it ran |
| `fatal: could not read from remote repository` | Internet was Off, or the repo is private |
| no `gradio.live` link printed | you forgot `share=True`, or Internet is Off |
| session died around 9 hours | Kaggle was busy. Lower `--deadline-hours` to 8 and rerun |
