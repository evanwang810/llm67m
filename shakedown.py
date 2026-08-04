#!/usr/bin/env python
"""Run the whole pipeline small, on the real accelerator, and bundle the evidence.

    python /kaggle/working/code/shakedown.py

Meant for an interactive session: it prints as it goes, takes about fifteen
minutes, and leaves a zip in /kaggle/working you can download and send on.

This is not the preflight. tpu_preflight.py answers one question, is this device
worth spending a session on, and it answers it in ten minutes using synthetic
tensors. The shakedown answers a different one: does the entire pipeline work
here, end to end, on the real hardware, using the real data path. So it really
tokenizes FineWeb-Edu, really builds the corpus, really trains the preset you
are going to use, really saves a checkpoint, really resumes from it, really
fine-tunes, and really loads the result into the chat session.

Everything a long run does, at a scale where a mistake costs minutes.

The stage that matters most is the resume check. Kaggle sessions die, and the
entire design assumes a restart picks up where the last checkpoint left off with
no loss discontinuity. That property is invisible in a short run and expensive
to discover is broken in a long one, so it gets tested explicitly: train, stop,
resume, and confirm the loss either side of the boundary is continuous.

Stages keep going after a failure rather than stopping at the first one, because
a bundle showing four failures is more useful than four separate runs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BAR = "=" * 72

# This process must never initialise the XLA runtime. A TPU is claimed by one
# process at a time, so a single call to xr.global_runtime_device_count() here
# takes the device and every stage that spawns a trainer then dies with
# "Check failed: reporting_closure_ == nullptr", which reads like a bug in the
# trainer and is not. All XLA facts are gathered in throwaway subprocesses that
# claim the device, answer, and exit. These pops are belt and braces for
# anything that imports torch_xla despite that.
for _var in ("TPU_PROCESS_ADDRESSES", "CLOUD_TPU_TASK_ID"):
    os.environ.pop(_var, None)

def probe_xla() -> dict:
    """Version and device facts, gathered without holding the device here."""
    from xla_probe import probe

    return probe(cwd=str(HERE))


class Bundle:
    """Tees everything to the console and to a log file, and collects facts."""

    def __init__(self, outdir: Path) -> None:
        outdir.mkdir(parents=True, exist_ok=True)
        self.dir = outdir
        self.logfile = outdir / "shakedown.log"
        self.fh = self.logfile.open("w", encoding="utf-8", errors="replace")
        self.stages: list[dict] = []
        self.facts: dict = {}
        self.t0 = time.time()

    def say(self, line: str = "") -> None:
        enc = sys.stdout.encoding or "utf-8"
        safe = line.encode(enc, errors="replace").decode(enc, errors="replace")
        print(safe, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()

    def run(self, cmd: list[str], timeout: float = 1800) -> tuple[int, str]:
        """Run a real command, streaming its output into both sinks."""
        self.say(f"$ {' '.join(str(c) for c in cmd)}")
        lines: list[str] = []
        try:
            p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace", bufsize=1)
        except OSError as e:
            self.say(f"  could not start: {e}")
            return 127, str(e)
        # A timer, not a check inside the read loop. A process that hangs with no
        # output never yields another line, so an in-loop deadline never fires and
        # the shakedown would sit there until the session died, which is the exact
        # failure it exists to catch quickly.
        import threading

        killed = threading.Event()

        def reap():
            killed.set()
            with contextlib.suppress(Exception):
                p.kill()

        timer = threading.Timer(timeout, reap)
        timer.start()
        try:
            for line in p.stdout:
                lines.append(line.rstrip("\n"))
                self.say("  " + line.rstrip("\n"))
            p.wait()
        finally:
            timer.cancel()
        if killed.is_set():
            self.say(f"  TIMEOUT: killed after {timeout:.0f}s with no exit")
            return 124, "\n".join(lines)
        return p.returncode, "\n".join(lines)


def stage(b: Bundle, name: str, proves: str):
    """Decorator-ish helper: run fn, record pass/fail and duration, never raise."""
    def wrap(fn):
        b.say("")
        b.say(BAR)
        b.say(f"STAGE {len(b.stages) + 1}: {name}")
        b.say(f"  proves: {proves}")
        b.say(BAR)
        t = time.time()
        rec = {"stage": name, "proves": proves}
        try:
            out = fn() or {}
            rec.update({"ok": True, **out})
        except Exception as e:  # a broken stage must not hide the ones after it
            import traceback
            rec.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
            b.say(f"  FAILED: {type(e).__name__}: {e}")
            b.fh.write(traceback.format_exc())
        rec["seconds"] = round(time.time() - t, 1)
        b.stages.append(rec)
        b.say(f"  -> {'ok' if rec.get('ok') else 'FAILED'} in {rec['seconds']:.1f}s")
        return rec
    return wrap


def detect_device(xla: dict) -> str:
    try:
        import torch
        if torch.cuda.device_count() > 0:
            return "gpu"
    except Exception:
        pass
    return "tpu" if xla.get("xla_device_type") == "TPU" else "cpu"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="", help="default: matches the device")
    ap.add_argument("--out", default="/kaggle/working/shakedown")
    ap.add_argument("--tokens", type=float, default=3e6,
                    help="how much FineWeb-Edu to actually tokenize")
    ap.add_argument("--steps", type=int, default=30, help="steps per training leg")
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--tokens-dir", default="",
                    help="use tokens from here instead of making new ones")
    ap.add_argument("--skip-tokenize", action="store_true",
                    help="reuse tokens already on disk (offline, or a repeat run)")
    ap.add_argument("--no-install", action="store_true",
                    help="do not pip install missing packages")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    b = Bundle(out)
    work = out / "work"
    # An explicit tokens dir lives outside the output tree, which is wiped on
    # every run, so a second shakedown does not re-download the corpus.
    tokens_dir = Path(args.tokens_dir) if args.tokens_dir else work / "tokens"
    run_dir = work / "run"

    xla = probe_xla()
    device = detect_device(xla)
    preset = args.preset or {"tpu": "tpu1session", "gpu": "67m"}.get(device, "1session")
    trainer = "train_tpu.py" if device == "tpu" else "train.py"
    tuner = "finetune_tpu.py" if device == "tpu" else "finetune.py"

    b.say(BAR)
    b.say("llm67m shakedown")
    b.say(f"  device  {device}")
    b.say(f"  preset  {preset}")
    b.say(f"  trainer {trainer}")
    b.say(f"  output  {out}")
    b.say(BAR)
    b.facts.update({"device": device, "preset": preset, "trainer": trainer})

    # ---------------------------------------------------------------- 1. env
    @stage(b, "environment", "the box is what you think it is")
    def _env():
        import torch
        facts = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_devices": torch.cuda.device_count(),
        }
        facts.update(xla)  # gathered by a subprocess, see probe_xla
        if torch.cuda.device_count():
            facts["gpu_names"] = [torch.cuda.get_device_name(i)
                                  for i in range(torch.cuda.device_count())]
        du = shutil.disk_usage("/kaggle/working" if Path("/kaggle/working").exists() else ".")
        facts["disk_free_gb"] = round(du.free / 1e9, 1)
        for k, v in facts.items():
            b.say(f"  {k:18} {v}")
        if facts["disk_free_gb"] < 5:
            raise RuntimeError(f"only {facts['disk_free_gb']}GB free, a real run needs more")
        b.facts.update(facts)
        return facts

    # --------------------------------------------------------------- 2. deps
    @stage(b, "dependencies", "the packages a real run installs are importable here")
    def _deps():
        missing = []
        for mod in ("tiktoken", "datasets"):
            try:
                __import__(mod)
            except ImportError:
                missing.append(mod)
        if missing and not args.no_install:
            b.say(f"  installing {' '.join(missing)}")
            b.run([sys.executable, "-m", "pip", "install", "-q", *missing], timeout=900)
            missing = [m for m in missing
                       if subprocess.run([sys.executable, "-c", f"import {m}"],
                                         capture_output=True).returncode != 0]
        if missing:
            raise RuntimeError(f"still missing: {', '.join(missing)}")
        b.say("  tiktoken and datasets both import")
        return {}

    # ----------------------------------------------------------- 3. tokenize
    @stage(b, "tokenize", "network, HuggingFace streaming, and the shard writer all work")
    def _tok():
        if (args.skip_tokenize or args.tokens_dir) and (tokens_dir / "meta.json").exists():
            b.say(f"  reusing existing tokens at {tokens_dir}")
        else:
            code, _ = b.run([sys.executable, "tokenize_fineweb.py", "--out-dir", str(tokens_dir),
                             "--max-tokens", str(args.tokens)], timeout=1200)
            # Same reasoning as kaggle_run.sh: the streaming reader can abort
            # during shutdown after the shards are already complete, so the exit
            # code is not the authority. Ask the data.
            code, _ = b.run([sys.executable, "tokenize_fineweb.py", "--out-dir", str(tokens_dir),
                             "--max-tokens", str(args.tokens), "--verify-only"])
            if code != 0:
                raise RuntimeError("tokenizer did not produce a complete shard set")
        meta = json.loads((tokens_dir / "meta.json").read_text())
        b.say(f"  {meta.get('total_tokens', 0):,} tokens, tokenizer={meta.get('tokenizer')}")
        return {"total_tokens": meta.get("total_tokens"), "shards": len(meta["shards"]["train"])}

    # ---------------------------------------------------------------- 3. data
    @stage(b, "dataloader", "batches are the right shape and identical for the same step")
    def _data():
        sys.path.insert(0, str(HERE))
        import torch
        from data import BatchSampler, Corpus
        c = Corpus(str(tokens_dir), args.block_size, "train")
        s = BatchSampler(c, args.micro_batch, args.grad_accum, 1, 0, 1337)
        x1, y1 = s.batch(7, 0, torch.device("cpu"))
        x2, y2 = s.batch(7, 0, torch.device("cpu"))
        same = bool(torch.equal(x1, x2) and torch.equal(y1, y2))
        drifted = not torch.equal(x1, s.batch(8, 0, torch.device("cpu"))[0])
        b.say(f"  corpus {c.total_tokens:,} tokens, {c.n_blocks:,} blocks")
        b.say(f"  batch  {tuple(x1.shape)} {x1.dtype}, targets shifted by one: "
              f"{bool(torch.equal(x1[0, 1:], y1[0, :-1]))}")
        b.say(f"  step 7 reproducible: {same}   step 8 differs: {drifted}")
        if not same:
            raise RuntimeError("same step gave different batches; resume would not be clean")
        if not drifted:
            raise RuntimeError("different steps gave the same batch; the sampler is stuck")
        return {"blocks": c.n_blocks, "deterministic": same}

    # --------------------------------------------------------------- 4. model
    @stage(b, "model", "the preset really is the size config.py claims")
    def _model():
        from config import PRESETS, GPTConfig, non_embedding_params
        from model import GPT
        cfg = GPTConfig(**PRESETS[preset])
        m = GPT(cfg)
        counted = sum(p.numel() for n, p in m.named_parameters() if "wte" not in n)
        formula = non_embedding_params(cfg)
        b.say(m.param_report())
        b.say(f"  formula says {formula / 1e6:.2f}M non-embedding, "
              f"model has {counted / 1e6:.2f}M")
        if abs(counted - formula) / formula > 0.01:
            raise RuntimeError("parameter count disagrees with the formula")
        return {"non_embedding_params": formula, "total_params": sum(p.numel()
                                                                    for p in m.parameters())}

    # ------------------------------------------------------- 5+6. train/resume
    common = ["--preset", preset, "--data-dir", str(tokens_dir), "--run-dir", str(run_dir),
              "--micro-batch", str(args.micro_batch), "--grad-accum", str(args.grad_accum),
              "--block-size", str(args.block_size),
              "--log-every", "5", "--eval-every", "0", "--warmup-steps", "10",
              "--save-every-min", "0.05", "--deadline-hours", "0.5"]

    @stage(b, "train", "the real trainer runs the real preset on the real device")
    def _train():
        code, txt = b.run([sys.executable, trainer, *common,
                           "--max-steps", str(args.steps)], timeout=2400)
        if code != 0:
            raise RuntimeError(f"trainer exited {code}")
        ck = sorted(run_dir.glob("ckpt_step*.pt"))
        if not ck:
            raise RuntimeError("no checkpoint was written")
        b.say(f"  wrote {ck[-1].name}, {ck[-1].stat().st_size / 1e6:.0f} MB")
        rate = [l for l in txt.splitlines() if "tok/s" in l]
        return {"checkpoint": ck[-1].name,
                "checkpoint_mb": round(ck[-1].stat().st_size / 1e6),
                "last_log_line": rate[-1].strip() if rate else None}

    @stage(b, "resume", "a restart continues the run instead of starting over")
    def _resume():
        code, txt = b.run([sys.executable, trainer, *common,
                           "--max-steps", str(args.steps * 2)], timeout=2400)
        if code != 0:
            raise RuntimeError(f"trainer exited {code} on resume")
        if "resum" not in txt.lower():
            raise RuntimeError("second run did not resume; it started from scratch")
        code, chk = b.run([sys.executable, "check_resume.py", "--run-dir", str(run_dir)])
        return {"resume_report": chk.strip().splitlines()[-3:], "check_resume_exit": code}

    # ----------------------------------------------------------------- 7. sft
    @stage(b, "finetune", "instruction tuning runs and writes an sft checkpoint")
    def _sft():
        code, _ = b.run([sys.executable, tuner, "--run-dir", str(run_dir),
                         "--max-examples", "400", "--max-len", "256",
                         "--batch-size", "4", "--grad-accum", "2", "--epochs", "1",
                         "--log-every", "10", "--warmup", "5", "--hours", "0.15"],
                        timeout=1800)
        if code != 0:
            raise RuntimeError(f"finetune exited {code}")
        sft = sorted(run_dir.glob("sft_step*.pt"))
        if not sft:
            raise RuntimeError("no sft checkpoint was written")
        return {"sft_checkpoint": sft[-1].name}

    # ---------------------------------------------------------------- 8. chat
    @stage(b, "chat", "the checkpoint loads back and generates text")
    def _chat():
        import chat as chatmod
        chatmod.no_color()
        sft = sorted(run_dir.glob("sft_step*.pt"))
        target = sft[-1] if sft else sorted(run_dir.glob("ckpt_step*.pt"))[-1]
        s = chatmod.Session(target, "cpu")
        replies = []
        for q in ("What is the capital of France?", "Name three animals."):
            s.history.clear()
            r = s.reply(q, 24, 0.8, 40, False, False)
            replies.append({"q": q, "a": " ".join(r.split())})
            b.say(f"  Q {q}")
            b.say(f"  A {' '.join(r.split())}")
        return {"sft_flag": s.sft, "replies": replies}

    # -------------------------------------------------------------- 9. bundle
    b.say("")
    b.say(BAR)
    b.say("SUMMARY")
    b.say(BAR)
    passed = sum(1 for s in b.stages if s.get("ok"))
    for s in b.stages:
        b.say(f"  {'PASS' if s.get('ok') else 'FAIL'}  {s['stage']:<12} "
              f"{s['seconds']:>6.1f}s   {s.get('error', '')}")
    b.say("")
    b.say(f"  {passed}/{len(b.stages)} stages passed in "
          f"{(time.time() - b.t0) / 60:.1f} minutes")

    b.facts["stages"] = b.stages
    b.facts["passed"] = passed
    b.facts["total_stages"] = len(b.stages)
    b.facts["wall_minutes"] = round((time.time() - b.t0) / 60, 2)
    (out / "shakedown.json").write_text(json.dumps(b.facts, indent=2, default=str))

    # Copy the artifacts a long run would have produced, so the bundle shows the
    # loss curve and the samples rather than just this script's opinion of them.
    for name in ("loss.csv", "status.json", "samples.txt"):
        src = run_dir / name
        if src.exists():
            shutil.copy(src, out / name)

    zpath = Path("/kaggle/working/shakedown.zip")
    if not zpath.parent.exists():
        zpath = out.parent / "shakedown.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in out.iterdir():
            if f.is_file():
                z.write(f, f.name)
    b.say("")
    b.say(f"  bundle: {zpath}  ({zpath.stat().st_size / 1e3:.0f} KB)")
    b.say(f"  log:    {b.logfile}")
    b.say("")
    b.say("  Download the zip from the Kaggle file browser on the right.")
    b.fh.close()
    return 0 if passed == len(b.stages) else 1


if __name__ == "__main__":
    raise SystemExit(main())
