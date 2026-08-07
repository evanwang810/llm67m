#!/usr/bin/env python
"""Terminal chat against any checkpoint. Runs anywhere, CPU by default.

    python chat.py                          # menu: pick a model and talk to it
    python chat.py --model run/sft_step0001183.pt
    python chat.py --dir path/to/run        # search somewhere specific

It works out whether the checkpoint is instruction-tuned and behaves
accordingly: an sft model gets real <|user|> / <|assistant|> turns and answers
you, a base model continues whatever text you type, because that is all it knows
how to do.

Commands inside the session:

    /menu             settings, model switching, session stats
    /model            pick a different checkpoint
    /temp 0.8         sampling temperature
    /topk 40          top-k cutoff, 0 turns it off
    /tokens 128       max new tokens per reply
    /greedy           toggle argmax sampling
    /probs            toggle the per-token probability table
    /stats            what this session has generated so far
    /reset            clear the conversation
    /help  /quit
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

import chatui as ui
from model import load_model_from_checkpoint
from runstate import default_search_dirs, find_checkpoints

USER_TOKEN = 50257
ASSISTANT_TOKEN = 50258

KIND_LABEL = {
    "sft": ("instruction tuned", "answers questions"),
    "milestone": ("milestone", "permanent snapshot, base model"),
    "full": ("full checkpoint", "includes optimizer state, base model"),
    "weights": ("rolling weights", "base model"),
}


def no_color() -> None:
    """Kept for callers that drive this module non-interactively."""
    ui.off()


@dataclass
class Settings:
    tokens: int = 128
    temp: float = 0.8
    topk: int = 40
    greedy: bool = False
    probs: bool = False

    def summary(self) -> str:
        mode = "greedy" if self.greedy else f"temp {self.temp:g} · top-k {self.topk}"
        return f"{mode} · max {self.tokens} tok"


@dataclass
class Stats:
    replies: int = 0
    tokens: int = 0
    seconds: float = 0.0

    @property
    def rate(self) -> float:
        return self.tokens / self.seconds if self.seconds else 0.0


# --------------------------------------------------------------------------- #
# picking a checkpoint
# --------------------------------------------------------------------------- #


def model_search_dirs() -> list[Path]:
    """Where to look when nobody said. Ordered nearest-first.

    The tool is meant to be launched from anywhere, so the script's own folder
    counts as much as the working directory, and LLM67M_MODELS lets you point at
    wherever the downloads actually land without typing --dir every time.
    """
    here = Path(__file__).resolve().parent
    dirs = [Path.cwd(), Path.cwd() / "run", here, here / "run"]
    env = os.environ.get("LLM67M_MODELS", "")
    dirs += [Path(p) for p in env.split(os.pathsep) if p.strip()]
    dirs += [Path.home() / ".llm67m", Path.home() / "Downloads"]
    dirs += list(default_search_dirs())
    seen, out = set(), []
    for d in dirs:
        key = str(d).lower()
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def checkpoint_menu(search_dirs) -> Path | None:
    cands = find_checkpoints(search_dirs)
    if not cands:
        print(ui.bad("\nno checkpoints found") + " under:")
        for d in search_dirs:
            print(f"  {ui.dim(str(d))}")
        print("\nPass --model with a path, or --dir with the folder holding your .pt files.")
        return None

    now = time.time()
    options = []
    for c in cands:
        name, note = KIND_LABEL.get(c["kind"], (c["kind"], ""))
        try:
            age = ui.human_age(now - c["path"].stat().st_mtime)
        except OSError:
            age = "?"
        tag = ui.good(name) if c["kind"] == "sft" else ui.dim(name)
        label = f"{tag}  step {c['step']:,}"
        options.append((label, f"{c['size_mb']:.0f} MB · {age} · {note}"))

    idx = ui.choose("checkpoints found", options, allow_back=True)
    return None if idx is None else cands[idx]["path"]


# --------------------------------------------------------------------------- #
# session
# --------------------------------------------------------------------------- #


class Session:
    def __init__(self, path: Path, device: str) -> None:
        print(ui.dim(f"loading {path.name} ..."), flush=True)
        t0 = time.time()
        self.model, ckpt = load_model_from_checkpoint(path, device=device)
        self.load_s = time.time() - t0
        self.device = torch.device(device)
        self.path = path
        self.sft = bool(ckpt.get("sft"))
        self.user_token = ckpt.get("user_token", USER_TOKEN)
        self.assistant_token = ckpt.get("assistant_token", ASSISTANT_TOKEN)
        self.step = ckpt.get("step", 0)
        self.params = sum(p.numel() for p in self.model.parameters())
        import tiktoken

        self.enc = tiktoken.get_encoding("gpt2")
        self.history: list[tuple[str, str]] = []
        self.stats = Stats()
        self.last_ctx = 0

    def header(self, settings: Settings) -> str:
        kind = (ui.good("instruction tuned") if self.sft
                else ui.warn("base model, continues your text"))
        rows = [
            ("model", f"{ui.bold(self.path.name)}  {kind}"),
            ("size", f"{self.params / 1e6:.1f}M params · ctx {self.model.cfg.block_size} · "
                     f"trained to step {self.step:,}"),
            ("sampling", ui.dim(settings.summary())),
        ]
        return ui.panel("llm67m chat", rows)

    def build_prompt(self, message: str) -> list[int]:
        if self.sft:
            ids: list[int] = []
            for role, content in self.history:
                ids += [self.user_token if role == "user" else self.assistant_token]
                ids += self.enc.encode_ordinary(content)
            ids += [self.user_token] + self.enc.encode_ordinary(message)
            ids += [self.assistant_token]
            return ids
        # A base model has never seen a turn structure, so just hand it the text.
        text = "".join(c for _, c in self.history) + message
        return self.enc.encode_ordinary(text) or [self.enc.eot_token]

    @torch.no_grad()
    def reply(self, message: str, max_tokens: int, temperature: float,
              top_k: int, greedy: bool, show_probs: bool,
              stream: bool = True) -> str:
        ids = self.build_prompt(message)[-(self.model.cfg.block_size - 1):]
        self.last_ctx = len(ids)
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        out_parts: list[str] = []
        rows = []
        streamer = ui.Streamer(indent=2, code=ui.BOT_CODE) if stream else None

        t0 = time.time()
        first_token_s = None
        for _ in range(max_tokens):
            window = idx[:, -self.model.cfg.block_size:]
            logits, _ = self.model(window)
            logits = logits[0, -1].float()
            probs_full = F.softmax(logits, dim=-1)

            if greedy:
                nxt = int(logits.argmax())
            else:
                scaled = logits / max(1e-4, temperature)
                if top_k > 0:
                    kth = torch.topk(scaled, min(top_k, scaled.numel()))[0][-1]
                    scaled = scaled.masked_fill(scaled < kth, float("-inf"))
                nxt = int(torch.multinomial(F.softmax(scaled, dim=-1), 1))

            if first_token_s is None:
                first_token_s = time.time() - t0
            if nxt == self.enc.eot_token or nxt >= self.enc.n_vocab:
                break
            piece = self.enc.decode([nxt])
            out_parts.append(piece)
            if streamer:
                streamer.feed(piece)
            if show_probs:
                top_p, top_i = probs_full.topk(5)
                rows.append((piece, float(probs_full[nxt]),
                             [(self.enc.decode([int(t)]), float(p))
                              for p, t in zip(top_p, top_i)]))
            idx = torch.cat([idx, torch.tensor([[nxt]], device=self.device)], dim=1)

        elapsed = time.time() - t0
        if streamer:
            streamer.done()
        n = len(out_parts)
        self.stats.replies += 1
        self.stats.tokens += n
        self.stats.seconds += elapsed

        if stream:
            rate = n / elapsed if elapsed else 0.0
            ctx_pct = 100 * (self.last_ctx + n) / self.model.cfg.block_size
            print(ui.faint(
                f"  {n} tok · {elapsed:.1f}s · {rate:.1f} tok/s · "
                f"first {first_token_s or 0:.2f}s · "
                f"ctx {self.last_ctx + n}/{self.model.cfg.block_size} ({ctx_pct:.0f}%)"))
        if show_probs and rows:
            print()
            print(ui.dim(f"  {'token':<14}{'p':>7}   top 5"))
            for piece, p, top5 in rows[:40]:
                bar = ui.G["bar"] * max(1, int(p * 12))
                alts = "  ".join(f"{t!r}={q:.2f}" for t, q in top5)
                print(f"  {piece!r:<14}{p:>7.3f} {ui.bot(bar):<14} {ui.faint(alts)}")

        reply = "".join(out_parts)
        self.history.append(("user", message))
        self.history.append(("assistant", reply))
        return reply


# --------------------------------------------------------------------------- #
# menus
# --------------------------------------------------------------------------- #


def settings_menu(s: Settings) -> None:
    while True:
        idx = ui.choose("settings", [
            (f"temperature      {ui.accent(f'{s.temp:g}')}", "higher is more random"),
            (f"top-k            {ui.accent(str(s.topk))}", "0 disables the cutoff"),
            (f"max new tokens   {ui.accent(str(s.tokens))}", "per reply"),
            (f"greedy           {ui.accent('on' if s.greedy else 'off')}",
             "always take the likeliest token"),
            (f"probability table {ui.accent('on' if s.probs else 'off')}",
             "show the top 5 per token"),
        ])
        if idx is None:
            return
        if idx == 0:
            s.temp = ui.ask("  temperature", s.temp, float)
        elif idx == 1:
            s.topk = ui.ask("  top-k", s.topk, int)
        elif idx == 2:
            s.tokens = ui.ask("  max new tokens", s.tokens, int)
        elif idx == 3:
            s.greedy = not s.greedy
        elif idx == 4:
            s.probs = not s.probs


def show_stats(session: Session) -> None:
    st = session.stats
    print()
    print(ui.panel("session", [
        ("replies", str(st.replies)),
        ("tokens", f"{st.tokens:,}"),
        ("generating", f"{st.seconds:.1f}s"),
        ("average", f"{st.rate:.1f} tok/s"),
        ("model load", f"{session.load_s:.1f}s"),
        ("turns held", str(len(session.history) // 2)),
    ]))


def session_menu(session: Session, settings: Settings, search) -> Session | None:
    """Returns a replacement Session, or the same one, or None to quit."""
    while True:
        idx = ui.choose("menu", [
            ("back to the chat", ""),
            ("settings", settings.summary()),
            ("switch model", ui.dim(session.path.name)),
            ("session stats", f"{session.stats.replies} replies"),
            ("clear the conversation", f"{len(session.history) // 2} turns held"),
            ("quit", ""),
        ], allow_back=False)
        if idx is None or idx == 0:
            return session
        if idx == 1:
            settings_menu(settings)
        elif idx == 2:
            new = checkpoint_menu(search)
            if new:
                return Session(new, str(session.device))
        elif idx == 3:
            show_stats(session)
        elif idx == 4:
            session.history.clear()
            print(ui.good("  conversation cleared"))
        elif idx == 5:
            return None


# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="", help="path to a .pt checkpoint")
    p.add_argument("--dir", default="", help="folder to search for checkpoints")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--topk", type=int, default=40)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--probs", action="store_true", help="show per-token probabilities")
    p.add_argument("--no-color", action="store_true")
    args = p.parse_args()

    ui.detect()
    if args.no_color:
        ui.off()

    search = [Path(args.dir)] if args.dir else model_search_dirs()
    print(ui.banner())

    path = Path(args.model) if args.model else checkpoint_menu(search)
    if path is None:
        return
    if not path.exists():
        raise SystemExit(f"no such file: {path}")

    session = Session(path, args.device)
    settings = Settings(args.tokens, args.temp, args.topk, args.greedy, args.probs)
    print()
    print(session.header(settings))
    print(ui.dim("  /menu for settings, /help for commands, /quit to leave"))

    while True:
        try:
            line = input(f"\n{ui.you('you')} {ui.faint(ui.G['arrow'])} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue

        if line.startswith("/"):
            parts = line.split()
            cmd, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else "")
            if cmd in ("/quit", "/q", "/exit"):
                return
            if cmd == "/menu":
                nxt = session_menu(session, settings, search)
                if nxt is None:
                    return
                if nxt is not session:
                    session = nxt
                    print()
                    print(session.header(settings))
            elif cmd == "/help":
                print(ui.dim(__doc__.split("Commands inside the session:")[1]))
            elif cmd == "/stats":
                show_stats(session)
            elif cmd == "/reset":
                session.history.clear()
                print(ui.good("  conversation cleared"))
            elif cmd == "/model":
                new = checkpoint_menu(search)
                if new:
                    session = Session(new, args.device)
                    print()
                    print(session.header(settings))
            elif cmd == "/greedy":
                settings.greedy = not settings.greedy
                print(ui.dim(f"  greedy = {settings.greedy}"))
            elif cmd == "/probs":
                settings.probs = not settings.probs
                print(ui.dim(f"  probs = {settings.probs}"))
            elif cmd in ("/temp", "/topk", "/tokens"):
                key = cmd[1:]
                try:
                    value = float(arg) if key == "temp" else int(arg)
                    setattr(settings, key, value)
                    print(ui.dim(f"  {key} = {value}"))
                except ValueError:
                    now = getattr(settings, key)
                    print(ui.bad(f"  usage: {cmd} <number>   (currently {now})"))
            else:
                print(ui.bad(f"  unknown command {cmd}, try /help"))
            continue

        label = "MODEL" if session.sft else "CONT."
        print(ui.chip(label, ui.BOT_CODE_N))
        try:
            session.reply(line, settings.tokens, settings.temp, settings.topk,
                          settings.greedy, settings.probs)
        except KeyboardInterrupt:
            print(ui.dim("\n  stopped"))


if __name__ == "__main__":
    main()
