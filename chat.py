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
    /history 0        previous turns to carry, 0 for a single-turn tuned model
    /rep 1.1          repetition penalty, 1 turns it off
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
import tools
from model import KVCache, load_model_from_checkpoint
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
    # How many previous exchanges to put in front of the question. Zero for an
    # Alpaca-tuned model on purpose: see Session.build_prompt.
    history: int = 0
    # Above 1, tokens the model already used in this reply or its recent answers
    # get less likely. A small model's favourite failure is copying its own
    # previous answer back out, or looping on one sentence.
    rep: float = 1.1

    def summary(self) -> str:
        mode = "greedy" if self.greedy else f"temp {self.temp:g} · top-k {self.topk}"
        hist = "no history" if self.history == 0 else f"{self.history} turns of history"
        rep = "" if self.rep == 1 else f" · rep {self.rep:g}"
        return f"{mode}{rep} · max {self.tokens} tok · {hist}"


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


def resolve_device(name: str) -> str:
    """auto picks a CUDA GPU if there is one, otherwise the CPU.

    Not an Intel integrated GPU, measured: with the cache it holds steady
    memory, but generating one token at a time is hundreds of tiny kernels per
    token, and an Arc 130V ran 2.2 to 2.9 tok/s against the same laptop's CPU
    at 6.4. Ask for xpu explicitly to use it anyway.
    """
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


class Session:
    def __init__(self, path: Path, device: str) -> None:
        device = resolve_device(device)
        print(ui.dim(f"loading {path.name} on {device} ..."), flush=True)
        t0 = time.time()
        self.model, ckpt = load_model_from_checkpoint(path, device=device)
        if device != "cpu":
            # Generation reads every weight once per token, so halving their
            # size roughly doubles the speed. A laptop CPU has no fast bf16, so
            # it keeps fp32.
            self.model = self.model.to(torch.bfloat16)
        self.dtype = next(self.model.parameters()).dtype
        self.load_s = time.time() - t0
        self.device = torch.device(device)
        self.path = path
        self.sft = bool(ckpt.get("sft"))
        self.user_token = ckpt.get("user_token", USER_TOKEN)
        self.assistant_token = ckpt.get("assistant_token", ASSISTANT_TOKEN)
        # Only a checkpoint tuned on tool conversations knows the call format;
        # for any other, a stray special token just ends the reply as before.
        self.tools = bool(ckpt.get("tools"))
        self.call_token = ckpt.get("tool_call_token", tools.TOOL_CALL)
        self.result_token = ckpt.get("tool_result_token", tools.TOOL_RESULT)
        self.step = ckpt.get("step", 0)
        # Whether carrying history helps is a property of the tuning data, not
        # of the chat tool: a model tuned only on single turn data reads a
        # previous exchange as a pattern to repeat rather than as context.
        self.sft_dataset = str(ckpt.get("sft_dataset", ""))
        self.multi_turn = self.sft and "alpaca" not in self.sft_dataset.lower() \
            and bool(self.sft_dataset)
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

    def build_prompt(self, message: str, turns: int = 0) -> list[int]:
        """Assemble the prompt, carrying only `turns` previous exchanges.

        Alpaca is single-turn data. The model has never seen one exchange
        followed by another, so a conversation in front of the question does not
        read as context, it reads as the pattern to continue: ask it four things
        in a row with history on and it answers the first one four times. Zero is
        the honest default until it is tuned on multi-turn data.
        """
        keep = self.history[-2 * turns:] if turns > 0 else []
        if self.sft:
            ids: list[int] = []
            for role, content in keep:
                ids += [self.user_token if role == "user" else self.assistant_token]
                ids += self.enc.encode_ordinary(content)
            ids += [self.user_token] + self.enc.encode_ordinary(message)
            ids += [self.assistant_token]
            return ids
        # A base model has never seen a turn structure, so just hand it the text.
        text = "".join(c for _, c in keep) + message
        return self.enc.encode_ordinary(text) or [self.enc.eot_token]

    @torch.no_grad()
    def reply(self, message: str, max_tokens: int, temperature: float,
              top_k: int, greedy: bool, show_probs: bool,
              stream: bool = True, turns: int = 0, rep: float = 1.0) -> str:
        block = self.model.cfg.block_size
        max_tokens = min(max_tokens, block // 2)
        # Leave room for the reply: with a cache the context cannot slide, so a
        # long conversation drops its oldest tokens up front instead.
        ids = self.build_prompt(message, turns)[-(block - max_tokens):]
        self.last_ctx = len(ids)
        out_parts: list[str] = []
        rows = []
        streamer = ui.Streamer(indent=2, code=ui.BOT_CODE) if stream else None

        # Tokens the repetition penalty applies to: the recent answers this
        # prompt carries, then everything this reply says as it goes. Not the
        # user's words, which a good answer is supposed to reuse.
        seen: set[int] = set()
        if rep != 1 and turns > 0:
            for role, content in self.history[-2 * turns:]:
                if role == "assistant":
                    seen.update(self.enc.encode_ordinary(content))

        cache = KVCache(self.model.cfg, 1, self.device, self.dtype)
        # The prompt goes through once, padded to a multiple of 64 on a GPU so
        # there are only a handful of prompt shapes ever. The padded tail writes
        # junk keys past the prompt; the cache mask keeps them out of view and
        # generation overwrites them in order.
        n = len(ids)
        padded = n if self.device.type == "cpu" else min(block, -(-n // 64) * 64)
        prompt = torch.zeros((1, padded), dtype=torch.long, device=self.device)
        prompt[0, :n] = torch.tensor(ids, dtype=torch.long)

        t0 = time.time()
        first_token_s = None
        logits = self.model.step(prompt, cache, 0)[0, n - 1].float()
        pos = n

        def feed(chunk: list[int]) -> torch.Tensor:
            """Run tokens through the cache, return the logits after the last."""
            nonlocal pos
            t = torch.tensor([chunk], dtype=torch.long, device=self.device)
            out = self.model.step(t, cache, pos)[0, -1].float()
            pos += len(chunk)
            return out

        call_ids: list[int] | None = None  # set while the model is writing a call
        calls = 0
        # Room for the calls on top of the answer budget, which counts only
        # what the user gets to read.
        for _ in range(max_tokens + 48 * tools.MAX_CALLS):
            if len(out_parts) >= max_tokens or pos >= block:
                break
            if seen and call_ids is None:
                hit = torch.tensor(sorted(seen), device=logits.device)
                lg = logits[hit]
                logits[hit] = torch.where(lg > 0, lg / rep, lg * rep)
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

            if self.tools and nxt == self.call_token and call_ids is None and calls < tools.MAX_CALLS:
                call_ids = []
                logits = feed([nxt])
                continue
            if call_ids is not None:
                if nxt != self.result_token and len(call_ids) < 48:
                    call_ids.append(nxt)
                    logits = feed([nxt])
                    continue
                # The model closed its call: run it, show it, and hand the
                # output back with a fresh assistant turn to answer from.
                name, arg, output = tools.run(self.enc.decode(call_ids))
                calls += 1
                call_ids = None
                if streamer and streamer.started:
                    streamer.done()
                    streamer = ui.Streamer(indent=2, code=ui.BOT_CODE)
                if stream:
                    print(ui.dim(f"  {ui.G['arrow']} {name}: {arg}"))
                    if show_probs:
                        print(ui.faint("    " + output.replace("\n", "\n    ")))
                result = self.enc.encode_ordinary(output)
                room = block - pos - max(32, max_tokens - len(out_parts)) - 2
                result = result[:max(0, room)]
                logits = feed([self.result_token] + result + [self.assistant_token])
                continue

            if nxt == self.enc.eot_token or nxt >= self.enc.n_vocab:
                break
            piece = self.enc.decode([nxt])
            out_parts.append(piece)
            if rep != 1:
                seen.add(nxt)
            if streamer:
                streamer.feed(piece)
            if show_probs:
                top_p, top_i = probs_full.topk(5)
                rows.append((piece, float(probs_full[nxt]),
                             [(self.enc.decode([int(t)]), float(p))
                              for p, t in zip(top_p, top_i)]))
            logits = feed([nxt])

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
            (f"history turns    {ui.accent(str(s.history))}",
             "0 suits a model tuned on single-turn data"),
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
        elif idx == 5:
            s.history = max(0, ui.ask("  history turns", s.history, int))


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
    p.add_argument("--device", default="auto", help="auto, cpu, cuda or xpu")
    p.add_argument("--rep", type=float, default=1.1, help="repetition penalty, 1 is off")
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--topk", type=int, default=40)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--probs", action="store_true", help="show per-token probabilities")
    p.add_argument("--history", type=int, default=-1,
                   help="previous turns to carry, -1 picks from the tuning data")
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
    settings.rep = max(1.0, args.rep)
    settings.history = (args.history if args.history >= 0
                        else (4 if session.multi_turn else 0))
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
            elif cmd == "/rep":
                try:
                    settings.rep = max(1.0, float(arg))
                    print(ui.dim(f"  repetition penalty {settings.rep:g}"))
                except ValueError:
                    print(ui.bad(f"  usage: /rep <x>   (currently {settings.rep:g}, 1 is off)"))
            elif cmd == "/history":
                try:
                    settings.history = max(0, int(arg))
                    print(ui.dim(f"  carrying {settings.history} previous turns"))
                except ValueError:
                    print(ui.bad(f"  usage: /history <n>   (currently {settings.history})"))
                    print(ui.dim("  0 is recommended for an Alpaca-tuned model"))
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
                          settings.greedy, settings.probs, turns=settings.history,
                          rep=settings.rep)
        except KeyboardInterrupt:
            print(ui.dim("\n  stopped"))


if __name__ == "__main__":
    main()
