#!/usr/bin/env python
"""Terminal chat and completion against any checkpoint. Runs anywhere, CPU by default.

    python chat.py                          # pick a model from a numbered list
    python chat.py --model run/sft_step0001183.pt
    python chat.py --dir path/to/run        # search somewhere specific

It works out whether the checkpoint is instruction-tuned and behaves accordingly:
an sft model gets real <|user|> / <|assistant|> turns and answers you, a base
model continues whatever text you type, because that is all it knows how to do.

Commands inside the session:

    /model            pick a different checkpoint
    /temp 0.8         sampling temperature
    /topk 40          top-k cutoff, 0 turns it off
    /tokens 128       max new tokens per reply
    /greedy           toggle argmax sampling
    /probs            toggle the per-token probability table
    /reset            clear the conversation
    /help  /quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from model import load_model_from_checkpoint
from runstate import default_search_dirs, find_checkpoints

USER_TOKEN = 50257
ASSISTANT_TOKEN = 50258

C = {"dim": "\x1b[2m", "bold": "\x1b[1m", "cyan": "\x1b[36m", "green": "\x1b[32m",
     "yellow": "\x1b[33m", "red": "\x1b[31m", "reset": "\x1b[0m"}


def no_color() -> None:
    for k in C:
        C[k] = ""


def pick_checkpoint(search_dirs) -> Path | None:
    cands = find_checkpoints(search_dirs)
    if not cands:
        print(f"{C['red']}no checkpoints found{C['reset']} under:")
        for d in search_dirs:
            print(f"  {d}")
        print("\nPass --model with a path, or --dir with the folder holding your .pt files.")
        return None

    print(f"\n{C['bold']}checkpoints found{C['reset']}")
    for i, c in enumerate(cands, 1):
        tag = {"sft": f"{C['green']}sft, answers questions{C['reset']}",
               "full": "full, includes optimizer state",
               "milestone": "milestone snapshot",
               "weights": "rolling snapshot"}[c["kind"]]
        print(f"  {C['cyan']}{i:>2}{C['reset']}  step {c['step']:>8,}  {c['size_mb']:>5.0f} MB  "
              f"{tag}\n      {C['dim']}{c['path']}{C['reset']}")
    while True:
        raw = input(f"\npick a number (or q): ").strip()
        if raw.lower() in ("q", "quit", ""):
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(cands):
            return cands[int(raw) - 1]["path"]
        print("not a valid choice")


class Session:
    def __init__(self, path: Path, device: str) -> None:
        print(f"{C['dim']}loading {path.name}...{C['reset']}", flush=True)
        self.model, ckpt = load_model_from_checkpoint(path, device=device)
        self.device = torch.device(device)
        self.path = path
        self.sft = bool(ckpt.get("sft"))
        self.user_token = ckpt.get("user_token", USER_TOKEN)
        self.assistant_token = ckpt.get("assistant_token", ASSISTANT_TOKEN)
        self.step = ckpt.get("step", 0)
        import tiktoken

        self.enc = tiktoken.get_encoding("gpt2")
        self.history: list[tuple[str, str]] = []
        n = sum(p.numel() for p in self.model.parameters())
        mode = (f"{C['green']}instruction tuned{C['reset']}, ask it things"
                if self.sft else
                f"{C['yellow']}base model{C['reset']}, it continues your text rather than answering")
        print(f"{C['bold']}{path.name}{C['reset']}  step {self.step:,}  {n / 1e6:.1f}M params  "
              f"ctx {self.model.cfg.block_size}\n{mode}")

    def build_prompt(self, message: str) -> tuple[list[int], str]:
        if self.sft:
            ids: list[int] = []
            for role, content in self.history:
                ids += [self.user_token if role == "user" else self.assistant_token]
                ids += self.enc.encode_ordinary(content)
            ids += [self.user_token] + self.enc.encode_ordinary(message)
            ids += [self.assistant_token]
            return ids, ""
        # A base model has never seen a turn structure, so just hand it the text.
        text = "".join(c for _, c in self.history) + message
        return self.enc.encode_ordinary(text) or [self.enc.eot_token], text

    @torch.no_grad()
    def reply(self, message: str, max_tokens: int, temperature: float,
              top_k: int, greedy: bool, show_probs: bool) -> str:
        ids, _ = self.build_prompt(message)
        ids = ids[-(self.model.cfg.block_size - 1):]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        out_parts: list[str] = []
        rows = []

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

            if nxt == self.enc.eot_token or nxt >= self.enc.n_vocab:
                break
            piece = self.enc.decode([nxt])
            out_parts.append(piece)
            sys.stdout.write(piece)
            sys.stdout.flush()
            if show_probs:
                top_p, top_i = probs_full.topk(5)
                rows.append((piece, float(probs_full[nxt]),
                             [(self.enc.decode([int(t)]), float(p))
                              for p, t in zip(top_p, top_i)]))
            idx = torch.cat([idx, torch.tensor([[nxt]], device=self.device)], dim=1)

        print()
        reply = "".join(out_parts)
        if show_probs and rows:
            print(f"\n{C['dim']}{'token':<14}{'p':>7}   top 5{C['reset']}")
            for piece, p, top5 in rows[:40]:
                alts = "  ".join(f"{t!r}={q:.2f}" for t, q in top5)
                print(f"{piece!r:<14}{p:>7.3f}   {C['dim']}{alts}{C['reset']}")
        self.history.append(("user", message))
        self.history.append(("assistant", reply))
        return reply


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
    if args.no_color:
        no_color()

    search = [Path(args.dir)] if args.dir else [Path("run"), Path.cwd(), *default_search_dirs()]
    path = Path(args.model) if args.model else pick_checkpoint(search)
    if path is None:
        return
    if not path.exists():
        raise SystemExit(f"no such file: {path}")

    session = Session(path, args.device)
    cfg = {"tokens": args.tokens, "temp": args.temp, "topk": args.topk,
           "greedy": args.greedy, "probs": args.probs}
    print(f"{C['dim']}/help for commands, /quit to leave{C['reset']}")

    while True:
        try:
            line = input(f"\n{C['cyan']}you{C['reset']} ").strip()
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
            if cmd == "/help":
                print(__doc__.split("Commands inside the session:")[1])
            elif cmd == "/reset":
                session.history.clear()
                print("conversation cleared")
            elif cmd == "/model":
                new = pick_checkpoint(search)
                if new:
                    session = Session(new, args.device)
            elif cmd == "/greedy":
                cfg["greedy"] = not cfg["greedy"]
                print(f"greedy = {cfg['greedy']}")
            elif cmd == "/probs":
                cfg["probs"] = not cfg["probs"]
                print(f"probs = {cfg['probs']}")
            elif cmd in ("/temp", "/topk", "/tokens"):
                key = cmd[1:]
                try:
                    cfg[key] = float(arg) if key == "temp" else int(arg)
                    print(f"{key} = {cfg[key]}")
                except ValueError:
                    print(f"usage: {cmd} <number>   (currently {cfg[key]})")
            else:
                print(f"unknown command {cmd}, try /help")
            continue

        label = "model" if session.sft else "continues"
        print(f"{C['green']}{label}{C['reset']} ", end="", flush=True)
        try:
            session.reply(line, cfg["tokens"], cfg["temp"], cfg["topk"],
                          cfg["greedy"], cfg["probs"])
        except KeyboardInterrupt:
            print(f"\n{C['dim']}stopped{C['reset']}")


if __name__ == "__main__":
    main()
