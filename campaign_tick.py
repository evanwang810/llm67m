#!/usr/bin/env python
"""Advance a multi-session campaign by one step, then exit.

    python campaign_tick.py --user ewang330 --sessions 3 --preset medium

Where kaggle_campaign.py sits in a polling loop and needs your machine on, this
does one pass and quits, so it can run from cron or a GitHub Actions schedule
with nothing of yours turned on.

It keeps no state. Every decision comes from asking Kaggle what the session
kernels are doing right now, which means a missed tick, a double tick, or two
of them racing all land on the same answer. The rule is only:

    the first session that is not complete is the one to care about,
    and it gets pushed only if it does not exist yet.

so a session that is queued or running is left alone rather than relaunched,
which is what stops a stray tick from spending quota twice.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

MISSING = "missing"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user", required=True)
    p.add_argument("--sessions", type=int, default=3)
    p.add_argument("--preset", default="medium")
    p.add_argument("--hours", type=float, default=8.5)
    p.add_argument("--tokens", default="7.5e9")
    p.add_argument("--data", default="")
    p.add_argument("--device", default="tpu")
    p.add_argument("--sft-hours", type=float, default=1.0)
    p.add_argument("--no-tokenize", action="store_true",
                   help="skip the tokenize kernel, the corpus already exists")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def kaggle(*cmd: str) -> tuple[int, str]:
    try:
        r = subprocess.run(["kaggle", *cmd], capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit("the kaggle CLI is not installed. pip install kaggle")
    return r.returncode, (r.stdout + r.stderr).strip()


def existing_kernels(user: str) -> set[str]:
    """Every kernel ref this account owns.

    Existence has to be read from a listing rather than probed with `kernels
    status`, because a kernel that was never pushed comes back as an HTTP 403
    carrying a permissions message, which is the identical response to a kernel
    that exists but is not readable. Reading "missing" out of that would push a
    session on top of a permissions problem and spend hours of quota doing it.
    """
    rc, out = kaggle("kernels", "list", "--mine", "--format", "json",
                     "--page-size", "200")
    if rc != 0:
        raise SystemExit("could not list your kernels, so nothing here is safe "
                         "to decide:\n" + out)
    try:
        rows = json.loads(out)
    except json.JSONDecodeError:
        # An empty account prints a human sentence rather than an empty array.
        if "no kernels" in out.lower():
            return set()
        raise SystemExit("unexpected output from kernels list:\n" + out)
    slugs = set()
    for row in rows:
        ref = row.get("ref") or ""
        if ref:
            slugs.add(ref if "/" in ref else f"{user}/{ref}")
    return slugs


def status_of(kernel_id: str) -> str:
    """running, queued, complete, error or cancel, for a kernel known to exist."""
    _, out = kaggle("kernels", "status", kernel_id)
    low = out.lower()
    for word in ("complete", "error", "cancel", "running", "queued"):
        if word in low:
            return word
    return "unknown"


def tokens_slug(args) -> str:
    return f"llm67m-tokens-{args.tokens}".replace(".", "-")


def push(args, session: int, mount: str = "") -> int:
    cmd = [sys.executable, str(Path(__file__).with_name("kaggle_launch.py")),
           "--user", args.user, "--session", str(session), "--preset", args.preset,
           "--hours", str(args.hours), "--tokens", args.tokens, "--device", args.device]
    if args.data:
        cmd += ["--data", args.data]
    if mount:
        cmd += ["--mount", mount]
    if session > 1:
        cmd += ["--resume", f"{args.user}/llm67m-{args.preset}-s{session - 1}"]
    if session == args.sessions and args.sft_hours > 0:
        cmd += ["--sft-hours", str(args.sft_hours)]
    if args.dry_run:
        cmd += ["--dry-run"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    return r.returncode


def main() -> None:
    args = parse_args()
    have = existing_kernels(args.user)
    print(f"{len(have)} kernels on the account", flush=True)

    # The corpus is tokenized once, in its own CPU kernel, and every training
    # session mounts it. Doing it inside session one instead would redo the work
    # each session, or carry 15GB of shards through the output of every one.
    tokens_id = f"{args.user}/{tokens_slug(args)}"
    if not args.no_tokenize:
        if tokens_id not in have:
            print(f"pushing the tokenize kernel {tokens_id}", flush=True)
            cmd = [sys.executable, str(Path(__file__).with_name("kaggle_launch.py")),
                   "--user", args.user, "--mode", "tokenize", "--tokens", args.tokens]
            if args.dry_run:
                cmd += ["--dry-run"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            sys.stdout.write(r.stdout)
            sys.stderr.write(r.stderr)
            raise SystemExit(r.returncode)
        st = status_of(tokens_id)
        print(f"tokens: {tokens_id} is {st}", flush=True)
        if st in ("running", "queued"):
            print("corpus still tokenizing, nothing to do this tick")
            return
        if st != "complete":
            raise SystemExit(f"the tokenize kernel ended as {st}, campaign stuck")

    for session in range(1, args.sessions + 1):
        kernel_id = f"{args.user}/llm67m-{args.preset}-s{session}"
        st = status_of(kernel_id) if kernel_id in have else MISSING
        print(f"session {session}: {kernel_id} is {st}", flush=True)

        if st == "complete":
            continue
        if st in ("running", "queued"):
            print("still going, nothing to do this tick")
            return
        if st in ("error", "cancel"):
            raise SystemExit(
                f"session {session} ended as {st} and the campaign is stuck.\n"
                f"  kaggle kernels output {kernel_id} -p ./out\n"
                "Fix the cause, delete that kernel on Kaggle, and the next tick "
                "will push it again.")
        if st == MISSING:
            print(f"pushing session {session}", flush=True)
            raise SystemExit(push(args, session,
                                  "" if args.no_tokenize else tokens_id))
        raise SystemExit(f"unrecognised status for {kernel_id}, not guessing")

    print("all sessions complete")


if __name__ == "__main__":
    main()
