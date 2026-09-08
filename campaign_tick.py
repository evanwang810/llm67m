#!/usr/bin/env python
"""Advance a multi-session campaign by one step, then exit.

    python campaign_tick.py --user evanwang810 --sessions 3 --preset medium

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
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def status_of(kernel_id: str) -> str:
    """running, queued, complete, error, cancel, or missing."""
    try:
        r = subprocess.run(["kaggle", "kernels", "status", kernel_id],
                           capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit("the kaggle CLI is not installed. pip install kaggle")
    out = (r.stdout + r.stderr).lower()
    # A kernel that was never pushed reports as not found rather than failing in
    # a way worth distinguishing, and both mean the same thing here: push it.
    if "404" in out or "not found" in out or "could not find" in out:
        return MISSING
    for word in ("complete", "error", "cancel", "running", "queued"):
        if word in out:
            return word
    return "unknown"


def push(args, session: int) -> int:
    cmd = [sys.executable, str(Path(__file__).with_name("kaggle_launch.py")),
           "--user", args.user, "--session", str(session), "--preset", args.preset,
           "--hours", str(args.hours), "--tokens", args.tokens, "--device", args.device]
    if args.data:
        cmd += ["--data", args.data]
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

    for session in range(1, args.sessions + 1):
        kernel_id = f"{args.user}/llm67m-{args.preset}-s{session}"
        st = status_of(kernel_id)
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
                f"Fix the cause, delete that kernel on Kaggle, and the next tick "
                f"will push it again.")
        if st == MISSING:
            print(f"pushing session {session}", flush=True)
            raise SystemExit(push(args, session))
        raise SystemExit(f"unrecognised status for {kernel_id}, not guessing")

    print("all sessions complete")


if __name__ == "__main__":
    main()
