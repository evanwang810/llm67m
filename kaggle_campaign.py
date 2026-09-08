#!/usr/bin/env python
"""Run a whole multi-session model end to end, unattended.

    python kaggle_campaign.py --user evanwang810 --sessions 3 \
        --preset medium --data evanwang810/fineweb-edu-tokens

Pushes session 1, waits for it to finish, pushes session 2 with session 1
mounted so it resumes from that checkpoint, and so on, then instruction tunes
on the last one. Between sessions it does nothing but poll, so it is safe to
leave running and come back to a finished model.

State lives in campaign.json next to this file. Killing the script and starting
it again with the same arguments picks up where it left off rather than
relaunching sessions that already ran, which matters because a session is
several hours of quota you do not get back.

Credentials are the kaggle CLI's business, not this script's: it shells out and
the CLI resolves them itself, from `kaggle auth login` or KAGGLE_API_TOKEN.

Quota is the thing that decides the calendar. A free account gets a fixed
number of TPU hours per week, so a three session model is spread over more than
one week no matter how it is launched. This script waits through that rather
than failing, but expect the wall clock to be days.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from campaign_tick import existing_kernels, status_of

STATE = Path(__file__).with_name("campaign.json")
POLL_SECONDS = 300


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
    p.add_argument("--sft-hours", type=float, default=1.0,
                   help="instruction tuning appended to the final session")
    p.add_argument("--poll", type=int, default=POLL_SECONDS)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_state(args) -> dict:
    if not STATE.exists():
        return {"preset": args.preset, "sessions": {}}
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("preset") != args.preset:
        raise SystemExit(
            f"campaign.json is for preset {state.get('preset')}, you asked for "
            f"{args.preset}. Delete it to start a new campaign.")
    return state


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def launch(args, session: int, resume: str) -> str:
    cmd = [sys.executable, str(Path(__file__).with_name("kaggle_launch.py")),
           "--user", args.user, "--session", str(session), "--preset", args.preset,
           "--hours", str(args.hours), "--tokens", args.tokens, "--device", args.device]
    if args.data:
        cmd += ["--data", args.data]
    if resume:
        cmd += ["--resume", resume]
    # Only the last session tunes: doing it earlier spends TPU hours making an
    # instruction model out of a checkpoint that is about to be trained further.
    if session == args.sessions and args.sft_hours > 0:
        cmd += ["--sft-hours", str(args.sft_hours)]
    if args.dry_run:
        cmd += ["--dry-run"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"session {session} failed to push")
    return f"{args.user}/llm67m-{args.preset}-s{session}"


def wait_for(kernel_id: str, poll: int) -> str:
    print(f"waiting on {kernel_id}, polling every {poll}s", flush=True)
    while True:
        st = status_of(kernel_id)
        stamp = time.strftime("%H:%M")
        print(f"  {stamp}  {st}", flush=True)
        if st in ("complete", "error", "cancel"):
            return st
        time.sleep(poll)


def main() -> None:
    args = parse_args()
    state = load_state(args)

    for session in range(1, args.sessions + 1):
        key = str(session)
        record = state["sessions"].get(key, {})
        if record.get("status") == "complete":
            print(f"session {session} already complete, skipping")
            continue

        resume = f"{args.user}/llm67m-{args.preset}-s{session - 1}" if session > 1 else ""
        kernel_id = record.get("id")
        if not kernel_id:
            print(f"\n=== pushing session {session} of {args.sessions} ===", flush=True)
            kernel_id = launch(args, session, resume)
            state["sessions"][key] = {"id": kernel_id, "status": "running"}
            save_state(state)
            if args.dry_run:
                continue
        else:
            print(f"\n=== session {session} already pushed as {kernel_id} ===", flush=True)

        st = wait_for(kernel_id, args.poll)
        state["sessions"][key]["status"] = st
        save_state(state)
        if st != "complete":
            raise SystemExit(
                f"session {session} ended as {st}. Pull the log with\n"
                f"  kaggle kernels output {kernel_id} -p ./out\n"
                f"Fix it, delete that entry from campaign.json, and rerun this.")

    print("\nall sessions complete. Pull the final model with")
    last = state["sessions"][str(args.sessions)]["id"]
    print(f"  kaggle kernels output {last} -p ./final")


if __name__ == "__main__":
    main()
