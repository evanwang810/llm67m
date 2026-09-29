#!/usr/bin/env python
"""Advance a multi-session campaign by one step, then exit.

    python campaign_tick.py --user ewang330 --name llm67m-medium-r2 --sessions 10 \\
        --preset medium --seed-from ewang330/llm67m-medium-s1

Where kaggle_campaign.py sits in a polling loop and needs your machine on, this
does one pass and quits, so it can run from cron or a GitHub Actions schedule
with nothing of yours turned on.

It keeps no state. Every decision comes from asking Kaggle what the campaign's
kernels are doing right now, so a missed tick, a double tick, or two of them
racing all land on the same answer. Per tick it does at most one thing:

    tokenize the corpus if that has not happened,
    otherwise find the first session with no completed attempt, and
        leave it alone if an attempt is queued or running,
        wait if another session would go over the weekly TPU quota,
        otherwise push its next attempt.

A failed session is retried a limited number of times, each attempt a new
kernel that mounts the earlier ones, so whatever checkpoint a crashed attempt
got to is where the retry resumes. After that it stops and says why, because a
failure that repeats is a bug, and retrying a bug spends quota on nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

MISSING = "missing"
WEEK = timedelta(days=7)


def quota_week_start(now: datetime, weekday: int) -> datetime:
    """The most recent quota reset at or before now, at 00:00 UTC.

    Kaggle resets the weekly allowance at a fixed time, Saturday 00:00 UTC, not
    on a rolling seven days. Pacing against a rolling window held a session
    back for most of a week while the whole allowance sat unused.
    """
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight - timedelta(days=(now.weekday() - weekday) % 7)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user", required=True)
    p.add_argument("--name", default="", help="campaign name, default llm67m-<preset>")
    p.add_argument("--sessions", type=int, default=3)
    p.add_argument("--preset", default="medium")
    p.add_argument("--hours", type=float, default=8.5)
    p.add_argument("--tokens", default="7.5e9")
    p.add_argument("--data", default="")
    p.add_argument("--device", default="tpu")
    p.add_argument("--sft-hours", type=float, default=1.0)
    p.add_argument("--seed-from", default="",
                   help="kernel whose checkpoint session 1 continues from, user/slug")
    p.add_argument("--retries", type=int, default=2, help="extra attempts per session")
    p.add_argument("--tpu-quota-hours", type=float, default=20.0,
                   help="weekly TPU allowance; sessions are paced to stay under it")
    p.add_argument("--preview-after", type=int, default=0,
                   help="after this session, tune a chat preview of its checkpoint; 0 is off")
    p.add_argument("--preview-hours", type=float, default=0.5)
    p.add_argument("--quota-reset-weekday", type=int, default=5,
                   help="day Kaggle resets the quota, Monday=0; Kaggle uses Saturday")
    p.add_argument("--decay-fraction", type=float, default=0.75,
                   help="share of the final session spent decaying the LR")
    p.add_argument("--no-tokenize", action="store_true",
                   help="skip the tokenize kernel, the corpus already exists")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    args.name = args.name or f"llm67m-{args.preset}"
    return args


def kaggle(*cmd: str) -> tuple[int, str]:
    # Kernel logs carry whatever the model printed, and a Windows console
    # cannot encode most of it; the CLI dies mid-print rather than replacing.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    try:
        r = subprocess.run(["kaggle", *cmd], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=env)
    except FileNotFoundError:
        raise SystemExit("the kaggle CLI is not installed. pip install kaggle")
    return r.returncode, ((r.stdout or "") + (r.stderr or "")).strip()


def existing_kernels(user: str) -> dict[str, datetime | None]:
    """Every kernel ref this account owns, with when it last ran.

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
            return {}
        raise SystemExit("unexpected output from kernels list:\n" + out)
    found: dict[str, datetime | None] = {}
    for row in rows:
        ref = row.get("ref") or ""
        if not ref:
            continue
        ref = ref if "/" in ref else f"{user}/{ref}"
        when = None
        if row.get("lastRunTime"):
            try:
                when = datetime.fromisoformat(row["lastRunTime"]).replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        found[ref] = when
    return found


def status_of(kernel_id: str) -> str:
    """running, queued, complete, error or cancel, for a kernel known to exist."""
    _, out = kaggle("kernels", "status", kernel_id)
    low = out.lower()
    # NEW_SCRIPT is what a kernel reports in the minutes after a push, before
    # it is queued. Treating it as unknown stopped the campaign right there.
    if "new_script" in low:
        return "queued"
    for word in ("complete", "error", "cancel", "running", "queued"):
        if word in low:
            return word
    return "unknown"


def log_entries(kernel_id: str) -> list[dict]:
    """A finished kernel's log as parsed entries, or [] if it cannot be read."""
    _, out = kaggle("kernels", "logs", kernel_id)
    entries = []
    for line in out.splitlines():
        line = line.strip().lstrip("[,").rstrip("]")
        if not line.startswith("{"):
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def hours_used(kernel_id: str, status: str, budget: float) -> float:
    """TPU hours a session attempt consumed, read from its log where possible.

    A running attempt, or one whose log will not parse, is charged its whole
    budget. Overcounting only makes the next session wait longer; undercounting
    is what runs into the quota wall halfway through a session.
    """
    if status in ("running", "queued"):
        return budget
    times = [e.get("time", 0) for e in log_entries(kernel_id)]
    return max(times) / 3600 if times else budget


def tail_log(kernel_id: str, lines: int = 25) -> str:
    text = "".join(e.get("data", "") for e in log_entries(kernel_id))
    return "\n".join(text.splitlines()[-lines:])


def launcher() -> list[str]:
    return [sys.executable, str(Path(__file__).with_name("kaggle_launch.py"))]


def run(cmd: list[str]) -> int:
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    return r.returncode


def tokens_slug(args) -> str:
    return f"llm67m-tokens-{args.tokens}".replace(".", "-")


def attempt_ids(args, session: int) -> list[str]:
    base = f"{args.user}/{args.name}-s{session}"
    return [base] + [f"{base}-a{k}" for k in range(2, args.retries + 2)]


def push_session(args, session: int, slug: str, mounts: list[str]) -> int:
    cmd = launcher() + [
        "--user", args.user, "--slug", slug, "--session", str(session),
        "--preset", args.preset, "--hours", str(args.hours), "--tokens", args.tokens,
        "--device", args.device]
    if args.data:
        cmd += ["--data", args.data]
    for m in mounts:
        cmd += ["--mount", m]
    final = session == args.sessions
    # Only the final session decays. Decaying at the end of an earlier one marks
    # the run finished, and every later session resumes a model that is already
    # done and trains nothing, which is how two sessions got spent before.
    if not final:
        cmd += ["--no-decay"]
    else:
        cmd += ["--decay-fraction", str(args.decay_fraction)]
        if args.sft_hours > 0:
            cmd += ["--sft-hours", str(args.sft_hours)]
    if session == 1 and args.seed_from:
        cmd += ["--reset-decay"]
    if args.dry_run:
        cmd += ["--dry-run"]
    return run(cmd)


def main() -> None:
    args = parse_args()
    have = existing_kernels(args.user)
    print(f"{len(have)} kernels on the account; campaign {args.name}, "
          f"{args.sessions} sessions of {args.preset}", flush=True)

    # The corpus is tokenized once, in its own CPU kernel, and every training
    # session mounts it. Doing it inside session one instead would redo the work
    # each session, or carry 15GB of shards through the output of every one.
    tokens_id = f"{args.user}/{tokens_slug(args)}"
    if not args.no_tokenize:
        if tokens_id not in have:
            print(f"pushing the tokenize kernel {tokens_id}", flush=True)
            cmd = launcher() + ["--user", args.user, "--mode", "tokenize",
                                "--tokens", args.tokens]
            raise SystemExit(run(cmd + (["--dry-run"] if args.dry_run else [])))
        st = status_of(tokens_id)
        print(f"tokens: {tokens_id} is {st}", flush=True)
        if st in ("running", "queued"):
            print("corpus still tokenizing, nothing to do this tick")
            return
        if st != "complete":
            raise SystemExit(f"the tokenize kernel ended as {st}, campaign stuck")

    # Session 1 continues from the seed's checkpoint, so the seed has to have
    # finished. This is also what gates a campaign on a verification run: until
    # that kernel completes the campaign waits, and if it fails, it stops.
    if args.seed_from:
        if args.seed_from not in have:
            raise SystemExit(f"seed kernel {args.seed_from} does not exist")
        st = status_of(args.seed_from)
        if st in ("running", "queued"):
            print(f"seed {args.seed_from} is {st}, waiting for it")
            return
        if st != "complete":
            raise SystemExit(f"seed {args.seed_from} ended as {st}, not starting on it")

    now = datetime.now(timezone.utc)
    week_start = quota_week_start(now, args.quota_reset_weekday)
    previous = args.seed_from

    def charge(kernel: str, state: str) -> float:
        """TPU hours a kernel spent since this week's reset.

        lastRunTime is when it started running, so a kernel that began before
        the reset and ran across it only counts for the part after, which is
        how Kaggle charges it.
        """
        start = have.get(kernel)
        if not start or start + timedelta(hours=args.hours + 1) < week_start:
            return 0.0
        end = start + timedelta(hours=hours_used(kernel, state, args.hours))
        return max(0.0, (end - max(start, week_start)).total_seconds() / 3600)

    preview_id = (f"{args.user}/{args.name}-preview-s{args.preview_after}"
                  if args.preview_after else "")
    preview_state = status_of(preview_id) if preview_id in have else MISSING
    window_used = charge(preview_id, preview_state) if preview_state != MISSING else 0.0

    for session in range(1, args.sessions + 1):
        attempts = [a for a in attempt_ids(args, session) if a in have]
        states = {a: status_of(a) for a in attempts}
        done = next((a for a in attempts if states[a] == "complete"), None)

        for a in attempts:
            window_used += charge(a, states[a])

        if done:
            print(f"session {session}: complete as {done}")
            previous = done
            continue

        latest = attempts[-1] if attempts else None
        state = states.get(latest, MISSING)
        print(f"session {session}: {len(attempts)} attempt(s), latest "
              f"{latest or '-'} is {state}", flush=True)

        if state in ("running", "queued"):
            print("still going, nothing to do this tick")
            return
        if state == "unknown":
            raise SystemExit(f"unrecognised status for {latest}, not guessing")
        if state in ("error", "cancel"):
            print(f"--- last lines of {latest} ---\n{tail_log(latest)}\n---", flush=True)
            if len(attempts) > args.retries:
                raise SystemExit(
                    f"session {session} failed {len(attempts)} times; that is a bug, "
                    f"not bad luck. Fix it, delete the failed kernels, and the next "
                    f"tick starts the session again.")

        # A chat preview of a mid-training checkpoint. It goes after session N
        # because the next session usually waits on the weekly reset there
        # anyway, so it costs the campaign nothing; Kaggle allows one batch TPU
        # session at a time, so while it runs the campaign waits. A preview that
        # fails does not hold the campaign up.
        if preview_id and session == args.preview_after + 1 and not attempts:
            if preview_state in ("running", "queued"):
                print(f"preview {preview_id} is {preview_state} and holds the TPU slot")
                return
            if (preview_state == MISSING
                    and window_used + args.preview_hours <= args.tpu_quota_hours):
                print(f"pushing the SFT preview {preview_id}, tuned from {previous}",
                      flush=True)
                cmd = launcher() + [
                    "--user", args.user, "--mode", "sft",
                    "--slug", preview_id.split("/", 1)[1], "--preset", args.preset,
                    "--device", args.device, "--sft-hours", str(args.preview_hours),
                    "--mount", previous]
                raise SystemExit(run(cmd + (["--dry-run"] if args.dry_run else [])))

        if args.device == "tpu" and window_used + args.hours > args.tpu_quota_hours:
            print(f"waiting on quota: {window_used:.1f}h of TPU used since the reset on "
                  f"{week_start:%a %b %d}, another {args.hours}h session would pass "
                  f"{args.tpu_quota_hours}h")
            return

        slug = attempt_ids(args, session)[len(attempts)].split("/", 1)[1]
        # Everything that could hold the newest checkpoint: the previous session,
        # and this session's own failed attempts if there were any. The trainer
        # resumes from whichever mounted checkpoint has the highest step.
        mounts = [m for m in [previous, *attempts] if m]
        if not args.no_tokenize:
            mounts.append(tokens_id)
        print(f"pushing session {session} as {slug}, mounting {mounts}", flush=True)
        raise SystemExit(push_session(args, session, slug, mounts))

    print("all sessions complete")


if __name__ == "__main__":
    main()
