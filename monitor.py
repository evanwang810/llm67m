#!/usr/bin/env python
"""Terminal training monitor. Everything the dashboard shows, minus the mouse.

    python monitor.py                          # one snapshot
    python monitor.py --watch                  # refresh every 30s
    python monitor.py --watch --interval 300   # refresh every 5 min
    python monitor.py --last 2000              # zoom the chart to recent steps
    python monitor.py --tail 30                # raw train.log lines instead

Same three buttons the dashboard has, as flags:

    python monitor.py --save     # checkpoint now
    python monitor.py --decay    # start the lr decay phase
    python monitor.py --stop     # final checkpoint, then exit cleanly

Works on a live run or on a finished run folder you downloaded.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from runstate import RunDir

C = {
    "reset": "\x1b[0m", "dim": "\x1b[2m", "bold": "\x1b[1m",
    "red": "\x1b[31m", "green": "\x1b[32m", "yellow": "\x1b[33m",
    "blue": "\x1b[34m", "cyan": "\x1b[36m",
}
NO_COLOR = {k: "" for k in C}


def fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def bar(frac: float, width: int = 46) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "#" * filled + "-" * (width - filled)


def ascii_plot(series: dict[str, tuple[list, list]], width: int = 72, height: int = 16) -> list[str]:
    """Bucket-average each series into `width` columns and draw them on one grid.

    series maps label -> (xs, ys, marker). Later series draw over earlier ones.
    """
    all_x = [x for xs, _, _ in series.values() for x in xs]
    all_y = [y for _, ys, _ in series.values() for y in ys]
    if not all_y:
        return ["  no data yet"]
    xlo, xhi = min(all_x), max(all_x)
    ylo, yhi = min(all_y), max(all_y)
    if yhi - ylo < 1e-9:
        yhi = ylo + 1e-9
    xspan = max(1e-9, xhi - xlo)

    grid = [[" "] * width for _ in range(height)]
    for xs, ys, marker in series.values():
        buckets: list[list[float]] = [[] for _ in range(width)]
        for x, y in zip(xs, ys):
            col = min(width - 1, int((x - xlo) / xspan * (width - 1)))
            buckets[col].append(y)
        for col, vals in enumerate(buckets):
            if not vals:
                continue
            avg = sum(vals) / len(vals)
            row = int(round((yhi - avg) / (yhi - ylo) * (height - 1)))
            grid[max(0, min(height - 1, row))][col] = marker

    lines = []
    for i, row in enumerate(grid):
        if i == 0:
            tag = f"{yhi:6.3f}"
        elif i == height - 1:
            tag = f"{ylo:6.3f}"
        elif i == height // 2:
            tag = f"{(yhi + ylo) / 2:6.3f}"
        else:
            tag = "      "
        lines.append(f"  {tag} |{''.join(row)}")
    lines.append("         +" + "-" * width)
    left, right = f"step {xlo:,.0f}", f"step {xhi:,.0f}"
    lines.append("          " + left + " " * max(1, width - len(left) - len(right)) + right)
    return lines


def render(rs: RunDir, last: int, width: int, c: dict) -> str:
    st = rs.read_status()
    rows = rs.read_csv()
    out: list[str] = []

    if st is None and not rows:
        return (f"nothing to report in {rs.path}\n"
                "  no status.json and no loss.csv. Either training has not logged yet, "
                "or this is the wrong --run-dir.")

    if st:
        stale = time.time() - st.get("heartbeat", 0)
        if st.get("stop_reason"):
            state, col = f"finished ({st['stop_reason']})", c["blue"]
        elif st.get("alive") and stale < 180:
            state, col = "training", c["green"]
        else:
            state, col = f"no heartbeat for {fmt_hms(stale)}", c["red"]
        out.append(f"{c['bold']}llm67m{c['reset']}  {col}{state}{c['reset']}"
                   f"{c['dim']}  {st.get('non_embedding_params', 0) / 1e6:.1f}M non-emb"
                   f" | {st.get('world_size', 1)} gpu"
                   f" | {st.get('tokens_per_step', 0):,} tok/step{c['reset']}")
        out.append("")

        elapsed, remaining = st.get("elapsed_s", 0.0), st.get("remaining_s", 0.0)
        total = elapsed + remaining
        out.append(f"  session  [{bar(elapsed / total if total else 1.0)}] "
                   f"{fmt_hms(elapsed)} / {fmt_hms(total)}")
        if st.get("max_steps"):
            out.append(f"  steps    [{bar(st['step'] / st['max_steps'])}] "
                       f"{st['step']:,} / {st['max_steps']:,}")
        if st.get("decay_start") is not None:
            done = max(0, st["step"] - st["decay_start"])
            out.append(f"  lr decay [{bar(done / max(1, st.get('decay_steps', 1)))}] "
                       f"{done:,} / {st.get('decay_steps', 0):,} steps")
        out.append("")

        loss = st.get("loss_ema") or st.get("loss")
        ppl = f"{math.exp(min(20.0, loss)):,.1f}" if loss else "n/a"
        val = f"{st['val_loss']:.4f}" if st.get("val_loss") else "pending"
        best = f"{st['best_val']:.4f}" if st.get("best_val") else "n/a"
        tok_s = st.get("tok_per_s") or 0
        pairs = [
            ("step", f"{st.get('step', 0):,}"),
            ("phase", str(st.get("phase", "?"))),
            ("train loss", f"{loss:.4f}" if loss else "n/a"),
            ("perplexity", ppl),
            ("val loss", val),
            ("best val", best),
            ("throughput", f"{tok_s / 1e3:.1f}k tok/s"),
            ("sec/step", f"{st.get('secs_per_step', 0):.2f}"),
            ("tokens seen", f"{st.get('tokens_seen', 0) / 1e9:.3f}B"),
            ("epoch", f"{st.get('epoch', 0):.2f}"),
            ("lr", f"{st.get('lr', 0):.2e}"),
            ("last save", f"step {st.get('last_save_step', 0):,}"),
        ]
        for i in range(0, len(pairs), 3):
            out.append("  " + "".join(f"{c['dim']}{k:<13}{c['reset']}{v:<16}"
                                      for k, v in pairs[i : i + 3]))
        out.append("")

    if rows:
        def col(name):
            vals = []
            for r in rows:
                v = r.get(name, "")
                vals.append(float(v) if v not in ("", None) else None)
            return vals

        steps = [int(s) for s in col("step")]
        loss, ema, val = col("loss"), col("loss_ema"), col("val_loss")
        keep = [i for i, s in enumerate(steps) if last <= 0 or s >= steps[-1] - last]
        series = {
            "loss": ([steps[i] for i in keep if loss[i] is not None],
                     [loss[i] for i in keep if loss[i] is not None], "."),
            "ema": ([steps[i] for i in keep if ema[i] is not None],
                    [ema[i] for i in keep if ema[i] is not None], "o"),
            "val": ([steps[i] for i in keep if val[i] is not None],
                    [val[i] for i in keep if val[i] is not None], "V"),
        }
        series = {k: v for k, v in series.items() if v[0]}
        span = "whole run" if last <= 0 else f"last {last:,} steps"
        out.append(f"  {c['cyan']}cross entropy{c['reset']} {c['dim']}({span}) "
                   f". raw   o ema   V val{c['reset']}")
        out.extend(ascii_plot(series, width=width))

    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default="/kaggle/working/run")
    p.add_argument("--watch", action="store_true", help="loop instead of printing once")
    p.add_argument("--interval", type=float, default=30.0, help="seconds between refreshes")
    p.add_argument("--count", type=int, default=0, help="stop after N refreshes, 0 is forever")
    p.add_argument("--until-done", action="store_true",
                   help="with --watch, exit when training finishes or stops sending a heartbeat")
    p.add_argument("--stale-minutes", type=float, default=15.0,
                   help="--until-done treats a heartbeat older than this as a dead trainer")
    p.add_argument("--last", type=int, default=0, help="chart only the last N steps, 0 is all")
    p.add_argument("--width", type=int, default=72, help="chart width in columns")
    p.add_argument("--clear", action="store_true", help="redraw in place (terminals, not notebooks)")
    p.add_argument("--no-color", action="store_true")
    p.add_argument("--tail", type=int, default=0, help="print the last N train.log lines and exit")
    p.add_argument("--save", action="store_true", help="ask the trainer to checkpoint now")
    p.add_argument("--decay", action="store_true", help="ask the trainer to start lr decay")
    p.add_argument("--stop", action="store_true", help="ask the trainer to save and exit")
    args = p.parse_args()

    rs = RunDir(args.run_dir)
    c = NO_COLOR if args.no_color else C

    for flag, name in ((args.save, "SAVE_NOW"), (args.decay, "DECAY_NOW"), (args.stop, "STOP_NOW")):
        if flag:
            rs.request(name)
            print(f"{name} requested. The trainer picks it up within a few steps.")
    if args.save or args.decay or args.stop:
        return

    if args.tail:
        log = Path(args.run_dir) / "train.log"
        if not log.exists():
            raise SystemExit(f"no {log}. train.log only exists if you redirected stdout to it.")
        print("".join(log.read_text(errors="replace").splitlines(keepends=True)[-args.tail :]))
        return

    shown = 0
    started = time.time()
    while True:
        text = render(rs, args.last, args.width, c)
        if args.clear:
            print("\x1b[2J\x1b[H", end="")
        print(text, flush=True)
        shown += 1
        if not args.watch or (args.count and shown >= args.count):
            return

        if args.until_done:
            st = rs.read_status()
            if st and st.get("stop_reason"):
                print(f"\n{c['green']}training finished: {st['stop_reason']}, "
                      f"final checkpoint {st.get('final_checkpoint')}{c['reset']}")
                return
            # Only trust a stale heartbeat once the trainer has had time to write one.
            grace = max(args.stale_minutes * 60, 600)
            stale = time.time() - (st or {}).get("heartbeat", 0)
            if time.time() - started > grace and stale > args.stale_minutes * 60:
                print(f"\n{c['red']}no heartbeat for {fmt_hms(stale)}, treating the trainer as "
                      f"dead{c['reset']}")
                log = Path(args.run_dir) / "train.log"
                if log.exists():
                    tail = log.read_text(errors="replace").splitlines()[-40:]
                    print(f"{c['dim']}--- last 40 lines of train.log ---{c['reset']}")
                    print("\n".join(tail))
                raise SystemExit(1)

        print(f"\n{c['dim']}refreshing in {args.interval:.0f}s, ctrl-c to stop{c['reset']}\n",
              flush=True)
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
