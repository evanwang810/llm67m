#!/usr/bin/env bash
# One command that does the whole run. Safe to re-run: tokenization resumes
# where it left off and training resumes from the newest checkpoint.
#
#   bash kaggle_run.sh [HOURS] [PRESET] [MAX_TOKENS]
#
#   bash kaggle_run.sh 9                  # 9 hour budget, 1session preset
#   bash kaggle_run.sh 11.5 125m 4e9      # long session, bigger model, more data
#
# HOURS is your total budget including tokenizing. Everything else is derived
# from it, so there is one number to think about.
set -euo pipefail

HOURS="${1:-9}"
PRESET="${2:-1session}"
MAX_TOKENS="${3:-1.5e9}"

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKENS="${TOKENS_DIR:-/kaggle/working/tokens}"
RUN="${RUN_DIR:-/kaggle/working/run}"

echo "=== llm67m: ${HOURS}h budget, preset ${PRESET}, ${MAX_TOKENS} tokens ==="
cd "$CODE"

if [ "${SKIP_PIP:-0}" != "1" ]; then
  pip install -q tiktoken datasets
  # Optional, for braille charts in the monitor. Not the PyPI package of the same
  # name. Failure here is fine, the monitor falls back to ASCII.
  if [ -n "${TERMPLOT_REPO:-}" ]; then
    pip install -q "git+${TERMPLOT_REPO}" || echo "termplot install failed, using ascii charts"
  fi
fi

# Kaggle mounts a notebook's output at /kaggle/input/notebooks/<user>/<slug>/,
# which nobody guesses correctly. Rather than make you type a path, go and find
# a finished token set anywhere under /kaggle/input and use the biggest one.
find_tokens() {
  python - "$1" <<'PY'
import json, sys
from pathlib import Path

want = Path(sys.argv[1])
if (want / "meta.json").exists():
    print(want)
    raise SystemExit

best = None
patterns = ["meta.json", "*/meta.json", "*/*/meta.json", "*/*/*/meta.json",
            "*/*/*/*/meta.json", "*/*/*/*/*/meta.json"]
for root in (Path("/kaggle/input"), Path("/kaggle/working")):
    if not root.exists():
        continue
    for pat in patterns:
        for meta in root.glob(pat):
            try:
                m = json.loads(meta.read_text())
            except Exception:
                continue
            if "shards" not in m:
                continue
            n = int(m.get("total_tokens", 0))
            if n > 0 and (best is None or n > best[0]):
                best = (n, meta.parent)
if best:
    print(best[1])
PY
}

if [ "${SMOKE:-0}" = "1" ]; then
  # Whole pipeline in a few minutes on synthetic data, same code path.
  echo "SMOKE=1: skipping tokenization, training a tiny model on generated data"
else
  FOUND="$(find_tokens "$TOKENS" 2>/dev/null || true)"
  if [ -n "$FOUND" ] && [ "$FOUND" != "$TOKENS" ]; then
    echo "found existing tokens at $FOUND, using those instead of $TOKENS"
    TOKENS="$FOUND"
  fi
  # Never try to write into the read-only input mount.
  case "$TOKENS" in
    /kaggle/input/*)
      if [ ! -f "$TOKENS/meta.json" ]; then
        echo "TOKENS_DIR=$TOKENS is under the read-only /kaggle/input and has no" >&2
        echo "meta.json. Check the real path in the file browser, or unset" >&2
        echo "TOKENS_DIR to tokenize fresh into /kaggle/working." >&2
        exit 1
      fi
      ;;
  esac
fi

if [ "${SMOKE:-0}" = "1" ]; then
  :
else
  # Resumable: if meta.json already has enough tokens this returns immediately.
  # A non-zero exit here is not conclusive. The HF streaming reader can abort
  # during interpreter shutdown, after the shards are already written, so ask
  # separately whether the data is actually complete rather than trusting the
  # exit code.
  python tokenize_fineweb.py --out-dir "$TOKENS" --max-tokens "$MAX_TOKENS" \
    || echo "tokenizer exited non-zero, checking the shards themselves"
  python tokenize_fineweb.py --out-dir "$TOKENS" --max-tokens "$MAX_TOKENS" --verify-only
fi

# Reserve time for tokenizing, and derive the schedule knobs from the budget:
# decay over roughly the last 15% of steps, and about 8 permanent milestones.
read -r TRAIN_HOURS DECAY_STEPS MILESTONE_MIN <<EOF
$(python - "$HOURS" <<'PY'
import sys
hours = float(sys.argv[1])
train = max(0.4, hours - 0.45)
print(f"{train:.2f} {max(200, int(120 * train))} {max(15, int(train * 60 / 8))}")
PY
)
EOF

# Fall back to a single process if this box does not actually have two GPUs.
NGPU="$(python -c 'import torch; print(torch.cuda.device_count())')"
if [ "$NGPU" -ge 2 ]; then
  LAUNCH=(torchrun --nproc_per_node="$NGPU")
else
  echo "WARNING: found $NGPU gpu(s), not 2. Check the Accelerator setting is GPU T4 x2."
  LAUNCH=(python)
fi

echo "=== training ${TRAIN_HOURS}h on ${NGPU} gpu, decay ${DECAY_STEPS} steps, "\
"milestone every ${MILESTONE_MIN} min ==="

START_TS="$(date +%s)"
DEADLINE_SECONDS="$(python -c "print(int($TRAIN_HOURS * 3600))")"

TRAIN_CMD=("${LAUNCH[@]}" train.py
  --preset "$PRESET"
  --data-dir "$TOKENS"
  --run-dir "$RUN"
  --deadline-hours "$TRAIN_HOURS"
  --session-start "$START_TS"
  --auto-decay --decay-steps "$DECAY_STEPS"
  --keep-checkpoints 1 --keep-weights "${KEEP_WEIGHTS:-0}"
  --save-every-min "${SAVE_EVERY_MIN:-25}"
  --milestone-every-min "$MILESTONE_MIN")

# Continuing a previous session: point at the ckpt_step*.pt from its output.
# Without this, resume still works by globbing /kaggle/input, but being explicit
# means a wrong path fails loudly instead of silently starting from scratch.
if [ -n "${RESUME_FROM:-}" ]; then
  if [ ! -f "$RESUME_FROM" ]; then
    echo "RESUME_FROM=$RESUME_FROM does not exist" >&2
    exit 1
  fi
  echo "resuming from $RESUME_FROM"
  TRAIN_CMD+=(--resume-from "$RESUME_FROM")
fi

# A long run on free hardware will occasionally die on something outside our
# control: a CUDA fault, an NCCL abort, a preempted GPU. Training resumes from
# its last checkpoint with optimizer and loss-scale state intact, so the right
# response is to start it again rather than lose the session. --session-start
# pins the deadline to the original clock, so restarts cannot extend the run.
train_with_restarts() {
  local attempt code now
  for attempt in $(seq 1 "${ATTEMPTS:-8}"); do
    if [ "$attempt" -gt 1 ]; then
      echo "=== restart $((attempt - 1)), resuming from the last checkpoint ==="
    fi
    code=0
    "${TRAIN_CMD[@]}" || code=$?
    if [ "$code" -eq 0 ]; then
      return 0
    fi
    now="$(date +%s)"
    if [ $((now - START_TS)) -ge "$DEADLINE_SECONDS" ]; then
      echo "=== training exited $code but the deadline has passed, stopping ==="
      return 0
    fi
    echo "=== training exited $code with $(( (DEADLINE_SECONDS - now + START_TS) / 60 )) min left, retrying in 20s ==="
    sleep 20
  done
  echo "=== gave up after ${ATTEMPTS:-8} attempts ==="
  return 1
}

# Turn NCCL faults into a Python exception with a message instead of an opaque
# abort, so the next failure says what it actually was.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1

if [ "${SMOKE:-0}" = "1" ]; then
  TRAIN_CMD+=(--smoke-test --max-steps "${SMOKE_STEPS:-80}")
fi

# With MONITOR=1, train in the background and put the monitor in front, so a
# committed run's log becomes readable status blocks with a loss chart instead
# of a wall of step lines.
CODE_EXIT=0
if [ "${MONITOR:-0}" = "1" ]; then
  mkdir -p "$RUN"
  rm -f "$RUN/train.exit"
  echo "monitor mode: full training output goes to $RUN/train.log"
  # The marker file, not the pid, tells the monitor when to stop. A finished
  # background job lingers as a zombie that still answers kill(pid, 0), so a
  # liveness check would never return. A stale heartbeat is not a signal either:
  # it just means rank 0 is mid-save or mid-restart.
  { c=0; train_with_restarts || c=$?; echo "$c" > "$RUN/train.exit"; } \
    > "$RUN/train.log" 2>&1 &
  TRAIN_PID=$!
  python monitor.py --run-dir "$RUN" --watch --exit-file "$RUN/train.exit" \
    --interval "${MONITOR_INTERVAL:-300}" --until-done --no-color || true
  wait "$TRAIN_PID" || true
  CODE_EXIT="$(cat "$RUN/train.exit" 2>/dev/null || echo 1)"
else
  train_with_restarts || CODE_EXIT=$?
fi

if [ "$CODE_EXIT" -ne 0 ]; then
  echo "=== training failed with $CODE_EXIT, not going on to fine-tuning ===" >&2
  exit "$CODE_EXIT"
fi

# Fine-tuning as part of the same command, so it cannot run off a stale
# checkpoint when training never happened. Unset SFT_HOURS to skip it.
if [ -n "${SFT_HOURS:-}" ]; then
  echo "=== instruction tuning for ${SFT_HOURS}h ==="
  exec python finetune.py --run-dir "$RUN" --hours "$SFT_HOURS"
fi
