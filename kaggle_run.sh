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

if [ "${SMOKE:-0}" = "1" ]; then
  # Whole pipeline in a few minutes on synthetic data, same code path.
  echo "SMOKE=1: skipping tokenization, training a tiny model on generated data"
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
  --keep-checkpoints 1 --keep-weights 2
  --save-every-min "${SAVE_EVERY_MIN:-15}"
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

if [ "${MONITOR:-0}" = "1" ]; then
  # Train in the background and put the monitor in front, so a committed run's
  # log becomes readable status blocks with a loss chart instead of a wall of
  # step lines. --until-done ends the cell as soon as the final checkpoint
  # lands, or bails out and dumps the log if the trainer dies.
  mkdir -p "$RUN"
  echo "monitor mode: full training output goes to $RUN/train.log"
  train_with_restarts > "$RUN/train.log" 2>&1 &
  TRAIN_PID=$!
  trap 'kill $TRAIN_PID 2>/dev/null || true' EXIT
  # --stale-minutes has to exceed a restart gap, or the monitor would call a
  # recovering run dead and end the notebook out from under it.
  exec python monitor.py --run-dir "$RUN" --watch \
    --interval "${MONITOR_INTERVAL:-300}" --until-done --stale-minutes 25 --no-color
fi

train_with_restarts
