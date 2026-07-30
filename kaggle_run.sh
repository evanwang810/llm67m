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

pip install -q tiktoken datasets

# Resumable: if meta.json already has enough tokens this returns immediately.
python tokenize_fineweb.py --out-dir "$TOKENS" --max-tokens "$MAX_TOKENS"

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

exec "${LAUNCH[@]}" train.py \
  --preset "$PRESET" \
  --data-dir "$TOKENS" \
  --run-dir "$RUN" \
  --deadline-hours "$TRAIN_HOURS" \
  --auto-decay --decay-steps "$DECAY_STEPS" \
  --keep-checkpoints 1 --keep-weights 2 \
  --milestone-every-min "$MILESTONE_MIN"
