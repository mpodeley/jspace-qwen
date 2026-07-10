#!/usr/bin/env bash
# 8B only, N=32 (big-model fits are ~5 min/prompt; 32 keeps overnight sane).
# 8B bf16 resumes from its prompt-4 checkpoint. STOPS before 32B.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/models"
PY="$PWD/.venv/bin/python"
NP=32
LOG=out/batch.log
mark() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

analyze() {  # <key> [--int8]
  local key="$1"; shift; local flag="${1:-}"
  mark "  metrics $key $flag";   $PY scripts/metrics.py    "$key" $flag 2>>"$LOG" | tee -a "$LOG"
  mark "  lens_eval $key $flag"; $PY scripts/lens_eval.py  "$key" $flag --limit 40 2>>"$LOG" | tee -a "$LOG"
  mark "  causal $key $flag";    $PY scripts/causal_swap.py "$key" $flag 2>>"$LOG" | tee -a "$LOG"
}

fit_if_needed() {  # <tag> <key> [--int8]
  local tag="$1" key="$2"; shift 2; local flag="${1:-}"
  if [ -f "out/lenses/${tag}.pt" ]; then mark "SKIP fit $tag (exists)"; return 0; fi
  local extra="--compile"; [ "$flag" = "--int8" ] && extra=""
  mark "FIT $tag (N=$NP, resume if ckpt) ..."
  $PY scripts/fit_lens.py "$key" $flag $extra --n-prompts "$NP" 2>>"$LOG" \
    | grep -E 'prompt [0-9]+/|saved|int8:' >>"$LOG"
}

mark "=== 8B BATCH START (N=$NP; resume bf16; NO 32B) ==="
fit_if_needed "8b" 8b            && analyze 8b
fit_if_needed "8b-int8" 8b --int8 && analyze 8b --int8
mark "=== BATCH COMPLETE (through 8B) — STOP before 32B ==="
