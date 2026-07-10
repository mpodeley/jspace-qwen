#!/usr/bin/env bash
# Overnight batch: finish 1.7B-int8, then 8B (bf16 + int8), with all analyses.
# STOPS before 32B (user reviews the 8B quantization control first).
# Resumable: fits checkpoint every few prompts; re-running skips finished lenses.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/models"
PY="$PWD/.venv/bin/python"
NP=64
LOG=out/batch.log
mark() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

analyze() {  # <key> [--int8]
  local key="$1"; shift; local flag="${1:-}"
  mark "  metrics $key $flag";    $PY scripts/metrics.py    "$key" $flag 2>>"$LOG" | tee -a "$LOG"
  mark "  lens_eval $key $flag";  $PY scripts/lens_eval.py  "$key" $flag --limit 40 2>>"$LOG" | tee -a "$LOG"
  mark "  causal $key $flag";     $PY scripts/causal_swap.py "$key" $flag 2>>"$LOG" | tee -a "$LOG"
}

fit_if_needed() {  # <tag> <key> [--int8]  -> fits to out/lenses/<tag>.pt unless present
  local tag="$1" key="$2"; shift 2; local flag="${1:-}"
  if [ -f "out/lenses/${tag}.pt" ]; then mark "SKIP fit $tag (exists)"; return 0; fi
  local extra="--compile"; [ "$flag" = "--int8" ] && extra=""   # compile not composed with int8
  mark "FIT $tag ..."
  $PY scripts/fit_lens.py "$key" $flag $extra --n-prompts "$NP" 2>>"$LOG" | grep -E 'prompt [0-9]+/|saved|int8:' >>"$LOG"
}

: > "$LOG"; mark "=== BATCH START (through 8B; NO 32B) ==="

# 1) finish the already-running 1.7B int8 fit, then analyse it
mark "waiting for running 1.7B-int8 fit ..."
while pgrep -f "fit_lens.py 1.7b --int8" >/dev/null; do sleep 20; done
fit_if_needed "1.7b-int8" 1.7b --int8
analyze 1.7b --int8

# 2) 8B bf16  (downloads ~16 GB on first use)
fit_if_needed "8b" 8b
analyze 8b

# 3) 8B int8  (quantization control)
fit_if_needed "8b-int8" 8b --int8
analyze 8b --int8

mark "=== BATCH COMPLETE (through 8B) — STOP before 32B ==="
