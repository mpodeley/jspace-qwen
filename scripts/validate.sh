#!/usr/bin/env bash
# Wait for a running fit of <tag> to finish, then run the three validations.
#   scripts/validate.sh 1.7b
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/models"
PY="$PWD/.venv/bin/python"
TAG="${1:?usage: validate.sh <tag>}"
LOG="out/validate_${TAG}.log"
: > "$LOG"

echo "waiting for fit_lens.py $TAG to finish ..." | tee -a "$LOG"
while pgrep -f "fit_lens.py $TAG" >/dev/null; do sleep 20; done
echo "fit done; lens: $(ls -la out/lenses/${TAG}.pt 2>/dev/null)" | tee -a "$LOG"

echo "=== DEMO: boot-country probe (J-lens vs logit-lens) ===" | tee -a "$LOG"
$PY scripts/apply_lens.py "$TAG" --demo 2>/dev/null | tee -a "$LOG"

echo "=== LENS-EVAL: multihop pass@k (first 40 items) ===" | tee -a "$LOG"
$PY scripts/lens_eval.py "$TAG" --limit 40 2>/dev/null | tee -a "$LOG"

echo "=== METRICS: per-layer ===" | tee -a "$LOG"
$PY scripts/metrics.py "$TAG" 2>/dev/null | tee -a "$LOG"
echo "=== VALIDATION COMPLETE ($TAG) ===" | tee -a "$LOG"
