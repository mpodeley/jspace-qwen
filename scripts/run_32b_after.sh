#!/usr/bin/env bash
# Wait for the running 32B int8 fit, then run all analyses + figures for 32B,
# regenerate the scale plots, and rebuild the site. Everything is forward-only
# (the fit is already done), so no OOM risk once the fit frees the GPU.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/models"
PY="$PWD/.venv/bin/python"
LOG=out/after32b.log
mark() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run() { mark "RUN $*"; "$PY" "$@" 2>>"$LOG" | tee -a "$LOG" || mark "  (failed: $*)"; }
: > "$LOG"

mark "waiting for 32B int8 fit to finish ..."
while pgrep -f "fit_lens.py 32b" >/dev/null; do sleep 60; done
[ -f out/lenses/32b-int8.pt ] || { mark "ERROR: 32b-int8.pt not found"; exit 1; }
mark "32B lens present: $(ls -la out/lenses/32b-int8.pt | awk '{print $5}') bytes"

# core scale + causal
run scripts/metrics.py     32b --int8
run scripts/lens_eval.py   32b --int8 --limit 40
run scripts/causal_swap.py 32b --int8

# reservoir-view figures on the big model (cleanest)
run scripts/reservoir_field.py 32b --int8 --concepts Italy euro
run scripts/permeability.py    32b --int8 --answer euro
run scripts/flow.py            32b --int8 --intermediate Italy --answer euro
run scripts/inject.py          32b --int8 --to "United States" Japan France Russia
run scripts/injector_sweep.py  32b --int8 --to "United States" --currency-to dollar
run scripts/kr_curves.py       32b --int8 --concepts euro dollar yen pound

# regenerate scale figures (now with 32B) and rebuild the site
run scripts/plots.py
mark "=== 32B ANALYSES COMPLETE ==="
