#!/usr/bin/env bash
# The BlackboxNLP reviewer-round experiment queue. Runs sequentially (one GPU);
# every step writes to results/ablation/ and logs to out/revision/.
#
#   bash scripts/run_revision.sh 1.7b          # the full battery on one model
#   bash scripts/run_revision.sh 8b            # the replication
#   bash scripts/run_revision.sh google/gemma-2-9b
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
M="${1:-1.7b}"
TAG="$(basename "$M")"
LOG="out/revision/${TAG}"
mkdir -p "$LOG"

run() {  # run <name> <args...>
  local name="$1"; shift
  echo "=== [$TAG] $name  $(date +%H:%M:%S)"
  if "$@" >"$LOG/$name.log" 2>&1; then
    echo "    ok -> $LOG/$name.log"
  else
    echo "    FAILED (exit $?) -> $LOG/$name.log"; tail -5 "$LOG/$name.log"
  fi
}

# --- relations: the reviewer-round experiments -------------------------------
run patch_relations       $PY scripts/op_patch_decomp.py "$M" --domain relations
run patch_relations_1L    $PY scripts/op_patch_decomp.py "$M" --domain relations --scope single
run positions_relations   $PY scripts/op_positions.py    "$M" --domain relations
run nulls_relations       $PY scripts/op_nulls.py        "$M" --domain relations --seeds 20
run layersweep_relations  $PY scripts/op_layer_sweep.py  "$M" --domain relations
# dose-response with NEGATIVE alphas: is the direction a signed axis?
run dose_relations        $PY scripts/op_dose.py         "$M" --domain relations \
                              --alphas -12 -8 -4 -2 -1 -0.5 0.5 1 2 4 6 8 12

# --- animals: the non-geographic domain, full pipeline ------------------------
run paradigm_animals      $PY scripts/operator_paradigm.py  "$M" --domain animals
run factorize_animals     $PY scripts/operator_factorize.py "$M" --domain animals
run templates_animals     $PY scripts/operator_templates.py "$M" --domain animals
run minimal_animals       $PY scripts/op_minimal.py         "$M" --domain animals
run patch_animals         $PY scripts/op_patch_decomp.py    "$M" --domain animals
run nulls_animals         $PY scripts/op_nulls.py           "$M" --domain animals --seeds 20
run positions_animals     $PY scripts/op_positions.py       "$M" --domain animals
run dose_animals          $PY scripts/op_dose.py            "$M" --domain animals
run geometry_animals      $PY scripts/op_geometry_dump.py   "$M" --domain animals

echo "=== [$TAG] done  $(date +%H:%M:%S)"
