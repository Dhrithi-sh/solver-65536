#!/usr/bin/env bash
# P7: official hybrid comparison, all through eval/run.py on the SAME seeds.
#
#   WEIGHTS=weights/ntuple/run1/latest.ntw GAMES=100 SEED0=0 bash scripts/hybrid_experiments.sh
#
# Dev seeds are 0-999; the final report uses the test seeds (SEED0=100000) exactly once.
set -euo pipefail
WEIGHTS=${WEIGHTS:?set WEIGHTS=path/to/latest.ntw}
GAMES=${GAMES:-100}
SEED0=${SEED0:-0}
OUT=${OUT:-results}
export NTUPLE_WEIGHTS="$WEIGHTS"

for agent in ntuple hybrid-d1 hybrid-d2 hybrid-d3 hybrid-65k expectimax-d2; do
  echo "=== $agent"
  python -m eval.run --agent "$agent" --games "$GAMES" --seed0 "$SEED0" --out-dir "$OUT" \
    | tee "$OUT/${agent}.summary.txt"
done
echo "commit: $(git rev-parse --short HEAD)  machine: $(uname -sm)"
