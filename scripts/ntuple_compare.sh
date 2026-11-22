#!/usr/bin/env bash
# Compare n-tuple training variants on equal terms (P5).
#
# Trains 4 configurations with the SAME game count, seed and alpha:
#   plain        TD(0), one network, all weights start at 0
#   multistage   --stages 2 --stage-tile $STAGE_TILE      (Yeh et al. 2017; default 2048, see below)
#   optimistic   --init-value $INIT_VALUE                  (Guei et al. 2022)
#   both         multi-stage + optimistic
# then benchmarks each with eval.run on the shared dev seeds (the only numbers
# that go in the report) and prints a summary table.
#
# Usage (from the repo root, Git Bash / macOS / Linux):
#   scripts/ntuple_compare.sh                 # 50,000 games each (overnight-safe)
#   GAMES=2000 EVAL_GAMES=100 scripts/ntuple_compare.sh   # quick smoke check
#
# Settings (environment variables):
#   GAMES=50000  SEED=10000000  ALPHA=0.1  STAGE_TILE=2048  INIT_VALUE=50000
#
# Why STAGE_TILE defaults to 2048, not Yeh et al.'s late-game 16384: in 50,000
# 1-ply training games almost no game reaches 16,384, so stage 2 would never be
# used and "multistage" would equal "plain". At 2048 most games cross the split,
# so the comparison measures something. Once long runs reach 8,192-16,384
# regularly, rerun with STAGE_TILE=16384 for the late-game version.
#   EVAL_GAMES=1000 (seeds 0..EVAL_GAMES-1)  OUT=weights/ntuple/compare  PYTHON=python
#
# Re-running continues each run from its latest checkpoint, so an interrupted
# night can simply be restarted. Disk: ~1.6 GB of checkpoints (268 MB per stage).

set -euo pipefail

GAMES=${GAMES:-50000}
SEED=${SEED:-10000000}
ALPHA=${ALPHA:-0.1}
STAGE_TILE=${STAGE_TILE:-2048}
INIT_VALUE=${INIT_VALUE:-50000}
EVAL_GAMES=${EVAL_GAMES:-1000}
OUT=${OUT:-weights/ntuple/compare}
PYTHON=${PYTHON:-python}

cd "$(dirname "$0")/.."
mkdir -p "$OUT"

train() {
  local name=$1; shift
  local dir="$OUT/$name"
  local resume=()
  if [[ -f "$dir/latest.ntw" ]]; then
    resume=(--resume "$dir/latest.ntw")
  fi
  echo "=== train $name: $* ==="
  "$PYTHON" -u -m agents.ntuple.train --games "$GAMES" --alpha "$ALPHA" --seed "$SEED" \
    --out "$dir" --checkpoint-every 10000 --log-every 1000 "${resume[@]}" "$@" \
    2>&1 | tee -a "$OUT/$name.train.txt"
}

evaluate() {
  local name=$1
  echo "=== eval $name ==="
  "$PYTHON" -m eval.run --agent ntuple --games "$EVAL_GAMES" --seed0 0 \
    --weights "$OUT/$name/latest.ntw" --out-dir "$OUT/$name" \
    | tee "$OUT/$name.eval.txt"
}

STAGED=(--stages 2 --stage-tile "$STAGE_TILE")
OPTIMISTIC=(--init-value "$INIT_VALUE")

train plain
train multistage "${STAGED[@]}"
train optimistic "${OPTIMISTIC[@]}"
train both "${STAGED[@]}" "${OPTIMISTIC[@]}"

for name in plain multistage optimistic both; do
  evaluate "$name"
done

echo
echo "=== summary: $GAMES training games each, eval on seeds 0..$((EVAL_GAMES - 1)), commit $(git rev-parse --short HEAD) ==="
printf "%-12s %12s %12s %10s %10s %10s\n" config mean_score median reach2048 reach4096 reach8192
for name in plain multistage optimistic both; do
  f="$OUT/$name.eval.txt"
  mean=$(awk '/mean score/ {print $3}' "$f")
  median=$(awk '/median score/ {print $3}' "$f")
  r2k=$(awk '$1 == "reach" && $2 == "2048" {print $3}' "$f")
  r4k=$(awk '$1 == "reach" && $2 == "4096" {print $3}' "$f")
  r8k=$(awk '$1 == "reach" && $2 == "8192" {print $3}' "$f")
  printf "%-12s %12s %12s %10s %10s %10s\n" "$name" "$mean" "$median" "$r2k" "$r4k" "$r8k"
done
echo "training curves: $OUT/<config>/train_log.csv"
