# agents/heuristics — heuristic evaluator and weight tuning

A hand-designed board evaluation function, exposed as `ValueFunction.evaluate(board) -> float`
(see `agents/base.py`), with weights tuned automatically by CMA-ES. It is used directly by a
1-ply agent and is meant to be the leaf evaluator of the expectimax search.

```python
from agents.heuristics import HeuristicEvaluator
evaluator = HeuristicEvaluator.from_file("tuned")   # any ValueFunction consumer can use this
evaluator.evaluate(board)                           # board: 4x4 list of tile values, 0 = empty
```

## Layout

```
agents/heuristics/
  features.py        the six feature functions, exponents(), FEATURES
  evaluator.py       HeuristicEvaluator: weighted sum, per-phase weights, JSON save/load
  agent.py           HeuristicAgent: 1-ply, best afterstate
  __init__.py        heuristic_agents(): one benchmark agent per weight file present
  tune.py            CMA-ES tuning
  ablation.py        retune with each feature removed
  report.py          results tables from eval.run
  requirements.txt   cma (tuning only; not needed to play or to run the tests)
  weights/
    untuned.json     hand-set starting point            -> agent heuristic-untuned
    tuned.json       CMA-ES, one weight set             -> agent heuristic
    phased.json      CMA-ES, per-phase sets (if kept)   -> agent heuristic-phased
    ablation/        full.json, no_<feature>.json       -> heuristic-full, heuristic-no-<feature>
tests/test_heuristics.py
logs/heuristics/     per-generation tuning CSVs (git-ignored)
```

## Features

Each feature is a separate pure function of a 4x4 grid of **log2 exponents** (0 = empty,
k = tile 2^k, 65,536 = 16) and is oriented so that **higher is always better**; penalties
are returned negated. All six are invariant under the 8 rotations/reflections of the board.

| name | function | value | why |
|---|---|---|---|
| `empty` | `empty_count` | number of empty cells | Free cells are what keeps the game alive; every spawn eats one. |
| `mono` | `monotonicity` | −Σ over the 8 lines of min(sum of rises, sum of falls) | A line that only rises or only falls costs 0. Otherwise it is charged for the *smaller* direction, i.e. how far it is from its dominant ordering. Ordered lines keep big tiles from being trapped between small ones. Empty cells count as 0. |
| `smooth` | `smoothness` | −Σ \|a − b\| over adjacent occupied pairs | Neighbours with close exponents are a few merges away from combining. |
| `corner` | `max_in_corner` | 1 if a largest tile is in a corner, else 0 | The classic "keep the big tile in the corner" rule as a single bit. |
| `snake` | `snake` | max over 8 orientations of Σ 0.5^k · e(path_k) | Rewards tiles sorted in descending order along a zig-zag path starting in a corner. |
| `merge` | `merge_potential` | number of adjacent equal non-empty pairs | Merges available right now. |

**Why exponents and not tile values.** Tile values double at every step, so any sum of raw
values or raw differences is dominated by the one or two largest tiles and says almost nothing
about the rest of the board. On the exponent scale a gap of 1 always means "one merge away",
whether the tiles are 4 and 8 or 16,384 and 32,768, so one weight works across the whole game.
It also matches the bitboard engine, which stores exponents.

**Why smoothness skips empty cells.** An empty cell is exponent 0, so without the skip a lone
2048 next to an empty cell would be charged a gap of 11, and the evaluator would learn to
dislike open space, the opposite of `empty`. Only pairs where both cells hold a tile are counted.
(Monotonicity does treat empties as 0: an empty cell at the low end of a line is consistent with
the ordering, and one in the middle of a line of big tiles genuinely breaks it.)

**The snake ratio (0.5).** The k-th cell on the path has weight 0.5^k. The ratio is the trade-off
between "the head of the snake matters most" and "the tail still counts". With 0.5 the first cell
outweighs all later cells put together (1 > ½ + ¼ + …), so moving the largest tile off the corner
can never be paid for by tidying the rest, and the same holds from each cell to the ones after it.
Because the feature works on exponents (0–16) rather than tile values, a geometric fall-off this
steep is still numerically mild. The price is that the last cells have almost no weight, so the
tail of the board is left to `mono`, `smooth` and `empty`. The ratio is a constant
(`features.SNAKE_RATIO`), not a tuned parameter. Taking the max over the 8 symmetries means no
particular corner is preferred and keeps the feature symmetric.

## Evaluator and weight files

`HeuristicEvaluator(weights).evaluate(board)` is the weighted sum of the features.

- A feature missing from the weights counts as 0, and zero-weight features are not computed.
  The ablation runs rely on this: dropping a feature is just leaving it out.
- Unknown feature names raise `ValueError` (a typo must not silently become a 0 weight).
- `weights` is either one set, or three per-phase sets chosen by the board's **max tile**:

  | phase | max tile |
  |---|---|
  | `early` | < 2048 |
  | `mid` | 2048 – 8192 |
  | `late` | ≥ 16,384 |

  The bounds are stored in the file (`phase_bounds`: the smallest max tile of `mid` and `late`),
  so a file fully describes its own behaviour.

Single-set file:

```json
{
  "meta": {"source": "hand-set", "note": "..."},
  "weights": {"empty": 2.7, "mono": 1.0, "smooth": 0.1, "corner": 2.0, "snake": 1.0, "merge": 0.7}
}
```

Per-phase file:

```json
{
  "meta": {"source": "cma-es", "command": "...", "dev_seeds": [0, 63], "...": "..."},
  "phase_bounds": {"mid": 2048, "late": 16384},
  "phases": {"early": {"empty": 0.6, "...": 0}, "mid": {"...": 0}, "late": {"...": 0}}
}
```

`meta` records how the file was produced: for tuned files the exact command, mode, features,
tuned phases, player, dev seeds, CMA-ES settings, generations run, and the dev score of the
starting point and of the result. `HeuristicEvaluator.from_file("tuned")` and `.save("tuned")`
resolve a bare name (also `"ablation/full"`) to the `weights/` folder; anything ending in
`.json` is used as a path.

**The weights folder is an exception to `.gitignore`.** The repo ignores every folder called
`weights/` because learned weights are large. These files are a few hundred bytes and the
benchmark agents are built from them, so `.gitignore` re-includes `agents/heuristics/weights/`.
Check with `git check-ignore -v agents/heuristics/weights/tuned.json` (prints nothing).

## 1-ply agent

`HeuristicAgent` slides the board in each legal direction with `Game.slide` (the *afterstate*:
after the slide, before the random spawn) and plays the move whose afterstate scores highest.
Ties go to the lowest move index, so the agent is deterministic. It never mutates the `Game`.

The merge reward is **not** added to the value. The features are on an exponent scale and the
reward on a tile-value scale, so adding it would let the reward swamp the heuristic late in the
game; what a merge is worth is already visible in the afterstate (more empty cells).

**Only the direction of the weight vector matters.** The agent compares weighted sums, so
multiplying every weight by the same positive number changes no decision. The tuner therefore
fixes the scale (largest |weight| = 1). The same is true inside expectimax, since expectation and
max both commute with positive scaling, as long as the search does not add raw rewards to leaf
values.

## Tuning

`tune.py` maximises the **mean benchmark score over a fixed block of dev seeds** (default
0–63), playing each game through `eval.run.play_game`, the benchmark's own code path.
The (candidate, seed) games of a generation run in a `multiprocessing.Pool`.

**Why CMA-ES.** The objective is a black box: the weights only act through an argmax over moves,
so the score is piecewise constant in the weights and has no useful gradient. The search space is
small (6 weights, 12–18 with phases). CMA-ES is built for this: it only uses the *ranking* of
candidates, so heavy-tailed game scores do no harm; it adapts its step size and a full covariance
matrix, which matters because the features are correlated (`snake`, `mono` and `corner` all
reward similar boards); and it has essentially one setting to choose, the initial step size.
A population also parallelises trivially.

**Why fixed seeds.** Game scores are very noisy (standard deviation of the same order as the
mean). If every candidate played different games, most of the difference between two candidates
would be luck. Using the same seeds for every candidate (common random numbers) makes the
objective a deterministic function of the weights, so comparisons inside a generation are fair
and a run is reproducible from its command line.

**Why the saved best's tuning score is optimistic.** The saved file is the best of several
hundred candidates on those 64 games. Even though the seeds are fixed, 64 games are a small
sample, and picking the maximum of many noisy estimates overstates the winner (it is partly the
candidate that happens to suit these particular seeds). `meta.dev_mean_score` is therefore an
upper bound, not a result. Real numbers come only from `eval.run` on the test seeds, which
tuning never touches: `tune.py` refuses any seed block overlapping 10000–10999 or
100000–100999.

Other points:

- The starting point is scored on the same seeds first and counts as the first "best so far",
  so a saved file is never worse than its start on the dev seeds; `meta.improved_on_init`
  says whether the run found anything better. The best-so-far file is rewritten every generation,
  so an interrupted run still leaves usable weights.
- Weights are unconstrained. A feature is "higher is better" on its own, but next to
  correlated features its best weight can come out negative.
- `--sigma` is relative to the scaled start (largest |weight| = 1). 0.5 explores widely and
  suits a hand-set start. For a warm start from tuned weights use 0.2: at 0.5 a short run
  around an already good point mostly samples worse vectors.
- **Late phase is tied to mid by default.** In phased mode only `early` and `mid` are tuned and
  `late` copies `mid`. At 1-ply, games almost never reach 16,384, so late weights would never
  be used during tuning and CMA-ES would return noise for them. `--phases` chooses which phases
  are tuned; an untuned phase copies the nearest tuned one. Once a search player reaches the
  late game, tune it with `--phases early mid late`.
- `--player expectimax --depth D` plays the tuning games with
  `agents.expectimax.ExpectimaxAgent(evaluator, depth)` instead of the 1-ply agent. The import
  is lazy and the script exits with a clear message while that class does not exist.

## Commands

```bash
pip install -r requirements.txt -r agents/heuristics/requirements.txt

# benchmark and replay (same as every other agent)
python -m eval.run --agent heuristic --games 100 --seed0 0
python -m eval.replay results/heuristic.jsonl --index 0

# single weight set: weights/untuned.json -> weights/tuned.json
python -m agents.heuristics.tune --games 64 --iters 40

# per-phase weights, warm-started from tuned -> weights/phased.json
python -m agents.heuristics.tune --mode phased --init tuned --sigma 0.2 --games 64 --iters 40

# ablation: weights/ablation/full.json and no_<feature>.json, same budget and CMA seed
python -m agents.heuristics.ablation --games 64 --iters 40

# results tables on the test seeds
python -m agents.heuristics.report --table all --games 1000 --seed0 100000 --workers 4

# retune for the search once agents.expectimax.ExpectimaxAgent exists
python -m agents.heuristics.tune --player expectimax --depth 1 --init tuned --sigma 0.2 \
    --games 64 --iters 40 --out tuned_expectimax_d1
```

`tune.py` flags: `--mode single|phased`, `--phases`, `--drop FEATURE` (repeatable), `--init`,
`--games`, `--seed0`, `--iters`, `--sigma`, `--popsize`, `--cma-seed`, `--workers`,
`--player 1ply|expectimax`, `--depth`, `--out`, `--log` (default `logs/heuristics/<out>.csv`).
`ablation.py` adds `--cma-seed` and `--only`, and passes everything else through to `tune.py`.
`report.py`: `--table tuning|phases|ablation|all`, `--games`, `--seed0`, `--out-dir`,
`--workers`, `--out`. A new weight file under a registered name shows up in `eval.run`
automatically.

## Results

Seeds 100000–100999 (1000 games) via `python -m eval.run`. Reach columns: share of games whose max tile is at least that value.

### Tuning (1-ply)

| agent | mean score | vs greedy | median | 2048 | 4096 | 8192 | 16384 | ms/move |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `greedy` | 3,141 | — | 2,996 | 0.0% | 0.0% | 0.0% | 0.0% | 0.043 |
| `heuristic-untuned` | 13,844 | +10,703 (+340.8%) | 13,036 | 13.6% | 0.0% | 0.0% | 0.0% | 0.072 |
| `heuristic` | 20,882 | +17,741 (+564.9%) | 16,960 | 39.2% | 2.9% | 0.0% | 0.0% | 0.073 |

### Single vs per-phase weights (1-ply)

| agent | mean score | vs heuristic | median | 2048 | 4096 | 8192 | 16384 | ms/move |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `heuristic` | 20,882 | — | 16,960 | 39.2% | 2.9% | 0.0% | 0.0% | 0.073 |

### Ablation (1-ply, each run retuned)

| agent | mean score | vs heuristic-full | median | 2048 | 4096 | 8192 | 16384 | ms/move |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `heuristic-full` | 20,882 | — | 16,960 | 39.2% | 2.9% | 0.0% | 0.0% | 0.074 |
| `heuristic-no-empty` | 18,893 | -1,990 (-9.5%) | 16,370 | 35.4% | 0.6% | 0.0% | 0.0% | 0.070 |
| `heuristic-no-mono` | 15,813 | -5,069 (-24.3%) | 15,622 | 21.4% | 0.2% | 0.0% | 0.0% | 0.068 |
| `heuristic-no-smooth` | 19,956 | -926 (-4.4%) | 16,656 | 35.4% | 0.9% | 0.0% | 0.0% | 0.067 |
| `heuristic-no-corner` | 21,251 | +368 (+1.8%) | 17,062 | 43.6% | 2.1% | 0.0% | 0.0% | 0.067 |
| `heuristic-no-snake` | 18,195 | -2,688 (-12.9%) | 16,296 | 32.1% | 0.2% | 0.0% | 0.0% | 0.054 |
| `heuristic-no-merge` | 14,312 | -6,570 (-31.5%) | 13,262 | 18.6% | 0.5% | 0.0% | 0.0% | 0.066 |

Produced with `python -m agents.heuristics.report --table all --games 1000 --seed0 100000 --workers 4`
(four agents benchmarked at once, so ms/move is slightly noisier than a single-process run).

What the numbers say:

- Tuning lifts the 1-ply agent from 13,844 to 20,882 mean score, and 2048 reach from 13.6% to 39.2%.
  No 1-ply agent reaches 8192; that needs search on top of this evaluator.
- The tuning run had not fully converged after 40 generations (the best score was still rising and the
  step size was about 0.22), so a longer run may find slightly better weights.
- The dev-seed score of the tuned weights (23,937 on seeds 0–63) is optimistic, as expected for the
  best of many noisy candidates; the test-seed figure above is the honest one.
- Ablation: `merge` and `mono` matter most (−31.5% and −24.3% without them), then `snake` and `empty`.
  `smooth` is a small loss (−4.4%).
- `corner` does not help: the tuner gave it a negative weight in `tuned.json`, and the run without it
  scores 368 points higher (+1.8%, within noise for 1,000 games). `snake` already rewards a cornered
  max tile, so the binary feature is redundant at 1-ply.
- Per-phase weights: the first phased run (warm start from `tuned`, same dev seeds) never beat its
  starting point, so no `phased.json` is committed and the phases table has only the baseline row.


## TODO

- [x] Features on exponents: empty cells, monotonicity, smoothness, corner, snake, merge potential
- [x] Weighted-sum evaluator with JSON weight files and a 1-ply agent
- [x] Agents registered in `eval/run.py` from the weight files present
- [x] CMA-ES tuning on fixed dev seeds, separate from the test seeds
- [ ] Per-phase weights that beat the single set (tooling is in place; the first run did not improve)
- [x] Ablation: retune with each feature removed, report the score drop
- [ ] Retune with depth-1 expectimax as the player once `agents.expectimax.ExpectimaxAgent` exists
- [ ] Tune the late phase once a player reaches 16,384 often enough to give it signal
- [ ] Agree the official test seed range (`eval/README.md` suggests 10000–10999, the task brief 100000–100999)
- [ ] Try tuning `SNAKE_RATIO` and a longer / larger-seed-block CMA-ES run
