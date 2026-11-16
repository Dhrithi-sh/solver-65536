# agents/ntuple — P5 (N-tuple network)

**Owner:** P5 · **Review buddy:** P6 · **Label:** `P5-ntuple`

An n-tuple network value function trained by self-play with TD(0) on afterstates. It
provides:

- **`NTupleEvaluator.evaluate(board) -> float`**, which P3 (expectimax) and P7 (hybrid)
  use as their board evaluator, and
- **`NTupleAgent`**, a 1-ply greedy agent registered as `ntuple` in `eval/run.py`.

The full design and test plan is in [PLAN.md](PLAN.md).

## Quick start

```bash
pip install -r requirements.txt          # includes numba

# 1. Train (about 30 s for 10k games on a laptop; checkpoints go to --out)
python -u -m agents.ntuple.train --games 10000 --alpha 0.1 --seed 10000000 \
    --out weights/ntuple/run1 --checkpoint-every 10000 --log-every 1000

# 2. Benchmark it on the shared seeds
python -m eval.run --agent ntuple --games 100 --seed0 0 --weights weights/ntuple/run1/latest.ntw
```

Long overnight run (Ctrl+C saves a checkpoint; rerun the same command to continue):

```bash
python -u -m agents.ntuple.train --games 1000000 --alpha 0.1 --seed 10000000 \
    --out weights/ntuple/run1 --checkpoint-every 10000 --log-every 1000 \
    --resume weights/ntuple/run1/latest.ntw
```

Leave out `--resume` the first time. Training writes `train_log.csv` in `--out`, one row
per `--log-every` games, with the mean score, max score, mean moves, reach counts and rates
for 2048–32768, elapsed seconds and games/sec. That log is for learning curves only. Numbers
for the report come from `eval.run`.

## Which weights file is used

The first match wins:

1. `--weights PATH` on `eval.run`, or `NTupleEvaluator("PATH")` in code
2. the `NTUPLE_WEIGHTS` environment variable
3. `weights/ntuple/latest.ntw` in the repo

If none exists, the agent refuses to start and prints how to get weights. `weights/` is
git-ignored. Trained weights are shared through GitHub Releases. Download the `.ntw` file
and use any of the three options above.

## For P3 / P7: using the evaluator

```python
from agents.ntuple.agent import NTupleEvaluator

ev = NTupleEvaluator("weights/ntuple/run1/latest.ntw")   # or NTupleEvaluator() to use $NTUPLE_WEIGHTS
value = ev.evaluate(board)    # board = engine 4x4 list of tile values; returns float
```

- **What the value means.** `V(board)` estimates the **score still to come** from this
  board, treating it as an **afterstate**: the board right after a slide, before the random
  tile appears. The network was trained on exactly that kind of board. So in expectimax,
  evaluate the board after the player's move and add the move's merge reward:
  `reward + ev.evaluate(afterstate)`.
- **Pure and read-only.** It's safe to call from many places. Evaluators loading the same
  file share one 268 MB copy of the tables.
- **Speed.** About 4 µs per call on an engine board, mostly the list-to-array conversion,
  and about 0.7 µs on a flat `uint8[16]` exponent array, which `evaluate` also accepts.
  For search in numba or C++, read `ev.network.weights` (`float32[S, 4, 16**6]`, S = number
  of stages, 1 for plain weights), `ev.network.stage_exps` and the cell layout
  `agents.ntuple.network.CELLS` directly, and call
  `agents.ntuple.network.evaluate_staged(weights, stage_exps, exps)` inside your own jitted
  code. It picks the stage from the board's max tile.
- **Multi-stage weights** need nothing extra: `evaluate` picks the right stage network for
  each board. `ev.stage_of(board)` tells you which one it used.
- **Helpers.** `ev.afterstate_value(board, move)` returns `reward + V(afterstate)`, or `None`
  if the move is illegal. `NTupleAgent.move_values(game)` returns that for every legal move.
- **65,536.** Boards holding 65,536 evaluate fine. The table index caps exponents at 15, so
  65,536 is seen as 32,768.

## Network

- **Board:** 16 log2 exponents, row-major. 0 = empty, 1 = 2, …, 16 = 65,536.
- **Tuples:** 4 six-cell tuples (Yeh et al. 2017), cells numbered 0–15 row-major:
  `{0,1,2,3,4,5}`, `{4,5,6,7,8,9}`, `{0,1,2,4,5,6}`, `{4,5,6,8,9,10}`.
- **Symmetries:** each tuple is read in all 8 rotations and reflections, and the 8 share
  one table. That makes 32 lookups per evaluation, and every symmetric version of a board
  gets the same value.
- **Index:** the 6 exponents, each capped at 15, written as a 6-digit hex number.
- **Memory:** 16⁶ entries × 4 bytes = 67 MB per table, **268 MB** in total.
- **Policy:** the agent picks the legal move that maximises `reward + V(afterstate)`. Ties go
  to the lowest move number, so play is deterministic.

## Training (TD(0) on afterstates)

Play a whole game with the current policy and record every (afterstate, reward). Then walk
the afterstates backwards:

- the last afterstate's target is 0 (the game ended after it)
- afterstate *t*'s target is `reward[t+1] + V(afterstate[t+1])`
- update with `V += alpha * (target - V)`, spread as `alpha / 32` of the error over each of
  the 32 weights it reads

Default `alpha = 0.1`. Training uses its own numba board (`board.py`, tested against
`engine.game.Game.slide`) and its own seeded random generator, starting at seed 10,000,000.

### Improvements (flags)

| Flag | Idea | Details |
|------|------|---------|
| `--stages 2 --stage-tile 16384` | Multi-stage TD (Yeh et al. 2017) | A separate network for boards whose max tile is ≥ 16,384. Each afterstate is valued and updated by its own stage's network, including the TD target across the boundary. Stage 2 is warm-started (copied from stage 1) the first time a game reaches it. More stages: `--stages 3 --stage-tile 8192 16384`. Memory: 268 MB per stage. |
| `--init-value V` | Optimistic initialisation (Guei et al. 2022) | Every board starts valued at V (each weight = V/32), so unvisited positions look attractive and get explored. Default 0. |

Compare them on equal terms with `scripts/ntuple_compare.sh`. It trains plain, multistage,
optimistic and both with the same games, seed and alpha (default 50,000 games), benchmarks
each with `eval.run`, and prints a table. Use `GAMES=2000 EVAL_GAMES=100` for a quick check.

## Weights file format (`.ntw`)

| bytes | content |
|-------|---------|
| 4 | magic `NTUP` |
| 4 | header length *n*, uint32 little-endian |
| *n* | JSON header, space-padded so the weights start on a 64-byte boundary |
| S × 268,435,456 | weights, little-endian float32, shape `[S, 4, 16**6]`, row-major (S = stages) |

The header holds `format`, `version` (currently 2), `dtype`, `shape`, `stage_tiles`
(e.g. `[16384]`), `tuples`, `symmetries`, `max_exponent`, and `meta`. `meta` is the training
info: alpha, seed, games, elapsed seconds, init value, which stages have started, and the
random-generator state used for `--resume`. Version-1 files (single stage, shape
`[4, 16**6]`) still load. `StagedNetwork.load` reads any file. `NTupleNetwork.load` reads
single-stage files only. Both reject files with the wrong magic, version or tuple layout. The raw
floats can be read from C++ with one `fread` after skipping the header.

## Files

| File | Contents |
|------|----------|
| `board.py` | numba board core: slide, legal moves, spawn, symmetries |
| `network.py` | `NTupleNetwork` (one stage), `StagedNetwork` (one or more), kernels, save/load |
| `train.py` | self-play TD(0) training CLI |
| `agent.py` | `NTupleEvaluator`, `NTupleAgent`, weights lookup |
| `_jit.py` | numba import with a pure-Python fallback |
| `../../scripts/ntuple_compare.sh` | trains and benchmarks plain / multistage / optimistic / both |

Tests: `tests/test_ntuple_*.py`. `tests/conftest.py` points `NTUPLE_WEIGHTS` at an
all-zero weights file, so the shared agent tests run in CI without trained weights.

## Results so far

| Run | Command | Mean score | Reach 2048 | Reach 4096 |
|-----|---------|-----------:|-----------:|-----------:|
| 10k training games | `python -m eval.run --agent ntuple --games 100 --seed0 0 --weights weights/ntuple/run1/latest.ntw` | 24,353 | 48% | 4% |
| greedy baseline | `python -m eval.run --agent greedy --games 100 --seed0 0` | 3,297 | 0% | 0% |

These are early numbers from the dev seeds; the target is ≥ 80% reach-2048 after the long
run. The commit, command and machine for each reported number go in the PR.

## TODO
- [x] 4 x 6-tuple network with all 8 board symmetries (rotations + reflections)
- [x] TD(0) afterstate learning
- [x] Save/load weights; register a greedy-on-value agent in `eval/run.py`
- [ ] Long training run; publish weights on GitHub Releases
- [ ] Learning curves (score and reach rates vs training games) with P8's plots
- [x] Multi-stage weights (`--stages`, `--stage-tile`)
- [x] Optimistic initialisation (`--init-value`)
- [ ] Overnight comparison run (`scripts/ntuple_compare.sh`) and its results table
