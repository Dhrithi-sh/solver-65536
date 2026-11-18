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

Training writes `train_log.csv` in `--out`, one row per `--log-every` games, with the mean
score, max score, mean moves, reach counts and rates for 2048–32768, elapsed seconds and
games/sec. That log is for learning curves only. Numbers for the report come from `eval.run`.

## Running the real experiments

### Resources (measured on this machine)

Machine: AMD Ryzen laptop, 12 threads, Windows 11, Python 3.14, numba 0.68.1. Training is
single-threaded.

- **Memory:** about **270 MB per stage** (268 MB of tables). Plain = 1 stage ≈ 270 MB;
  `--stages 2` ≈ 540 MB. Add about 10 MB for buffers. Checkpoints are the same size on disk.
- **Speed:** about **400,000 moves/sec**. Games/sec falls as the agent improves, because
  better games last longer:

  | point in training | games/sec | mean moves per game |
  |---|---|---|
  | first 1,000 games | ~600 | ~530 |
  | around 10,000 games | ~330 | ~1,230 |
  | around 50,000 games | ~160 | ~2,100 |

  50,000 games took 221 s for plain (157–221 s across the four configurations).
- **Estimate for a long run:** roughly **1–3 hours for 500,000 games** and 3–6 hours for
  1,000,000. This is an extrapolation: it depends on how long games get, so watch the
  `games_per_s` column.
- **Benchmarking:** 1,000 games with `eval.run` took about 2.5 minutes per configuration.

### 1. Long overnight run (best configuration: plain)

In the 50k-game comparison below, the plain configuration was best, so the long run uses
no `--stages` and no `--init-value`. Run it from the repo root:

```bash
python -u -m agents.ntuple.train --games 1000000 --alpha 0.1 --seed 10000000 \
    --out weights/ntuple/long --checkpoint-every 10000 --log-every 1000 2>&1 | tee -a weights/ntuple/long.txt
```

- **Interrupted, crashed or rebooted:** Ctrl+C saves a checkpoint first. Continue with the
  same command plus `--resume`; the result is identical to an uninterrupted run:

  ```bash
  python -u -m agents.ntuple.train --games 1000000 --alpha 0.1 --seed 10000000 \
      --out weights/ntuple/long --checkpoint-every 10000 --log-every 1000 \
      --resume weights/ntuple/long/latest.ntw 2>&1 | tee -a weights/ntuple/long.txt
  ```
- **Starting from an existing run:** add `--resume weights/ntuple/compare/plain/latest.ntw`
  to the first command. That continues the 50k-game plain run instead of starting over (same
  seed and RNG stream).
- **Checkpoints:** only `latest.ntw` is kept, overwritten every 10,000 games. With
  `--keep-checkpoints` each one is another 268 MB, so use something like
  `--checkpoint-every 100000`.
- **Progress:** in a second terminal, `tail -f weights/ntuple/long.txt`. Plot the curve any
  time with `python scripts/plot_ntuple.py weights/ntuple/long`.

### 2. Benchmark on the shared seeds

Tune and compare on the **dev seeds 0–999**. Use the **test seeds 100000–100999** only for
the final numbers in the report.

```bash
# dev set
python -m eval.run --agent ntuple --games 1000 --seed0 0 --weights weights/ntuple/long/latest.ntw

# final test set (once, at the end)
python -m eval.run --agent ntuple --games 1000 --seed0 100000 --weights weights/ntuple/long/latest.ntw \
    --out-dir results/test
```

Each run writes `results/<agent>.jsonl` (or the `--out-dir`). When you post numbers, give
the command, the commit (`git rev-parse --short HEAD`) and the machine (CONTRIBUTING.md).

### 3. Publish the weights (GitHub Release, not git)

`weights/` is git-ignored: never commit `.ntw` files. Give the file a descriptive name and
attach it to a release. GitHub allows up to 2 GB per file.

```bash
cp weights/ntuple/long/latest.ntw ntuple-plain-1M.ntw
sha256sum ntuple-plain-1M.ntw            # PowerShell: Get-FileHash ntuple-plain-1M.ntw

gh release create ntuple-v1 ntuple-plain-1M.ntw \
    --target main \
    --title "N-tuple weights v1 (plain, 1M games)" \
    --notes "1-ply n-tuple TD(0), 4x6-tuples x 8 symmetries, 1 stage, alpha 0.1, seed 10000000,
1,000,000 training games, trained at commit <hash>. sha256: <hash>.
Dev-seed benchmark (0-999): <paste eval.run summary>. Use: --weights ntuple-plain-1M.ntw"
```

Use `--target main` once this PR is merged; before that, use `--target p5-ntuple`.
Teammates download it with:

```bash
gh release download ntuple-v1 --pattern "*.ntw" --dir weights/ntuple
python -m eval.run --agent ntuple --games 100 --seed0 0 --weights weights/ntuple/ntuple-plain-1M.ntw
```

Alternatively, copy it to `weights/ntuple/latest.ntw`, or set `NTUPLE_WEIGHTS`, so no flag is
needed.

### 4. Plots

```bash
pip install matplotlib                     # not in requirements.txt (keeps CI light)
python scripts/plot_ntuple.py              # every run in weights/ntuple/compare
python scripts/plot_ntuple.py weights/ntuple/compare/plain weights/ntuple/long --label long=plain-1M
```

This writes `results/plots/ntuple_mean_score.png` (mean score vs games trained, one line per
run) and `results/plots/ntuple_reach.png` (2048 / 8192 / 16384 reach rate vs games trained).

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

The script splits stages at **2048** by default, not Yeh et al.'s 16,384. In 50,000 1-ply
training games almost no game reaches 16,384, so stage 2 would never be used and multistage
would equal plain (a 2,000-game check confirmed this). At 2048 most games cross the split, so
the comparison measures whether a separate later-game network helps at our training scale.
It does not show that the 16,384 split itself helps. Once long runs regularly reach
8,192–16,384, rerun with `STAGE_TILE=16384`.

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
| `../../scripts/plot_ntuple.py` | learning-curve PNGs from training CSVs |

Tests: `tests/test_ntuple_*.py`. `tests/conftest.py` points `NTUPLE_WEIGHTS` at an
all-zero weights file, so the shared agent tests run in CI without trained weights.

## Results so far

All rows: `eval.run` on dev seeds 0–999 (1,000 games) unless noted, same machine as above.

| Run | Training games | Mean score | Median | Reach 2048 | Reach 4096 | Reach 8192 |
|-----|---------------:|-----------:|-------:|-----------:|-----------:|-----------:|
| plain | 50,000 | **51,902** | 49,942 | **85.0%** | **50.7%** | **3.2%** |
| optimistic (V = 50,000) | 50,000 | 48,228 | 44,466 | 88.1% | 46.4% | 0.9% |
| multistage (split at 2048) | 50,000 | 37,342 | 36,358 | 80.5% | 18.5% | 0.1% |
| both | 50,000 | 31,959 | 32,526 | 82.6% | 5.7% | 0.0% |
| plain, seeds 0–99 only | 10,000 | 24,353 | 22,496 | 48% | 4% | 0% |
| greedy baseline, seeds 0–99 | — | 3,297 | 3,210 | 0% | 0% | 0% |

The four 50k rows come from `scripts/ntuple_compare.sh` (commit `4d613ad`; same seed and
alpha, 50,000 games each). Plain already meets the brief's ≥ 80% reach-2048 target on the
dev seeds.

How to read the comparison:
- **Optimistic init reaches 2048 slightly more often but scores less.** It's a different
  trade-off, not clearly better, at this length.
- **The 2048 multi-stage split hurts at 50k games.** Stage 2 starts as a copy of stage 1 and
  then learns only from the part of each game past 2048. Each network sees less data, and
  50k games isn't enough to make up for it. This tests a split at 2048, not Yeh et al.'s
  16,384 split.
- **Combining them is worst.**

Training curves: `python scripts/plot_ntuple.py`.

## TODO
- [x] 4 x 6-tuple network with all 8 board symmetries (rotations + reflections)
- [x] TD(0) afterstate learning
- [x] Save/load weights; register a greedy-on-value agent in `eval/run.py`
- [ ] Long training run; publish weights on GitHub Releases
- [x] Multi-stage weights (`--stages`, `--stage-tile`)
- [x] Optimistic initialisation (`--init-value`)
- [x] 50k-game comparison run (`scripts/ntuple_compare.sh`) and its results table
- [x] Learning-curve plots (`scripts/plot_ntuple.py`)
