# P5 plan: n-tuple network trained by TD(0)

Status: plan only, no implementation yet. Branch: `p5-ntuple`.

## Goal

Build a value function `V(board)` that learns, from self-play, how good a 2048
position is. It has two users:

- **`NTupleEvaluator.evaluate(board) -> float`**: implements `agents.base.ValueFunction`.
  P3 (expectimax) and P7 (hybrid) call it at their search leaves.
- **`NTupleAgent.choose_move(game) -> int`**: implements `agents.base.Agent`. It plays
  1-ply greedy: it picks the legal move with the highest `reward + V(afterstate)`.

Done when (task brief): after training, the 1-ply agent reaches 2048 in at least 80% of
shared-seed games, and P7 can load the weights.

## Design

### Board encoding

- Inside this package a board is a flat `np.uint8[16]` of **log2 exponents**, row-major
  (cell `4*r + c`). `0` = empty, `1` = 2, ..., `15` = 32,768, `16` = 65,536.
- The public API takes the shared board format (`list[list[int]]` of tile values) and
  converts it with `to_exponents(board)`.
- Training games keep the true exponent, so 32,768 + 32,768 really becomes 16.
- **Clamp at 15 only when building a table index**: `min(e, 15)`. Every index digit then
  fits in 4 bits (16 values), so a board with a 65,536 tile still evaluates. The cost is
  that the network sees 65,536 as if it were 32,768. That is acceptable: such boards are
  extremely rare in training, and P7's search handles the late game.

### Tuples (network shape)

The 4 base six-tuples, with cells numbered 0-15 row-major:

```
T0 = {0,1,2,3,4,5}     T1 = {4,5,6,7,8,9}     T2 = {0,1,2,4,5,6}     T3 = {4,5,6,8,9,10}

 0  1  2  3            .  .  .  .             0  1  2  .             .  .  .  .
 4  5  .  .            4  5  6  7             4  5  6  .             4  5  6  .
 .  .  .  .            8  9  .  .             .  .  .  .             8  9 10  .
 .  .  .  .            .  .  .  .             .  .  .  .             .  .  .  .
```

This is the layout from Yeh et al. 2017 (*Multistage Temporal Difference Learning for
2048-like Games*, IEEE TCIAIG). Check it against the paper's figure before citing.

- **Symmetries**: each base tuple is applied in all 8 orientations of the board (4
  rotations x optional mirror). An orientation is a permutation of the 16 cells. All 8
  orientations of a tuple **share one table**, giving 4 x 8 = **32 lookups per
  evaluation**. Because the tables are shared, `V` gives the same value for all 8
  symmetric versions of a board, and every update teaches all 8 orientations at once.
- Precomputed once: `CELLS: np.int64[4, 8, 6]`, where `CELLS[t, s, k]` is the board cell
  for digit `k` of tuple `t` in orientation `s`.

### Lookup tables (data layout)

- `W: np.float32[4, 16**6]`, one C-contiguous array with one row per base tuple.
- Index of tuple `t`, orientation `s`: `idx = sum_k min(b[CELLS[t,s,k]], 15) * 16**(5-k)`.
  This is a 24-bit number, 4 bits per cell.
- `V(b) = sum over t, s of W[t, idx(t, s, b)]`.
- **Memory**: 16^6 = 16,777,216 entries x 4 bytes = **67.1 MB per tuple, 268 MB in total**.
  Multi-stage training (below) needs one copy per stage, so 2 stages = 537 MB.
  `np.zeros` reserves the memory but the OS only commits pages that get touched, so an
  untrained agent (for example in CI) costs very little RAM.

### TD(0) on afterstates

The afterstate is the board after the slide and before the random spawn. Learning `V` of
afterstates means the player never has to average over spawns to choose a move. Per the
task brief:

1. In state `s`, choose `a = argmax over legal moves of r(s,a) + V(afterstate(s,a))`.
   Ties go to the lowest move number, so play is deterministic.
2. The game spawns a tile, which gives the next state `s''`.
3. Choose the next move the same way, with reward `r'` and afterstate `s'_next`.
4. Update the previous afterstate: `delta = r' + V(s'_next) - V(s')`, then add
   `(alpha / 32) * delta` to each of the 32 weights `s'` uses.
5. If `s''` has no legal move (game over), the target is 0: `delta = 0 - V(s')`.

Default `alpha = 0.1`, spread over the 32 lookups as the brief says.

### Speed: numba

The hot loops are written as `@njit` functions over the flat exponent array: slide,
legality check, spawn, evaluate, TD update, and one whole self-play episode. Python only
loops over games, logging and checkpoints.

- Training does not use the shared `engine.game.Game`. It is pure Python and far too slow
  for 100k+ games, and P1's fast engine isn't ready. The numba slide is **tested against
  `Game.slide`** so the two cannot disagree. When P1's engine lands we can switch to it if
  it is faster.
- Training spawns use numba's own seeded RNG. Training games don't have to match the
  engine's RNG; they only have to be reproducible from the training seed.
- **numba is optional at import time.** If it isn't installed, a no-op `njit` decorator
  runs the same functions as plain Python. That is slow but correct, and it is all that
  `evaluate` and the agent need. numba is in the top-level `requirements.txt`, so CI
  installs it and runs the jitted path. The fallback is there so that a teammate (P3, P7)
  can still import the evaluator if numba fails to install on their machine.

## Modules

| File | Contents |
|------|----------|
| `requirements.txt` (top level) | `numba>=0.60` (see "Differences from the repo rules") |
| `agents/ntuple/_jit.py` | `njit` import with the pure-Python fallback |
| `agents/ntuple/tuples.py` | `BASE_TUPLES`, the 8 symmetry permutations, building `CELLS` (pure numpy) |
| `agents/ntuple/board.py` | `to_exponents(board)` (shared board to `uint8[16]`); jitted `slide`, `is_legal`, `spawn` on exponent arrays |
| `agents/ntuple/kernels.py` | jitted `tuple_index`, `evaluate_exps`, `td_update`, `best_move`, `play_episode_td` |
| `agents/ntuple/network.py` | `NTupleNetwork` (holds `W`, stages, metadata; `save`/`load`); `NTupleEvaluator(ValueFunction)` with `evaluate(board)` |
| `agents/ntuple/agent.py` | `NTupleAgent(Agent)`: 1-ply greedy on `reward + V(afterstate)`; never mutates `game` |
| `agents/ntuple/train.py` | CLI: `--games --alpha --seed --stages --optimistic --checkpoint-every --out`; CSV log every 1,000 games |
| `agents/ntuple/__init__.py` | exports `NTupleEvaluator`, `NTupleAgent`, `NTupleNetwork` |
| `eval/run.py` | one line: register `"ntuple"` in `AGENTS` (as the brief and AGENTS.md require) |
| `tests/test_ntuple.py` | tests below (`tests/` is the only path pytest/CI collects) |

### Weights

- Saved with `np.savez_compressed`: `weights` (`float32[stages, 4, 16**6]`), `tuples`
  (`int64[4, 6]`), and `meta` (a JSON string with the format version, alpha, games trained,
  seed, stage thresholds and git commit). `load` rejects files whose tuple layout or version
  doesn't match.
- Written to `weights/ntuple/` (already git-ignored). Trained weights are published on
  GitHub Releases.
- The registered `ntuple` agent loads `$NTUPLE_WEIGHTS`, or else `weights/ntuple/latest.npz`.
  If neither exists it uses zero weights and prints a warning. The agent still plays legal
  moves, so the shared tests pass without a weights file.

### Training seeds

The shared evaluation seeds are dev 0-999 and test 100000-100999 (brief; `eval/README.md`
hasn't fixed them yet). Training seeds start at **10,000,000** so they never overlap either
range, as CONTRIBUTING.md requires.

## Build order (each step is a small PR or commit)

1. `tuples.py` + `board.py` + their tests (encoding, symmetries, slide matches the engine).
2. `kernels.py` + `network.py` + evaluate/update/save tests. `NTupleEvaluator` is usable
   by P3/P7 from this step (with zero or random weights) so they can integrate early.
3. `agent.py` + registration in `eval/run.py`.
4. `train.py`, then a ~10k-game run to show it learns, then the overnight command.
5. Improvements behind flags: multi-stage weights (switch when 16,384 first appears),
   optimistic initialisation (Guei et al. 2022), and optionally temporal coherence
   learning. Each one is compared against the plain run on the shared seeds.

## Testing

All tests go in `tests/test_ntuple.py` and run in CI with numba installed. Because numba
compiles each function on first call, the tests use small inputs so CI stays fast. One test
also runs the plain-Python fallback and checks that it gives the same values.

| What | How |
|------|-----|
| Exponent encoding | `to_exponents` on hand-made boards; 65,536 becomes 16; empty becomes 0 |
| Slide matches the engine | 2,000 random boards x 4 moves: numba `slide` (board + reward) equals `Game.slide`, including `[2,2,2,2]`, `[4,2,2,0]` and 32,768+32,768 -> 65,536 |
| Symmetry permutations | the 8 permutations are distinct, each is a bijection on 0..15, and they match `np.rot90`/`np.fliplr` on a labelled board |
| Index formula | a hand-made board gives a hand-computed index for one tuple and orientation |
| Clamp | a board with 65,536 evaluates without error and equals the same board with 32,768 in that cell |
| Symmetry invariance | random weights: `evaluate` gives the same value on all 8 transforms of random boards (`rtol=1e-5`, because float sums are added in a different order) |
| Reference check | `evaluate_exps` equals a slow, explicit Python loop over the 32 lookups |
| TD update | after one update toward a target, `abs(V - target)` shrinks by exactly `alpha * abs(delta)` when the 32 lookups don't collide; a terminal update moves `V` toward 0 |
| Save/load | round trip gives byte-identical `W` and identical `evaluate` results; a wrong format version is rejected |
| Memory layout | `W.dtype == float32`, `W.shape == (4, 16**6)`, `W.nbytes == 268,435,456` |
| Agent | covered by the existing legal-move / no-mutation / reproducibility tests once it is in `AGENTS`; plus: on a hand-made board where one move merges two 1,024s, a network trained briefly picks it |
| Training smoke | `train.py --games 50` runs, writes a checkpoint and a CSV with the expected columns |
| Learns (manual) | ~10k-game run: mean score over the last 1,000 games is clearly higher than over the first 1,000 (reported from `eval.run`, not from the training log) |

## Differences from the repo rules

- **numba is in the top-level `requirements.txt`** (P5's decision). AGENTS.md and
  CONTRIBUTING.md ask for heavy dependencies to live in per-folder files to keep CI fast.
  The trade-off: everyone and CI now install numba + llvmlite (about 30-40 MB of wheels),
  in exchange for P3 and P7 getting a fast evaluator with no extra install step. Flag this
  in the PR description so the review buddy (P6) and the group can agree to it.
- Tests go in `tests/test_ntuple.py`, because `pyproject.toml` only collects `tests/`.
- Seed ranges: the brief says test = 100000-100999, while `eval/README.md` gives
  10000-10999 only as an example. P8 should fix this. Training seeds (>= 10,000,000) are
  clear of both.
