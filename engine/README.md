# engine — P1 (Engine)

**Owner:** P1 · **Review buddy:** P2 · **Label:** `P1-engine`

`game.py` is the engine every agent and the benchmark runs on. Its public API is
shared by everyone, so don't change it without telling the group:

```python
Game(seed); game.reset(seed); game.step(move) -> (board, reward, done)
game.legal_moves(); game.max_tile(); Game.slide(board, move) -> (new_board, gained)
```

Attributes: `board`, `score`, `moves`, `seed`, `done`. Moves: 0=up, 1=right,
2=down, 3=left. Spawn: one uniformly random empty cell gets 2 (p=0.9) or 4 (p=0.1).

## Status

**Batch 1 (pure-Python engine, done):** 5-bit exponent row table inside
`Game.slide`, 65,536 support, tests in `tests/test_engine_p1.py`.
**Batch 2 (Python helpers, done):** `Game.from_board`, `afterstate`, `empty_cells`,
`spawn_outcomes`, `exponents`, `play_batch`; tests in `tests/test_engine_helpers.py`.
**Batch 3 (C++ / pybind11, not started).**

## Representation (and why)

* **Public board = 4x4 nested list of tile values** (0, 2, 4, ...). Every agent
  reads `game.board`, and the agent tests compare it directly, so this does not change.
* **Inside `slide`, cells are 5-bit exponents**: 0 = empty, k = tile 2**k. A row is
  4 cells x 5 bits = a 20-bit key (cell 0 in the lowest bits). 5 bits hold exponents
  0..31. A 4-bit cell stops at 2**15 = 32,768, so 32,768 + 32,768 = 2**16 = 65,536
  would not fit; with 5 bits it does, and so does 131,072 (exponent 17).
* **Row table:** `key -> (slid row key, score gained)` for a left move. Right, up
  and down reuse it by reversing / transposing lines (`_get_lines` / `_set_lines`).
  Score for a merge of two exponent-e tiles is `2**(e+1)`, the same as the reference.
* **Lazy table (trade-off):** the full table has 2**20 rows. Filling it in Python would
  cost seconds at import, and games only visit a small fraction of rows, so entries are
  computed on first use and cached in a dict. The C++ core will precompute all of it.
* **Fallback (trade-off):** if any cell is not 0 or a power of two in 2..2**31, `slide`
  uses the original `_merge_line`, so odd hand-made boards behave exactly as before.
* **Speed:** Batch 1 optimises correctness, not speed. No speed numbers are claimed;
  the micro-benchmark (`engine/bench.py`) comes with Batch 3.

## Helpers (Batch 2)

All are plain Python on top of the unchanged engine. `from engine.game import ...`.

| Helper | What it does | For |
|---|---|---|
| `Game.from_board(board, seed=0)` | New game from any position. Board is copied, nothing spawns, RNG seeded with `seed`. Wrong shape -> `ValueError`. | P3, P8 late-game benchmark |
| `afterstate(board, move)` | Board after sliding, **before** the spawn. Pure. An illegal move returns an equal copy (check `Game.is_legal`). Need the score? `Game.slide` returns it too. | P5 |
| `empty_cells(board)` | `[(row, col), ...]` of zeros, row-major, same order as `Game.empty_cells()`. | P3, P7 |
| `spawn_outcomes(board)` | `[(board, prob), ...]`: for each empty cell a "2" board (0.9/n) then a "4" board (0.1/n). Probabilities sum to 1. Full board -> `[]`. | P3, P7 |
| `exponents(board)` | 4x4 NumPy `int32` array of log2 values (0 empty, 65536 -> 16). Bad cell -> `ValueError`. numpy is imported only when this is called. | P5, P6 |
| `play_batch(agent_fn, seeds, processes=1)` | One full game per seed; `agent_fn(game) -> move`. Returns dicts `{seed, score, max_tile, moves, board}` in seed order. `processes>1` uses multiprocessing (`agent_fn` must be picklable). An illegal move raises instead of looping forever. | P5, P6 |

Design notes: `exponents` uses `int32` so index maths like `e0 + 32*e1 + 1024*e2` cannot
wrap around (int8 would). `play_batch` returns the move list so any game can be replayed
with `Game(seed)`. It is a plain-Python version; the fast parallel C++ version comes in Batch 3
with the same signature.

## Compatibility guarantees

* `_spawn`, `reset`, `step`, `legal_moves` and the RNG call order (`randrange(len(empties))`
  for the cell, then `random()` for the value) are untouched, so a seed plus a move list
  gives identical games to the old engine (checked in `test_engine_matches_original_rules_move_for_move`).
* Illegal moves stay no-ops: reward 0, no spawn, no RNG use, not recorded.
* The original helpers `_merge_line`, `_get_lines`, `_set_lines` are kept as the oracle.

## Tests

`tests/test_engine_p1.py` (Batch 1; it does not replace `tests/test_engine.py`): all four
directions, no double merge (`[2,2,2,2] -> [4,4,0,0]`), scoring, illegal moves,
invalid move values, 32,768 + 32,768 -> 65,536 in every direction, the row table
against `_merge_line` for all 17^4 rows (tiles up to 65,536), 3,000 random boards,
original-rules oracle game vs `Game`, seed repeat / reset / replay, 90/10 spawn ratio.
`tests/test_engine_helpers.py` (Batch 2) covers every helper in the table above.

## TODO
- [x] 5-bit exponent representation, 65,536 handled explicitly (Python)
- [ ] 64-bit-style bitboard (16 cells x 5-bit exponents) with row lookup tables for all 4 moves in C++
- [ ] Same public API as `game.py`; `tests/test_engine.py` must pass unchanged
- [ ] Target ≥ 1M moves/sec; add a micro-benchmark script
- [ ] Python bindings (pybind11) for the C/C++ core, Python fallback when not built
- [ ] Differential test: C++ engine vs Python engine on random games
- [x] Helpers (Python): `Game.from_board`, `afterstate`, `empty_cells`, `spawn_outcomes`, `exponents`, `play_batch`
- [ ] Fast C++ `play_batch`
- [ ] Build docs (Windows / macOS / Linux) and CI
