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
**Batch 3a (C++ core, done):** `engine/_core/core.*`, standalone self-test.
**Batch 3b (pybind11 binding + build + CI):** `engine._cengine`, `setup.py`, CI on three OSes.
Verification status of 3b is in "Optional C++ extension" below.
**Batch 3c (fast batch play, CPython-exact RNG):** `engine/_core/rng.hpp`, `batch.*`, `engine/batch.py`,
deterministic `play_batch` option, `engine.bench` batch mode. See "Fast batch play (3c)" below.

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
  measure it with `python -m engine.bench` (see below).

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
with `Game(seed)`. It is a plain-Python version; the fast parallel C++ version comes in Batch 3c
with the same signature (3c keeps it and adds options; see below).

## Fast batch play (Batch 3c)

Two different jobs, two tools:

| Need | Use | Speed |
|---|---|---|
| Many **random-policy** games as fast as possible (benchmarks, the 1M moves/sec target, baselines) | `engine.batch.play_random_batch(seeds, board=None, threads=0, record_moves=True)` | C++, all cores, GIL released |
| Many games played by **any Python agent** (P5/P6 self-play) | `engine.game.play_batch(agent_fn, seeds, processes=N, deterministic=True)` | limited by the agent; scales with processes |

A Python `agent_fn` cannot run inside C++, so only the random policy is fast. This is a deliberate
trade-off: it covers the brief's "moves per second for random play" benchmark without changing the
`agent_fn(game) -> move` contract.

**How the C++ games stay identical to Python (the part to defend).** Spawns must come from
`random.Random(seed)`, so `engine/_core/rng.hpp` re-implements the pieces of CPython's `random` the engine
uses: the Mersenne Twister MT19937 with CPython's `init_by_array` seeding (`abs(seed)` split into 32-bit
words, minimum one word, so negative seeds equal positive ones), `getrandbits(k)` = top k bits of one
output, `random()` = `(a*2^26 + b)/2^53` from two outputs (`a = out>>5`, `b = out>>6`), and
`randrange(n)` = draw `getrandbits(n.bit_length())` until `< n` (rejection loop; n = 1 still draws).
`batch.cpp` then plays exactly `Game.reset`/`Game.step`: row-major empties, **cell first, then value**.
Seeds are accepted only as exact Python ints with `abs(seed) < 2^64`; anything else (numpy ints, floats,
huge ints) makes the extension answer `None` and Python plays the game.

**The policy** is "uniform over the legal moves in order up, right, down, left", drawn from a *second*
generator `random.Random(abs(seed) ^ 0x2545F4914F6CDD1D)` so picking moves never shifts the spawn sequence
(same seed = same spawns for every agent). It exists in C++ and in plain Python (`RandomLegalPolicy`);
the tests prove both play the same games.

Return value: one dict per seed, in seed order, like `play_batch`:
`{seed, score, max_tile, n_moves, moves (list, or None with record_moves=False), board}`. With `board=...`
every game starts from that position (late-game benchmark); no tiles spawn at the start. Results never depend
on `threads`. If a hand-made start board chains up to a 2^19 tile (outside the C++ tables, which accept
exponents <= 18) the game is replayed by the Python engine, so the answer is still right.

**`play_batch` determinism.** The signature and default behaviour are unchanged. Two optional keyword
arguments were added: `deterministic=True` runs every game on a fresh deep copy of `agent_fn` and calls
`reset(seed)` first when `agent_fn` is a bound method of an `Agent` (as `eval/run.py` does), so a game depends only on
`(agent, seed)` and results are identical for any `processes`. Without it, an agent that keeps state (RNG,
cache) gives results that depend on the process count and on how the pool chunks the seeds. `start_method`
selects `fork`/`spawn`/`forkserver`; Windows and macOS default to `spawn`, where `agent_fn` must be importable
and the script needs an `if __name__ == "__main__":` guard.

### Speed (measured, not promised)

`python -m engine.bench --seconds 3` times both loops end to end (spawn, RNG, legal check and move choice
all inside the clock). Numbers from the development container (x86_64, **4 vCPU**, Python 3.13 unless noted;
a laptop will differ):

| Measurement | moves/sec |
|---|---|
| Python engine, `Game.step()` loop (`ENGINE_PURE_PYTHON=1`) | ~11,800 |
| C++ core behind `Game.step()` (Python spawn + RNG) | ~190,000 |
| `play_random_batch`, pure Python fallback | ~12,800 |
| `play_random_batch`, C++, 1 thread | ~3,600,000 |
| `play_random_batch`, C++, 4 threads | ~11,900,000 (Python 3.11: ~9,800,000) |

Windows 11 laptop (AMD64, 14 logical CPUs, Python 3.11.9, MSVC), same command, single 3 s runs:
`Game.step()` ~5,800 (pure Python) / ~103,000 (C++); `play_random_batch` ~5,100 (pure Python) /
~2,270,000 (C++, 1 thread) / ~18,900,000 (C++, all 14 threads).

So the 1M moves/sec target is met by the C++ batch path (including a single thread), and is **not** met by
`Game.step()`, which is bound by Python's per-call overhead. Re-measure on your own machine before quoting.

## Optional C++ extension (`engine._cengine`)

`engine/game.py` imports the compiled module if it exists and falls back to the pure-Python
engine otherwise, so **nothing below is required to use the engine**. The extension only speeds up
`Game.slide` / `Game.legal_moves`; spawning and the seeded `random.Random` always stay in Python, so
games are identical with or without it. It answers `None` for anything it does not support (not a
4x4 list/tuple of exact ints, a cell that is not 0 or a power of two up to 2^18, a bad move) and the
Python code then runs unchanged.

Switches: `ENGINE_PURE_PYTHON=1` ignores the extension; `python -c "from engine import game; print(game.backend())"`
prints `cpp` or `python`.

### Build (always run from the repo root, inside your virtual environment)

You need Python 3.11, a C++17 compiler, and `pybind11` + `setuptools` (build-time only; not runtime dependencies).
The build is *optional*: if the compiler is missing or the compile fails, setuptools only prints a warning,
so always check `backend()` afterwards.

**Windows 10/11 (PowerShell).** Install "Build Tools for Visual Studio 2022" and tick the workload
**Desktop development with C++** (MSVC v143 + a Windows SDK). Then:
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install "setuptools>=64" wheel "pybind11>=2.11"
python setup.py build_ext --inplace
python -c "from engine import game; print(game.backend())"     # expect: cpp
python -m pytest
python -m engine.bench --seconds 3                              # C++ backend
$env:ENGINE_PURE_PYTHON = "1"; python -m engine.bench --seconds 3; Remove-Item Env:ENGINE_PURE_PYTHON
```
In `cmd.exe` use `.venv\Scripts\activate.bat` and `set ENGINE_PURE_PYTHON=1` / `set ENGINE_PURE_PYTHON=`.
Run the build from the "x64 Native Tools Command Prompt" only if setuptools cannot find the compiler by itself.

**macOS.** `xcode-select --install`, then:
```bash
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install "setuptools>=64" wheel "pybind11>=2.11"
python setup.py build_ext --inplace
python -c "from engine import game; print(game.backend())"     # expect: cpp
python -m pytest && python -m engine.bench --seconds 3
ENGINE_PURE_PYTHON=1 python -m engine.bench --seconds 3
```

**Linux (Debian/Ubuntu).** `sudo apt install build-essential python3.11-dev python3.11-venv`, then the same commands as macOS.

After editing any C++ file rebuild with `python setup.py build_ext --inplace --force`. To go back to pure
Python delete `engine/_cengine*.so` / `engine/_cengine*.pyd` and the `build/` folder (do not commit them).

### Verification status (honest record)

* Python integration, fallback, the `None` contract, and C++-vs-Python equality were tested through a
  stand-in module built from the same `pyglue.hpp` + `core.cpp` on Linux (Python 3.12).
* `engine/_core/binding.cpp` (the pybind11 wrapper) and the Windows/macOS builds had **not** been
  compiled by the author when this was written. CI (`.github/workflows/ci.yml`) is what builds and tests them
  on Ubuntu, Windows and macOS; update this section with real results once it has run.
* Batch 3c (verified on Linux x86_64, g++, Python 3.13 and 3.11): full suite passes with and without the
  extension (`ENGINE_PURE_PYTHON=1`), including the 10,000-game parity test. The pybind11 file `binding.cpp`
  now compiles under g++/Linux only; the MSVC/clang builds of 3b and 3c still rely on CI.
* Batch 3c on **Windows 11, Visual Studio Build Tools 18 (MSVC 14.52), Python 3.11.9**, verified locally: the extension builds
  with `python setup.py build_ext --inplace` and `backend()` reports `cpp`; `pytest` passes in full with
  `ENGINE_REQUIRE_CPP=1` (130 passed, incl. the 10,000-game parity test) and with `ENGINE_PURE_PYTHON=1`
  (102 passed, 28 C++-only tests skipped); the standalone `selftest.cpp` compiles under `cl /W4` without
  warnings and passes against `gen_golden.py`. `eval.run` logs are identical across runs and backends and replay
  with `eval.replay`. macOS/clang still relies on CI.
* Speed: see "Speed (measured, not promised)" above. The 1M moves/sec target applies to the C++ batch loop,
  not to `Game.step()`.

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
`tests/test_engine_rng.py` (3c) compares the C++ generator with `random.Random`: seeding (whole 624-word
state, incl. negative/huge seeds), `getrandbits` widths 1..32, `random()` bit for bit, `randrange` and its output
consumption, and the state after long runs. `tests/test_engine_batch.py` (3c) holds the brief's **10,000 seeded
games** test (C++ batch vs the pure-Python engine in a `ENGINE_PURE_PYTHON=1` subprocess: scores, max tiles, move
lists and final boards equal; `ENGINE_PARITY_GAMES` lowers the count for quick runs, ~40 s at 10,000), replay
of C++ games on `Game`, thread-count independence, 32,768+32,768 start boards, out-of-range and unsupported
inputs, and `play_batch` determinism (1/2/3 processes, `spawn`).

## TODO
- [x] 5-bit exponent representation, 65,536 handled explicitly (Python)
- [x] Bitboard-style C++ core (16 cells x 5-bit exponents) with row lookup tables for all 4 moves
- [x] Same public API as `game.py` (extension used only behind it); `tests/test_engine.py` must pass unchanged
- [x] Target ≥ 1M moves/sec: measured ~3.6M (1 thread) / ~11.9M (4 threads) for `play_random_batch` on the dev container; `Game.step()` stays ~0.19M; benchmark: `engine/bench.py`
- [x] pybind11 binding `engine._cengine` + Python fallback (CI-verified status: see above)
- [x] Differential tests: C++ vs Python (`tests/test_engine_cengine.py`)
- [x] Helpers (Python): `Game.from_board`, `afterstate`, `empty_cells`, `spawn_outcomes`, `exponents`, `play_batch`
- [x] Fast C++ random self-play (`engine.batch.play_random_batch`) + deterministic parallel `play_batch` for arbitrary Python agents
- [x] Build docs and CI (see above)
