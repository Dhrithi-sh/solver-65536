# P1 Engine — Testing and Benchmarks

**Snapshot:** commit `d8379a0`. "Verified locally" means run on the Windows machine described in section 7 during
the handover work; "CI-verified" means reported as passed by GitHub Actions.

## 1. Environment requirements

| Need | Version | Where it is declared |
|---|---|---|
| Python | **3.11** (CI uses 3.11; `requires-python = ">=3.11"`) | `pyproject.toml`, `.github/workflows/ci.yml` |
| Runtime packages | `numpy>=1.26`, `pytest>=8.0` | `requirements.txt` |
| Build-only (optional C++ extension) | `setuptools>=64`, `wheel`, `pybind11>=2.11`, a C++17 compiler | `pyproject.toml` `[build-system]`, `setup.py` |

The compiled module is tied to the Python version that built it (e.g. `engine/_cengine.cp311-win_amd64.pyd` only
loads in CPython 3.11 on 64-bit Windows). Run everything with the interpreter of the virtual environment you built
in. On the handover machine `python` on `PATH` is 3.13 without pytest, so `.venv\Scripts\python.exe` (3.11.9) must
be used. Packages in that venv at handover: numpy 2.4.6, pytest 9.2.0, pybind11 3.1.0, setuptools 65.5.0.

## 2. Python 3.11 setup (all OSes)

```bash
# Windows PowerShell: py -3.11 -m venv .venv ; .venv\Scripts\Activate.ps1
# macOS / Linux:      python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Without a compiler you can stop here: the engine runs in pure Python and the C++-only tests skip themselves.

## 3. Building the C++ extension

Always from the repo root, inside the venv:

```bash
pip install "setuptools>=64" wheel "pybind11>=2.11"
python setup.py build_ext --inplace
python -c "from engine import game; print(game.backend())"     # must print: cpp
```

The build is optional (`optional=True` in `setup.py`): a missing compiler or failed compile only prints a warning,
so the `backend()` check is mandatory. After changing any `.cpp`/`.hpp`: `python setup.py build_ext --inplace --force`.
To return to pure Python: delete `engine/_cengine*.pyd` / `engine/_cengine*.so` and `build/` (never commit them).

**Windows (MSVC).** Install Visual Studio Build Tools with the **Desktop development with C++** workload, then run
the commands above in PowerShell. `engine/README.md` names "Build Tools for Visual Studio 2022"; the build that was
actually verified used **Visual Studio Build Tools 18 (MSVC 14.52.37025)** with Python 3.11.9. VS 2022 itself was not
tried by us (CI's `windows-latest` image builds successfully; its exact MSVC version is not recorded here).
`setup.py` adds `/EHsc` on Windows. If setuptools cannot find the compiler, build from the "x64 Native Tools Command
Prompt".

**macOS.** `xcode-select --install`, then the same commands. CI-verified on `macos-latest`; **not tested on a local Mac.**

**Linux (Debian/Ubuntu).** `sudo apt install build-essential python3.11-dev python3.11-venv`, then the same commands.
CI-verified on `ubuntu-latest`; the earlier development container (Linux x86_64, g++) also passed per `engine/README.md`.

## 4. Running the tests

```bash
# Full suite with the C++ extension, failing loudly if it is missing (what CI does):
ENGINE_REQUIRE_CPP=1 python -m pytest

# Same tests on the pure-Python engine (extension present but ignored):
ENGINE_PURE_PYTHON=1 ENGINE_REQUIRE_CPP=0 python -m pytest
```

PowerShell: `$env:ENGINE_REQUIRE_CPP = "1"; python -m pytest; Remove-Item Env:ENGINE_REQUIRE_CPP`, and likewise for
`ENGINE_PURE_PYTHON`. `pyproject.toml` already sets `-q` and `testpaths = ["tests"]` (adding another `-q` hides the
summary line).

| Variable | Effect |
|---|---|
| `ENGINE_PURE_PYTHON` | Any value except `""`/`"0"`: `engine.game` ignores the extension (`backend() == "python"`) |
| `ENGINE_REQUIRE_CPP` | `1`: `test_extension_is_built_when_required` fails unless the extension is in use. Unset: that test skips |
| `ENGINE_PARITY_GAMES` | Number of games for the parity test (default 10000). For quick local runs only |

**Results at `d8379a0` (verified locally, Windows, Python 3.11.9):**

| Run | Result |
|---|---|
| `ENGINE_REQUIRE_CPP=1 python -m pytest` | **130 passed** in 65.4 s |
| `ENGINE_PURE_PYTHON=1 ENGINE_REQUIRE_CPP=0 python -m pytest` | **102 passed, 28 skipped** in 16.2 s (all 28 skips need the compiled extension) |

Test inventory (130 tests): `test_engine.py` 23 (skeleton), `test_engine_p1.py` 19 (Batch 1),
`test_engine_helpers.py` 32 (Batch 2), `test_engine_cengine.py` 21 (3b), `test_build_config.py` 4 (3b/3c),
`test_engine_batch.py` 19 (3c), `test_engine_rng.py` 7 (3c), `test_agents.py` 5 (skeleton; agents x engine).

Also verified once with the extension genuinely **unloadable** (Python 3.13 cannot load the cp311 module):
`backend()` returned `"python"` and the engine imported normally. pytest was not installed for 3.13, so the suite
was not run that way; `test_missing_extension_import_falls_back` covers this case in a subprocess.

## 5. The 10,000-game parity test

`tests/test_engine_batch.py::test_cpp_and_python_agree_on_10000_seeded_games`:

1. A subprocess with `ENGINE_PURE_PYTHON=1` (so no C++ can leak in) plays seeds `0..9999` with
   `play_batch(RandomLegalPolicy(), range(n), processes=min(cpu_count, 8), deterministic=True)`.
2. The test process plays the same seeds with `play_random_batch` (C++).
3. Asserts equality of seed, score, max tile, move count, **full move list** and **final board** for every game.

It needs the extension (skips otherwise) and takes ~42 s on the handover laptop. It ran at the full 10,000 games
locally and in every CI job of the C++ matrix. Other parity checks: `test_seeded_games_identical_with_and_without_extension`
(25 `Game.step` games including the final RNG state), `test_batch_matches_pure_python_engine_in_process`,
`test_from_board_games_identical_across_backends`, `test_batch2_helpers_identical_across_backends`, and the row tests.

## 6. C++ standalone self-test (not in CI)

Tests the Batch 3a core only (no Python, no pybind11): known cases, every one of the 19^4 rows in all directions,
200,000 random boards x 4 moves against an independent reference, and golden vectors made by the real Python engine.
Run from the **repo root** (`gen_golden.py` imports `engine` relative to the current directory); keep outputs out of
the repo (do not commit `golden.txt`, `selftest(.exe)`, `*.obj`).

```bash
# Linux / macOS (clang++ instead of g++ on macOS)
g++ -std=c++17 -O2 -Wall -Wextra -o /tmp/selftest engine/_core/selftest.cpp engine/_core/core.cpp
python engine/_core/gen_golden.py /tmp/golden.txt
/tmp/selftest /tmp/golden.txt          # exit code 0 and "ALL C++ CORE CHECKS PASSED"
```

```bat
:: Windows, in an "x64 Native Tools Command Prompt" (or after calling vcvars64.bat), from the repo root
cl /std:c++17 /O2 /EHsc /W4 /Fe:%TEMP%\selftest.exe engine\_core\selftest.cpp engine\_core\core.cpp
python engine\_core\gen_golden.py %TEMP%\golden.txt
%TEMP%\selftest.exe %TEMP%\golden.txt
```

Verified locally on Windows (MSVC 14.52, `/W4`, no warnings): `golden: compared 80028 Python results`,
`ALL C++ CORE CHECKS PASSED`. (Note: `cl` writes `.obj` files to the current directory unless `/Fo` is given; delete
them or point `/Fo` elsewhere.)

## 7. Benchmarks

```bash
python -m engine.bench --seconds 3                          # both modes, C++ if built
python -m engine.bench --seconds 3 --mode step
python -m engine.bench --seconds 3 --mode batch --threads 1
ENGINE_PURE_PYTHON=1 python -m engine.bench --seconds 3     # pure Python
```

It prints the machine, CPU count, Python version, backend and thread count with every number. Both loops are end to
end (spawn, RNG, legal check and move choice inside the clock).

* **`step` mode**: one `Game.step()` call per move with moves from a separate `random.Random`. This is the speed a
  Python agent, `eval/run.py` and `play_batch` get.
* **`batch` mode**: `play_random_batch(..., record_moves=False)` in chunks (2000 games per chunk with C++, 100 in pure
  Python) until the time is up. Whole random-policy games inside C++. Thread start-up and table use are inside the timing.

**Measured results** (moves/second). Compare only within a column.

| Measurement | Windows laptop (verified locally) | Dev container (from `engine/README.md`, not re-verified) |
|---|---|---|
| Machine | AMD64, 14 logical CPUs, Windows 11, Python 3.11.9, MSVC 14.52 | x86_64, 4 vCPU, Python 3.13 unless noted |
| Original skeleton engine (`0e5e863`), step loop | 5,672 (1 run) | not measured |
| Batch-1 Python engine, step (`ENGINE_PURE_PYTHON=1`) | 5,775 – 5,915 (2 runs) | ~11,800 |
| C++ behind `Game.step()` | 68,180 – 102,717 (4 runs) | ~190,000 |
| `play_random_batch`, pure Python | 5,108 | ~12,800 |
| `play_random_batch`, C++, 1 thread | 2,274,174 | ~3,600,000 |
| `play_random_batch`, C++, all threads | 18,870,104 (14 threads) | ~11,900,000 (4 threads; Python 3.11: ~9,800,000) |

The original-engine baseline was measured by running the `run()` loop of `engine/bench.py` against a copy of
`engine/game.py` taken with `git show 0e5e863:engine/game.py`, outside the repo. PR #10's description also
reports 90,277 (C++ step) and 4,426 (Python step) from an earlier Windows run; those were not reproduced here.

**Interpretation.**
* Batch 1 is about as fast as the original engine; it was a correctness change (5-bit cells, 65,536).
* C++ makes `Game.step()` roughly 12–18x faster than the original on the laptop, still far below 1M, because every
  move remains several Python calls plus Python RNG and list copies.
* **The 1M moves/sec target is met only by `play_random_batch()`** (C++, ≥ 2.27M on one thread). It is **not** met by
  `Game.step()` or by `play_batch()`.
* Laptop numbers vary between runs (the C++ step results spread by ~50%). Re-measure and quote the machine, mode and
  thread count.

## 8. Evaluation and replay workflow (verified locally)

```bash
python -m eval.run --agent random --games 10 --seed0 0      # same as CI (Linux job)
python -m eval.replay results/random.jsonl --index 0        # prints "replay matches log"
```

Checked during handover (outputs written outside the repo with `--out-dir`): `random` and `greedy`, 50 games each,
were identical across two C++ runs and a pure-Python run (all fields except `ms_per_move`); all 200 logged games
replayed with matching score/max tile/move count on the other backend. `play_random_batch` results on 1,510 seeds
(including negative seeds, 2^63, 2^64−1, 2^64, 2^70) plus 300 late-game starts were identical for 1, 3 and 14 threads,
a repeat run and pure Python, and all 1,810 games replayed exactly on `Game`.

## 9. What is verified where

| Check | CI (`d8379a0`, all 4 jobs passed) | Locally (Windows) |
|---|---|---|
| Extension builds, `backend()=="cpp"` | ubuntu, windows, macos | yes |
| Full pytest incl. 10,000-game parity, C++ required | ubuntu, windows, macos | yes (130 passed) |
| Full pytest with `ENGINE_PURE_PYTHON=1` | ubuntu, windows, macos | yes (102 passed, 28 skipped) |
| No compiler, no pybind11, pure-Python only | `test (pure-Python fallback)` on ubuntu | not done (extension was present) |
| `engine.bench` runs | ubuntu, windows, macos (informational, numbers not reviewed) | yes |
| `eval.run` random + greedy 10 games | ubuntu jobs only | yes |
| `selftest.cpp` | **not in CI** | yes (MSVC) |
| macOS | CI only | **not tested locally** |

CI results were read from the GitHub API (job and step conclusions). The CI logs themselves were not reviewed, so the
CI benchmark numbers and the exact compiler versions on the runners are unknown to us.

CI workflow (`.github/workflows/ci.yml`) runs on `push` to `main` and on `pull_request` (default activity types).
Jobs: `test (ubuntu-latest | windows-latest | macos-latest)` with `ENGINE_REQUIRE_CPP=1`: install, build, assert
`cpp`, pytest, pytest with `ENGINE_PURE_PYTHON=1`, `python -m engine.bench --seconds 2`, and on Linux only
`eval.run` for random and greedy; plus `test (pure-Python fallback)` with `ENGINE_PURE_PYTHON=1`, no build tools.

## 10. Known performance limitations

* `Game.step()` is Python-bound (~0.07–0.10M moves/s with C++ on the laptop). Agents that search (P3, P7) or train
  (P5) need their own C++/numba inner loops; P1's helpers are correct but not fast enough for deep search in Python.
* `play_batch` speed is never measured; it scales with processes but each worker is Python-bound and builds its own
  16 MiB C++ tables.
* `play_random_batch` falls back to Python for the **whole** batch if any seed is not an exact `int` with
  `abs(seed) < 2^64` (huge-seed batches are slow, not wrong).
* The Python row table is a dict that grows with the rows a process sees (not precomputed).
* Each process that imports the extension builds the C++ tables at import time (16 MiB).
