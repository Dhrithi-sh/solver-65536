# engine/_core — C++ core (P1, Batch 3a)

Status: **3a = C++ core + standalone checks.** The pybind11 binding, Python
wiring, CI and `pyproject.toml` changes are sub-batch 3b (not here yet).
Nothing in this folder is imported by Python yet, so the pure-Python engine
behaves exactly as before.

## What is here

| File | Purpose |
|---|---|
| `core.hpp` / `core.cpp` | Board type, row tables, `move`, `is_legal`, `legal_mask`, `empty_count`, `max_exponent`, conversions |
| `selftest.cpp` | Standalone checks (no framework, exit code 0 = pass) |
| `gen_golden.py` | Writes test vectors made by the **real Python engine** for `selftest` |
| `bench.cpp` | Raw core speed (core only, no Python, not the game RNG) |

## Design (beginner-friendly)

* **Board** = 4 rows, each a 20-bit number: 4 cells x 5 bits, each cell an exponent
  (0 empty, 16 = 65,536). Cell 0 is in the lowest bits, same as `pack_row()` in
  `engine/game.py`. Four `uint32_t` rows are used instead of `__uint128_t` because
  **MSVC on Windows has no 128-bit integer**.
* **Row tables:** for each of the 2^20 possible rows we precompute the row after a
  left move and its score; a second table does right moves (reverse the row, slide
  left, reverse back). Tables are built once on first use (small fraction of a
  second; ~16 MB of memory per process).
* **Directions:** LEFT/RIGHT look up each row. UP/DOWN first transpose (rows <->
  columns), do LEFT/RIGHT, then transpose back. Numbering: 0 up, 1 right, 2 down, 3 left.
* **Supported range:** every exponent must be <= 18 (tile 262,144). A move can then
  create at most exponent 19, and one row's score stays far below 2^32. Real 4x4
  games never exceed 131,072 (exponent 17); 262,144 only appears on artificial boards
  (two 131,072 tiles from `Game.from_board`). Exponent 18 is accepted as input so such
  a tile can keep moving. Rows with exponent > 18 are stored unchanged with score 0, so
  nothing overflows, and `move()` does NOT report an error for them: callers must check
  `is_supported(board)` first. Whatever limit is chosen, the board right after the
  first merge above it is unsupported, so the Python engine (exact up to exponent 30)
  stays the fallback for those boards.
* **No spawning / RNG here.** The Python engine keeps its own `random.Random`.

## Build and run the checks

You need a C++17 compiler. Nothing else for 3a (no Python packages, no pybind11).

**Linux** (`sudo apt install g++` if missing), from the repo root:
```bash
g++ -std=c++17 -O2 -Wall -Wextra -o selftest engine/_core/selftest.cpp engine/_core/core.cpp
python engine/_core/gen_golden.py golden.txt
./selftest golden.txt
g++ -std=c++17 -O2 -o bench engine/_core/bench.cpp engine/_core/core.cpp && ./bench 2
```

**macOS** (`xcode-select --install` for clang): same commands, with `clang++` instead of `g++`.

**Windows, MSVC** (install "Build Tools for Visual Studio 2022" with the
*Desktop development with C++* workload; open **x64 Native Tools Command Prompt for
VS 2022**), from the repo root:
```bat
cl /std:c++17 /O2 /EHsc /W4 /Fe:selftest.exe engine\_core\selftest.cpp engine\_core\core.cpp
python engine\_core\gen_golden.py golden.txt
selftest.exe golden.txt
cl /std:c++17 /O2 /EHsc /Fe:bench.exe engine\_core\bench.cpp engine\_core\core.cpp
bench.exe 2
```
**Windows, MinGW-w64** (e.g. via MSYS2): the Linux `g++` commands work unchanged.

Do not commit `golden.txt`, `selftest(.exe)`, `bench(.exe)` or `*.obj`.

## What the checks cover

Known cases (`[2,2,2,2] -> [4,4,0,0]`, 32,768 + 32,768 -> 65,536 in all four directions,
65,536 + 65,536 -> 131,072 and 131,072 + 131,072 -> 262,144 in all four directions, a
262,144 tile at every cell moved in every direction, the chain 131,072 x2 -> 262,144 ->
another move on that exponent-18 board, the pinned behaviour for exponent 19, dead boards,
invalid move numbers throw); every one of the
19^4 rows (exponents 0..18) in all directions, as a row and as a column; 200,000 random
boards x 4 moves against an independent reference; and ~80,000 results produced by the
real Python `Game.slide`.

## Contract for the Python binding (sub-batch 3b)

The binding must never hand an unsupported board to the C++ core. Planned rules:

1. The Python wrapper pre-checks the board: 4x4, every cell an `int` that is 0 or a
   power of two with exponent <= 18. Anything else (too big, not a power of two, wrong
   shape) takes the existing pure-Python path, unchanged.
2. The C++ entry point converts the board itself and raises `ValueError` if
   `is_supported()` is false (defence in depth), instead of returning an unchanged board.
3. If the extension is not built, `import` fails quietly and the Python engine is used.
4. Tests for 3b will force each path (a supported board, an exponent-19 board, a
   non-power-of-two board, extension missing) and compare with the Python engine.

These rules are a plan; the binding does not exist yet, so none of this is tested.
