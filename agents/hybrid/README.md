# agents/hybrid — P7 (Hybrid: n-tuple inside expectimax, 65,536 mode)

**Owner:** P7 · **Review buddy:** P8 · **Label:** `P7-hybrid`

The strongest agent we expect to have: P5's n-tuple value function is the leaf
evaluator of an expectimax search, written in C++ for speed, with a late-game
"65,536 mode".

## Quick start

```bash
pip install -r requirements.txt -r agents/hybrid/requirements.txt
python agents/hybrid/cpp/setup.py build_ext      # builds agents/hybrid/_hsearch.*  (needs a C++17 compiler)

python -m eval.run --agent hybrid-d1 --games 100 --seed0 0 --weights path/to/latest.ntw
python -m agents.hybrid.bench --weights path/to/latest.ntw         # speed
python -m agents.hybrid.sweep --weights W --games 30 --depths 1 2   # tuning, DEV seeds only
WEIGHTS=W GAMES=100 bash scripts/hybrid_experiments.sh              # the official comparison
python -m pytest tests/test_hybrid.py -q
```

Compilers: Linux `g++`; macOS `xcode-select --install`; Windows "Build Tools for Visual
Studio" (C++ workload). Without the build the agent still works through the pure-Python
reference, about 30x slower, and prints a warning.

Registered in `eval/run.py` (all deterministic, all accept `--weights`):

| Name | Settings |
|---|---|
| `hybrid-d1` / `-d2` / `-d3` | depth 1 / 2 / 3, cutoff 1e-4 |
| `hybrid-65k` | depth 2; once the board has a tile >= 16,384: depth 3, cutoff 1e-5 |

For the live demo (not registered, because wall-clock budgets are not reproducible):
`from agents.hybrid.registry import make_timed_agent` (100 ms/move, iterative deepening, 4 threads).

## How it works

1. **Value function.** P5's network estimates the *score still to come* from an
   **afterstate** (the board right after our slide, before the random tile). That is
   what it was trained on, so we must use it on afterstates and add the move's own
   merge reward: `Q(move) = reward + V(afterstate)`.
2. **Search** (`reference.py` documents it exactly; `cpp/hsearch.cpp` is the fast version).
   `depth` counts our moves, as in P3. Chance nodes average over every empty cell
   (2 with 0.9, 4 with 0.1); after the spawn we take the best move. At depth 0 the
   agent is exactly P5's greedy agent. At the leaves, a post-spawn board is scored as
   the best `reward + V(afterstate)` over its moves (0 if the game is over).
3. **Speed-ups.** A probability cutoff (branches rarer than 1e-4 become leaves) and a
   transposition table keyed on (board, remaining depth). Row slides come from a
   lookup table; the n-tuple tables are read in place (no copy of the 268 MB).
4. **Threads.** Root moves are searched in parallel, each with its own table, so the
   answer is identical for 1 or 8 threads (tested bit-for-bit).
5. **Time control.** With `time_ms`, iterative deepening runs depth 0, 1, 2, ... and
   returns the deepest *finished* depth when the deadline hits. Depth 0 always finishes.
   An iteration that is predicted not to fit is not started.
6. **65,536 mode** (`Mode65k`): once the max tile >= `trigger_tile` (16,384), add
   `extra_depth` plies, lower the cutoff and optionally raise the time budget.

## Design decisions (for the viva)

* **Search over afterstates with rewards added.** The network was trained on
  afterstates. P3's search scores post-spawn boards and omits move rewards, which is fine for a
  heuristic but wrong for a value measured in score.
* **Own 16-byte board in C++** (one exponent per byte, exponents up to 17). It makes 65,536
  trivial and keeps the search independent of the engine bindings. Correctness is
  tested against `Game.slide`, not against our own code.
* **Exact transposition table, no eviction**, and the same traversal order in Python
  and C++. With a cutoff a cached value depends slightly on the first path that reached
  the board; that is deterministic, so results are reproducible.
* **`-ffp-contract=off`** so C++ rounds like Python and tests can compare to 1e-9.
* **No wall-clock budget in registered agents**, so the shared-seed benchmark is
  reproducible across machines. Time control exists for the demo.
* **Threads default to 1** because the harness will run games in parallel processes.
* **Tuning only on dev seeds.** `sweep.py` refuses seeds outside 0-999.

## Tests (`tests/test_hybrid.py`)

C++ slide == engine (incl. 32,768+32,768 and no double merges) · C++ leaf value ==
`NTupleEvaluator` exactly, both stages · C++ search == pure-Python reference on 1,000
boards at depth 0 and 1 · depth 1 == P3's expectimax (afterstate-greedy leaf), depth 2 ==
P3's with merge rewards added · TT does not change values (cutoff 0) · cutoff prunes ·
1 thread == 4 threads · iterative == fixed depth · time limit respected · 65k mode
switches only past the trigger · two 32,768s are merged into 65,536 · all registered
agents are legal, non-mutating and reproducible on capped games.
`hybrid-d3` is excluded from the whole-game tests in `tests/test_agents.py` (about 100 s
per game with CI's all-zero weights) and covered here instead.

## Results

**Caveat first:** these use a small development network I trained myself (120k games of P5's trainer,
single stage), run on 1 CPU. They show the *direction* of the effect, not final numbers. Rerun with P5's
official weights. All rows come from `eval.run` on shared dev seeds; `agents/hybrid/compare.py` restricts each
table to seeds every listed agent played (identical tile spawns).

**Table A: 40 dev seeds (0-39)**

| Agent | mean score | median | ms/move | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|
| ntuple (1-ply) | 81,562 | 75,944 | 0.01 | 88% | 65% | 40% | 0% |
| hybrid-d1 | 127,196 | 134,198 | 0.13 | 100% | 98% | 65% | 2% |
| hybrid-d2 | 142,752 | 150,030 | 1.55 | 100% | 100% | 82% | 5% |
| hybrid-65k | 143,491 | 150,030 | 1.11 | 100% | 100% | 82% | 5% |

Paired by seed: d1 beats n-tuple on 31/40, d2 beats d1 on 25/40, d2 beats n-tuple on 33/40.
With this network no agent reached 32,768, and 16,384 only in 2 to 5% of games, so 65,536 mode
(trigger 16,384) almost never activated: the +0.7k mean difference vs d2 is not evidence of an effect.
The 50%-reach-16,384 target needs much stronger weights (longer / multi-stage P5 training).

**Table B: hybrid-d3, only 2 games (seeds 0-1): anecdotal, not a result.** d3 scored 249,982 mean
(one game reached 16,384) at 14.7 ms/move vs n-tuple 121,180, d1 147,930, d2 117,560 on the same two seeds.
Two games cannot rank d2 vs d3; the run was cut short by the sandbox.

**Table C: expectimax + P2/P3 heuristic baseline, 5 seeds (0-4)**

| Agent | mean score | 2048 | 4096 | 8192 | ms/move |
|---|---|---|---|---|---|
| expectimax-d2 (heuristic) | 29,305 | 60% | 0% | 0% | 28.05 |
| ntuple | 89,432 | 80% | 60% | 60% | 0.02 |
| hybrid-d1 | 146,869 | 100% | 100% | 100% | 0.13 |
| hybrid-d2 | 125,135 | 100% | 60% | 60% | 1.93 |

Here the heuristic expectimax also fails P3's own "2048 in 90% of games" target at only 5 games.
Small sample; the p-value is not meaningful but the gap is large.

**Speed (16 positions from a real game, cutoff 1e-4, 1 CPU core):**

| Backend | depth 0 | depth 1 | depth 2 | depth 3 |
|---|---|---|---|---|
| Python reference | 0.98 ms | 2.28 ms | - | - |
| C++, 1 thread | 0.030 ms | 0.071 ms | 1.15 ms | 22.8 ms |

C++ is about 32x faster than Python at depths 0 and 1. Multithreading could not be measured (the sandbox has
one core; 4 threads were slightly slower from overhead): run `bench.py` on a multi-core machine.
Cost note: depth 3 with CI's all-zero weights cost about 100 ms/move, because zero values prune poorly.

## TODO
- [x] Hybrid agent, C++ search, threads, time control, 65,536 mode, registration, tests
- [ ] Rerun the table with P5's official long-run weights, on the dev seeds, then once on the test seeds
- [ ] Sweep depth / cutoff / trigger (`sweep.py`) on dev seeds with those weights
- [ ] Late-game benchmark with P8: `python -m eval.late_game --agent hybrid-65k` once the 32,768 bank exists
- [ ] Measure thread scaling on a multi-core machine (`bench.py`)
