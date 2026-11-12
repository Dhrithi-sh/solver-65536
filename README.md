# solver-65536

AI agents that play **2048** on a 4x4 board and push toward the **65,536** tile.
We compare search-based, learned and hybrid agents on identical seeded games.

## Novelty

1. **Fair head-to-head.** Every approach (baselines, heuristics, expectimax, n-tuple
   networks, deep RL, hybrids) plays the *same* seeded games through one benchmark
   harness, so tile spawns are identical across agents.
2. **Late-game benchmark.** We start from saved boards that already hold a 32,768 tile
   and measure how often each agent converts them to 65,536. This isolates the hardest
   part of the game, which is too rare to measure from fresh starts.

## Team

| Role | Area | Folder | Label | Review buddy | Name / GitHub |
|------|------|--------|-------|--------------|---------------|
| P1 | Bitboard engine | `engine/` | `P1-engine` | P2 | TBD |
| P2 | Heuristics + weight tuning | `agents/heuristics/` | `P2-heuristics` | P1 | TBD |
| P3 | Expectimax | `agents/expectimax/` | `P3-expectimax` | P4 | TBD |
| P4 | Baselines (corner, minimax, MCTS) | `agents/baselines/` | `P4-baselines` | P3 | TBD |
| P5 | N-tuple network + TD learning | `agents/ntuple/` | `P5-ntuple` | P6 | TBD |
| P6 | Deep RL (DQN / PPO) | `agents/deeprl/` | `P6-deeprl` | P5 | TBD |
| P7 | Hybrid (n-tuple + expectimax), 65,536 mode | `agents/hybrid/` | `P7-hybrid` | P8 | TBD |
| P8 | Benchmarks, dashboard, demo, report | `eval/`, `ui/`, `report/` | `P8-eval` | P7 | TBD |

## Quick start

Requires Python 3.11+.

```bash
git clone https://github.com/Dhrithi-sh/solver-65536.git
cd solver-65536
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

pytest                                                   # run the tests
python -m eval.run --agent greedy --games 100 --seed0 0  # benchmark an agent
python -m eval.replay results/greedy.jsonl --index 0     # replay a logged game
```

`eval.run` writes `results/<agent>.jsonl` (one game per line: `seed`, `score`,
`max_tile`, `n_moves`, `moves`, `ms_per_move`) and prints the mean and median score and
the reach rates for 2048 through 65,536.

## Shared interfaces

Changes to these need a heads-up to the whole group (see CONTRIBUTING.md).

**Engine** (`engine/game.py`): 4x4 list of ints, 0 = empty. Moves are `0=up 1=right 2=down 3=left`.
Spawns a 2 (90%) or a 4 (10%) from a seeded RNG.

```python
from engine.game import Game
g = Game(seed=0)
board, reward, done = g.step(3)        # reward = score gained by merges
g.legal_moves(); g.max_tile(); g.score; g.moves
new_board, gained = Game.slide(board, move)   # pure, no spawn
g.reset(seed=1)                        # or Game(seed, board=saved_board)
```

**Agents** (`agents/base.py`):

```python
class Agent:
    name: str
    def reset(self, seed): ...            # reseed any internal randomness
    def choose_move(self, game) -> int: ... # must return a legal move, must not mutate game

class ValueFunction:
    def evaluate(self, board) -> float: ... # heuristics, n-tuple nets, value networks
```

**Benchmark** (`eval/run.py`): register an agent by adding a zero-argument factory to `AGENTS`.

## Layout

```
engine/            reference engine (P1 replaces internals with a bitboard)
agents/base.py     Agent and ValueFunction interfaces
agents/<area>/     one folder per role, each with a README + TODO list
eval/              run.py (benchmark), replay.py
tests/             engine and agent tests (run in CI)
ui/  report/       demo and report (P8)
```
