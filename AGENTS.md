# AGENTS.md — context for coding agents (Codex, Claude Code, etc.)

## Project

An 8-person college project: AI agents that play 2048 on a 4x4 board and push toward the
65,536 tile. We compare search-based, learned and hybrid agents on identical seeded games.
Our novelty: (1) a fair head-to-head of every approach on the same seeds, and (2) a late-game
benchmark that starts from saved 32,768 boards and measures how often each agent reaches 65,536.

## Folder ownership

Only edit files in the folder owned by the role you are working for, unless the task says otherwise.

| Folder | Owner |
|--------|-------|
| `engine/` | P1: bitboard engine (shared API, see below) |
| `agents/heuristics/` | P2 |
| `agents/expectimax/` | P3 |
| `agents/baselines/` | P4 |
| `agents/ntuple/` | P5 |
| `agents/deeprl/` | P6 |
| `agents/hybrid/` | P7 |
| `eval/`, `ui/`, `report/` | P8 |
| `agents/base.py`, `tests/`, CI, top-level docs | everyone, via a reviewed PR |

## Interfaces (do not change without the group's agreement)

- Board: `list[list[int]]`, 4x4, `0` = empty, otherwise the tile value (2, 4, ..., 65536).
- Moves: `0=up 1=right 2=down 3=left`.
- Spawns: a 2 with probability 0.9 or a 4 with 0.1, in a uniformly chosen empty cell, from
  `random.Random(seed)`. RNG call order: cell index first, then value.
- `engine.game.Game`: `Game(seed, board=None)`, `reset(seed, board=None)`,
  `step(move) -> (board, reward, done)` (an illegal move is a no-op with reward 0),
  `legal_moves()`, `max_tile()`, attributes `board`, `score`, `moves`, `done`, `seed`;
  static, pure `Game.slide(board, move) -> (new_board, gained)`.
- `agents.base.Agent`: `name`, `reset(seed)`, `choose_move(game) -> int` (must be legal; must
  not mutate `game`).
- `agents.base.ValueFunction`: `evaluate(board) -> float` (pure).
- Benchmark: `eval/run.py`, where `AGENTS` maps a name to a zero-argument factory. Output is
  `results/<agent>.jsonl` with `agent, seed, score, max_tile, n_moves, moves (digit string), ms_per_move`.
- When P1's bitboard lands, the API above stays the same. 4-bit cells cap at 32,768, so 65,536
  needs explicit handling; `tests/test_engine.py::test_65536_merge` guards this.

## Coding rules

- Python 3.11. Use type hints on public functions and keep the code readable.
- Seed every source of randomness. Results must be reproducible from `(agent, seed)`.
- Never report numbers that didn't come from `python -m eval.run` on the shared seeds.
- Register new agents in `eval/run.py` → `AGENTS`; the tests then check them automatically.
- Keep heavy dependencies out of the top-level `requirements.txt` (use per-folder files).
- Don't commit `results/`, `weights/`, `logs/` or virtual environments.
- Branches are named `p<N>-<topic>`; open PRs into `main`. Never push to `main` directly.

## Running

```bash
pip install -r requirements.txt
pytest                                                   # all tests (must pass before a PR)
python -m eval.run --agent random --games 10 --seed0 0   # quick benchmark, same as CI
python -m eval.replay results/random.jsonl --index 0     # verify a logged game replays
```
