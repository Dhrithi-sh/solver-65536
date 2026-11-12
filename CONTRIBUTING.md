# Contributing

## Branches

- One branch per person per topic, named `p<N>-<topic>`, e.g. `p3-expectimax-tt`, `p5-td-learning`.
- Branch from an up-to-date `main`; keep branches short-lived and rebase or merge `main` in often.
- Never push directly to `main`; it is protected.

## Pull requests

- Open a pull request into `main`. Link the issue (`Closes #N`) and add your role label.
- Each PR needs **one approval from your review buddy**:

  **P1 ↔ P2 · P3 ↔ P4 · P5 ↔ P6 · P7 ↔ P8**

  If your buddy is away for more than two days, any teammate can review.
- `main` must pass CI (pytest plus a 10-game benchmark for the random and greedy agents).
  Run `pytest` locally before pushing.
- Add tests for new code. Every new agent goes into `eval/run.py` → `AGENTS`, which
  automatically includes it in the "only legal moves" test.

## Results

- Report numbers **only** from `python -m eval.run` on the shared seeds (the seed ranges
  are set in `eval/README.md`). No numbers from ad-hoc scripts, notebooks or a different engine.
- When you post results, give the exact command, the commit hash and the machine.
- `results/`, `weights/` and `logs/` are git-ignored. Share large files through releases or the team drive.
- Training must use seeds that don't overlap the evaluation seeds.

## Shared interfaces

Don't change the public API in `engine/` or the interfaces in `agents/base.py` without
**telling the group first** (in the team chat plus an issue). These are used by every agent and
by the benchmark. Adding new optional helpers is fine; renaming or changing signatures is not.

## Style

- Python 3.11, type hints on public functions, short docstrings.
- Agents must not mutate the `Game` they are given; look ahead with `Game.slide` on copies.
- Keep heavy dependencies (torch, etc.) out of the top-level `requirements.txt`; put them
  in a per-folder requirements file so CI stays fast.
