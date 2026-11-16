"""Fast random self-play (P1, Batch 3c).

    from engine.batch import play_random_batch
    results = play_random_batch(range(10_000))          # C++ if built, else Python

Plays one complete game per seed with the "uniformly random legal move" policy and
returns the same dicts as ``engine.game.play_batch``:
    {"seed", "score", "max_tile", "n_moves", "moves", "board"}
(``moves`` is the replayable list, or None with ``record_moves=False``; with moves
recorded, ``Game(seed)`` plus those moves reproduces the game exactly).

TWO IMPLEMENTATIONS, ONE ANSWER. With the compiled extension, whole games run in C++
across several threads with the GIL released (engine/_core/batch.cpp). Without it,
or for input the extension does not support, the same games are played by the
pure-Python ``Game`` below. tests/test_engine_batch.py proves they agree on 10,000
seeded games (scores, tiles, move lists and final boards), so the fast path is just
a faster way to get the identical result.

THE POLICY (kept identical in C++ and Python; see engine/_core/batch.hpp):
    game RNG   = random.Random(seed)                       -> spawns, as always
    policy RNG = random.Random(abs(seed) ^ POLICY_SEED_MASK)
    move       = legal[policy.randrange(len(legal))], legal in order up, right, down, left
A separate policy RNG means choosing moves never shifts the spawn sequence.

This is NOT how arbitrary agents are run: a Python ``agent_fn`` cannot run at C++
speed. Use ``engine.game.play_batch(agent_fn, seeds, processes=N,
deterministic=True)`` for those.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

from . import game as G

# Must equal p1::kPolicySeedMask in engine/_core/batch.hpp (a test checks it).
POLICY_SEED_MASK = 0x2545F4914F6CDD1D


def policy_seed(seed: int) -> int:
    """Seed of the move-choosing RNG for a game with this seed."""
    return abs(seed) ^ POLICY_SEED_MASK


class RandomLegalPolicy:
    """``agent_fn`` form of the policy, usable with ``engine.game.play_batch``.

    Picklable. It re-seeds its RNG at the first move of every game (when
    ``game.moves`` is empty), so it needs no ``reset`` and gives the same moves
    wherever and whenever a game is played.
    """

    def __init__(self) -> None:
        self._rng = random.Random(0)

    def __call__(self, game: G.Game) -> int:
        if not game.moves:
            self._rng = random.Random(policy_seed(game.seed))
        legal = game.legal_moves()  # ascending: up, right, down, left
        return legal[self._rng.randrange(len(legal))]


def _play_python(seed: int, board, record_moves: bool) -> dict:
    """Reference implementation: the ordinary Game plus the policy above."""
    game = G.Game(seed, board)
    policy = RandomLegalPolicy()
    while not game.done:
        game.step(policy(game))
    return _result(seed, game.score, game.max_tile(), len(game.moves),
                   list(game.moves) if record_moves else None, G.copy_board(game.board))


def _result(seed, score, max_tile, n_moves, moves, board) -> dict:
    return {"seed": seed, "score": score, "max_tile": max_tile, "n_moves": n_moves,
            "moves": moves, "board": board}


def play_random_batch(seeds: Sequence[int], board: Optional[Sequence[Sequence[int]]] = None,
                      threads: int = 0, record_moves: bool = True) -> List[dict]:
    """One random-policy game per seed; results in the order of ``seeds``.

    ``board`` starts every game from that position (late-game benchmark) instead of
    a fresh game; no tiles are spawned at the start, as in ``Game.from_board``.
    ``threads`` is the C++ worker count (0 = all cores); the results never depend on
    it. ``record_moves=False`` skips building move lists (faster for benchmarks).
    With no extension (or unsupported input) the games are played by the pure-Python
    engine, one after another, and ``threads`` is ignored.
    """
    seeds = list(seeds)
    if board is not None:
        G._check_board_shape(board)
    ext = G._cengine
    raw = None
    if ext is not None and hasattr(ext, "play_random"):
        raw = ext.play_random(seeds, board, threads, record_moves)  # None = unsupported input
    out: List[dict] = []
    for i, seed in enumerate(seeds):
        if raw is None or raw[i][5]:  # no C++ answer, or the game left the C++ range ("bailed")
            out.append(_play_python(seed, board, record_moves))
            continue
        score, max_tile, n_moves, moves, final_board, _ = raw[i]
        out.append(_result(seed, score, max_tile, n_moves, list(moves) if moves is not None else None, final_board))
    return out
