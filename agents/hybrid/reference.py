"""Pure-Python reference for the hybrid search (slow, simple, obviously correct).

Why it exists
-------------
The fast search lives in C++ (cpp/hsearch.cpp). This file is the *specification*
it must match, written on top of code we did not write ourselves:
  * moves come from the shared engine (``Game.slide``), not from our own C++ slide
  * leaf values come from P5's ``evaluator.evaluate`` (numba), not our C++ lookup
so a test that compares the two (tests/test_hybrid.py) catches mistakes in
either direction. It is also the fallback when the C++ module is not built.

The algorithm (depth ``d`` counts player moves, like P3's expectimax)
---------------------------------------------------------------------
The n-tuple network scores AFTERSTATES (board after our slide, before the random
tile). So we search over afterstates:

    q(board, move)    = reward(move) + chance(afterstate, d)
    chance(a, 0)      = V(a)                            # leaf
    chance(a, d)      = average over every empty cell and spawn (2: 0.9, 4: 0.1)
                        of  max_node(board_after_spawn, d - 1)
    max_node(b, d)    = max over legal moves of  reward + chance(afterstate, d)
                        (0 if the game is over: no more score can be earned)

``d = 0`` is therefore plain 1-ply greedy (identical to NTupleAgent); ``d = 1``
looks one random tile ahead and then plays greedily at the leaf, and so on.

Two speed-ups, both optional:
  * probability cutoff: a chance node reached with probability < ``cutoff`` is
    scored as V(afterstate) instead of being expanded;
  * transposition table keyed on (board, remaining depth), one per root move.
    Note the cutoff makes a cached value depend slightly on which path reached
    the board first. That is fine (it is deterministic) and is why both
    implementations use an identical traversal order.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from engine.game import Game

SPAWNS = ((2, 0.9), (4, 0.1))


def _key(board) -> tuple:
    return tuple(tuple(r) for r in board)


def _empties(board) -> List[tuple]:
    return [(r, c) for r in range(4) for c in range(4) if board[r][c] == 0]


class ReferenceSearch:
    """Search one position. Create one per root move (it owns the TT)."""

    def __init__(self, evaluator, cutoff: float = 1e-4, use_tt: bool = True):
        self.evaluator = evaluator
        self.cutoff = cutoff
        self.use_tt = use_tt
        self._tt: Dict[tuple, float] = {}
        self.nodes = 0

    def chance(self, after, d: int, prob: float) -> float:
        if d == 0 or prob < self.cutoff:
            return float(self.evaluator.evaluate(after))
        self.nodes += 1
        cells = _empties(after)
        n = len(cells)
        if n == 0:
            return float(self.evaluator.evaluate(after))
        total = 0.0
        for r, c in cells:
            for tile, p in SPAWNS:
                child = [list(row) for row in after]
                child[r][c] = tile
                total += p * self.max_node(child, d - 1, prob * p / n)
        return total / n

    def max_node(self, board, d: int, prob: float) -> float:
        key = (_key(board), d)
        if d >= 1 and self.use_tt and key in self._tt:
            return self._tt[key]
        best: Optional[float] = None
        for move in range(4):
            after, reward = Game.slide(board, move)
            if after == board:
                continue
            v = float(reward) + self.chance(after, d, prob)
            if best is None or v > best:      # strict: ties keep the lowest move
                best = v
        if best is None:
            best = 0.0
        if d >= 1 and self.use_tt:
            self._tt[key] = best
        return best


def move_values(
    board: Sequence[Sequence[int]],
    evaluator,
    depth: int,
    cutoff: float = 1e-4,
    use_tt: bool = True,
) -> Dict[int, float]:
    """{move: expected future score} for every legal move at the given depth."""
    board = [list(r) for r in board]
    out: Dict[int, float] = {}
    for move in range(4):
        after, reward = Game.slide(board, move)
        if after == board:
            continue
        search = ReferenceSearch(evaluator, cutoff, use_tt)
        out[move] = float(reward) + search.chance(after, depth, 1.0)
    return out


class AfterstateGreedyLeaf:
    """Adapter that lets P3's ExpectimaxAgent compute the same thing as our depth d.

    P3's leaves are scored with ``evaluator.evaluate(post_spawn_board)``. Our
    leaves (post-spawn boards) are scored as max over moves of reward + V(after),
    so wrapping V in this class makes ``ExpectimaxAgent(AfterstateGreedyLeaf(V),
    depth=d)`` an independent cross-check of the hybrid at depth d.
    """

    def __init__(self, evaluator):
        self.evaluator = evaluator

    def evaluate(self, board) -> float:
        best: Optional[float] = None
        for move in range(4):
            after, reward = Game.slide(board, move)
            if after == [list(r) for r in board]:
                continue
            v = float(reward) + float(self.evaluator.evaluate(after))
            if best is None or v > best:
                best = v
        return 0.0 if best is None else best
