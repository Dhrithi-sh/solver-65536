"""Board features for the heuristic evaluator.

Every feature is a pure function of a 4x4 grid of log2 exponents
(0 = empty, k = tile 2**k, so 65,536 is 16) and returns a float where
higher is always better. Penalties are therefore returned negated.

All features are invariant under the 8 rotations/reflections of the board.

Naming follows the engine: boards hold tile values, ``exponents`` uses the
same convention as ``engine.game.exponents``, and the cell *count* feature
is ``empty_count`` because ``engine.game.empty_cells`` is the coordinate list.
"""

from __future__ import annotations

from functools import lru_cache
from operator import mul
from typing import Callable, Dict, List, Sequence, Tuple

from engine.game import SIZE

Grid = Sequence[Sequence[int]]

_CORNERS = ((0, 0), (0, SIZE - 1), (SIZE - 1, 0), (SIZE - 1, SIZE - 1))

# Weight ratio between consecutive cells of the snake path.
SNAKE_RATIO = 0.5


# Same range as the engine's 5-bit cells: 0 (empty) or a power of two up to 2**31.
_VALUE_TO_EXP = {0: 0, **{1 << k: k for k in range(1, 32)}}


def exponents(board: Sequence[Sequence[int]]) -> List[List[int]]:
    """Convert tile values to log2 exponents (0 stays 0, 2 -> 1, 65536 -> 16).

    Same values as ``engine.game.exponents`` (and the same ``ValueError`` for
    a cell that is not 0 or a power of two), but as plain nested lists: the
    features index single cells, which is much faster on lists than on a
    NumPy array, and this keeps the evaluator free of a NumPy dependency.
    """
    try:
        return [[_VALUE_TO_EXP[v] for v in row] for row in board]
    except KeyError as bad:
        raise ValueError(f"cell {bad.args[0]!r} is not 0 or a power of two from 2 to 2**31") from None


def symmetries(grid: Grid) -> List[List[List[int]]]:
    """The 8 rotations and reflections of a square grid."""
    out = []
    g = [list(row) for row in grid]
    for _ in range(4):
        g = [list(row) for row in zip(*g[::-1])]  # rotate 90 degrees clockwise
        out.append(g)
        out.append([row[::-1] for row in g])  # and its mirror image
    return out


# The per-line helpers are memoised: a line has at most 17**4 states and the
# tuner evaluates hundreds of millions of lines, so the cache is a large win.
# Memoising a pure function keeps the public features pure.


@lru_cache(maxsize=None)
def _mono_line(a: int, b: int, c: int, d: int) -> int:
    rises = falls = 0
    for x, y in ((a, b), (b, c), (c, d)):
        if y > x:
            rises += y - x
        else:
            falls += x - y
    return min(rises, falls)


@lru_cache(maxsize=None)
def _smooth_line(a: int, b: int, c: int, d: int) -> int:
    return sum(abs(x - y) for x, y in ((a, b), (b, c), (c, d)) if x and y)


@lru_cache(maxsize=None)
def _pairs_line(a: int, b: int, c: int, d: int) -> int:
    return sum(1 for x, y in ((a, b), (b, c), (c, d)) if x and x == y)


def empty_count(e: Grid) -> float:
    """Number of empty cells."""
    return float(sum(row.count(0) for row in map(list, e)))


def monotonicity(e: Grid) -> float:
    """Minus the cost of making every row and column monotonic.

    A line that only rises or only falls costs 0. Otherwise it costs the
    smaller of (sum of rises, sum of falls): the amount by which it deviates
    from its dominant direction. Empty cells count as exponent 0.
    """
    total = 0
    for a, b, c, d in e:
        total += _mono_line(a, b, c, d)
    for a, b, c, d in zip(*e):
        total += _mono_line(a, b, c, d)
    return -float(total)


def smoothness(e: Grid) -> float:
    """Minus the sum of exponent gaps between adjacent occupied cells.

    Pairs with an empty cell are skipped: an empty neighbour is not a
    "gap of k", it is free space (already rewarded by ``empty_count``).
    """
    total = 0
    for a, b, c, d in e:
        total += _smooth_line(a, b, c, d)
    for a, b, c, d in zip(*e):
        total += _smooth_line(a, b, c, d)
    return -float(total)


def max_in_corner(e: Grid) -> float:
    """1 if a largest tile sits in a corner, else 0 (0 for an empty board)."""
    top = max(map(max, e))
    if top == 0:
        return 0.0
    return 1.0 if any(e[r][c] == top for r, c in _CORNERS) else 0.0


def _snake_weight_sets() -> Tuple[Tuple[float, ...], ...]:
    """Flattened snake weights for each of the 8 board symmetries."""
    base = [[0.0] * SIZE for _ in range(SIZE)]
    k = 0
    for r in range(SIZE):
        cols = range(SIZE) if r % 2 == 0 else reversed(range(SIZE))
        for c in cols:
            base[r][c] = SNAKE_RATIO**k
            k += 1
    return tuple(tuple(v for row in g for v in row) for g in symmetries(base))


_SNAKE_WEIGHTS = _snake_weight_sets()


def snake(e: Grid) -> float:
    """Weighted sum of exponents along a zig-zag path from a corner.

    The k-th cell on the path has weight ``SNAKE_RATIO**k``, so the score is
    highest when tiles are sorted in descending order along the path. The
    best of the 8 orientations is used, so no corner is preferred.
    """
    flat = [v for row in e for v in row]
    return max(sum(map(mul, w, flat)) for w in _SNAKE_WEIGHTS)


def merge_potential(e: Grid) -> float:
    """Number of adjacent pairs of equal, non-empty tiles."""
    total = 0
    for a, b, c, d in e:
        total += _pairs_line(a, b, c, d)
    for a, b, c, d in zip(*e):
        total += _pairs_line(a, b, c, d)
    return float(total)


FEATURES: Dict[str, Callable[[Grid], float]] = {
    "empty": empty_count,
    "mono": monotonicity,
    "smooth": smoothness,
    "corner": max_in_corner,
    "snake": snake,
    "merge": merge_potential,
}
