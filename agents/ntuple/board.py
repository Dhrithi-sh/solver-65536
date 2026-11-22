"""Fast board core for n-tuple *training* only (numba-compiled).

This does not replace or touch ``engine/``. The shared engine stays the single
source of truth for benchmarks; this module exists because training needs
100k+ games and the pure-Python engine is far too slow. ``tests/test_ntuple_board.py``
checks that ``slide`` agrees exactly with ``engine.game.Game.slide``.

Board layout
------------
A board is a flat ``np.uint8`` array of 16 **log2 exponents**, row-major
(cell ``4*r + c``). 0 = empty, 1 = tile 2, ..., 15 = 32,768, 16 = 65,536.
uint8 holds exponents up to 255, so 32,768 + 32,768 -> 65,536 (15 + 15 -> 16)
is represented exactly; clamping to 15 happens later, only when indexing the
n-tuple tables.

Moves: 0 = up, 1 = right, 2 = down, 3 = left (same as the engine).

Randomness
----------
``spawn_random`` uses its own tiny RNG (xorshift32) whose whole state is one
int64 in a length-1 array, created by ``make_rng(seed)``. It is deterministic
from the seed in both numba and plain Python. It does NOT reproduce the
engine's ``random.Random`` stream; training games only need to be
reproducible from the training seed, not identical to benchmark games.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from agents.ntuple._jit import njit

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3


# --------------------------------------------------------------------- tables
def _build_lines() -> np.ndarray:
    """LINES[move, line, pos] = cell index, ordered so tiles slide toward pos 0.

    Every move is the same operation ("slide a line of 4 toward its start")
    applied to 4 lines; only the order in which cells are read differs. This is
    the flat-array equivalent of ``engine.game._get_lines``.
    """
    lines = np.zeros((4, 4, 4), dtype=np.int64)
    for i in range(4):
        for j in range(4):
            lines[UP, i, j] = 4 * j + i  # column i, top to bottom
            lines[RIGHT, i, j] = 4 * i + (3 - j)  # row i, right to left
            lines[DOWN, i, j] = 4 * (3 - j) + i  # column i, bottom to top
            lines[LEFT, i, j] = 4 * i + j  # row i, left to right
    return lines


def _build_symmetries() -> np.ndarray:
    """SYMMETRIES[s] is a permutation p of 0..15 with sym_board[i] = board[p[i]].

    The 8 symmetries of a square (the dihedral group D4): 4 rotations, each
    with and without a left-right mirror. Row 0 is the identity.
    """
    idx = np.arange(16).reshape(4, 4)
    perms = []
    for k in range(4):
        rotated = np.rot90(idx, k)
        perms.append(rotated.ravel())
        perms.append(np.fliplr(rotated).ravel())
    return np.ascontiguousarray(np.array(perms, dtype=np.int64))


LINES = _build_lines()
SYMMETRIES = _build_symmetries()

# P(spawn is a 2) = 0.9, as a threshold on a uniform 32-bit integer.
_TWO_THRESHOLD = int(0.9 * 2**32)
_MASK32 = 0xFFFFFFFF


# ----------------------------------------------------------------- converters
def to_exponents(board: Sequence[Sequence[int]]) -> np.ndarray:
    """Engine board (4x4 tile values, 0 = empty) -> flat uint8[16] exponents."""
    out = np.zeros(16, dtype=np.uint8)
    for r in range(4):
        for c in range(4):
            v = int(board[r][c])
            if v == 0:
                continue
            if v < 2 or v & (v - 1):
                raise ValueError(f"cell ({r},{c}) holds {v}, not a power of two >= 2")
            out[4 * r + c] = v.bit_length() - 1
    return out


def from_exponents(arr: np.ndarray) -> List[List[int]]:
    """Flat 16 exponents -> engine board (4x4 list of tile values)."""
    flat = [int(e) for e in np.asarray(arr).ravel()]
    if len(flat) != 16:
        raise ValueError(f"expected 16 cells, got {len(flat)}")
    return [[(1 << e) if e else 0 for e in flat[4 * r : 4 * r + 4]] for r in range(4)]


# ------------------------------------------------------------------------ rng
def make_rng(seed: int) -> np.ndarray:
    """Create RNG state for ``spawn_random`` from any integer seed.

    The seed is scrambled with splitmix64 so nearby seeds give unrelated
    streams, then reduced to a non-zero 32-bit value (xorshift32 must never
    hold 0). Returned as a length-1 int64 array so jitted code can update it
    in place.
    """
    mask64 = (1 << 64) - 1
    z = (int(seed) + 0x9E3779B97F4A7C15) & mask64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask64
    z ^= z >> 31
    x = z & _MASK32
    return np.array([x if x else 1], dtype=np.int64)


@njit(cache=True)
def next_u32(state: np.ndarray) -> int:
    """Advance xorshift32 and return a uniform integer in [1, 2**32 - 1].

    Values stay below 2**32 and shifts are masked, so int64 never overflows;
    that keeps numba and plain-Python results bit-identical.
    """
    x = state[0]
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    state[0] = x
    return x


# ----------------------------------------------------------------- game rules
@njit(cache=True)
def slide(board: np.ndarray, move: int) -> Tuple[np.ndarray, int, bool]:
    """Slide ``board`` in ``move``. Returns (afterstate, reward, changed).

    Pure: ``board`` is not modified and no tile is spawned. ``reward`` is the
    sum of merged tile values, exactly like the engine's score.
    """
    out = board.copy()
    reward = 0
    tmp = np.zeros(4, dtype=np.int64)
    for line in range(4):
        # 1. Compact: copy the non-empty tiles of this line, in slide order.
        n = 0
        for pos in range(4):
            e = int(board[LINES[move, line, pos]])
            if e != 0:
                tmp[n] = e
                n += 1
        # 2. Merge left-to-right. A merged tile is written once and we skip
        #    past both inputs, so it can't merge again in this move.
        w = 0
        i = 0
        while i < n:
            e = tmp[i]
            if i + 1 < n and tmp[i + 1] == e:
                e += 1
                reward += 1 << e
                i += 2
            else:
                i += 1
            out[LINES[move, line, w]] = e
            w += 1
        # 3. Pad the rest of the line with empties.
        while w < 4:
            out[LINES[move, line, w]] = 0
            w += 1
    changed = False
    for k in range(16):
        if out[k] != board[k]:
            changed = True
            break
    return out, reward, changed


@njit(cache=True)
def can_move(board: np.ndarray, move: int) -> bool:
    """True if ``move`` changes the board (cheaper than calling ``slide``).

    A line can slide iff an empty cell comes before a tile (in slide order),
    or two equal tiles are adjacent once the empties are skipped.
    """
    for line in range(4):
        prev = 0  # last tile seen in this line (0 = none yet)
        seen_empty = False
        for pos in range(4):
            e = board[LINES[move, line, pos]]
            if e == 0:
                seen_empty = True
            else:
                if seen_empty or e == prev:
                    return True
                prev = e
    return False


@njit(cache=True)
def legal_moves(board: np.ndarray) -> np.ndarray:
    """Boolean mask of length 4: ``mask[m]`` is True if move m is legal."""
    mask = np.zeros(4, dtype=np.bool_)
    for m in range(4):
        mask[m] = can_move(board, m)
    return mask


@njit(cache=True)
def is_game_over(board: np.ndarray) -> bool:
    """True if no move is legal: no empty cell and no equal neighbours."""
    for k in range(16):
        if board[k] == 0:
            return False
    for r in range(4):
        for c in range(4):
            e = board[4 * r + c]
            if c < 3 and board[4 * r + c + 1] == e:
                return False
            if r < 3 and board[4 * (r + 1) + c] == e:
                return False
    return True


@njit(cache=True)
def spawn_random(board: np.ndarray, rng_state: np.ndarray) -> int:
    """Place a 2 (90%) or 4 (10%) in a uniformly chosen empty cell, IN PLACE.

    Returns the cell index, or -1 if the board is full. Like the engine, it
    draws the cell first and the value second.
    """
    n_empty = 0
    for k in range(16):
        if board[k] == 0:
            n_empty += 1
    if n_empty == 0:
        return -1
    target = next_u32(rng_state) % n_empty
    cell = -1
    for k in range(16):
        if board[k] == 0:
            if target == 0:
                cell = k
                break
            target -= 1
    board[cell] = 1 if next_u32(rng_state) < _TWO_THRESHOLD else 2
    return cell


@njit(cache=True)
def symmetries(board: np.ndarray) -> np.ndarray:
    """All 8 rotations/reflections of ``board`` as a (8, 16) uint8 array.

    Row 0 is the board itself.
    """
    out = np.empty((8, 16), dtype=np.uint8)
    for s in range(8):
        for i in range(16):
            out[s, i] = board[SYMMETRIES[s, i]]
    return out


# ------------------------------------------------------------------ benchmark
@njit(cache=True)
def random_play(n_games: int, rng_state: np.ndarray) -> int:
    """Play ``n_games`` with uniformly random legal moves; return total moves."""
    total = 0
    for _ in range(n_games):
        board = np.zeros(16, dtype=np.uint8)
        spawn_random(board, rng_state)
        spawn_random(board, rng_state)
        while True:
            mask = legal_moves(board)
            n_legal = 0
            for m in range(4):
                if mask[m]:
                    n_legal += 1
            if n_legal == 0:
                break
            pick = next_u32(rng_state) % n_legal
            move = 0
            for m in range(4):
                if mask[m]:
                    if pick == 0:
                        move = m
                        break
                    pick -= 1
            board, _, _ = slide(board, move)
            spawn_random(board, rng_state)
            total += 1
    return total


def benchmark(n_games: int = 20_000, seed: int = 0) -> float:
    """Moves per second for random play (compile time excluded)."""
    import time

    random_play(2, make_rng(seed))  # warm-up: trigger compilation
    state = make_rng(seed)
    t0 = time.perf_counter()
    moves = random_play(n_games, state)
    return moves / (time.perf_counter() - t0)


if __name__ == "__main__":
    print(f"random play: {benchmark():,.0f} moves/sec")
