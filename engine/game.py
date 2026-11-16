"""Reference 2048 engine (simple, readable, correct; not fast).

Board: 4x4 list of lists of ints. 0 = empty, other cells hold the tile value
(2, 4, 8, ...). Moves: 0 = up, 1 = right, 2 = down, 3 = left.

Spawn rule: after every legal move, one empty cell is chosen uniformly at
random and receives a 2 with probability 0.9 or a 4 with probability 0.1.
All randomness comes from a per-game ``random.Random(seed)``, so a seed plus
a move list fully determines a game (see eval/replay.py).

Internal representation (P1, Batch 1): ``Game.board`` stays a 4x4 nested list
of tile VALUES (every agent reads it). Inside ``Game.slide`` each line is
mapped to 5-bit exponents (0 = empty, k = tile 2**k) and packed into a 20-bit
row key (4 cells x 5 bits), which indexes a lazily filled row table holding the
slid row and the score gained. 5 bits hold exponents 0..31, so 32,768 + 32,768
-> 65,536 (exponent 16) fits, which 4-bit cells cannot do. Boards containing
anything that is not 0 or a power of two between 2 and 2**31 fall back to the
original per-line merge, so behaviour never changes for odd inputs.
See engine/README.md. The C++ core (engine/_core, Batch 3) uses the same
layout, with the whole 2**20-row table precomputed when the extension loads.

Optional C++ core (P1, Batch 3b): if the compiled extension ``engine._cengine``
is importable, Game.slide and Game.legal_moves ask it first. It answers None for
anything it does not support (not a 4x4 list/tuple of exact ints, a cell that is
not 0 or a power of two up to 2**18, a bad move), and then the pure-Python code
below runs exactly as before. Spawning and the RNG always stay in Python. Set the
environment variable ENGINE_PURE_PYTHON=1 to skip the extension; see backend().

Helpers (P1, Batch 2, pure Python): Game.from_board, afterstate, empty_cells,
spawn_outcomes, exponents (needs numpy only when called) and play_batch.
They sit below the Game class and never change how a Game plays.
"""

from __future__ import annotations

import copy
import numbers
import os
import random
from typing import List, Optional, Sequence, Tuple

# ---------------------------------------------------- optional C++ extension
# Never required: if it is missing, fails to load, or ENGINE_PURE_PYTHON is set
# (anything except "" or "0"), _cengine stays None and everything runs in Python.
_cengine = None
if os.environ.get("ENGINE_PURE_PYTHON", "") in ("", "0"):
    try:
        from . import _cengine  # type: ignore[attr-defined]
    except ImportError:  # not built / not loadable / not imported as a package
        _cengine = None


def backend() -> str:
    """"cpp" if the compiled extension is in use, otherwise "python"."""
    return "python" if _cengine is None else "cpp"


Board = List[List[int]]

SIZE = 4
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
MOVES = (UP, RIGHT, DOWN, LEFT)
MOVE_NAMES = {UP: "up", RIGHT: "right", DOWN: "down", LEFT: "left"}
PROB_TWO = 0.9


def empty_board() -> Board:
    return [[0] * SIZE for _ in range(SIZE)]


def copy_board(board: Sequence[Sequence[int]]) -> Board:
    return [list(row) for row in board]


def _merge_line(line: Sequence[int]) -> Tuple[List[int], int]:
    """Slide one line toward index 0. Each tile merges at most once per move."""
    tiles = [v for v in line if v != 0]
    out: List[int] = []
    gained = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            merged = tiles[i] * 2
            out.append(merged)
            gained += merged
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    out.extend([0] * (SIZE - len(out)))
    return out, gained


def _get_lines(board: Sequence[Sequence[int]], move: int) -> List[List[int]]:
    """Lines ordered so that tiles move toward index 0."""
    if move == LEFT:
        return [list(board[r]) for r in range(SIZE)]
    if move == RIGHT:
        return [list(reversed(board[r])) for r in range(SIZE)]
    if move == UP:
        return [[board[r][c] for r in range(SIZE)] for c in range(SIZE)]
    if move == DOWN:
        return [[board[r][c] for r in reversed(range(SIZE))] for c in range(SIZE)]
    raise ValueError(f"invalid move {move!r}; expected 0..3")


def _set_lines(lines: List[List[int]], move: int) -> Board:
    board = empty_board()
    for i, line in enumerate(lines):
        for j, v in enumerate(line):
            if move == LEFT:
                board[i][j] = v
            elif move == RIGHT:
                board[i][SIZE - 1 - j] = v
            elif move == UP:
                board[j][i] = v
            else:  # DOWN
                board[SIZE - 1 - j][i] = v
    return board


# ---------------------------------------------------------------- row table
# 5 bits per cell: exponent 0 = empty, k = tile 2**k (max exponent 31).
CELL_BITS = 5
CELL_MASK = (1 << CELL_BITS) - 1
MAX_EXP = CELL_MASK  # 31
_VALUE_TO_EXP = {0: 0, **{1 << k: k for k in range(1, MAX_EXP + 1)}}

# row key (20 bits) -> (slid row key, score gained). Filled lazily: a full
# 2**20-entry table would cost seconds at import, and real games only touch a
# small fraction of rows. The C++ core precomputes the whole table.
_ROW_TABLE: dict = {}


def pack_row(exps: Sequence[int]) -> int:
    """Pack 4 exponents into a 20-bit key; cell 0 sits in the lowest bits."""
    key = 0
    for i, e in enumerate(exps):
        key |= e << (CELL_BITS * i)
    return key


def unpack_row(key: int) -> List[int]:
    return [(key >> (CELL_BITS * i)) & CELL_MASK for i in range(SIZE)]


def _slide_row_exps(exps: Sequence[int]) -> Tuple[List[int], int]:
    """Left-slide one row of exponents. Returns (exponents, score gained).

    A merge of two tiles with exponent e makes exponent e+1 and scores
    2**(e+1), exactly what _merge_line scores on tile values.
    """
    tiles = [e for e in exps if e]
    out: List[int] = []
    gained = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            out.append(tiles[i] + 1)
            gained += 1 << (tiles[i] + 1)
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    out.extend([0] * (SIZE - len(out)))
    return out, gained


def row_table_entry(key: int) -> Tuple[int, int]:
    """(slid row key, score gained) for a packed row, computed once, then cached."""
    hit = _ROW_TABLE.get(key)
    if hit is None:
        out, gained = _slide_row_exps(unpack_row(key))
        hit = _ROW_TABLE[key] = (pack_row(out), gained)
    return hit


def _slide_lines_table(lines: List[List[int]]):
    """Table-driven slide of oriented lines; None if a cell is not encodable."""
    result: List[List[int]] = []
    gained = 0
    for line in lines:
        try:
            key = pack_row([_VALUE_TO_EXP[v] for v in line])
        except (KeyError, TypeError):
            return None
        out_key, g = row_table_entry(key)
        result.append([(1 << e) if e else 0 for e in unpack_row(out_key)])
        gained += g
    return result, gained


# ------------------------------------------------------------ Batch 2 helpers
def _check_board_shape(board: Sequence[Sequence[int]]) -> None:
    if len(board) != SIZE or any(len(row) != SIZE for row in board):
        raise ValueError(f"board must be {SIZE}x{SIZE}")


def afterstate(board: Sequence[Sequence[int]], move: int) -> Board:
    """Board right after sliding in ``move``, BEFORE the random tile spawns.

    Pure: ``board`` is not modified and a new board is returned. If the move
    changes nothing (illegal) you simply get an equal copy back, so check
    ``Game.is_legal(board, move)`` first when it matters. Need the merge score
    too? Use ``Game.slide(board, move)``, which returns (afterstate, gained).
    Raises ValueError for a move outside 0..3.
    """
    return Game.slide(board, move)[0]


def empty_cells(board: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    """All (row, col) positions holding 0, in row-major order.

    Same order as ``Game.empty_cells()``, which is what the spawn RNG indexes
    into. (The method needs a Game; this function works on any board.)
    """
    return [(r, c) for r in range(SIZE) for c in range(SIZE) if board[r][c] == 0]


def spawn_outcomes(board: Sequence[Sequence[int]]) -> List[Tuple[Board, float]]:
    """Every board the next spawn can produce, with its probability.

    A spawn picks one empty cell uniformly, then a 2 (p = PROB_TWO = 0.9) or a
    4 (p = 0.1). With n empty cells that is 2n outcomes: p = 0.9 / n for each
    "2" board and 0.1 / n for each "4" board, summing to 1. Order: empty cells
    in row-major order, the 2 before the 4. A full board returns [] (no
    spawn is possible). Boards are fresh copies; ``board`` is not modified.
    Used by expectimax chance nodes (P3, P7).
    """
    cells = empty_cells(board)
    n = len(cells)
    outcomes: List[Tuple[Board, float]] = []
    for r, c in cells:
        for value, prob in ((2, PROB_TWO), (4, 1.0 - PROB_TWO)):
            child = copy_board(board)
            child[r][c] = value
            outcomes.append((child, prob / n))
    return outcomes


def exponents(board: Sequence[Sequence[int]]):
    """The board as a 4x4 NumPy array of log2 exponents (0 = empty, 2 -> 1,
    65536 -> 16). Used by the n-tuple network (P5) and deep RL (P6).

    dtype is int32 on purpose: smaller types such as int8 silently wrap around
    when you compute table indexes like e0 + 16 * e1 + 256 * e2. Raises
    ValueError for a cell that is not 0 or a power of two >= 2, and ImportError
    with a clear message if numpy is not installed (numpy is only needed here).
    """
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("exponents() needs numpy: pip install numpy") from exc
    _check_board_shape(board)
    out = np.zeros((SIZE, SIZE), dtype=np.int32)
    for r in range(SIZE):
        for c in range(SIZE):
            v = board[r][c]
            exp = _VALUE_TO_EXP.get(int(v)) if isinstance(v, numbers.Integral) else None
            if exp is None:
                raise ValueError(f"cell ({r},{c}) = {v!r} is not 0 or a power of two >= 2")
            out[r, c] = exp
    return out


def _play_one(args):
    agent_fn, seed, *rest = args
    if rest and rest[0]:
        # deterministic mode: this game gets its own copy of the agent (so no state
        # leaks from other games, whatever the process count or chunking) and, for a
        # bound method of an Agent, reset(seed) exactly like eval/run.py does.
        agent_fn = copy.deepcopy(agent_fn)
        reset = getattr(getattr(agent_fn, "__self__", None), "reset", None)
        if callable(reset):
            reset(seed)
    game = Game(seed)
    while not game.done:
        move = agent_fn(game)
        if move not in game.legal_moves():
            raise ValueError(f"agent chose {move!r} on seed {seed}; legal moves: {game.legal_moves()}")
        game.step(move)
    return {"seed": seed, "score": game.score, "max_tile": game.max_tile(),
            "moves": list(game.moves), "board": copy_board(game.board)}


def play_batch(agent_fn, seeds: Sequence[int], processes: int = 1, *,
               deterministic: bool = False, start_method: Optional[str] = None) -> List[dict]:
    """Play one full game per seed with ``agent_fn(game) -> move``.

    ``agent_fn`` is called exactly like ``Agent.choose_move`` (e.g. pass
    ``agent.choose_move``) and must return one of ``game.legal_moves()``;
    anything else raises ValueError instead of looping forever. It must not
    change the game it is given. Returns one dict per seed, in the SAME order
    as ``seeds``: {"seed", "score", "max_tile", "moves" (list, replayable with
    Game(seed)), "board" (final)}.

    ``processes=1`` runs in this process. ``processes > 1`` uses a
    multiprocessing pool, which needs ``agent_fn`` to be picklable (a
    module-level function or a bound method of a picklable object, not a
    lambda). Results are identical either way FOR AGENTS WITHOUT STATE.

    Agents that keep state between games (an RNG, a cache, a move counter) are the
    catch: serially one object is shared by all games, while a pool hands each worker
    a pickled copy and gives games to workers in chunks, so results would depend on
    the process count. Pass ``deterministic=True`` to remove that: every game then
    runs on a fresh deep copy of ``agent_fn`` and, if it is a bound method of an
    object with ``reset(seed)`` (an ``Agent``), that is called first, so a game
    depends only on (agent, seed), identical for any ``processes``.

    ``start_method`` ("fork", "spawn" or "forkserver") picks the multiprocessing
    start method; the default is the platform's (spawn on Windows and macOS, where
    ``agent_fn`` must also be importable from a module, and the calling script needs
    an ``if __name__ == "__main__":`` guard). Both keywords are optional additions;
    the original call forms behave as before.
    """
    jobs = [(agent_fn, seed, deterministic) for seed in seeds]
    if processes <= 1 or len(jobs) <= 1:
        return [_play_one(job) for job in jobs]
    import multiprocessing
    with multiprocessing.get_context(start_method).Pool(processes) as pool:
        # chunksize=1 in deterministic mode: one pickled agent copy per game.
        return pool.map(_play_one, jobs, chunksize=1 if deterministic else None)


class Game:
    """A seeded 2048 game.

    Public API (shared by every agent and by the benchmark; do not change
    without telling the group):
        Game(seed), reset(seed), step(move) -> (board, reward, done),
        legal_moves(), max_tile(), Game.slide(board, move) -> (new_board, gained)
    Attributes: board, score, moves (list of moves played), seed, done.
    """

    def __init__(self, seed: int = 0, board: Optional[Sequence[Sequence[int]]] = None):
        self.reset(seed, board)

    @classmethod
    def from_board(cls, board: Sequence[Sequence[int]], seed: int = 0) -> "Game":
        """Start a game from any position (late-game benchmark, tests, search).

        Same as ``Game(seed, board=board)`` plus a shape check. The board is
        copied, no tiles are spawned at the start, and the RNG is seeded with
        ``seed`` so the spawns that follow are reproducible. (``reset()`` with
        no board still starts a normal game from the seed, as it always did.)
        """
        _check_board_shape(board)
        return cls(seed, board)

    # ------------------------------------------------------------------ setup
    def reset(self, seed: Optional[int] = None, board: Optional[Sequence[Sequence[int]]] = None) -> Board:
        """Start a new game. With ``board`` given (e.g. a saved 32,768 position
        for the late-game benchmark) no initial tiles are spawned."""
        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)
        self.score = 0
        self.moves: List[int] = []
        if board is not None:
            self.board = copy_board(board)
        else:
            self.board = empty_board()
            self._spawn()
            self._spawn()
        self.done = not self.legal_moves()
        return copy_board(self.board)

    # ------------------------------------------------------------ pure helpers
    @staticmethod
    def slide(board: Sequence[Sequence[int]], move: int) -> Tuple[Board, int]:
        """Pure: return (board after sliding in ``move``, score gained). No spawn."""
        if _cengine is not None and move.__class__ is int:
            # None = "board/move not supported by the C++ core": use Python below.
            result = _cengine.slide(board, move)
            if result is not None:
                return result
        lines = _get_lines(board, move)
        fast = _slide_lines_table(lines)
        if fast is not None:
            return _set_lines(fast[0], move), fast[1]
        # Fallback for cells that are not 0 / a power of two: original merge.
        merged = [_merge_line(line) for line in lines]
        new_board = _set_lines([m[0] for m in merged], move)
        return new_board, sum(m[1] for m in merged)

    @staticmethod
    def is_legal(board: Sequence[Sequence[int]], move: int) -> bool:
        new_board, _ = Game.slide(board, move)
        return new_board != [list(r) for r in board]

    # ---------------------------------------------------------------- queries
    def legal_moves(self) -> List[int]:
        if _cengine is not None:
            mask = _cengine.legal_mask(self.board)
            if mask is not None:
                return [m for m in MOVES if (mask >> m) & 1]
        return [m for m in MOVES if Game.is_legal(self.board, m)]

    def max_tile(self) -> int:
        return max(max(row) for row in self.board)

    def empty_cells(self) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(SIZE) for c in range(SIZE) if self.board[r][c] == 0]

    # ---------------------------------------------------------------- actions
    def step(self, move: int) -> Tuple[Board, int, bool]:
        """Play ``move``. Returns (board copy, reward, done).

        reward is the score gained by merges. An illegal move (one that does
        not change the board) is a no-op: reward 0, no spawn, not recorded.
        """
        if move not in MOVES:
            raise ValueError(f"invalid move {move!r}; expected 0..3")
        if self.done:
            return copy_board(self.board), 0, True
        new_board, gained = Game.slide(self.board, move)
        if new_board == self.board:
            return copy_board(self.board), 0, self.done
        self.board = new_board
        self.score += gained
        self.moves.append(move)
        self._spawn()
        self.done = not self.legal_moves()
        return copy_board(self.board), gained, self.done

    def _spawn(self) -> None:
        empties = self.empty_cells()
        if not empties:
            return
        # Fixed RNG call order (cell, then value) keeps replays reproducible.
        r, c = empties[self.rng.randrange(len(empties))]
        self.board[r][c] = 2 if self.rng.random() < PROB_TWO else 4

    def __str__(self) -> str:
        return "\n".join(" ".join(f"{v:6d}" for v in row) for row in self.board)
