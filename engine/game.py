"""Reference 2048 engine (simple, readable, correct; not fast).

Board: 4x4 list of lists of ints. 0 = empty, other cells hold the tile value
(2, 4, 8, ...). Moves: 0 = up, 1 = right, 2 = down, 3 = left.

Spawn rule: after every legal move, one empty cell is chosen uniformly at
random and receives a 2 with probability 0.9 or a 4 with probability 0.1.
All randomness comes from a per-game ``random.Random(seed)``, so a seed plus
a move list fully determines a game (see eval/replay.py).

NOTE (P1): this module is the *reference* implementation. P1 will replace the
internals with a 64-bit bitboard (16 cells x 4 bits, each cell storing the
exponent, with row lookup tables for slides) while keeping this exact public
API. 4-bit cells can only hold exponents 0..15, i.e. tiles up to 2**15 =
32,768, so the 65,536 tile (2**16) does NOT fit and must be handled
explicitly (e.g. a 5th bit / side flag, or treating the merge of two 32,768s
as a special case). The tests in tests/test_engine.py, including the 65,536
merge test, must keep passing against the bitboard version.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

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
        lines = _get_lines(board, move)
        merged = [_merge_line(line) for line in lines]
        new_board = _set_lines([m[0] for m in merged], move)
        return new_board, sum(m[1] for m in merged)

    @staticmethod
    def is_legal(board: Sequence[Sequence[int]], move: int) -> bool:
        new_board, _ = Game.slide(board, move)
        return new_board != [list(r) for r in board]

    # ---------------------------------------------------------------- queries
    def legal_moves(self) -> List[int]:
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
