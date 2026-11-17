"""Expectimax search agent (P3).

Max nodes: we pick a move. Chance nodes: a tile spawns on a random empty cell.
Depth counts our moves (depth 1 = one move + one spawn, then evaluate).
"""
from agents.base import Agent
from engine.game import Game

SPAWNS = ((2, 0.9), (4, 0.1))


def _key(board):
    return tuple(tuple(r) for r in board)


def _empties(board):
    return [(r, c) for r in range(4) for c in range(4) if board[r][c] == 0]


def _with_tile(board, r, c, v):
    new = [list(row) for row in board]
    new[r][c] = v
    return new


class ExpectimaxAgent(Agent):
    def __init__(self, evaluator, depth=2, cutoff=1e-4, use_tt=True,
                 adaptive=False, dead_value=None, name=None):
        self.evaluator = evaluator      # anything with .evaluate(board) -> float
        self.depth = depth
        self.cutoff = cutoff            # prune branches rarer than this
        self.use_tt = use_tt            # transposition table on/off
        self.adaptive = adaptive        # search deeper when the board is dangerous
        self.dead_value = dead_value    # score for "no legal move" boards (None = ask evaluator)
        self.name = name or f"expectimax-d{depth}"
        self._tt = {}                   # (board, depth) -> value, cleared every move
        self._eval_cache = {}           # board -> evaluator score, kept across moves
        self.nodes = 0                  # for the speed experiment

    def reset(self, seed):
        self._tt.clear()

    def choose_move(self, game):
        values = self.move_values(game.board)
        if not values:
            return game.legal_moves()[0]
        return max(values, key=values.get)

    def move_values(self, board, depth=None):
        """Value of every legal move. P8's 'explain the move' view can reuse this."""
        depth = depth or self._pick_depth(board)
        self._tt.clear()
        self.nodes = 0
        out = {}
        for move in range(4):
            after, _ = Game.slide(board, move)
            if _key(after) == _key(board):      # illegal: nothing moved
                continue
            out[move] = self._chance(after, depth, 1.0)
        return out

    def _pick_depth(self, board):
        if not self.adaptive:
            return self.depth
        empties = sum(1 for row in board for v in row if v == 0)
        distinct = len({v for row in board for v in row if v})
        extra = (empties <= 4) + (distinct >= 9)   # crowded board, many different tiles
        return min(self.depth + extra, 4)

    def _eval(self, board):
        """Evaluator call with a cache: the same boards reach the leaves again and again."""
        k = _key(board)
        v = self._eval_cache.get(k)
        if v is None:
            v = self.evaluator.evaluate(board)
            if len(self._eval_cache) > 500_000:     # keep memory bounded
                self._eval_cache.clear()
            self._eval_cache[k] = v
        return v

    def _max(self, board, depth, prob):
        key = (_key(board), depth)
        if self.use_tt and key in self._tt:
            return self._tt[key]
        best = None
        for move in range(4):
            after, _ = Game.slide(board, move)
            if _key(after) == _key(board):
                continue
            v = self._chance(after, depth, prob)
            if best is None or v > best:
                best = v
        if best is None:                        # no legal move: game over
            best = self.dead_value if self.dead_value is not None else self._eval(board)
        if self.use_tt:
            self._tt[key] = best
        return best

    def _chance(self, after, depth, prob):
        self.nodes += 1
        if prob < self.cutoff:                  # too rare to matter: treat as leaf
            return self._eval(after)
        cells = _empties(after)
        total = 0.0
        for r, c in cells:
            for tile, p in SPAWNS:
                child = _with_tile(after, r, c, tile)
                if depth == 1:
                    v = self._eval(child)
                else:
                    v = self._max(child, depth - 1, prob * p / len(cells))
                total += p * v
        return total / len(cells)