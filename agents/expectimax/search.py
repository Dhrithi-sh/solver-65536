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
                 adaptive=False, name=None):
        self.evaluator = evaluator      # anything with .evaluate(board) -> float
        self.depth = depth
        self.cutoff = cutoff            # prune branches rarer than this
        self.use_tt = use_tt            # transposition table on/off
        self.adaptive = adaptive        # search deeper when the board is dangerous
        self.name = name or f"expectimax-d{depth}"
        self._tt = {}
        self.nodes = 0                  # for the speed experiment

    def reset(self, seed):
        self._tt.clear()
    def choose_move(self, game):
        values = self.move_values(game.board)   # check the attribute name in engine/game.py
        if not values:
            return game.legal_moves()[0]
        return max(values, key=values.get)
    def move_values(self, board, depth=None):
        """Value of every legal move"""
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
        return self.depth                    

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
            best = self.evaluator.evaluate(board)
        if self.use_tt:
            self._tt[key] = best
        return best

    def _chance(self, after, depth, prob):
        self.nodes += 1
        if prob < self.cutoff:                  # too rare to matter: treat as leaf
            return self.evaluator.evaluate(after)
        cells = _empties(after)
        total = 0.0
        for r, c in cells:
            for tile, p in SPAWNS:
                child = _with_tile(after, r, c, tile)
                if depth == 1:
                    v = self.evaluator.evaluate(child)
                else:
                    v = self._max(child, depth - 1, prob * p / len(cells))
                total += p * v
        return total / len(cells)               # average over cells, p already weights tiles

