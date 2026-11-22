"""1-ply agent: play the move whose afterstate the evaluator likes best."""

from __future__ import annotations

from agents.base import Agent, ValueFunction
from engine.game import MOVES, Game


class HeuristicAgent(Agent):
    """Scores each legal move's afterstate (board after the slide, before the
    spawn) and plays the best. Ties go to the lowest move index.

    The afterstate is ``Game.slide(board, move)[0]``, which is exactly what
    the engine's ``afterstate`` helper returns; ``slide`` is used directly
    because it is part of the original shared interface.

    The merge reward is deliberately not added to the value: the features
    live on an exponent scale and the reward on a tile-value scale, so adding
    them would let the reward swamp the heuristic late in the game.
    """

    def __init__(self, evaluator: ValueFunction, name: str = "heuristic") -> None:
        self.evaluator = evaluator
        self.name = name

    def choose_move(self, game: Game) -> int:
        board = game.board
        best_move, best_value = None, float("-inf")
        for move in MOVES:
            after, _ = Game.slide(board, move)
            if after == board:  # illegal: same test as Game.is_legal
                continue
            value = self.evaluator.evaluate(after)
            if best_move is None or value > best_value:
                best_move, best_value = move, value
        assert best_move is not None, "choose_move called on a finished game"
        return best_move
