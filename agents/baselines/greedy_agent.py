from __future__ import annotations

from agents.base import Agent
from engine.game import Game


class GreedyAgent(Agent):
    """Picks the legal move with the highest immediate merge score.
    Ties go to the lowest move index (deterministic)."""

    name = "greedy"

    def choose_move(self, game: Game) -> int:
        best_move, best_gain = None, -1
        for move in game.legal_moves():
            _, gained = Game.slide(game.board, move)
            if gained > best_gain:
                best_move, best_gain = move, gained
        assert best_move is not None, "choose_move called on a finished game"
        return best_move
