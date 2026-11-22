
from __future__ import annotations

from agents.base import Agent
from engine.game import Game


class CornerAgent(Agent):
    """Chooses the first legal move in a fixed priority order."""

    name = "corner"

    def __init__(self, order: tuple[int, ...] = (2, 3, 1, 0)):
        if set(order) != {0, 1, 2, 3} or len(order) != 4:
            raise ValueError("order must contain each move exactly once")

        self.order = order

    def choose_move(self, game: Game) -> int:
        legal = game.legal_moves()

        for move in self.order:
            if move in legal:
                return move

        raise RuntimeError("choose_move called on a finished game")
