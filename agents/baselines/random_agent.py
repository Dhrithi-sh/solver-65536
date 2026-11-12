from __future__ import annotations

import random
from typing import Optional

from agents.base import Agent
from engine.game import Game


class RandomAgent(Agent):
    """Uniformly random legal move. Seeded per game for reproducibility."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def choose_move(self, game: Game) -> int:
        return self.rng.choice(game.legal_moves())
