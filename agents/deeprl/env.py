"""Gym-style wrapper (no Gym dependency) around the team's engine."""
import numpy as np
from engine.game import Board, Game
from .model import encode


class GameEnv:
    def __init__(self, seed: int = 0) -> None:
        self.game = Game(seed)

    def info(self) -> dict:
        mask = np.zeros(4, dtype=bool)
        mask[self.game.legal_moves()] = True
        return {"action_mask": mask, "score": self.game.score,
                "max_tile": self.game.max_tile()}

    def reset(self, seed: int = 0, board: Board | None = None) -> tuple:
        self.game.reset(seed, board)
        return encode(self.game.board), self.info()

    def step(self, action: int) -> tuple:
        board, reward, done = self.game.step(action)
        return encode(board), reward, done, False, self.info()
