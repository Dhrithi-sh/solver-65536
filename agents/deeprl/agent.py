"""Checkpoint-backed agent implementing the shared choose_move(game) interface."""
import os
from pathlib import Path
import torch
from agents.base import Agent
from engine.game import Game
from .model import QNetwork, encode


class DQNAgent(Agent):
    name = "dqn"

    def __init__(self, checkpoint: str | None = None, device: str = "cpu") -> None:
        path = Path(checkpoint or os.environ.get("DQN_CHECKPOINT", "weights/dqn.pt"))
        if not path.is_file():
            raise FileNotFoundError(f"No trained checkpoint at {path}. Run python -m agents.deeprl.train first.")
        self.device = torch.device(device)
        self.network = QNetwork().to(self.device)
        saved = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(saved["model"])
        self.network.eval()

    def reset(self, seed: int | None = None) -> None:
        # Greedy evaluation has no exploration or random tie breaking.
        pass

    def move_scores(self, game: Game) -> list[float]:
        """Q estimates for P8's viewer; illegal moves are negative infinity."""
        with torch.inference_mode():
            board = torch.from_numpy(encode(game.board)).unsqueeze(0).to(self.device)
            scores = self.network(board)[0].cpu().tolist()
        legal = game.legal_moves()
        return [score if move in legal else float("-inf") for move, score in enumerate(scores)]

    def choose_move(self, game: Game) -> int:
        if not game.legal_moves():
            raise ValueError("Cannot choose a move on a terminal board")
        scores = self.move_scores(game)
        return max(range(4), key=lambda move: scores[move])
