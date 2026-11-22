
from __future__ import annotations

import random
from typing import Optional

from agents.base import Agent
from engine.game import Game


class MonteCarloAgent(Agent):
    """Chooses moves by averaging scores from random future games."""

    name = "monte_carlo"

    def __init__(self, rollouts: int = 5, rollout_depth: int = 100):
        if rollouts < 1:
            raise ValueError("rollouts must be at least 1")

        if rollout_depth < 1:
            raise ValueError("rollout_depth must be at least 1")

        self.rollouts = rollouts
        self.rollout_depth = rollout_depth
        self.rng = random.Random(0)

    def reset(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def simulate(self, game: Game, first_move: int) -> float:
        # Give this simulated game its own seed.
        seed = self.rng.randrange(2**32)
        simulated_game = Game(seed=seed, board=game.board)

        # Use a separate random generator for the rollout policy.
        policy_rng = random.Random(seed)

        # Apply the candidate move first.
        simulated_game.step(first_move)

        steps = 1

        # Continue with random legal moves.
        while not simulated_game.done and steps < self.rollout_depth:
            legal_moves = simulated_game.legal_moves()
            move = policy_rng.choice(legal_moves)
            simulated_game.step(move)
            steps += 1

        return float(simulated_game.score)

    def choose_move(self, game: Game) -> int:
        legal_moves = game.legal_moves()
        assert legal_moves, "choose_move called on a finished game"

        best_move = legal_moves[0]
        best_average = float("-inf")

        for move in legal_moves:
            total_score = 0.0

            for _ in range(self.rollouts):
                total_score += self.simulate(game, move)

            average_score = total_score / self.rollouts

            if average_score > best_average:
                best_average = average_score
                best_move = move

        return best_move
