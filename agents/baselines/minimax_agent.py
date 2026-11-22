
from __future__ import annotations

from agents.base import Agent, ValueFunction
from engine.game import Board, Game, MOVES, SIZE, copy_board


class MinimaxAgent(Agent):
    """Minimax search treating tile spawns as an adversary."""

    name = "minimax"

    def __init__(self, depth: int = 2, evaluator: ValueFunction | None = None):
        if depth < 1:
            raise ValueError("depth must be at least 1")

        self.depth = depth
        self.evaluator = evaluator

    def evaluate(self, board: Board) -> float:
        if self.evaluator is not None:
            return self.evaluator.evaluate(board)

        return float(max(max(row) for row in board))

    def spawn_outcomes(self, board: Board) -> list[Board]:
        outcomes = []

        for r in range(SIZE):
            for c in range(SIZE):
                if board[r][c] == 0:
                    for tile in (2, 4):
                        new_board = copy_board(board)
                        new_board[r][c] = tile
                        outcomes.append(new_board)

        return outcomes

    def minimax(self, board: Board, depth: int) -> float:
        legal_moves = [
            move for move in MOVES
            if Game.is_legal(board, move)
        ]

        if depth == 0 or not legal_moves:
            return self.evaluate(board)

        best_value = float("-inf")

        for move in legal_moves:
            new_board, gained = Game.slide(board, move)
            outcomes = self.spawn_outcomes(new_board)

            if not outcomes:
                future_value = self.evaluate(new_board)
            else:
                future_value = min(
                    self.minimax(spawn_board, depth - 1)
                    for spawn_board in outcomes
                )

            if self.evaluator is None:
                value = gained + future_value
            else:
                value = future_value

            best_value = max(best_value, value)

        return best_value

    def choose_move(self, game: Game) -> int:
        legal_moves = game.legal_moves()
        assert legal_moves, "choose_move called on a finished game"

        best_move = legal_moves[0]
        best_value = float("-inf")

        for move in legal_moves:
            new_board, gained = Game.slide(game.board, move)
            outcomes = self.spawn_outcomes(new_board)

            if not outcomes:
                future_value = self.evaluate(new_board)
            else:
                future_value = min(
                    self.minimax(spawn_board, self.depth - 1)
                    for spawn_board in outcomes
                )

            if self.evaluator is None:
                value = gained + future_value
            else:
                value = future_value

            if value > best_value:
                best_value = value
                best_move = move

        return best_move
