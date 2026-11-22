
import pytest

from agents.baselines.minimax_agent import MinimaxAgent
from engine.game import Game


def test_spawn_outcomes_count():
    agent = MinimaxAgent(depth=1)
    board = [
        [2, 0, 0, 0],
        [0, 4, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]

    outcomes = agent.spawn_outcomes(board)

    # There are 14 empty cells and 2 possible tile values.
    assert len(outcomes) == 28


def test_spawn_outcomes_add_one_tile():
    agent = MinimaxAgent(depth=1)
    board = [
        [2, 0, 0, 0],
        [0, 4, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]

    outcomes = agent.spawn_outcomes(board)

    for outcome in outcomes:
        # Each outcome must add exactly one tile.
        assert sum(cell != 0 for row in outcome for cell in row) == 3
        assert sum(cell for row in outcome for cell in row) in (8, 10)


def test_spawn_outcomes_do_not_change_original_board():
    agent = MinimaxAgent(depth=1)
    board = [
        [2, 0, 0, 0],
        [0, 4, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    original = [row[:] for row in board]

    agent.spawn_outcomes(board)

    assert board == original


def test_choose_move_is_legal_and_does_not_change_game():
    game = Game(seed=0)
    original_board = [row[:] for row in game.board]
    agent = MinimaxAgent(depth=1)

    move = agent.choose_move(game)

    assert move in game.legal_moves()
    assert game.board == original_board


def test_minimax_rejects_zero_depth():
    with pytest.raises(ValueError):
        MinimaxAgent(depth=0)
