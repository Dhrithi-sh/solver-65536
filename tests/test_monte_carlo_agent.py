
from agents.baselines.monte_carlo_agent import MonteCarloAgent
from engine.game import Game


def test_monte_carlo_returns_legal_move():
    game = Game(seed=0)
    agent = MonteCarloAgent(rollouts=2, rollout_depth=20)
    agent.reset(0)

    move = agent.choose_move(game)

    assert move in game.legal_moves()


def test_monte_carlo_does_not_modify_real_game():
    game = Game(seed=0)
    original_board = [row[:] for row in game.board]
    original_score = game.score
    original_moves = game.moves[:]

    agent = MonteCarloAgent(rollouts=2, rollout_depth=20)
    agent.reset(0)
    agent.choose_move(game)

    assert game.board == original_board
    assert game.score == original_score
    assert game.moves == original_moves


def test_monte_carlo_is_reproducible():
    game1 = Game(seed=0)
    agent1 = MonteCarloAgent(rollouts=2, rollout_depth=20)
    agent1.reset(42)
    move1 = agent1.choose_move(game1)

    game2 = Game(seed=0)
    agent2 = MonteCarloAgent(rollouts=2, rollout_depth=20)
    agent2.reset(42)
    move2 = agent2.choose_move(game2)

    assert move1 == move2


def test_monte_carlo_rejects_invalid_rollouts():
    import pytest

    with pytest.raises(ValueError):
        MonteCarloAgent(rollouts=0)


def test_monte_carlo_rejects_invalid_depth():
    import pytest

    with pytest.raises(ValueError):
        MonteCarloAgent(rollout_depth=0)
