
import pytest

from agents.baselines.corner_agent import CornerAgent
from engine.game import Game


def test_corner_agent_follows_priority():
    agent = CornerAgent(order=(2, 3, 1, 0))

    # Down is preferred, so it should be chosen if legal.
    game = Game(seed=0)

    move = agent.choose_move(game)

    assert move in game.legal_moves()


def test_corner_agent_uses_custom_order():
    agent = CornerAgent(order=(3, 2, 1, 0))

    # Check the configured order directly.
    assert agent.order == (3, 2, 1, 0)


def test_corner_agent_rejects_invalid_order():
    with pytest.raises(ValueError):
        CornerAgent(order=(2, 3, 1, 1))
