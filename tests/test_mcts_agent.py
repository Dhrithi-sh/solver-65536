
import pytest

from agents.baselines.mcts_agent import MCTSAgent, MCTSNode
from engine.game import Game, MOVES


def test_player_expansion_creates_legal_moves():
    game = Game(seed=0)
    agent = MCTSAgent()
    root = MCTSNode(game.board, "player")

    agent.expand_player_node(root)

    assert len(root.children) == len(game.legal_moves())
    assert all(child.node_type == "chance" for child in root.children)


def test_chance_probabilities_sum_to_one():
    game = Game(seed=0)
    agent = MCTSAgent()
    root = MCTSNode(game.board, "player")

    agent.expand_player_node(root)
    chance_node = root.children[0]
    agent.expand_chance_node(chance_node)

    assert chance_node.children
    assert sum(child.probability for child in chance_node.children) == pytest.approx(1.0)


def test_chance_outcomes_contain_two_and_four():
    game = Game(seed=0)
    agent = MCTSAgent()
    root = MCTSNode(game.board, "player")

    agent.expand_player_node(root)
    chance_node = root.children[0]
    agent.expand_chance_node(chance_node)

    spawned_tiles = {
        max(max(row) for row in child.board)
        for child in chance_node.children
    }

    assert 2 in spawned_tiles
    assert 4 in spawned_tiles


def test_backpropagation_updates_ancestors():
    game = Game(seed=0)
    agent = MCTSAgent()
    root = MCTSNode(game.board, "player")

    agent.expand_player_node(root)
    child = root.children[0]
    agent.expand_chance_node(child)
    outcome = child.children[0]

    agent.backpropagate(outcome, 960.0)

    assert root.visits == 1
    assert child.visits == 1
    assert outcome.visits == 1
    assert root.total_reward == 960.0
    assert child.total_reward == 960.0


def test_choose_move_is_legal_and_does_not_change_board():
    game = Game(seed=0)
    original_board = [row[:] for row in game.board]
    agent = MCTSAgent(iterations=10, rollout_depth=20)

    move = agent.choose_move(game)

    assert move in game.legal_moves()
    assert game.board == original_board


def test_choose_move_is_reproducible():
    game = Game(seed=0)
    agent = MCTSAgent(iterations=10, rollout_depth=20)

    agent.reset(42)
    first_move = agent.choose_move(game)

    agent.reset(42)
    second_move = agent.choose_move(game)

    assert first_move == second_move


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        MCTSAgent(iterations=0)

    with pytest.raises(ValueError):
        MCTSAgent(rollout_depth=0)

    with pytest.raises(ValueError):
        MCTSAgent(exploration_constant=-1)
