"""Run explicitly: python -m pytest agents/deeprl/test_deeprl.py."""
import copy
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from engine.game import Game
from .model import QNetwork, encode, transform, action_map
from .env import GameEnv
from .agent import DQNAgent


def test_encoding_large_tiles():
    board = [[0, 2, 32768, 65536], [131072, 4, 8, 16], [0]*4, [0]*4]
    encoded = encode(board)
    assert encoded.shape == (18, 4, 4)
    assert np.all(encoded.sum(0) == 1)
    assert encoded[16, 0, 3] == 1
    assert encoded[15, 0, 2] == 1
    assert encoded[17, 1, 0] == 1


@pytest.mark.parametrize("symmetry", range(8))
def test_symmetry_commutes_with_all_slides(symmetry):
    board = [[2, 2, 4, 0], [8, 0, 8, 4], [16, 32, 0, 2], [0, 4, 4, 8]]
    transformed = transform(np.array(board), symmetry).tolist()
    for move in range(4):
        after, reward = Game.slide(board, move)
        transformed_after, other_reward = Game.slide(transformed, int(action_map(symmetry)[move]))
        assert transformed_after == transform(np.array(after), symmetry).tolist()
        assert reward == other_reward


def test_env_matches_engine_and_terminal_board():
    env = GameEnv(8)
    game = Game(8)
    state, info = env.reset(8)
    for _ in range(20):
        legal = game.legal_moves()
        assert np.flatnonzero(info["action_mask"]).tolist() == legal
        move = legal[0]
        board, reward, done = game.step(move)
        state, other_reward, other_done, truncated, info = env.step(move)
        assert (reward, done) == (other_reward, other_done)
        assert not truncated
        assert np.array_equal(state, encode(board))
    dead = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
    _, info = env.reset(9, dead)
    assert not info["action_mask"].any()
    assert env.step(0)[2]


def test_agent_legal_pure_reproducible(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(1)
    path = tmp_path / "test.pt"
    torch.save({"model": QNetwork().state_dict()}, path)
    agent = DQNAgent(str(path))
    sequences = []
    for _ in range(2):
        game = Game(10)
        agent.reset(10)
        moves = []
        for _ in range(80):
            if game.done:
                break
            before, rng = copy.deepcopy(game.board), game.rng.getstate()
            move = agent.choose_move(game)
            assert move in game.legal_moves()
            assert game.board == before and game.rng.getstate() == rng
            moves.append(move)
            game.step(move)
        sequences.append(moves)
    assert sequences[0] == sequences[1]


def test_missing_checkpoint_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="No trained checkpoint"):
        DQNAgent(str(tmp_path / "missing.pt"))
