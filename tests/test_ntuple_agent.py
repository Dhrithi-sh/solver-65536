"""Tests for agents/ntuple/agent.py and its eval/run.py registration."""

import json

import numpy as np
import pytest

from agents.ntuple import agent as na
from agents.ntuple.board import make_rng, slide, to_exponents
from agents.ntuple.network import NTupleNetwork
from agents.ntuple.train import MAX_MOVES, play_and_learn
from engine.game import MOVES, Game
from eval.run import AGENTS, main


@pytest.fixture(scope="module")
def trained_net():
    """A network after 50 self-play games: realistic, non-trivial weights."""
    net = NTupleNetwork()
    buf = np.zeros((MAX_MOVES, 16), np.uint8), np.zeros(MAX_MOVES, np.int64)
    state = make_rng(123)
    w3, no_stages, started = net.weights[None], np.zeros(0, np.int64), np.ones(1, np.uint8)
    for _ in range(50):
        play_and_learn(w3, no_stages, started, state, 0.1, *buf)
    return net


def agent_for(net):
    return na.NTupleAgent(evaluator=na.NTupleEvaluator(network=net))


def test_agent_only_returns_legal_moves_and_never_mutates(trained_net):
    agent = agent_for(trained_net)
    for seed in range(3):
        game = Game(seed)
        while not game.done:
            before = [row[:] for row in game.board]
            move = agent.choose_move(game)
            assert move in game.legal_moves()
            assert game.board == before
            game.step(move)


# A board where all four moves are legal and none merges, so the only thing
# separating the moves is V(afterstate).
NO_MERGE_BOARD = [
    [2, 0, 4, 0],
    [0, 8, 0, 0],
    [0, 0, 0, 16],
    [32, 0, 0, 0],
]


@pytest.mark.parametrize("favoured", MOVES)
def test_hand_set_weights_make_agent_pick_that_move(favoured):
    game = Game(0, board=NO_MERGE_BOARD)
    assert game.legal_moves() == [0, 1, 2, 3]
    net = NTupleNetwork()
    after, reward, _ = slide(to_exponents(NO_MERGE_BOARD), favoured)
    assert reward == 0
    net.update(after, 1000.0)  # raise V(afterstate of the favoured move) by ~1000
    agent = agent_for(net)
    assert agent.choose_move(game) == favoured
    values = agent.move_values(game)
    assert max(values, key=values.get) == favoured


def test_value_can_outweigh_immediate_reward():
    """Left/right merge two 2s (reward 4); give 'down' (reward 0) a large value.

    The board is busy with distinct tiles on purpose: on a nearly empty board
    different afterstates share many table entries (all-empty windows), so
    raising one afterstate would raise the others too.
    """
    board = [
        [2, 2, 8, 16],
        [32, 64, 128, 256],
        [512, 0, 1024, 0],
        [0, 4, 0, 2048],
    ]
    game = Game(0, board=board)
    net = NTupleNetwork()
    agent = agent_for(net)
    assert agent.choose_move(game) == 1  # zero weights: merging wins (tie -> right)
    after_down, reward, _ = slide(to_exponents(board), 2)
    assert reward == 0
    net.update(after_down, 100.0)
    assert agent.choose_move(game) == 2


def test_choose_move_matches_move_values(trained_net):
    agent = agent_for(trained_net)
    game = Game(5)
    for _ in range(100):
        if game.done:
            break
        values = agent.move_values(game)
        assert sorted(values) == game.legal_moves()
        best = max(sorted(values), key=lambda m: values[m])  # ties -> lowest move
        assert agent.choose_move(game) == best
        game.step(best)


def test_evaluator_accepts_engine_board(trained_net):
    ev = na.NTupleEvaluator(network=trained_net)
    board = Game(9).board
    assert ev.evaluate(board) == trained_net.evaluate(to_exponents(board))
    assert isinstance(ev.evaluate(board), float)


# ------------------------------------------------------------------ weights
def test_missing_weights_fail_with_clear_message(tmp_path, monkeypatch):
    missing = tmp_path / "nope.ntw"
    with pytest.raises(na.WeightsNotFoundError, match="weights not found") as e:
        na.NTupleEvaluator(missing)
    assert "--weights" in str(e.value) and "NTUPLE_WEIGHTS" in str(e.value)
    monkeypatch.setenv("NTUPLE_WEIGHTS", str(missing))
    with pytest.raises(na.WeightsNotFoundError, match=r"\$NTUPLE_WEIGHTS"):
        AGENTS["ntuple"]()


def test_weights_path_priority(tmp_path, monkeypatch, ntuple_test_weights):
    explicit = tmp_path / "explicit.ntw"
    explicit.write_bytes(b"")
    monkeypatch.setenv("NTUPLE_WEIGHTS", str(ntuple_test_weights))
    assert na.resolve_weights_path(explicit) == explicit
    assert na.resolve_weights_path() == ntuple_test_weights
    monkeypatch.delenv("NTUPLE_WEIGHTS")
    assert na.resolve_weights_path.__defaults__ == (None,)
    if not na.DEFAULT_WEIGHTS.exists():
        with pytest.raises(na.WeightsNotFoundError, match="default location"):
            na.resolve_weights_path()


def test_evaluators_share_loaded_tables(ntuple_test_weights):
    a = na.NTupleEvaluator(ntuple_test_weights)
    b = na.NTupleEvaluator(ntuple_test_weights)
    assert a.network is b.network


def test_eval_run_weights_flag(tmp_path, ntuple_test_weights):
    main(["--agent", "ntuple", "--games", "2", "--seed0", "0",
          "--out-dir", str(tmp_path), "--weights", str(ntuple_test_weights)])
    lines = (tmp_path / "ntuple.jsonl").read_text().splitlines()
    assert [json.loads(l)["agent"] for l in lines] == ["ntuple", "ntuple"]


def test_eval_run_rejects_weights_for_agents_without_them(tmp_path):
    with pytest.raises(SystemExit):
        main(["--agent", "greedy", "--games", "1", "--out-dir", str(tmp_path), "--weights", "x.ntw"])
