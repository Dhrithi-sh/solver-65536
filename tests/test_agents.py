import json
import os
from importlib.util import find_spec
from pathlib import Path

import pytest

from engine.game import Game
from eval.replay import replay
from eval.run import AGENTS, main


def agent_test_parameters():
    params = []

    dqn_checkpoint = Path(
        os.environ.get("DQN_CHECKPOINT", "weights/dqn.pt")
    )

    cpp_available = find_spec("agents.hybrid._hsearch") is not None

    for name in sorted(AGENTS):
        reason = None

        if name == "dqn" and not dqn_checkpoint.is_file():
            reason = "P6 trained checkpoint has not been supplied yet."

        if name in ("hybrid-d3", "hybrid-65k") and not cpp_available:
            reason = "P7 C++ extension not built; Python fallback is too slow for full-game smoke tests."

        if name in {
            "mcts",
            "minimax",
            "minimax_heuristic",
            "monte_carlo",
        }:
            reason = "P4 search agents have dedicated focused tests."

        if reason:
            params.append(
                pytest.param(name, marks=pytest.mark.skip(reason=reason))
            )
        else:
            params.append(pytest.param(name))

    return params


@pytest.mark.parametrize("name", agent_test_parameters())
def test_agent_only_returns_legal_moves(name):
    agent = AGENTS[name]()

    for seed in range(3):
        game = Game(seed)
        agent.reset(seed)

        for _ in range(10):
            if game.done:
                break

            legal = game.legal_moves()
            before = [row[:] for row in game.board]

            move = agent.choose_move(game)

            assert move in legal, f"{name} chose {move}, legal {legal}"
            assert game.board == before, f"{name} mutated the game"

            game.step(move)


@pytest.mark.parametrize("name", agent_test_parameters())
def test_agent_is_reproducible(name):
    agent_a = AGENTS[name]()
    agent_b = AGENTS[name]()

    game_a = Game(seed=11)
    game_b = Game(seed=11)

    agent_a.reset(11)
    agent_b.reset(11)

    for _ in range(10):
        assert game_a.done == game_b.done

        if game_a.done:
            break

        move_a = agent_a.choose_move(game_a)
        move_b = agent_b.choose_move(game_b)

        assert move_a == move_b, f"{name} produced different moves"

        game_a.step(move_a)
        game_b.step(move_b)

        assert game_a.board == game_b.board
        assert game_a.score == game_b.score


def test_benchmark_writes_jsonl_and_replays(tmp_path, capsys):
    summary = main([
        "--agent", "greedy",
        "--games", "3",
        "--seed0", "0",
        "--out-dir", str(tmp_path),
    ])

    lines = (tmp_path / "greedy.jsonl").read_text().splitlines()

    assert len(lines) == 3
    assert summary["games"] == 3

    for line in lines:
        rec = json.loads(line)

        assert {
            "seed", "score", "max_tile", "moves", "ms_per_move"
        } <= rec.keys()

        game = replay(rec)

        assert game.score == rec["score"]
        assert game.max_tile() == rec["max_tile"]

    out = capsys.readouterr().out

    assert "mean score" in out
    assert "reach 65536" in out