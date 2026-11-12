import json

import pytest

from engine.game import Game
from eval.replay import replay
from eval.run import AGENTS, main, play_game


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_agent_only_returns_legal_moves(name):
    agent = AGENTS[name]()
    for seed in range(3):
        game = Game(seed)
        agent.reset(seed)
        while not game.done:
            legal = game.legal_moves()
            before = [row[:] for row in game.board]
            move = agent.choose_move(game)
            assert move in legal, f"{name} chose {move}, legal {legal}"
            assert game.board == before, f"{name} mutated the game"
            game.step(move)


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_agent_is_reproducible(name):
    a = play_game(AGENTS[name](), seed=11)
    b = play_game(AGENTS[name](), seed=11)
    assert (a["score"], a["moves"]) == (b["score"], b["moves"])


def test_benchmark_writes_jsonl_and_replays(tmp_path, capsys):
    summary = main(["--agent", "greedy", "--games", "3", "--seed0", "0", "--out-dir", str(tmp_path)])
    lines = (tmp_path / "greedy.jsonl").read_text().splitlines()
    assert len(lines) == 3 and summary["games"] == 3
    for line in lines:
        rec = json.loads(line)
        assert {"seed", "score", "max_tile", "moves", "ms_per_move"} <= rec.keys()
        game = replay(rec)
        assert game.score == rec["score"]
        assert game.max_tile() == rec["max_tile"]
    out = capsys.readouterr().out
    assert "mean score" in out and "reach 65536" in out
