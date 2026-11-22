"""Late-game fine-tuning: promotion, position extraction, --init-from / --start-positions."""

from __future__ import annotations

import csv
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pytest

from agents.ntuple import train as tr
from agents.ntuple.board import make_rng
from agents.ntuple.network import NTupleNetwork, StagedNetwork
from engine.game import Game

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _random_game(seed: int) -> dict:
    g, rng = Game(seed), random.Random(seed)
    while not g.done:
        g.step(rng.choice(g.legal_moves()))
    return {"seed": seed, "moves": "".join(map(str, g.moves)), "score": g.score, "max_tile": g.max_tile()}


def test_pick_start_is_uniform_over_games_not_positions():
    pos = np.zeros((1001, 16), dtype=np.uint8)
    offsets = np.array([0, 1, 1001], dtype=np.int64)  # game 0: 1 position, game 1: 1000
    rng = make_rng(3)
    first = sum(tr.pick_start(pos, offsets, rng) == 0 for _ in range(4000))
    assert 1800 < first < 2200


def test_extract_excludes_seed_and_replays(tmp_path):
    ex = _load_script("extract_late_positions")
    games = [_random_game(s) for s in (5, 6, 128)]
    (tmp_path / "logs").mkdir()
    with (tmp_path / "logs" / "a.jsonl").open("w") as f:
        for g in games + games[:1]:  # duplicate game: must add nothing
            f.write(json.dumps(g) + "\n")
    r = ex.extract(tmp_path / "logs", min_tile=8, exclude_seed=128)
    assert r["kept"] and r["held"]
    assert {x["source_seed"] for x in r["kept"]} == {5, 6}
    assert {x["source_seed"] for x in r["held"]} == {128}
    assert r["stats"]["dup_games"] == 1 and r["n_kept_games"] == 2
    for x in r["kept"][:50]:  # each saved board is the replayed state before move_index
        g = next(gg for gg in games if gg["seed"] == x["source_seed"])
        game = Game(g["seed"])
        for c in g["moves"][: x["move_index"]]:
            game.step(int(c))
        assert game.board == x["board"] and max(map(max, x["board"])) >= 8


def test_start_positions_train_only_touches_late_stage(tmp_path):
    ex = _load_script("extract_late_positions")
    (tmp_path / "logs").mkdir()
    with (tmp_path / "logs" / "a.jsonl").open("w") as f:
        for s in (11, 12, 13):
            f.write(json.dumps(_random_game(s)) + "\n")
    r = ex.extract(tmp_path / "logs", min_tile=64, exclude_seed=128)
    positions = tmp_path / "pos.jsonl"
    ex.write_jsonl(positions, r["kept"])

    base = StagedNetwork(2, [64])
    base.weights[0, :, :1000] = np.arange(1000, dtype=np.float32)  # recognisable stage 0
    base.weights[1] = base.weights[0]
    base.meta["stages_started"] = [1, 1]
    init = tmp_path / "init.ntw"
    base.save(init)
    stage0 = base.weights[0].copy()
    del base

    runs = []
    for name in ("a", "b"):
        res = tr.train(30, alpha=0.1, seed=99, out=tmp_path / name, log_every=10, checkpoint_every=1000,
                       verbose=False, init_from=init, start_positions=positions)
        net = res["net"]
        assert np.array_equal(net.weights[0].view(np.uint32), stage0.view(np.uint32))
        assert not np.array_equal(net.weights[1], stage0)
        assert net.meta["start_positions"]["n_games"] == 3
        with (tmp_path / name / "starts.csv").open() as f:
            runs.append([row["row"] for row in csv.DictReader(f)])
        saved = StagedNetwork.load(tmp_path / name / "latest.ntw")
        assert saved.meta["games"] == 30 and saved.meta["init_from"]
    assert len(runs[0]) == 30 and runs[0] == runs[1]  # same seed -> same starts


WEIGHTS_1M = ROOT / "weights" / "ntuple" / "long" / "ntuple-plain-1m.ntw"


@pytest.mark.skipif(not WEIGHTS_1M.exists(), reason="1M weights not downloaded")
def test_promoted_1m_file_evaluates_like_source(tmp_path):
    pr = _load_script("ntuple_promote")
    out = tmp_path / "promoted.ntw"
    pr.main([str(WEIGHTS_1M), str(out), "--check-boards", "1000"])  # main() runs verify()
    assert isinstance(NTupleNetwork.load(WEIGHTS_1M), NTupleNetwork)
