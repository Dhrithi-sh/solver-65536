"""Tests for agents/ntuple/train.py (self-play TD(0) on afterstates)."""

import csv

import numpy as np
import pytest

from agents.ntuple import train as tr
from agents.ntuple.board import make_rng
from agents.ntuple.network import NTupleNetwork


def distinct_board(seed: int) -> np.ndarray:
    """All 16 cells different, so the 32 lookups never share a table entry."""
    return np.random.default_rng(seed).permutation(16).astype(np.uint8)


def test_backward_pass_worked_example():
    """Two afterstates A -> B, zero weights, alpha = 0.5, the move after A scored 4.

    B is last:  target 0,           V(B) = 0 + 0.5 * (0 - 0) = 0
    A:          target 4 + V(B) = 4, V(A) = 0 + 0.5 * (4 - 0) = 2
    """
    net = NTupleNetwork()
    a, b = distinct_board(1), distinct_board(2)
    afterstates = np.stack([a, b])
    rewards = np.array([0, 4], dtype=np.int64)  # rewards[1] = reward of the move that made B
    tr.learn_backward(net.weights, afterstates, rewards, 2, 0.5)
    assert net.evaluate(b) == pytest.approx(0.0)
    assert net.evaluate(a) == pytest.approx(2.0)


def test_last_afterstate_moves_toward_zero():
    net = NTupleNetwork()
    s = distinct_board(3)
    net.update(s, 10.0)  # V(s) = 10
    tr.learn_backward(net.weights, s[None, :], np.zeros(1, dtype=np.int64), 1, 0.1)
    assert net.evaluate(s) == pytest.approx(9.0, rel=1e-5)  # 10 + 0.1 * (0 - 10)


def test_200_game_run_beats_random_and_untrained_play(tmp_path):
    result = tr.train(200, alpha=0.1, seed=10_000_000, out=tmp_path, log_every=100,
                      checkpoint_every=1_000, verbose=False)
    rows = result["rows"]
    trained_mean = np.mean([r["mean_score"] for r in rows])
    random_mean = tr.random_play_scores(200, make_rng(10_000_000)).mean()
    assert trained_mean > 2 * random_mean, (trained_mean, random_mean)

    # Stronger check that it *learns*: the second window beats the same 1-ply
    # player with learning switched off (alpha = 0, i.e. greedy on reward).
    untrained = tr.train(200, alpha=0.0, seed=10_000_000, out=tmp_path / "a0", log_every=100,
                         checkpoint_every=1_000, verbose=False)
    untrained_mean = np.mean([r["mean_score"] for r in untrained["rows"]])
    assert rows[-1]["mean_score"] > 1.3 * untrained_mean, (rows[-1]["mean_score"], untrained_mean)


def test_csv_log_columns_and_checkpoint(tmp_path):
    tr.train(40, out=tmp_path, log_every=20, checkpoint_every=40, verbose=False)
    with (tmp_path / "train_log.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r["games"]) for r in rows] == [20, 40]
    assert list(rows[0].keys()) == tr.CSV_FIELDS
    for r in rows:
        assert 0.0 <= float(r["reach_2048"]) <= 1.0
        assert int(r["n_2048"]) >= int(r["n_4096"]) >= int(r["n_32768"])
    net = NTupleNetwork.load(tmp_path / "latest.ntw")
    assert net.meta["games"] == 40 and net.meta["alpha"] == 0.1


def test_resume_is_identical_to_uninterrupted_run(tmp_path):
    """30 games + resume to 60 == 60 games straight (same weights, same log),
    including a log window (20-40) that straddles the checkpoint."""
    straight = tr.train(60, seed=7, out=tmp_path / "straight", log_every=20,
                        checkpoint_every=1_000, verbose=False)
    tr.train(30, seed=7, out=tmp_path / "split", log_every=20, checkpoint_every=1_000, verbose=False)
    resumed = tr.train(60, out=tmp_path / "split", log_every=20, checkpoint_every=1_000,
                       resume=tmp_path / "split" / "latest.ntw", verbose=False)
    assert np.array_equal(straight["net"].weights, resumed["net"].weights)

    def scores(d):
        with (d / "train_log.csv").open() as f:
            return [(r["games"], r["mean_score"], r["n_2048"]) for r in csv.DictReader(f)]

    assert scores(tmp_path / "straight") == scores(tmp_path / "split")
