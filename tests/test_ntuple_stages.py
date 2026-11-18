"""Tests for multi-stage networks (--stages) and optimistic initialisation (--init-value)."""

import json

import numpy as np
import pytest

from agents.ntuple import train as tr
from agents.ntuple.agent import NTupleAgent, NTupleEvaluator
from agents.ntuple.board import from_exponents, to_exponents
from agents.ntuple.network import FORMAT_MAGIC, NTupleNetwork, StagedNetwork, tuple_index
from engine.game import Game


def board_with_max(tile: int) -> np.ndarray:
    """A busy board whose biggest tile is ``tile``."""
    exps = np.array([1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 1, 2, 3, 0, 0, 0], dtype=np.uint8)
    exps[5] = tile.bit_length() - 1
    return exps


def has_32_distinct_lookups(exps) -> bool:
    return len({(t, tuple_index(exps, t, s)) for t in range(4) for s in range(8)}) == 32


# Max tile 8,192 (stage 1 of a 16,384 split), all 32 lookups distinct.
STAGE1_BOARD = np.array([2, 11, 3, 10, 0, 4, 7, 5, 0, 12, 6, 9, 13, 8, 1, 1], dtype=np.uint8)
# Max tile 32,768 (stage 2), every cell different.
STAGE2_BOARD = np.random.default_rng(5).permutation(16).astype(np.uint8)


# ------------------------------------------------------------ stage choice
@pytest.mark.parametrize(
    "tile, stage",
    [(2, 0), (2048, 0), (8192, 0), (16384, 1), (32768, 1), (65536, 1)],
)
def test_stage_switches_at_the_stage_tile(tile, stage):
    net = StagedNetwork(2, [16384])
    assert net.stage_of(board_with_max(tile)) == stage
    assert net.stage_of(from_exponents(board_with_max(tile))) == stage  # engine board too


def test_three_stages():
    net = StagedNetwork(3, [8192, 16384])
    assert [net.stage_of(board_with_max(t)) for t in (4096, 8192, 16384, 65536)] == [0, 1, 2, 2]


def test_single_stage_is_always_stage_0():
    net = StagedNetwork()
    assert net.n_stages == 1 and net.stage_of(board_with_max(65536)) == 0


@pytest.mark.parametrize("n, tiles", [(2, []), (2, [1000]), (3, [16384, 8192]), (1, [16384])])
def test_bad_stage_tiles_are_rejected(n, tiles):
    with pytest.raises(ValueError):
        StagedNetwork(n, tiles)


def test_each_board_is_scored_by_its_own_stage():
    net = StagedNetwork(2, [16384])
    net.stages[1].update(STAGE2_BOARD, 100.0)  # only stage 2's tables change
    assert net.evaluate(STAGE2_BOARD) == pytest.approx(100.0, rel=1e-5)
    assert net.stages[0].evaluate(STAGE2_BOARD) == 0.0
    assert net.evaluate(STAGE1_BOARD) == net.stages[0].evaluate(STAGE1_BOARD) == 0.0
    ev = NTupleEvaluator(network=net)  # the evaluator picks the stage itself
    assert ev.stage_of(from_exponents(STAGE2_BOARD)) == 1
    assert ev.evaluate(from_exponents(STAGE2_BOARD)) == pytest.approx(100.0, rel=1e-5)


def test_td_target_crosses_the_stage_boundary():
    """A (stage 1) is followed by B (stage 2, the last afterstate). alpha = 0.5,
    the move into B scored 4, and stage 2 already values B at 10.

    B:  target 0               -> V2(B) = 10 + 0.5 * (0 - 10) = 5
    A:  target 4 + V2(B) = 9   -> V1(A) =  0 + 0.5 * (9 - 0)  = 4.5
    """
    assert has_32_distinct_lookups(STAGE1_BOARD) and has_32_distinct_lookups(STAGE2_BOARD)
    net = StagedNetwork(2, [16384])
    net.stages[1].update(STAGE2_BOARD, 10.0)
    tr.learn_backward(net.weights, net.stage_exps, np.stack([STAGE1_BOARD, STAGE2_BOARD]),
                      np.array([0, 4], dtype=np.int64), 2, 0.5)
    assert net.stages[1].evaluate(STAGE2_BOARD) == pytest.approx(5.0, rel=1e-5)
    assert net.stages[0].evaluate(STAGE1_BOARD) == pytest.approx(4.5, rel=1e-5)


# --------------------------------------------------------------- warm start
MERGE_TO_256 = [[128, 128, 2, 4], [8, 16, 32, 64], [0, 0, 0, 0], [0, 0, 0, 0]]


def test_training_warm_starts_a_new_stage_from_the_previous_one():
    net = StagedNetwork(2, [256])
    net.stages[0].update(STAGE1_BOARD, 7.0)
    started = np.array([1, 0], dtype=np.uint8)
    tr.choose_move(net.weights, net.stage_exps, started, to_exponents(MERGE_TO_256))
    assert started[1] == 1
    assert np.array_equal(net.weights[1], net.weights[0])


def test_agent_never_warm_starts():
    net = StagedNetwork(2, [256])
    net.stages[0].update(STAGE1_BOARD, 7.0)
    NTupleAgent(NTupleEvaluator(network=net)).choose_move(Game(0, board=MERGE_TO_256))
    assert not net.weights[1].any()


# -------------------------------------------------------- optimistic init
@pytest.mark.parametrize("value", [0.0, 320.0, 100_000.0])
def test_init_value_gives_that_value_on_an_empty_board(value):
    empty = np.zeros(16, dtype=np.uint8)
    assert NTupleNetwork(init_value=value).evaluate(empty) == pytest.approx(value, rel=1e-6)
    staged = StagedNetwork(2, [16384], init_value=value)
    assert staged.evaluate(empty) == pytest.approx(value, rel=1e-6)
    assert staged.evaluate(board_with_max(32768)) == pytest.approx(value, rel=1e-6)  # stage 2 too
    if value:
        assert float(staged.weights[0, 0, 0]) == pytest.approx(value / 32)


# --------------------------------------------------------------- save/load
def test_staged_save_load_round_trip(tmp_path):
    net = StagedNetwork(2, [16384], meta={"games": 5})
    net.stages[1].update(STAGE2_BOARD, 3.0)
    net.save(tmp_path / "s.ntw")
    loaded = StagedNetwork.load(tmp_path / "s.ntw")
    assert loaded.n_stages == 2 and loaded.stage_tiles == [16384]
    assert np.array_equal(loaded.weights, net.weights)
    assert loaded.meta["games"] == 5
    with pytest.raises(ValueError, match="StagedNetwork.load"):
        NTupleNetwork.load(tmp_path / "s.ntw")


def test_version_1_files_still_load(tmp_path):
    """Weights saved before multi-stage support (format v1) load as one stage."""
    path = tmp_path / "v1.ntw"
    header = {"format": "ntuple", "version": 1, "dtype": "<f4", "shape": [4, 16**6],
              "tuples": [[0, 1, 2, 3, 4, 5], [4, 5, 6, 7, 8, 9], [0, 1, 2, 4, 5, 6], [4, 5, 6, 8, 9, 10]],
              "meta": {"games": 1}}
    raw = json.dumps(header).encode()
    raw += b" " * ((-(8 + len(raw))) % 64)
    w = np.zeros((4, 16**6), dtype="<f4")
    w[0, 0] = 1.5
    with path.open("wb") as f:
        f.write(FORMAT_MAGIC + len(raw).to_bytes(4, "little") + raw)
        w.tofile(f)
    net = StagedNetwork.load(path)
    assert net.n_stages == 1 and net.weights[0, 0, 0] == 1.5 and net.meta["games"] == 1
    assert NTupleNetwork.load(path).weights[0, 0] == 1.5


# ------------------------------------------------------------- end to end
def test_multistage_training_run(tmp_path):
    """A low stage tile (256) so short games actually reach stage 2."""
    tr.train(30, seed=1, out=tmp_path, log_every=10, checkpoint_every=1_000, verbose=False,
             stages=2, stage_tiles=[256])
    net = StagedNetwork.load(tmp_path / "latest.ntw")
    assert net.n_stages == 2 and net.stage_tiles == [256]
    assert net.meta["stages_started"] == [1, 1]
    assert not np.array_equal(net.weights[0], net.weights[1])  # both stages learned separately
    ev = NTupleEvaluator(tmp_path / "latest.ntw")
    assert ev.stage_of(from_exponents(board_with_max(512))) == 1


def test_optimistic_training_run_keeps_unvisited_boards_at_init(tmp_path):
    tr.train(20, seed=1, out=tmp_path, log_every=10, checkpoint_every=1_000, verbose=False,
             init_value=50_000.0)
    net = StagedNetwork.load(tmp_path / "latest.ntw")
    assert net.meta["init_value"] == 50_000.0
    never_seen = np.full(16, 15, dtype=np.uint8)  # a board of sixteen 32,768s
    assert net.evaluate(never_seen) == pytest.approx(50_000.0, rel=1e-5)


def test_resume_keeps_stages(tmp_path):
    tr.train(10, seed=2, out=tmp_path, log_every=5, checkpoint_every=1_000, verbose=False,
             stages=2, stage_tiles=[256])
    result = tr.train(20, out=tmp_path, log_every=5, checkpoint_every=1_000, verbose=False,
                      resume=tmp_path / "latest.ntw")
    assert result["net"].n_stages == 2 and result["net"].stage_tiles == [256]
