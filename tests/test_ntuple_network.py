"""Tests for agents/ntuple/network.py (n-tuple value network)."""

import numpy as np
import pytest

from agents.ntuple import network as nn
from agents.ntuple.board import from_exponents, symmetries


def random_boards(n: int, seed: int = 0, high: int = 17) -> np.ndarray:
    """Random exponent boards with values 0..high-1 (17 -> includes 65,536)."""
    return np.random.default_rng(seed).integers(0, high, size=(n, 16)).astype(np.uint8)


def distinct_boards(n: int, seed: int = 0) -> np.ndarray:
    """Boards holding each exponent 0..15 exactly once. With every cell
    different, the 32 lookups always hit 32 different table entries."""
    rng = np.random.default_rng(seed)
    return np.array([rng.permutation(16) for _ in range(n)], dtype=np.uint8)


@pytest.fixture(scope="module")
def random_net():
    """A network with random weights (zero weights would make most checks trivial)."""
    rng = np.random.default_rng(2048)
    w = rng.random((nn.N_TUPLES, nn.TABLE_SIZE), dtype=np.float32) - np.float32(0.5)
    return nn.NTupleNetwork(weights=w)


# ----------------------------------------------------------------- layout
def test_table_layout_and_memory():
    net = nn.NTupleNetwork()
    assert net.weights.dtype == np.float32
    assert net.weights.shape == (4, 16**6)
    assert net.weights.nbytes == 268_435_456
    assert nn.N_LOOKUPS == 32


def test_index_packs_six_exponents_in_base_16():
    exps = np.arange(16, dtype=np.uint8)
    # Tuple 0, identity orientation, reads cells 0..5 = exponents 0..5.
    # In base 16 each exponent is one hex digit: 0x012345.
    assert nn.tuple_index(exps, 0, 0) == 0x012345
    exps[0] = 16  # a 65,536 tile is clamped to 15 -> digit F
    assert nn.tuple_index(exps, 0, 0) == 0xF12345


def test_evaluate_matches_slow_reference(random_net):
    w = random_net.weights
    for exps in random_boards(50, seed=1):
        expected = 0.0
        for t in range(4):
            for s in range(8):
                idx = 0
                for cell in nn.CELLS[t, s]:
                    idx = idx * 16 + min(int(exps[cell]), 15)
                expected += float(w[t, idx])
        assert random_net.evaluate(exps) == pytest.approx(expected, rel=1e-9, abs=1e-9)


# ------------------------------------------------------------- symmetries
def test_evaluate_is_identical_for_all_8_symmetries(random_net):
    for exps in random_boards(300, seed=2):
        values = [random_net.evaluate(sym) for sym in symmetries(exps)]
        # Same 32 weights, summed in a different order -> equal up to float rounding.
        assert values == pytest.approx([values[0]] * 8, rel=1e-9, abs=1e-9)


# ----------------------------------------------------------------- update
@pytest.mark.parametrize("delta", [1.0, -0.37, 12.5])
def test_update_changes_value_by_exactly_delta(random_net, delta):
    net = nn.NTupleNetwork()
    # Start from random weights for the entries these boards use.
    for exps in distinct_boards(100, seed=3):
        for t in range(4):
            for s in range(8):
                idx = nn.tuple_index(exps, t, s)
                net.weights[t, idx] = random_net.weights[t, idx]
    for exps in distinct_boards(100, seed=3):
        before = net.evaluate(exps)
        net.update(exps, delta)
        assert net.evaluate(exps) - before == pytest.approx(delta, rel=1e-5, abs=1e-6)


def test_update_on_fully_symmetric_board_counts_shared_entries():
    """Documents the collision case: on an empty board all 8 orientations of a
    tuple hit the same entry, so it gets 8 * d/32 and is summed 8 times:
    4 tuples x 8 x (8 * d/32) = 8d."""
    net = nn.NTupleNetwork()
    empty = np.zeros(16, dtype=np.uint8)
    net.update(empty, 1.0)
    assert net.evaluate(empty) == pytest.approx(8.0)


def test_evaluate_is_pure(random_net):
    exps = random_boards(1, seed=4)[0]
    before = exps.copy()
    a = random_net.evaluate(exps)
    assert random_net.evaluate(exps) == a
    assert np.array_equal(exps, before)


# ------------------------------------------------------------------- init
def test_constant_init():
    net = nn.NTupleNetwork(init_value=16.0)
    # init_value is the starting value of a whole board: each weight is 16 / 32.
    assert float(net.weights.min()) == float(net.weights.max()) == 0.5
    assert net.evaluate(random_boards(1, seed=5)[0]) == pytest.approx(16.0)


# -------------------------------------------------------------- save/load
def test_save_load_gives_identical_evaluations(random_net, tmp_path):
    path = tmp_path / "net.ntw"
    random_net.meta["games"] = 123
    random_net.save(path)
    loaded = nn.NTupleNetwork.load(path)
    assert np.array_equal(loaded.weights, random_net.weights)
    assert loaded.meta["games"] == 123
    for exps in random_boards(1_000, seed=6):
        assert loaded.evaluate(exps) == random_net.evaluate(exps)


def test_saved_file_is_compact(random_net, tmp_path):
    path = tmp_path / "net.ntw"
    random_net.save(path)
    size = path.stat().st_size
    assert random_net.weights.nbytes < size < random_net.weights.nbytes + 4096


def test_load_rejects_bad_files(tmp_path):
    bad_magic = tmp_path / "bad.ntw"
    bad_magic.write_bytes(b"XXXX" + bytes(100))
    with pytest.raises(ValueError, match="not an n-tuple"):
        nn.NTupleNetwork.load(bad_magic)

    header = b'{"version": 999}'
    bad_version = tmp_path / "v.ntw"
    bad_version.write_bytes(b"NTUP" + len(header).to_bytes(4, "little") + header)
    with pytest.raises(ValueError, match="version"):
        nn.NTupleNetwork.load(bad_version)


# ------------------------------------------------------------------ 65536
def test_board_with_65536_evaluates(random_net):
    board = [[65536, 32768, 16384, 8192], [2, 4, 8, 16], [0, 0, 2, 0], [0, 0, 0, 4]]
    value = random_net.evaluate(board)  # engine-format board works too
    assert np.isfinite(value)
    # Clamping means 65,536 is read like 32,768.
    clamped = [row[:] for row in board]
    clamped[0][0] = 32768
    assert value == random_net.evaluate(clamped)


def test_engine_board_and_exponents_agree(random_net):
    for exps in random_boards(20, seed=7):
        assert random_net.evaluate(from_exponents(exps)) == random_net.evaluate(exps)
