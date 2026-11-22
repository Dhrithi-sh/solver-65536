"""Tests for agents/ntuple/board.py, the numba training board core.

The key property: the fast slide must agree exactly with the shared engine.
"""

import numpy as np
import pytest

from agents.ntuple import board as nb
from engine.game import MOVES, Game


def random_boards(n: int, seed: int = 1234) -> np.ndarray:
    """Random exponent boards. Half use small exponents (lots of merges and
    empties), half use the full range 0..16 (incl. 32,768 and 65,536)."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 5, size=(n // 2, 16))
    wide = rng.integers(0, 17, size=(n - n // 2, 16))
    return np.vstack([small, wide]).astype(np.uint8)


def row_board(row):
    return nb.to_exponents([list(row), [0] * 4, [0] * 4, [0] * 4])


# ------------------------------------------------------------- converters
def test_converters_round_trip():
    board = [[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [4096, 8192, 32768, 65536]]
    exps = nb.to_exponents(board)
    assert exps.dtype == np.uint8 and exps.shape == (16,)
    assert list(exps) == list(range(16))[:14] + [15, 16]
    assert nb.from_exponents(exps) == board


def test_to_exponents_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        nb.to_exponents([[3, 0, 0, 0]] + [[0] * 4] * 3)


# ------------------------------------------------------------------ slide
def test_slide_matches_engine_on_10000_random_boards():
    for exps in random_boards(10_000):
        engine_board = nb.from_exponents(exps)
        for move in MOVES:
            want_board, want_reward = Game.slide(engine_board, move)
            got, reward, changed = nb.slide(exps, move)
            assert nb.from_exponents(got) == want_board, (engine_board, move)
            assert reward == want_reward, (engine_board, move)
            assert changed == (want_board != engine_board)


def test_slide_does_not_mutate_input():
    exps = row_board([2, 2, 0, 0])
    before = exps.copy()
    nb.slide(exps, 3)
    assert np.array_equal(exps, before)


def test_no_double_merge_2222():
    got, reward, changed = nb.slide(row_board([2, 2, 2, 2]), 3)  # left
    assert nb.from_exponents(got)[0] == [4, 4, 0, 0]
    assert reward == 8 and changed


def test_two_exponent_15_tiles_merge_to_16():
    exps = np.zeros(16, dtype=np.uint8)
    exps[0] = exps[1] = 15  # two 32,768 tiles
    got, reward, _ = nb.slide(exps, 3)
    assert got[0] == 16 and got[1] == 0
    assert reward == 65536


def test_python_fallback_matches_compiled():
    """The plain-Python path (used if numba is missing) gives the same results."""
    py_slide = getattr(nb.slide, "py_func", nb.slide)
    for exps in random_boards(200, seed=7):
        for move in MOVES:
            a = nb.slide(exps, move)
            b = py_slide(exps, move)
            assert np.array_equal(a[0], b[0]) and a[1] == b[1] and a[2] == b[2]


# ----------------------------------------------------- legality / game over
def test_legal_moves_and_game_over_match_engine():
    for exps in random_boards(3_000, seed=99):
        g = Game(0, board=nb.from_exponents(exps))
        mask = nb.legal_moves(exps)
        assert [m for m in MOVES if mask[m]] == g.legal_moves()
        assert nb.is_game_over(exps) == (not g.legal_moves())


def test_game_over_on_full_checkerboard():
    exps = np.array([1, 2, 1, 2, 2, 1, 2, 1] * 2, dtype=np.uint8)
    assert nb.is_game_over(exps)
    assert not nb.legal_moves(exps).any()


# ------------------------------------------------------------------ spawn
def test_spawn_fills_one_empty_cell_with_2_or_4():
    exps = row_board([2, 4, 8, 0])
    cell = nb.spawn_random(exps, nb.make_rng(5))
    assert cell >= 3 and exps[cell] in (1, 2)
    assert np.count_nonzero(exps) == 4


def test_spawn_on_full_board_returns_minus_one():
    exps = np.ones(16, dtype=np.uint8)
    assert nb.spawn_random(exps, nb.make_rng(0)) == -1


def test_spawn_is_seeded_and_about_90_percent_twos():
    def run(seed):
        state, values, cells = nb.make_rng(seed), [], []
        for _ in range(20_000):
            exps = np.zeros(16, dtype=np.uint8)
            cells.append(nb.spawn_random(exps, state))
            values.append(exps.max())
        return np.array(values), np.array(cells)

    v1, c1 = run(42)
    v2, c2 = run(42)
    assert np.array_equal(v1, v2) and np.array_equal(c1, c2)
    assert 0.88 < np.mean(v1 == 1) < 0.92
    assert set(np.unique(c1)) == set(range(16))  # every cell gets chosen


# ------------------------------------------------------------- symmetries
def test_symmetries_are_valid_distinct_permutations_with_identity():
    perms = nb.SYMMETRIES
    assert perms.shape == (8, 16)
    for p in perms:
        assert sorted(p) == list(range(16))
    assert len({tuple(p) for p in perms}) == 8
    assert list(perms[0]) == list(range(16))


def test_symmetries_of_board_match_numpy_rotations():
    exps = np.arange(16, dtype=np.uint8)  # every cell distinct
    syms = nb.symmetries(exps)
    assert np.array_equal(syms[0], exps)
    grid = exps.reshape(4, 4)
    expected = set()
    for k in range(4):
        r = np.rot90(grid, k)
        expected.add(tuple(r.ravel()))
        expected.add(tuple(np.fliplr(r).ravel()))
    assert {tuple(s) for s in syms} == expected


# ---------------------------------------------------------------- random play
def test_random_play_is_reproducible():
    assert nb.random_play(20, nb.make_rng(3)) == nb.random_play(20, nb.make_rng(3))
