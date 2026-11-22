"""P1 Batch 2 tests: from_board, afterstate, empty_cells, spawn_outcomes,
exponents, play_batch. New file; tests/test_engine.py and
tests/test_engine_p1.py are not touched. Plain asserts only.

exponents() tests skip themselves (pytest.skip) if numpy is not installed.
"""

import random

from engine import game as G
from engine.game import DOWN, LEFT, RIGHT, UP, Game, afterstate, empty_cells, exponents, play_batch, spawn_outcomes

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def _need_numpy():
    if np is None:
        import pytest
        pytest.skip("numpy not installed")


def empty():
    return [[0] * 4 for _ in range(4)]


def random_board(rng):
    tiles = [0, 0, 0, 2, 4, 8, 64, 2048, 32768, 65536]
    return [[rng.choice(tiles) for _ in range(4)] for _ in range(4)]


# ------------------------------------------------------------- from_board
def test_from_board_uses_given_board_and_spawns_nothing():
    b = empty(); b[0][0] = b[3][3] = 2
    g = Game.from_board(b, seed=5)
    assert g.board == b and g.score == 0 and g.moves == [] and g.seed == 5 and not g.done


def test_from_board_copies_input():
    b = empty(); b[0][0] = b[0][1] = 2
    g = Game.from_board(b, 1)
    g.step(LEFT)
    assert b[0] == [2, 2, 0, 0]
    b[1][1] = 8
    assert g.board[1][1] != 8 or g.moves  # later edits to b never reach the game
    assert 8 not in [v for row in g.board for v in row]


def test_from_board_default_seed_and_equivalence_with_constructor():
    b = empty(); b[2][2] = 4; b[0][0] = 2
    a, c = Game.from_board(b), Game(0, board=b)
    assert a.seed == 0
    for m in (UP, LEFT, DOWN, RIGHT, UP, LEFT):
        if m in a.legal_moves():
            assert a.step(m) == c.step(m)
    assert (a.board, a.score, a.moves) == (c.board, c.score, c.moves)


def test_from_board_is_deterministic_per_seed():
    b = empty(); b[0][0] = b[0][1] = 2
    runs = []
    for seed in (3, 3, 4):
        g = Game.from_board(b, seed)
        for m in (LEFT, DOWN, RIGHT, UP, LEFT, DOWN):
            g.step(m)
        runs.append(g.board)
    assert runs[0] == runs[1]


def test_from_board_32768_late_game_converts_to_65536():
    b = empty(); b[0][0] = b[0][1] = 32768
    g = Game.from_board(b, 0)
    _, reward, _ = g.step(LEFT)
    assert reward == 65536 and g.max_tile() == 65536


def test_from_board_rejects_wrong_shape():
    for bad in ([[0] * 4] * 3, [[0] * 3] * 4, []):
        try:
            Game.from_board(bad, 0)
        except ValueError:
            continue
        raise AssertionError(f"shape {bad!r} should raise ValueError")


def test_from_board_full_dead_board_is_done():
    dead = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
    assert Game.from_board(dead, 0).done


# ------------------------------------------------------------- afterstate
def test_afterstate_matches_slide_and_all_directions():
    rng = random.Random(9)
    for _ in range(500):
        b = random_board(rng)
        for m in range(4):
            assert afterstate(b, m) == Game.slide(b, m)[0]


def test_afterstate_known_values_and_no_spawn():
    b = [[2, 2, 0, 0], [0] * 4, [0] * 4, [0, 0, 4, 4]]
    after = afterstate(b, LEFT)
    assert after == [[4, 0, 0, 0], [0] * 4, [0] * 4, [8, 0, 0, 0]]
    assert sum(v != 0 for r in after for v in r) == 2   # nothing spawned


def test_afterstate_is_pure():
    b = [[2, 2, 4, 4], [0, 2, 0, 2], [8, 0, 8, 0], [0, 0, 0, 2]]
    snap = [r[:] for r in b]
    for m in range(4):
        afterstate(b, m)
    assert b == snap


def test_afterstate_illegal_move_returns_equal_copy():
    b = [[2, 4, 8, 16], [0] * 4, [0] * 4, [0] * 4]
    after = afterstate(b, UP)
    assert after == b and after is not b and after[0] is not b[0]


def test_afterstate_invalid_move_raises():
    for bad in (-1, 4, None):
        try:
            afterstate(empty(), bad)
        except ValueError:
            continue
        raise AssertionError(f"move {bad!r} should raise ValueError")


def test_afterstate_65536():
    b = empty(); b[0][0] = b[1][0] = 32768
    assert afterstate(b, UP)[0][0] == 65536


# ------------------------------------------------------------ empty_cells
def test_empty_cells_row_major_and_matches_game_method():
    b = [[2, 0, 4, 0], [0] * 4, [8, 8, 8, 8], [0, 2, 0, 2]]
    assert empty_cells(b) == [(0, 1), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (3, 0), (3, 2)]
    assert Game.from_board(b, 0).empty_cells() == empty_cells(b)


def test_empty_cells_full_and_empty_boards():
    assert empty_cells([[2] * 4 for _ in range(4)]) == []
    assert len(empty_cells(empty())) == 16


def test_empty_cells_matches_game_on_random_boards():
    rng = random.Random(2)
    for _ in range(200):
        b = random_board(rng)
        assert empty_cells(b) == Game.from_board(b, 0).empty_cells()


# --------------------------------------------------------- spawn_outcomes
def test_spawn_outcomes_count_probabilities_and_values():
    b = [[2, 4, 8, 16], [32, 64, 128, 256], [2, 4, 8, 16], [0, 0, 0, 2]]
    out = spawn_outcomes(b)
    assert len(out) == 6                           # 3 empty cells x {2, 4}
    assert abs(sum(p for _, p in out) - 1.0) < 1e-12
    probs = sorted({round(p, 12) for _, p in out})
    assert probs == [round(0.1 / 3, 12), round(0.9 / 3, 12)]


def test_spawn_outcomes_each_board_differs_by_one_new_tile():
    b = [[2, 0, 4, 8], [0] * 4, [0] * 4, [16, 2, 0, 4]]
    cells = empty_cells(b)
    out = spawn_outcomes(b)
    assert len(out) == 2 * len(cells)
    for i, (child, prob) in enumerate(out):
        r, c = cells[i // 2]
        value, p = (2, 0.9) if i % 2 == 0 else (4, 0.1)   # order: cell by cell, 2 then 4
        assert child[r][c] == value and abs(prob - p / len(cells)) < 1e-12
        diff = [(x, y) for x in range(4) for y in range(4) if child[x][y] != b[x][y]]
        assert diff == [(r, c)]


def test_spawn_outcomes_full_board_is_empty_list_and_input_untouched():
    full = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
    assert spawn_outcomes(full) == []
    b = [[2, 0, 0, 0]] + [[0] * 4 for _ in range(3)]
    snap = [r[:] for r in b]
    spawn_outcomes(b)
    assert b == snap


def test_spawn_outcomes_agree_with_engine_spawn_frequencies():
    """The engine's real spawn (seeded RNG) should follow these probabilities."""
    b = [[2, 4, 8, 16], [32, 64, 128, 256], [2, 4, 8, 16], [32, 64, 0, 0]]
    expected = {}
    for child, p in spawn_outcomes(b):
        expected[str(child)] = p
    counts = {k: 0 for k in expected}
    n = 20000
    for seed in range(n):
        g = Game.from_board(b, seed)
        g._spawn()
        counts[str(g.board)] += 1
    for k, p in expected.items():
        assert abs(counts[k] / n - p) < 0.015, (k, counts[k] / n, p)


# -------------------------------------------------------------- exponents
def test_exponents_values_shape_dtype():
    _need_numpy()
    b = [[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [4096, 8192, 32768, 65536]]
    e = exponents(b)
    assert e.shape == (4, 4) and e.dtype == np.int32
    assert e.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 15, 16]]


def test_exponents_roundtrip_and_131072():
    _need_numpy()
    rng = random.Random(4)
    for _ in range(200):
        b = random_board(rng)
        e = exponents(b)
        assert [[(1 << int(x)) if x else 0 for x in row] for row in e] == b
    b = empty(); b[0][0] = 131072
    assert exponents(b)[0, 0] == 17


def test_exponents_no_overflow_when_building_indexes():
    _need_numpy()
    b = empty(); b[0] = [65536] * 4          # exponent 16 in every cell
    e = exponents(b)
    assert int(e[0, 0] + 32 * e[0, 1] + 1024 * e[0, 2]) == 16 + 512 + 16384


def test_exponents_returns_copy_independent_of_board():
    _need_numpy()
    b = empty(); b[0][0] = 2
    e = exponents(b)
    e[0, 0] = 9
    assert b[0][0] == 2


def test_exponents_accepts_numpy_ints_and_rejects_bad_cells():
    _need_numpy()
    b = np.array(empty()); b[0, 0] = 4
    assert exponents(b)[0, 0] == 2
    for bad_value in (3, 1, -2, 2.0, "2", None):
        bad = empty(); bad[1][1] = bad_value
        try:
            exponents(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad_value!r} should raise ValueError")
    try:
        exponents([[0] * 4] * 3)
    except ValueError:
        pass
    else:
        raise AssertionError("shape check")


# ------------------------------------------------------------- play_batch
def first_legal(game):
    return game.legal_moves()[0]


def seeded_random_agent(game):
    # Depends only on the board, so results are reproducible in any process.
    legal = game.legal_moves()
    return legal[sum(map(sum, game.board)) % len(legal)]


def test_play_batch_basic_shape_and_order():
    seeds = [5, 1, 9]
    res = play_batch(first_legal, seeds)
    assert [r["seed"] for r in res] == seeds
    for r in res:
        assert {"seed", "score", "max_tile", "moves", "board"} <= r.keys()
        assert isinstance(r["moves"], list) and r["max_tile"] == max(max(row) for row in r["board"])


def test_play_batch_matches_manual_game():
    res = play_batch(seeded_random_agent, [3])[0]
    g = Game(3)
    while not g.done:
        g.step(seeded_random_agent(g))
    assert (res["score"], res["moves"], res["board"], res["max_tile"]) == (g.score, g.moves, g.board, g.max_tile())


def test_play_batch_is_deterministic_and_replayable():
    a = play_batch(seeded_random_agent, range(4))
    b = play_batch(seeded_random_agent, range(4))
    assert a == b
    for rec in a:
        g = Game(rec["seed"])
        for m in rec["moves"]:
            g.step(m)
        assert g.score == rec["score"] and g.board == rec["board"] and g.done


def test_play_batch_parallel_equals_sequential():
    seq = play_batch(seeded_random_agent, range(6), processes=1)
    par = play_batch(seeded_random_agent, range(6), processes=2)
    assert seq == par


def test_play_batch_empty_seed_list():
    assert play_batch(first_legal, []) == []


def test_play_batch_illegal_move_raises_instead_of_hanging():
    def bad(game):
        legal = game.legal_moves()
        illegal = [m for m in range(4) if m not in legal]
        return illegal[0] if illegal else legal[0]   # misbehaves as soon as it can
    try:
        play_batch(bad, [0])
    except ValueError:
        return
    raise AssertionError("illegal agent move must raise ValueError")


def test_play_batch_works_with_bound_method():
    class A:
        def choose_move(self, game):
            return game.legal_moves()[-1]
    res = play_batch(A().choose_move, [0, 1])
    assert len(res) == 2 and all(r["score"] > 0 for r in res)
