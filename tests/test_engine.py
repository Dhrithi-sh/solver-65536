import pytest

from engine.game import DOWN, LEFT, MOVES, RIGHT, UP, Game


def slide(board, move):
    return Game.slide(board, move)


def row_board(row):
    return [list(row), [0] * 4, [0] * 4, [0] * 4]


# ---------------------------------------------------------------- merges
def test_simple_merge_left():
    new, gained = slide(row_board([2, 2, 0, 0]), LEFT)
    assert new[0] == [4, 0, 0, 0]
    assert gained == 4


def test_slide_without_merge():
    new, gained = slide(row_board([0, 2, 0, 4]), LEFT)
    assert new[0] == [2, 4, 0, 0]
    assert gained == 0


def test_no_double_merge_2222():
    new, gained = slide(row_board([2, 2, 2, 2]), LEFT)
    assert new[0] == [4, 4, 0, 0]
    assert gained == 8


def test_no_chain_merge_4_2_2():
    # 2+2 -> 4 must not then merge with the existing 4 in the same move.
    new, _ = slide(row_board([4, 2, 2, 0]), LEFT)
    assert new[0] == [4, 4, 0, 0]


def test_merge_priority_toward_move_direction():
    new, _ = slide(row_board([2, 2, 2, 0]), LEFT)
    assert new[0] == [4, 2, 0, 0]
    new, _ = slide(row_board([2, 2, 2, 0]), RIGHT)
    assert new[0] == [0, 0, 2, 4]


def test_65536_merge():
    # 4-bit bitboards cap at 32,768; the engine must still produce 65,536.
    new, gained = slide(row_board([32768, 32768, 0, 0]), LEFT)
    assert new[0] == [65536, 0, 0, 0]
    assert gained == 65536


# ------------------------------------------------------------ directions
BOARD = [
    [2, 0, 0, 2],
    [0, 4, 0, 0],
    [0, 4, 8, 0],
    [2, 0, 8, 0],
]


def test_left():
    new, gained = slide(BOARD, LEFT)
    assert new == [[4, 0, 0, 0], [4, 0, 0, 0], [4, 8, 0, 0], [2, 8, 0, 0]]
    assert gained == 4


def test_right():
    new, gained = slide(BOARD, RIGHT)
    assert new == [[0, 0, 0, 4], [0, 0, 0, 4], [0, 0, 4, 8], [0, 0, 2, 8]]
    assert gained == 4


def test_up():
    new, gained = slide(BOARD, UP)
    assert new == [[4, 8, 16, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    assert gained == 28


def test_down():
    new, gained = slide(BOARD, DOWN)
    assert new == [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [4, 8, 16, 2]]
    assert gained == 28


def test_slide_is_pure():
    before = [row[:] for row in BOARD]
    slide(BOARD, LEFT)
    assert BOARD == before


def test_invalid_move_raises():
    with pytest.raises(ValueError):
        slide(BOARD, 4)
    with pytest.raises(ValueError):
        Game(0).step(7)


# --------------------------------------------------------- illegal moves
def test_illegal_moves_detected():
    g = Game(0, board=[[2, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert LEFT not in g.legal_moves()
    assert UP not in g.legal_moves()
    assert set(g.legal_moves()) == {RIGHT, DOWN}


def test_illegal_step_is_noop():
    start = [[2, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    g = Game(0, board=start)
    board, reward, done = g.step(LEFT)
    assert board == start and reward == 0 and not done
    assert g.moves == [] and g.score == 0


# -------------------------------------------------------- seeded spawns
def test_new_game_has_two_tiles():
    for seed in range(20):
        tiles = [v for row in Game(seed).board for v in row if v]
        assert len(tiles) == 2
        assert all(v in (2, 4) for v in tiles)


def _play_fixed(seed, n=200):
    g = Game(seed)
    for i in range(n):
        if g.done:
            break
        g.step(g.legal_moves()[i % len(g.legal_moves())])
    return g.board, g.score, g.moves


def test_seeded_games_repeat():
    assert _play_fixed(123) == _play_fixed(123)
    g = Game(5)
    first = _play_fixed(5)
    g.reset(5)
    assert g.board == Game(5).board
    assert first != _play_fixed(6)


def test_spawn_distribution_roughly_90_10():
    twos = fours = 0
    for seed in range(2000):
        for v in (v for row in Game(seed).board for v in row if v):
            twos += v == 2
            fours += v == 4
    frac_four = fours / (twos + fours)
    assert 0.07 < frac_four < 0.13


def test_step_spawns_one_tile_and_scores():
    g = Game(0, board=[[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    board, reward, done = g.step(LEFT)
    assert reward == 4 and g.score == 4 and not done
    assert sum(1 for row in board for v in row if v) == 2
    assert g.moves == [LEFT]


# ------------------------------------------------------------- game end
def test_game_over_on_full_board_without_merges():
    full = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
    g = Game(0, board=full)
    assert g.legal_moves() == []
    assert g.done
    board, reward, done = g.step(LEFT)
    assert done and reward == 0 and board == full


def test_full_board_with_merge_is_not_over():
    g = Game(0, board=[[2, 2, 4, 8], [4, 8, 16, 32], [8, 16, 32, 64], [16, 32, 64, 128]])
    assert not g.done
    assert LEFT in g.legal_moves()


def test_random_games_terminate():
    import random

    for seed in range(10):
        g = Game(seed)
        rng = random.Random(seed)
        steps = 0
        while not g.done:
            g.step(rng.choice(g.legal_moves()))
            steps += 1
            assert steps < 100_000
        assert g.legal_moves() == []
        assert g.max_tile() >= 4


def test_max_tile():
    g = Game(0, board=[[2, 0, 0, 0], [0, 1024, 0, 0], [0, 0, 0, 0], [0, 0, 0, 4]])
    assert g.max_tile() == 1024


def test_moves_constant():
    assert MOVES == (0, 1, 2, 3) and (UP, RIGHT, DOWN, LEFT) == MOVES
