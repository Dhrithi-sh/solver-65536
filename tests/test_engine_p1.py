"""P1 engine tests (Batch 1): movement, merging, scoring, illegal moves, 65,536,
the 5-bit row table vs. the original merge, and RNG/replay compatibility.

This file is NEW and sits next to the team's existing tests/test_engine.py,
which it does not replace. Plain asserts only (runs under pytest).
"""

import itertools
import random

from engine import game as G
from engine.game import DOWN, LEFT, RIGHT, UP, Game


def board_from_row(row):
    return [list(row), [0] * 4, [0] * 4, [0] * 4]


# ------------------------------------------------------------- movement
def test_all_four_directions_move_and_merge():
    b = [[2, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [4, 0, 0, 4]]
    assert Game.slide(b, LEFT)[0] == [[4, 0, 0, 0], [0] * 4, [0] * 4, [8, 0, 0, 0]]
    assert Game.slide(b, RIGHT)[0] == [[0, 0, 0, 4], [0] * 4, [0] * 4, [0, 0, 0, 8]]
    t = [[2, 0, 0, 4], [0] * 4, [0] * 4, [2, 0, 0, 4]]
    assert Game.slide(t, UP)[0] == [[4, 0, 0, 8], [0] * 4, [0] * 4, [0] * 4]
    assert Game.slide(t, DOWN)[0] == [[0] * 4, [0] * 4, [0] * 4, [4, 0, 0, 8]]


def test_slide_is_pure_and_does_not_mutate_input():
    b = [[2, 2, 4, 4], [0, 2, 0, 2], [8, 0, 8, 0], [0, 0, 0, 2]]
    snapshot = [r[:] for r in b]
    for m in range(4):
        Game.slide(b, m)
    assert b == snapshot


# --------------------------------------------------------------- merging
def test_no_double_merge():
    new, gained = Game.slide(board_from_row([2, 2, 2, 2]), LEFT)
    assert new[0] == [4, 4, 0, 0] and gained == 8
    new, gained = Game.slide(board_from_row([4, 2, 2, 0]), LEFT)
    assert new[0] == [4, 4, 0, 0] and gained == 4      # 4 does not merge with new 4
    new, gained = Game.slide(board_from_row([2, 2, 2, 0]), LEFT)
    assert new[0] == [4, 2, 0, 0] and gained == 4
    new, gained = Game.slide(board_from_row([0, 2, 2, 2]), RIGHT)
    assert new[0] == [0, 0, 2, 4] and gained == 4      # leading pair merges toward the wall


def test_merge_with_gaps():
    new, gained = Game.slide(board_from_row([2, 0, 2, 4]), LEFT)
    assert new[0] == [4, 4, 0, 0] and gained == 4


# ---------------------------------------------------------------- scoring
def test_score_is_sum_of_created_tiles():
    b = [[2, 2, 4, 4], [8, 8, 8, 8], [0] * 4, [0] * 4]
    new, gained = Game.slide(b, LEFT)
    assert gained == (4 + 8) + (16 + 16)   # [2,2,4,4] -> 4+8, [8,8,8,8] -> 16+16
    g = Game(0, board=b)
    g.score = 0
    _, reward, _ = g.step(LEFT)
    assert reward == gained and g.score == gained


# ----------------------------------------------------------- illegal moves
def test_illegal_move_is_noop():
    b = [[2, 4, 8, 16], [0] * 4, [0] * 4, [0] * 4]
    g = Game(5, board=b)
    state = g.rng.getstate()
    board, reward, done = g.step(UP)           # row already against the top wall
    assert board == b and reward == 0 and done is False
    assert g.moves == [] and g.score == 0
    assert g.rng.getstate() == state           # no spawn, no RNG consumed


def test_invalid_move_value_raises():
    g = Game(1)
    for bad in (-1, 4, None, "up"):
        try:
            g.step(bad)
        except ValueError:
            continue
        raise AssertionError(f"step({bad!r}) should raise ValueError")


def test_done_when_no_moves():
    full = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
    g = Game(0, board=full)
    assert g.done and g.legal_moves() == []
    board, reward, done = g.step(LEFT)
    assert board == full and reward == 0 and done is True


# ---------------------------------------------------------------- 65,536
def test_32768_plus_32768_makes_65536():
    new, gained = Game.slide(board_from_row([32768, 32768, 0, 0]), LEFT)
    assert new[0] == [65536, 0, 0, 0] and gained == 65536
    g = Game(3, board=board_from_row([32768, 0, 0, 32768]))
    _, reward, _ = g.step(LEFT)
    assert reward == 65536 and g.score == 65536
    assert g.max_tile() == 65536 and g.board[0][0] == 65536


def test_65536_in_every_direction_and_beyond():
    for move, cells in ((LEFT, ((0, 0), (0, 1))), (RIGHT, ((0, 3), (0, 2))),
                        (UP, ((0, 0), (1, 0))), (DOWN, ((3, 0), (2, 0)))):
        b = [[0] * 4 for _ in range(4)]
        for r, c in cells:
            b[r][c] = 32768
        new, gained = Game.slide(b, move)
        assert max(max(r) for r in new) == 65536 and gained == 65536
    new, gained = Game.slide(board_from_row([65536, 65536, 0, 0]), LEFT)
    assert new[0][0] == 131072 and gained == 131072   # exponent 17 still fits in 5 bits


def test_row_table_roundtrip_keeps_exponent_16():
    key = G.pack_row([16, 16, 0, 1])
    assert G.unpack_row(key) == [16, 16, 0, 1] and key < 1 << 20
    out, gained = G.row_table_entry(key)
    assert G.unpack_row(out) == [17, 1, 0, 0] and gained == 1 << 17


# --------------------------------- table vs original merge (differential)
def test_row_table_matches_reference_for_every_row_up_to_65536():
    """All 17^4 rows with exponents 0..16 (tiles up to 65,536)."""
    for exps in itertools.product(range(17), repeat=4):
        line = [(1 << e) if e else 0 for e in exps]
        ref_line, ref_gain = G._merge_line(line)
        out_key, gained = G.row_table_entry(G.pack_row(exps))
        got = [(1 << e) if e else 0 for e in G.unpack_row(out_key)]
        assert got == ref_line and gained == ref_gain, (line, got, ref_line)


def _ref_slide(board, move):
    lines = G._get_lines(board, move)
    merged = [G._merge_line(l) for l in lines]
    return G._set_lines([m[0] for m in merged], move), sum(m[1] for m in merged)


def test_slide_matches_reference_on_random_boards():
    rng = random.Random(1234)
    tiles = [0, 0, 0, 2, 4, 8, 16, 64, 1024, 32768, 65536]
    for _ in range(3000):
        b = [[rng.choice(tiles) for _ in range(4)] for _ in range(4)]
        for m in range(4):
            assert Game.slide(b, m) == _ref_slide(b, m)


def test_non_power_of_two_boards_use_fallback_unchanged():
    b = [[3, 3, 0, 0], [0] * 4, [0] * 4, [0] * 4]   # not valid 2048, must not crash
    assert Game.slide(b, LEFT) == _ref_slide(b, LEFT)


# ------------------------------------------- determinism / RNG compatibility
def _oracle_game(seed, move_source):
    """Independent re-implementation of the ORIGINAL engine rules (original
    per-line merge, original RNG call order: randrange(cell), then random())."""
    rng = random.Random(seed)
    board = [[0] * 4 for _ in range(4)]

    def spawn():
        empt = [(r, c) for r in range(4) for c in range(4) if board[r][c] == 0]
        if empt:
            r, c = empt[rng.randrange(len(empt))]
            board[r][c] = 2 if rng.random() < 0.9 else 4

    spawn(); spawn()
    score, played = 0, []
    while True:
        legal = [m for m in range(4) if _ref_slide(board, m)[0] != board]
        if not legal:
            return board, score, played
        m = move_source(legal)
        nb, gained = _ref_slide(board, m)
        board[:] = nb
        score += gained
        played.append(m)
        spawn()


def test_engine_matches_original_rules_move_for_move():
    for seed in range(20):
        picker = random.Random(seed + 999)
        choices = []

        def source(legal):
            m = picker.choice(legal)
            choices.append(m)
            return m

        o_board, o_score, o_moves = _oracle_game(seed, source)
        g = Game(seed)
        for m in o_moves:
            g.step(m)
        assert g.done
        assert (g.board, g.score, g.moves) == (o_board, o_score, o_moves)


def test_seeded_games_repeat_and_reset_restarts():
    def play(seed):
        g = Game(seed)
        picker = random.Random(7)
        while not g.done:
            g.step(picker.choice(g.legal_moves()))
        return g.board, g.score, g.moves

    assert play(42) == play(42)
    assert play(42) != play(43)
    g = Game(9)
    first = [r[:] for r in g.board]
    g.step(g.legal_moves()[0])
    g.reset(9)
    assert g.board == first and g.score == 0 and g.moves == []


def test_replay_of_move_list_reproduces_game():
    g1 = Game(77)
    picker = random.Random(3)
    while not g1.done:
        g1.step(picker.choice(g1.legal_moves()))
    g2 = Game(77)
    for m in g1.moves:
        g2.step(m)
    assert (g2.board, g2.score, g2.moves) == (g1.board, g1.score, g1.moves)


def test_spawn_distribution_is_roughly_90_10():
    twos = fours = 0
    for seed in range(2000):
        for row in Game(seed).board:
            for v in row:
                twos += v == 2
                fours += v == 4
    assert abs(fours / (twos + fours) - 0.1) < 0.02


def test_reset_with_board_spawns_nothing_and_copies_input():
    b = [[0] * 4 for _ in range(4)]
    b[0][0] = b[0][1] = 2
    g = Game(0, board=b)
    assert sum(v != 0 for row in g.board for v in row) == 2
    g.step(LEFT)
    assert b[0] == [2, 2, 0, 0]          # caller's board untouched
