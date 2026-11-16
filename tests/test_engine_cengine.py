"""P1 Batch 3b tests: the optional C++ extension ``engine._cengine`` and how
engine/game.py uses it.

* Tests that need the compiled extension skip themselves (SkipTest) when it is
  not built, so this file is safe in the pure-Python CI job.
* Tests of the dispatch/fallback logic run everywhere (they use fake extension
  objects, a subprocess, or switch the extension off).
* Python results are always produced by switching the extension off
  (`python_only()`), never by re-implementing the engine here.
"""

import os
import random
import subprocess
import sys
import unittest

from engine import game as G
from engine.game import DOWN, LEFT, RIGHT, UP, Game

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CPP = G._cengine  # the real extension, or None


def need_cpp():
    if CPP is None:
        raise unittest.SkipTest("compiled extension engine._cengine not available")


class python_only:
    """Context manager: run the pure-Python engine inside the block."""

    def __enter__(self):
        self.saved = G._cengine
        G._cengine = None

    def __exit__(self, *exc):
        G._cengine = self.saved


class using:
    """Context manager: temporarily replace the extension object."""

    def __init__(self, fake):
        self.fake = fake

    def __enter__(self):
        self.saved = G._cengine
        G._cengine = self.fake

    def __exit__(self, *exc):
        G._cengine = self.saved


def empty():
    return [[0] * 4 for _ in range(4)]


def random_board(rng, max_exp=18):
    tiles = [0, 0, 0, 2, 4, 8, 64, 2048, 32768, 65536, 131072, 262144]
    tiles = [t for t in tiles if t == 0 or t.bit_length() - 1 <= max_exp]
    return [[rng.choice(tiles) for _ in range(4)] for _ in range(4)]


def py_slide(board, move):
    with python_only():
        return Game.slide(board, move)


def test_extension_is_built_when_required():
    """CI sets ENGINE_REQUIRE_CPP=1 after building, so a silent build failure
    (the extension is optional) cannot hide behind the pure-Python fallback."""
    if os.environ.get("ENGINE_REQUIRE_CPP", "0") in ("", "0"):
        raise unittest.SkipTest("ENGINE_REQUIRE_CPP not set")
    assert CPP is not None, "engine._cengine was required but is not importable"
    assert G.backend() == "cpp"


# ===================================================== always-run: fallback
def test_backend_reports_active_engine():
    assert G.backend() == ("python" if G._cengine is None else "cpp")
    with python_only():
        assert G.backend() == "python"


def test_fallback_engine_still_plays_normally():
    with python_only():
        g = Game(3)
        picker = random.Random(1)
        while not g.done:
            g.step(picker.choice(g.legal_moves()))
        assert g.score > 0 and g.moves
        assert Game.slide([[2, 2, 0, 0]] + [[0] * 4] * 3, LEFT)[0][0] == [4, 0, 0, 0]


def _run_python(code, env_extra=None):
    env = dict(os.environ)
    env.pop("ENGINE_PURE_PYTHON", None)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)


def test_env_var_forces_pure_python():
    code = "from engine import game as G; assert G._cengine is None and G.backend() == 'python'; print('ok')"
    r = _run_python(code, {"ENGINE_PURE_PYTHON": "1"})
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


def test_env_var_zero_or_empty_does_not_force_python():
    code = "from engine import game as G; print(G.backend())"
    baseline = _run_python(code)  # no variable: "cpp" if the extension is built, else "python"
    assert baseline.returncode == 0, baseline.stderr
    for value in ("", "0"):
        r = _run_python(code, {"ENGINE_PURE_PYTHON": value})
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == baseline.stdout.strip()


def test_missing_extension_import_falls_back():
    # sys.modules[name] = None makes `import name` raise ImportError, i.e. "not built".
    code = ("import sys; sys.modules['engine._cengine'] = None\n"
            "from engine import game as G\n"
            "assert G._cengine is None and G.backend() == 'python'\n"
            "g = G.Game(1); g.step(g.legal_moves()[0]); print('ok')")
    r = _run_python(code)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


# ============================================== always-run: dispatch logic
class FakeAlwaysNone:
    """An extension that declines everything: Python must answer."""
    calls = 0

    def slide(self, board, move):
        FakeAlwaysNone.calls += 1
        return None

    def legal_mask(self, board):
        FakeAlwaysNone.calls += 1
        return None


class FakeAnswers:
    """An extension that answers with sentinel values: proves the fast path is used."""

    def slide(self, board, move):
        return ([[7] * 4] * 4, 123)

    def legal_mask(self, board):
        return 0b0101  # UP and DOWN


def test_none_from_extension_means_python_answers():
    b = [[2, 2, 4, 4], [0, 2, 0, 2], [8, 0, 8, 0], [0, 0, 0, 2]]
    expected = {m: py_slide(b, m) for m in range(4)}
    FakeAlwaysNone.calls = 0
    with using(FakeAlwaysNone()):
        for m in range(4):
            assert Game.slide(b, m) == expected[m]
        assert Game.from_board(b, 0).legal_moves() == [m for m in range(4) if expected[m][0] != b]
    assert FakeAlwaysNone.calls >= 5


def test_extension_answer_is_used_when_given():
    with using(FakeAnswers()):
        assert Game.slide(empty(), LEFT) == ([[7] * 4] * 4, 123)
        g = Game.from_board([[2, 4, 8, 16]] + [[0] * 4] * 3, 0)
        assert g.legal_moves() == [UP, DOWN]


def test_non_int_moves_never_reach_the_extension():
    class Spy:
        seen = []

        def slide(self, board, move):
            Spy.seen.append(move)
            return None

    with using(Spy()):
        for ok in (True, 1.0):  # Python path accepts these as before
            Game.slide(empty(), ok)
        for bad in (4, -1, None, "up"):
            try:
                Game.slide(empty(), bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{bad!r} must raise ValueError")
    assert Spy.seen == [4, -1]  # exact ints are offered; extension says None; Python raises


# ================================================ needs the compiled module
def test_cpp_constants_and_none_contract():
    need_cpp()
    assert CPP.MAX_SUPPORTED_EXP == 18
    good = empty()
    bad_boards = {
        "exponent 19": [[524288] + [0] * 3] + [[0] * 4] * 3,
        "not a power of two": [[3] + [0] * 3] + [[0] * 4] * 3,
        "tile 1": [[1] + [0] * 3] + [[0] * 4] * 3,
        "negative": [[-2] + [0] * 3] + [[0] * 4] * 3,
        "float": [[2.0] + [0] * 3] + [[0] * 4] * 3,
        "float zero": [[0.0] + [0] * 3] + [[0] * 4] * 3,
        "bool": [[True] + [0] * 3] + [[0] * 4] * 3,
        "bool False": [[False] + [0] * 3] + [[0] * 4] * 3,
        "string": [["2"] + [0] * 3] + [[0] * 4] * 3,
        "None cell": [[None] + [0] * 3] + [[0] * 4] * 3,
        "huge int": [[2 ** 70] + [0] * 3] + [[0] * 4] * 3,
        "3 rows": [[0] * 4] * 3,
        "3 columns": [[0] * 3] * 4,
        "ragged": [[0] * 4, [0] * 4, [0] * 4, [0] * 3],
        "empty": [],
        "not a sequence": 5,
        "string board": "0000000000000000",
        "list subclass": type("L", (list,), {})([[0] * 4] * 4),
    }
    try:
        import numpy as np
        bad_boards["numpy int"] = [[np.int64(2)] + [0] * 3] + [[0] * 4] * 3
    except ImportError:
        pass
    for name, b in bad_boards.items():
        for m in range(4):
            assert CPP.slide(b, m) is None, name
        assert CPP.legal_mask(b) is None, name
    for bad_move in (4, -1, 99, True, 1.0, None, "up", 2 ** 70):
        assert CPP.slide(good, bad_move) is None, repr(bad_move)
    assert CPP.slide(good, LEFT) == ([[0] * 4 for _ in range(4)], 0)
    assert CPP.legal_mask(good) == 0


def test_unsupported_boards_give_python_results_through_game():
    need_cpp()
    cases = [
        [[524288, 524288, 0, 0]] + [[0] * 4] * 3,   # exponent 19
        [[3, 3, 0, 0]] + [[0] * 4] * 3,             # not a power of two
        [[2.0, 2.0, 0, 0]] + [[0] * 4] * 3,         # floats
        [[True, True, 0, 0]] + [[0] * 4] * 3,       # bools
    ]
    for b in cases:
        for m in range(4):
            assert Game.slide(b, m) == py_slide(b, m), (b[0], m)
    # Odd shapes: whatever the Python engine does (result or exception type), the
    # integrated engine must do the same, because C++ declines and Python runs.
    def outcome(board, m):
        try:
            return ("ok", Game.slide(board, m))
        except Exception as exc:  # noqa: BLE001 - comparing behaviour, not handling it
            return ("error", type(exc))

    for odd in ([[0] * 3] * 4, [[0] * 4] * 3, [[0] * 4, [0] * 4, [0] * 4, [0] * 3]):
        for m in range(4):
            with python_only():
                want = outcome(odd, m)
            assert outcome(odd, m) == want, (odd, m)


def test_slide_matches_python_all_directions_and_scores():
    need_cpp()
    rng = random.Random(77)
    for _ in range(4000):
        b = random_board(rng)
        for m in range(4):
            assert CPP.slide(b, m) == py_slide(b, m), (b, m)


def test_every_row_up_to_exponent_18_matches_python():
    need_cpp()
    import itertools
    for exps in itertools.product(range(19), repeat=4):
        row = [(1 << e) if e else 0 for e in exps]
        b = [row, [0] * 4, [0] * 4, [0] * 4]
        for m in (LEFT, RIGHT):
            assert CPP.slide(b, m) == py_slide(b, m), (row, m)


def test_tuple_boards_are_accepted_and_match():
    need_cpp()
    b = ((2, 2, 4, 4), (0, 2, 0, 2), (8, 0, 8, 0), (0, 0, 0, 2))
    mixed = [(2, 2, 4, 4), [0, 2, 0, 2], (8, 0, 8, 0), [0, 0, 0, 2]]
    for m in range(4):
        assert CPP.slide(b, m) is not None and CPP.slide(mixed, m) is not None
        assert CPP.slide(b, m) == py_slide(b, m) == CPP.slide(mixed, m)


def test_65536_merge_and_exponent_18_edge_cases():
    need_cpp()
    for tile, result in ((32768, 65536), (65536, 131072), (131072, 262144)):
        for m in range(4):
            pairs = {UP: ((0, 0), (1, 0)), RIGHT: ((0, 3), (0, 2)), DOWN: ((3, 0), (2, 0)), LEFT: ((0, 0), (0, 1))}[m]
            b = empty()
            for r, c in pairs:
                b[r][c] = tile
            out = Game.slide(b, m)
            assert out == CPP.slide(b, m) == py_slide(b, m)
            assert max(max(r) for r in out[0]) == result and out[1] == result
    # a 262,144 tile (exponent 18) is a supported INPUT and keeps moving
    for pos in range(16):
        b = empty()
        b[pos // 4][pos % 4] = 262144
        for m in range(4):
            assert CPP.slide(b, m) is not None and CPP.slide(b, m) == py_slide(b, m)
    chain = [[131072, 131072, 262144, 0]] + [[0] * 4] * 3
    first = CPP.slide(chain, LEFT)
    assert first[0][0][:2] == [262144, 262144] and first[1] == 262144
    second = CPP.slide(first[0], LEFT)               # input has exponent 18
    assert second is not None and second == py_slide(first[0], LEFT)
    assert second[0][0][0] == 524288 and second[1] == 524288
    # the board after that holds exponent 19: C++ declines, Python continues
    assert CPP.slide(second[0], RIGHT) is None
    assert Game.slide(second[0], RIGHT) == py_slide(second[0], RIGHT)
    g = Game.from_board(second[0], 5)
    assert g.max_tile() == 524288 and g.legal_moves() == [m for m in range(4) if py_slide(g.board, m)[0] != g.board]


def test_legal_mask_matches_python():
    need_cpp()
    rng = random.Random(5)
    for _ in range(3000):
        b = random_board(rng)
        mask = CPP.legal_mask(b)
        with python_only():
            expected = [m for m in range(4) if Game.is_legal(b, m)]
            py_moves = Game.from_board(b, 0).legal_moves()
        assert [m for m in range(4) if (mask >> m) & 1] == expected == py_moves
        assert Game.from_board(b, 0).legal_moves() == expected


def test_results_are_fresh_lists_and_input_is_untouched():
    need_cpp()
    b = [[2, 2, 4, 4], [0, 2, 0, 2], [8, 0, 8, 0], [0, 0, 0, 2]]
    snap = [r[:] for r in b]
    out, _ = CPP.slide(b, LEFT)
    assert b == snap
    out[0][0] = 999
    out[1] = None
    assert b == snap and CPP.slide(b, LEFT)[0][0][0] == 4
    assert all(type(v) is int for row in CPP.slide(b, UP)[0] for v in row)
    assert type(CPP.slide(b, UP)[1]) is int


def _play(seed, picker_seed):
    g = Game(seed)
    picker = random.Random(picker_seed)
    while not g.done:
        g.step(picker.choice(g.legal_moves()))
    return g.board, g.score, list(g.moves), g.rng.getstate()


def test_seeded_games_identical_with_and_without_extension():
    need_cpp()
    for seed in range(25):
        with python_only():
            expected = _play(seed, seed + 100)
        assert _play(seed, seed + 100) == expected  # board, score, moves AND final RNG state


def test_replay_on_one_backend_reproduces_game_from_the_other():
    need_cpp()
    with python_only():
        board, score, moves, _ = _play(11, 3)
    g = Game(11)
    for m in moves:
        g.step(m)
    assert (g.board, g.score, g.moves) == (board, score, moves)


def test_from_board_games_identical_across_backends():
    need_cpp()
    start = empty()
    start[0][0] = start[0][1] = 131072
    start[3][3] = start[3][2] = 32768

    def run():
        g = Game.from_board(start, 9)
        picker = random.Random(2)
        for _ in range(40):
            if g.done:
                break
            g.step(picker.choice(g.legal_moves()))
        return g.board, g.score, g.moves, g.rng.getstate()

    with python_only():
        expected = run()
    assert run() == expected


def test_batch2_helpers_identical_across_backends():
    need_cpp()
    rng = random.Random(8)
    for _ in range(200):
        b = random_board(rng, max_exp=17)
        with python_only():
            want = ([G.afterstate(b, m) for m in range(4)], G.spawn_outcomes(b))
        assert ([G.afterstate(b, m) for m in range(4)], G.spawn_outcomes(b)) == want
