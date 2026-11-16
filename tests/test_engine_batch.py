"""P1 Batch 3c: fast random self-play (engine.batch) and play_batch determinism.

The headline test is ``test_cpp_and_python_agree_on_10000_seeded_games`` (brief,
P1 task 7): the C++ batch and the PURE-PYTHON engine, in a separate process with
ENGINE_PURE_PYTHON=1 so no C++ can leak in, must produce identical scores, max
tiles, move lists and final boards. Set ENGINE_PARITY_GAMES to run fewer games
while developing (default 10000).
"""

import json
import os
import random
import subprocess
import sys
import threading
import unittest

from engine import batch as B
from engine import game as G
from engine.game import Game, play_batch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CPP = G._cengine
PARITY_GAMES = int(os.environ.get("ENGINE_PARITY_GAMES", "10000"))


def need_cpp():
    if CPP is None or not hasattr(CPP, "play_random"):
        raise unittest.SkipTest("compiled extension engine._cengine not available")


class python_only:
    def __enter__(self):
        self.saved = G._cengine
        G._cengine = None

    def __exit__(self, *exc):
        G._cengine = self.saved


def empty():
    return [[0] * 4 for _ in range(4)]


def flat(results):
    return [[r["seed"], r["score"], r["max_tile"], r["n_moves"], r["moves"], r["board"]] for r in results]


# ------------------------------------------------------------------ the 10,000-game test
_REFERENCE = """
import json, os, sys
from engine import game as G
from engine.batch import RandomLegalPolicy
assert G.backend() == "python", "reference run must be pure Python"
n, procs = int(sys.argv[1]), int(sys.argv[2])
rows = G.play_batch(RandomLegalPolicy(), range(n), processes=procs, deterministic=True)
json.dump([[r["seed"], r["score"], r["max_tile"], len(r["moves"]), r["moves"], r["board"]] for r in rows], sys.stdout)
"""


def test_cpp_and_python_agree_on_10000_seeded_games():
    need_cpp()
    assert CPP is not None and G.backend() == "cpp"
    procs = max(1, min(os.cpu_count() or 1, 8))
    env = dict(os.environ, ENGINE_PURE_PYTHON="1", PYTHONPATH=ROOT)
    out = subprocess.run([sys.executable, "-c", _REFERENCE, str(PARITY_GAMES), str(procs)], env=env, cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    reference = json.loads(out)
    fast = flat(B.play_random_batch(range(PARITY_GAMES)))
    assert len(fast) == len(reference) == PARITY_GAMES
    assert fast == reference  # every score, max tile, full move list and final board


# ----------------------------------------------------------- smaller, faster parity checks
def test_batch_matches_pure_python_engine_in_process():
    need_cpp()
    seeds = list(range(150)) + [-1, -2, 2**31, 2**32 + 5, 2**63 + 9]
    fast = B.play_random_batch(seeds)
    with python_only():
        slow = B.play_random_batch(seeds)
    assert flat(fast) == flat(slow)


def test_cpp_games_replay_on_the_real_engine():
    need_cpp()
    for r in B.play_random_batch(range(40)):
        g = Game(r["seed"])
        for m in r["moves"]:
            g.step(m)
        assert g.done and g.board == r["board"] and g.score == r["score"] and g.max_tile() == r["max_tile"]
        assert len(r["moves"]) == r["n_moves"]


def test_results_do_not_depend_on_thread_count():
    need_cpp()
    base = flat(B.play_random_batch(range(300), threads=1))
    for t in (2, 3, 7, 0):
        assert flat(B.play_random_batch(range(300), threads=t)) == base


def test_python_policy_via_play_batch_equals_batch_python_path():
    seeds = list(range(15))
    with python_only():
        a = play_batch(B.RandomLegalPolicy(), seeds)
        b = B.play_random_batch(seeds)
    assert [(r["score"], r["moves"], r["board"]) for r in a] == [(r["score"], r["moves"], r["board"]) for r in b]


def test_works_without_extension_and_is_repeatable():
    with python_only():
        a, b = B.play_random_batch(range(8)), B.play_random_batch(range(8))
    assert a == b and all(r["max_tile"] >= 8 for r in a)


# --------------------------------------------------------------------- start boards
def late_game_board():
    b = empty()
    b[0][0] = b[0][1] = 32768   # one merge away from 65,536
    b[1][0], b[1][1], b[1][2] = 16384, 8192, 4096
    return b


def test_from_board_start_matches_python_and_can_make_65536():
    need_cpp()
    seeds = range(60)
    fast = B.play_random_batch(seeds, board=late_game_board())
    with python_only():
        slow = B.play_random_batch(seeds, board=late_game_board())
    assert flat(fast) == flat(slow)
    assert any(r["max_tile"] >= 65536 for r in fast)  # the 65,536 tile is reachable end to end


def test_start_board_is_not_modified():
    b = late_game_board()
    before = [row[:] for row in b]
    B.play_random_batch(range(5), board=b)
    assert b == before


def test_huge_start_tiles_leave_cpp_range_and_still_match_python():
    """Sixteen 2**16 tiles can chain up to 2**19, beyond the C++ tables; those games are
    handed back to Python and must still give the Python answer."""
    need_cpp()
    b = [[65536] * 4 for _ in range(4)]
    raw = CPP.play_random(list(range(6)), b, 1, True)
    assert any(item[5] for item in raw), "expected at least one game to leave the supported range"
    fast = B.play_random_batch(range(6), board=b)
    with python_only():
        slow = B.play_random_batch(range(6), board=b)
    assert flat(fast) == flat(slow)


def test_unsupported_inputs_fall_back_to_python():
    need_cpp()
    odd = empty()
    odd[0][0] = 3  # not a power of two: the extension declines, Python plays it
    assert CPP.play_random([1], odd, 1, True) is None
    assert CPP.play_random([1.5], None, 1, True) is None
    assert CPP.play_random([2**64], None, 1, True) is None
    assert B.play_random_batch([2**70])[0]["n_moves"] > 0  # big seed handled by Python
    with python_only():
        want = flat(B.play_random_batch([5], board=odd))
    assert flat(B.play_random_batch([5], board=odd)) == want


def test_bad_board_shape_raises():
    try:
        B.play_random_batch([1], board=[[0] * 4] * 3)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_record_moves_false_and_empty_input():
    r = B.play_random_batch(range(5), record_moves=False)
    full = B.play_random_batch(range(5))
    assert all(x["moves"] is None for x in r)
    assert [x["n_moves"] for x in r] == [x["n_moves"] for x in full]
    assert [x["score"] for x in r] == [x["score"] for x in full]
    assert B.play_random_batch([]) == []


def test_concurrent_python_threads_get_correct_results():
    need_cpp()
    want = flat(B.play_random_batch(range(200), threads=1))
    results = {}

    def run(i):
        results[i] = flat(B.play_random_batch(range(200), threads=2))

    ts = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert all(results[i] == want for i in range(4))


# ------------------------------------------------------------ play_batch (arbitrary agents)
class StatefulRandomAgent:
    """Keeps ONE RNG across games: results depend on game order unless isolated."""

    def __init__(self):
        self.rng = random.Random(1)
        self.resets = 0

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.resets += 1

    def choose_move(self, game):
        return self.rng.choice(game.legal_moves())


def stateless_first_move(game):
    return game.legal_moves()[0]


def summary(rows):
    return [(r["seed"], r["score"], r["moves"], r["board"]) for r in rows]


def test_play_batch_default_behaviour_unchanged():
    seeds = list(range(6))
    a = play_batch(stateless_first_move, seeds)
    assert summary(a) == summary(play_batch(stateless_first_move, seeds, processes=2))
    assert [r["seed"] for r in a] == seeds


def test_play_batch_deterministic_mode_is_independent_of_process_count():
    seeds = list(range(12))
    runs = [summary(play_batch(StatefulRandomAgent().choose_move, seeds, processes=p, deterministic=True))
            for p in (1, 2, 3)]
    assert runs[0] == runs[1] == runs[2]
    # and equals what eval/run.py-style play does: a fresh reset(seed) per game
    agent = StatefulRandomAgent()
    want = []
    for s in seeds:
        agent.reset(s)
        g = Game(s)
        while not g.done:
            g.step(agent.choose_move(g))
        want.append((s, g.score, g.moves, g.board))
    assert runs[0] == want


def test_play_batch_deterministic_mode_does_not_mutate_callers_agent():
    agent = StatefulRandomAgent()
    play_batch(agent.choose_move, range(3), deterministic=True)
    assert agent.resets == 0


def test_play_batch_stateful_agents_differ_without_deterministic_flag_documented():
    """Why the flag exists: a shared stateful agent depends on game order."""
    a = summary(play_batch(StatefulRandomAgent().choose_move, [3, 4]))
    b = summary(play_batch(StatefulRandomAgent().choose_move, [4]))
    assert a[1] != b[0]  # seed 4 played second inherits state from seed 3


def test_play_batch_works_with_spawn_start_method():
    """Windows/macOS default: agent_fn must be importable and picklable."""
    seeds = list(range(4))
    a = play_batch(B.RandomLegalPolicy(), seeds, processes=2, deterministic=True, start_method="spawn")
    b = play_batch(B.RandomLegalPolicy(), seeds, processes=1, deterministic=True)
    assert summary(a) == summary(b)


def test_play_batch_still_rejects_illegal_moves_in_parallel():
    try:
        play_batch(lambda g: 9, [1], processes=1)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
