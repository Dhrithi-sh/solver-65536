"""Tests for P7's hybrid agent (agents/hybrid).

The C++ search is checked against things we did not write:
  * the shared engine (Game.slide) for moves and rewards,
  * P5's NTupleEvaluator for leaf values,
  * P3's ExpectimaxAgent for the search recursion,
  * a plain-Python reference (agents/hybrid/reference.py).
A randomly filled two-stage network is used, so values are non-trivial (the
all-zero weights used by conftest would make most comparisons vacuous).
"""

import math
import random

import numpy as np
import pytest

# Needs P5's n-tuple package. Until it is merged into main, this whole file is skipped (CI stays green).
pytest.importorskip("agents.ntuple.network", reason="merge P5 (agents/ntuple) first")

from agents.hybrid import HybridAgent, Mode65k, cpp_available, reference
from agents.hybrid.registry import hybrid_agents
from agents.ntuple.agent import NTupleAgent, NTupleEvaluator
from agents.ntuple.board import from_exponents, to_exponents
from agents.ntuple.network import CELLS, StagedNetwork
from engine.game import Game

needs_cpp = pytest.mark.skipif(not cpp_available(), reason="build: python agents/hybrid/cpp/setup.py build_ext")


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def evaluator():
    """Random 2-stage network (stage switch at 2048). ~540 MB, built once."""
    rng = np.random.default_rng(7)
    w = rng.standard_normal(size=(2, 4, 16**6), dtype=np.float32)
    w *= 40.0
    return NTupleEvaluator(network=StagedNetwork(n_stages=2, stage_tiles=[2048], weights=w))


def make_boards(n, seed=0, scale_up=True):
    """Mid-game boards from random play; some scaled up to hold big tiles."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        g = Game(rng.randrange(10**6))
        for _ in range(rng.randrange(5, 220)):
            legal = g.legal_moves()
            if not legal:
                break
            g.step(rng.choice(legal))
        if g.done:
            continue
        b = [row[:] for row in g.board]
        if scale_up and rng.random() < 0.5:
            k = rng.randrange(1, 5)
            if g.max_tile() * 2**k <= 65536:
                b = [[v * 2**k if v else 0 for v in row] for row in b]
        out.append(b)
    return out


def pick(values):
    best = max(values.values())
    return min(m for m, v in values.items() if v == best)


# --------------------------------------------------------------- C++ vs engine
@needs_cpp
def test_cpp_slide_matches_engine(evaluator):
    from agents.hybrid import _hsearch

    s = _hsearch.HybridSearcher(evaluator.network.weights, evaluator.network.stage_exps, CELLS)
    boards = make_boards(1500, seed=1)
    # Edge cases: no double merges, and two 32,768 tiles make 65,536.
    boards += [
        [[2, 2, 2, 2], [0] * 4, [0] * 4, [0] * 4],
        [[4, 2, 2, 0], [2, 2, 0, 0], [0] * 4, [8, 8, 8, 0]],
        [[32768, 32768, 0, 0], [2, 0, 0, 0], [0] * 4, [0] * 4],
        [[32768, 2, 0, 0], [32768, 4, 0, 0], [0] * 4, [0] * 4],
    ]
    for b in boards:
        for move in range(4):
            want, reward = Game.slide(b, move)
            got, got_reward, changed = s.slide(to_exponents(b), move)
            assert from_exponents(got) == want, (b, move)
            assert got_reward == reward
            assert changed == (want != b)


@needs_cpp
def test_cpp_eval_matches_ntuple_evaluator(evaluator):
    from agents.hybrid import _hsearch

    s = _hsearch.HybridSearcher(evaluator.network.weights, evaluator.network.stage_exps, CELLS)
    boards = make_boards(2000, seed=2)
    boards.append([[65536, 32768, 0, 0], [2, 0, 0, 0], [0] * 4, [0] * 4])   # exponent 16 is clamped
    stages = set()
    for b in boards:
        stages.add(evaluator.stage_of(b))
        assert s.evaluate(to_exponents(b)) == evaluator.evaluate(b)   # same arithmetic order: exact
    assert stages == {0, 1}   # both stage networks were exercised


@needs_cpp
def test_cpp_rejects_bad_input(evaluator):
    from agents.hybrid import _hsearch

    net = evaluator.network
    with pytest.raises(TypeError):
        _hsearch.HybridSearcher(net.weights.astype(np.float64)[:, :, :4], net.stage_exps, CELLS)
    s = _hsearch.HybridSearcher(net.weights, net.stage_exps, CELLS)
    with pytest.raises(ValueError):
        s.evaluate(np.full(16, 30, dtype=np.uint8))
    with pytest.raises(ValueError):
        s.evaluate(np.zeros(15, dtype=np.uint8))


# ------------------------------------------------------------ C++ vs reference
@needs_cpp
@pytest.mark.parametrize("depth,n_boards,cutoff", [(0, 1000, 0.0), (1, 1000, 1e-4), (2, 12, 1e-2)])
def test_cpp_search_matches_reference(evaluator, depth, n_boards, cutoff):
    """Same values (to rounding) and the same chosen move on many boards."""
    agent = HybridAgent(evaluator=evaluator, depth=depth, cutoff=cutoff, backend="cpp")
    for b in make_boards(n_boards, seed=3 + depth):
        want = reference.move_values(b, evaluator, depth, cutoff)
        got = agent.move_values(b)
        assert set(got) == set(want)
        for m in want:
            assert got[m] == pytest.approx(want[m], rel=1e-9, abs=1e-6), (b, m)
        # The chosen move may only differ from the reference's when values tie.
        assert want[pick(got)] == pytest.approx(max(want.values()), rel=1e-9, abs=1e-6)


@needs_cpp
def test_cpp_and_python_agents_play_identically(evaluator):
    cpp = HybridAgent(evaluator=evaluator, depth=1, backend="cpp")
    py = HybridAgent(evaluator=evaluator, depth=1, backend="python")
    g = Game(seed=5)
    for _ in range(40):
        if g.done:
            break
        assert cpp.choose_move(g) == py.choose_move(g)
        g.step(cpp.choose_move(g))


def test_depth0_is_the_ntuple_agent(evaluator):
    hybrid = HybridAgent(evaluator=evaluator, depth=0)
    ntuple = NTupleAgent(evaluator=evaluator)
    for b in make_boards(300, seed=9, scale_up=False):
        g = Game(seed=0, board=b)
        assert hybrid.choose_move(g) == ntuple.choose_move(g)


# ------------------------------------------------------------------ vs P3
def test_depth1_matches_p3_expectimax(evaluator):
    """P3's expectimax with an afterstate-greedy leaf is the same recursion."""
    search = pytest.importorskip("agents.expectimax.search")
    leaf = reference.AfterstateGreedyLeaf(evaluator)
    p3 = search.ExpectimaxAgent(leaf, depth=1, cutoff=0, use_tt=False)
    hybrid = HybridAgent(evaluator=evaluator, depth=1, cutoff=0.0, use_tt=False)
    for b in make_boards(25, seed=11):
        got = hybrid.move_values(b)
        for move, v in p3.move_values(b).items():
            _, reward = Game.slide(b, move)
            assert got[move] == pytest.approx(reward + v, rel=1e-9, abs=1e-6)


def test_depth2_matches_p3_expectimax_plus_move_rewards(evaluator):
    """At depth 2, P3's search differs from ours in ONE way: P3's max node does not add the
    merge reward of the intermediate move (its heuristic is not measured in score). The n-tuple
    value V estimates *score still to come*, so for it Q = reward + V(afterstate) is required.
    Add exactly that to P3's class and the two searches must agree: it checks the recursion,
    chance weighting and leaf handling against code we did not write."""
    search = pytest.importorskip("agents.expectimax.search")

    class P3WithRewards(search.ExpectimaxAgent):
        def _max(self, board, depth, prob):
            best = None
            for move in range(4):
                after, reward = Game.slide(board, move)
                if after == board:
                    continue
                v = reward + self._chance(after, depth, prob)
                if best is None or v > best:
                    best = v
            return 0.0 if best is None else best

    leaf = reference.AfterstateGreedyLeaf(evaluator)
    p3 = P3WithRewards(leaf, depth=2, cutoff=0, use_tt=False)
    hybrid = HybridAgent(evaluator=evaluator, depth=2, cutoff=0.0, use_tt=False)
    for b in make_boards(2, seed=12):
        got = hybrid.move_values(b)
        for move, v in p3.move_values(b).items():
            _, reward = Game.slide(b, move)
            assert got[move] == pytest.approx(reward + v, rel=1e-9, abs=1e-6)


# ---------------------------------------------- cutoff / TT / threads / time
@needs_cpp
def test_transposition_table_does_not_change_values_without_cutoff(evaluator):
    a = HybridAgent(evaluator=evaluator, depth=2, cutoff=0.0, use_tt=True)
    b = HybridAgent(evaluator=evaluator, depth=2, cutoff=0.0, use_tt=False)
    for board in make_boards(6, seed=13):
        va, vb = a.move_values(board), b.move_values(board)
        assert va.keys() == vb.keys()
        for m in va:
            assert va[m] == pytest.approx(vb[m], rel=1e-9, abs=1e-6)


@needs_cpp
def test_cutoff_prunes_nodes_but_cutoff_zero_is_full_search(evaluator):
    full = HybridAgent(evaluator=evaluator, depth=2, cutoff=0.0)
    cut = HybridAgent(evaluator=evaluator, depth=2, cutoff=1e-2)
    # An open board: with ~12 empty cells, branch probabilities fall below 1e-2 by depth 2.
    board = [[2, 4, 0, 0], [8, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]]
    full.move_values(board)
    cut.move_values(board)
    assert cut.last_info["nodes"] < full.last_info["nodes"]


@needs_cpp
def test_threads_do_not_change_the_answer(evaluator):
    one = HybridAgent(evaluator=evaluator, depth=2, threads=1)
    four = HybridAgent(evaluator=evaluator, depth=2, threads=4)
    for board in make_boards(8, seed=15):
        assert one.move_values(board) == four.move_values(board)   # bit-for-bit


@needs_cpp
def test_iterative_deepening_matches_fixed_depth(evaluator):
    from agents.hybrid import _hsearch

    s = _hsearch.HybridSearcher(evaluator.network.weights, evaluator.network.stage_exps, CELLS)
    for board in make_boards(5, seed=16):
        e = to_exponents(board)
        fixed = s.search(e, 2, 1e-4, True, 0.0, 1, False)
        iterative = s.search(e, 2, 1e-4, True, 0.0, 1, True)
        assert iterative["depth"] == 2 and not iterative["aborted"]
        assert iterative["values"] == fixed["values"] or all(
            (math.isnan(a) and math.isnan(b)) or a == b for a, b in zip(iterative["values"], fixed["values"])
        )


@needs_cpp
@pytest.mark.parametrize("budget_ms", [5.0, 30.0, 80.0])
def test_time_limit_is_respected(evaluator, budget_ms):
    agent = HybridAgent(evaluator=evaluator, depth=8, time_ms=budget_ms, cutoff=0.0, threads=2)
    for board in make_boards(5, seed=17, scale_up=False):
        g = Game(seed=0, board=board)
        t0 = __import__("time").perf_counter()
        move = agent.choose_move(g)
        elapsed = 1000 * (__import__("time").perf_counter() - t0)
        assert move in g.legal_moves()
        assert agent.last_info["depth"] >= 0                       # at least the greedy pass finished
        assert elapsed < budget_ms + 150, f"{elapsed:.1f} ms for a {budget_ms} ms budget"


@needs_cpp
def test_more_time_gives_deeper_search(evaluator):
    board = make_boards(1, seed=18, scale_up=False)[0]
    g = Game(seed=0, board=board)
    quick = HybridAgent(evaluator=evaluator, depth=6, time_ms=2.0, cutoff=1e-4)
    slow = HybridAgent(evaluator=evaluator, depth=6, time_ms=400.0, cutoff=1e-4)
    quick.choose_move(g)
    slow.choose_move(g)
    assert slow.last_info["depth"] > quick.last_info["depth"]


# -------------------------------------------------------------- 65,536 mode
def test_mode65k_switches_settings(evaluator):
    agent = HybridAgent(
        evaluator=evaluator, depth=2, cutoff=1e-4, time_ms=10,
        mode65k=Mode65k(trigger_tile=16384, extra_depth=2, cutoff=1e-6, time_ms=50),
    )
    small = [[2048, 4, 0, 0], [0] * 4, [0] * 4, [0] * 4]
    big = [[16384, 4, 0, 0], [0] * 4, [0] * 4, [0] * 4]
    s, b = agent.settings_for(small), agent.settings_for(big)
    assert (s.depth, s.cutoff, s.time_ms, s.mode65k_active) == (2, 1e-4, 10, False)
    assert (b.depth, b.cutoff, b.time_ms, b.mode65k_active) == (4, 1e-6, 50, True)
    plain = HybridAgent(evaluator=evaluator, depth=2)
    assert plain.settings_for(big).depth == 2           # no mode -> never changes


def test_mode65k_searches_deeper_only_past_the_trigger(evaluator):
    agent = HybridAgent(evaluator=evaluator, depth=0, mode65k=Mode65k(trigger_tile=1024, extra_depth=1))
    low = Game(seed=0, board=[[512, 2, 0, 0], [0] * 4, [0] * 4, [0] * 4])
    high = Game(seed=0, board=[[1024, 2, 0, 0], [0] * 4, [0] * 4, [0] * 4])
    agent.choose_move(low)
    assert agent.last_info["requested_depth"] == 0 and not agent.last_info["mode65k"]
    agent.choose_move(high)
    assert agent.last_info["requested_depth"] == 1 and agent.last_info["mode65k"]


def test_late_game_board_makes_65536(evaluator):
    """From a position with two 32,768 tiles the agent takes the merge."""
    board = [[32768, 32768, 0, 0], [2, 4, 0, 0], [0] * 4, [0] * 4]
    for depth in (0, 1, 2):
        g = Game(seed=3, board=board)
        agent = HybridAgent(evaluator=evaluator, depth=depth)
        move = agent.choose_move(g)
        assert move in (1, 3)                    # a horizontal merge
        g.step(move)
        assert g.max_tile() == 65536 and g.score >= 65536


def test_65536_boards_are_searchable(evaluator):
    g = Game(seed=1, board=[[65536, 32768, 16384, 0], [2, 0, 0, 0], [0] * 4, [0] * 4])
    agent = HybridAgent(evaluator=evaluator, depth=2, mode65k=Mode65k())
    for _ in range(25):
        if g.done:
            break
        g.step(agent.choose_move(g))
    assert g.max_tile() >= 65536


# ------------------------------------------------------------------- agent API
def test_dead_board_has_no_values(evaluator):
    dead = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
    agent = HybridAgent(evaluator=evaluator, depth=1)
    assert agent.move_values(dead) == {}
    with pytest.raises(RuntimeError):
        agent.choose_move(Game(seed=0, board=dead))


def test_agent_does_not_mutate_game(evaluator):
    agent = HybridAgent(evaluator=evaluator, depth=1)
    g = Game(seed=4)
    for _ in range(30):
        before, moves, score = [r[:] for r in g.board], list(g.moves), g.score
        m = agent.choose_move(g)
        assert (g.board, g.moves, g.score) == (before, moves, score)
        assert m in g.legal_moves()
        g.step(m)


def test_registry_names_and_weights_keyword():
    import inspect

    agents = hybrid_agents()
    assert set(agents) == {"hybrid-d1", "hybrid-d2", "hybrid-d3", "hybrid-65k"}
    for factory in agents.values():
        assert "weights" in inspect.signature(factory).parameters   # --weights works like for `ntuple`


def test_registered_agents_are_legal_and_do_not_mutate():
    """Capped-length version of tests/test_agents.py's checks, for ALL hybrid agents
    (hybrid-d3 is excluded there because whole games are too slow for CI).
    conftest points $NTUPLE_WEIGHTS at all-zero weights, so the agents play on merge reward."""
    for name, factory in hybrid_agents().items():
        agent = factory()
        assert agent.name == name
        for seed in range(2):
            g = Game(seed)
            agent.reset(seed)
            for _ in range(40):
                if g.done:
                    break
                before = [r[:] for r in g.board]
                move = agent.choose_move(g)
                assert move in g.legal_moves(), (name, seed)
                assert g.board == before, f"{name} mutated the game"
                g.step(move)


def test_registered_agents_are_reproducible_over_a_capped_game():
    """Same seed -> same moves (no clock, no thread races, no hidden randomness)."""
    for name, factory in hybrid_agents().items():
        runs = []
        for _ in range(2):
            agent, g = factory(), Game(11)
            agent.reset(11)
            for _ in range(40):
                if g.done:
                    break
                g.step(agent.choose_move(g))
            runs.append("".join(map(str, g.moves)))
        assert runs[0] == runs[1], name
