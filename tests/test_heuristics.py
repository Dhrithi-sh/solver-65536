import csv
import json
import random

import pytest

import engine.game

from agents.base import ValueFunction
from agents.heuristics import HeuristicAgent, HeuristicEvaluator, agent_name, heuristic_agents
from agents.heuristics import ablation, report, tune
from agents.heuristics import evaluator as evaluator_module
from agents.heuristics.evaluator import DEFAULT_PHASE_BOUNDS, PHASES, WEIGHTS_DIR, phase_of, resolve_weights_path
from agents.heuristics.features import (
    FEATURES,
    empty_count,
    exponents,
    max_in_corner,
    merge_potential,
    monotonicity,
    smoothness,
    snake,
    symmetries,
)
from engine.game import Game
from eval.run import AGENTS

# Exponent grids with answers worked out by hand.
ROW = [[4, 3, 2, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
MESSY = [[1, 3, 2, 0], [1, 3, 0, 0], [0, 0, 5, 0], [0, 0, 0, 0]]
EMPTY = [[0] * 4 for _ in range(4)]
SNAKE = [[16, 15, 14, 13], [9, 10, 11, 12], [8, 7, 6, 5], [1, 2, 3, 4]]

PHASED = {"early": {"empty": 1.0}, "mid": {"empty": 2.0, "snake": 0.5}, "late": {"empty": 3.0, "mono": 1.0}}


def to_board(e):
    return [[2**k if k else 0 for k in row] for row in e]


def random_grids(n, seed=0):
    rng = random.Random(seed)
    return [[[rng.choice([0, 0, 1, 1, 2, 3, 4, 5, 8, 11, 16]) for _ in range(4)] for _ in range(4)] for _ in range(n)]


# ------------------------------------------------------------------ features
def test_exponents():
    board = [[0, 2, 4, 8], [16, 2048, 16384, 32768], [65536, 0, 0, 0], [0, 0, 0, 2]]
    assert exponents(board) == [[0, 1, 2, 3], [4, 11, 14, 15], [16, 0, 0, 0], [0, 0, 0, 1]]


def test_exponents_follows_the_engine_convention():
    with pytest.raises(ValueError):
        exponents([[3, 0, 0, 0]] + EMPTY[1:])
    assert exponents([[2**31, 131072, 0, 0]] + EMPTY[1:])[0] == [31, 17, 0, 0]
    assert exponents(tuple(map(tuple, to_board(SNAKE)))) == SNAKE  # tuples work, like Game.slide
    engine_exponents = getattr(engine.game, "exponents", None)
    if engine_exponents is not None:  # present once the engine helpers are merged
        for grid in random_grids(20, seed=3) + [SNAKE]:
            assert engine_exponents(to_board(grid)).tolist() == exponents(to_board(grid))


def test_symmetries_are_the_8_distinct_images():
    images = symmetries(SNAKE)
    assert len({tuple(map(tuple, g)) for g in images}) == 8
    assert SNAKE in images and [list(r) for r in zip(*SNAKE)] in images


def test_empty_count():
    assert empty_count(ROW) == 12
    assert empty_count(MESSY) == 10
    assert empty_count(EMPTY) == 16
    assert empty_count(SNAKE) == 0


def test_monotonicity():
    assert monotonicity(ROW) == 0  # every line only falls
    assert monotonicity(SNAKE) == 0
    # rows: [1,3,2,0] -> min(2,3)=2, [1,3,0,0] -> 2, [0,0,5,0] -> 5; column [2,0,5,0] -> min(5,7)=5
    assert monotonicity(MESSY) == -14
    assert monotonicity([[1, 2, 1, 2]] + EMPTY[1:]) == -1  # rises 2, falls 1


def test_smoothness_skips_empty_cells():
    assert smoothness(ROW) == -3  # 4-3, 3-2, 2-1; the empty rows below add nothing
    assert smoothness(MESSY) == -5  # rows 2+1 and 2; the lone 5 has only empty neighbours
    assert smoothness([[7, 0, 7, 0]] + EMPTY[1:]) == 0
    assert smoothness([[3, 3, 0, 0], [3, 3, 0, 0]] + EMPTY[2:]) == 0


def test_max_in_corner():
    assert max_in_corner(ROW) == 1
    assert max_in_corner(MESSY) == 0
    assert max_in_corner(EMPTY) == 0
    assert max_in_corner([[5, 0, 0, 0], [0, 5, 0, 0]] + EMPTY[2:]) == 1  # a tied max in a corner counts
    assert max_in_corner([[1, 5, 0, 0]] + EMPTY[1:]) == 0


def test_snake():
    assert snake(EMPTY) == 0
    assert snake(ROW) == pytest.approx(4 + 3 / 2 + 2 / 4 + 1 / 8)
    assert snake(SNAKE) == pytest.approx(sum((16 - k) * 0.5**k for k in range(16)))
    # A centre cell is at best 6th on a zig-zag path, so weight 0.5**5.
    assert snake([[0, 0, 0, 0], [0, 5, 0, 0]] + EMPTY[2:]) == pytest.approx(5 / 32)
    # Sorted along the path beats the same tiles shuffled.
    assert snake(SNAKE) > snake([row[::-1] if i == 0 else row for i, row in enumerate(SNAKE)])


def test_merge_potential():
    assert merge_potential(ROW) == 0
    assert merge_potential(MESSY) == 2  # the vertical 1-1 and 3-3 pairs
    assert merge_potential(EMPTY) == 0  # empty neighbours are not a pair
    assert merge_potential([[2, 2, 2, 0], [2, 0, 0, 0]] + EMPTY[2:]) == 3


def test_features_dict_is_complete():
    assert list(FEATURES) == ["empty", "mono", "smooth", "corner", "snake", "merge"]


@pytest.mark.parametrize("name", FEATURES)
def test_feature_is_symmetric(name):
    for grid in random_grids(25) + [ROW, MESSY, SNAKE]:
        expected = FEATURES[name](grid)
        for image in symmetries(grid):
            assert FEATURES[name](image) == pytest.approx(expected)


# ----------------------------------------------------------------- evaluator
def test_evaluator_is_the_weighted_sum():
    weights = {"empty": 2.0, "mono": 1.5, "smooth": 0.5, "corner": 3.0, "snake": 1.0, "merge": 4.0}
    for grid in (ROW, MESSY, SNAKE):
        expected = sum(w * FEATURES[f](grid) for f, w in weights.items())
        assert HeuristicEvaluator(weights).evaluate(to_board(grid)) == pytest.approx(expected)
    # MESSY by hand: 2*10 + 1.5*-14 + 0.5*-5 + 3*0 + 4*2, plus the snake term
    assert HeuristicEvaluator({**weights, "snake": 0.0}).evaluate(to_board(MESSY)) == pytest.approx(4.5)


def test_missing_features_count_as_zero():
    assert HeuristicEvaluator({"empty": 2.0}).evaluate(to_board(MESSY)) == 20
    assert HeuristicEvaluator({"empty": 2.0, "mono": 0.0}).evaluate(to_board(MESSY)) == 20
    assert HeuristicEvaluator({}).evaluate(to_board(MESSY)) == 0


def test_zero_weight_features_are_not_computed(monkeypatch):
    def boom(e):
        raise AssertionError("zero-weight feature was computed")

    monkeypatch.setitem(evaluator_module.FEATURES, "snake", boom)
    assert HeuristicEvaluator({"empty": 1.0, "snake": 0.0}).evaluate(to_board(ROW)) == 12


def test_unknown_feature_names_are_rejected():
    with pytest.raises(ValueError, match="unknown feature"):
        HeuristicEvaluator({"empty": 1.0, "emtpy": 1.0})
    with pytest.raises(ValueError, match="unknown feature"):
        HeuristicEvaluator({**PHASED, "late": {"bogus": 1.0}})
    with pytest.raises(ValueError, match="phases"):
        HeuristicEvaluator({"early": {"empty": 1.0}, "mid": {"empty": 1.0}})


@pytest.mark.parametrize(
    "max_tile, phase",
    [(2, "early"), (1024, "early"), (2048, "mid"), (4096, "mid"), (8192, "mid"), (16384, "late"), (32768, "late"), (65536, "late")],
)
def test_phase_selection(max_tile, phase):
    assert phase_of(max_tile) == phase
    ev = HeuristicEvaluator(PHASED)
    board = to_board(EMPTY)
    board[1][2] = max_tile
    assert ev.phase(board) == phase
    expected = sum(w * FEATURES[f](exponents(board)) for f, w in PHASED[phase].items())
    assert ev.evaluate(board) == pytest.approx(expected)


def test_custom_phase_bounds():
    ev = HeuristicEvaluator(PHASED, phase_bounds={"mid": 512, "late": 4096})
    assert ev.phase([[256, 0, 0, 0]] + EMPTY[1:]) == "early"
    assert ev.phase([[512, 0, 0, 0]] + EMPTY[1:]) == "mid"
    assert ev.phase([[4096, 0, 0, 0]] + EMPTY[1:]) == "late"
    assert HeuristicEvaluator({"empty": 1.0}).phase(to_board(ROW)) is None


@pytest.mark.parametrize("weights", [{"empty": 2.7, "mono": 1.0, "smooth": 0.1, "corner": 2.0, "snake": 1.0, "merge": 0.7}, PHASED])
def test_evaluator_is_symmetric(weights):
    ev = HeuristicEvaluator(weights)
    for grid in random_grids(25, seed=1) + [SNAKE]:
        expected = ev.evaluate(to_board(grid))
        for image in symmetries(grid):
            assert ev.evaluate(to_board(image)) == pytest.approx(expected)


def test_evaluate_does_not_mutate_the_board():
    board = to_board(MESSY)
    before = [row[:] for row in board]
    HeuristicEvaluator(PHASED).evaluate(board)
    assert board == before


@pytest.mark.parametrize("weights, bounds", [({"empty": 1.5, "snake": -0.25}, None), (PHASED, {"mid": 1024, "late": 8192})])
def test_save_load_roundtrip(tmp_path, weights, bounds):
    path = tmp_path / "sub" / "w.json"
    ev = HeuristicEvaluator(weights, phase_bounds=bounds, meta={"source": "test"})
    assert ev.save(path) == path
    loaded = HeuristicEvaluator.from_file(path)
    assert loaded.to_dict() == ev.to_dict()
    assert loaded.meta == {"source": "test"}
    for grid in random_grids(10, seed=2):
        assert loaded.evaluate(to_board(grid)) == ev.evaluate(to_board(grid))
    data = json.loads(path.read_text())
    assert ("phase_bounds" in data) == (bounds is not None)


def test_bare_names_resolve_to_the_weights_folder(tmp_path):
    assert resolve_weights_path("tuned") == WEIGHTS_DIR / "tuned.json"
    assert resolve_weights_path("ablation/no_snake") == WEIGHTS_DIR / "ablation" / "no_snake.json"
    assert resolve_weights_path(tmp_path / "x.json") == tmp_path / "x.json"


def test_weight_file_needs_exactly_one_weight_block():
    with pytest.raises(ValueError):
        HeuristicEvaluator.from_dict({"meta": {}})
    with pytest.raises(ValueError):
        HeuristicEvaluator.from_dict({"weights": {}, "phases": PHASED})


COMMITTED = sorted(WEIGHTS_DIR.rglob("*.json"))


def test_untuned_weights_are_committed():
    assert WEIGHTS_DIR / "untuned.json" in COMMITTED


@pytest.mark.parametrize("path", COMMITTED, ids=lambda p: p.relative_to(WEIGHTS_DIR).as_posix())
def test_committed_weight_file_loads(path):
    ev = HeuristicEvaluator.from_file(path)
    assert ev.meta.get("source"), "every weight file records how it was produced"
    if ev.phased:
        assert ev.phase_bounds == DEFAULT_PHASE_BOUNDS
    assert isinstance(ev.evaluate(to_board(MESSY)), float)
    assert agent_name(path.relative_to(WEIGHTS_DIR).with_suffix("").as_posix()) in AGENTS


# --------------------------------------------------------------------- agent
class Scripted(ValueFunction):
    """Returns a fixed value per afterstate (default 0)."""

    def __init__(self, values=()):
        self.values = list(values)

    def evaluate(self, board):
        return next((v for b, v in self.values if b == board), 0.0)


def test_agent_picks_the_best_afterstate():
    game = Game(0, board=[[2, 0, 0, 0], [0, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert game.legal_moves() == [0, 1, 2, 3]
    for target in range(4):
        after, _ = Game.slide(game.board, target)
        assert HeuristicAgent(Scripted([(after, 1.0)])).choose_move(game) == target


def test_agent_ties_go_to_the_lowest_legal_move():
    assert HeuristicAgent(Scripted()).choose_move(Game(0, board=to_board(MESSY))) == 0
    game = Game(0, board=[[2, 4, 0, 0]] + EMPTY[1:])  # up is illegal here
    assert game.legal_moves() == [1, 2]
    assert HeuristicAgent(Scripted()).choose_move(game) == 1
    after2, _ = Game.slide(game.board, 2)
    after1, _ = Game.slide(game.board, 1)
    assert HeuristicAgent(Scripted([(after1, 5.0), (after2, 5.0)])).choose_move(game) == 1


def test_agent_ignores_the_merge_reward():
    game = Game(0, board=EMPTY[:3] + [[1024, 1024, 0, 0]])
    assert Game.slide(game.board, 0)[1] == 0 and Game.slide(game.board, 1)[1] == 2048
    assert HeuristicAgent(Scripted()).choose_move(game) == 0


def test_agent_does_not_mutate_the_game():
    game = Game(3)
    before = [row[:] for row in game.board]
    move = HeuristicAgent(HeuristicEvaluator(PHASED)).choose_move(game)
    assert move in game.legal_moves() and game.board == before and game.moves == []


# -------------------------------------------------------------- registration
def test_registration_follows_the_weight_files(tmp_path):
    assert heuristic_agents(tmp_path) == {}
    ev = HeuristicEvaluator({"empty": 1.0})
    for stem in ("untuned", "tuned", "ablation/full", "ablation/no_snake", "scratch"):
        ev.save(tmp_path / f"{stem}.json")
    HeuristicEvaluator(PHASED).save(tmp_path / "phased.json")
    agents = heuristic_agents(tmp_path)
    assert sorted(agents) == ["heuristic", "heuristic-full", "heuristic-no-snake", "heuristic-phased", "heuristic-untuned"]
    for name, factory in agents.items():
        agent = factory()
        assert agent.name == name
        assert agent.choose_move(Game(0)) in Game(0).legal_moves()
    assert agents["heuristic-phased"]().evaluator.phased


def test_untuned_agent_is_in_the_benchmark():
    assert "heuristic-untuned" in AGENTS
    assert AGENTS["heuristic-untuned"]().name == "heuristic-untuned"


def test_ablation_variants_match_registration_and_report():
    names = [agent_name(f"ablation/{v}") for v in ablation.variants()]
    assert names == report.TABLES["ablation"][1]
    assert ablation.variants()["no_snake"] == ["snake"] and ablation.variants()["full"] == []


# -------------------------------------------------------------------- tuning
def test_single_layout():
    layout = tune.make_layout("single")
    assert layout.features == tuple(FEATURES) and layout.dim == 6
    assert layout.to_weights([2, 1, 0, -4, 0.5, 1]) == {
        "empty": 0.5, "mono": 0.25, "smooth": 0.0, "corner": -1.0, "snake": 0.125, "merge": 0.25,
    }  # scaled so the largest |weight| is 1


def test_layout_drops_features():
    layout = tune.make_layout("single", drop=["snake", "corner"])
    assert layout.features == ("empty", "mono", "smooth", "merge") and layout.dim == 4
    assert "snake" not in layout.to_weights([1, 1, 1, 1])
    with pytest.raises(ValueError):
        tune.make_layout("single", drop=["nope"])
    with pytest.raises(ValueError):
        layout.to_weights([1, 2, 3])


def test_phased_layout_ties_untuned_phases_to_the_nearest_tuned_one():
    layout = tune.make_layout("phased", drop=["mono", "smooth", "corner", "merge"])
    assert layout.phases == ("early", "mid") and layout.dim == 4
    w = layout.to_weights([1.0, 0.5, 0.25, -0.5])
    assert w["early"] == {"empty": 1.0, "snake": 0.5}
    assert w["mid"] == {"empty": 0.25, "snake": -0.5}
    assert w["late"] == w["mid"] and w["late"] is not w["mid"]
    assert set(w) == set(PHASES)

    only_mid = tune.make_layout("phased", phases=["mid"])
    assert [only_mid.source_phase(p) for p in PHASES] == ["mid", "mid", "mid"]
    ends = tune.make_layout("phased", phases=["late", "early"])
    assert ends.phases == ("early", "late")
    assert [ends.source_phase(p) for p in PHASES] == ["early", "early", "late"]
    assert tune.make_layout("phased", phases=list(PHASES)).dim == 18


def test_layout_start_vector():
    single = HeuristicEvaluator({"empty": 2.0, "snake": 1.0})
    phased = HeuristicEvaluator(PHASED)
    assert tune.make_layout("single").to_vector(single) == [2.0, 0, 0, 0, 1.0, 0]
    layout = tune.make_layout("phased", drop=["smooth", "corner", "merge"])
    assert layout.to_vector(single) == [2.0, 0, 1.0, 2.0, 0, 1.0]  # every phase starts from the single set
    assert layout.to_vector(phased) == [1.0, 0, 0, 2.0, 0, 0.5]
    with pytest.raises(ValueError):
        tune.make_layout("single").to_vector(phased)


def test_tuning_refuses_test_seeds():
    tune._check_seeds(0, 64)
    tune._check_seeds(999, 1)
    for seed0, games in ((100_000, 1), (99_990, 20), (10_500, 10), (100_999, 5)):
        with pytest.raises(SystemExit):
            tune._check_seeds(seed0, games)


def test_expectimax_player_fails_clearly_until_it_exists():
    try:
        from agents.expectimax import ExpectimaxAgent  # noqa: F401
    except ImportError:
        with pytest.raises(SystemExit, match="ExpectimaxAgent"):
            tune.make_agent(HeuristicEvaluator({"empty": 1.0}), "expectimax", 1)
    assert isinstance(tune.make_agent(HeuristicEvaluator({"empty": 1.0})), HeuristicAgent)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("mode", ["single", "phased"])
def test_tiny_tuning_run(tmp_path, mode, capsys):
    pytest.importorskip("cma")
    out, log = tmp_path / "w.json", tmp_path / "w.csv"
    meta = tune.main([
        "--mode", mode, "--games", "1", "--iters", "1", "--popsize", "2", "--workers", "1",
        "--drop", "snake", "--out", str(out), "--log", str(log),
    ])  # fmt: skip
    ev = HeuristicEvaluator.from_file(out)
    assert ev.phased == (mode == "phased")
    assert ev.meta == meta and meta["generations"] == 1 and meta["dev_seeds"] == [0, 0]
    assert meta["dev_mean_score"] >= meta["init_dev_mean_score"]  # best-so-far includes the start
    sets = ev.phases.values() if ev.phased else [ev.weights]
    assert all(set(w) == set(FEATURES) - {"snake"} for w in sets)
    rows = list(csv.reader(log.open()))
    assert rows[0][0] == "generation" and len(rows) == 3


# -------------------------------------------------------------------- report
def test_report_table_formats_and_skips_missing_agents():
    def summary(mean):
        return {"mean_score": mean, "median_score": mean - 1, "mean_ms_per_move": 0.05, "reach": {t: 0.25 for t in report.REACH}}

    text = report.markdown_table("T", ["a", "missing", "b"], {"a": summary(1000.0), "b": summary(1500.0)})
    lines = text.splitlines()
    assert "missing" not in text and len(lines) == 6
    assert lines[4].startswith("| `a` | 1,000 | — | 999 | 25.0% |")
    assert "+500 (+50.0%)" in lines[5]
    assert "No agents" in report.markdown_table("T", ["missing"], {})
