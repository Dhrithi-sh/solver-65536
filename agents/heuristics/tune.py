"""Tune the heuristic weights with CMA-ES.

    python -m agents.heuristics.tune --games 64 --iters 40                 # -> weights/tuned.json
    python -m agents.heuristics.tune --mode phased --init tuned --sigma 0.2  # -> weights/phased.json

Objective: mean benchmark score over a fixed block of dev seeds. Games are
played through ``eval.run.play_game`` so tuning and benchmarking share one
code path. The best-so-far weights are saved after every generation.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import statistics
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from agents.base import Agent
from agents.heuristics.agent import HeuristicAgent
from agents.heuristics.evaluator import (
    DEFAULT_PHASE_BOUNDS,
    PHASES,
    WEIGHTS_DIR,
    HeuristicEvaluator,
    resolve_weights_path,
)
from agents.heuristics.features import FEATURES

LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "heuristics"

# Final-result seeds (the brief's range and the one in eval/README.md).
# Tuning on either would leak the test set, so it is refused outright.
TEST_SEED_RANGES = ((10_000, 10_999), (100_000, 100_999))


@dataclass(frozen=True)
class Layout:
    """How a flat CMA-ES vector maps onto weight sets.

    Single mode: one value per feature. Phased mode: one block of
    ``features`` per tuned phase, in ``PHASES`` order; phases that are not
    tuned copy the nearest tuned one (ties go to the earlier phase).
    """

    mode: str
    features: Tuple[str, ...]
    phases: Tuple[str, ...] = ()

    @property
    def dim(self) -> int:
        return len(self.features) * max(1, len(self.phases))

    def source_phase(self, phase: str) -> str:
        """The tuned phase whose weights ``phase`` uses."""
        i = PHASES.index(phase)
        return min(self.phases, key=lambda p: (abs(PHASES.index(p) - i), PHASES.index(p)))

    def to_weights(self, x: Sequence[float]) -> Dict[str, Any]:
        """Vector -> weights, scaled so the largest |weight| is 1.

        Play only depends on the direction of the weight vector (see README),
        so fixing the scale loses nothing and keeps files comparable. The
        scaled, rounded weights are what gets *played*, so a saved file
        reproduces its tuning score exactly.
        """
        if len(x) != self.dim:
            raise ValueError(f"expected {self.dim} values, got {len(x)}")
        scale = max(abs(float(v)) for v in x) or 1.0
        vals = [round(float(v) / scale, 6) for v in x]
        n = len(self.features)
        if self.mode == "single":
            return dict(zip(self.features, vals))
        blocks = {p: dict(zip(self.features, vals[k * n : (k + 1) * n])) for k, p in enumerate(self.phases)}
        return {p: dict(blocks[self.source_phase(p)]) for p in PHASES}

    def to_vector(self, evaluator: HeuristicEvaluator) -> List[float]:
        """Starting vector from existing weights (missing features start at 0)."""
        if self.mode == "single":
            if evaluator.phased:
                raise ValueError("cannot start a single-set run from per-phase weights")
            return [evaluator.weights.get(f, 0.0) for f in self.features]
        sets = evaluator.phases if evaluator.phased else {p: evaluator.weights for p in PHASES}
        return [sets[p].get(f, 0.0) for p in self.phases for f in self.features]


def make_layout(mode: str, drop: Sequence[str] = (), phases: Optional[Sequence[str]] = None) -> Layout:
    """Build the layout for a run; ``drop`` removes features (ablation)."""
    unknown = sorted(set(drop) - set(FEATURES))
    if unknown:
        raise ValueError(f"cannot drop unknown feature(s) {unknown}")
    features = tuple(f for f in FEATURES if f not in drop)
    if not features:
        raise ValueError("every feature was dropped")
    if mode == "single":
        return Layout("single", features)
    # Late copies mid by default: 1-ply games almost never reach 16,384, so
    # the late weights would get no signal from the objective.
    chosen = set(phases or ("early", "mid"))
    return Layout("phased", features, tuple(p for p in PHASES if p in chosen))


def make_agent(evaluator: HeuristicEvaluator, player: str = "1ply", depth: int = 1) -> Agent:
    """The agent that plays the tuning games."""
    if player == "1ply":
        return HeuristicAgent(evaluator)
    try:
        from agents.expectimax import ExpectimaxAgent  # lazy: may not exist yet
    except ImportError:
        raise SystemExit(
            "--player expectimax needs agents.expectimax.ExpectimaxAgent(evaluator, depth), "
            "which is not in the repo yet. Use --player 1ply until it lands."
        ) from None
    return ExpectimaxAgent(evaluator, depth)


def _score(task: Tuple[Dict[str, Any], int, str, int]) -> int:
    """Worker: play one benchmark game with the given weights."""
    from eval.run import play_game  # imported here to avoid an import cycle with eval.run

    weights, seed, player, depth = task
    bounds = DEFAULT_PHASE_BOUNDS if "early" in weights else None
    return play_game(make_agent(HeuristicEvaluator(weights, bounds), player, depth), seed)["score"]


def mean_scores(
    candidates: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    player: str = "1ply",
    depth: int = 1,
    pool: Optional[Any] = None,
) -> List[float]:
    """Mean score of each candidate over ``seeds`` (all games in one batch)."""
    tasks = [(dict(w), s, player, depth) for w in candidates for s in seeds]
    scores = pool.map(_score, tasks, chunksize=4) if pool else list(map(_score, tasks))
    n = len(seeds)
    return [statistics.fmean(scores[i * n : (i + 1) * n]) for i in range(len(candidates))]


def _check_seeds(seed0: int, games: int) -> None:
    for lo, hi in TEST_SEED_RANGES:
        if seed0 <= hi and seed0 + games - 1 >= lo:
            raise SystemExit(f"seeds {seed0}..{seed0 + games - 1} overlap the test seeds {lo}..{hi}; never tune on those")


def default_log_path(out_path: Path) -> Path:
    """logs/heuristics/<name>.csv, e.g. ablation/no_snake -> ablation_no_snake.csv."""
    try:
        rel = out_path.resolve().relative_to(WEIGHTS_DIR)
    except ValueError:
        rel = Path(out_path.name)
    return LOG_DIR / ("_".join(rel.with_suffix("").parts) + ".csv")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("single", "phased"), default="single")
    p.add_argument("--phases", nargs="+", choices=PHASES, help="phases to tune in phased mode (default: early mid)")
    p.add_argument("--drop", action="append", default=[], metavar="FEATURE", help="leave a feature out (repeatable)")
    p.add_argument("--init", default="untuned", help="starting weights: a name in the weights folder or a .json path")
    p.add_argument("--games", type=int, default=64, help="dev seeds per candidate")
    p.add_argument("--seed0", type=int, default=0, help="first dev seed")
    p.add_argument("--iters", type=int, default=40, help="CMA-ES generations")
    p.add_argument("--sigma", type=float, default=0.5, help="initial step size (weights are scaled to max |w| = 1)")
    p.add_argument("--popsize", type=int, default=None, help="candidates per generation (default: CMA-ES's own)")
    p.add_argument("--cma-seed", type=int, default=1, help="CMA-ES sampling seed (must be > 0)")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--player", choices=("1ply", "expectimax"), default="1ply")
    p.add_argument("--depth", type=int, default=1, help="search depth for --player expectimax")
    p.add_argument("--out", default=None, help="output weights (default: tuned or phased, by mode)")
    p.add_argument("--log", default=None, help="per-generation CSV (default: logs/heuristics/<out>.csv)")
    return p


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run the tuner. Returns the saved file's ``meta`` block."""
    args = build_parser().parse_args(argv)
    try:
        import cma
    except ImportError:
        raise SystemExit("tuning needs cma: pip install -r agents/heuristics/requirements.txt") from None

    _check_seeds(args.seed0, args.games)
    if args.cma_seed <= 0:
        raise SystemExit("--cma-seed must be > 0 (cma treats 0 as 'pick a random seed')")
    if args.phases and args.mode != "phased":
        raise SystemExit("--phases only applies to --mode phased")
    layout = make_layout(args.mode, args.drop, args.phases)
    seeds = list(range(args.seed0, args.seed0 + args.games))
    out_path = resolve_weights_path(args.out or ("tuned" if args.mode == "single" else "phased"))
    log_path = Path(args.log) if args.log else default_log_path(out_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    make_agent(HeuristicEvaluator({}), args.player, args.depth)  # fail early if the player is unavailable

    x0 = layout.to_vector(HeuristicEvaluator.from_file(args.init))
    scale = max(abs(v) for v in x0) or 1.0
    x0 = [v / scale for v in x0]  # same scale as the candidates, so sigma is relative

    opts: Dict[str, Any] = {"seed": args.cma_seed, "verbose": -9}
    if args.popsize:
        opts["popsize"] = args.popsize
    es = cma.CMAEvolutionStrategy(x0, args.sigma, opts)

    command = "python -m agents.heuristics.tune " + shlex.join(sys.argv[1:] if argv is None else argv)
    meta: Dict[str, Any] = {
        "source": "cma-es",
        "command": command,
        "mode": args.mode,
        "features": list(layout.features),
        "tuned_phases": list(layout.phases),
        "player": args.player if args.player == "1ply" else f"expectimax-d{args.depth}",
        "dev_seeds": [seeds[0], seeds[-1]],
        "init": args.init,
        "sigma0": args.sigma,
        "popsize": es.popsize,
        "cma_seed": args.cma_seed,
        "iters": args.iters,
        "note": "dev_mean_score is the best of many candidates on the tuning seeds, so it is optimistic; "
        "report numbers come from eval.run on the test seeds.",
    }

    def save(weights: Mapping[str, Any]) -> None:
        bounds = DEFAULT_PHASE_BOUNDS if args.mode == "phased" else None
        HeuristicEvaluator(weights, bounds, meta).save(out_path)

    pool = Pool(args.workers) if args.workers > 1 else None
    t0 = time.time()
    try:
        best_w = layout.to_weights(x0)
        best = init_score = mean_scores([best_w], seeds, args.player, args.depth, pool)[0]
        meta.update(init_dev_mean_score=round(init_score, 1), dev_mean_score=round(best, 1), generations=0, improved_on_init=False)
        save(best_w)
        print(f"init {args.init}: dev mean {init_score:.1f} | dim {layout.dim} popsize {es.popsize} -> {out_path}", flush=True)

        with log_path.open("w", newline="", encoding="utf-8") as f:
            log = csv.writer(f)
            log.writerow(["generation", "gen_best", "gen_mean", "best_so_far", "sigma", "elapsed_s", "best_weights"])
            log.writerow([0, init_score, init_score, best, args.sigma, 0.0, best_w])
            for gen in range(1, args.iters + 1):
                xs = es.ask()
                cands = [layout.to_weights(x) for x in xs]
                means = mean_scores(cands, seeds, args.player, args.depth, pool)
                es.tell(xs, [-m for m in means])  # CMA-ES minimises
                top = max(range(len(means)), key=means.__getitem__)
                if means[top] > best:
                    best, best_w = means[top], cands[top]
                meta.update(dev_mean_score=round(best, 1), generations=gen, improved_on_init=best > init_score)
                save(best_w)
                elapsed = time.time() - t0
                log.writerow([gen, means[top], statistics.fmean(means), best, es.sigma, round(elapsed, 1), best_w])
                f.flush()
                print(
                    f"gen {gen:3d}/{args.iters}  best {means[top]:9.1f}  mean {statistics.fmean(means):9.1f}  "
                    f"best-so-far {best:9.1f}  sigma {es.sigma:.3f}  {elapsed:6.0f}s",
                    flush=True,
                )
                if es.stop():
                    print(f"CMA-ES stopped early: {es.stop()}", flush=True)
                    break
    finally:
        if pool:
            pool.close()
            pool.join()

    print(f"saved {out_path} (dev mean {best:.1f}, init {init_score:.1f}); log {log_path}")
    return meta


if __name__ == "__main__":
    main()
