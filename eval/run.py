"""Shared-seed benchmark. The ONLY source of numbers for the report.

    python -m eval.run --agent random --games 100 --seed0 0

Game i uses seed ``seed0 + i`` for both the engine and the agent, so every
agent faces identical tile spawns. Writes results/<agent>.jsonl with one
line per game:
    {"agent", "seed", "score", "max_tile", "n_moves", "moves", "ms_per_move"}
``moves`` is the move list as a digit string ("0"=up "1"=right "2"=down
"3"=left); eval/replay.py turns it back into the game.

To add an agent: import it and add a zero-argument factory to AGENTS.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable, Dict, List

from agents.base import Agent
from agents.baselines import GreedyAgent, RandomAgent
from agents.expectimax.search import ExpectimaxAgent
from agents.heuristics import heuristic_agents
from engine.game import Game

AGENTS: Dict[str, Callable[[], Agent]] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    **heuristic_agents(),
}

HEUR_KEY = "heuristic"   # P2's tuned evaluator


def _heuristic_evaluator():
    # Reuse P2's evaluator by taking it from P2's own registered agent
    return heuristic_agents()[HEUR_KEY]().evaluator


def _expectimax(depth, **kw):
    name = kw.pop("name", f"expectimax-d{depth}")
    return lambda: ExpectimaxAgent(_heuristic_evaluator(), depth=depth, name=name, **kw)


AGENTS["expectimax-d1"] = _expectimax(1)
AGENTS["expectimax-d2"] = _expectimax(2)
AGENTS["expectimax-d2-c01"] = _expectimax(2, cutoff=0.01, name="expectimax-d2-c01")
AGENTS["expectimax-adaptive"] = _expectimax(1, adaptive=True, name="expectimax-adaptive")

MILESTONES = (2048, 4096, 8192, 16384, 32768, 65536)


def play_game(agent: Agent, seed: int) -> dict:
    game = Game(seed)
    agent.reset(seed)
    think_time = 0.0
    while not game.done:
        legal = game.legal_moves()
        t0 = time.perf_counter()
        move = agent.choose_move(game)
        think_time += time.perf_counter() - t0
        if move not in legal:
            raise RuntimeError(
                f"agent {agent.name!r} returned illegal move {move!r} (legal: {legal}) on seed {seed}"
            )
        game.step(move)
    n = len(game.moves)
    return {
        "agent": agent.name,
        "seed": seed,
        "score": game.score,
        "max_tile": game.max_tile(),
        "n_moves": n,
        "moves": "".join(str(m) for m in game.moves),
        "ms_per_move": round(1000 * think_time / n, 4) if n else 0.0,
    }


def summarize(results: List[dict]) -> dict:
    scores = [r["score"] for r in results]
    return {
        "games": len(results),
        "mean_score": statistics.fmean(scores),
        "median_score": statistics.median(scores),
        "mean_ms_per_move": statistics.fmean(r["ms_per_move"] for r in results),
        "reach": {t: sum(r["max_tile"] >= t for r in results) / len(results) for t in MILESTONES},
    }


def print_summary(name: str, s: dict) -> None:
    print(f"agent={name} games={s['games']}")
    print(f"  mean score   {s['mean_score']:.1f}")
    print(f"  median score {s['median_score']:.1f}")
    print(f"  ms/move      {s['mean_ms_per_move']:.4f}")
    for t, rate in s["reach"].items():
        print(f"  reach {t:>5}  {100 * rate:6.2f}%")


def main(argv: List[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", required=True, choices=sorted(AGENTS))
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--out-dir", default="results")
    args = p.parse_args(argv)

    agent = AGENTS[args.agent]()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.agent}.jsonl"

    results = []
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(args.games):
            r = play_game(agent, args.seed0 + i)
            results.append(r)
            f.write(json.dumps(r) + "\n")

    s = summarize(results)
    print_summary(args.agent, s)
    print(f"  wrote {out_path}")
    return s


if __name__ == "__main__":
    main()
