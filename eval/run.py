"""Shared-seed benchmark and late-game board collector."""

from __future__ import annotations

import argparse
import inspect
import json
import statistics
import time
from pathlib import Path
from typing import Callable, Dict, List

from agents.base import Agent
from agents.baselines import (
    CornerAgent,
    GreedyAgent,
    MCTSAgent,
    MinimaxAgent,
    MonteCarloAgent,
    RandomAgent,
)
from agents.heuristics import HeuristicEvaluator, heuristic_agents
from agents.hybrid.registry import hybrid_agents
from agents.ntuple.agent import make_ntuple_agent
from engine.game import Game


def make_corner_agent():
    return CornerAgent()


def make_mcts_agent():
    return MCTSAgent()


def make_minimax_agent():
    return MinimaxAgent()


def make_monte_carlo_agent():
    return MonteCarloAgent()


def make_minimax_heuristic():
    evaluator = HeuristicEvaluator.from_file("tuned")
    return MinimaxAgent(depth=1, evaluator=evaluator)


def make_dqn_agent(weights: str | None = None) -> Agent:
    from agents.deeprl.agent import DQNAgent

    return DQNAgent(checkpoint=weights)


AGENTS: Dict[str, Callable[..., Agent]] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "corner": make_corner_agent,
    "mcts": make_mcts_agent,
    "minimax": make_minimax_agent,
    "monte_carlo": make_monte_carlo_agent,
    "minimax_heuristic": make_minimax_heuristic,
    "ntuple": make_ntuple_agent,
    "dqn": make_dqn_agent,
}

AGENTS.update(heuristic_agents())
AGENTS.update(hybrid_agents())
MILESTONES = (2048, 4096, 8192, 16384, 32768, 65536)


def play_game(agent: Agent, seed: int, boards_file=None) -> dict:
    game = Game(seed)
    agent.reset(seed)
    think_time = 0.0
    board_saved = False

    while not game.done:
        legal = game.legal_moves()

        t0 = time.perf_counter()
        move = agent.choose_move(game)
        think_time += time.perf_counter() - t0

        if move not in legal:
            raise RuntimeError(
                f"agent {agent.name!r} returned illegal move "
                f"{move!r} (legal: {legal}) on seed {seed}"
            )

        game.step(move)

        # Save the first board that reaches 32768.
        if not board_saved and game.max_tile() >= 32768:
            if boards_file is not None:
                board_snapshot = (
                    [row[:] for row in game.board]
                    if game.board and isinstance(game.board[0], (list, tuple))
                    else list(game.board)
                )
                board_record = {
                    "agent": agent.name,
                    "seed": seed,
                    "score": game.score,
                    "move_number": len(game.moves),
                    "max_tile": game.max_tile(),
                    "board": board_snapshot,
                }

                boards_file.write(json.dumps(board_record) + "\n")
                boards_file.flush()

                print(
                    f"Saved 32768 board: "
                    f"agent={agent.name}, seed={seed}"
                )

            board_saved = True

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
        "mean_ms_per_move": statistics.fmean(
            r["ms_per_move"] for r in results
        ),
        "reach": {
            t: sum(r["max_tile"] >= t for r in results) / len(results)
            for t in MILESTONES
        },
    }


def print_summary(name: str, s: dict) -> None:
    print(f"agent={name} games={s['games']}")
    print(f"  mean score   {s['mean_score']:.1f}")
    print(f"  median score {s['median_score']:.1f}")
    print(f"  ms/move      {s['mean_ms_per_move']:.4f}")

    for t, rate in s["reach"].items():
        print(f"  reach {t:>5}  {100 * rate:6.2f}%")


def main(argv: List[str] | None = None) -> dict:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--agent", required=True, choices=sorted(AGENTS))
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--out-dir", default="results")
    p.add_argument(
        "--append",
        action="store_true",
        help="Append results instead of replacing an existing file",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing result file",
    )

    p.add_argument(
        "--weights",
        default=None,
        help="Path to trained weights (currently used by the N-Tuple agent)",
    )

    p.add_argument(
        "--boards-out",
        default="results/late_game_boards.jsonl",
        help="File where boards reaching 32768 are saved",
    )

    args = p.parse_args(argv)

    if args.games < 1:
        p.error("--games must be at least 1")
    if args.append and args.overwrite:
        p.error("Use either --append or --overwrite, not both")

    factory = AGENTS[args.agent]

    if args.weights is not None:
        if "weights" not in inspect.signature(factory).parameters:
            p.error(f"--weights is not supported by agent {args.agent!r}")
        agent = factory(weights=args.weights)
    else:
        agent = factory()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.agent}.jsonl"
    if out_path.exists() and not (args.append or args.overwrite):
        p.error(
            f"{out_path} already exists. Use --append to add games "
            "or --overwrite to replace it."
        )

    boards_path = Path(args.boards_out)
    boards_path.parent.mkdir(parents=True, exist_ok=True)

    results = []

    with (
        out_path.open(
            "a" if args.append else "w",
            encoding="utf-8",
        ) as f,
        boards_path.open("a", encoding="utf-8") as boards_file,
    ):
        for i in range(args.games):
            r = play_game(agent, args.seed0 + i, boards_file)
            results.append(r)
            f.write(json.dumps(r) + "\n")

    s = summarize(results)

    print_summary(args.agent, s)
    print(f"  wrote {out_path}")
    print(f"  late-game boards saved in {boards_path}")

    return s


if __name__ == "__main__":
    main()