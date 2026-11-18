
"""Shared-seed benchmark and late-game board collector."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable, Dict, List

from agents.base import Agent
from agents.baselines import GreedyAgent, RandomAgent
from engine.game import Game

AGENTS: Dict[str, Callable[[], Agent]] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
}

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
                board_record = {
                    "agent": agent.name,
                    "seed": seed,
                    "score": game.score,
                    "move_number": len(game.moves),
                    "max_tile": game.max_tile(),
                    "board": [row[:] for row in game.board],
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
        "--boards-out",
        default="results/late_game_boards.jsonl",
        help="File where boards reaching 32768 are saved",
    )

    args = p.parse_args(argv)

    if args.games < 1:
        p.error("--games must be at least 1")

    agent = AGENTS[args.agent]()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.agent}.jsonl"

    boards_path = Path(args.boards_out)
    boards_path.parent.mkdir(parents=True, exist_ok=True)

    results = []

    with out_path.open("w", encoding="utf-8") as f, \
         boards_path.open("a", encoding="utf-8") as boards_file:

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
