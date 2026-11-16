"""Measure Game.step() throughput of the CURRENT engine (P1).

    python -m engine.bench [--seconds 3] [--seed 0]
    ENGINE_PURE_PYTHON=1 python -m engine.bench      # force the Python fallback

Plays random games (moves chosen with their own random.Random, so the game's
seeded RNG is not touched) and reports full Game.step() calls per second,
including the Python spawn and the legal-move check. It prints which backend
ran, so numbers from the two modes are never confused. Results depend on the
machine; quote them together with the machine and Python version.
"""

import argparse
import random
import time

from engine import game as G


def run(seconds: float, seed: int = 0) -> dict:
    chooser = random.Random(seed)
    steps = games = 0
    start = time.perf_counter()
    elapsed = 0.0
    while elapsed < seconds:
        game = G.Game(seed + games)
        while not game.done:
            game.step(chooser.choice(game.legal_moves()))
            steps += 1
        games += 1
        elapsed = time.perf_counter() - start
    return {"backend": G.backend(), "games": games, "moves": steps, "seconds": elapsed,
            "moves_per_sec": steps / elapsed}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    r = run(args.seconds, args.seed)
    print(f"backend={r['backend']} games={r['games']} moves={r['moves']} "
          f"seconds={r['seconds']:.2f} moves/sec={r['moves_per_sec']:.0f}")
    return r


if __name__ == "__main__":
    main()
