"""Measure random-play throughput of the CURRENT engine (P1).

    python -m engine.bench [--seconds 3] [--seed 0] [--mode step|batch|both] [--threads N]
    ENGINE_PURE_PYTHON=1 python -m engine.bench      # force the Python fallback

Two measurements, both END TO END (spawns, RNG, legal-move checks and the move
choice are all inside the timed loop; nothing is precomputed outside it):

  step   one Python Game.step() per move, moves chosen by a separate
         random.Random. This is what a Python agent sees.
  batch  engine.batch.play_random_batch: whole random games, in C++ across
         threads when the extension is built (pure Python otherwise, so the
         number is still honest). Table build and thread start-up are included
         in the timing of the first call, so a warm-up call is NOT made.

It prints the backend and thread count with every number, so results from
different modes are never confused. Results depend on the machine; quote them
together with the machine and Python version (printed).
"""

import argparse
import os
import platform
import random
import time

from engine import batch as B
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


def run_batch(seconds: float, seed: int = 0, threads: int = 0, chunk: int = 2000) -> dict:
    """Play random-policy games in chunks of ``chunk`` until ``seconds`` have passed."""
    moves = games = 0
    start = time.perf_counter()
    elapsed = 0.0
    while elapsed < seconds:
        results = B.play_random_batch(range(seed + games, seed + games + chunk), threads=threads,
                                      record_moves=False)
        moves += sum(r["n_moves"] for r in results)
        games += chunk
        elapsed = time.perf_counter() - start
    return {"backend": G.backend(), "threads": threads, "games": games, "moves": moves,
            "seconds": elapsed, "moves_per_sec": moves / elapsed}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=("step", "batch", "both"), default="both")
    ap.add_argument("--threads", type=int, default=0, help="batch mode worker threads (0 = all cores)")
    args = ap.parse_args(argv)
    print(f"machine: {platform.machine()} cpus={os.cpu_count()} python={platform.python_version()}")
    out = {}
    if args.mode in ("step", "both"):
        r = out["step"] = run(args.seconds, args.seed)
        print(f"step  backend={r['backend']} games={r['games']} moves={r['moves']} "
              f"seconds={r['seconds']:.2f} moves/sec={r['moves_per_sec']:.0f}")
    if args.mode in ("batch", "both"):
        # Games are played in chunks and the clock is checked between chunks, so a chunk
        # must be short: 2000 games take ~0.1 s in C++ but ~20 s in pure Python (hence 100).
        r = out["batch"] = run_batch(args.seconds, args.seed, args.threads,
                                     chunk=2000 if G.backend() == "cpp" else 100)
        print(f"batch backend={r['backend']} threads={r['threads'] or 'all'} games={r['games']} moves={r['moves']} "
              f"seconds={r['seconds']:.2f} moves/sec={r['moves_per_sec']:.0f}")
    return out


if __name__ == "__main__":
    main()
