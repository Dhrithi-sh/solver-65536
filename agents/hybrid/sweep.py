"""Tune the hybrid agent on the DEV seeds only (0-999).

    python -m agents.hybrid.sweep --weights W --games 30 --depths 1 2 --cutoffs 1e-3 1e-4 \
        --triggers none 4096 8192 --procs 2

For every (depth, cutoff, 65k-trigger) combination it plays the same dev seeds through
eval.run.play_game (the shared harness) and prints/saves a table. It REFUSES seeds
outside the dev range: the final test seeds (100000-100999) must never be used for
tuning, otherwise the reported result is optimistically biased.

`--triggers` are max-tile thresholds for 65,536 mode ("none" = mode off). The real
trigger is 16,384, but short experiments rarely reach it, so you can also sweep
lower values (4096, 8192) to see whether "think harder when the board is big" helps.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from multiprocessing import Pool
from pathlib import Path

from agents.hybrid import HybridAgent, Mode65k
from agents.ntuple.agent import NTupleEvaluator
from eval.run import MILESTONES, play_game

DEV_SEEDS = range(0, 1000)
_EV = None


def _init(weights):
    global _EV
    _EV = NTupleEvaluator(weights)


def _make(depth, cutoff, trigger, boost):
    mode = None if trigger is None else Mode65k(trigger_tile=trigger, extra_depth=boost, cutoff=cutoff / 10)
    name = f"hybrid-d{depth}-c{cutoff:g}" + ("" if trigger is None else f"-t{trigger}+{boost}")
    return HybridAgent(evaluator=_EV, depth=depth, cutoff=cutoff, mode65k=mode, name=name)


def _play(job):
    depth, cutoff, trigger, boost, seed = job
    return play_game(_make(depth, cutoff, trigger, boost), seed)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=None)
    p.add_argument("--games", type=int, default=30)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2])
    p.add_argument("--cutoffs", type=float, nargs="+", default=[1e-4])
    p.add_argument("--triggers", nargs="+", default=["none"])
    p.add_argument("--boost", type=int, default=1, help="extra depth in 65k mode")
    p.add_argument("--procs", type=int, default=1)
    p.add_argument("--out", default="results/hybrid_sweep.csv")
    args = p.parse_args(argv)

    seeds = list(range(args.seed0, args.seed0 + args.games))
    if not all(s in DEV_SEEDS for s in seeds):
        raise SystemExit(f"refusing seeds {seeds[0]}..{seeds[-1]}: tuning is restricted to dev seeds 0-999")
    triggers = [None if t == "none" else int(t) for t in args.triggers]

    rows = []
    with Pool(args.procs, initializer=_init, initargs=(args.weights,)) as pool:
        for depth in args.depths:
            for cutoff in args.cutoffs:
                for trig in triggers:
                    res = pool.map(_play, [(depth, cutoff, trig, args.boost, s) for s in seeds])
                    row = {
                        "depth": depth, "cutoff": cutoff, "trigger": trig or "none", "games": len(res),
                        "mean_score": statistics.fmean(r["score"] for r in res),
                        "median_score": statistics.median(r["score"] for r in res),
                        "ms_per_move": statistics.fmean(r["ms_per_move"] for r in res),
                        **{f"reach_{t}": sum(r["max_tile"] >= t for r in res) / len(res) for t in MILESTONES},
                    }
                    rows.append(row)
                    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
