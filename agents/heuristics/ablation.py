"""Ablation sweep: retune with all features, then with each feature removed.

    python -m agents.heuristics.ablation --games 64 --iters 40

Writes weights/ablation/full.json and weights/ablation/no_<feature>.json.
Every run gets the same budget, dev seeds and CMA-ES seed, so the only
difference between runs is the missing feature. Arguments this script does
not know (--games, --iters, --sigma, --popsize, --workers, ...) are passed
straight to tune.py.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

from agents.heuristics import tune
from agents.heuristics.features import FEATURES


def variants() -> Dict[str, List[str]]:
    """Run name -> features dropped in that run."""
    return {"full": [], **{f"no_{f}": [f] for f in FEATURES}}


def main(argv: Optional[List[str]] = None) -> Dict[str, dict]:
    """Run the sweep. Returns each run's ``meta`` block by run name."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cma-seed", type=int, default=1, help="CMA-ES seed shared by every run")
    p.add_argument("--only", nargs="+", choices=sorted(variants()), help="run just these variants")
    args, passthrough = p.parse_known_args(argv)
    for flag in ("--out", "--drop", "--mode", "--log"):
        if flag in passthrough:
            raise SystemExit(f"{flag} is set by the sweep itself")

    metas = {}
    for name, dropped in variants().items():
        if args.only and name not in args.only:
            continue
        print(f"=== ablation/{name} ===", flush=True)
        run = ["--mode", "single", "--out", f"ablation/{name}", "--cma-seed", str(args.cma_seed)]
        for feature in dropped:
            run += ["--drop", feature]
        metas[name] = tune.main(run + passthrough)
    return metas


if __name__ == "__main__":
    main()
