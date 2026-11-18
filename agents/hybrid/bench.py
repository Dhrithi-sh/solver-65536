"""Speed benchmark for the hybrid search: pure Python (before) vs C++ (after) vs C++ + threads.

    python -m agents.hybrid.bench --weights weights/ntuple/run1/latest.ntw

Positions are snapshots from a real greedy n-tuple game, so memory-access
patterns in the weight tables are realistic. This measures speed only; strength
numbers must come from `python -m eval.run` (see scripts/hybrid_experiments.sh).
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

from agents.hybrid import HybridAgent, cpp_available
from agents.ntuple.agent import NTupleAgent, NTupleEvaluator
from engine.game import Game


def collect_positions(evaluator: NTupleEvaluator, n: int, seed: int = 0, every: int = 40):
    """Boards from one greedy n-tuple game, one every `every` moves."""
    agent, g, out = NTupleAgent(evaluator=evaluator), Game(seed), []
    while not g.done and len(out) < n:
        g.step(agent.choose_move(g))
        if len(g.moves) % every == 0:
            out.append([row[:] for row in g.board])
    return out


def time_config(evaluator, positions, **kw) -> dict:
    agent = HybridAgent(evaluator=evaluator, **kw)
    ms, nodes = [], []
    for b in positions:
        t0 = time.perf_counter()
        agent.move_values(b)
        ms.append(1000 * (time.perf_counter() - t0))
        nodes.append(agent.last_info.get("nodes", 0))
    return {"ms": statistics.fmean(ms), "worst_ms": max(ms), "nodes": statistics.fmean(nodes)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=None)
    p.add_argument("--positions", type=int, default=20)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--cutoff", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--python-depth", type=int, default=1, help="deepest depth to time in pure Python")
    args = p.parse_args(argv)

    ev = NTupleEvaluator(args.weights)
    positions = collect_positions(ev, args.positions)
    print(f"{len(positions)} positions from a greedy n-tuple game; cutoff={args.cutoff}; cpus={os.cpu_count()}")
    rows = []
    for d in range(args.python_depth + 1):
        rows.append(("python", d, 1, time_config(ev, positions[: max(3, len(positions) // 4)], depth=d,
                                                 cutoff=args.cutoff, backend="python")))
    if cpp_available():
        for d in range(args.max_depth + 1):
            for th in sorted({1, args.threads}):
                rows.append(("cpp", d, th, time_config(ev, positions, depth=d, cutoff=args.cutoff,
                                                       backend="cpp", threads=th)))
    else:
        print("C++ extension not built: python agents/hybrid/cpp/setup.py build_ext")
    print(f"\n{'backend':8} {'depth':>5} {'threads':>7} {'ms/move':>10} {'worst ms':>10} {'chance nodes':>13}")
    for be, d, th, r in rows:
        print(f"{be:8} {d:>5} {th:>7} {r['ms']:>10.3f} {r['worst_ms']:>10.3f} {r['nodes']:>13.0f}")
    py = {d: r["ms"] for be, d, th, r in rows if be == "python"}
    for be, d, th, r in rows:
        if be == "cpp" and th == 1 and d in py:
            print(f"speed-up C++ vs Python at depth {d}: {py[d] / r['ms']:.0f}x")


if __name__ == "__main__":
    main()
