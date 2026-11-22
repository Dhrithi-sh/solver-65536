"""Build the results tables from the shared benchmark.

    python -m agents.heuristics.report --table all --games 1000 --seed0 100000

Every number comes from ``eval.run.main`` on the test seeds; this script
only arranges them into markdown. Agents whose weight file does not exist
yet are skipped.
"""

from __future__ import annotations

import argparse
import contextlib
import io
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agents.heuristics.features import FEATURES

REACH = (2048, 4096, 8192, 16384)

# Table -> (title, agents); the first agent is the baseline of the diff column.
TABLES: Dict[str, Tuple[str, List[str]]] = {
    "tuning": ("Tuning (1-ply)", ["greedy", "heuristic-untuned", "heuristic"]),
    "phases": ("Single vs per-phase weights (1-ply)", ["heuristic", "heuristic-phased"]),
    "ablation": ("Ablation (1-ply, each run retuned)", ["heuristic-full"] + [f"heuristic-no-{f}" for f in FEATURES]),
}


def _benchmark(job: Tuple[str, int, int, str]) -> Tuple[str, dict]:
    from eval.run import main as run_benchmark

    agent, games, seed0, out_dir = job
    with contextlib.redirect_stdout(io.StringIO()):
        summary = run_benchmark(["--agent", agent, "--games", str(games), "--seed0", str(seed0), "--out-dir", out_dir])
    return agent, summary


def markdown_table(title: str, agents: List[str], summaries: Dict[str, dict]) -> str:
    """One markdown table; the first agent present is the baseline."""
    rows = [a for a in agents if a in summaries]
    if not rows:
        return f"### {title}\n\n_No agents available yet._\n"
    base = summaries[rows[0]]["mean_score"]
    lines = [
        f"### {title}",
        "",
        "| agent | mean score | vs " + rows[0] + " | median | " + " | ".join(str(t) for t in REACH) + " | ms/move |",
        "|---|---:|---:|---:|" + "---:|" * len(REACH) + "---:|",
    ]
    for a in rows:
        s = summaries[a]
        diff = "—" if a == rows[0] else f"{s['mean_score'] - base:+,.0f} ({100 * (s['mean_score'] - base) / base:+.1f}%)"
        reach = " | ".join(f"{100 * s['reach'][t]:.1f}%" for t in REACH)
        lines.append(
            f"| `{a}` | {s['mean_score']:,.0f} | {diff} | {s['median_score']:,.0f} | {reach} | {s['mean_ms_per_move']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> str:
    """Run the benchmarks and return (and print) the markdown."""
    from eval.run import AGENTS

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", choices=[*TABLES, "all"], default="all")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--seed0", type=int, default=100_000, help="first test seed")
    p.add_argument("--out-dir", default="results", help="where eval.run writes its .jsonl files")
    p.add_argument("--workers", type=int, default=1, help="agents benchmarked in parallel (ms/move is noisier above 1)")
    p.add_argument("--out", type=Path, default=None, help="also write the markdown to this file")
    args = p.parse_args(argv)

    tables = list(TABLES) if args.table == "all" else [args.table]
    wanted: List[str] = []
    for t in tables:
        for agent in TABLES[t][1]:
            if agent not in AGENTS:
                print(f"skipping {agent}: not registered (no weight file yet)")
            elif agent not in wanted:
                wanted.append(agent)

    jobs = [(a, args.games, args.seed0, args.out_dir) for a in wanted]
    if args.workers > 1 and len(jobs) > 1:
        with Pool(args.workers) as pool:
            summaries = dict(pool.map(_benchmark, jobs, chunksize=1))
    else:
        summaries = dict(map(_benchmark, jobs))

    header = (
        f"Seeds {args.seed0}–{args.seed0 + args.games - 1} ({args.games} games) via `python -m eval.run`. "
        "Reach columns: share of games whose max tile is at least that value.\n"
    )
    text = header + "\n" + "\n".join(markdown_table(TABLES[t][0], TABLES[t][1], summaries) for t in tables)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    main()
