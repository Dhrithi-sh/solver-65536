"""Paired comparison table from eval.run outputs (results/<agent>.jsonl).

    python -m agents.hybrid.compare results/dev/ntuple.jsonl results/dev/hybrid-d1.jsonl ...

Only seeds played by EVERY listed agent are used, so all rows see identical tile
spawns even if the runs had different game counts. Numbers still come from eval.run.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

MILESTONES = (2048, 4096, 8192, 16384, 32768, 65536)


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return rows[0]["agent"], {r["seed"]: r for r in rows}


def main(paths):
    data = dict(load(p) for p in paths)
    common = sorted(set.intersection(*(set(d) for d in data.values())))
    if not common:
        raise SystemExit("no seeds in common")
    print(f"{len(common)} common seeds ({common[0]}..{common[-1]})\n")
    head = f"{'agent':14} {'mean':>9} {'median':>9} {'ms/move':>8} " + " ".join(f"{t:>6}" for t in MILESTONES)
    print(head)
    for name, d in data.items():
        rs = [d[s] for s in common]
        reach = " ".join(f"{100 * sum(r['max_tile'] >= t for r in rs) / len(rs):5.0f}%" for t in MILESTONES)
        print(f"{name:14} {statistics.fmean(r['score'] for r in rs):9.0f} {statistics.median(r['score'] for r in rs):9.0f} "
              f"{statistics.fmean(r['ms_per_move'] for r in rs):8.2f} {reach}")


if __name__ == "__main__":
    main(sys.argv[1:])
