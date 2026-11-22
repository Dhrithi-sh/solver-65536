"""Quick late-game sanity check: 1-ply n-tuple agent from saved positions, several weights files.

    python scripts/ntuple_late_sanity.py weights/ntuple/late/late_positions_seed128_eval.jsonl \\
        --weights weights/ntuple/long/ntuple-plain-1m.ntw weights/ntuple/late/run50k/latest.ntw \\
        --spawn-seeds 65000-65009 --out results/late-sanity-seed128.jsonl

Start boards: by default the FIRST position of every source game in the file (so with
the seed-128 held-out file: 4 starts, two holding 16,384 and two holding 32,768); use
--min-tile to require a bigger max tile, or --all-positions-every K to take every K-th.
Each (start, spawn seed) game is ``Game(spawn_seed, board=start)`` played to the end by
``NTupleAgent``, so every weights file sees the identical spawn sequence from the same
start. Prints, per weights file, how many games reached 32,768 / 65,536 and the mean
score gained after the start.

This is a sanity check for P5's fine-tune, not a report number: benchmark numbers for
the report must come from ``python -m eval.run`` (or P8's late-game benchmark).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.ntuple.agent import NTupleAgent  # noqa: E402
from engine.game import Game  # noqa: E402


def parse_seeds(s: str) -> List[int]:
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def select_starts(path: Path, min_tile: int, every: int) -> List[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["max_tile"] >= min_tile]
    if every:
        return rows[::every]
    first, seen = [], set()
    for r in rows:
        key = r.get("game_id", r.get("source_file"))
        if key not in seen:
            seen.add(key)
            first.append(r)
    return first


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("positions", type=Path)
    p.add_argument("--weights", nargs="+", required=True)
    p.add_argument("--spawn-seeds", default="65000-65009")
    p.add_argument("--min-tile", type=int, default=0)
    p.add_argument("--all-positions-every", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="optional JSONL, one line per game")
    a = p.parse_args(argv)
    starts = select_starts(a.positions, a.min_tile, a.all_positions_every)
    seeds = parse_seeds(a.spawn_seeds)
    print(f"{len(starts)} start positions x {len(seeds)} spawn seeds ({seeds[0]}..{seeds[-1]}) per weights file")
    out_f = None
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        out_f = a.out.open("w", encoding="utf-8", newline="\n")
    for w in a.weights:
        agent = NTupleAgent(weights_path=w)
        n = r32 = r65 = 0
        gained = 0
        t0 = time.perf_counter()
        for st in starts:
            for s in seeds:
                g = Game(s, board=[row[:] for row in st["board"]])
                agent.reset(s)
                while not g.done:
                    g.step(agent.choose_move(g))
                n += 1
                r32 += g.max_tile() >= 32768
                r65 += g.max_tile() >= 65536
                gained += g.score
                if out_f:
                    out_f.write(json.dumps({
                        "weights": w, "source_seed": st["source_seed"], "move_index": st["move_index"],
                        "start_max_tile": st["max_tile"], "spawn_seed": s, "score_gained": g.score,
                        "max_tile": g.max_tile(), "n_moves": len(g.moves),
                        "moves": "".join(map(str, g.moves)),
                    }) + "\n")
        dt = time.perf_counter() - t0
        print(f"{w}: games {n} | reached 32768: {r32} | reached 65536: {r65} | "
              f"mean score gained {gained / n:,.0f} | {dt:.1f}s")
    if out_f:
        out_f.close()


if __name__ == "__main__":
    main()
