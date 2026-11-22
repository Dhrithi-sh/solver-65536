"""Extract late-game start positions (max tile >= 16,384) from logged games.

    python scripts/extract_late_positions.py RESULTS_DIR \\
        --out weights/ntuple/late/late_positions_train.jsonl \\
        --heldout-out weights/ntuple/late/late_positions_seed128_eval.jsonl

Every game is REPLAYED from its seed and move string with ``engine.game.Game`` (the
engine P7 used; its branch does not change ``engine/``). A replay that does not end on
the logged score / max tile / board is an error. For each move i we record the board
BEFORE move i (after the previous spawn, i.e. a real state, not an afterstate) when its
max tile is >= --min-tile. The terminal board (no move follows) is not recorded.

Inputs: every ``*.jsonl`` (eval.run format, one game per line) and ``*.json`` (one game
with ``seed`` and ``moves``) under RESULTS_DIR, read in sorted path order.

Duplicates: many logs share games or prefixes (e.g. P7's depth-4 runs continue a
depth-3 game from its switch move). A position is identified by (source_seed,
move_index, board); it is kept only in the first game (in path order) that has it, so
a fully duplicated game contributes nothing and does not get double weight when
training samples source games uniformly.

Held-out seed: every position from source seed --exclude-seed (default 128) goes to
--heldout-out only, never to --out. Each output line:
    {"board": 4x4, "max_tile", "source_seed", "move_index", "source_file", "game_id"}
``game_id`` numbers the source games that contributed positions (0, 1, ...).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.game import Game  # noqa: E402


def iter_games(root: Path) -> Iterator[Tuple[str, int, dict]]:
    """Yield (relative file path, record index in file, record) for every game."""
    paths = sorted(p for p in root.rglob("*") if p.suffix in (".json", ".jsonl") and p.is_file())
    for p in paths:
        rel = p.relative_to(root).as_posix()
        if p.suffix == ".jsonl":
            with p.open(encoding="utf-8") as f:
                for i, line in enumerate(l for l in f if l.strip()):
                    yield rel, i, json.loads(line)
        else:
            rec = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(rec, dict) and "seed" in rec and "moves" in rec:
                yield rel, 0, rec


def replay_late(rec: dict, min_tile: int) -> List[Tuple[int, List[List[int]]]]:
    """Replay one game; return [(move_index, board before that move)] with max tile >= min_tile."""
    g = Game(int(rec["seed"]))
    out = []
    for i, c in enumerate(rec["moves"]):
        if g.max_tile() >= min_tile:
            out.append((i, [row[:] for row in g.board]))
        before = len(g.moves)
        g.step(int(c))
        if len(g.moves) == before:
            raise ValueError(f"seed {rec['seed']}: move {i} ({c}) is illegal in replay")
    if "score" in rec and g.score != rec["score"]:
        raise ValueError(f"seed {rec['seed']}: replay score {g.score} != logged {rec['score']}")
    if "max_tile" in rec and g.max_tile() != rec["max_tile"]:
        raise ValueError(f"seed {rec['seed']}: replay max tile {g.max_tile()} != logged {rec['max_tile']}")
    if "board" in rec and g.board != rec["board"]:
        raise ValueError(f"seed {rec['seed']}: replay final board differs from the logged one")
    return out


def extract(root: Path, min_tile: int, exclude_seed: int) -> Dict:
    seen = set()
    kept: List[dict] = []
    held: List[dict] = []
    stats = Counter()
    game_ids = {"kept": 0, "held": 0}
    for rel, idx, rec in iter_games(root):
        stats["games_read"] += 1
        seed = int(rec["seed"])
        late = replay_late(rec, min_tile)
        if not late:
            continue
        stats["games_late"] += 1
        new = []
        for i, board in late:
            key = (seed, i, tuple(v for row in board for v in row))
            if key in seen:
                stats["dup_positions"] += 1
                continue
            seen.add(key)
            new.append((i, board))
        if not new:
            stats["dup_games"] += 1
            continue
        bucket, which = (held, "held") if seed == exclude_seed else (kept, "kept")
        gid = game_ids[which]
        game_ids[which] += 1
        src = f"{rel}#{idx}"
        for i, board in new:
            bucket.append({
                "board": board, "max_tile": max(max(r) for r in board), "source_seed": seed,
                "move_index": i, "source_file": src, "game_id": gid,
            })
    return {"kept": kept, "held": held, "stats": stats, "n_kept_games": game_ids["kept"],
            "n_held_games": game_ids["held"]}


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path, help="folder of logged games (searched recursively)")
    p.add_argument("--out", type=Path, required=True, help="training positions (JSONL)")
    p.add_argument("--heldout-out", type=Path, required=True, help="excluded-seed positions, eval only (JSONL)")
    p.add_argument("--min-tile", type=int, default=16384)
    p.add_argument("--exclude-seed", type=int, default=128)
    a = p.parse_args(argv)
    r = extract(a.root, a.min_tile, a.exclude_seed)
    assert all(x["source_seed"] != a.exclude_seed for x in r["kept"])
    write_jsonl(a.out, r["kept"])
    write_jsonl(a.heldout_out, r["held"])
    s = r["stats"]
    kept_seeds = sorted({x["source_seed"] for x in r["kept"]})
    print(f"games read: {s['games_read']}, with a board >= {a.min_tile}: {s['games_late']}, "
          f"fully duplicated (dropped): {s['dup_games']}, duplicate positions skipped: {s['dup_positions']}")
    print(f"KEPT for training: {len(r['kept'])} positions from {r['n_kept_games']} source games "
          f"({len(kept_seeds)} seeds) -> {a.out}")
    print(f"EXCLUDED (seed {a.exclude_seed}): {len(r['held'])} positions from {r['n_held_games']} source games "
          f"-> {a.heldout_out} (evaluation only)")
    tiles = Counter(x["max_tile"] for x in r["kept"])
    print("kept positions by max tile:", dict(sorted(tiles.items())))


if __name__ == "__main__":
    main()
