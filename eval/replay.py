"""Replay a logged game from its seed and move list, and check it matches.

    python -m eval.replay results/greedy.jsonl --index 0
    python -m eval.replay results/greedy.jsonl --seed 7 --show

A record may carry an optional "start_board" (4x4 list) for games that began
from a saved position, e.g. the late-game benchmark's 32,768 boards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from engine.game import MOVE_NAMES, Game


def replay(record: dict, show: bool = False) -> Game:
    game = Game(record["seed"], board=record.get("start_board"))
    if show:
        print(f"start\n{game}\n")
    for i, ch in enumerate(record["moves"]):
        move = int(ch)
        if move not in game.legal_moves():
            raise ValueError(f"move {i} ({MOVE_NAMES[move]}) is illegal on replay")
        game.step(move)
        if show:
            print(f"move {i + 1}: {MOVE_NAMES[move]}  score={game.score}\n{game}\n")
    return game


def load_record(path: Path, index: Optional[int], seed: Optional[int]) -> dict:
    with path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if seed is not None:
        for r in records:
            if r["seed"] == seed:
                return r
        raise SystemExit(f"no game with seed {seed} in {path}")
    return records[index or 0]


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", type=Path)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--index", type=int, help="line number in the file (default 0)")
    g.add_argument("--seed", type=int)
    p.add_argument("--show", action="store_true", help="print the board after every move")
    args = p.parse_args(argv)

    record = load_record(args.file, args.index, args.seed)
    game = replay(record, show=args.show)
    ok = game.score == record["score"] and game.max_tile() == record["max_tile"]
    print(f"seed={record['seed']} moves={len(game.moves)} score={game.score} max_tile={game.max_tile()}")
    print(game)
    if not ok:
        raise SystemExit(
            f"MISMATCH: logged score={record['score']} max_tile={record['max_tile']}"
        )
    print("replay matches log")


if __name__ == "__main__":
    main()
