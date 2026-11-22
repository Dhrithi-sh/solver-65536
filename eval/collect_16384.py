"""Collect the first playable 16,384 board from each saved replay.

Run:
    python -m eval.collect_16384 results/hybrid-d3.jsonl

Output:
    results/boards_16384.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.game import Game


TARGET_TILE = 16384


def collect_record(record: dict, source_file: str) -> tuple[dict | None, str]:
    seed = record["seed"]
    moves = record["moves"]

    game = Game(seed)
    saved_board = None
    reached_but_unplayable = False

    for move_number, move_char in enumerate(moves, start=1):
        move = int(move_char)

        if move not in game.legal_moves():
            raise ValueError(
                f"Seed {seed}: move {move_number} ({move}) is illegal"
            )

        # step() performs the move and then spawns the new tile.
        game.step(move)

        if saved_board is None and game.max_tile() >= TARGET_TILE:
            if not game.done:
                saved_board = {
                    "agent": record.get("agent", "unknown"),
                    "seed": seed,
                    "move_number": move_number,
                    "score": game.score,
                    "max_tile": game.max_tile(),
                    "threshold": TARGET_TILE,
                    "board": [row[:] for row in game.board],
                    "source_file": source_file,
                    "source_final_moves": record.get("n_moves", len(moves)),
                    "playable_after_spawn": True,
                }
            else:
                reached_but_unplayable = True

    # Check that replaying the moves reproduces the logged final result.
    expected_moves = record.get("n_moves", len(moves))
    if (
        game.score != record["score"]
        or game.max_tile() != record["max_tile"]
        or len(game.moves) != expected_moves
    ):
        raise ValueError(
            f"Seed {seed}: replay mismatch. "
            f"score {game.score}/{record['score']}, "
            f"max tile {game.max_tile()}/{record['max_tile']}, "
            f"moves {len(game.moves)}/{expected_moves}"
        )

    if saved_board is not None:
        return saved_board, "saved"
    if reached_but_unplayable:
        return None, "reached_16384_but_game_over"
    if game.max_tile() < TARGET_TILE:
        return None, "never_reached_16384"
    return None, "no_playable_16384_board"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_file", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/boards_16384.jsonl"),
    )
    args = parser.parse_args()

    if not args.replay_file.is_file():
        raise SystemExit(f"Replay file not found: {args.replay_file}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.replay_file.open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]

    saved = 0
    print(f"Reading {len(records)} replay(s) from {args.replay_file}")

    with args.out.open("w", encoding="utf-8") as output:
        for record in records:
            board_record, status = collect_record(
                record, args.replay_file.name
            )
            if board_record is not None:
                output.write(json.dumps(board_record) + "\n")
                saved += 1
                print(
                    f"Saved seed {board_record['seed']}: "
                    f"move {board_record['move_number']}, "
                    f"max tile {board_record['max_tile']}"
                )
            else:
                print(f"Seed {record['seed']}: {status}")

    print(f"\nSaved {saved} playable board(s) to {args.out}")


if __name__ == "__main__":
    main()
