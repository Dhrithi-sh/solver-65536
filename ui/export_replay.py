import copy
import json
import sys
from pathlib import Path

from engine.game import Game


def find_merged_tiles(board, move):
    merged = []

    # Arrange coordinates in the direction tiles move.
    if move == 0:  # UP
        lines = [[(r, c) for r in range(4)] for c in range(4)]
    elif move == 1:  # RIGHT
        lines = [[(r, c) for c in range(3, -1, -1)] for r in range(4)]
    elif move == 2:  # DOWN
        lines = [[(r, c) for r in range(3, -1, -1)] for c in range(4)]
    else:  # LEFT
        lines = [[(r, c) for c in range(4)] for r in range(4)]

    for line in lines:
        tiles = [
            (board[r][c], (r, c))
            for r, c in line
            if board[r][c] != 0
        ]

        i = 0
        output_index = 0

        while i < len(tiles):
            if (
                i + 1 < len(tiles)
                and tiles[i][0] == tiles[i + 1][0]
            ):
                # Two equal tiles merge at this destination.
                merged.append(list(line[output_index]))
                i += 2
            else:
                i += 1

            output_index += 1

    return merged


def export_replay(input_file, index=0):
    input_path = Path(input_file)

    with input_path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    record = records[index]
    game = Game(record["seed"], board=record.get("start_board"))

    frames = [{
        "board": copy.deepcopy(game.board),
        "score": game.score,
        "max_tile": game.max_tile(),
        "move": "Start",
        "merged": []
    }]

    for move_char in record["moves"]:
        move = int(move_char)

        merged = find_merged_tiles(game.board, move)
        game.step(move)

        frames.append({
            "board": copy.deepcopy(game.board),
            "score": game.score,
            "max_tile": game.max_tile(),
            "move": ["Up", "Right", "Down", "Left"][move],
            "merged": merged
        })

    if game.score != record["score"] or game.max_tile() != record["max_tile"]:
        raise ValueError("Replay does not match the original game!")

    output = {
        "agent": record["agent"],
        "seed": record["seed"],
        "frames": frames
    }

    output_path = Path("ui/replay_data.json")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f)

    print(f"Replay exported successfully: {output_path}")
    print(f"Agent: {record['agent']}")
    print(f"Seed: {record['seed']}")
    print(f"Total frames: {len(frames)}")


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "results/greedy.jsonl"
    index = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    export_replay(input_file, index)