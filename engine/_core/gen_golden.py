"""Write test vectors made by the REAL Python engine (engine/game.py) for the
C++ self-test. Usage (from the repo root):

    python engine/_core/gen_golden.py golden.txt [n_boards]

Each line: 16 tile values (row-major), move, 16 resulting tile values, score
gained. The file is a build artefact; do not commit it.
"""
import random
import sys

sys.path.insert(0, ".")
from engine.game import Game  # noqa: E402


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "golden.txt"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    rng = random.Random(2024)
    tiles = [0, 0, 0, 0, 2, 4, 8, 16, 64, 256, 1024, 4096, 16384, 32768, 65536, 131072, 262144]
    with open(path, "w") as f:
        for _ in range(n):
            board = [[rng.choice(tiles) for _ in range(4)] for _ in range(4)]
            for move in range(4):
                new, gained = Game.slide(board, move)
                flat_in = [v for row in board for v in row]
                flat_out = [v for row in new for v in row]
                f.write(" ".join(map(str, flat_in + [move] + flat_out + [gained])) + "\n")
        # hand-picked: the 65,536 merge and double-merge cases
        for row in ([32768, 32768, 0, 0], [2, 2, 2, 2], [4, 2, 2, 0], [65536, 65536, 0, 0],
                    [131072, 131072, 262144, 0], [262144, 0, 0, 262144], [262144, 262144, 0, 0]):
            board = [list(row), [0] * 4, [0] * 4, [0] * 4]
            for move in range(4):
                new, gained = Game.slide(board, move)
                flat_in = [v for r in board for v in r]
                flat_out = [v for r in new for v in r]
                f.write(" ".join(map(str, flat_in + [move] + flat_out + [gained])) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
