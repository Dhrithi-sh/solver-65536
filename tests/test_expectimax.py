import pytest
from engine.game import Game
from agents.expectimax.search import ExpectimaxAgent


class TopLeft:                     
    def evaluate(self, board):
        return float(board[0][0])


# Only left/right are legal (row 0 merges). Left keeps 4 in the corner, right doesn't.
BOARD = [[2, 2, 4, 8], [16, 32, 64, 128],
         [256, 512, 1024, 2048], [4096, 8192, 16384, 32768]]


def test_picks_obvious_move():
    g = Game(seed=0, board=BOARD)
    assert ExpectimaxAgent(TopLeft(), depth=1).choose_move(g) == 3   # 3 = left


def test_depth1_matches_brute_force():
    vals = ExpectimaxAgent(TopLeft(), depth=1, cutoff=0).move_values(BOARD)
    for move, v in vals.items():
        after, _ = Game.slide(BOARD, move)
        cells = [(r, c) for r in range(4) for c in range(4) if after[r][c] == 0]
        expected = 0.0
        for r, c in cells:
            for tile, p in ((2, 0.9), (4, 0.1)):
                b = [list(row) for row in after]
                b[r][c] = tile
                expected += p * b[0][0] / len(cells)
        assert v == pytest.approx(expected)


def test_transposition_table_does_not_change_values():
    a = ExpectimaxAgent(TopLeft(), depth=2, cutoff=0, use_tt=True)
    b = ExpectimaxAgent(TopLeft(), depth=2, cutoff=0, use_tt=False)
    assert a.move_values(BOARD) == pytest.approx(b.move_values(BOARD))

def test_adaptive_depth_goes_deeper_when_crowded():
    crowded = [[2, 4, 8, 16], [32, 64, 128, 256],
               [512, 1024, 2048, 4096], [8192, 0, 0, 0]]
    open_ = [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    a = ExpectimaxAgent(TopLeft(), depth=2, adaptive=True)
    assert a._pick_depth(crowded) > 2
    assert a._pick_depth(open_) == 2
    assert ExpectimaxAgent(TopLeft(), depth=2)._pick_depth(crowded) == 2