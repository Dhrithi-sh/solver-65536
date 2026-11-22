// engine/_core/core.hpp  --  P1 Batch 3a: C++ 2048 core (no Python, no pybind11).
//
// Board layout (mirrors engine/game.py from Batch 1):
//   * a board is 4 rows; each row is a 20-bit number holding 4 cells x 5 bits
//   * a cell stores the EXPONENT: 0 = empty, k = tile 2^k (so 65,536 = 16)
//   * cell c of a row sits at bits [5c, 5c+4]  (cell 0 = lowest bits), exactly
//     like pack_row() in engine/game.py
// We use four uint32_t rows instead of __uint128_t because MSVC (Windows) has
// no 128-bit integer type.
//
// Moves: 0 = up, 1 = right, 2 = down, 3 = left (same as the Python engine).
// Spawning/RNG is NOT here: the Python engine keeps its own random.Random.
#pragma once

#include <cstdint>

namespace p1 {

constexpr int kSize = 4;
constexpr int kCellBits = 5;
constexpr uint32_t kCellMask = 31;
// Boards are "supported" when every exponent is <= 18 (tile 262,144). A move
// then creates at most exponent 19, and a row's score (at most two merges of
// 2^19) is 2^20, far inside 32 bits. Real 4x4 play never gets past 131,072
// (exponent 17), so exponents 18 and 19 only appear on artificial boards
// (e.g. Game.from_board with two 131,072 tiles). Exponent 18 is accepted as
// INPUT so a 262,144 tile can keep moving after it was created. Boards with a
// larger exponent are outside the tables; callers must check is_supported().
constexpr int kMaxSupportedExp = 18;

enum Dir : int { UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3 };

struct Board {
    uint32_t row[kSize];
    bool operator==(const Board& o) const {
        return row[0] == o.row[0] && row[1] == o.row[1] && row[2] == o.row[2] && row[3] == o.row[3];
    }
    bool operator!=(const Board& o) const { return !(*this == o); }
};

struct MoveResult {
    Board board;
    uint32_t gained;  // score gained by the merges of this move
};

// Build the row tables. Safe to call many times / from many threads; the
// other functions call it for you on first use.
void init_tables();

// Conversion: 16 exponents in row-major order (row 0 first).
Board from_exponents(const int8_t* exps16);
void to_exponents(const Board& b, int8_t* exps16);

int get_cell(const Board& b, int r, int c);  // exponent at (r, c)

// Slide the board in direction dir (0..3), no spawn. Throws
// std::invalid_argument for any other dir. Result is only defined when
// is_supported(b) is true.
MoveResult move(const Board& b, int dir);

bool is_supported(const Board& b);
bool is_legal(const Board& b, int dir);   // does the move change the board?
int legal_mask(const Board& b);           // bit m set if move m is legal
int empty_count(const Board& b);
int max_exponent(const Board& b);

}  // namespace p1
