// engine/_core/core.cpp  --  implementation, see core.hpp for the layout.
#include "core.hpp"

#include <stdexcept>
#include <vector>

namespace p1 {
namespace {

constexpr uint32_t kRowKeys = 1u << (kCellBits * kSize);  // 2^20 possible rows

struct RowEntry {
    uint32_t row;    // row after the move (20-bit key)
    uint32_t score;  // score gained by that row's merges
};

struct Tables {
    std::vector<RowEntry> left;
    std::vector<RowEntry> right;
};

inline uint32_t cell_of(uint32_t row, int c) { return (row >> (kCellBits * c)) & kCellMask; }

uint32_t pack(const uint32_t e[kSize]) {
    uint32_t key = 0;
    for (int i = 0; i < kSize; ++i) key |= e[i] << (kCellBits * i);
    return key;
}

uint32_t reverse_row(uint32_t row) {
    uint32_t e[kSize];
    for (int i = 0; i < kSize; ++i) e[i] = cell_of(row, kSize - 1 - i);
    return pack(e);
}

// Left-slide one row: compact tiles toward cell 0, merge equal neighbours once
// (a freshly merged tile never merges again in the same move).
RowEntry slide_left(uint32_t key) {
    uint32_t e[kSize];
    bool supported = true;
    for (int i = 0; i < kSize; ++i) {
        e[i] = cell_of(key, i);
        if (e[i] > static_cast<uint32_t>(kMaxSupportedExp)) supported = false;
    }
    // Unsupported rows (exponent > 18) are never produced by real play. Keep
    // them well defined (no overflow): unchanged row, score 0.
    if (!supported) return RowEntry{key, 0};

    uint32_t tiles[kSize];
    int n = 0;
    for (int i = 0; i < kSize; ++i)
        if (e[i]) tiles[n++] = e[i];

    uint32_t out[kSize] = {0, 0, 0, 0};
    int w = 0;
    uint32_t score = 0;
    for (int i = 0; i < n;) {
        if (i + 1 < n && tiles[i] == tiles[i + 1]) {
            out[w++] = tiles[i] + 1;
            score += 1u << (tiles[i] + 1);  // value of the new tile
            i += 2;
        } else {
            out[w++] = tiles[i];
            i += 1;
        }
    }
    return RowEntry{pack(out), score};
}

const Tables& tables() {
    // C++11 guarantees this runs once, even with several threads.
    static const Tables t = [] {
        Tables x;
        x.left.resize(kRowKeys);
        x.right.resize(kRowKeys);
        for (uint32_t key = 0; key < kRowKeys; ++key) x.left[key] = slide_left(key);
        // Right move = reverse the row, slide left, reverse the result.
        for (uint32_t key = 0; key < kRowKeys; ++key) {
            const RowEntry l = x.left[reverse_row(key)];
            x.right[key] = RowEntry{reverse_row(l.row), l.score};
        }
        return x;
    }();
    return t;
}

// Rows become columns and vice versa. Used for UP / DOWN.
Board transpose(const Board& b) {
    Board t;
    for (int c = 0; c < kSize; ++c) {
        uint32_t v = 0;
        for (int r = 0; r < kSize; ++r) v |= cell_of(b.row[r], c) << (kCellBits * r);
        t.row[c] = v;
    }
    return t;
}

MoveResult slide_rows(const Board& b, const std::vector<RowEntry>& table) {
    MoveResult res;
    res.gained = 0;
    for (int r = 0; r < kSize; ++r) {
        const RowEntry& e = table[b.row[r]];
        res.board.row[r] = e.row;
        res.gained += e.score;
    }
    return res;
}

}  // namespace

void init_tables() { (void)tables(); }

Board from_exponents(const int8_t* exps16) {
    Board b;
    for (int r = 0; r < kSize; ++r) {
        uint32_t e[kSize];
        for (int c = 0; c < kSize; ++c) e[c] = static_cast<uint32_t>(exps16[r * kSize + c]) & kCellMask;
        b.row[r] = pack(e);
    }
    return b;
}

void to_exponents(const Board& b, int8_t* exps16) {
    for (int r = 0; r < kSize; ++r)
        for (int c = 0; c < kSize; ++c) exps16[r * kSize + c] = static_cast<int8_t>(cell_of(b.row[r], c));
}

int get_cell(const Board& b, int r, int c) { return static_cast<int>(cell_of(b.row[r], c)); }

MoveResult move(const Board& b, int dir) {
    const Tables& t = tables();
    switch (dir) {
        case LEFT:
            return slide_rows(b, t.left);
        case RIGHT:
            return slide_rows(b, t.right);
        case UP: {
            MoveResult m = slide_rows(transpose(b), t.left);
            m.board = transpose(m.board);
            return m;
        }
        case DOWN: {
            MoveResult m = slide_rows(transpose(b), t.right);
            m.board = transpose(m.board);
            return m;
        }
        default:
            throw std::invalid_argument("invalid move; expected 0..3");
    }
}

bool is_supported(const Board& b) { return max_exponent(b) <= kMaxSupportedExp; }

bool is_legal(const Board& b, int dir) { return move(b, dir).board != b; }

int legal_mask(const Board& b) {
    int mask = 0;
    for (int m = 0; m < 4; ++m)
        if (is_legal(b, m)) mask |= 1 << m;
    return mask;
}

int empty_count(const Board& b) {
    int n = 0;
    for (int r = 0; r < kSize; ++r)
        for (int c = 0; c < kSize; ++c)
            if (cell_of(b.row[r], c) == 0) ++n;
    return n;
}

int max_exponent(const Board& b) {
    int best = 0;
    for (int r = 0; r < kSize; ++r)
        for (int c = 0; c < kSize; ++c) {
            const int v = static_cast<int>(cell_of(b.row[r], c));
            if (v > best) best = v;
        }
    return best;
}

}  // namespace p1
