// engine/_core/selftest.cpp -- standalone checks for the C++ core (no Python,
// no pybind11, no test framework). Exit code 0 = all passed.
//
//   selftest                 built-in checks against an independent reference
//   selftest golden.txt      also compare with results produced by the real
//                            Python engine (see gen_golden.py)
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "core.hpp"

using namespace p1;

static int g_failures = 0;
#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            ++g_failures;                                 \
            if (g_failures <= 20) {                       \
                std::printf("FAIL line %d: ", __LINE__);  \
                std::printf(__VA_ARGS__);                 \
                std::printf("\n");                        \
            }                                             \
        }                                                 \
    } while (0)

typedef uint64_t Grid[4][4];  // tile VALUES (0, 2, 4, ...), like the Python board

// ---- independent reference: the same algorithm as _get_lines/_merge_line/_set_lines
static void ref_merge_line(const uint64_t in[4], uint64_t out[4], uint64_t* gained) {
    uint64_t tiles[4];
    int n = 0;
    for (int i = 0; i < 4; ++i)
        if (in[i]) tiles[n++] = in[i];
    for (int i = 0; i < 4; ++i) out[i] = 0;
    int w = 0;
    for (int i = 0; i < n;) {
        if (i + 1 < n && tiles[i] == tiles[i + 1]) {
            out[w++] = tiles[i] * 2;
            *gained += tiles[i] * 2;
            i += 2;
        } else {
            out[w++] = tiles[i++];
        }
    }
}

static uint64_t ref_slide(const Grid b, int move_, Grid out) {
    uint64_t gained = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t line[4], res[4];
        for (int j = 0; j < 4; ++j) {  // line[j] = j-th cell counted from the wall tiles move toward
            if (move_ == LEFT) line[j] = b[i][j];
            else if (move_ == RIGHT) line[j] = b[i][3 - j];
            else if (move_ == UP) line[j] = b[j][i];
            else line[j] = b[3 - j][i];
        }
        ref_merge_line(line, res, &gained);
        for (int j = 0; j < 4; ++j) {
            if (move_ == LEFT) out[i][j] = res[j];
            else if (move_ == RIGHT) out[i][3 - j] = res[j];
            else if (move_ == UP) out[j][i] = res[j];
            else out[3 - j][i] = res[j];
        }
    }
    return gained;
}

static int exp_of(uint64_t v) {
    int e = 0;
    while (v > 1) { v >>= 1; ++e; }
    return e;
}

static Board board_from_grid(const Grid g) {
    int8_t e[16];
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) e[r * 4 + c] = g[r][c] ? static_cast<int8_t>(exp_of(g[r][c])) : 0;
    return from_exponents(e);
}

static void grid_from_board(const Board& b, Grid g) {
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            const int e = get_cell(b, r, c);
            g[r][c] = e ? (uint64_t(1) << e) : 0;
        }
}

static bool grids_equal(const Grid a, const Grid b) {
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            if (a[r][c] != b[r][c]) return false;
    return true;
}

static void compare_move(const Grid g, int m, const char* what) {
    Grid want;
    const uint64_t want_gain = ref_slide(g, m, want);
    const MoveResult got = move(board_from_grid(g), m);
    Grid got_grid;
    grid_from_board(got.board, got_grid);
    CHECK(grids_equal(want, got_grid) && got.gained == want_gain, "%s: move %d differs from reference", what, m);
    Grid copy;
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) copy[r][c] = g[r][c];
    CHECK(is_legal(board_from_grid(g), m) == !grids_equal(want, copy), "%s: is_legal %d", what, m);
}

static void test_every_row() {
    // All 19^4 rows with exponents 0..18, in all four directions, placed in
    // every row/column position so transposition is exercised too.
    for (int a = 0; a <= kMaxSupportedExp; ++a)
        for (int b = 0; b <= kMaxSupportedExp; ++b)
            for (int c = 0; c <= kMaxSupportedExp; ++c)
                for (int d = 0; d <= kMaxSupportedExp; ++d) {
                    const int e[4] = {a, b, c, d};
                    Grid g = {{0}};
                    for (int j = 0; j < 4; ++j) g[1][j] = e[j] ? (uint64_t(1) << e[j]) : 0;
                    for (int m = 0; m < 4; ++m) compare_move(g, m, "row");
                    Grid h = {{0}};
                    for (int j = 0; j < 4; ++j) h[j][2] = e[j] ? (uint64_t(1) << e[j]) : 0;
                    for (int m = 0; m < 4; ++m) compare_move(h, m, "column");
                }
}

static uint64_t g_state = 88172645463325252ULL;  // test-only xorshift, NOT the game RNG
static uint64_t rnd() {
    g_state ^= g_state << 13;
    g_state ^= g_state >> 7;
    g_state ^= g_state << 17;
    return g_state;
}

static void test_random_boards() {
    for (int i = 0; i < 200000; ++i) {
        Grid g;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) {
                const uint64_t x = rnd() % 100;
                const int e = x < 35 ? 0 : static_cast<int>(rnd() % (kMaxSupportedExp + 1));
                g[r][c] = e ? (uint64_t(1) << e) : 0;
            }
        for (int m = 0; m < 4; ++m) compare_move(g, m, "random");
    }
}

static void test_known_cases() {
    Grid g = {{2, 2, 2, 2}, {0}, {0}, {0}};
    MoveResult r = move(board_from_grid(g), LEFT);
    CHECK(get_cell(r.board, 0, 0) == 2 && get_cell(r.board, 0, 1) == 2 && get_cell(r.board, 0, 2) == 0 && r.gained == 8,
          "[2,2,2,2] left");
    for (int m = 0; m < 4; ++m) {  // 32,768 + 32,768 -> 65,536 in every direction
        Grid h = {{0}};
        const int pr[4][2][2] = {{{0, 0}, {1, 0}}, {{0, 3}, {0, 2}}, {{3, 0}, {2, 0}}, {{0, 0}, {0, 1}}};  // UP,RIGHT,DOWN,LEFT
        h[pr[m][0][0]][pr[m][0][1]] = 32768;
        h[pr[m][1][0]][pr[m][1][1]] = 32768;
        MoveResult x = move(board_from_grid(h), m);
        CHECK(max_exponent(x.board) == 16 && x.gained == 65536, "32768+32768 dir %d", m);
    }
    Grid big = {{65536, 65536, 0, 0}, {0}, {0}, {0}};
    MoveResult b = move(board_from_grid(big), LEFT);
    CHECK(get_cell(b.board, 0, 0) == 17 && b.gained == 131072, "65536+65536");
    Grid ok18 = {{262144, 0, 0, 0}, {0}, {0}, {0}};  // exponent 18: supported input
    Grid bad = {{524288, 0, 0, 0}, {0}, {0}, {0}};   // exponent 19: first unsupported exponent
    CHECK(is_supported(board_from_grid(ok18)) && is_supported(board_from_grid(big)) &&
              !is_supported(board_from_grid(bad)),
          "is_supported");

    Grid full = {{2, 4, 2, 4}, {4, 2, 4, 2}, {2, 4, 2, 4}, {4, 2, 4, 2}};
    CHECK(legal_mask(board_from_grid(full)) == 0 && empty_count(board_from_grid(full)) == 0, "dead board");
    Grid top = {{2, 4, 8, 16}, {0}, {0}, {0}};
    CHECK(!is_legal(board_from_grid(top), UP) && is_legal(board_from_grid(top), DOWN), "top row cannot move up");
    CHECK(empty_count(board_from_grid(top)) == 12 && max_exponent(board_from_grid(top)) == 4, "counts");

    bool threw = false;
    try { move(board_from_grid(top), 4); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "move(4) must throw");
    threw = false;
    try { move(board_from_grid(top), -1); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "move(-1) must throw");
}


// Where the two tiles of a pair sit so that `m` merges them (touching the wall they move toward).
static void pair_grid(int m, uint64_t value, Grid g) {
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) g[r][c] = 0;
    const int pr[4][2][2] = {{{0, 0}, {1, 0}}, {{0, 3}, {0, 2}}, {{3, 0}, {2, 0}}, {{0, 0}, {0, 1}}};  // UP,RIGHT,DOWN,LEFT
    g[pr[m][0][0]][pr[m][0][1]] = value;
    g[pr[m][1][0]][pr[m][1][1]] = value;
}

static void test_large_tiles() {
    // 1. 65,536 + 65,536 -> 131,072 and 2. 131,072 + 131,072 -> 262,144, all four directions.
    const uint64_t vals[2] = {65536, 131072};
    for (int k = 0; k < 2; ++k)
        for (int m = 0; m < 4; ++m) {
            Grid g;
            pair_grid(m, vals[k], g);
            compare_move(g, m, "pair merge");  // 4. board + score + legality vs independent reference
            const MoveResult r = move(board_from_grid(g), m);
            CHECK(max_exponent(r.board) == exp_of(vals[k] * 2) && r.gained == vals[k] * 2, "pair %llu dir %d",
                  static_cast<unsigned long long>(vals[k]), m);
            CHECK(empty_count(r.board) == 15, "pair merge leaves one tile");
        }

    // 3. A 262,144 tile (exponent 18) as INPUT: every cell, every direction.
    for (int pos = 0; pos < 16; ++pos) {
        Grid g = {{0}};
        g[pos / 4][pos % 4] = 262144;
        for (int m = 0; m < 4; ++m) {
            compare_move(g, m, "262144 single");
            const MoveResult r = move(board_from_grid(g), m);
            CHECK(max_exponent(r.board) == 18 && r.gained == 0 && empty_count(r.board) == 15,
                  "262144 must survive a move unchanged in value (pos %d dir %d)", pos, m);
        }
    }
    // ...and with neighbours that do and do not merge with it.
    Grid mix = {{262144, 262144, 131072, 131072}, {131072, 131072, 262144, 0}, {65536, 65536, 262144, 262144}, {0}};
    for (int m = 0; m < 4; ++m) compare_move(mix, m, "262144 mixed");

    // The exact edge case: the chain 131,072 x2 -> 262,144, then ANOTHER move on that exponent-18 board.
    Grid chain = {{131072, 131072, 262144, 0}, {0}, {0}, {0}};
    const MoveResult first = move(board_from_grid(chain), LEFT);
    CHECK(get_cell(first.board, 0, 0) == 18 && get_cell(first.board, 0, 1) == 18 && first.gained == 262144,
          "chain step 1");
    CHECK(is_supported(first.board), "exponent-18 board must be a supported input");
    Grid after;
    grid_from_board(first.board, after);
    compare_move(after, LEFT, "chain step 2 (input has exponent 18)");
    const MoveResult second = move(first.board, LEFT);
    CHECK(get_cell(second.board, 0, 0) == 19 && second.gained == 524288, "262144+262144 -> 524288");

    // First unsupported exponent (19): behaviour is pinned, never an overflow. Row unchanged, score 0.
    Grid beyond = {{524288, 524288, 0, 0}, {0}, {0}, {0}};
    const MoveResult stuck = move(board_from_grid(beyond), LEFT);
    CHECK(!is_supported(board_from_grid(beyond)) && stuck.board == board_from_grid(beyond) && stuck.gained == 0,
          "exponent-19 board is unsupported: unchanged, score 0 (callers must check is_supported)");
}

static void test_roundtrip() {
    for (int i = 0; i < 10000; ++i) {
        int8_t e[16], back[16];
        for (int k = 0; k < 16; ++k) e[k] = static_cast<int8_t>(rnd() % 32);
        to_exponents(from_exponents(e), back);
        for (int k = 0; k < 16; ++k) CHECK(e[k] == back[k], "roundtrip");
    }
}

static void test_golden(const char* path) {
    std::ifstream in(path);
    CHECK(in.good(), "cannot open golden file %s", path);
    long lines = 0;
    for (;;) {
        Grid g, want;
        int m;
        uint64_t want_gain;
        bool ok = true;
        for (int r = 0; r < 4 && ok; ++r)
            for (int c = 0; c < 4 && ok; ++c) ok = static_cast<bool>(in >> g[r][c]);
        if (!ok || !(in >> m)) break;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) in >> want[r][c];
        in >> want_gain;
        const MoveResult got = move(board_from_grid(g), m);
        Grid got_grid;
        grid_from_board(got.board, got_grid);
        CHECK(grids_equal(want, got_grid) && got.gained == want_gain, "golden line %ld differs from Python engine", lines);
        ++lines;
    }
    std::printf("golden: compared %ld Python results\n", lines);
    CHECK(lines > 0, "golden file was empty");
}

int main(int argc, char** argv) {
    init_tables();
    test_known_cases();
    test_large_tiles();
    test_roundtrip();
    test_every_row();
    test_random_boards();
    if (argc > 1) test_golden(argv[1]);
    if (g_failures == 0) {
        std::printf("ALL C++ CORE CHECKS PASSED\n");
        return 0;
    }
    std::printf("%d CHECK(S) FAILED\n", g_failures);
    return 1;
}
