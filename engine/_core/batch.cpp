// engine/_core/batch.cpp  --  see batch.hpp for the rules this must follow.
#include "batch.hpp"

#include <algorithm>
#include <atomic>
#include <thread>

#include "rng.hpp"

namespace p1 {
namespace {

// Same as Game._spawn: pick among empty cells (row-major) then the value.
// Returns false (and changes nothing) on a full board, like Python.
bool spawn(Board& b, PyRandom& rng) {
    uint8_t cells[16];  // flat index r*4 + c of each empty cell, row-major
    uint32_t n = 0;
    for (int r = 0; r < kSize; ++r)
        for (int c = 0; c < kSize; ++c)
            if (get_cell(b, r, c) == 0) cells[n++] = static_cast<uint8_t>(r * kSize + c);
    if (n == 0) return false;
    const uint8_t cell = cells[rng.randbelow(n)];  // 1st draw: which cell
    const uint32_t value_exp = rng.random() < 0.9 ? 1u : 2u;  // 2nd draw: 2 (p=.9) or 4
    b.row[cell / kSize] |= value_exp << (kCellBits * (cell % kSize));
    return true;
}

}  // namespace

GameResult play_random_game(uint64_t abs_seed, const Board* start, bool record_moves) {
    init_tables();
    PyRandom rng = PyRandom::from_seed(abs_seed);
    PyRandom policy = PyRandom::from_seed(policy_seed(abs_seed));

    GameResult res;
    Board b{};
    if (start) {
        b = *start;
    } else {
        spawn(b, rng);
        spawn(b, rng);
    }
    // Normal games can never leave the supported range (4x4 caps at 131,072).
    // A hand-made start board might (many 2^16 tiles can chain into 2^19), so
    // for those we check each move and hand the game back to Python if needed.
    const bool watch = start != nullptr;

    for (;;) {
        // Slide all four ways once; a move is legal iff it changes the board.
        MoveResult next[4];
        uint8_t legal[4];
        int n_legal = 0;
        for (int m = 0; m < 4; ++m) {
            next[m] = move(b, m);
            if (next[m].board != b) legal[n_legal++] = static_cast<uint8_t>(m);
        }
        if (n_legal == 0) break;  // game over

        const uint8_t m = legal[policy.randbelow(static_cast<uint32_t>(n_legal))];
        b = next[m].board;
        res.score += next[m].gained;
        ++res.n_moves;
        if (record_moves) res.moves.push_back(m);
        spawn(b, rng);
        if (watch && !is_supported(b)) {
            res.bailed = true;
            break;
        }
    }
    res.board = b;
    res.max_exp = max_exponent(b);
    return res;
}

void play_random_batch(const uint64_t* abs_seeds, size_t n, const Board* start, bool record_moves,
                       int threads, GameResult* results) {
    init_tables();  // build once before any worker starts
    size_t workers = threads > 0 ? static_cast<size_t>(threads) : std::thread::hardware_concurrency();
    workers = std::max<size_t>(1, std::min(workers, n));
    std::atomic<size_t> next_index{0};
    auto work = [&] {
        for (size_t i; (i = next_index.fetch_add(1)) < n;)
            results[i] = play_random_game(abs_seeds[i], start, record_moves);
    };
    if (workers == 1) {
        work();
        return;
    }
    std::vector<std::thread> pool;
    for (size_t w = 1; w < workers; ++w) pool.emplace_back(work);
    work();  // the calling thread helps too
    for (auto& t : pool) t.join();
}

}  // namespace p1
