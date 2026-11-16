// engine/_core/batch.hpp  --  P1 Batch 3c: whole random games played in C++.
//
// One game here is exactly one game of the Python engine (engine/game.py) played
// by the "uniformly random legal move" policy defined below. Nothing in this
// file touches Python, so it is fast, thread-safe, and testable on its own.
//
// THE GAME (mirrors Game.reset / Game.step line by line):
//   start : empty board + two spawns, or a caller-supplied board (no spawn)
//   spawn : empties in row-major order; cell = rng.randbelow(#empties), THEN
//           value = 2 if rng.random() < 0.9 else 4   (same RNG order as Python)
//   step  : slide, add the merge score, spawn. Over when no move is legal.
//
// THE POLICY. A second generator, seeded with policy_seed(seed), picks the
// move: among the legal moves in ascending order (up, right, down, left) it
// takes legal[policy.randbelow(len(legal))]. It is separate from the game RNG
// so choosing moves never shifts the spawn sequence, which keeps
// "same seed => same spawns for every agent" true. engine/batch.py contains the
// same policy written in plain Python; tests prove both play identical games.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core.hpp"

namespace p1 {

// seed -> policy RNG seed. XOR with a fixed odd constant so the policy stream
// differs from the spawn stream of the same game (identical seeds would make
// the move choices correlated with the spawns). Mirrored in engine/batch.py.
constexpr uint64_t kPolicySeedMask = 0x2545F4914F6CDD1DULL;
inline uint64_t policy_seed(uint64_t abs_seed) { return abs_seed ^ kPolicySeedMask; }

struct GameResult {
    uint64_t score = 0;
    uint32_t n_moves = 0;
    int max_exp = 0;            // exponent of the largest tile (0 if none)
    Board board{};              // final position
    std::vector<uint8_t> moves; // 0..3 per move; empty unless recorded
    bool bailed = false;        // true: a board left the supported range, redo in Python
};

// Play one game. `abs_seed` = abs(seed) of the Python int seed. `start` null =
// a normal game, else begin from that board (is_supported(*start) required).
GameResult play_random_game(uint64_t abs_seed, const Board* start, bool record_moves);

// Play n games, results[i] for seeds[i], using `threads` worker threads
// (0 = hardware concurrency). Results never depend on the thread count: each
// game is independent and written to its own slot.
void play_random_batch(const uint64_t* abs_seeds, size_t n, const Board* start, bool record_moves,
                       int threads, GameResult* results);

}  // namespace p1
