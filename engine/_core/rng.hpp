// engine/_core/rng.hpp  --  P1 Batch 3c: CPython-compatible random.Random.
//
// WHY THIS EXISTS. The whole project promises that a game is reproducible from
// (agent, seed), and the reference engine draws its spawns from
// random.Random(seed). To play games in C++ and still get EXACTLY the same
// games, this file re-implements the pieces of CPython's `random` module that
// the engine uses, bit for bit:
//
//   random.Random(seed)        -> PyRandom::from_seed(|seed|)
//   rng.getrandbits(k)         -> PyRandom::getrandbits(k)        (k <= 32)
//   rng.random()               -> PyRandom::random()
//   rng.randrange(n) (n >= 1)  -> PyRandom::randbelow(n)
//
// Python's generator is the standard 32-bit Mersenne Twister (MT19937), so
// nothing here is invented. The details that must match CPython are:
//
//  1. SEEDING. For an int seed CPython takes abs(seed), splits it into 32-bit
//     little-endian words (always at least one word, so seed 0 -> key [0])
//     and runs the reference `init_by_array`. Negative seeds therefore give the
//     same stream as positive ones: Random(-5) == Random(5).
//  2. getrandbits(k) for k <= 32 is the TOP k bits of one 32-bit output:
//     genrand_uint32() >> (32 - k).
//  3. random() uses two outputs a = out1 >> 5 (27 bits), b = out2 >> 6
//     (26 bits) and returns (a * 2^26 + b) / 2^53, a double with 53 random bits.
//  4. randrange(n) is `_randbelow_with_getrandbits`: k = n.bit_length();
//     draw getrandbits(k) until the value is < n (rejection sampling, so the
//     number of outputs consumed varies). Note n = 1 still draws (k = 1, retry
//     until the bit is 0), which is why the spawn code must call it even when
//     there is only one empty cell.
//
// tests/test_engine_rng.py checks all four against random.Random, including
// the full internal state after thousands of draws.
#pragma once

#include <cstddef>
#include <cstdint>

namespace p1 {

class PyRandom {
public:
    static constexpr int kN = 624;
    static constexpr int kM = 397;

    // Same as random.Random(seed) for a Python int whose absolute value is
    // `abs_seed` (it fits 64 bits here; the Python wrapper checks that).
    static PyRandom from_seed(uint64_t abs_seed) {
        uint32_t key[2] = {static_cast<uint32_t>(abs_seed), static_cast<uint32_t>(abs_seed >> 32)};
        // CPython: key length = number of 32-bit words needed, minimum 1.
        return PyRandom(key, (abs_seed >> 32) ? 2 : 1);
    }

    // Reference MT19937 init_by_array (what CPython calls for int seeds).
    PyRandom(const uint32_t* key, size_t key_len) {
        init_genrand(19650218u);
        size_t i = 1, j = 0;
        for (size_t k = (kN > key_len ? kN : key_len); k; --k) {
            mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1664525u)) + key[j] + static_cast<uint32_t>(j);
            ++i;
            ++j;
            if (i >= static_cast<size_t>(kN)) {
                mt_[0] = mt_[kN - 1];
                i = 1;
            }
            if (j >= key_len) j = 0;
        }
        for (size_t k = kN - 1; k; --k) {
            mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1566083941u)) - static_cast<uint32_t>(i);
            ++i;
            if (i >= static_cast<size_t>(kN)) {
                mt_[0] = mt_[kN - 1];
                i = 1;
            }
        }
        mt_[0] = 0x80000000u;  // MSB set: the initial array is non-zero
        index_ = kN;           // CPython starts "exhausted": first draw twists
    }

    // One tempered 32-bit output (CPython's genrand_uint32).
    uint32_t next32() {
        if (index_ >= kN) twist();
        uint32_t y = mt_[index_++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= (y >> 18);
        return y;
    }

    // random.Random.getrandbits(k) for 0 <= k <= 32.
    uint32_t getrandbits(int k) { return k == 0 ? 0u : next32() >> (32 - k); }

    // random.Random.random(): float in [0, 1) with 53 random bits.
    double random() {
        const uint32_t a = next32() >> 5, b = next32() >> 6;
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
    }

    // random.Random.randrange(n) / _randbelow(n) for 1 <= n < 2^32.
    uint32_t randbelow(uint32_t n) {
        int k = 0;  // n.bit_length()
        for (uint32_t t = n; t; t >>= 1) ++k;
        uint32_t r = getrandbits(k);
        while (r >= n) r = getrandbits(k);
        return r;
    }

    // Internal state in the layout of random.Random.getstate()[1]:
    // 624 words, then the position (index) as a 625th element.
    void state(uint32_t* out625) const {
        for (int i = 0; i < kN; ++i) out625[i] = mt_[i];
        out625[kN] = static_cast<uint32_t>(index_);
    }

private:
    void init_genrand(uint32_t s) {
        mt_[0] = s;
        for (int i = 1; i < kN; ++i) mt_[i] = 1812433253u * (mt_[i - 1] ^ (mt_[i - 1] >> 30)) + static_cast<uint32_t>(i);
        index_ = kN;
    }

    void twist() {
        constexpr uint32_t kUpper = 0x80000000u, kLower = 0x7fffffffu, kMatrix = 0x9908b0dfu;
        auto mix = [&](uint32_t u, uint32_t v) { return (u & kUpper) | (v & kLower); };
        auto mag = [&](uint32_t y) { return (y >> 1) ^ ((y & 1u) ? kMatrix : 0u); };
        int kk = 0;
        for (; kk < kN - kM; ++kk) mt_[kk] = mt_[kk + kM] ^ mag(mix(mt_[kk], mt_[kk + 1]));
        for (; kk < kN - 1; ++kk) mt_[kk] = mt_[kk + (kM - kN)] ^ mag(mix(mt_[kk], mt_[kk + 1]));
        mt_[kN - 1] = mt_[kM - 1] ^ mag(mix(mt_[kN - 1], mt_[0]));
        index_ = 0;
    }

    uint32_t mt_[kN];
    int index_;
};

}  // namespace p1
