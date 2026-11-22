// hsearch.cpp -- P7: expectimax over afterstates with an n-tuple value function.
//
// WHAT THIS FILE IS
//   The hot path of the hybrid agent. It plays no games itself; Python hands it
//   a board and it returns the expected value of each of the four moves.
//
// KEY DESIGN CHOICES (each is defended in agents/hybrid/README.md)
//   * Board = 16 bytes, one log2 exponent per cell, row-major (0 empty, 1 = 2,
//     ..., 16 = 65,536). Exponents go up to 17 so two 32,768s merging is fine.
//   * Slides use a precomputed table of "what does this row become?" (18^4 rows).
//   * The n-tuple weights are NOT copied: we read the numpy float32 array that
//     P5's StagedNetwork already holds (268 MB per stage). The cell layout also
//     comes from Python (CELLS), so P5 and this file can never disagree.
//   * The value function V was trained on AFTERSTATES (board after the slide,
//     before the random tile). So leaves are scored as V(afterstate) + reward.
//   * Floating point is evaluated in the same order as the Python reference
//     (agents/hybrid/reference.py) and with -ffp-contract=off, so both give the
//     same numbers and tests can compare them directly.
//   * Threads: each ROOT move is searched independently (own transposition
//     table), so the answer never depends on thread scheduling.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

constexpr int N_TUPLES = 4;
constexpr int N_SYM = 8;
constexpr int TUPLE_LEN = 6;
constexpr int MAX_EXP_INDEX = 15;          // n-tuple index clamp (4 bits per cell)
constexpr int64_t TABLE_SIZE = 1 << 24;    // 16^6 entries per tuple table
constexpr int ROW_BASE = 18;               // exponents 0..17 are legal inputs
constexpr int ROW_COUNT = ROW_BASE * ROW_BASE * ROW_BASE * ROW_BASE;
constexpr int MAX_BOARD_EXP = ROW_BASE - 1;

using Clock = std::chrono::steady_clock;

struct Board {
    uint8_t c[16];
};

// ---------------------------------------------------------------- row table
struct RowEntry {
    uint8_t out[4];
    uint32_t reward;
    bool changed;
};

std::vector<RowEntry> build_rows() {
    std::vector<RowEntry> t(ROW_COUNT);
    for (int key = 0; key < ROW_COUNT; ++key) {
        int e[4], k = key;
        for (int i = 0; i < 4; ++i) {
            e[i] = k % ROW_BASE;
            k /= ROW_BASE;
        }
        int tiles[4], n = 0;
        for (int i = 0; i < 4; ++i)
            if (e[i]) tiles[n++] = e[i];
        int out[4] = {0, 0, 0, 0}, m = 0;
        uint32_t reward = 0;
        for (int i = 0; i < n;) {
            if (i + 1 < n && tiles[i] == tiles[i + 1]) {   // each tile merges at most once
                int merged = std::min(tiles[i] + 1, MAX_BOARD_EXP);
                out[m++] = merged;
                reward += 1u << merged;                     // value of the new tile
                i += 2;
            } else {
                out[m++] = tiles[i++];
            }
        }
        RowEntry& r = t[key];
        r.reward = reward;
        r.changed = false;
        for (int i = 0; i < 4; ++i) {
            r.out[i] = static_cast<uint8_t>(out[i]);
            if (out[i] != e[i]) r.changed = true;
        }
    }
    return t;
}

const std::vector<RowEntry>& row_table() {
    static const std::vector<RowEntry> t = build_rows();   // thread-safe init (C++11)
    return t;
}

inline int row_key(int a, int b, int c, int d) {
    return a + ROW_BASE * (b + ROW_BASE * (c + ROW_BASE * d));
}

// Moves: 0 = up, 1 = right, 2 = down, 3 = left (same as the Python engine).
// Returns whether the board changed; `out` and `reward` are always filled.
inline bool slide(const RowEntry* rows, const Board& b, int move, Board& out, uint32_t& reward) {
    reward = 0;
    bool changed = false;
    switch (move) {
        case 3:
            for (int r = 0; r < 4; ++r) {
                const uint8_t* p = &b.c[4 * r];
                const RowEntry& e = rows[row_key(p[0], p[1], p[2], p[3])];
                for (int k = 0; k < 4; ++k) out.c[4 * r + k] = e.out[k];
                reward += e.reward;
                changed |= e.changed;
            }
            break;
        case 1:
            for (int r = 0; r < 4; ++r) {
                const uint8_t* p = &b.c[4 * r];
                const RowEntry& e = rows[row_key(p[3], p[2], p[1], p[0])];
                for (int k = 0; k < 4; ++k) out.c[4 * r + 3 - k] = e.out[k];
                reward += e.reward;
                changed |= e.changed;
            }
            break;
        case 0:
            for (int c = 0; c < 4; ++c) {
                const RowEntry& e = rows[row_key(b.c[c], b.c[4 + c], b.c[8 + c], b.c[12 + c])];
                for (int k = 0; k < 4; ++k) out.c[4 * k + c] = e.out[k];
                reward += e.reward;
                changed |= e.changed;
            }
            break;
        case 2:
            for (int c = 0; c < 4; ++c) {
                const RowEntry& e = rows[row_key(b.c[12 + c], b.c[8 + c], b.c[4 + c], b.c[c])];
                for (int k = 0; k < 4; ++k) out.c[4 * (3 - k) + c] = e.out[k];
                reward += e.reward;
                changed |= e.changed;
            }
            break;
        default:
            throw std::invalid_argument("move must be 0..3");
    }
    return changed;
}

// ------------------------------------------------------------------ context
// Everything the search reads but never writes. Shared by all threads.
struct Ctx {
    const float* weights = nullptr;      // [n_stages, 4, 16^6], C order
    int n_stages = 1;
    std::vector<int> stage_exps;         // ascending thresholds (exponent of max tile)
    uint8_t cells[N_TUPLES][N_SYM][TUPLE_LEN];
    const RowEntry* rows = nullptr;

    // V(afterstate). Same arithmetic order as agents/ntuple/network.py:
    // loop tuples, then symmetries, accumulate float32 table entries in double.
    double eval(const Board& b) const {
        int mx = 0;
        for (int i = 0; i < 16; ++i) mx = std::max<int>(mx, b.c[i]);
        int s = 0;
        for (size_t j = 0; j < stage_exps.size(); ++j)
            if (mx >= stage_exps[j]) s = static_cast<int>(j) + 1;
        const float* w = weights + static_cast<int64_t>(s) * N_TUPLES * TABLE_SIZE;
        double v = 0.0;
        for (int t = 0; t < N_TUPLES; ++t) {
            const float* wt = w + static_cast<int64_t>(t) * TABLE_SIZE;
            for (int sy = 0; sy < N_SYM; ++sy) {
                uint32_t idx = 0;
                for (int k = 0; k < TUPLE_LEN; ++k) {
                    uint32_t e = b.c[cells[t][sy][k]];
                    if (e > MAX_EXP_INDEX) e = MAX_EXP_INDEX;   // 65,536 is seen as 32,768
                    idx = idx * 16 + e;
                }
                v += wt[idx];
            }
        }
        return v;
    }
};

// ------------------------------------------------- transposition table (exact)
// Open addressing, grows at 50% load, never evicts: it behaves exactly like the
// Python dict in the reference, so the two implementations stay comparable.
class TT {
   public:
    TT() { reset(1 << 12); }
    bool get(const Board& b, int depth, double& val) const {
        uint64_t lo, hi;
        pack(b, lo, hi);
        size_t i = hash(lo, hi, depth) & mask_;
        while (t_[i].depth >= 0) {
            if (t_[i].lo == lo && t_[i].hi == hi && t_[i].depth == depth) {
                val = t_[i].val;
                return true;
            }
            i = (i + 1) & mask_;
        }
        return false;
    }
    void put(const Board& b, int depth, double val) {
        if ((count_ + 1) * 2 > t_.size()) grow();
        uint64_t lo, hi;
        pack(b, lo, hi);
        insert(lo, hi, depth, val);
    }

   private:
    struct E {
        uint64_t lo, hi;
        int32_t depth;   // -1 = empty slot
        double val;
    };
    std::vector<E> t_;
    size_t mask_ = 0, count_ = 0;

    static void pack(const Board& b, uint64_t& lo, uint64_t& hi) {
        std::memcpy(&lo, b.c, 8);
        std::memcpy(&hi, b.c + 8, 8);
    }
    static uint64_t mix(uint64_t x) {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    }
    static uint64_t hash(uint64_t lo, uint64_t hi, int depth) {
        return mix(lo ^ mix(hi + 0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(depth + 1)));
    }
    void reset(size_t n) {
        t_.assign(n, E{0, 0, -1, 0.0});
        mask_ = n - 1;
        count_ = 0;
    }
    void insert(uint64_t lo, uint64_t hi, int depth, double val) {
        size_t i = hash(lo, hi, depth) & mask_;
        while (t_[i].depth >= 0) {
            if (t_[i].lo == lo && t_[i].hi == hi && t_[i].depth == depth) {
                t_[i].val = val;
                return;
            }
            i = (i + 1) & mask_;
        }
        t_[i] = E{lo, hi, depth, val};
        ++count_;
    }
    void grow() {
        std::vector<E> old;
        old.swap(t_);
        reset(old.size() * 2);
        for (const E& e : old)
            if (e.depth >= 0) insert(e.lo, e.hi, e.depth, e.val);
    }
};

// ----------------------------------------------------------------- searcher
// One Searcher = one thread's worth of state (own TT, own node counter).
struct Searcher {
    const Ctx& ctx;
    double cutoff;
    bool use_tt;
    bool timed;
    Clock::time_point deadline;
    std::atomic<bool>& abort_flag;
    uint64_t nodes = 0;
    TT tt;

    Searcher(const Ctx& c, double cut, bool tt_on, bool is_timed, Clock::time_point dl,
             std::atomic<bool>& ab)
        : ctx(c), cutoff(cut), use_tt(tt_on), timed(is_timed), deadline(dl), abort_flag(ab) {}

    // Value of an afterstate with `d` player moves still to search.
    //   d == 0 or too improbable -> V(afterstate)  (leaf)
    //   otherwise average over every empty cell and spawn value (2: 0.9, 4: 0.1).
    double chance(const Board& after, int d, double prob) {
        if (d == 0 || prob < cutoff) return ctx.eval(after);
        if (timed) {
            if (abort_flag.load(std::memory_order_relaxed)) return 0.0;
            // Look at the clock every 16 chance nodes (a node costs ~tens of microseconds).
            if (((++nodes) & 15u) == 0 && Clock::now() >= deadline) {
                abort_flag.store(true);
                return 0.0;
            }
        } else {
            ++nodes;
        }
        int cells[16], n = 0;
        for (int i = 0; i < 16; ++i)
            if (after.c[i] == 0) cells[n++] = i;
        if (n == 0) return ctx.eval(after);
        double total = 0.0;
        for (int k = 0; k < n; ++k) {
            for (int tile = 1; tile <= 2; ++tile) {
                double p = (tile == 1) ? 0.9 : 0.1;
                Board child = after;
                child.c[cells[k]] = static_cast<uint8_t>(tile);
                total += p * max_node(child, d - 1, prob * p / n);
            }
        }
        return total / n;
    }

    // Value of a board right after a tile spawned: best of reward + chance(...).
    // At d == 0 this is plain 1-ply greedy on reward + V(afterstate), i.e. exactly
    // the policy the n-tuple network was trained with.
    double max_node(const Board& b, int d, double prob) {
        double cached;
        if (d >= 1 && use_tt && tt.get(b, d, cached)) return cached;
        double best = 0.0;
        bool any = false;
        for (int m = 0; m < 4; ++m) {
            Board after;
            uint32_t reward;
            if (!slide(ctx.rows, b, m, after, reward)) continue;
            double v = static_cast<double>(reward) + chance(after, d, prob);
            if (!any || v > best) {     // strict > : ties keep the lowest move number
                best = v;
                any = true;
            }
        }
        if (!any) best = 0.0;           // game over: no more score to come
        if (d >= 1 && use_tt && !(timed && abort_flag.load(std::memory_order_relaxed)))
            tt.put(b, d, best);
        return best;
    }
};

struct Root {
    int move;
    Board after;
    uint32_t reward;
};

}  // namespace

// ------------------------------------------------------------- python class
class HybridSearcher {
   public:
    HybridSearcher(py::array_t<float, py::array::c_style> weights,
                   py::array_t<int64_t, py::array::c_style | py::array::forcecast> stage_exps,
                   py::array_t<int64_t, py::array::c_style | py::array::forcecast> cells)
        : weights_(weights) {
        if (weights.ndim() != 3 || weights.shape(1) != N_TUPLES || weights.shape(2) != TABLE_SIZE)
            throw std::invalid_argument("weights must have shape [stages, 4, 16**6] (float32)");
        ctx_.weights = weights.data();
        ctx_.n_stages = static_cast<int>(weights.shape(0));
        ctx_.stage_exps.assign(stage_exps.data(), stage_exps.data() + stage_exps.size());
        if (static_cast<int>(ctx_.stage_exps.size()) != ctx_.n_stages - 1)
            throw std::invalid_argument("stage_exps must have n_stages - 1 entries");
        if (cells.ndim() != 3 || cells.shape(0) != N_TUPLES || cells.shape(1) != N_SYM ||
            cells.shape(2) != TUPLE_LEN)
            throw std::invalid_argument("cells must have shape [4, 8, 6]");
        const int64_t* cp = cells.data();
        for (int i = 0; i < N_TUPLES * N_SYM * TUPLE_LEN; ++i) {
            if (cp[i] < 0 || cp[i] > 15) throw std::invalid_argument("cell index out of range");
            (&ctx_.cells[0][0][0])[i] = static_cast<uint8_t>(cp[i]);
        }
        ctx_.rows = row_table().data();
    }

    double evaluate(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> board) const {
        return ctx_.eval(to_board(board));
    }

    py::tuple slide_py(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> board,
                       int move) const {
        Board in = to_board(board), out;
        uint32_t reward;
        bool changed = slide(ctx_.rows, in, move, out, reward);
        py::array_t<uint8_t> arr(16);
        std::memcpy(arr.mutable_data(), out.c, 16);
        return py::make_tuple(arr, static_cast<int64_t>(reward), changed);
    }

    // search(board, depth, cutoff, use_tt, time_ms, threads, iterative) -> dict
    //   time_ms > 0  : iterative deepening 0,1,2,...,depth, stopping at the deadline.
    //   iterative    : same without a clock (used to test the machinery).
    //   otherwise    : one search at exactly `depth`.
    // Depth 0 always completes, so a legal move is always available.
    py::dict search(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> board_arr,
                    int max_depth, double cutoff, bool use_tt, double time_ms, int threads,
                    bool iterative) const {
        if (max_depth < 0) throw std::invalid_argument("depth must be >= 0");
        const Board board = to_board(board_arr);
        std::vector<Root> roots;
        for (int m = 0; m < 4; ++m) {
            Root r;
            r.move = m;
            if (slide(ctx_.rows, board, m, r.after, r.reward)) roots.push_back(r);
        }
        const double nan = std::numeric_limits<double>::quiet_NaN();
        std::array<double, 4> best_values = {nan, nan, nan, nan};
        int depth_done = -1;
        uint64_t total_nodes = 0;
        bool aborted = false;
        const auto t_start = Clock::now();
        const bool timed = time_ms > 0.0;
        const auto deadline =
            t_start + std::chrono::duration_cast<Clock::duration>(
                          std::chrono::duration<double, std::milli>(timed ? time_ms : 0.0));
        auto ms_since = [&](Clock::time_point t0) {
            return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        };

        if (!roots.empty()) {
            py::gil_scoped_release release;     // the search never touches Python objects
            std::atomic<bool> abort_flag{false};
            const int d0 = (timed || iterative) ? 0 : max_depth;
            double last_ms = 0.0, prev_ms = 0.0;
            for (int d = d0; d <= max_depth; ++d) {
                if (timed && d > d0) {
                    // Don't start an iteration that probably can't finish: assume it costs
                    // `growth` x the previous one (measured from the last two, clamped).
                    double growth = prev_ms > 0.0 ? std::min(30.0, std::max(2.0, last_ms / prev_ms)) : 8.0;
                    if (ms_since(t_start) + last_ms * growth > time_ms) break;
                }
                const auto t_iter = Clock::now();
                std::array<double, 4> vals = {nan, nan, nan, nan};
                uint64_t nodes = 0;
                run_depth(roots, d, cutoff, use_tt, timed, deadline, threads, abort_flag, vals, nodes);
                total_nodes += nodes;
                if (abort_flag.load()) {        // unfinished iteration: throw it away
                    aborted = true;
                    break;
                }
                best_values = vals;
                depth_done = d;
                prev_ms = last_ms;
                last_ms = ms_since(t_iter);
            }
        }
        py::dict out;
        py::list values;
        for (double v : best_values) values.append(v);
        out["values"] = values;
        out["depth"] = depth_done;
        out["nodes"] = static_cast<py::int_>(total_nodes);
        out["elapsed_ms"] = ms_since(t_start);
        out["aborted"] = aborted;
        return out;
    }

   private:
    py::array_t<float, py::array::c_style> weights_;   // keeps the 268 MB table alive
    Ctx ctx_;

    static Board to_board(const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& a) {
        if (a.size() != 16) throw std::invalid_argument("board must have 16 exponents");
        Board b;
        std::memcpy(b.c, a.data(), 16);
        for (int i = 0; i < 16; ++i)
            if (b.c[i] > MAX_BOARD_EXP) throw std::invalid_argument("exponent too large");
        return b;
    }

    void run_depth(const std::vector<Root>& roots, int d, double cutoff, bool use_tt, bool timed,
                   Clock::time_point deadline, int threads, std::atomic<bool>& abort_flag,
                   std::array<double, 4>& vals, uint64_t& nodes) const {
        std::atomic<size_t> next{0};
        std::vector<uint64_t> node_counts(roots.size(), 0);
        auto worker = [&]() {
            for (;;) {
                size_t i = next.fetch_add(1);
                if (i >= roots.size()) return;
                // Fresh searcher (and TT) per root move: results never depend on which
                // thread got which move, so 1 thread and 4 threads agree exactly.
                Searcher s(ctx_, cutoff, use_tt, timed, deadline, abort_flag);
                double v = static_cast<double>(roots[i].reward) + s.chance(roots[i].after, d, 1.0);
                vals[roots[i].move] = v;
                node_counts[i] = s.nodes;
            }
        };
        int n_workers = std::max(1, std::min<int>(threads, static_cast<int>(roots.size())));
        if (n_workers == 1) {
            worker();
        } else {
            std::vector<std::thread> pool;
            for (int i = 0; i < n_workers; ++i) pool.emplace_back(worker);
            for (auto& t : pool) t.join();
        }
        for (uint64_t c : node_counts) nodes += c;
    }
};

PYBIND11_MODULE(_hsearch, m) {
    m.doc() = "P7 hybrid search: expectimax over afterstates with n-tuple leaves (C++)";
    py::class_<HybridSearcher>(m, "HybridSearcher")
        .def(py::init<py::array_t<float, py::array::c_style>,
                      py::array_t<int64_t, py::array::c_style | py::array::forcecast>,
                      py::array_t<int64_t, py::array::c_style | py::array::forcecast>>(),
             py::arg("weights"), py::arg("stage_exps"), py::arg("cells"))
        .def("evaluate", &HybridSearcher::evaluate, py::arg("board"))
        .def("slide", &HybridSearcher::slide_py, py::arg("board"), py::arg("move"))
        .def("search", &HybridSearcher::search, py::arg("board"), py::arg("depth"),
             py::arg("cutoff") = 1e-4, py::arg("use_tt") = true, py::arg("time_ms") = 0.0,
             py::arg("threads") = 1, py::arg("iterative") = false);
    m.def("hardware_threads", []() { return static_cast<int>(std::thread::hardware_concurrency()); });
}
