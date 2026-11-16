// engine/_core/binding.cpp -- pybind11 module  engine._cengine
//
// All conversion/validation logic lives in pyglue.hpp (plain C API, tested on
// its own). This file only exposes it to Python:
//
//   engine._cengine.slide(board, move)  -> (new_board, gained) or None
//   engine._cengine.legal_mask(board)   -> int or None
//   engine._cengine.MAX_SUPPORTED_EXP   -> 18
//   engine._cengine.play_random(seeds, board, threads, record_moves) -> list or None
//   engine._cengine.PyRandom(seed)      -> CPython-compatible generator (for tests)
//
// None always means "not supported here, use the pure-Python engine".
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

#include "batch.hpp"
#include "pyglue.hpp"
#include "rng.hpp"

namespace py = pybind11;

namespace {
// Wrap a new reference returned by pyglue; NULL means a Python error is set.
py::object take(PyObject* raw) {
    if (!raw) throw py::error_already_set();
    return py::reinterpret_steal<py::object>(raw);
}

// abs(seed) as uint64 for an exact Python int; false if not an exact int or >= 2^64.
bool read_abs_seed(PyObject* o, uint64_t* out) {
    if (!PyLong_CheckExact(o)) return false;
    PyObject* a = PyNumber_Absolute(o);
    if (!a) {
        PyErr_Clear();
        return false;
    }
    const unsigned long long v = PyLong_AsUnsignedLongLong(a);
    Py_DECREF(a);
    if (v == static_cast<unsigned long long>(-1) && PyErr_Occurred()) {
        PyErr_Clear();
        return false;
    }
    *out = v;
    return true;
}

// play_random(seeds, board=None, threads=0, record_moves=True)
// -> list of (score, max_tile, n_moves, moves_bytes_or_None, final_board, bailed)
// or None when an input is not supported (caller then uses the Python engine).
py::object play_random(py::object seeds, py::object board, int threads, bool record_moves) {
    if (!PyList_CheckExact(seeds.ptr()) && !PyTuple_CheckExact(seeds.ptr())) return py::none();
    const py::ssize_t n = PySequence_Size(seeds.ptr());
    std::vector<uint64_t> abs_seeds(static_cast<size_t>(n));
    for (py::ssize_t i = 0; i < n; ++i) {
        PyObject* item = PySequence_GetItem(seeds.ptr(), i);  // new reference
        if (!item) throw py::error_already_set();
        const bool ok = read_abs_seed(item, &abs_seeds[static_cast<size_t>(i)]);
        Py_DECREF(item);
        if (!ok) return py::none();
    }
    p1::Board start{};
    const bool has_start = !board.is_none();
    if (has_start && (!p1glue::read_board(board.ptr(), &start) || !p1::is_supported(start))) return py::none();

    std::vector<p1::GameResult> results(static_cast<size_t>(n));
    {
        py::gil_scoped_release release;  // pure C++ from here: other Python threads may run
        p1::play_random_batch(abs_seeds.data(), abs_seeds.size(), has_start ? &start : nullptr, record_moves, threads,
                              results.data());
    }
    py::list out;
    for (const p1::GameResult& r : results) {
        py::object moves = py::none();
        if (record_moves)
            moves = py::reinterpret_steal<py::object>(
                PyBytes_FromStringAndSize(reinterpret_cast<const char*>(r.moves.data()), static_cast<Py_ssize_t>(r.moves.size())));
        py::object brd = py::reinterpret_steal<py::object>(p1glue::make_board(r.board));
        if (!brd) throw py::error_already_set();
        out.append(py::make_tuple(py::int_(r.score), py::int_(r.max_exp ? (1ULL << r.max_exp) : 0ULL), r.n_moves,
                                  moves, brd, r.bailed));
    }
    return out;
}

// Test-only wrapper so Python can compare our generator with random.Random.
struct PyRandomHandle {
    p1::PyRandom rng;
    explicit PyRandomHandle(py::int_ seed) : rng(make(seed)) {}
    static p1::PyRandom make(py::int_ seed) {
        py::int_ a = py::reinterpret_steal<py::int_>(PyNumber_Absolute(seed.ptr()));
        std::vector<uint32_t> key;
        py::int_ mask(0xffffffffULL), shift(32);
        while (key.empty() || a > py::int_(0)) {
            key.push_back(static_cast<uint32_t>(py::int_(a & mask)));
            a = py::int_(a >> shift);
        }
        return p1::PyRandom(key.data(), key.size());
    }
};
}  // namespace

PYBIND11_MODULE(_cengine, m) {
    m.doc() = "C++ 2048 core (5-bit exponent rows, precomputed tables). None = unsupported input.";
    p1::init_tables();  // ~16 MB of row tables, built once per process
    m.attr("MAX_SUPPORTED_EXP") = p1::kMaxSupportedExp;
    m.def(
        "slide", [](py::object board, py::object move) { return take(p1glue::slide(board.ptr(), move.ptr())); },
        py::arg("board"), py::arg("move"),
        "Slide `board` in direction `move` (0 up, 1 right, 2 down, 3 left), no spawn. "
        "Returns (new_board, gained), or None if the input is unsupported.");
    m.def(
        "legal_mask", [](py::object board) { return take(p1glue::legal_mask(board.ptr())); }, py::arg("board"),
        "Bit m is set if move m changes the board; None if the board is unsupported.");
    m.def("play_random", &play_random, py::arg("seeds"), py::arg("board") = py::none(), py::arg("threads") = 0,
          py::arg("record_moves") = true,
          "Play one random-policy game per seed entirely in C++ (GIL released). Each result is "
          "(score, max_tile, n_moves, moves_bytes_or_None, final_board, bailed). None = unsupported input.");
    m.attr("POLICY_SEED_MASK") = py::int_(p1::kPolicySeedMask);
    py::class_<PyRandomHandle>(m, "PyRandom", "CPython-compatible random.Random (test helper).")
        .def(py::init<py::int_>(), py::arg("seed"))
        .def("getrandbits", [](PyRandomHandle& h, int k) { return h.rng.getrandbits(k); }, py::arg("k"))
        .def("random", [](PyRandomHandle& h) { return h.rng.random(); })
        .def("randbelow", [](PyRandomHandle& h, uint32_t n) { return h.rng.randbelow(n); }, py::arg("n"))
        .def("getstate", [](PyRandomHandle& h) {
            std::vector<uint32_t> st(p1::PyRandom::kN + 1);
            h.rng.state(st.data());
            return py::tuple(py::cast(st));
        });
}
