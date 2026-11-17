// engine/_core/binding.cpp -- pybind11 module  engine._cengine
//
// All conversion/validation logic lives in pyglue.hpp (plain C API, tested on
// its own). This file only exposes it to Python:
//
//   engine._cengine.slide(board, move)  -> (new_board, gained) or None
//   engine._cengine.legal_mask(board)   -> int or None
//   engine._cengine.MAX_SUPPORTED_EXP   -> 18
//
// None always means "not supported here, use the pure-Python engine".
#include <pybind11/pybind11.h>

#include "pyglue.hpp"

namespace py = pybind11;

namespace {
// Wrap a new reference returned by pyglue; NULL means a Python error is set.
py::object take(PyObject* raw) {
    if (!raw) throw py::error_already_set();
    return py::reinterpret_steal<py::object>(raw);
}
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
}
