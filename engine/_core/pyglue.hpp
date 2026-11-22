// engine/_core/pyglue.hpp -- converts between Python boards and p1::Board.
//
// Uses ONLY the plain CPython C API (no pybind11), so the same code can be
// compiled and tested on its own. binding.cpp is a thin pybind11 wrapper
// around slide() and legal_mask().
//
// Contract (this is what keeps unsupported boards on the Python engine):
//   * Both functions return Python None, never an exception, whenever the
//     input is not exactly what the core supports. None means "not handled,
//     use the pure-Python engine".
//   * Supported = board is a list/tuple of 4 list/tuples of 4 EXACT Python
//     ints (bool, numpy ints, floats, subclasses -> None), each cell 0 or a
//     power of two between 2 and 2^kMaxSupportedExp (262,144); the move is an
//     exact int 0..3.
//   * Results are fresh nested lists of Python ints, like the Python engine's.
#pragma once

#include <Python.h>

#include "core.hpp"

namespace p1glue {

// Cell -> exponent. False for anything the core does not support.
inline bool read_cell(PyObject* item, int8_t* exp_out) {
    if (!PyLong_CheckExact(item)) return false;  // rejects bool, float, numpy ints, str, None
    int overflow = 0;
    const long long v = PyLong_AsLongLongAndOverflow(item, &overflow);
    if (overflow != 0) return false;  // far bigger than any tile
    if (v == -1 && PyErr_Occurred()) {
        PyErr_Clear();
        return false;
    }
    if (v == 0) {
        *exp_out = 0;
        return true;
    }
    if (v < 2 || (v & (v - 1)) != 0) return false;  // negative, 1, or not a power of two
    int e = 0;
    while ((1LL << e) < v) ++e;
    if (e > p1::kMaxSupportedExp) return false;
    *exp_out = static_cast<int8_t>(e);
    return true;
}

// Borrowed references to the 4 items of an exact list/tuple of length 4.
inline bool items4(PyObject* o, PyObject** out) {
    if (PyList_CheckExact(o)) {
        if (PyList_GET_SIZE(o) != 4) return false;
        for (int i = 0; i < 4; ++i) out[i] = PyList_GET_ITEM(o, i);
        return true;
    }
    if (PyTuple_CheckExact(o)) {
        if (PyTuple_GET_SIZE(o) != 4) return false;
        for (int i = 0; i < 4; ++i) out[i] = PyTuple_GET_ITEM(o, i);
        return true;
    }
    return false;
}

inline bool read_board(PyObject* obj, p1::Board* out) {
    PyObject* rows[4];
    if (!items4(obj, rows)) return false;
    int8_t e[16];
    for (int r = 0; r < 4; ++r) {
        PyObject* cells[4];
        if (!items4(rows[r], cells)) return false;
        for (int c = 0; c < 4; ++c)
            if (!read_cell(cells[c], &e[r * 4 + c])) return false;
    }
    *out = p1::from_exponents(e);
    return true;
}

inline bool read_move(PyObject* obj, int* out) {
    if (!PyLong_CheckExact(obj)) return false;
    const long m = PyLong_AsLong(obj);
    if (m == -1 && PyErr_Occurred()) {
        PyErr_Clear();
        return false;
    }
    if (m < 0 || m > 3) return false;
    *out = static_cast<int>(m);
    return true;
}

// New reference to a nested list of tile VALUES, or NULL with an exception set.
inline PyObject* make_board(const p1::Board& b) {
    PyObject* rows = PyList_New(4);
    if (!rows) return nullptr;
    for (int r = 0; r < 4; ++r) {
        PyObject* row = PyList_New(4);
        if (!row) {
            Py_DECREF(rows);
            return nullptr;
        }
        PyList_SET_ITEM(rows, r, row);  // rows owns row now
        for (int c = 0; c < 4; ++c) {
            const int e = p1::get_cell(b, r, c);
            PyObject* v = PyLong_FromLongLong(e ? (1LL << e) : 0LL);
            if (!v) {
                Py_DECREF(rows);
                return nullptr;
            }
            PyList_SET_ITEM(row, c, v);
        }
    }
    return rows;
}

// slide(board, move) -> (new_board, gained) or None.  New reference / NULL on error.
inline PyObject* slide(PyObject* board, PyObject* move) {
    int m = 0;
    p1::Board b;
    if (!read_move(move, &m) || !read_board(board, &b)) Py_RETURN_NONE;
    const p1::MoveResult r = p1::move(b, m);
    PyObject* nb = make_board(r.board);
    if (!nb) return nullptr;
    PyObject* gained = PyLong_FromUnsignedLong(r.gained);
    if (!gained) {
        Py_DECREF(nb);
        return nullptr;
    }
    PyObject* out = PyTuple_Pack(2, nb, gained);
    Py_DECREF(nb);
    Py_DECREF(gained);
    return out;
}

// legal_mask(board) -> int (bit m set if move m changes the board) or None.
inline PyObject* legal_mask(PyObject* board) {
    p1::Board b;
    if (!read_board(board, &b)) Py_RETURN_NONE;
    return PyLong_FromLong(p1::legal_mask(b));
}

}  // namespace p1glue
