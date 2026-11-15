"""N-tuple value network: 4 six-tuples x 8 symmetries, float32 lookup tables.

How a board is scored
---------------------
The board (16 log2 exponents, see ``board.py``) is read through 4 small
"windows" of 6 cells each (the base tuples below). The 6 exponents a window
sees, each clamped at 15, are packed into one base-16 number: that number is
an index into the window's lookup table. The board's value is the sum of the
entries found.

Each window is also read in all 8 orientations of the board (4 rotations, each
optionally mirrored). The 8 orientations of a window SHARE one table, so there
are 4 tables but 4 x 8 = 32 lookups per evaluation.

Tuple layout (cells numbered row-major, 0 = top-left), as in Yeh et al. 2017,
"Multistage Temporal Difference Learning for 2048-like Games" (IEEE TCIAIG):

    T0 = {0,1,2,3,4,5}   T1 = {4,5,6,7,8,9}   T2 = {0,1,2,4,5,6}   T3 = {4,5,6,8,9,10}

Memory: 16**6 = 16,777,216 float32 entries = 67.1 MB per table, 268 MB total.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from agents.base import ValueFunction
from agents.ntuple._jit import njit
from agents.ntuple.board import SYMMETRIES, to_exponents

BASE_TUPLES = np.array(
    [
        [0, 1, 2, 3, 4, 5],
        [4, 5, 6, 7, 8, 9],
        [0, 1, 2, 4, 5, 6],
        [4, 5, 6, 8, 9, 10],
    ],
    dtype=np.int64,
)
N_TUPLES, TUPLE_LEN = BASE_TUPLES.shape
N_SYMMETRIES = SYMMETRIES.shape[0]
N_LOOKUPS = N_TUPLES * N_SYMMETRIES  # 32
TABLE_SIZE = 16**TUPLE_LEN  # 16,777,216
MAX_EXPONENT = 15  # clamp for indexing: 4 bits per cell

FORMAT_MAGIC = b"NTUP"
FORMAT_VERSION = 1
_ALIGN = 64  # weights start on a 64-byte boundary in the file


def _build_cells() -> np.ndarray:
    """CELLS[t, s, k] = board cell read as digit k of tuple t in orientation s.

    ``symmetries()`` defines the s-th symmetric board as sym[i] = board[P[s, i]].
    Reading tuple t on that symmetric board therefore reads the original board
    at P[s, BASE_TUPLES[t, k]]. Precomputing this means evaluation never
    builds the 8 symmetric boards; it just reads 32 different cell patterns.
    """
    cells = np.zeros((N_TUPLES, N_SYMMETRIES, TUPLE_LEN), dtype=np.int64)
    for t in range(N_TUPLES):
        for s in range(N_SYMMETRIES):
            cells[t, s] = SYMMETRIES[s, BASE_TUPLES[t]]
    return cells


CELLS = _build_cells()


# -------------------------------------------------------------------- kernels
@njit(cache=True)
def tuple_index(exps: np.ndarray, t: int, s: int) -> int:
    """Table index for tuple t in orientation s: 6 clamped exponents in base 16.

    The first cell of the tuple is the most significant digit, so the index
    is e0*16^5 + e1*16^4 + ... + e5. Clamping at 15 keeps every digit in 4
    bits, so a 65,536 tile (exponent 16) is read as 32,768 instead of
    overflowing into the next digit.
    """
    idx = 0
    for k in range(TUPLE_LEN):
        e = int(exps[CELLS[t, s, k]])  # int(): avoid uint8 overflow in plain Python
        if e > MAX_EXPONENT:
            e = MAX_EXPONENT
        idx = idx * 16 + e
    return idx


@njit(cache=True)
def evaluate_exps(weights: np.ndarray, exps: np.ndarray) -> float:
    """V(board) = sum of the 32 looked-up weights. Accumulated in float64."""
    v = 0.0
    for t in range(N_TUPLES):
        for s in range(N_SYMMETRIES):
            v += weights[t, tuple_index(exps, t, s)]
    return v


@njit(cache=True)
def update_exps(weights: np.ndarray, exps: np.ndarray, delta: float) -> None:
    """Add delta / 32 to each of the 32 weights ``exps`` looks up (in place).

    If all 32 lookups hit different entries, V(board) changes by exactly
    ``delta``. If some lookups share an entry (very symmetric boards, e.g. an
    empty board), that entry is both bumped and counted more than once, so V
    moves by more than ``delta``. This is the standard n-tuple update; it is
    rare and harmless on real game positions.
    """
    step = np.float32(delta / N_LOOKUPS)
    for t in range(N_TUPLES):
        for s in range(N_SYMMETRIES):
            idx = tuple_index(exps, t, s)
            weights[t, idx] += step


# -------------------------------------------------------------------- network
BoardLike = Union[np.ndarray, Sequence[Sequence[int]]]


def as_exponents(board: BoardLike) -> np.ndarray:
    """Accept a flat uint8[16] exponent array (fast path) or an engine board."""
    if isinstance(board, np.ndarray) and board.dtype == np.uint8 and board.shape == (16,):
        return board
    return to_exponents(board)


class NTupleNetwork(ValueFunction):
    """4 shared lookup tables, float32[4, 16**6].

    ``evaluate`` accepts either the engine's 4x4 tile-value board (what P3/P7
    pass) or a flat uint8[16] exponent array (what training uses).
    """

    def __init__(
        self,
        init_value: float = 0.0,
        weights: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        """Create a network.

        init_value: every weight starts at this constant. A board's initial
            value is then 32 * init_value. Optimistic initialisation sets this
            > 0 so unseen positions look good and get explored.
        weights: use this array instead (must be float32[4, 16**6]).
        meta: free-form info saved with the weights (alpha, games, seed, ...).
        """
        if weights is None:
            # np.zeros lets the OS hand out pages lazily (cheap until touched);
            # np.full has to write every entry.
            if init_value == 0.0:
                weights = np.zeros((N_TUPLES, TABLE_SIZE), dtype=np.float32)
            else:
                weights = np.full((N_TUPLES, TABLE_SIZE), init_value, dtype=np.float32)
        else:
            weights = np.ascontiguousarray(weights, dtype=np.float32)
            if weights.shape != (N_TUPLES, TABLE_SIZE):
                raise ValueError(f"weights must have shape {(N_TUPLES, TABLE_SIZE)}, got {weights.shape}")
        self.weights = weights
        self.meta: Dict[str, Any] = dict(meta or {})
        self.meta.setdefault("init_value", float(init_value))

    # ------------------------------------------------------------ value / learn
    def evaluate(self, board: BoardLike) -> float:
        """Value of ``board`` (pure; does not modify anything)."""
        return float(evaluate_exps(self.weights, as_exponents(board)))

    def update(self, board: BoardLike, delta: float) -> None:
        """Add ``delta / 32`` to each looked-up weight, so V(board) moves by ~delta.

        The caller passes the already-scaled step, e.g. alpha * TD error.
        """
        update_exps(self.weights, as_exponents(board), float(delta))

    # --------------------------------------------------------------- save/load
    def _header(self) -> Dict[str, Any]:
        return {
            "format": "ntuple",
            "version": FORMAT_VERSION,
            "dtype": "<f4",
            "shape": [N_TUPLES, TABLE_SIZE],
            "tuples": BASE_TUPLES.tolist(),
            "symmetries": N_SYMMETRIES,
            "max_exponent": MAX_EXPONENT,
            "meta": self.meta,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Write one binary file: magic, header length, JSON header, raw float32.

        Layout: b"NTUP" | uint32 LE header length | JSON (space-padded so the
        weights start on a 64-byte boundary) | weights as little-endian float32,
        row-major. Plain raw floats load fast with np.fromfile and can be read
        from any language.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = json.dumps(self._header()).encode("utf-8")
        prefix = len(FORMAT_MAGIC) + 4
        header += b" " * ((-(prefix + len(header))) % _ALIGN)
        with path.open("wb") as f:
            f.write(FORMAT_MAGIC)
            f.write(len(header).to_bytes(4, "little"))
            f.write(header)
            self.weights.astype("<f4", copy=False).tofile(f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NTupleNetwork":
        """Read a file written by ``save``; rejects mismatched format or layout."""
        with Path(path).open("rb") as f:
            if f.read(len(FORMAT_MAGIC)) != FORMAT_MAGIC:
                raise ValueError(f"{path}: not an n-tuple weights file")
            header_len = int.from_bytes(f.read(4), "little")
            header = json.loads(f.read(header_len).decode("utf-8"))
            if header.get("version") != FORMAT_VERSION:
                raise ValueError(f"{path}: format version {header.get('version')}, expected {FORMAT_VERSION}")
            if header.get("tuples") != BASE_TUPLES.tolist() or header.get("shape") != [N_TUPLES, TABLE_SIZE]:
                raise ValueError(f"{path}: tuple layout does not match this code")
            count = N_TUPLES * TABLE_SIZE
            weights = np.fromfile(f, dtype="<f4", count=count)
        if weights.size != count:
            raise ValueError(f"{path}: truncated ({weights.size} of {count} weights)")
        return cls(weights=weights.reshape(N_TUPLES, TABLE_SIZE), meta=header.get("meta"))
