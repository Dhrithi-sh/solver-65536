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
FORMAT_VERSION = 2  # 2: multi-stage shape [S, 4, 16**6]; version 1 files still load
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


@njit(cache=True)
def stage_of(exps: np.ndarray, stage_exps: np.ndarray) -> int:
    """Stage index of a board: how many stage thresholds its biggest tile has reached.

    ``stage_exps`` holds the threshold exponents in ascending order (empty for a
    single-stage network). With thresholds [14] (= 16,384): boards whose max tile
    is below 16,384 are stage 0, boards holding 16,384 or more are stage 1. The
    max tile never shrinks during a game, so "max tile >= threshold" is the same
    as "the board has (at some point) contained the stage tile".
    """
    m = 0
    for k in range(16):
        if exps[k] > m:
            m = exps[k]
    s = 0
    for j in range(stage_exps.shape[0]):
        if m >= stage_exps[j]:
            s = j + 1
    return s


@njit(cache=True)
def evaluate_staged(weights3: np.ndarray, stage_exps: np.ndarray, exps: np.ndarray) -> float:
    """V(board) using the network of the board's own stage. weights3: [S, 4, 16**6]."""
    return evaluate_exps(weights3[stage_of(exps, stage_exps)], exps)


@njit(cache=True)
def update_staged(weights3: np.ndarray, stage_exps: np.ndarray, exps: np.ndarray, delta: float) -> None:
    """update_exps on the network of the board's own stage."""
    update_exps(weights3[stage_of(exps, stage_exps)], exps, delta)


@njit(cache=True)
def ensure_stage_started(weights3: np.ndarray, started: np.ndarray, s: int) -> None:
    """Warm-start stage s from stage s-1 the first time training reaches it.

    A brand-new stage network knows nothing (all weights at their initial
    value). Copying the previous stage's tables gives it a sensible starting
    point, so the TD target across the stage boundary isn't garbage while the
    new stage is still rare. ``started`` (uint8[S]) records which stages are
    live; read-only users (the agent) pass all ones so nothing is ever copied.
    """
    if s > 0 and started[s] == 0:
        weights3[s, :, :] = weights3[s - 1, :, :]
        started[s] = 1


# -------------------------------------------------------------------- network
BoardLike = Union[np.ndarray, Sequence[Sequence[int]]]
PathLike = Union[str, Path]


def as_exponents(board: BoardLike) -> np.ndarray:
    """Accept a flat uint8[16] exponent array (fast path) or an engine board."""
    if isinstance(board, np.ndarray) and board.dtype == np.uint8 and board.shape == (16,):
        return board
    return to_exponents(board)


def _initial_weights(shape, init_value: float) -> np.ndarray:
    """Weights such that every board's value starts at ``init_value``.

    A value is the sum of 32 lookups, so each weight is init_value / 32.
    np.zeros lets the OS hand out pages lazily (cheap until touched);
    np.full has to write every entry.
    """
    if init_value == 0.0:
        return np.zeros(shape, dtype=np.float32)
    return np.full(shape, init_value / N_LOOKUPS, dtype=np.float32)


class NTupleNetwork(ValueFunction):
    """One set of 4 shared lookup tables, float32[4, 16**6] (a single stage).

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

        init_value: every board's value starts at this (each weight is
            init_value / 32). Optimistic initialisation sets it high so unseen
            positions look good and get explored.
        weights: use this array instead (must be float32[4, 16**6]; not copied
            if it already is, so a view into a bigger array stays a view).
        meta: free-form info saved with the weights (alpha, games, seed, ...).
        """
        if weights is None:
            weights = _initial_weights((N_TUPLES, TABLE_SIZE), init_value)
        else:
            weights = np.ascontiguousarray(weights, dtype=np.float32)
            if weights.shape != (N_TUPLES, TABLE_SIZE):
                raise ValueError(f"weights must have shape {(N_TUPLES, TABLE_SIZE)}, got {weights.shape}")
        self.weights = weights
        self.meta: Dict[str, Any] = dict(meta or {})
        self.meta.setdefault("init_value", float(init_value))

    def evaluate(self, board: BoardLike) -> float:
        """Value of ``board`` (pure; does not modify anything)."""
        return float(evaluate_exps(self.weights, as_exponents(board)))

    def update(self, board: BoardLike, delta: float) -> None:
        """Add ``delta / 32`` to each looked-up weight, so V(board) moves by ~delta.

        The caller passes the already-scaled step, e.g. alpha * TD error.
        """
        update_exps(self.weights, as_exponents(board), float(delta))

    def save(self, path: PathLike) -> None:
        """Save as a one-stage weights file (see ``StagedNetwork.save``)."""
        _write_weights(path, self.weights[None], [], self.meta)

    @classmethod
    def load(cls, path: PathLike) -> "NTupleNetwork":
        """Load a one-stage weights file. Multi-stage files need ``StagedNetwork.load``."""
        weights3, stage_tiles, meta = _read_weights(path)
        if weights3.shape[0] != 1:
            raise ValueError(f"{path}: has {weights3.shape[0]} stages; load it with StagedNetwork.load")
        return cls(weights=weights3[0], meta=meta)


class StagedNetwork(ValueFunction):
    """Multi-stage n-tuple network (Yeh et al. 2017): one NTupleNetwork per stage.

    All stages live in one contiguous float32[S, 4, 16**6] array (so numba can
    take them in a single argument); ``stages[k]`` is an NTupleNetwork view of
    slice k. A board is scored by the stage its biggest tile puts it in. With
    S = 1 this is exactly a plain NTupleNetwork.
    """

    def __init__(
        self,
        n_stages: int = 1,
        stage_tiles: Sequence[int] = (),
        init_value: float = 0.0,
        weights: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        """n_stages: number of stages S.
        stage_tiles: S - 1 ascending tile values (powers of two) where the next
            stage begins, e.g. [16384] for two stages.
        init_value: every board's value starts at this, in every stage.
        """
        tiles = [int(t) for t in stage_tiles]
        if len(tiles) != n_stages - 1:
            raise ValueError(f"{n_stages} stages need {n_stages - 1} stage tile(s), got {tiles}")
        for t in tiles:
            if t < 4 or t & (t - 1):
                raise ValueError(f"stage tile {t} is not a power of two >= 4")
        if tiles != sorted(set(tiles)):
            raise ValueError(f"stage tiles must be strictly increasing, got {tiles}")
        shape = (n_stages, N_TUPLES, TABLE_SIZE)
        if weights is None:
            weights = _initial_weights(shape, init_value)
        else:
            weights = np.ascontiguousarray(weights, dtype=np.float32)
            if weights.shape != shape:
                raise ValueError(f"weights must have shape {shape}, got {weights.shape}")
        self.weights = weights
        self.stage_tiles = tiles
        self.stage_exps = np.array([t.bit_length() - 1 for t in tiles], dtype=np.int64)
        self.stages = [NTupleNetwork(weights=weights[k]) for k in range(n_stages)]
        self.meta: Dict[str, Any] = dict(meta or {})
        self.meta.setdefault("init_value", float(init_value))

    @property
    def n_stages(self) -> int:
        return self.weights.shape[0]

    @classmethod
    def from_network(cls, net: NTupleNetwork) -> "StagedNetwork":
        """Wrap a single NTupleNetwork as a 1-stage network (shares its tables)."""
        return cls(weights=net.weights[None], meta=net.meta)

    def stage_of(self, board: BoardLike) -> int:
        """Which stage's network scores ``board`` (0-based)."""
        return int(stage_of(as_exponents(board), self.stage_exps))

    def evaluate(self, board: BoardLike) -> float:
        """Value of ``board`` from its own stage's network (pure)."""
        return float(evaluate_staged(self.weights, self.stage_exps, as_exponents(board)))

    def update(self, board: BoardLike, delta: float) -> None:
        """``NTupleNetwork.update`` on the board's own stage."""
        update_staged(self.weights, self.stage_exps, as_exponents(board), float(delta))

    def save(self, path: PathLike) -> None:
        """Write all stages to one file (format below, in ``_write_weights``)."""
        _write_weights(path, self.weights, self.stage_tiles, self.meta)

    @classmethod
    def load(cls, path: PathLike) -> "StagedNetwork":
        """Load any weights file (1 or more stages; format version 1 or 2)."""
        weights3, stage_tiles, meta = _read_weights(path)
        return cls(weights3.shape[0], stage_tiles, weights=weights3, meta=meta)


# ------------------------------------------------------------- file format
def _write_weights(path: PathLike, weights3: np.ndarray, stage_tiles: Sequence[int], meta: Dict[str, Any]) -> None:
    """Write one binary file: magic, header length, JSON header, raw float32.

    Layout: b"NTUP" | uint32 LE header length | JSON (space-padded so the
    weights start on a 64-byte boundary) | weights as little-endian float32,
    shape [stages, 4, 16**6], row-major. Plain raw floats load fast with
    np.fromfile and can be read from any language.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "format": "ntuple",
        "version": FORMAT_VERSION,
        "dtype": "<f4",
        "shape": list(weights3.shape),
        "stage_tiles": [int(t) for t in stage_tiles],
        "tuples": BASE_TUPLES.tolist(),
        "symmetries": N_SYMMETRIES,
        "max_exponent": MAX_EXPONENT,
        "meta": meta,
    }
    raw = json.dumps(header).encode("utf-8")
    prefix = len(FORMAT_MAGIC) + 4
    raw += b" " * ((-(prefix + len(raw))) % _ALIGN)
    with path.open("wb") as f:
        f.write(FORMAT_MAGIC)
        f.write(len(raw).to_bytes(4, "little"))
        f.write(raw)
        weights3.astype("<f4", copy=False).tofile(f)


def _read_weights(path: PathLike):
    """Read a weights file -> (float32[S, 4, 16**6], stage_tiles, meta).

    Version 1 files (single stage, shape [4, 16**6]) are still accepted.
    """
    with Path(path).open("rb") as f:
        if f.read(len(FORMAT_MAGIC)) != FORMAT_MAGIC:
            raise ValueError(f"{path}: not an n-tuple weights file")
        header_len = int.from_bytes(f.read(4), "little")
        header = json.loads(f.read(header_len).decode("utf-8"))
        version = header.get("version")
        if version not in (1, FORMAT_VERSION):
            raise ValueError(f"{path}: format version {version}, expected 1 or {FORMAT_VERSION}")
        shape = header.get("shape")
        if version == 1 and shape == [N_TUPLES, TABLE_SIZE]:
            shape = [1, N_TUPLES, TABLE_SIZE]
        if header.get("tuples") != BASE_TUPLES.tolist() or not shape or shape[1:] != [N_TUPLES, TABLE_SIZE]:
            raise ValueError(f"{path}: tuple layout does not match this code")
        count = shape[0] * N_TUPLES * TABLE_SIZE
        weights = np.fromfile(f, dtype="<f4", count=count)
    if weights.size != count:
        raise ValueError(f"{path}: truncated ({weights.size} of {count} weights)")
    return weights.reshape(shape), header.get("stage_tiles", []), header.get("meta") or {}
