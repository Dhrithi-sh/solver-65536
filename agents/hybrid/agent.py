"""HybridAgent: expectimax search whose leaves are scored by P5's n-tuple network.

    from agents.hybrid import HybridAgent, Mode65k
    agent = HybridAgent(weights="weights/ntuple/run1/latest.ntw", depth=2)

How it fits together
--------------------
* P5 supplies the value function V (``NTupleEvaluator``), trained on afterstates.
* The search (cpp/hsearch.cpp, fast; reference.py, slow but simple) averages V
  over random tile spawns for a few moves ahead. See reference.py for the exact
  recursion and what "depth" means.
* ``Mode65k`` is our late-game switch: once the board holds a big tile, search
  deeper, prune less, and (optionally) allow more thinking time.
* Time control: with ``time_ms`` set the C++ search runs iterative deepening
  (depth 0, 1, 2, ...) and returns the deepest *completed* iteration when the
  clock runs out, so a legal, sensible move is always returned in time.

Reproducibility: with ``time_ms=None`` the agent is a pure function of the board
(thread count does not change the answer), so shared-seed benchmarks are
repeatable. A time budget makes results depend on machine speed, so the
registered benchmark agents do not use one (see registry.py).
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, replace
from typing import Dict, Optional, Union

from agents.base import Agent
from agents.hybrid import reference
from agents.ntuple.agent import NTupleEvaluator
from agents.ntuple.board import to_exponents
from agents.ntuple.network import CELLS
from engine.game import Game

try:  # compiled by:  python agents/hybrid/cpp/setup.py build_ext
    from agents.hybrid import _hsearch
except ImportError:  # pragma: no cover - depends on the local build
    _hsearch = None

_WARNED = False


def cpp_available() -> bool:
    return _hsearch is not None


@dataclass(frozen=True)
class Mode65k:
    """Late-game settings. Active when the board's max tile >= ``trigger_tile``.

    trigger_tile: 16,384 by default; the 65,536 tile needs the 32,768 first, and
                  once 16,384 is on the board every mistake is very expensive.
    extra_depth:  how many extra search plies to add.
    cutoff:       probability cutoff while active (lower = explore rarer
                  branches = more accurate but slower). None keeps the base value.
    time_ms:      per-move budget while active. None keeps the base budget.
    """

    trigger_tile: int = 16384
    extra_depth: int = 1
    cutoff: Optional[float] = 1e-5
    time_ms: Optional[float] = None


@dataclass(frozen=True)
class SearchSettings:
    depth: int
    cutoff: float
    time_ms: Optional[float]
    mode65k_active: bool


class HybridAgent(Agent):
    def __init__(
        self,
        evaluator: Optional[NTupleEvaluator] = None,
        weights: Optional[str] = None,
        depth: int = 1,
        cutoff: float = 1e-4,
        use_tt: bool = True,
        threads: int = 1,
        time_ms: Optional[float] = None,
        mode65k: Optional[Mode65k] = None,
        backend: str = "auto",
        name: Optional[str] = None,
    ):
        if backend not in ("auto", "cpp", "python"):
            raise ValueError("backend must be 'auto', 'cpp' or 'python'")
        if depth < 0:
            raise ValueError("depth must be >= 0")
        self.evaluator = evaluator if evaluator is not None else NTupleEvaluator(weights)
        self.depth = depth
        self.cutoff = cutoff
        self.use_tt = use_tt
        self.threads = max(1, int(threads))
        self.time_ms = time_ms
        self.mode65k = mode65k
        self.name = name or f"hybrid-d{depth}"
        self.last_info: dict = {}

        net = getattr(self.evaluator, "network", None)
        can_cpp = _hsearch is not None and net is not None
        if backend == "cpp" and not can_cpp:
            raise RuntimeError(
                "C++ backend unavailable: build it with `python agents/hybrid/cpp/setup.py build_ext`"
            )
        if backend == "auto" and not can_cpp:
            global _WARNED
            if not _WARNED and _hsearch is None:
                warnings.warn(
                    "agents.hybrid C++ extension not built; using the slow pure-Python search. "
                    "Build it: python agents/hybrid/cpp/setup.py build_ext",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _WARNED = True
        self.backend = "cpp" if (backend != "python" and can_cpp) else "python"
        self._cpp = (
            _hsearch.HybridSearcher(net.weights, net.stage_exps, CELLS) if self.backend == "cpp" else None
        )

    # ------------------------------------------------------------ settings
    def settings_for(self, board) -> SearchSettings:
        """Search settings for this position (base settings, or 65,536 mode)."""
        m = self.mode65k
        if m is not None and max(max(r) for r in board) >= m.trigger_tile:
            return SearchSettings(
                depth=self.depth + m.extra_depth,
                cutoff=self.cutoff if m.cutoff is None else m.cutoff,
                time_ms=self.time_ms if m.time_ms is None else m.time_ms,
                mode65k_active=True,
            )
        return SearchSettings(self.depth, self.cutoff, self.time_ms, False)

    # -------------------------------------------------------------- search
    def move_values(self, position: Union[Game, list], settings: Optional[SearchSettings] = None) -> Dict[int, float]:
        """{move: expected future score} for every legal move.

        Accepts a Game or a bare board. P8's "explain the move" view uses this.
        """
        board = position.board if isinstance(position, Game) else position
        st = settings or self.settings_for(board)
        t0 = time.perf_counter()
        if self.backend == "cpp":
            res = self._cpp.search(
                to_exponents(board),
                st.depth,
                st.cutoff,
                self.use_tt,
                float(st.time_ms or 0.0),
                self.threads,
                False,
            )
            values = {m: v for m, v in enumerate(res["values"]) if not math.isnan(v)}
            info = {k: res[k] for k in ("depth", "nodes", "aborted")}
        else:
            values, info = self._python_search(board, st)
        info.update(
            backend=self.backend,
            ms=1000 * (time.perf_counter() - t0),
            mode65k=st.mode65k_active,
            requested_depth=st.depth,
        )
        self.last_info = info
        return values

    def _python_search(self, board, st: SearchSettings):
        """Reference search. A time budget is only checked between depths here."""
        if not st.time_ms:
            vals = reference.move_values(board, self.evaluator, st.depth, st.cutoff, self.use_tt)
            return vals, {"depth": st.depth, "nodes": 0, "aborted": False}
        deadline = time.perf_counter() + st.time_ms / 1000
        vals, done = {}, -1
        for d in range(st.depth + 1):
            if d > 0 and time.perf_counter() >= deadline:
                break
            vals = reference.move_values(board, self.evaluator, d, st.cutoff, self.use_tt)
            done = d
        return vals, {"depth": done, "nodes": 0, "aborted": done < st.depth}

    def choose_move(self, game: Game) -> int:
        values = self.move_values(game)
        if not values:
            raise RuntimeError("choose_move called on a finished game")
        best = max(values.values())
        return min(m for m, v in values.items() if v == best)   # ties -> lowest move
