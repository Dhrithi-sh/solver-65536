"""Public entry points for the trained n-tuple network.

    from agents.ntuple.agent import NTupleEvaluator, NTupleAgent

    ev = NTupleEvaluator("weights/ntuple/run1/latest.ntw")
    ev.evaluate(game.board)          # float: expected future score from this board

``NTupleEvaluator`` is the interface P3 (expectimax) and P7 (hybrid) import.
``NTupleAgent`` plays 1-ply greedy: the legal move with the highest
``reward + V(afterstate)``, exactly the policy the network was trained with.

Which weights file is used (first match wins):
    1. the path passed in (``--weights`` on ``eval.run``)
    2. the ``NTUPLE_WEIGHTS`` environment variable
    3. ``weights/ntuple/latest.ntw`` in the repo
If none exists, loading fails with ``WeightsNotFoundError`` explaining how to get one.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from agents.base import Agent, ValueFunction
from agents.ntuple.board import slide, to_exponents
import numpy as np

from agents.ntuple.network import NTupleNetwork, StagedNetwork, as_exponents
from agents.ntuple.train import choose_move as _choose_move_kernel
from engine.game import Game

ENV_VAR = "NTUPLE_WEIGHTS"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "ntuple" / "latest.ntw"

PathLike = Union[str, Path]


class WeightsNotFoundError(FileNotFoundError):
    """No n-tuple weights file at the requested / default location."""


def resolve_weights_path(path: Optional[PathLike] = None) -> Path:
    """Pick the weights file: explicit path, then $NTUPLE_WEIGHTS, then the default."""
    if path:
        source, candidate = "--weights / weights_path", Path(path)
    elif os.environ.get(ENV_VAR):
        source, candidate = f"${ENV_VAR}", Path(os.environ[ENV_VAR])
    else:
        source, candidate = "default location", DEFAULT_WEIGHTS
    if not candidate.is_file():
        raise WeightsNotFoundError(
            f"n-tuple weights not found: {candidate} (from {source}).\n"
            f"  Train some:   python -m agents.ntuple.train --games 10000 --out weights/ntuple/run1\n"
            f"  or download the .ntw file from the team's GitHub release,\n"
            f"  then pass --weights PATH, set {ENV_VAR}=PATH, or copy it to {DEFAULT_WEIGHTS}."
        )
    return candidate


@lru_cache(maxsize=2)
def _load_cached(resolved: str, mtime_ns: int, size: int) -> StagedNetwork:
    """Load each weights file once per process (268 MB per stage). The key
    includes mtime and size, so a re-trained file at the same path is picked up."""
    return StagedNetwork.load(resolved)


class NTupleEvaluator(ValueFunction):
    """V(board) from a trained n-tuple network. Read-only and pure.

    Pass ``weights_path`` (or rely on $NTUPLE_WEIGHTS / the default), or an
    already-built ``network`` (a StagedNetwork, or a plain NTupleNetwork which
    is treated as one stage). Multi-stage weights pick the right stage per
    board automatically. Evaluators loading the same file share one copy.
    """

    def __init__(
        self,
        weights_path: Optional[PathLike] = None,
        network: Optional[Union[StagedNetwork, NTupleNetwork]] = None,
    ):
        if network is None:
            path = resolve_weights_path(weights_path)
            st = path.stat()
            network = _load_cached(str(path.resolve()), st.st_mtime_ns, st.st_size)
            self.weights_path: Optional[Path] = path
        else:
            if isinstance(network, NTupleNetwork):
                network = StagedNetwork.from_network(network)
            self.weights_path = None
        self.network: StagedNetwork = network
        # "Every stage already started": the shared move kernel must never
        # warm-start (copy) stages when it is only being used to play.
        self._all_started = np.ones(network.n_stages, dtype=np.uint8)

    def stage_of(self, board: Sequence[Sequence[int]]) -> int:
        """Which stage network scores this board (0 for single-stage weights)."""
        return self.network.stage_of(board)

    def evaluate(self, board: Sequence[Sequence[int]]) -> float:
        """Value of an engine board (4x4 tile values, 0 = empty).

        Also accepts a flat uint8[16] exponent array, the fast path for code
        that already works in exponents. Higher = more future score expected.
        """
        return self.network.evaluate(board)

    def afterstate_value(self, board: Sequence[Sequence[int]], move: int) -> Optional[float]:
        """reward + V(afterstate) for one move, or None if the move is illegal."""
        after, reward, changed = slide(as_exponents(board), move)
        if not changed:
            return None
        return float(reward) + self.network.evaluate(after)


class NTupleAgent(Agent):
    """1-ply greedy on reward + V(afterstate). Deterministic (ties -> lowest move)."""

    name = "ntuple"

    def __init__(self, evaluator: Optional[NTupleEvaluator] = None, weights_path: Optional[PathLike] = None):
        self.evaluator = evaluator if evaluator is not None else NTupleEvaluator(weights_path)

    def move_values(self, game: Game) -> Dict[int, float]:
        """reward + V(afterstate) for every legal move (handy for 'explain the move')."""
        values = {}
        for m in range(4):
            v = self.evaluator.afterstate_value(game.board, m)
            if v is not None:
                values[m] = v
        return values

    def choose_move(self, game: Game) -> int:
        # Same computation as move_values, done in one numba call for speed.
        # Only reads game.board; the game itself is never modified.
        ev = self.evaluator
        move, _, _ = _choose_move_kernel(
            ev.network.weights, ev.network.stage_exps, ev._all_started, to_exponents(game.board)
        )
        if move < 0:
            raise RuntimeError("choose_move called on a finished game")
        return int(move)


def make_ntuple_agent(weights: Optional[PathLike] = None) -> NTupleAgent:
    """Factory registered in eval/run.py; ``weights`` comes from ``--weights``."""
    return NTupleAgent(weights_path=weights)
