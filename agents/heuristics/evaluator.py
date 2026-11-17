"""Weighted-sum heuristic evaluator with optional per-phase weights."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from agents.base import ValueFunction
from agents.heuristics.features import FEATURES, Grid, exponents
from engine.game import Board

Weights = Dict[str, float]

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

PHASES = ("early", "mid", "late")
# Smallest max tile of the mid and late phases; anything below "mid" is early.
DEFAULT_PHASE_BOUNDS: Dict[str, int] = {"mid": 2048, "late": 16384}


def resolve_weights_path(name: Union[str, Path]) -> Path:
    """Map a bare name ("tuned", "ablation/full") to the weights folder.

    Anything ending in ``.json`` is used as a path as given.
    """
    text = str(name)
    if text.endswith(".json"):
        return Path(text)
    return WEIGHTS_DIR / f"{text}.json"


def phase_of(max_tile: int, bounds: Mapping[str, int] = DEFAULT_PHASE_BOUNDS) -> str:
    """Game phase for a board whose largest tile is ``max_tile``."""
    if max_tile >= bounds["late"]:
        return "late"
    if max_tile >= bounds["mid"]:
        return "mid"
    return "early"


def _check_weights(weights: Mapping[str, float]) -> Weights:
    unknown = sorted(set(weights) - set(FEATURES))
    if unknown:
        raise ValueError(f"unknown feature(s) {unknown}; known: {sorted(FEATURES)}")
    return {name: float(w) for name, w in weights.items()}


class HeuristicEvaluator(ValueFunction):
    """``evaluate(board)`` = sum of weight * feature over the weighted features.

    ``weights`` is either one set ``{feature: weight}`` or per-phase sets
    ``{"early": {...}, "mid": {...}, "late": {...}}`` selected by the board's
    max tile. Features missing from a set count as 0 and zero-weight features
    are not computed, which is what the ablation runs rely on.
    """

    def __init__(
        self,
        weights: Mapping[str, Any],
        phase_bounds: Optional[Mapping[str, int]] = None,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.meta: Dict[str, Any] = dict(meta or {})
        self.phased = bool(weights) and all(isinstance(v, Mapping) for v in weights.values())
        if self.phased:
            if set(weights) != set(PHASES):
                raise ValueError(f"phased weights need exactly the phases {PHASES}, got {sorted(weights)}")
            self.phases: Optional[Dict[str, Weights]] = {p: _check_weights(weights[p]) for p in PHASES}
            self.weights: Optional[Weights] = None
            self.phase_bounds: Optional[Dict[str, int]] = dict(phase_bounds or DEFAULT_PHASE_BOUNDS)
            if not self.phase_bounds["mid"] < self.phase_bounds["late"]:
                raise ValueError(f"phase bounds must satisfy mid < late, got {self.phase_bounds}")
            self._terms = {p: self._compile(w) for p, w in self.phases.items()}
        else:
            if phase_bounds is not None:
                raise ValueError("phase_bounds given with a single weight set")
            self.weights = _check_weights(weights)
            self.phases = None
            self.phase_bounds = None
            self._single_terms = self._compile(self.weights)

    @staticmethod
    def _compile(weights: Weights) -> List[Tuple[Callable[[Grid], float], float]]:
        # Follow FEATURES order so the float sum does not depend on dict order.
        return [(fn, weights[name]) for name, fn in FEATURES.items() if weights.get(name, 0.0) != 0.0]

    def phase(self, board: Board) -> Optional[str]:
        """Phase used for ``board`` (None for a single weight set)."""
        if not self.phased:
            return None
        return phase_of(max(map(max, board)), self.phase_bounds)

    def evaluate(self, board: Board) -> float:
        terms = self._terms[self.phase(board)] if self.phased else self._single_terms
        e = exponents(board)
        return sum(w * fn(e) for fn, w in terms)

    # ------------------------------------------------------------------ files
    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"meta": self.meta}
        if self.phased:
            data["phase_bounds"] = self.phase_bounds
            data["phases"] = self.phases
        else:
            data["weights"] = self.weights
        return data

    def save(self, path: Union[str, Path]) -> Path:
        """Write the weights as JSON. A bare name goes to the weights folder."""
        path = resolve_weights_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename, so a benchmark or test that loads the file while a
        # tuning run is rewriting it never sees half a file.
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return path

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HeuristicEvaluator":
        if ("weights" in data) == ("phases" in data):
            raise ValueError('a weight file needs exactly one of "weights" or "phases"')
        if "phases" in data:
            return cls(data["phases"], phase_bounds=data.get("phase_bounds"), meta=data.get("meta"))
        return cls(data["weights"], meta=data.get("meta"))

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "HeuristicEvaluator":
        """Load a weight file. A bare name is looked up in the weights folder."""
        return cls.from_dict(json.loads(resolve_weights_path(path).read_text(encoding="utf-8")))
