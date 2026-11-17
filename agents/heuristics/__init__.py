"""Hand-crafted evaluation function (ValueFunction), its tuned weights and a 1-ply agent."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

from agents.heuristics.agent import HeuristicAgent
from agents.heuristics.evaluator import WEIGHTS_DIR, HeuristicEvaluator
from agents.heuristics.features import FEATURES, exponents

__all__ = ["FEATURES", "HeuristicAgent", "HeuristicEvaluator", "exponents", "heuristic_agents"]

# Weight file (relative to the weights folder, no suffix) -> benchmark name.
_TOP_LEVEL = {"untuned": "heuristic-untuned", "tuned": "heuristic", "phased": "heuristic-phased"}


def agent_name(stem: str) -> Optional[str]:
    """Benchmark name for a weight file stem, or None if it is not registered."""
    if stem in _TOP_LEVEL:
        return _TOP_LEVEL[stem]
    if stem.startswith("ablation/"):
        return "heuristic-" + stem.split("/", 1)[1].replace("_", "-")
    return None


def _factory(path: Path, name: str) -> Callable[[], HeuristicAgent]:
    return lambda: HeuristicAgent(HeuristicEvaluator.from_file(path), name=name)


def heuristic_agents(weights_dir: Path = WEIGHTS_DIR) -> Dict[str, Callable[[], HeuristicAgent]]:
    """One zero-argument benchmark factory per weight file that exists."""
    agents: Dict[str, Callable[[], HeuristicAgent]] = {}
    for path in sorted(weights_dir.glob("*.json")) + sorted(weights_dir.glob("ablation/*.json")):
        stem = path.relative_to(weights_dir).with_suffix("").as_posix()
        name = agent_name(stem)
        if name is not None:
            agents[name] = _factory(path, name)
    return agents
