"""Agents P7 registers in eval/run.py.

In eval/run.py add:

    from agents.hybrid.registry import hybrid_agents
    AGENTS.update(hybrid_agents())

All registered agents are deterministic (no wall-clock budget), so the shared-seed
benchmark is reproducible and tests/test_agents.py's reproducibility test passes.
Each factory accepts ``weights=`` so ``--weights PATH`` works like for ``ntuple``.

    hybrid-d1   depth 1   one random tile ahead, then greedy at the leaf
    hybrid-d2   depth 2
    hybrid-d3   depth 3
    hybrid-65k  depth 2, and from a 16,384 tile on: depth 3 with cutoff 1e-5
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from agents.base import Agent
from agents.hybrid.agent import HybridAgent, Mode65k

CONFIGS: Dict[str, dict] = {
    "hybrid-d1": dict(depth=1),
    "hybrid-d2": dict(depth=2),
    "hybrid-d3": dict(depth=3),
    "hybrid-65k": dict(depth=2, mode65k=Mode65k(trigger_tile=16384, extra_depth=1, cutoff=1e-5)),
}


def _factory(name: str, **kw) -> Callable[..., Agent]:
    def make(weights: Optional[str] = None) -> Agent:
        return HybridAgent(weights=weights, name=name, **kw)

    make.__name__ = f"make_{name.replace('-', '_')}"
    return make


def hybrid_agents() -> Dict[str, Callable[..., Agent]]:
    return {name: _factory(name, **kw) for name, kw in CONFIGS.items()}


def make_timed_agent(weights: Optional[str] = None, time_ms: float = 100.0, max_depth: int = 4, threads: int = 4):
    """For the live demo (NOT registered: wall-clock budgets are not reproducible)."""
    return HybridAgent(
        weights=weights,
        depth=max_depth,
        time_ms=time_ms,
        threads=threads,
        mode65k=Mode65k(trigger_tile=16384, extra_depth=1, cutoff=1e-5, time_ms=time_ms * 3),
        name=f"hybrid-timed-{int(time_ms)}ms",
    )
