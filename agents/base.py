"""Shared agent interfaces. Do not change without telling the group.

Every agent subclasses ``Agent`` and implements ``choose_move(game) -> int``,
returning one of ``game.legal_moves()`` (0=up 1=right 2=down 3=left).
Agents must not mutate ``game``; use ``Game.slide`` on copies to look ahead.

Anything that scores positions (heuristics, n-tuple nets, value networks)
implements ``ValueFunction.evaluate(board) -> float`` so it can be plugged
into search (e.g. P7's n-tuple-inside-expectimax hybrid).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from engine.game import Board, Game


class Agent(ABC):
    name: str = "agent"

    def reset(self, seed: Optional[int] = None) -> None:
        """Called by the benchmark before each game. Agents that use their own
        randomness must reseed from ``seed`` so results are reproducible."""

    @abstractmethod
    def choose_move(self, game: Game) -> int:
        """Return a legal move for the current position."""


class ValueFunction(ABC):
    @abstractmethod
    def evaluate(self, board: Board) -> float:
        """Higher is better for the player. Must be pure (no side effects)."""
