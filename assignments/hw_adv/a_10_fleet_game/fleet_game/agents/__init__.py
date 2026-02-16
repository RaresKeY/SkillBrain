"""Routing agents exposed by the fleet game package."""

from .base import RouteAgent
from .strategies import (
    BacktrackingAgent,
    BlankAgent,
    CheatingRAgent,
    GreedyAgent,
    PruningBacktrackingAgent,
    RAgent,
    StudentAgent,
)

__all__ = [
    "RouteAgent",
    "GreedyAgent",
    "BacktrackingAgent",
    "PruningBacktrackingAgent",
    "StudentAgent",
    "RAgent",
    "CheatingRAgent",
    "BlankAgent",
]

try:
    from .strategies import NeuralRAgent
except Exception:  # pragma: no cover - optional dependency (numpy)
    NeuralRAgent = None  # type: ignore[assignment]
else:
    __all__.append("NeuralRAgent")
