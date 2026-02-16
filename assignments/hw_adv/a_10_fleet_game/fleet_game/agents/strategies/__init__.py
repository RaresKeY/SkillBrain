"""Built-in routing strategies."""

from .agent_r import RAgent
from .agent_r_cheating import RAgent as CheatingRAgent
from .backtracking import BacktrackingAgent, PruningBacktrackingAgent
from .blank_agent import CustomAgent as BlankAgent
from .greedy import GreedyAgent
from .student import StudentAgent

__all__ = [
    "GreedyAgent",
    "BacktrackingAgent",
    "PruningBacktrackingAgent",
    "StudentAgent",
    "RAgent",
    "CheatingRAgent",
    "BlankAgent",
]

try:
    from .agent_r_neural import NeuralRAgent
except Exception:  # pragma: no cover - optional dependency (numpy)
    NeuralRAgent = None  # type: ignore[assignment]
else:
    __all__.append("NeuralRAgent")
