"""
Base agent class for routing algorithms.

This module defines the abstract base class that all routing agents must implement.
This follows the Strategy pattern, allowing different algorithms to be swapped easily.
"""

from abc import ABC, abstractmethod
from typing import List, Dict
from ..models import Package, Vehicle, Route, DeliveryMap
from ..utils import calculate_route_metrics


class RouteAgent(ABC):
    """
    Abstract base class for all routing algorithms.

    All routing agents must implement the plan_routes method, which takes
    a list of packages and available vehicles and produces optimized routes.

    Attributes:
        delivery_map: Map for distance calculations
        name: Human-readable agent name
        description: Description of the algorithm
    """

    def __init__(self, delivery_map: DeliveryMap, name: str = "Base Agent"):
        """
        Initialize routing agent.

        Args:
            delivery_map: Map for distance calculations
            name: Agent name
        """
        self.delivery_map = delivery_map
        self.name = name
        self.description = "Base routing agent"

    @abstractmethod
    def plan_routes(self, packages: List[Package], fleet: List[Vehicle]) -> List[Route]:
        """
        Create optimized routes for given packages and fleet.

        This is the main method that must be implemented by all agents.
        It should:
        1. Assign packages to vehicles (respecting capacity constraints)
        2. Determine the order of deliveries for each vehicle
        3. Create Route objects with proper stops

        Args:
            packages: List of packages to deliver
            fleet: Available vehicles

        Returns:
            List of routes (one per vehicle used)

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement plan_routes()")

    def calculate_metrics(self, routes: List[Route]) -> Dict:
        """
        Calculate performance metrics for routes.

        Args:
            routes: List of routes to analyze

        Returns:
            Dictionary with performance metrics
        """
        metrics = calculate_route_metrics(routes)
        metrics['agent_name'] = self.name
        return metrics

    def validate_inputs(self, packages: List[Package], fleet: List[Vehicle]) -> bool:
        """
        Validate that inputs are suitable for planning.

        Args:
            packages: Packages to deliver
            fleet: Available vehicles

        Returns:
            True if inputs are valid
        """
        if not packages:
            print(f"[{self.name}] Warning: No packages to deliver")
            return False

        if not fleet:
            print(f"[{self.name}] Warning: No vehicles available")
            return False

        # Check if total package volume can fit in fleet
        total_volume = sum(pkg.volume_m3 for pkg in packages)
        total_capacity = sum(v.vehicle_type.capacity_m3 for v in fleet)

        if total_volume > total_capacity:
            print(f"[{self.name}] Warning: Total volume ({total_volume:.1f}m³) "
                  f"exceeds fleet capacity ({total_capacity:.1f}m³)")
            return False

        return True

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"

    def __repr__(self) -> str:
        return f"RouteAgent(name='{self.name}')"
